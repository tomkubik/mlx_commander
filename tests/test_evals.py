import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlx_commander.evals.metrics import (
    build_confusion_matrix,
    build_migration_matrix,
    classify_transition,
    compute_exact_match,
    compute_substring_match,
    compute_word_metrics,
    format_cli_categorical_confusion_matrix,
    format_cli_eval_summary_matrices,
    format_cli_migration_matrix,
    normalize_answer,
)
from mlx_commander.evals.runner import (
    active_checkpoint_context,
    clean_generation_answer,
    discover_adapter_checkpoints,
    estimate_eval_throughput,
    extract_generation_text,
    format_prompt_for_model,
    load_test_dataset,
    parse_chat_prompt,
    parse_training_losses_from_log,
    resolve_eval_checkpoints,
    run_generative_eval,
    run_inference_batch,
)
from mlx_commander.evals.storage import (
    append_to_leaderboard_csv,
    generate_html_dashboard,
    save_run_predictions,
)
from mlx_commander.lora.config import (
    EVAL_STRATEGY_ALL,
    EVAL_STRATEGY_FINAL,
    EVAL_STRATEGY_MIN_TRAIN_LOSS,
    EVAL_STRATEGY_MIN_VAL_LOSS,
    LoraRunConfig,
)
from mlx_commander.lora.multi_config import MultiLoraRunConfig, SWEEP_FIELD_DEFS
from mlx_commander.lora.tracking import WandbTracker
from mlx_commander.tui.state import CommanderState
from mlx_commander.tui.widgets import draw_multi_field


class TestEvalMetrics(unittest.TestCase):
    def test_normalize_answer(self):
        self.assertEqual(normalize_answer("  The Blue, Sky!  "), "blue sky")
        self.assertEqual(normalize_answer("An Apple a day."), "apple day")
        self.assertEqual(normalize_answer(""), "")

    def test_compute_exact_match(self):
        # Strict exact match
        self.assertTrue(compute_exact_match("Paris", "Paris", normalize=False))
        self.assertFalse(compute_exact_match("Paris ", "Paris", normalize=False))
        self.assertFalse(compute_exact_match("paris", "Paris", normalize=False))

        # Normalized exact match
        self.assertTrue(compute_exact_match("  The Paris! ", "paris", normalize=True))
        self.assertTrue(compute_exact_match("blue and gold", "The blue and gold.", normalize=True))
        self.assertFalse(compute_exact_match("London", "Paris", normalize=True))

    def test_compute_substring_match(self):
        # Golden answer is contained within generated answer
        self.assertTrue(compute_substring_match("The capital of France is Paris.", "Paris"))
        self.assertTrue(compute_substring_match("The answer is 42!", "42"))
        self.assertFalse(compute_substring_match("I don't know the capital.", "Paris"))

    def test_compute_word_metrics(self):
        # Case 1: Exact match
        scores = compute_word_metrics("blue and gold", "blue and gold")
        self.assertEqual(scores["precision"], 1.0)
        self.assertEqual(scores["recall"], 1.0)
        self.assertEqual(scores["f1"], 1.0)

        # Case 2: Partial overlap / extra words
        # Golden: "blue and gold" (3 words)
        # Generated: "The colors are blue and gold" ("the" article removed -> 5 words: colors, are, blue, and, gold)
        scores = compute_word_metrics("The colors are blue and gold", "blue and gold")
        self.assertEqual(scores["recall"], 1.0)  # Found all 3 golden words
        self.assertAlmostEqual(scores["precision"], 3 / 5, places=2)  # 0.6
        expected_f1 = (2 * 0.6 * 1.0) / (0.6 + 1.0)  # 0.75
        self.assertAlmostEqual(scores["f1"], round(expected_f1, 4), places=3)

        # Case 3: Complete miss (zero overlap)
        scores = compute_word_metrics("red and green", "cyan magenta yellow")
        self.assertEqual(scores["f1"], 0.0)

    def test_classify_transition(self):
        # Both correct -> PRESERVED
        self.assertEqual(classify_transition(True, True), "PRESERVED")
        # Baseline wrong, LoRA correct -> FIXED (cured!)
        self.assertEqual(classify_transition(False, True), "FIXED")
        # Baseline correct, LoRA wrong -> REGRESSED (catastrophic forgetting!)
        self.assertEqual(classify_transition(True, False), "REGRESSED")
        # Both wrong -> PERSISTENT_FAIL
        self.assertEqual(classify_transition(False, False), "PERSISTENT_FAIL")

    def test_build_migration_matrix(self):
        transitions = [
            "PRESERVED", "PRESERVED",
            "FIXED", "FIXED", "FIXED",
            "REGRESSED",
            "PERSISTENT_FAIL",
        ]
        mat = build_migration_matrix(transitions)
        self.assertEqual(mat["total"], 7)
        self.assertEqual(mat["counts"]["FIXED"], 3)
        self.assertEqual(mat["counts"]["REGRESSED"], 1)
        self.assertEqual(mat["counts"]["PRESERVED"], 2)
        self.assertEqual(mat["counts"]["PERSISTENT_FAIL"], 1)
        self.assertEqual(mat["fixed_count"], 3)
        self.assertEqual(mat["regressed_count"], 1)

    def test_build_confusion_matrix(self):
        ground_truths = ["positive", "positive", "negative", "neutral", "positive"]
        predictions = ["positive", "neutral", "negative", "neutral", "positive"]
        cm = build_confusion_matrix(ground_truths, predictions)
        self.assertIsNotNone(cm)
        self.assertTrue(cm["is_categorical"])
        self.assertEqual(sorted(cm["classes"]), ["negative", "neutral", "positive"])
        self.assertEqual(cm["matrix"]["positive"]["positive"], 2)
        self.assertEqual(cm["matrix"]["positive"]["neutral"], 1)
        self.assertEqual(cm["matrix"]["negative"]["negative"], 1)
        self.assertEqual(cm["matrix"]["neutral"]["neutral"], 1)
        self.assertEqual(cm["accuracy"], 80.0)

    def test_format_cli_migration_matrix(self):
        transitions = [
            "PRESERVED", "PRESERVED",
            "FIXED", "FIXED", "FIXED",
            "REGRESSED",
            "PERSISTENT_FAIL",
        ]
        mat = build_migration_matrix(transitions)
        formatted = format_cli_migration_matrix(mat)
        self.assertIn("MODEL CONFUSION & MIGRATION MATRIX", formatted)
        self.assertIn("POST-TUNING (LoRA)", formatted)
        self.assertIn("PRESERVED:    2", formatted)
        self.assertIn("FIXED:        3", formatted)
        self.assertIn("REGRESSED:    1", formatted)
        self.assertIn("PERSISTENT:   1", formatted)
        self.assertIn("Net Model Accuracy Gain:     +2 (+28.6%)", formatted)

    def test_format_cli_categorical_confusion_matrix(self):
        ground_truths = ["positive", "positive", "negative", "neutral", "positive"]
        predictions = ["positive", "neutral", "negative", "neutral", "positive"]
        cm = build_confusion_matrix(ground_truths, predictions)
        formatted = format_cli_categorical_confusion_matrix(cm)
        self.assertIn("CATEGORICAL CONFUSION MATRIX", formatted)
        self.assertIn("Actual \\ Pred", formatted)
        self.assertIn("Recall", formatted)
        self.assertIn("positive", formatted)
        self.assertIn("Categorical Accuracy: 80.0%", formatted)


class TestEvalStorageAndRunner(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_load_test_dataset(self):
        ds_dir = self.base_path / "test_ds"
        ds_dir.mkdir()
        test_file = ds_dir / "test.jsonl"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": "Q1", "completion": "A1"}) + "\n")
            f.write(json.dumps({"messages": [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}]}) + "\n")
            f.write(json.dumps({"text": "Instruction: Q3. Response: A3."}) + "\n")

        samples = load_test_dataset(ds_dir)
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0]["prompt"], "Q1")
        self.assertEqual(samples[0]["golden"], "A1")
        self.assertIn("Q2", samples[1]["prompt"])
        self.assertEqual(samples[1]["golden"], "A2")

    def test_run_generative_eval_pipeline(self):
        ds_dir = self.base_path / "dataset"
        ds_dir.mkdir()
        test_file = ds_dir / "test.jsonl"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": "What is 2+2?", "completion": "4"}) + "\n")
            f.write(json.dumps({"prompt": "Capital of France?", "completion": "Paris"}) + "\n")
            f.write(json.dumps({"prompt": "Colors?", "completion": "blue and gold"}) + "\n")

        adapters_dir = self.base_path / "adapters" / "01_test_run"
        cfg = LoraRunConfig(
            name="01_test_run",
            data=str(ds_dir),
            adapter_path=str(adapters_dir),
            run_eval=True,
        )

        mock_baseline = ["I am not sure", "London", "The colors are blue and gold"]
        mock_predictions = ["4", "The capital of France is Paris.", "blue and gold"]

        eval_res = run_generative_eval(
            config=cfg,
            data_path=ds_dir,
            mock_predictions=mock_predictions,
            mock_baseline=mock_baseline,
        )

        self.assertIsNotNone(eval_res)
        summary = eval_res["summary"]
        self.assertEqual(summary["total_samples"], 3)
        self.assertGreater(summary["exact_match_pct"], 0)
        self.assertEqual(summary["substring_match_pct"], 100.0)
        self.assertGreater(summary["avg_word_f1"], 0.5)

        # Verify fixed count: Sample 1 (2+2) and Sample 2 (Paris) were wrong in baseline, right in LoRA!
        self.assertGreaterEqual(summary["fixed_count"], 2)

        # Verify Tier 1: eval_leaderboard.csv
        csv_file = Path(eval_res["leaderboard_csv"])
        self.assertTrue(csv_file.exists())
        with open(csv_file, "r", encoding="utf-8") as f:
            csv_content = f.read()
            self.assertIn("01_test_run", csv_content)
            self.assertIn("exact_match_pct", csv_content)

        # Verify Tier 2: eval_summary.json & eval_predictions.jsonl
        summary_file = adapters_dir / "eval_summary.json"
        preds_file = adapters_dir / "eval_predictions.jsonl"
        self.assertTrue(summary_file.exists())
        self.assertTrue(preds_file.exists())

        # Verify Tier 3: eval_comparison.html
        html_file = Path(eval_res["html_dashboard"])
        self.assertTrue(html_file.exists())
        with open(html_file, "r", encoding="utf-8") as f:
            html_text = f.read()
            self.assertIn("MLX Commander :: Generative Eval Dashboard", html_text)
            self.assertIn("Regressed (Model Broke)", html_text)
            self.assertIn("01_test_run", html_text)
            self.assertIn("Model Confusion Map & Migration Matrix", html_text)
            self.assertIn("quadrant-tile", html_text)
            self.assertIn("tile-preserved", html_text)
            self.assertIn("tile-fixed", html_text)
            self.assertIn("tile-regressed", html_text)
            self.assertIn("POST-TUNING (LoRA)", html_text)
            self.assertIn("BASELINE", html_text)
            self.assertIn("initConfusionMap", html_text)

    def test_run_generative_eval_cli_displays_confusion_matrix(self):
        ds_dir = self.base_path / "dataset_cli_matrix"
        ds_dir.mkdir()
        test_file = ds_dir / "test.jsonl"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": "Q1", "completion": "A1"}) + "\n")
            f.write(json.dumps({"prompt": "Q2", "completion": "A2"}) + "\n")

        adapters_dir = self.base_path / "adapters" / "03_matrix_run"
        cfg = LoraRunConfig(
            name="03_matrix_run",
            data=str(ds_dir),
            adapter_path=str(adapters_dir),
            run_eval=True,
        )

        out_buf = io.StringIO()
        with patch("sys.stdout", out_buf):
            run_generative_eval(
                config=cfg,
                data_path=ds_dir,
                mock_predictions=["A1", "A2"],
                mock_baseline=["Wrong", "A2"],
            )

        printed = out_buf.getvalue()
        self.assertIn("MODEL CONFUSION & MIGRATION MATRIX", printed)
        self.assertIn("PRESERVED:", printed)
        self.assertIn("FIXED:", printed)
        self.assertIn("REGRESSED:", printed)
        self.assertIn("Net Model Accuracy Gain:", printed)

    def test_html_dashboard_confusion_matrix_elements(self):
        html_out = self.base_path / "custom_dashboard.html"
        mock_summary = {
            "run_id": "test_run",
            "run_name": "Test Run",
            "exact_match_pct": 80.0,
            "avg_word_f1": 0.85,
            "fixed_count": 4,
            "regressed_count": 1,
            "migration_matrix": {
                "total": 10,
                "counts": {"PRESERVED": 4, "FIXED": 4, "REGRESSED": 1, "PERSISTENT_FAIL": 1},
                "percentages": {"PRESERVED": 40.0, "FIXED": 40.0, "REGRESSED": 10.0, "PERSISTENT_FAIL": 10.0},
            },
            "confusion_matrix": {
                "is_categorical": True,
                "classes": ["neg", "pos"],
                "matrix": {"neg": {"neg": 4, "pos": 1}, "pos": {"neg": 0, "pos": 5}},
                "accuracy": 90.0,
                "total_samples": 10,
            },
        }

        generate_html_dashboard(
            dashboard_path=html_out,
            current_run_summary=mock_summary,
            current_predictions=[],
        )

        self.assertTrue(html_out.exists())
        with open(html_out, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Model Confusion Map & Migration Matrix", content)
        self.assertIn("Categorical Confusion Matrix", content)
        self.assertIn("categoricalCmTable", content)
        self.assertIn("initConfusionMap", content)
        self.assertIn("Net Model Gain:", content)

    def test_run_inference_batch_milestone_output(self):
        prompts = ["Question 1", "Question 2", "Question 3"]
        out_buf = io.StringIO()
        with patch("sys.stdout", out_buf):
            preds, tok_sec, total_tok = run_inference_batch(
                model_name="test-model",
                adapter_path=None,
                prompts=prompts,
                phase_label="1/2 Baseline",
            )
        self.assertEqual(len(preds), 3)
        printed = out_buf.getvalue()
        self.assertIn("[1/2 Baseline]", printed)
        self.assertIn("Sample 1/3", printed)
        self.assertIn("Sample 3/3", printed)

    def test_run_generative_eval_cli_transparency_and_no_cap(self):
        ds_dir = self.base_path / "dataset_transparency"
        ds_dir.mkdir()
        test_file = ds_dir / "test.jsonl"
        with open(test_file, "w", encoding="utf-8") as f:
            for i in range(5):
                f.write(json.dumps({"prompt": f"Q{i}", "completion": f"A{i}"}) + "\n")

        adapters_dir = self.base_path / "adapters" / "02_transparent_run"
        cfg = LoraRunConfig(
            name="02_transparent_run",
            model="meta-llama/Llama-3.2-3B",
            data=str(ds_dir),
            adapter_path=str(adapters_dir),
            run_eval=True,
            engine="mlx_lm",
        )

        out_buf = io.StringIO()
        with patch("sys.stdout", out_buf):
            res = run_generative_eval(
                config=cfg,
                data_path=ds_dir,
                mock_predictions=["A0", "A1", "A2", "A3", "A4"],
                mock_baseline=["A0", "Wrong", "Wrong", "A3", "Wrong"],
            )

        printed = out_buf.getvalue()
        self.assertIn("Target Checkpoint:", printed)
        self.assertIn("02_transparent_run", printed)
        self.assertIn("Base Model:", printed)
        self.assertIn("meta-llama/Llama-3.2-3B", printed)
        self.assertIn("Full split: 5 samples", printed)
        self.assertIn("10 generations", printed)
        self.assertIn("Estimated Time:", printed)
        self.assertIn("tok/s", printed)
        self.assertIn("model: ~3B", printed)
        # Ensure full test set was evaluated (no artificial cap)
        self.assertEqual(res["summary"]["total_samples"], 5)

    def test_estimate_eval_throughput_runner_integration(self):
        est_small = estimate_eval_throughput("mlx-community/Llama-3.2-1B-Instruct-4bit", total_samples=100)
        est_large = estimate_eval_throughput("meta-llama/Llama-3.1-70B-Instruct-4bit", total_samples=100)
        self.assertGreater(est_small["est_tps"], est_large["est_tps"])
        self.assertLess(est_small["est_total_sec"], est_large["est_total_sec"])
        self.assertIn("tok/s on", est_small["summary_label"])
        self.assertIn("tok/s on", est_large["summary_label"])


class TestWandbEvalLogging(unittest.TestCase):
    @patch("mlx_commander.lora.tracking.is_wandb_available", return_value=True)
    @patch("mlx_commander.lora.tracking.is_wandb_logged_in", return_value=(True, "test_user"))
    def test_wandb_log_eval(self, mock_logged_in, mock_available):
        tracker = WandbTracker(project="test-proj", enabled=True)
        tracker._is_active = True
        tracker.run = MagicMock()

        eval_result = {
            "summary": {
                "exact_match_pct": 75.0,
                "substring_match_pct": 90.0,
                "avg_word_f1": 0.82,
                "avg_word_prec": 0.85,
                "avg_word_recall": 0.80,
                "fixed_count": 5,
                "regressed_count": 1,
                "tokens_per_sec": 48.0,
            },
            "predictions": [
                {
                    "id": 1,
                    "prompt": "Q1",
                    "golden": "A1",
                    "baseline_output": "Bad",
                    "model_output": "A1",
                    "status": "FIXED",
                    "word_f1": 1.0,
                    "exact_match_norm": True,
                }
            ],
        }

        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker.log_eval(eval_result)
            self.assertTrue(mock_wandb.log.called)
            self.assertTrue(mock_wandb.Table.called)


class TestEvalUIControls(unittest.TestCase):
    def test_sweep_field_defs_includes_run_eval(self):
        names = [f[0] for f in SWEEP_FIELD_DEFS]
        self.assertIn("run_eval", names)
        # Check label
        labels = {f[0]: f[1] for f in SWEEP_FIELD_DEFS}
        self.assertEqual(labels["run_eval"], "Run evals on test set (experimental)")

    def test_multi_lora_run_config_run_eval_propagation(self):
        cfg = MultiLoraRunConfig(run_eval=[True, False])
        runs = cfg.generate_runs()
        self.assertEqual(len(runs), 2)
        self.assertTrue(runs[0].run_eval)
        self.assertFalse(runs[1].run_eval)

    def test_state_toggle_run_eval(self):
        state = CommanderState()
        self.assertFalse(state.lora_config.run_eval)
        state.lora_config.run_eval = True
        self.assertTrue(state.lora_config.run_eval)

    def test_draw_multi_field_eval_yes_no(self):
        win = MagicMock()
        win.getmaxyx.return_value = (24, 80)
        draw_multi_field(win, 0, 0, "Run evals on test set", [True, False], is_focused=True)
        # Verify it drew Yes and No without brackets
        calls = [c[0][2] for c in win.addstr.call_args_list if len(c[0]) >= 3 and isinstance(c[0][2], str)]
        has_yes = any(" Yes " in s for s in calls)
        has_no = any(" No " in s for s in calls)
        self.assertTrue(has_yes)
        self.assertTrue(has_no)
        self.assertFalse(any("[ Yes ]" in s for s in calls))
        self.assertFalse(any("[ No ]" in s for s in calls))


class TestChatTemplateFormatting(unittest.TestCase):
    def test_parse_chat_prompt(self):
        raw = "system: You are a classifier.\nuser: Stock rose 5%."
        msgs = parse_chat_prompt(raw)
        self.assertIsNotNone(msgs)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "You are a classifier.")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[1]["content"], "Stock rose 5%.")

    def test_parse_non_chat_prompt(self):
        self.assertIsNone(parse_chat_prompt("What is 2+2?"))
        self.assertIsNone(parse_chat_prompt(""))

    def test_format_prompt_gemma_merges_system_and_adds_generation_prompt(self):
        class MockGemmaTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                if messages[0]["role"] == "system":
                    raise ValueError("System role not supported")
                res = ""
                for m in messages:
                    res += f"<start_of_turn>{m['role']}\n{m['content']}<end_of_turn>\n"
                if add_generation_prompt:
                    res += "<start_of_turn>model\n"
                return res

        sample = {
            "messages": [
                {"role": "system", "content": "You are a sentiment classifier."},
                {"role": "user", "content": "Stock rallied today."},
            ]
        }
        res = format_prompt_for_model(sample, MockGemmaTokenizer())
        self.assertIn("<start_of_turn>user", res)
        self.assertIn("You are a sentiment classifier.", res)
        self.assertIn("Stock rallied today.", res)
        self.assertTrue(res.endswith("<start_of_turn>model\n"))

    def test_format_prompt_llama_supports_system_and_adds_generation_prompt(self):
        class MockLlamaTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                res = "<|begin_of_text|>"
                for m in messages:
                    res += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
                if add_generation_prompt:
                    res += "<|start_header_id|>assistant<|end_header_id|>\n\n"
                return res

        sample = {
            "messages": [
                {"role": "system", "content": "System text"},
                {"role": "user", "content": "User question"},
            ]
        }
        res = format_prompt_for_model(sample, MockLlamaTokenizer())
        self.assertIn("<|start_header_id|>system<|end_header_id|>", res)
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>", res)

    def test_format_prompt_raw_string_without_template(self):
        # No chat template available: should preserve raw prompt
        tok = MagicMock(spec=[])
        res = format_prompt_for_model("What is 2+2?", tok)
        self.assertEqual(res, "What is 2+2?")

    def test_extract_generation_text_from_object(self):
        class MockGenResult:
            def __init__(self, text):
                self.text = text
            def __repr__(self):
                return f"GenerationResult(text='{self.text}', logprobs=[0.1])"

        res = extract_generation_text(MockGenResult("negative"))
        self.assertEqual(res, "negative")

    def test_extract_generation_text_from_dict(self):
        res = extract_generation_text({"text": "positive"})
        self.assertEqual(res, "positive")

    def test_extract_generation_text_from_leaked_string_repr(self):
        leaked = "GenerationResult(text='neutral', token=106, logprobs=array([-40.5]))"
        res = extract_generation_text(leaked)
        self.assertEqual(res, "neutral")

    def test_clean_generation_answer_strips_turn_tokens(self):
        self.assertEqual(clean_generation_answer("<start_of_turn>model\nnegative<end_of_turn>"), "negative")
        self.assertEqual(clean_generation_answer("assistant:\npositive<|eot_id|>"), "positive")
        self.assertEqual(clean_generation_answer("neutral</s>"), "neutral")


class TestAdapterCheckpointEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_training_losses_from_log(self):
        log_file = self.base_path / "run.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("=== MLX Commander LoRA Run ===\n")
            f.write("Iter 100: Train loss 2.500, Learning Rate 1e-4\n")
            f.write("Iter 100: Val loss 2.800, Val It/sec 2.0\n")
            f.write("Iter 200: Train loss 1.900, Learning Rate 1e-4\n")
            f.write("★ [Validation Loss Milestone] Iter 200: Val loss = 1.600 (val took 0.4s)\n")
            f.write("Iter 300: Train loss 1.400, Learning Rate 1e-4\n")
            f.write("Iter 300: Val loss 1.750, Val It/sec 2.0\n")

        train_losses, val_losses = parse_training_losses_from_log(log_file)
        self.assertEqual(train_losses[100], 2.5)
        self.assertEqual(train_losses[200], 1.9)
        self.assertEqual(train_losses[300], 1.4)
        self.assertEqual(val_losses[100], 2.8)
        self.assertEqual(val_losses[200], 1.6)
        self.assertEqual(val_losses[300], 1.75)

    def test_discover_adapter_checkpoints(self):
        adir = self.base_path / "adapters_test"
        adir.mkdir()
        ckpt100 = adir / "0000100_adapters.safetensors"
        ckpt200 = adir / "0000200_adapters.safetensors"
        ckpt_final = adir / "adapters.safetensors"
        ckpt100.touch()
        ckpt200.touch()
        ckpt_final.touch()

        cfg = LoraRunConfig(name="test_run", adapter_path=str(adir), iters=300)
        ckpts = discover_adapter_checkpoints(adir, config=cfg)
        self.assertIn(100, ckpts)
        self.assertIn(200, ckpts)
        self.assertIn(300, ckpts)
        self.assertEqual(ckpts[100], ckpt100)
        self.assertEqual(ckpts[200], ckpt200)
        self.assertEqual(ckpts[300], ckpt_final)

    def test_resolve_eval_checkpoints_strategies(self):
        adir = self.base_path / "adapters_strat"
        adir.mkdir()
        (adir / "0000100_adapters.safetensors").touch()
        (adir / "0000200_adapters.safetensors").touch()
        (adir / "0000300_adapters.safetensors").touch()
        (adir / "adapters.safetensors").touch()

        log_file = self.base_path / "run.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("Iter 100: Train loss 2.500\nIter 100: Val loss 2.800\n")
            f.write("Iter 200: Train loss 1.900\nIter 200: Val loss 1.350\n")  # Min val loss!
            f.write("Iter 300: Train loss 1.100\nIter 300: Val loss 1.750\n")  # Min train loss!

        cfg = LoraRunConfig(
            name="strat_run",
            adapter_path=str(adir),
            iters=300,
            log_file=str(log_file),
        )

        # 1. Final adapter
        cfg.eval_adapter_strategy = EVAL_STRATEGY_FINAL
        res_final = resolve_eval_checkpoints(cfg, log_file=log_file)
        self.assertEqual(len(res_final), 1)
        self.assertIn("Final", res_final[0][0])
        self.assertTrue(res_final[0][1].name in ("adapters.safetensors", "0000300_adapters.safetensors"))

        # 2. Min val loss
        cfg.eval_adapter_strategy = EVAL_STRATEGY_MIN_VAL_LOSS
        res_val = resolve_eval_checkpoints(cfg, log_file=log_file)
        self.assertEqual(len(res_val), 1)
        self.assertIn("minimal validation loss", res_val[0][0].lower())
        self.assertIn("200", res_val[0][1].name)

        # 3. Min train loss
        cfg.eval_adapter_strategy = EVAL_STRATEGY_MIN_TRAIN_LOSS
        res_train = resolve_eval_checkpoints(cfg, log_file=log_file)
        self.assertEqual(len(res_train), 1)
        self.assertIn("minimal training loss", res_train[0][0].lower())
        self.assertTrue(res_train[0][1].name in ("adapters.safetensors", "0000300_adapters.safetensors"))

        # 4. All adapters
        cfg.eval_adapter_strategy = EVAL_STRATEGY_ALL
        res_all = resolve_eval_checkpoints(cfg, log_file=log_file)
        self.assertEqual(len(res_all), 3)
        self.assertIn("100", res_all[0][1].name)
        self.assertIn("200", res_all[1][1].name)
        self.assertTrue(res_all[2][1].name in ("adapters.safetensors", "0000300_adapters.safetensors"))

    def test_resolve_eval_checkpoints_fallback_when_no_val_loss(self):
        adir = self.base_path / "adapters_fallback"
        adir.mkdir()
        (adir / "0000100_adapters.safetensors").touch()
        (adir / "adapters.safetensors").touch()

        # Empty log (no validation entries)
        log_file = self.base_path / "empty.log"
        log_file.touch()

        cfg = LoraRunConfig(
            name="fallback_run",
            adapter_path=str(adir),
            iters=100,
            eval_adapter_strategy=EVAL_STRATEGY_MIN_VAL_LOSS,
            log_file=str(log_file),
        )
        res = resolve_eval_checkpoints(cfg, log_file=log_file)
        self.assertEqual(len(res), 1)
        self.assertIn("fallback", res[0][0].lower())

    def test_run_generative_eval_all_checkpoints_execution(self):
        ds_dir = self.base_path / "ds"
        ds_dir.mkdir()
        test_file = ds_dir / "test.jsonl"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": "What is Apple?", "completion": "Tech company"}) + "\n")
            f.write(json.dumps({"prompt": "What is Python?", "completion": "Language"}) + "\n")

        adir = self.base_path / "adapters_multi_eval"
        adir.mkdir()
        (adir / "0000100_adapters.safetensors").touch()
        (adir / "0000200_adapters.safetensors").touch()
        (adir / "adapters.safetensors").touch()

        cfg = LoraRunConfig(
            name="multi_eval_run",
            data=str(ds_dir),
            adapter_path=str(adir),
            iters=200,
            run_eval=True,
            eval_adapter_strategy=EVAL_STRATEGY_ALL,
        )

        res = run_generative_eval(
            config=cfg,
            data_path=ds_dir,
            mock_predictions=["Tech company", "Language"],
            mock_baseline=["Wrong", "Language"],
        )
        self.assertIsNotNone(res)
        self.assertIn("all_eval_results", res)
        # Evaluated both step 100 and step 200
        self.assertEqual(len(res["all_eval_results"]), 2)
        # Check leaderboard CSV exists and has entries
        leaderboard_csv = self.base_path / "eval_leaderboard.csv"
        self.assertTrue(leaderboard_csv.exists())
        with open(leaderboard_csv, "r", encoding="utf-8") as f:
            csv_content = f.read()
        self.assertIn("multi_eval_run", csv_content)
        self.assertIn("100", csv_content)
        self.assertIn("200", csv_content)

    def test_lora_run_config_eval_strategy_serialization(self):
        cfg = LoraRunConfig(
            name="serial_run",
            eval_adapter_strategy=EVAL_STRATEGY_MIN_VAL_LOSS,
            run_eval=True,
        )
        d = cfg.to_dict()
        self.assertEqual(d["eval_adapter_strategy"], "best_val_loss")

        loaded = LoraRunConfig.from_dict(d)
        self.assertEqual(loaded.eval_adapter_strategy, "best_val_loss")

        # YAML parsing
        yaml_text = "eval_adapter_strategy: best_train_loss\nrun_eval: true\n"
        from_yaml_cfg = LoraRunConfig.from_yaml(yaml_text)
        self.assertEqual(from_yaml_cfg.eval_adapter_strategy, "best_train_loss")
        self.assertTrue(from_yaml_cfg.run_eval)


if __name__ == "__main__":
    unittest.main()
