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
    normalize_answer,
)
from mlx_commander.evals.runner import load_test_dataset, run_generative_eval
from mlx_commander.evals.storage import (
    append_to_leaderboard_csv,
    generate_html_dashboard,
    save_run_predictions,
)
from mlx_commander.lora.config import LoraRunConfig
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
        # Verify it drew [ Yes ] and [ No ]
        calls = [c[0][2] for c in win.addstr.call_args_list if len(c[0]) >= 3 and isinstance(c[0][2], str)]
        has_yes = any("[ Yes ]" in s for s in calls)
        has_no = any("[ No ]" in s for s in calls)
        self.assertTrue(has_yes)
        self.assertTrue(has_no)


if __name__ == "__main__":
    unittest.main()
