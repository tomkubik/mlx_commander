"""
Unit and Integration Tests for Dual-Engine Support (mlx-lm and mlx-vlm).
Verifies:
1. Auto-detection of VLM models vs text LLMs.
2. CLI command generation for mlx_vlm.lora vs mlx_lm.lora.
3. Propagation of engine metadata through LoraRunConfig and MultiLoraRunConfig.
4. Standalone script generation in QueueManager.
5. Pre-flight validation when engine is missing.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlx_commander.lora.config import LoraRunConfig
from mlx_commander.lora.model_info import (
    ModelMetadata,
    detect_model_engine,
    inspect_local_model,
    is_engine_installed,
)
from mlx_commander.lora.multi_config import MultiLoraRunConfig
from mlx_commander.lora.queue import QueueManager
from mlx_commander.lora.runner import execute_single_run


class TestVlmEngineDetection(unittest.TestCase):
    def test_detect_pure_llm(self):
        # LLaMA model
        self.assertEqual(
            detect_model_engine({"model_type": "llama", "num_hidden_layers": 32}),
            "mlx_lm",
        )
        # Mistral
        self.assertEqual(
            detect_model_engine({"model_type": "mistral"}, "mistralai/Mistral-7B-v0.1"),
            "mlx_lm",
        )
        # Gemma 2 (text)
        self.assertEqual(
            detect_model_engine({"model_type": "gemma2"}, "google/gemma-2-9b-it"),
            "mlx_lm",
        )

    def test_detect_vlm_via_vision_config(self):
        # Model with vision_config dict
        cfg = {
            "model_type": "gemma",
            "vision_config": {"hidden_size": 1152},
            "text_config": {"hidden_size": 2048},
        }
        self.assertEqual(detect_model_engine(cfg), "mlx_vlm")

    def test_detect_vlm_via_image_token_id(self):
        cfg = {"model_type": "custom", "image_token_id": 256000}
        self.assertEqual(detect_model_engine(cfg), "mlx_vlm")

    def test_detect_vlm_via_known_model_type(self):
        for vlm_type in ["gemma4", "paligemma", "paligemma2", "llava", "qwen2_vl", "smolvlm", "mllama"]:
            self.assertEqual(
                detect_model_engine({"model_type": vlm_type}),
                "mlx_vlm",
                f"Failed to detect {vlm_type} as mlx_vlm",
            )

    def test_detect_vlm_via_architectures(self):
        cfg1 = {"architectures": ["Gemma4ForConditionalGeneration"]}
        self.assertEqual(detect_model_engine(cfg1), "mlx_vlm")

        cfg2 = {"architectures": ["LlavaForConditionalGeneration"]}
        self.assertEqual(detect_model_engine(cfg2), "mlx_vlm")

        cfg3 = {"architectures": ["Qwen2VLForConditionalGeneration"]}
        self.assertEqual(detect_model_engine(cfg3), "mlx_vlm")

    def test_detect_vlm_via_model_name_or_path(self):
        # String heuristics when config is empty or remote Hub ID
        self.assertEqual(detect_model_engine({}, "google/gemma-4-4b-it"), "mlx_vlm")
        self.assertEqual(detect_model_engine({}, "Qwen/Qwen2-VL-7B-Instruct"), "mlx_vlm")
        self.assertEqual(detect_model_engine({}, "/local/models/llava-1.5-7b-hf"), "mlx_vlm")
        self.assertEqual(detect_model_engine({}, "HuggingFaceTB/SmolVLM-Instruct"), "mlx_vlm")

    def test_inspect_local_model_vlm(self):
        tmp_dir = Path(tempfile.mkdtemp(prefix="test_vlm_model_"))
        try:
            cfg = {
                "model_type": "gemma4",
                "architectures": ["Gemma4ForConditionalGeneration"],
                "num_hidden_layers": 28,
                "hidden_size": 2560,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "vocab_size": 256000,
                "vision_config": {"image_size": 896},
            }
            with open(tmp_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f)

            meta = inspect_local_model(str(tmp_dir))
            self.assertTrue(meta.is_valid)
            self.assertTrue(meta.is_vlm)
            self.assertEqual(meta.engine, "mlx_vlm")
            formatted = meta.format_hyperparameters_line()
            self.assertIn("engine=mlx-vlm", formatted)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestVlmCliAndConfig(unittest.TestCase):
    def test_lora_run_config_vlm_command(self):
        cfg = LoraRunConfig(
            model="google/gemma-4-4b-it",
            data="mlx_dataset",
            engine="mlx_vlm",
            iters=500,
            batch_size=2,
            learning_rate=2e-5,
            lora_rank=16,
            lora_alpha=32.0,
            save_every=50,
            grad_checkpoint=True,
            train_on_completions=True,
            adapter_path="adapters/01_gemma4",
        )
        cmd = cfg.to_cli_command()
        self.assertTrue(cmd.startswith("mlx_vlm.lora"))
        self.assertIn("--model-path google/gemma-4-4b-it", cmd)
        self.assertIn("--dataset mlx_dataset", cmd)
        self.assertIn("--batch-size 2", cmd)
        self.assertIn("--iters 500", cmd)
        self.assertIn("--learning-rate 2e-05", cmd)
        self.assertIn("--lora-rank 16", cmd)
        self.assertIn("--lora-alpha 32", cmd)
        self.assertIn("--steps-per-save 50", cmd)
        self.assertIn("--output-path adapters/01_gemma4", cmd)
        self.assertIn("--grad-checkpoint", cmd)
        self.assertIn("--train-on-completions", cmd)
        # Standard mlx_lm flags must NOT be used
        self.assertNotIn("--config", cmd)
        self.assertNotIn("--mask-prompt", cmd)

    def test_lora_run_config_serialization(self):
        cfg = LoraRunConfig(
            model="google/gemma-4-4b-it",
            engine="mlx_vlm",
            train_vision=True,
        )
        d = cfg.to_dict()
        self.assertEqual(d["engine"], "mlx_vlm")
        self.assertTrue(d["train_vision"])

        restored = LoraRunConfig.from_dict(d)
        self.assertEqual(restored.engine, "mlx_vlm")
        self.assertTrue(restored.train_vision)

    def test_multi_lora_run_config_vlm_propagation(self):
        multi = MultiLoraRunConfig(
            model="google/gemma-4-4b-it",
            engine="mlx_vlm",
            iters=[200, 400],
            lora_rank=[8, 16],
        )
        runs = multi.generate_runs()
        self.assertEqual(len(runs), 4)
        for r in runs:
            self.assertEqual(r.engine, "mlx_vlm")
            self.assertEqual(r.model, "google/gemma-4-4b-it")


class TestVlmQueueAndRunner(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_vlm_queue_"))
        self.mgr = QueueManager(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_generate_standalone_script_mixed_queue(self):
        # Run 1: MLX-LM
        r1 = LoraRunConfig(name="run_lm", model="meta-llama/Llama-3.2-3B", engine="mlx_lm")
        # Run 2: MLX-VLM
        r2 = LoraRunConfig(name="run_vlm", model="google/gemma-4-4b-it", engine="mlx_vlm")

        self.mgr.add_run(r1)
        self.mgr.add_run(r2)

        script_path = self.mgr.generate_standalone_script()
        content = script_path.read_text(encoding="utf-8")

        self.assertIn("mlx_lm.lora --config", content)
        self.assertIn("mlx_vlm.lora --model-path google/gemma-4-4b-it", content)

    @patch("mlx_commander.lora.runner.is_engine_installed")
    def test_execute_single_run_blocks_missing_vlm_dependency(self, mock_is_installed):
        mock_is_installed.return_value = False

        data_dir = self.test_dir / "dataset"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "train.jsonl").write_text('{"text": "hi"}\n', encoding="utf-8")

        r = LoraRunConfig(
            name="run_vlm_missing",
            model="google/gemma-4-4b-it",
            engine="mlx_vlm",
            data=str(data_dir),
        )
        self.mgr.add_run(r)

        code = execute_single_run(self.mgr, r.id, enable_wandb=False)
        self.assertEqual(code, 1)

        updated = self.mgr.get_run(r.id)
        self.assertEqual(updated.status, "failed")
        self.assertIn("pip install \"mlx-vlm[train]\"", updated.error_message)


class TestVlmTuiIntegration(unittest.TestCase):
    def test_state_inspect_current_model_syncs_vlm_engine(self):
        from mlx_commander.tui.state import CommanderState

        state = CommanderState()
        # Initial default is mlx-community/Llama-3.2-3B-Instruct-4bit -> mlx_lm
        self.assertEqual(state.lora_config.engine, "mlx_lm")

        # Set to Gemma 4 VLM
        state.lora_config.model = "google/gemma-4-4b-it"
        meta = state.inspect_current_model(force=True)
        self.assertTrue(meta.is_vlm)
        self.assertEqual(meta.engine, "mlx_vlm")
        self.assertEqual(state.lora_config.engine, "mlx_vlm")
        self.assertEqual(state.multi_lora_config.engine, "mlx_vlm")

    @patch("mlx_commander.tui.app.show_error_dialog")
    @patch("mlx_commander.lora.model_info.is_engine_installed")
    def test_validate_preconditions_blocks_missing_vlm(self, mock_is_installed, mock_err_dialog):
        from mlx_commander.tui.app import validate_lora_queue_preconditions

        mock_is_installed.return_value = False

        tmp_data = Path(tempfile.mkdtemp(prefix="test_vlm_data_"))
        try:
            (tmp_data / "train.jsonl").write_text('{"text": "hello"}\n', encoding="utf-8")
            valid = validate_lora_queue_preconditions(None, "google/gemma-4-4b-it", str(tmp_data))
            self.assertFalse(valid)
            mock_err_dialog.assert_called_once()
            call_args = mock_err_dialog.call_args[0]
            self.assertIn("Missing Dependency: mlx-vlm", call_args[1])
            self.assertIn("pip install \"mlx-vlm[train]\"", call_args[2])
        finally:
            shutil.rmtree(tmp_data, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
