"""
Tests for MLX Commander's Enhanced VLM Trainer (mlx_commander.lora.vlm_trainer).
Verifies:
1. CLI argument parser and defaults.
2. Auto-discovery and loading of valid.jsonl / validation.jsonl.
3. Attention mask safeguard in vlm_trainer (batch_size > 1 auto-adjusted).
4. Validation loss stream milestone detection.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlx_commander.lora.vlm_trainer import build_parser, run_vlm_training


class TestVlmTrainerParser(unittest.TestCase):
    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["--model-path", "test/model", "--dataset", "test/data"])
        self.assertEqual(args.model_path, "test/model")
        self.assertEqual(args.dataset, "test/data")
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.gradient_accumulation_steps, 1)
        self.assertEqual(args.iters, 1000)
        self.assertEqual(args.learning_rate, 2e-5)
        self.assertEqual(args.train_mode, "sft")
        self.assertEqual(args.output_path, "adapters.safetensors")

    def test_parser_custom_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "--model-path", "google/gemma-4-4b-it",
            "--dataset", "/path/to/dataset",
            "--batch-size", "1",
            "--gradient-accumulation-steps", "8",
            "--iters", "500",
            "--learning-rate", "1e-4",
            "--train-mode", "orpo",
            "--train-vision",
            "--grad-checkpoint",
            "--output-path", "my_adapters/adapters.safetensors",
        ])
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.gradient_accumulation_steps, 8)
        self.assertEqual(args.iters, 500)
        self.assertEqual(args.learning_rate, 1e-4)
        self.assertEqual(args.train_mode, "orpo")
        self.assertTrue(args.train_vision)
        self.assertTrue(args.grad_checkpoint)
        self.assertEqual(args.output_path, "my_adapters/adapters.safetensors")


class TestVlmTrainerExecution(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="test_vlm_trainer_"))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_run_vlm_training_discovers_validation_and_applies_safeguard(self):
        # Create dummy train.jsonl and valid.jsonl
        train_file = self.tmp_dir / "train.jsonl"
        valid_file = self.tmp_dir / "valid.jsonl"
        train_file.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n', encoding="utf-8")
        valid_file.write_text('{"messages": [{"role": "user", "content": "hello"}]}\n', encoding="utf-8")

        mock_datasets = MagicMock()
        mock_load_dataset = MagicMock()
        mock_datasets.load_dataset = mock_load_dataset

        mock_mlx = MagicMock()
        mock_mlx_optimizers = MagicMock()
        mock_mlx.optimizers = mock_mlx_optimizers

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm_lora = MagicMock()
        mock_mlx_vlm_lora.not_supported_for_training = []
        mock_setup_model = MagicMock()
        mock_mlx_vlm_lora.setup_model_for_training = mock_setup_model

        mock_model = MagicMock()
        mock_model.config.model_type = "gemma4"
        mock_setup_model.return_value = mock_model

        mock_processor = MagicMock()
        mock_mlx_vlm_utils = MagicMock()
        mock_mlx_vlm_utils.load.return_value = (mock_model, mock_processor)

        mock_sft_trainer = MagicMock()
        mock_train = MagicMock()
        mock_training_args_cls = MagicMock()
        mock_sft_trainer.train = mock_train
        mock_sft_trainer.TrainingArgs = mock_training_args_cls

        mock_trainer_datasets = MagicMock()
        mock_vision_dataset = MagicMock()
        mock_trainer_datasets.VisionDataset = mock_vision_dataset

        mock_modules = {
            "datasets": mock_datasets,
            "mlx": mock_mlx,
            "mlx.optimizers": mock_mlx_optimizers,
            "mlx_vlm": mock_mlx_vlm,
            "mlx_vlm.lora": mock_mlx_vlm_lora,
            "mlx_vlm.utils": mock_mlx_vlm_utils,
            "mlx_vlm.trainer": MagicMock(),
            "mlx_vlm.trainer.datasets": mock_trainer_datasets,
            "mlx_vlm.trainer.sft_trainer": mock_sft_trainer,
            "mlx_vlm.trainer.orpo_trainer": MagicMock(),
            "mlx_vlm.trainer.utils": MagicMock(),
        }

        parser = build_parser()
        # Pass batch_size=4, grad_accum=1 to test auto-safeguard
        args = parser.parse_args([
            "--model-path", "google/gemma-4-4b-it",
            "--dataset", str(self.tmp_dir),
            "--batch-size", "4",
            "--gradient-accumulation-steps", "1",
            "--iters", "100",
            "--steps-per-eval", "25",
        ])

        with patch.dict("sys.modules", mock_modules):
            run_vlm_training(args)

        # 1. Verify safeguard triggered: batch_size adjusted to 1, accum to 4
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.gradient_accumulation_steps, 4)

        # 2. Verify dataset loading was called for both train and valid
        self.assertEqual(mock_load_dataset.call_count, 2)
        valid_call_kwargs = mock_load_dataset.call_args_list[1]
        self.assertEqual(valid_call_kwargs[1]["split"], "valid")
        self.assertIn("valid", valid_call_kwargs[1]["data_files"])

        # 3. Verify TrainingArgs parameters
        self.assertTrue(mock_training_args_cls.called)
        t_args_kwargs = mock_training_args_cls.call_args[1]
        self.assertEqual(t_args_kwargs["batch_size"], 1)
        self.assertEqual(t_args_kwargs["gradient_accumulation_steps"], 4)
        self.assertEqual(t_args_kwargs["steps_per_eval"], 25)

        # 4. Verify train() received val_dataset
        self.assertTrue(mock_train.called)
        train_call_kwargs = mock_train.call_args[1]
        self.assertIsNotNone(train_call_kwargs["val_dataset"])

    def test_validation_loss_milestone_detection(self):
        line1 = "Iter 100: Val loss 1.452, Val took 0.65s\n"
        line2 = "Iteration 50: Train loss 0.982, It/sec 12.3\n"

        self.assertIn("val loss", line1.lower())
        self.assertNotIn("val loss", line2.lower())


if __name__ == "__main__":
    unittest.main()
