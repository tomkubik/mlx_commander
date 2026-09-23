"""
Tests for MLX-VLM Attention Mask Safeguards and Gradient Accumulation Tweaks.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlx_commander.lora.config import LoraRunConfig
from mlx_commander.lora.multi_config import MultiLoraRunConfig
from mlx_commander.lora.queue import QueueManager
from mlx_commander.lora.runner import execute_single_run
from mlx_commander.lora.vlm_safeguards import (
    calculate_vlm_batch_tweak,
    calculate_vlm_sweep_tweak,
    should_suggest_vlm_sweep_tweak,
    should_suggest_vlm_tweak,
)
from mlx_commander.tui.app import (
    prompt_vlm_batch_tweak,
    prompt_vlm_sweep_batch_tweak,
    validate_lora_queue_preconditions,
)


class TestVlmSafeguardsCalculations(unittest.TestCase):
    def test_calculate_vlm_batch_tweak(self):
        # Default batch_size 4, accum 1 -> batch 1, accum 4
        b, gas = calculate_vlm_batch_tweak(4, 1)
        self.assertEqual(b, 1)
        self.assertEqual(gas, 4)

        # batch_size 2, accum 4 -> batch 1, accum 8
        b, gas = calculate_vlm_batch_tweak(2, 4)
        self.assertEqual(b, 1)
        self.assertEqual(gas, 8)

        # batch_size 1, accum 4 -> batch 1, accum 4 (unchanged)
        b, gas = calculate_vlm_batch_tweak(1, 4)
        self.assertEqual(b, 1)
        self.assertEqual(gas, 4)

        # Edge cases (zero or negative)
        b, gas = calculate_vlm_batch_tweak(0, -1)
        self.assertEqual(b, 1)
        self.assertEqual(gas, 1)

    def test_calculate_vlm_sweep_tweak(self):
        # Sweeping over batch_size=[2, 4], grad_accums=[1]
        b_list, gas_list = calculate_vlm_sweep_tweak([2, 4], [1])
        self.assertEqual(b_list, [1])
        self.assertEqual(gas_list, [2, 4])

        # Single batch size 4
        b_list, gas_list = calculate_vlm_sweep_tweak([4], [1])
        self.assertEqual(b_list, [1])
        self.assertEqual(gas_list, [4])

        # Complex sweep: batch=[1, 2, 4], accum=[1, 2]
        # Products: 1*1=1, 1*2=2, 2*1=2, 2*2=4, 4*1=4, 4*2=8 -> [1, 2, 4, 8]
        b_list, gas_list = calculate_vlm_sweep_tweak([1, 2, 4], [1, 2])
        self.assertEqual(b_list, [1])
        self.assertEqual(gas_list, [1, 2, 4, 8])

    def test_should_suggest_vlm_tweak(self):
        self.assertTrue(should_suggest_vlm_tweak("mlx_vlm", 4))
        self.assertTrue(should_suggest_vlm_tweak("mlx_vlm", 2))
        self.assertFalse(should_suggest_vlm_tweak("mlx_vlm", 1))
        self.assertFalse(should_suggest_vlm_tweak("mlx_lm", 4))

    def test_should_suggest_vlm_sweep_tweak(self):
        self.assertTrue(should_suggest_vlm_sweep_tweak("mlx_vlm", [1, 2, 4]))
        self.assertTrue(should_suggest_vlm_sweep_tweak("mlx_vlm", [4]))
        self.assertFalse(should_suggest_vlm_sweep_tweak("mlx_vlm", [1]))
        self.assertFalse(should_suggest_vlm_sweep_tweak("mlx_lm", [2, 4]))


class TestVlmRunnerSafeguard(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_runner_safeguard_"))
        self.mgr = QueueManager(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("mlx_commander.lora.runner.is_engine_installed")
    @patch("subprocess.Popen")
    def test_runner_auto_tweaks_vlm_batch_size(self, mock_popen, mock_installed):
        mock_installed.return_value = True

        mock_proc = MagicMock()
        mock_proc.stdout.readline.side_effect = [
            "Iter 1: Val loss 1.234\n",
            "",
        ]
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        data_dir = self.test_dir / "dataset"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "train.jsonl").write_text('{"messages": [{"role": "user", "content": "hi"}]}\n', encoding="utf-8")

        r = LoraRunConfig(
            name="run_vlm_guard",
            model="google/gemma-4-4b-it",
            engine="mlx_vlm",
            data=str(data_dir),
            batch_size=4,
            grad_accumulation_steps=1,
            iters=100,
        )
        self.mgr.add_run(r)

        code = execute_single_run(self.mgr, r.id, enable_wandb=False)
        self.assertEqual(code, 0)

        # Check subprocess invocation: batch-size must be 1, gradient-accumulation-steps must be 4
        call_args = mock_popen.call_args[0][0]
        self.assertIn("--batch-size", call_args)
        b_idx = call_args.index("--batch-size")
        self.assertEqual(call_args[b_idx + 1], "1")

        self.assertIn("--gradient-accumulation-steps", call_args)
        gas_idx = call_args.index("--gradient-accumulation-steps")
        self.assertEqual(call_args[gas_idx + 1], "4")

    @patch("mlx_commander.lora.runner.is_engine_installed")
    @patch("subprocess.Popen")
    def test_runner_respects_disable_safeguard_env(self, mock_popen, mock_installed):
        mock_installed.return_value = True

        mock_proc = MagicMock()
        mock_proc.stdout.readline.side_effect = ["", ""]
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        data_dir = self.test_dir / "dataset"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "train.jsonl").write_text('{"messages": [{"role": "user", "content": "hi"}]}\n', encoding="utf-8")

        r = LoraRunConfig(
            name="run_vlm_disabled_guard",
            model="google/gemma-4-4b-it",
            engine="mlx_vlm",
            data=str(data_dir),
            batch_size=4,
            grad_accumulation_steps=1,
            iters=100,
        )
        self.mgr.add_run(r)

        with patch.dict(os.environ, {"MLX_DISABLE_VLM_SAFEGUARD": "1"}):
            code = execute_single_run(self.mgr, r.id, enable_wandb=False)
            self.assertEqual(code, 0)

            call_args = mock_popen.call_args[0][0]
            b_idx = call_args.index("--batch-size")
            self.assertEqual(call_args[b_idx + 1], "4")


class TestVlmTuiPrompts(unittest.TestCase):
    @patch("mlx_commander.tui.app.show_choice_dialog")
    def test_prompt_vlm_batch_tweak_accepted(self, mock_choice):
        cfg = LoraRunConfig(
            engine="mlx_vlm",
            batch_size=4,
            grad_accumulation_steps=1,
        )
        mock_win = MagicMock()
        mock_choice.return_value = "Auto-tweak: batch=1, grad_accum=4 (Recommended)"

        accepted = prompt_vlm_batch_tweak(mock_win, cfg)
        self.assertTrue(accepted)
        self.assertEqual(cfg.batch_size, 1)
        self.assertEqual(cfg.grad_accumulation_steps, 4)

    @patch("mlx_commander.tui.app.show_choice_dialog")
    def test_prompt_vlm_batch_tweak_kept(self, mock_choice):
        cfg = LoraRunConfig(
            engine="mlx_vlm",
            batch_size=4,
            grad_accumulation_steps=1,
        )
        mock_win = MagicMock()
        mock_choice.return_value = "Keep batch_size=4 (Experimental)"

        accepted = prompt_vlm_batch_tweak(mock_win, cfg)
        self.assertTrue(accepted)
        self.assertEqual(cfg.batch_size, 4)
        self.assertEqual(cfg.grad_accumulation_steps, 1)

    @patch("mlx_commander.tui.app.show_choice_dialog")
    def test_prompt_vlm_batch_tweak_cancelled(self, mock_choice):
        cfg = LoraRunConfig(
            engine="mlx_vlm",
            batch_size=4,
            grad_accumulation_steps=1,
        )
        mock_win = MagicMock()
        mock_choice.return_value = "Cancel"

        accepted = prompt_vlm_batch_tweak(mock_win, cfg)
        self.assertFalse(accepted)

    @patch("mlx_commander.tui.app.show_choice_dialog")
    def test_prompt_vlm_sweep_batch_tweak(self, mock_choice):
        mcfg = MultiLoraRunConfig(
            engine="mlx_vlm",
            batch_size=[2, 4],
            grad_accumulation_steps=[1],
        )
        mock_win = MagicMock()
        mock_choice.return_value = "Auto-tweak: batch=[1], grad_accum=[2, 4] (Recommended)"

        accepted = prompt_vlm_sweep_batch_tweak(mock_win, mcfg)
        self.assertTrue(accepted)
        self.assertEqual(mcfg.batch_size, [1])
        self.assertEqual(mcfg.grad_accumulation_steps, [2, 4])

    @patch("mlx_commander.lora.model_info.is_engine_installed")
    @patch("mlx_commander.tui.app.prompt_vlm_batch_tweak")
    def test_validate_preconditions_triggers_batch_tweak(self, mock_tweak, mock_installed):
        mock_installed.return_value = True
        mock_tweak.return_value = True

        tmp_data = Path(tempfile.mkdtemp(prefix="test_precon_data_"))
        try:
            (tmp_data / "train.jsonl").write_text('{"messages": [{"role": "user", "content": "1"}]}\n', encoding="utf-8")
            cfg = LoraRunConfig(
                model="google/gemma-4-4b-it",
                engine="mlx_vlm",
                batch_size=4,
            )
            mock_win = MagicMock()
            valid = validate_lora_queue_preconditions(mock_win, cfg.model, str(tmp_data), config=cfg)
            self.assertTrue(valid)
            mock_tweak.assert_called_once_with(mock_win, cfg)
        finally:
            shutil.rmtree(tmp_data, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
