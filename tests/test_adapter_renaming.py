import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from mlx_commander.lora.config import LoraRunConfig
from mlx_commander.lora.runner import rename_saved_adapters


class TestAdapterRenaming(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.adapter_dir = Path(self.temp_dir) / "adapters" / "01_lora_r16_a32_test"
        self.adapter_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = LoraRunConfig(
            model="mlx-community/Llama-3.2-3B-Instruct-4bit",
            fine_tune_type="lora",
            lora_rank=16,
            lora_alpha=32.0,
            learning_rate=1e-5,
            batch_size=4,
            iters=1000,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_rename_step_checkpoints(self):
        # Create mock step checkpoints
        step100 = self.adapter_dir / "0000100_adapters.safetensors"
        step200 = self.adapter_dir / "0000200_adapters.safetensors"
        step100.write_text("weights_100", encoding="utf-8")
        step200.write_text("weights_200", encoding="utf-8")

        renamed = rename_saved_adapters(self.adapter_dir, config=self.cfg)
        self.assertEqual(len(renamed), 2)

        expected_100 = (
            self.adapter_dir
            / "0000100_adapters_lora_r16_a32_lr1e-5_b4_i1000_Llama-3.2-3B-Instruct-4bit.safetensors"
        )
        expected_200 = (
            self.adapter_dir
            / "0000200_adapters_lora_r16_a32_lr1e-5_b4_i1000_Llama-3.2-3B-Instruct-4bit.safetensors"
        )

        self.assertTrue(expected_100.is_file())
        self.assertTrue(expected_200.is_file())
        self.assertEqual(expected_100.read_text(encoding="utf-8"), "weights_100")
        self.assertEqual(expected_200.read_text(encoding="utf-8"), "weights_200")

        # Verify symlinks were created for backward compatibility
        self.assertTrue(step100.is_symlink())
        self.assertTrue(step200.is_symlink())
        self.assertEqual(os.readlink(str(step100)), expected_100.name)
        self.assertEqual(os.readlink(str(step200)), expected_200.name)

        # Idempotency check: running again changes nothing
        second_run = rename_saved_adapters(self.adapter_dir, config=self.cfg)
        self.assertEqual(len(second_run), 0)

    def test_rename_final_adapter(self):
        # Create mock final weights
        final_file = self.adapter_dir / "adapters.safetensors"
        final_file.write_text("final_weights", encoding="utf-8")

        renamed = rename_saved_adapters(
            self.adapter_dir, config=self.cfg, final_iters=self.cfg.iters
        )
        self.assertEqual(len(renamed), 1)

        expected_final = (
            self.adapter_dir
            / "0001000_adapters_lora_r16_a32_lr1e-5_b4_i1000_Llama-3.2-3B-Instruct-4bit.safetensors"
        )
        self.assertTrue(expected_final.is_file())
        self.assertEqual(expected_final.read_text(encoding="utf-8"), "final_weights")

        # Verify adapters.safetensors is a symlink pointing to the renamed file
        self.assertTrue(final_file.is_symlink())
        self.assertEqual(os.readlink(str(final_file)), expected_final.name)

    def test_rename_final_when_step_checkpoint_already_exists(self):
        # Suppose checkpoint 1000 was already saved and renamed
        expected_final = (
            self.adapter_dir
            / "0001000_adapters_lora_r16_a32_lr1e-5_b4_i1000_Llama-3.2-3B-Instruct-4bit.safetensors"
        )
        expected_final.write_text("weights_1000", encoding="utf-8")

        # MLX also saves adapters.safetensors at completion
        final_file = self.adapter_dir / "adapters.safetensors"
        final_file.write_text("weights_1000", encoding="utf-8")

        rename_saved_adapters(
            self.adapter_dir, config=self.cfg, final_iters=self.cfg.iters
        )

        # final_file should now be a symlink to expected_final without error
        self.assertTrue(final_file.is_symlink())
        self.assertEqual(os.readlink(str(final_file)), expected_final.name)

    def test_fallback_using_adapter_config_json(self):
        # Create adapter_config.json
        ac_file = self.adapter_dir / "adapter_config.json"
        ac_file.write_text(
            json.dumps(
                {
                    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
                    "lora_parameters": {"rank": 8, "scale": 16.0, "dropout": 0.0},
                }
            ),
            encoding="utf-8",
        )

        step50 = self.adapter_dir / "0000050_adapters.safetensors"
        step50.write_text("weights_50", encoding="utf-8")

        # Call with config=None
        renamed = rename_saved_adapters(self.adapter_dir, config=None)
        self.assertEqual(len(renamed), 1)

        renamed_path = renamed[0][1]
        self.assertTrue(renamed_path.name.startswith("0000050_adapters_lora_r8_a16_"))
        self.assertIn("Qwen2.5-7B-Instruct-4bit", renamed_path.name)
        self.assertTrue(step50.is_symlink())

    def test_cli_rename_adapters(self):
        import subprocess
        import sys
        # Write config yaml
        cfg_file = Path(self.temp_dir) / "config.yaml"
        cfg_file.write_text(self.cfg.to_mlx_yaml(), encoding="utf-8")

        step300 = self.adapter_dir / "0000300_adapters.safetensors"
        step300.write_text("weights_300", encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "mlx_commander.lora.runner",
            "--rename-adapters",
            str(self.adapter_dir),
            "--config",
            str(cfg_file),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        self.assertIn("[OK] Renamed:", res.stdout)
        self.assertIn("0000300_adapters_lora_r16_a32_", res.stdout)
        self.assertTrue(step300.is_symlink())


if __name__ == "__main__":
    unittest.main()
