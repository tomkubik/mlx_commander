import unittest
from unittest.mock import patch
from mlx_commander.lora.config import LoraRunConfig
from mlx_commander.lora.estimator import (
    calculate_implied_epochs,
    estimate_duration,
    estimate_peak_memory,
    get_hardware_memory_bytes,
    get_apple_silicon_chip,
)


class TestLoraEstimator(unittest.TestCase):
    def test_calculate_implied_epochs(self):
        # 1000 iters * 4 batch_size = 4000 samples. Total records = 1000 -> 4.0 epochs
        self.assertAlmostEqual(calculate_implied_epochs(1000, 4, 1000), 4.0)

        # 500 iters * 2 batch_size = 1000 samples. Total records = 2000 -> 0.5 epochs
        self.assertAlmostEqual(calculate_implied_epochs(500, 2, 2000), 0.5)

        # Total records = 0 or negative -> None
        self.assertIsNone(calculate_implied_epochs(1000, 4, 0))
        self.assertIsNone(calculate_implied_epochs(1000, 4, -5))

    def test_hardware_memory_bytes(self):
        mem_bytes = get_hardware_memory_bytes()
        self.assertIsInstance(mem_bytes, int)
        self.assertGreater(mem_bytes, 0)

    def test_hardware_chip_name(self):
        chip = get_apple_silicon_chip()
        self.assertIsInstance(chip, str)
        self.assertGreater(len(chip), 0)

    @patch("mlx_commander.lora.estimator.get_hardware_memory_bytes", return_value=16 * 1024**3)
    def test_estimate_peak_memory_safe(self, mock_hw):
        cfg = LoraRunConfig(
            model="mlx-community/Llama-3.2-1B-Instruct-4bit",
            batch_size=2,
            max_seq_length=1024,
            lora_rank=8,
        )
        res = estimate_peak_memory(cfg)
        self.assertIn("est_gb", res)
        self.assertIn("hardware_gb", res)
        self.assertIn("safety_level", res)
        self.assertEqual(res["safety_level"], "SAFE")
        self.assertIn("SAFE", res["badge"])
        # Verify no decimals in integer GB values
        self.assertIsInstance(res["peak_gb"], int)
        self.assertIsInstance(res["model_gb"], int)
        self.assertIsInstance(res["act_gb"], int)
        self.assertIsInstance(res["lora_opt_gb"], int)
        self.assertIsInstance(res["total_ram_gb"], int)
        self.assertNotIn(".", res["badge"].replace("Est.", "").split("[")[0])

    @patch("mlx_commander.lora.estimator.get_hardware_memory_bytes", return_value=8 * 1024**3)
    def test_estimate_peak_memory_tight_and_oom(self, mock_hw):
        # 8B model with high seq length and batch size on 8GB machine -> OOM RISK
        cfg = LoraRunConfig(
            model="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            batch_size=8,
            max_seq_length=4096,
            lora_rank=32,
        )
        res = estimate_peak_memory(cfg)
        self.assertEqual(res["safety_level"], "OOM RISK")
        self.assertIn("OOM RISK", res["badge"])

    def test_estimate_duration_and_eta(self):
        cfg = LoraRunConfig(
            model="mlx-community/Llama-3.2-3B-Instruct-4bit",
            iters=600,
            batch_size=4,
            max_seq_length=2048,
        )
        res = estimate_duration(cfg)
        self.assertIn("seconds", res)
        self.assertGreater(res["seconds"], 0)
        self.assertIn("duration_str", res)
        self.assertIn("eta_clock", res)
        self.assertIn("iters_per_sec", res)

    def test_hardware_estimator_caching(self):
        from mlx_commander.lora.estimator import clear_estimator_cache
        clear_estimator_cache()
        # Verify lru_cache info
        info1 = get_hardware_memory_bytes.cache_info()
        get_hardware_memory_bytes()
        get_hardware_memory_bytes()
        info2 = get_hardware_memory_bytes.cache_info()
        self.assertGreaterEqual(info2.hits, info1.hits + 1)

    def test_state_record_count_and_estimate_caching(self):
        import tempfile
        from pathlib import Path
        from mlx_commander.tui.state import CommanderState

        state = CommanderState()
        with tempfile.TemporaryDirectory() as td:
            train_f = Path(td) / "train.jsonl"
            train_f.write_text('{"prompt": "hi", "completion": "hello"}\n{"prompt": "a", "completion": "b"}\n')
            state.lora_config.data = td

            # First call reads file
            c1 = state.get_train_record_count()
            self.assertEqual(c1, 2)

            # Second call should hit cache
            c2 = state.get_train_record_count()
            self.assertEqual(c2, 2)
            self.assertIn(str(train_f.resolve()), state._cached_train_counts)

            # Check estimate caching
            est1 = state.get_memory_estimate()
            est2 = state.get_memory_estimate()
            self.assertIs(est1, est2)

    def test_estimate_peak_memory_local_dir_with_safetensors(self):
        """Verify selecting a real local folder with .safetensors files never causes param_b UnboundLocalError."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            # Create dummy config.json and multiple safetensors shards
            (p / "config.json").write_text('{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}')
            (p / "model-00001-of-00004.safetensors").write_bytes(b"0" * (1024 * 1024 * 10))  # 10 MB
            (p / "model-00002-of-00004.safetensors").write_bytes(b"0" * (1024 * 1024 * 10))  # 10 MB

            cfg = LoraRunConfig(model=str(p))
            res = estimate_peak_memory(cfg)
            self.assertIn("peak_gb", res)
            self.assertIn("act_gb", res)
            self.assertGreater(res["peak_gb"], 0)

            # Test passing a specific shard file directly
            cfg_file = LoraRunConfig(model=str(p / "model-00001-of-00004.safetensors"))
            res_file = estimate_peak_memory(cfg_file)
            self.assertIn("peak_gb", res_file)
            self.assertGreater(res_file["peak_gb"], 0)
