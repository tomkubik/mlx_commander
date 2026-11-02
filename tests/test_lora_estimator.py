import unittest
from unittest.mock import patch
from mlx_commander.lora.config import LoraRunConfig
from mlx_commander.lora.estimator import (
    calculate_implied_epochs,
    estimate_duration,
    estimate_eval_throughput,
    estimate_peak_memory,
    get_apple_silicon_chip,
    get_chip_memory_bandwidth_gbps,
    get_hardware_memory_bytes,
    parse_model_param_billions,
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

    def test_parse_model_param_billions(self):
        self.assertEqual(parse_model_param_billions("mlx-community/Llama-3.2-1B-Instruct-4bit"), 1.0)
        self.assertEqual(parse_model_param_billions("mlx-community/Llama-3.2-3B-Instruct-4bit"), 3.0)
        self.assertEqual(parse_model_param_billions("google/gemma-4-4b-it"), 4.0)
        self.assertEqual(parse_model_param_billions("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"), 8.0)
        self.assertEqual(parse_model_param_billions("mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"), 14.0)
        self.assertEqual(parse_model_param_billions("meta-llama/Llama-3.1-70B-Instruct-4bit"), 70.0)

    def test_estimate_eval_throughput_inverse_scaling(self):
        """Verify tokens/sec is model-dependent and strictly scales inversely with model parameter size."""
        res_1b = estimate_eval_throughput("mlx-community/Llama-3.2-1B-Instruct-4bit", total_samples=50)
        res_8b = estimate_eval_throughput("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", total_samples=50)
        res_70b = estimate_eval_throughput("meta-llama/Llama-3.1-70B-Instruct-4bit", total_samples=50)

        # Smaller models must generate significantly faster (higher tok/s)
        self.assertGreater(res_1b["est_tps"], res_8b["est_tps"])
        self.assertGreater(res_8b["est_tps"], res_70b["est_tps"])

        # Smaller models must complete evaluation significantly faster (lower duration)
        self.assertLess(res_1b["est_total_sec"], res_8b["est_total_sec"])
        self.assertLess(res_8b["est_total_sec"], res_70b["est_total_sec"])

        # Model size in GB must scale with parameter count
        self.assertLess(res_1b["model_gb"], res_8b["model_gb"])
        self.assertLess(res_8b["model_gb"], res_70b["model_gb"])

    def test_estimate_eval_throughput_quantization_speedup(self):
        """Verify 4-bit quantization yields higher tok/s and smaller model footprint than fp16."""
        res_q4 = estimate_eval_throughput("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", total_samples=50)
        res_fp16 = estimate_eval_throughput("meta-llama/Meta-Llama-3.1-8B-Instruct", total_samples=50)

        self.assertEqual(res_q4["quant_label"], "4-bit")
        self.assertEqual(res_fp16["quant_label"], "fp16")
        self.assertGreater(res_q4["est_tps"], res_fp16["est_tps"])
        self.assertLess(res_q4["model_gb"], res_fp16["model_gb"])
        self.assertLess(res_q4["est_total_sec"], res_fp16["est_total_sec"])

    def test_estimate_eval_throughput_hardware_bandwidth(self):
        """Verify throughput scales with Apple Silicon chip memory bandwidth (Ultra > Max > Pro > Base)."""
        m = "mlx-community/Llama-3.2-3B-Instruct-4bit"
        tps_ultra = estimate_eval_throughput(m, chip_name="Apple M2 Ultra")["est_tps"]
        tps_max = estimate_eval_throughput(m, chip_name="Apple M3 Max")["est_tps"]
        tps_pro = estimate_eval_throughput(m, chip_name="Apple M3 Pro")["est_tps"]
        tps_base = estimate_eval_throughput(m, chip_name="Apple M1")["est_tps"]

        self.assertGreater(tps_ultra, tps_max)
        self.assertGreater(tps_max, tps_pro)
        self.assertGreater(tps_pro, tps_base)

    def test_estimate_duration_and_memory_scale_with_validation_set_size(self):
        import tempfile
        from pathlib import Path
        from mlx_commander.lora.estimator import clear_estimator_cache, get_validation_record_count

        clear_estimator_cache()
        with tempfile.TemporaryDirectory() as td:
            ds_dir = Path(td)
            valid_f = ds_dir / "valid.jsonl"
            with open(valid_f, "w", encoding="utf-8") as f:
                for i in range(400):
                    f.write(f'{{"prompt": "Q{i}", "completion": "A{i}"}}\n')

            self.assertEqual(get_validation_record_count(str(ds_dir)), 400)

            cfg_all = LoraRunConfig(
                model="mlx-community/Llama-3.2-3B-Instruct-4bit",
                data=str(ds_dir),
                iters=500,
                batch_size=4,
                steps_per_eval=100,
                val_batches=-1,  # All 100 batches (400 / 4)
            )
            cfg_capped = LoraRunConfig(
                model="mlx-community/Llama-3.2-3B-Instruct-4bit",
                data=str(ds_dir),
                iters=500,
                batch_size=4,
                steps_per_eval=100,
                val_batches=10,  # Only 10 batches
            )

            dur_no_val = estimate_duration(cfg_all, val_records=0)
            dur_capped = estimate_duration(cfg_capped)
            dur_all = estimate_duration(cfg_all)

            self.assertEqual(dur_all["val_records"], 400)
            self.assertEqual(dur_all["effective_val_batches"], 100)
            self.assertEqual(dur_capped["effective_val_batches"], 10)
            self.assertGreater(dur_capped["seconds"], dur_no_val["seconds"])
            self.assertGreater(dur_all["seconds"], dur_capped["seconds"])
            self.assertGreater(dur_all["val_seconds"], dur_capped["val_seconds"])

            mem_no_val = estimate_peak_memory(cfg_all, val_records=0)
            mem_capped = estimate_peak_memory(cfg_capped)
            mem_all = estimate_peak_memory(cfg_all)
            self.assertEqual(mem_no_val["val_mem_gb"], 0.0)
            self.assertGreater(mem_capped["val_mem_gb"], 0.0)
            self.assertGreater(mem_all["val_mem_gb"], mem_capped["val_mem_gb"])

