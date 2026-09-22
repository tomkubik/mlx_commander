import tempfile
import unittest
from pathlib import Path

from mlx_commander.loader import load_single_local_dataset
from mlx_commander.lora.model_info import (
    inspect_local_model,
    is_model_directory,
    normalize_model_path,
)


class TestModelSelection(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_is_model_directory_with_safetensors_and_config(self):
        model_dir = self.base / "Llama-3.2-3B"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "llama", "architectures": ["LlamaForCausalLM"]}')
        (model_dir / "model-00001-of-00004.safetensors").write_bytes(b"dummy")
        (model_dir / "model-00002-of-00004.safetensors").write_bytes(b"dummy")

        # Folder is recognized as model directory
        self.assertTrue(is_model_directory(model_dir))
        self.assertTrue(is_model_directory(str(model_dir)))

        # Individual files inside the folder are also recognized
        self.assertTrue(is_model_directory(model_dir / "model-00001-of-00004.safetensors"))
        self.assertTrue(is_model_directory(model_dir / "config.json"))

    def test_is_model_directory_false_for_normal_dataset(self):
        ds_dir = self.base / "my_dataset"
        ds_dir.mkdir()
        (ds_dir / "train.jsonl").write_text('{"prompt": "hi", "completion": "hello"}\n')

        self.assertFalse(is_model_directory(ds_dir))
        self.assertFalse(is_model_directory(ds_dir / "train.jsonl"))

    def test_normalize_model_path_from_folder(self):
        model_dir = self.base / "MyModel"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "qwen2"}')

        norm = normalize_model_path(str(model_dir))
        self.assertEqual(norm, str(model_dir.resolve()))

    def test_normalize_model_path_from_shard_file(self):
        model_dir = self.base / "Mistral-7B"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "mistral"}')
        shard = model_dir / "model-00001-of-00004.safetensors"
        shard.write_bytes(b"fake_weights")

        # Pointing at the shard file resolves to the containing folder
        norm = normalize_model_path(str(shard))
        self.assertEqual(norm, str(model_dir.resolve()))

    def test_normalize_model_path_from_config_file(self):
        model_dir = self.base / "Phi-3"
        model_dir.mkdir()
        cfg = model_dir / "config.json"
        cfg.write_text('{"model_type": "phi3"}')

        # Pointing at config.json resolves to the containing folder
        norm = normalize_model_path(str(cfg))
        self.assertEqual(norm, str(model_dir.resolve()))

    def test_inspect_local_model_normalizes_path(self):
        model_dir = self.base / "Qwen-7B"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "qwen2", "num_hidden_layers": 28, "hidden_size": 3584}')
        shard = model_dir / "model.safetensors"
        shard.write_bytes(b"12345678")

        meta = inspect_local_model(str(shard))
        self.assertTrue(meta.is_valid)
        self.assertEqual(meta.path, str(model_dir.resolve()))
        self.assertEqual(meta.architecture, "qwen2")

    def test_load_single_local_dataset_blocks_model_directory(self):
        model_dir = self.base / "Llama-3.2-3B"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "llama", "architectures": ["LlamaForCausalLM"]}')
        (model_dir / "model-00001-of-00004.safetensors").write_bytes(b"dummy")

        # Attempting to load the model directory as a dataset raises ValueError with helpful guidance
        with self.assertRaises(ValueError) as ctx:
            load_single_local_dataset(model_dir)

        self.assertIn("base model directory", str(ctx.exception).lower())
        self.assertIn("mode 2", str(ctx.exception).lower())

    def test_load_single_local_dataset_blocks_model_weight_file(self):
        model_dir = self.base / "Llama-3.2-3B"
        model_dir.mkdir()
        shard = model_dir / "model-00001-of-00004.safetensors"
        shard.write_bytes(b"dummy")

        with self.assertRaises(ValueError) as ctx:
            load_single_local_dataset(shard)

        self.assertIn("model weights file", str(ctx.exception).lower())
        self.assertIn("mode 2", str(ctx.exception).lower())

    def test_is_local_path(self):
        from mlx_commander.lora.model_info import is_local_path

        self.assertFalse(is_local_path("mlx-community/Llama-3.2-3B-Instruct-4bit"))
        self.assertFalse(is_local_path("meta-llama/Llama-3.2-1B-Instruct"))
        self.assertTrue(is_local_path("/Users/tomkubik/Models/Llama"))
        self.assertTrue(is_local_path("./models/Llama"))
        self.assertTrue(is_local_path("~/models/Llama"))
        self.assertTrue(is_local_path("/path/with/mlx-community:Llama-3.2-3B-Instruct"))

    def test_normalize_model_path_heals_colon_to_slash(self):
        # On disk: Models/mlx-community/Llama-3.2-3B-Instruct
        model_dir = self.base / "Models" / "mlx-community" / "Llama-3.2-3B-Instruct"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text('{"model_type": "llama"}')

        # Input: Models/mlx-community:Llama-3.2-3B-Instruct (colon in folder name)
        colon_input = str(self.base / "Models" / "mlx-community:Llama-3.2-3B-Instruct")
        healed = normalize_model_path(colon_input)
        self.assertEqual(healed, str(model_dir.resolve()))

    def test_normalize_model_path_heals_slash_to_colon(self):
        # On disk: Models/mlx-community:Llama-3.2-3B-Instruct (colon in folder name on disk)
        model_dir = self.base / "Models" / "mlx-community:Llama-3.2-3B-Instruct"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text('{"model_type": "llama"}')

        # Input: Models/mlx-community/Llama-3.2-3B-Instruct (slash entered)
        slash_input = str(self.base / "Models" / "mlx-community" / "Llama-3.2-3B-Instruct")
        healed = normalize_model_path(slash_input)
        self.assertEqual(healed, str(model_dir.resolve()))

    def test_execute_single_run_preflight_blocks_nonexistent_local_model(self):
        from mlx_commander.lora.config import LoraRunConfig
        from mlx_commander.lora.queue import QueueManager
        from mlx_commander.lora.runner import execute_single_run

        q_dir = self.base / "queue"
        mgr = QueueManager(q_dir)

        # Dataset with train.jsonl
        ds_dir = self.base / "dataset"
        ds_dir.mkdir()
        (ds_dir / "train.jsonl").write_text('{"prompt": "q", "completion": "a"}\n')

        # Run with nonexistent local model path
        bad_model_path = str(self.base / "NonExistentModel")
        cfg = LoraRunConfig(model=bad_model_path, data=str(ds_dir), iters=10)
        run = mgr.add_run(cfg)

        code = execute_single_run(mgr, run.id, enable_wandb=False)
        self.assertEqual(code, 1)

        # Status updated to failed with clear error message
        updated_r = mgr.get_run(run.id)
        self.assertEqual(updated_r.status, "failed")
        self.assertIn("not found on disk", updated_r.error_message.lower())

    def test_execute_single_run_preflight_blocks_missing_train_data(self):
        from mlx_commander.lora.config import LoraRunConfig
        from mlx_commander.lora.queue import QueueManager
        from mlx_commander.lora.runner import execute_single_run

        q_dir = self.base / "queue"
        mgr = QueueManager(q_dir)

        # Valid model
        model_dir = self.base / "ValidModel"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "llama"}')

        # Dataset folder missing train.jsonl
        empty_ds = self.base / "empty_dataset"
        empty_ds.mkdir()

        cfg = LoraRunConfig(model=str(model_dir), data=str(empty_ds), iters=10)
        run = mgr.add_run(cfg)

        code = execute_single_run(mgr, run.id, enable_wandb=False)
        self.assertEqual(code, 1)

        updated_r = mgr.get_run(run.id)
        self.assertEqual(updated_r.status, "failed")
        self.assertIn("train.jsonl", updated_r.error_message.lower())

    def test_execute_single_run_auto_heals_colon_model_path(self):
        from unittest.mock import patch, MagicMock
        from mlx_commander.lora.config import LoraRunConfig
        from mlx_commander.lora.queue import QueueManager
        from mlx_commander.lora.runner import execute_single_run

        q_dir = self.base / "queue"
        mgr = QueueManager(q_dir)

        # Real model on disk: Models/mlx-community/Llama-3.2-3B
        model_dir = self.base / "Models" / "mlx-community" / "Llama-3.2-3B"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text('{"model_type": "llama"}')

        # Valid dataset
        ds_dir = self.base / "dataset"
        ds_dir.mkdir()
        (ds_dir / "train.jsonl").write_text('{"prompt": "q", "completion": "a"}\n')

        # Run configured with colon in path
        colon_model = str(self.base / "Models" / "mlx-community:Llama-3.2-3B")
        cfg = LoraRunConfig(model=colon_model, data=str(ds_dir), iters=10)
        run = mgr.add_run(cfg)

        # Mock subprocess.Popen so we don't actually run mlx_lm
        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = ""
        mock_proc.wait.return_value = 0

        with patch("subprocess.Popen", return_value=mock_proc):
            code = execute_single_run(mgr, run.id, enable_wandb=False)
            self.assertEqual(code, 0)

        # Model path was auto-healed in run and YAML config
        updated_r = mgr.get_run(run.id)
        self.assertEqual(updated_r.model, str(model_dir.resolve()))
        yaml_content = (mgr.configs_dir / f"{run.id}.yaml").read_text()
        self.assertIn(str(model_dir.resolve()), yaml_content)

    def test_normalize_model_path_heals_ancestor_colon_folder_with_subfolder_hyphen(self):
        # On disk: Models/foo:bar/sub-hyphen
        model_dir = self.base / "Models" / "foo:bar" / "sub-hyphen"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text('{"model_type": "llama"}')

        # Input with forward slashes for the colon folder
        slash_input = str(self.base / "Models" / "foo" / "bar" / "sub-hyphen")
        healed = normalize_model_path(slash_input)
        self.assertEqual(healed, str(model_dir.resolve()))

    def test_normalize_model_path_heals_ancestor_colon_folder_from_file(self):
        # On disk: Models/foo:bar/sub-hyphen/config.json
        model_dir = self.base / "Models" / "foo:bar" / "sub-hyphen"
        model_dir.mkdir(parents=True)
        cfg_file = model_dir / "config.json"
        cfg_file.write_text('{"model_type": "llama"}')

        # Input pointing at config.json using slashes
        slash_file = str(self.base / "Models" / "foo" / "bar" / "sub-hyphen" / "config.json")
        healed = normalize_model_path(slash_file)
        self.assertEqual(healed, str(model_dir.resolve()))

    def test_normalize_model_path_auto_discovers_child_model_from_parent_folder(self):
        # On disk: Models/mlx-community/Llama-3.2-3B-Instruct
        model_dir = self.base / "Models" / "mlx-community" / "Llama-3.2-3B-Instruct"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text('{"model_type": "llama"}')

        # User selected parent container folder "mlx-community"
        parent_input = str(self.base / "Models" / "mlx-community")
        healed = normalize_model_path(parent_input)
        self.assertEqual(healed, str(model_dir.resolve()))

        # With trailing slash
        healed_slash = normalize_model_path(parent_input + "/")
        self.assertEqual(healed_slash, str(model_dir.resolve()))

    def test_normalize_model_path_heals_parent_prefix_to_colon_folder(self):
        # On disk: Models/mlx-community:Llama-3.2-3B-Instruct
        model_dir = self.base / "Models" / "mlx-community:Llama-3.2-3B-Instruct"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text('{"model_type": "llama"}')

        # User selected prefix "mlx-community"
        prefix_input = str(self.base / "Models" / "mlx-community")
        healed = normalize_model_path(prefix_input)
        self.assertEqual(healed, str(model_dir.resolve()))

    def test_normalize_model_path_heals_file_in_colon_folder(self):
        # On disk: Models/mlx-community:Llama-3.2-3B-Instruct/config.json
        model_dir = self.base / "Models" / "mlx-community:Llama-3.2-3B-Instruct"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text('{"model_type": "llama"}')

        # Input pointing at config.json inside the slash-represented path
        slash_cfg = str(self.base / "Models" / "mlx-community" / "Llama-3.2-3B-Instruct" / "config.json")
        healed = normalize_model_path(slash_cfg)
        self.assertEqual(healed, str(model_dir.resolve()))

    def test_normalize_model_path_sanitizes_quotes_file_url_and_escapes(self):
        model_dir = self.base / "Models" / "Model Tuning" / "Llama-3.2-3B"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text('{"model_type": "llama"}')

        expected = str(model_dir.resolve())

        # Surrounding quotes
        self.assertEqual(normalize_model_path(f'"{expected}"'), expected)
        self.assertEqual(normalize_model_path(f"'{expected}'"), expected)

        # file:// scheme
        self.assertEqual(normalize_model_path(f"file://{expected}"), expected)

        # URL-encoded space (%20)
        url_encoded = str(self.base / "Models" / "Model%20Tuning" / "Llama-3.2-3B")
        self.assertEqual(normalize_model_path(url_encoded), expected)

        # Shell backslash-escaped space
        escaped_space = str(self.base / "Models" / "Model\\ Tuning" / "Llama-3.2-3B")
        self.assertEqual(normalize_model_path(escaped_space), expected)

        # Unicode en-dash (\u2013) vs hyphen
        dash_model = self.base / "Models" / "Llama–3.2–3B"
        dash_model.mkdir(parents=True)
        (dash_model / "config.json").write_text('{"model_type": "llama"}')
        ascii_input = str(self.base / "Models" / "Llama-3.2-3B")
        self.assertEqual(normalize_model_path(ascii_input), str(dash_model.resolve()))


if __name__ == "__main__":

    unittest.main()
