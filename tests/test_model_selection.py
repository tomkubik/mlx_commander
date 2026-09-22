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


if __name__ == "__main__":
    unittest.main()
