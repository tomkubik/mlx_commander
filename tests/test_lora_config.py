import unittest
from mlx_commander.lora.config import (
    LoraRunConfig,
    POPULAR_MLX_MODELS,
    FINE_TUNE_TYPES,
    OPTIMIZERS,
)


class TestLoraConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = LoraRunConfig()
        self.assertEqual(cfg.model, "mlx-community/Llama-3.2-3B-Instruct-4bit")
        self.assertEqual(cfg.fine_tune_type, "lora")
        self.assertEqual(cfg.optimizer, "adamw")
        self.assertEqual(cfg.iters, 1000)
        self.assertEqual(cfg.batch_size, 4)
        self.assertEqual(cfg.learning_rate, 1e-5)
        self.assertEqual(cfg.lora_rank, 8)
        self.assertEqual(cfg.lora_alpha, 16.0)
        self.assertEqual(cfg.lora_dropout, 0.0)
        self.assertEqual(cfg.max_seq_length, 2048)
        self.assertEqual(cfg.num_layers, 16)
        self.assertTrue(cfg.grad_checkpoint)
        self.assertTrue(cfg.mask_prompt)
        self.assertTrue(cfg.train)
        self.assertFalse(cfg.test)
        self.assertEqual(cfg.status, "queued")
        self.assertEqual(cfg.steps_per_eval, 100)

    def test_popular_models_and_constants(self):
        self.assertGreater(len(POPULAR_MLX_MODELS), 5)
        self.assertIn("lora", FINE_TUNE_TYPES)
        self.assertIn("dora", FINE_TUNE_TYPES)
        self.assertIn("full", FINE_TUNE_TYPES)
        self.assertIn("adamw", OPTIMIZERS)
        self.assertIn("adam", OPTIMIZERS)

    def test_to_mlx_yaml(self):
        cfg = LoraRunConfig(
            model="mlx-community/Qwen2.5-7B-Instruct-4bit",
            data="data/custom",
            iters=500,
            batch_size=2,
            learning_rate=2e-5,
            lora_rank=16,
            lora_alpha=32.0,
            adapter_path="adapters/custom_run",
        )
        yaml_str = cfg.to_mlx_yaml()
        self.assertIn('model: "mlx-community/Qwen2.5-7B-Instruct-4bit"', yaml_str)
        self.assertIn('data: "data/custom"', yaml_str)
        self.assertIn('iters: 500', yaml_str)
        self.assertIn('batch_size: 2', yaml_str)
        self.assertIn('learning_rate: 2e-05', yaml_str)
        self.assertIn('adapter_path: "adapters/custom_run"', yaml_str)
        self.assertIn('rank: 16', yaml_str)
        self.assertIn('scale: 32.0', yaml_str)

    def test_to_cli_command(self):
        cfg = LoraRunConfig(
            model="mlx-community/Llama-3.2-1B-Instruct-4bit",
            data="mlx_dataset",
            iters=600,
            batch_size=4,
            learning_rate=1e-4,
            lora_rank=8,
            grad_checkpoint=True,
            mask_prompt=True,
            adapter_path="adapters/llama_1b",
        )
        cmd = cfg.to_cli_command()
        self.assertIn("mlx_lm.lora", cmd)
        self.assertIn("--model mlx-community/Llama-3.2-1B-Instruct-4bit", cmd)
        self.assertIn("--train", cmd)
        self.assertIn("--data mlx_dataset", cmd)
        self.assertIn("--iters 600", cmd)
        self.assertIn("--batch-size 4", cmd)
        self.assertIn("--learning-rate 0.0001", cmd)
        self.assertIn("--num-layers 16", cmd)
        self.assertIn("--adapter-path adapters/llama_1b", cmd)
        self.assertIn("--grad-checkpoint", cmd)
        self.assertIn("--mask-prompt", cmd)
        self.assertIn("--steps-per-eval 100", cmd)

    def test_dict_serialization_round_trip(self):
        cfg1 = LoraRunConfig(
            name="Test Run Alpha",
            model="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            data="data/test",
            iters=1200,
            batch_size=8,
            learning_rate=5e-5,
            lora_rank=32,
            fine_tune_type="dora",
            optimizer="adam",
        )
        d = cfg1.to_dict()
        cfg2 = LoraRunConfig.from_dict(d)

        self.assertEqual(cfg1.id, cfg2.id)
        self.assertEqual(cfg1.name, cfg2.name)
        self.assertEqual(cfg1.model, cfg2.model)
        self.assertEqual(cfg1.data, cfg2.data)
        self.assertEqual(cfg1.iters, cfg2.iters)
        self.assertEqual(cfg1.batch_size, cfg2.batch_size)
        self.assertEqual(cfg1.learning_rate, cfg2.learning_rate)
        self.assertEqual(cfg1.lora_rank, cfg2.lora_rank)
        self.assertEqual(cfg1.fine_tune_type, cfg2.fine_tune_type)
        self.assertEqual(cfg1.optimizer, cfg2.optimizer)

    def test_validation_errors(self):
        cfg = LoraRunConfig(iters=0)
        errors = cfg.validate()
        self.assertTrue(any("Iterations must be > 0" in e for e in errors))

        cfg_batch = LoraRunConfig(batch_size=0)
        errors = cfg_batch.validate()
        self.assertTrue(any("Batch size must be > 0" in e for e in errors))

        cfg_lr = LoraRunConfig(learning_rate=-1.0)
        errors = cfg_lr.validate()
        self.assertTrue(any("Learning rate must be > 0" in e for e in errors))

        cfg_dropout = LoraRunConfig(lora_dropout=0.8)
        errors = cfg_dropout.validate()
        self.assertTrue(any("Dropout must be between 0.0 and 0.5" in e for e in errors))
