import os
import unittest
from unittest.mock import MagicMock, patch

from mlx_commander.lora.config import LoraRunConfig
from mlx_commander.lora.tracking import (
    WandbTracker,
    is_wandb_available,
    is_wandb_logged_in,
    parse_mlx_log_line,
)


class TestLoraTracking(unittest.TestCase):
    def test_parse_mlx_log_line_train(self):
        line = "Iter 100: Train loss 1.450, Learning Rate 1.000e-05, It/sec 1.150, Tokens/sec 1200.0"
        metrics = parse_mlx_log_line(line)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["iter"], 100)
        self.assertAlmostEqual(metrics["train/loss"], 1.450)
        self.assertAlmostEqual(metrics["train/learning_rate"], 1e-5)
        self.assertAlmostEqual(metrics["train/it_per_sec"], 1.150)
        self.assertAlmostEqual(metrics["train/tokens_per_sec"], 1200.0)

    def test_parse_mlx_log_line_val(self):
        line = "Iter 200: Val loss 2.100, Val It/sec 2.000"
        metrics = parse_mlx_log_line(line)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["iter"], 200)
        self.assertAlmostEqual(metrics["val/loss"], 2.100)
        self.assertAlmostEqual(metrics["val/it_per_sec"], 2.000)

    def test_parse_mlx_log_line_unrelated(self):
        self.assertIsNone(parse_mlx_log_line("Loading pretrained model..."))
        self.assertIsNone(parse_mlx_log_line("Fetching 12 files: 100%"))
        self.assertIsNone(parse_mlx_log_line(""))

    def test_is_wandb_logged_in_env(self):
        with patch.dict(os.environ, {"WANDB_API_KEY": "test_fake_api_key"}):
            logged_in, entity = is_wandb_logged_in()
            self.assertTrue(logged_in)
            self.assertIsNotNone(entity)

    def test_is_wandb_logged_in_empty(self):
        # When WANDB_API_KEY is unset and no ~/.netrc exists
        with patch.dict(os.environ, {}, clear=True), \
             patch("pathlib.Path.exists", return_value=False), \
             patch("mlx_commander.lora.tracking.is_wandb_available", return_value=False):
            logged_in, entity = is_wandb_logged_in()
            self.assertFalse(logged_in)
            self.assertIsNone(entity)

    def test_wandb_tracker_disabled(self):
        tracker = WandbTracker(project="test-proj", enabled=False)
        cfg = LoraRunConfig(name="run1")
        started = tracker.start_run(cfg)
        self.assertFalse(started)
        self.assertIsNone(tracker.run_url)
        # log_line should not raise
        tracker.log_line("Iter 100: Train loss 1.234")
        url = tracker.finish_run()
        self.assertIsNone(url)

    @patch("mlx_commander.lora.tracking.is_wandb_available", return_value=True)
    @patch("mlx_commander.lora.tracking.is_wandb_logged_in", return_value=(True, "test_user"))
    def test_wandb_tracker_mock_run(self, mock_logged_in, mock_avail):
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.get_url.return_value = "https://wandb.ai/test_user/test-proj/runs/abc1234"
        mock_run.url = "https://wandb.ai/test_user/test-proj/runs/abc1234"
        mock_wandb.init.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker = WandbTracker(project="test-proj", enabled=True)
            cfg = LoraRunConfig(
                name="01_lora_r16_test",
                model="mlx-community/Llama-3.2-3B-Instruct-4bit",
            )
            started = tracker.start_run(cfg, implied_epochs=3.5)
            self.assertTrue(started)
            self.assertEqual(tracker.run_url, "https://wandb.ai/test_user/test-proj/runs/abc1234")

            # Check wandb.init call arguments
            mock_wandb.init.assert_called_once()
            _, kwargs = mock_wandb.init.call_args
            self.assertEqual(kwargs["project"], "test-proj")
            self.assertEqual(kwargs["name"], "01_lora_r16_test")
            self.assertEqual(kwargs["config"]["implied_epochs"], 3.5)

            # Test streaming metric logging
            tracker.log_line("Iter 50: Train loss 1.450, Learning Rate 1.000e-05, It/sec 1.150")
            mock_wandb.log.assert_called_once()
            log_metrics, log_kwargs = mock_wandb.log.call_args
            self.assertEqual(log_kwargs["step"], 50)
            self.assertIn("train/loss", log_metrics[0])
            self.assertAlmostEqual(log_metrics[0]["train/loss"], 1.450)

            # Test finish
            fin_url = tracker.finish_run(exit_code=0)
            self.assertEqual(fin_url, "https://wandb.ai/test_user/test-proj/runs/abc1234")
            mock_wandb.finish.assert_called_once_with(exit_code=0)

    def test_wandb_login_uses_local_settings_without_network(self):
        from mlx_commander.lora.tracking import clear_wandb_login_cache
        mock_wandb = MagicMock()
        mock_settings = MagicMock()
        mock_settings.api_key = "test_key_123"
        mock_settings.entity = "test_entity_direct"
        mock_wandb.setup.return_value.settings = mock_settings

        with patch.dict("sys.modules", {"wandb": mock_wandb}), \
             patch.dict(os.environ, {}, clear=True), \
             patch("mlx_commander.lora.tracking.is_wandb_available", return_value=True):
            clear_wandb_login_cache()
            logged_in, entity = is_wandb_logged_in()
            self.assertTrue(logged_in)
            self.assertEqual(entity, "test_entity_direct")
            # Verify no network calls were made to Api().viewer
            self.assertFalse(hasattr(mock_wandb, "Api") and mock_wandb.Api.called)


if __name__ == "__main__":
    unittest.main()
