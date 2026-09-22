import curses
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlx_commander.lora.multi_config import MultiLoraRunConfig, SWEEP_FIELD_DEFS
from mlx_commander.lora.config import LoraRunConfig
from mlx_commander.lora.queue import QueueManager
from mlx_commander.tui.app import run_commander_tui
from mlx_commander.tui.state import CommanderState
from mlx_commander.tui.widgets import (
    draw_multi_field,
    draw_sweep_grid,
    show_multi_value_edit_dialog,
)


class TestMultiLoraRunConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = MultiLoraRunConfig()
        self.assertEqual(cfg.total_runs_count, 1)
        runs = cfg.generate_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].model, "mlx-community/Llama-3.2-3B-Instruct-4bit")
        self.assertEqual(runs[0].iters, 1000)

    def test_cartesian_product_24_runs(self):
        # Mockup conditions: 3 learning rates, 4 ranks, 2 mask prompt settings = 24 runs
        cfg = MultiLoraRunConfig(
            learning_rate=[1e-4, 2e-4, 5e-4],
            lora_rank=[2, 4, 6, 8],
            mask_prompt=[True, False],
        )
        self.assertEqual(cfg.total_runs_count, 24)
        runs = cfg.generate_runs()
        self.assertEqual(len(runs), 24)

        # Check unique combinations
        combs = {(r.learning_rate, r.lora_rank, r.mask_prompt) for r in runs}
        self.assertEqual(len(combs), 24)
        self.assertIn((1e-4, 2, True), combs)
        self.assertIn((5e-4, 8, False), combs)

    def test_varying_hyperparameters(self):
        cfg = MultiLoraRunConfig(
            learning_rate=[1e-4, 2e-4, 5e-4],
            lora_rank=[2, 4, 6, 8],
            mask_prompt=[True, False],
        )
        varying = cfg.get_varying_hyperparameters()
        self.assertEqual(len(varying), 3)
        names = [v[0] for v in varying]
        self.assertEqual(names, ["learning_rate", "lora_rank", "mask_prompt"])

    def test_varying_hyperparameters_fallback(self):
        # When all fields have single value, fallback returns the 3 primary fields
        cfg = MultiLoraRunConfig()
        varying = cfg.get_varying_hyperparameters()
        self.assertEqual(len(varying), 3)
        names = [v[0] for v in varying]
        self.assertEqual(names, ["learning_rate", "lora_rank", "mask_prompt"])

    def test_sweep_estimates_aggregation(self):
        cfg = MultiLoraRunConfig(
            iters=[500, 1000],
            batch_size=[2, 4],
            lora_rank=[8, 16],
        )
        # 2 * 2 * 2 = 8 runs
        est = cfg.calculate_sweep_estimates(dataset_records=1000)
        self.assertEqual(est["total_runs"], 8)
        # Min implied epochs: (500 * 2) / 1000 = 1.0 epoch
        self.assertAlmostEqual(est["min_implied_epochs"], 1.0)
        # Max peak RAM should be > 0 and reflect the largest configuration
        self.assertGreater(est["max_peak_ram_gb"], 0)
        # Total duration should sum all 8 runs
        self.assertGreater(est["total_duration_seconds"], 0)
        self.assertTrue(len(est["total_duration_str"]) > 0)


class TestMultiRunStateAndUI(unittest.TestCase):
    def setUp(self):
        self.mock_win = MagicMock()
        self.mock_win.getmaxyx.return_value = (35, 120)
        self.temp_dir = tempfile.mkdtemp()
        self.queue_dir = Path(self.temp_dir) / "test_runs"

    def test_state_switch_mode_3(self):
        state = CommanderState()
        self.assertEqual(state.active_tab, 0)
        state.switch_mode(2)
        self.assertEqual(state.active_tab, 2)
        self.assertIn("Fine-Tuning Multi-Run", state.status_message)

    def test_add_multi_lora_runs_to_queue(self):
        state = CommanderState()
        state.queue_manager = QueueManager(self.queue_dir)
        state.multi_lora_config = MultiLoraRunConfig(
            learning_rate=[1e-4, 2e-4],
            lora_rank=[4, 8],
        )
        # 2 * 2 = 4 runs
        added = state.add_multi_lora_runs_to_queue()
        self.assertEqual(len(added), 4)
        self.assertEqual(len(state.queue_manager.runs), 4)
        for r in state.queue_manager.runs:
            self.assertEqual(r.status, "queued")

    def test_draw_multi_field(self):
        win = MagicMock()
        win.getmaxyx.return_value = (20, 80)
        draw_multi_field(
            win,
            y=2,
            x=2,
            label="LoRA Rank",
            values=[2, 4, 6, 8],
            is_focused=True,
            right_edge=75,
        )
        # Verify safe_addstr called for label and values
        calls = win.addstr.call_args_list
        all_text = " ".join(c[0][2] for c in calls if len(c[0]) >= 3 and isinstance(c[0][2], str))
        self.assertIn("LoRA Rank", all_text)
        self.assertIn("[ 2 ]", all_text)
        self.assertIn("[ 8 ]", all_text)

    def test_draw_sweep_grid(self):
        win = MagicMock()
        win.getmaxyx.return_value = (20, 100)
        varying = [
            ("learning_rate", "Learning rate", [1e-4, 2e-4, 5e-4]),
            ("lora_rank", "LoRA Rank", [2, 4, 6, 8]),
            ("mask_prompt", "Mask prompt", [True, False]),
        ]
        draw_sweep_grid(
            win,
            y=1,
            x=2,
            h=12,
            w=96,
            varying_params=varying,
            total_runs_count=24,
        )
        calls = win.addstr.call_args_list
        all_text = " ".join(c[0][2] for c in calls if len(c[0]) >= 3 and isinstance(c[0][2], str))
        self.assertIn("Learning rate", all_text)
        self.assertIn("LoRA Rank", all_text)
        self.assertIn("Mask prompt", all_text)
        self.assertIn("✖", all_text)
        self.assertIn("24 training runs", all_text)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_mode_switcher_navigation_3_tabs(self, mock_curs, mock_colors, mock_has_colors):
        state = CommanderState()
        state.queue_manager = QueueManager(self.queue_dir)

        # Mode 0 -> UP into mode switcher -> RIGHT to mode 1 -> RIGHT to mode 2 -> ENTER -> 'q'
        self.mock_win.getch.side_effect = [
            curses.KEY_UP,
            curses.KEY_RIGHT,
            curses.KEY_RIGHT,
            10,  # Enter
            ord("q"),
        ]
        run_commander_tui(self.mock_win, initial_state=state)
        self.assertEqual(state.active_tab, 2)
        self.assertIn("Fine-Tuning Multi-Run", state.status_message)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_mode3_add_all_f6(self, mock_curs, mock_colors, mock_has_colors):
        state = CommanderState()
        state.active_tab = 2
        state.queue_manager = QueueManager(self.queue_dir)
        state.multi_lora_config.lora_rank = [4, 8]
        state.multi_lora_config.learning_rate = [1e-4, 2e-4]

        # In Mode 3, press F6 to add all runs, then 'q'
        self.mock_win.getch.side_effect = [
            curses.KEY_F6,
            ord("q"),
        ]
        run_commander_tui(self.mock_win, initial_state=state)
        self.assertEqual(len(state.queue_manager.runs), 4)
        self.assertIn("Added 4 sweep run(s) to queue", state.status_message)

    @patch("mlx_commander.tui.widgets.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.widgets.safe_curs_set")
    def test_show_multi_value_edit_dialog_int(self, mock_curs, mock_colors):
        win = MagicMock()
        win.getmaxyx.return_value = (25, 80)
        # Type "2, 4, 8" then Enter
        # Backspace existing default (e.g. 5 chars) then type "2, 4, 8"
        keys = [curses.KEY_BACKSPACE] * 20 + [ord(c) for c in "2, 4, 8"] + [10]
        win.getch.side_effect = keys
        res = show_multi_value_edit_dialog(win, "Edit Rank", "Enter rank conditions:", [8], val_type=int)
        self.assertEqual(res, [2, 4, 8])

    @patch("mlx_commander.tui.widgets.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.widgets.safe_curs_set")
    def test_show_multi_value_edit_dialog_float(self, mock_curs, mock_colors):
        win = MagicMock()
        win.getmaxyx.return_value = (25, 80)
        keys = [curses.KEY_BACKSPACE] * 20 + [ord(c) for c in "1e-5, 2e-5"] + [10]
        win.getch.side_effect = keys
        res = show_multi_value_edit_dialog(win, "Edit LR", "Enter learning rates:", [1e-5], val_type=float)
        self.assertEqual(res, [1e-5, 2e-5])

    @patch("mlx_commander.tui.widgets.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.widgets.safe_curs_set")
    def test_show_multi_value_edit_dialog_bool(self, mock_curs, mock_colors):
        win = MagicMock()
        win.getmaxyx.return_value = (25, 80)
        keys = [curses.KEY_BACKSPACE] * 20 + [ord(c) for c in "True, False"] + [10]
        win.getch.side_effect = keys
        res = show_multi_value_edit_dialog(win, "Edit Mask", "Enter boolean conditions:", [True], val_type=bool)
        self.assertEqual(res, [True, False])

    @patch("mlx_commander.tui.widgets.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.widgets.safe_curs_set")
    def test_show_multi_value_edit_dialog_esc_cancels(self, mock_curs, mock_colors):
        win = MagicMock()
        win.getmaxyx.return_value = (25, 80)
        win.getch.side_effect = [27]  # ESC
        res = show_multi_value_edit_dialog(win, "Edit", "Prompt", [8], val_type=int)
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()

