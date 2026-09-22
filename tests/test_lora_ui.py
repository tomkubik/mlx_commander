import curses
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlx_commander.lora import LoraRunConfig, QueueManager
from mlx_commander.tui.app import run_commander_tui
from mlx_commander.tui.state import CommanderState
from mlx_commander.tui.widgets import (
    draw_queue_table,
    show_choice_dialog,
    show_dataset_picker_dialog,
    show_model_picker_dialog,
)


class TestLoraUI(unittest.TestCase):
    def setUp(self):
        self.mock_win = MagicMock()
        self.mock_win.getmaxyx.return_value = (35, 120)
        self.temp_dir = tempfile.mkdtemp()
        self.queue_dir = Path(self.temp_dir) / "test_runs"

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_mode_switch_f2(self, mock_curs, mock_colors, mock_has_colors):
        state = CommanderState()
        state.queue_manager = QueueManager(self.queue_dir)

        # Start in mode 0, press F2 (switches to mode 1), then 'q' to quit
        self.mock_win.getch.side_effect = [
            curses.KEY_F2,
            ord("q"),
        ]
        res = run_commander_tui(self.mock_win, initial_state=state)
        self.assertIsNone(res)
        self.assertEqual(state.active_tab, 1)
        self.assertIn("Fine-Tuning Single Run", state.status_message)
        calls = self.mock_win.addstr.call_args_list
        all_text = " ".join(c[0][2] for c in calls if len(c[0]) >= 3 and isinstance(c[0][2], str))
        self.assertIn("2: Fine-Tuning Single Run", all_text)
        self.assertNotIn("2: Fine-Tuning Single Run (F2)", all_text)
        self.assertIn("[F2] Mode", all_text)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_add_run_to_queue_f6(self, mock_curs, mock_colors, mock_has_colors):
        state = CommanderState()
        state.active_tab = 1
        state.queue_manager = QueueManager(self.queue_dir)
        state.lora_config.model = "mlx-community/Llama-3.2-3B-Instruct-4bit"

        # Press F6 to add to queue, then 'q'
        self.mock_win.getch.side_effect = [
            curses.KEY_F6,
            ord("q"),
        ]
        run_commander_tui(self.mock_win, initial_state=state)
        self.assertEqual(len(state.queue_manager.runs), 1)
        self.assertEqual(state.queue_manager.runs[0].model, "mlx-community/Llama-3.2-3B-Instruct-4bit")

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_queue_actions_clone_and_delete(self, mock_curs, mock_colors, mock_has_colors):
        state = CommanderState()
        state.active_tab = 1
        state.lora_active_panel = "queue"
        state.queue_manager = QueueManager(self.queue_dir)
        run1 = LoraRunConfig(id="r1", name="Original Run")
        state.queue_manager.add_run(run1)

        # Press 'c' to clone, then 'd' to delete, then 'q'
        self.mock_win.getch.side_effect = [
            ord("c"),  # Clones run1 -> now 2 runs
            ord("d"),  # Deletes selected run -> back to 1 run
            ord("q"),
        ]
        run_commander_tui(self.mock_win, initial_state=state)
        self.assertEqual(len(state.queue_manager.runs), 1)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_queue_clear_action_x(self, mock_curs, mock_colors, mock_has_colors):
        state = CommanderState()
        state.active_tab = 1
        state.lora_active_panel = "queue"
        state.queue_manager = QueueManager(self.queue_dir)
        state.queue_manager.add_run(LoraRunConfig(id="r1"))
        state.queue_manager.add_run(LoraRunConfig(id="r2"))
        self.assertEqual(len(state.queue_manager.runs), 2)

        # Press 'x' to clear queue, then 'q'
        self.mock_win.getch.side_effect = [
            ord("x"),
            ord("q"),
        ]
        run_commander_tui(self.mock_win, initial_state=state)
        self.assertEqual(len(state.queue_manager.runs), 0)

    @patch("mlx_commander.tui.widgets.show_text_edit_dialog", return_value="/local/path/to/model")
    def test_show_model_picker_dialog(self, mock_text_edit):
        # Press Down (select Enter Local Model Path Manually) then Enter
        self.mock_win.getch.side_effect = [curses.KEY_DOWN, 10]
        chosen = show_model_picker_dialog(self.mock_win)
        self.assertEqual(chosen, "/local/path/to/model")
        mock_text_edit.assert_called_once()

    @patch("mlx_commander.tui.widgets.show_text_edit_dialog", return_value="/local/path/to/dataset")
    def test_show_dataset_picker_dialog(self, mock_text_edit):
        # Press Down (select Enter Local Dataset Path Manually) then Enter
        self.mock_win.getch.side_effect = [curses.KEY_DOWN, 10]
        chosen = show_dataset_picker_dialog(self.mock_win)
        self.assertEqual(chosen, "/local/path/to/dataset")
        mock_text_edit.assert_called_once()

    def test_show_choice_dialog(self):
        # Select option 1 with Down then Enter
        options = ["adamw", "adam", "sgd"]
        self.mock_win.getch.side_effect = [curses.KEY_DOWN, 10]
        chosen = show_choice_dialog(self.mock_win, "Optimizer", "Select optimizer:", options, current_val="adamw")
        self.assertEqual(chosen, "adam")

    def test_draw_queue_table(self):
        runs = [
            LoraRunConfig(id="r1", name="Run 1", model="mlx-community/Llama-3.2-1B-Instruct-4bit", iters=500, batch_size=4, status="queued"),
            LoraRunConfig(id="r2", name="Run 2", model="mlx-community/Qwen2.5-7B-Instruct-4bit", iters=1000, batch_size=2, status="running"),
        ]
        selected = draw_queue_table(
            self.mock_win,
            2,
            2,
            8,
            100,
            runs,
            selected_idx=0,
            is_focused=True,
        )
        self.assertEqual(selected, 0)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_single_column_hyperparameter_navigation(self, mock_curs, mock_colors, mock_has_colors):
        state = CommanderState()
        state.active_tab = 1
        state.lora_active_panel = "right"
        state.lora_right_focus_idx = 0

        # Press KEY_DOWN (should go from 0 -> 1), then KEY_DOWN (1 -> 2), then KEY_UP (2 -> 1), then KEY_LEFT (to left panel), then 'q'
        self.mock_win.getch.side_effect = [
            curses.KEY_DOWN,
            curses.KEY_DOWN,
            curses.KEY_UP,
            curses.KEY_LEFT,
            ord("q"),
        ]
        run_commander_tui(self.mock_win, initial_state=state)
        self.assertEqual(state.lora_right_focus_idx, 1)
        self.assertEqual(state.lora_active_panel, "left")

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_no_redundant_model_and_params_in_right_panel(self, mock_curs, mock_colors, mock_has_colors):
        """Verify that redundant Model: and Params: rows are removed from the right panel."""
        from mlx_commander.lora.model_info import ModelMetadata
        state = CommanderState()
        state.active_tab = 1
        state.lora_active_panel = "right"
        state.lora_config.model = "/models/TestModel-3B"
        state.current_model_metadata = ModelMetadata(
            name="TestModel-3B",
            path="/models/TestModel-3B",
            architecture="llama",
            num_layers=28,
            hidden_size=3072,
            num_heads=24,
            num_kv_heads=8,
            context_length=131072,
            quantization="4-bit (group 64)",
            file_size_gb=1.9,
            is_valid=True,
        )

        self.mock_win.getch.side_effect = [ord("q")]
        run_commander_tui(self.mock_win, initial_state=state)

        calls = self.mock_win.addstr.call_args_list
        all_text = " ".join(c[0][2] for c in calls if len(c[0]) >= 3 and isinstance(c[0][2], str))
        self.assertNotIn("• Model:", all_text)
        self.assertNotIn("• Params:", all_text)
        self.assertNotIn("Select a local model on drive to inspect architecture", all_text)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_hyperparameters_pane_displays_all_fields_without_scrolling(self, mock_curs, mock_colors, mock_has_colors):
        """Verify that the hyperparameter pane displays all 14 fields simultaneously without scrolling."""
        state = CommanderState()
        state.active_tab = 1
        state.lora_active_panel = "right"
        self.mock_win.reset_mock()
        self.mock_win.getmaxyx.return_value = (35, 120)
        self.mock_win.getch.side_effect = [ord("q")]

        run_commander_tui(self.mock_win, initial_state=state)

        calls = self.mock_win.addstr.call_args_list
        rendered_texts = [c[0][2] for c in calls if len(c[0]) >= 3 and isinstance(c[0][2], str)]
        all_text = " ".join(rendered_texts)

        # Check all 14 fields and button are present in a single frame
        expected_fields = [
            "Training Iterations",
            "Batch Size",
            "Learning Rate",
            "LoRA Rank (r)",
            "LoRA Alpha (α)",
            "LoRA Dropout",
            "Max Seq Length",
            "Fine-Tuned Layers",
            "Grad Checkpoint",
            "Mask Prompt",
            "Save Every",
            'Steps per "Eval" (validation loss)',
            "Adapter Path",
            "+ Add to Queue (F6)",
        ]
        for field in expected_fields:
            self.assertIn(field, all_text, f"Field '{field}' must be rendered in view without scrolling")

    @patch("mlx_commander.tui.widgets.curses.color_pair", side_effect=lambda n: n * 256)
    @patch("mlx_commander.tui.app.curses.color_pair", side_effect=lambda n: n * 256)
    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_estimates_pane_typography_and_layout(self, mock_curs, mock_colors, mock_has_colors, mock_app_cp, mock_wid_cp):
        """
        Verify that in Resource & Training Estimates:
        - Only the pane title is bold. All inner text lines are strictly unbolded.
        - Implied Epochs: label and epoch count in white font; calculation in gray font.
        - Peak Unified RAM: metrics in green/safety color; description in gray font.
        - RAM Breakdown: in gray font.
        - Estimated Duration: in white font; hardware row and recommendation row removed.
        """
        from mlx_commander.tui.widgets import (
            COLOR_LABEL_GRAY,
            COLOR_NORMAL_TEXT,
            COLOR_SUCCESS,
            COLOR_TITLE_ACCENT,
        )

        state = CommanderState()
        state.active_tab = 1
        state.loaded_dataset = True
        state.get_split_counts = lambda: {"train": 1000}
        state.lora_config.iters = 500
        state.lora_config.batch_size = 4

        self.mock_win.reset_mock()
        self.mock_win.getmaxyx.return_value = (35, 120)
        self.mock_win.getch.side_effect = [ord("q")]

        run_commander_tui(self.mock_win, initial_state=state)

        calls = self.mock_win.addstr.call_args_list
        all_text = " ".join(c[0][2] for c in calls if len(c[0]) >= 3 and isinstance(c[0][2], str))

        # 1. Recommendation row must be completely removed
        self.assertNotIn("Recommendation:", all_text)
        self.assertNotIn("Headroom is optimal", all_text)

        # 2. Hardware info and old title must be completely removed
        self.assertNotIn("Hardware: Apple Silicon", all_text)
        self.assertNotIn("Apple Silicon Unified Memory", all_text)
        self.assertNotIn("Resource & Training Estimates", all_text)

        # 3. Verify specific lines and typography
        found_epochs_lbl = False
        found_epochs_calc = False
        found_ram_main = False
        found_ram_desc = False
        found_breakdown = False
        found_duration = False
        found_title = False

        for c in calls:
            args = c[0]
            if len(args) >= 3 and isinstance(args[2], str):
                text = args[2]
                attr = args[3] if len(args) >= 4 else 0

                if "Runtime Estimates" in text:
                    found_title = True
                    # Pane title must remain bold
                    self.assertTrue(bool(attr & curses.A_BOLD), "Pane title must be bold")

                # All text inside estimates pane must be UNBOLDED
                if any(k in text for k in ["Implied Epochs", "train records", "Peak Unified RAM", "RAM Breakdown", "Est. Duration"]):
                    self.assertEqual(attr & curses.A_BOLD, 0, f"Inner estimates text '{text}' must NOT be bold")

                if "• Implied Epochs:" in text:
                    found_epochs_lbl = True
                    # Must be white unbold (COLOR_NORMAL_TEXT)
                    self.assertEqual(attr & curses.A_BOLD, 0)
                    self.assertEqual((attr & ~curses.A_DIM & ~curses.A_BOLD) // 256, COLOR_NORMAL_TEXT)

                if "train records" in text:
                    found_epochs_calc = True
                    # Must be gray unbold (COLOR_LABEL_GRAY or A_DIM)
                    self.assertEqual(attr & curses.A_BOLD, 0)
                    pair_num = (attr & ~curses.A_DIM & ~curses.A_BOLD) // 256
                    self.assertTrue(pair_num == COLOR_LABEL_GRAY or bool(attr & curses.A_DIM))

                if "• Peak Unified RAM:" in text:
                    found_ram_main = True
                    self.assertNotIn("~", text, "Peak Unified RAM should not contain ~")
                    # Must be green unbold (COLOR_SUCCESS)
                    self.assertEqual(attr & curses.A_BOLD, 0)
                    self.assertEqual((attr & ~curses.A_DIM & ~curses.A_BOLD) // 256, COLOR_SUCCESS)

                if "Safe fine-tuning headroom" in text:
                    found_ram_desc = True
                    # Must be gray unbold
                    self.assertEqual(attr & curses.A_BOLD, 0)
                    pair_num = (attr & ~curses.A_DIM & ~curses.A_BOLD) // 256
                    self.assertTrue(pair_num == COLOR_LABEL_GRAY or bool(attr & curses.A_DIM))

                if "RAM Breakdown:" in text:
                    found_breakdown = True
                    # Must be gray unbold
                    self.assertEqual(attr & curses.A_BOLD, 0)
                    pair_num = (attr & ~curses.A_DIM & ~curses.A_BOLD) // 256
                    self.assertTrue(pair_num == COLOR_LABEL_GRAY or bool(attr & curses.A_DIM))

                if "• Est. Duration:" in text:
                    found_duration = True
                    self.assertNotIn("~", text, "Est. Duration should not contain ~")
                    # Must be white unbold
                    self.assertEqual(attr & curses.A_BOLD, 0)
                    self.assertEqual((attr & ~curses.A_DIM & ~curses.A_BOLD) // 256, COLOR_NORMAL_TEXT)

        self.assertTrue(found_title, "Estimates pane title should be found")
        self.assertTrue(found_epochs_lbl, "Implied Epochs label should be found in white font")
        self.assertTrue(found_epochs_calc, "Implied Epochs calculation should be found in gray font")
        self.assertTrue(found_ram_main, "Peak Unified RAM main string should be found in green font")
        self.assertTrue(found_ram_desc, "Safe fine-tuning headroom should be found in gray font")
        self.assertTrue(found_breakdown, "RAM Breakdown should be found in gray font")
        self.assertTrue(found_duration, "Est. Duration should be found in white font")

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    @patch("mlx_commander.tui.app.show_dataset_picker_dialog", return_value="/new/dataset/dir")
    def test_dataset_and_model_selector_consistency(self, mock_picker, mock_curs, mock_colors, mock_has_colors):
        """
        Verify that:
        - Base model and dataset both display 'None' when empty.
        - Neither has a separate Finder button row.
        - Pressing Enter on dataset (index 1) opens show_dataset_picker_dialog.
        - Navigation clamps between 0 and 5.
        """
        state = CommanderState()
        state.active_tab = 1
        state.lora_active_panel = "left"
        state.lora_config.model = ""
        state.lora_config.data = ""

        # Press KEY_DOWN (to focus Dataset, index 1), press Enter (open dialog), then 'q'
        self.mock_win.getch.side_effect = [
            curses.KEY_DOWN,
            10,  # Enter on Dataset
            ord("q"),
        ]
        run_commander_tui(self.mock_win, initial_state=state)

        # Picker should have been called with current data ("")
        mock_picker.assert_called_once()
        self.assertEqual(state.lora_config.data, "/new/dataset/dir")

        calls = self.mock_win.addstr.call_args_list
        all_text = " ".join(c[0][2] for c in calls if len(c[0]) >= 3 and isinstance(c[0][2], str))
        self.assertIn("None", all_text)
        self.assertNotIn("Finder (F2)", all_text)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_no_workflow_shortcuts_in_fine_tuning_left_panel(self, mock_curs, mock_colors, mock_has_colors):
        """Verify Workflow Shortcuts section is completely removed from Model & Dataset Selector pane."""
        state = CommanderState()
        state.active_tab = 1
        self.mock_win.getmaxyx.return_value = (35, 110)
        self.mock_win.getch.side_effect = [ord("q")]
        run_commander_tui(self.mock_win, initial_state=state)

        calls = self.mock_win.addstr.call_args_list
        all_text = " ".join(c[0][2] for c in calls if len(c[0]) >= 3 and isinstance(c[0][2], str))
        self.assertNotIn("Workflow Shortcuts", all_text)

    @patch("mlx_commander.tui.widgets.curses.color_pair", side_effect=lambda n: n * 256)
    @patch("mlx_commander.tui.app.curses.color_pair", side_effect=lambda n: n * 256)
    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_implied_epochs_color_under_one_epoch(self, mock_curs, mock_colors, mock_has_colors, mock_app_cp, mock_wid_cp):
        """Verify that when implied epochs < 1.0, it is marked in COLOR_ERROR (dark red font)."""
        from mlx_commander.tui.widgets import COLOR_ERROR, COLOR_NORMAL_TEXT

        state = CommanderState()
        state.active_tab = 1
        state.loaded_dataset = True
        state.get_split_counts = lambda: {"train": 1000}
        # iters: 100 * batch: 4 = 400 records -> 0.40 epochs (< 1.0)
        state.lora_config.iters = 100
        state.lora_config.batch_size = 4

        self.mock_win.reset_mock()
        self.mock_win.getmaxyx.return_value = (35, 120)
        self.mock_win.getch.side_effect = [ord("q")]

        run_commander_tui(self.mock_win, initial_state=state)

        calls = self.mock_win.addstr.call_args_list
        found_red_epoch = False
        for c in calls:
            args = c[0]
            if len(args) >= 3 and isinstance(args[2], str):
                text = args[2]
                attr = args[3] if len(args) >= 4 else 0
                if "• Implied Epochs:" in text:
                    self.assertIn("0.40 epochs", text)
                    pair_num = (attr & ~curses.A_DIM & ~curses.A_BOLD) // 256
                    self.assertEqual(pair_num, COLOR_ERROR, "Implied epochs < 1.0 should be styled with COLOR_ERROR (dark red font)")
                    found_red_epoch = True

        self.assertTrue(found_red_epoch, "Implied Epochs line (< 1.0) should be rendered in COLOR_ERROR")

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_mode2_toggle_run_eval(self, mock_curs, mock_colors, mock_has_colors):
        state = CommanderState()
        state.active_tab = 1
        state.lora_active_panel = "right"
        state.lora_right_focus_idx = 12  # Run evals on test set (experimental)
        self.assertFalse(state.lora_config.run_eval)

        # Press Enter (10) then 'q'
        self.mock_win.reset_mock()
        self.mock_win.getmaxyx.return_value = (35, 120)
        self.mock_win.getch.side_effect = [10, ord("q")]

        run_commander_tui(self.mock_win, initial_state=state)
        self.assertTrue(state.lora_config.run_eval)
        self.assertIn("Run evals on test set (experimental): Yes", state.status_message)


if __name__ == "__main__":
    unittest.main()



