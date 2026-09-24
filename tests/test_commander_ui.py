import curses
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from mlx_commander.formats import MLXFormat
from mlx_commander.tui.app import run_commander_tui
from mlx_commander.tui.state import ActivePanel, CommanderState
from mlx_commander.tui.widgets import (
    show_column_picker_dialog,
    show_error_dialog,
    show_help_dialog,
    show_message_dialog,
    show_results_dialog,
    show_text_edit_dialog,
)


class TestCommanderUI(unittest.TestCase):
    def setUp(self):
        self.mock_win = MagicMock()
        self.mock_win.getmaxyx.return_value = (30, 100)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_run_commander_tui_quit_q(self, mock_curs, mock_colors, mock_has_colors):
        # Simulate pressing 'q' immediately to exit
        self.mock_win.getch.side_effect = [ord("q")]
        res = run_commander_tui(self.mock_win)
        self.assertIsNone(res)
        self.assertTrue(self.mock_win.erase.called)
        self.assertTrue(self.mock_win.refresh.called)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_run_commander_tui_tab_and_arrow(self, mock_curs, mock_colors, mock_has_colors):
        # Press Tab (switches panel), Right arrow (cycles format), then 'q'
        self.mock_win.getch.side_effect = [
            9,                  # Tab
            curses.KEY_RIGHT,   # Right arrow
            ord("q"),           # Quit
        ]
        res = run_commander_tui(self.mock_win)
        self.assertIsNone(res)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_run_commander_tui_too_small_window(self, mock_curs, mock_colors, mock_has_colors):
        self.mock_win.getmaxyx.return_value = (10, 40)
        self.mock_win.getch.side_effect = [ord("q")]
        res = run_commander_tui(self.mock_win)
        self.assertIsNone(res)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_run_commander_tui_help_dialog(self, mock_curs, mock_colors, mock_has_colors):
        # Press '?' (help), close help with Enter, then 'q' to quit
        self.mock_win.getch.side_effect = [
            ord("?"),           # Help
            10,                 # Close help
            ord("q"),           # Quit
        ]
        res = run_commander_tui(self.mock_win)
        self.assertIsNone(res)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_run_commander_tui_randomize_seed(self, mock_curs, mock_colors, mock_has_colors):
        # Press 'r' (randomize seed), then 'q'
        self.mock_win.getch.side_effect = [
            ord("r"),           # Randomize
            ord("q"),           # Quit
        ]
        res = run_commander_tui(self.mock_win)
        self.assertIsNone(res)

    def test_show_column_picker_dialog(self):
        # Test selecting option 1 with Down then Enter (fast single-select without space)
        self.mock_win.getch.side_effect = [curses.KEY_DOWN, 10]
        chosen = show_column_picker_dialog(self.mock_win, "Pick Column", ["col_a", "col_b", "col_c"])
        self.assertEqual(chosen, "col_b")

    def test_show_column_picker_dialog_multi_select(self):
        # Select col_a with Space, navigate down to col_b, select with Space, then Enter
        self.mock_win.getch.side_effect = [
            32,                 # Space on col_a
            curses.KEY_DOWN,    # Move to col_b
            32,                 # Space on col_b
            10,                 # Enter to confirm
        ]
        chosen = show_column_picker_dialog(self.mock_win, "Pick Column", ["col_a", "col_b", "col_c"])
        self.assertEqual(chosen, "col_a + col_b")

    def test_show_column_picker_dialog_ordering(self):
        # Select col_b first, then col_a second -> order should be col_b + col_a
        self.mock_win.getch.side_effect = [
            curses.KEY_DOWN,    # Move to col_b
            32,                 # Space on col_b
            curses.KEY_UP,      # Move up to col_a
            32,                 # Space on col_a
            10,                 # Enter to confirm
        ]
        chosen = show_column_picker_dialog(self.mock_win, "Pick Column", ["col_a", "col_b", "col_c"])
        self.assertEqual(chosen, "col_b + col_a")

    def test_show_column_picker_dialog_deselect(self):
        # Select col_a, select col_b, deselect col_b -> only col_a remains
        self.mock_win.getch.side_effect = [
            32,                 # Space on col_a
            curses.KEY_DOWN,    # Move to col_b
            32,                 # Space on col_b
            32,                 # Space on col_b again (deselect)
            10,                 # Enter to confirm
        ]
        chosen = show_column_picker_dialog(self.mock_win, "Pick Column", ["col_a", "col_b", "col_c"])
        self.assertEqual(chosen, "col_a")

    def test_show_column_picker_dialog_esc_cancel(self):
        # Press ESC (27) to cancel
        self.mock_win.getch.side_effect = [27]
        chosen = show_column_picker_dialog(self.mock_win, "Pick Column", ["col_a", "col_b"], current_val="col_a")
        self.assertEqual(chosen, "col_a")

    def test_show_text_edit_dialog_esc_cancel(self):
        # Press ESC (27) to cancel
        self.mock_win.getch.side_effect = [27]
        val = show_text_edit_dialog(self.mock_win, "Edit", "Prompt", default_val="original")
        self.assertIsNone(val)

    def test_show_text_edit_dialog_left_arrow_scrolls_and_cursor_moves(self):
        """Verify pressing Left Arrow on long path moves cursor and scrolls viewport to expose beginning."""
        long_path = "adapters/01_lora_r16_a32_lr1e-5_b4_i1000_Llama-3.2-3B-Instruct-4bit"
        # Start at end, press Left Arrow 10 times, then Enter
        self.mock_win.reset_mock()
        self.mock_win.getmaxyx.return_value = (24, 80)
        self.mock_win.getch.side_effect = [curses.KEY_LEFT] * 10 + [10]
        val = show_text_edit_dialog(self.mock_win, "Adapter Path", "Directory to store weights:", default_val=long_path)
        self.assertEqual(val, long_path)
        # Verify stdscr.move was called with changing coordinates as cursor moved left
        move_calls = [c[0] for c in self.mock_win.move.call_args_list]
        self.assertGreater(len(move_calls), 1)
        # Cursor X should have decreased
        initial_x = move_calls[0][1]
        final_x = move_calls[10][1]
        self.assertLess(final_x, initial_x, "Cursor X must move to the left when pressing Left Arrow")

    def test_show_text_edit_dialog_home_key_exposes_beginning(self):
        """Verify Home key jumps to index 0, scrolls viewport to offset 0, and allows prepending text."""
        long_path = "adapters/01_lora_r16_a32_lr1e-5_b4_i1000_Llama-3.2-3B-Instruct-4bit"
        # Home key -> type 'my_' -> Enter
        self.mock_win.reset_mock()
        self.mock_win.getmaxyx.return_value = (24, 80)
        self.mock_win.getch.side_effect = [
            curses.KEY_HOME,
            ord("m"),
            ord("y"),
            ord("_"),
            10,
        ]
        val = show_text_edit_dialog(self.mock_win, "Adapter Path", "Directory to store weights:", default_val=long_path)
        self.assertEqual(val, f"my_{long_path}")
        # Verify that after Home key, cursor moved to the left-most position of the input box
        move_calls = [c[0] for c in self.mock_win.move.call_args_list]
        home_x = move_calls[1][1]
        # At index 0, cursor_col is 0, so cursor_x is start_x + 3
        w = min(80 - 4, max(50, min(80 - 8, 86)))
        start_x = max(1, (80 - w) // 2)
        expected_home_x = start_x + 3
        self.assertEqual(home_x, expected_home_x, "Home key must place cursor at the very left edge of the input box")

    def test_show_help_dialog_esc(self):
        # Press ESC (27) to dismiss help
        self.mock_win.getch.side_effect = [27]
        show_help_dialog(self.mock_win)
        self.assertTrue(self.mock_win.refresh.called)

    def test_show_help_dialog_displays_version(self):
        """Verify the version number for mlx_commander is displayed at the top of the help dialog."""
        from mlx_commander import __version__
        self.mock_win.reset_mock()
        self.mock_win.getmaxyx.return_value = (25, 80)
        self.mock_win.getch.side_effect = [27]
        show_help_dialog(self.mock_win)
        calls = [c[0][2] for c in self.mock_win.addstr.call_args_list if len(c[0]) >= 3 and isinstance(c[0][2], str)]
        all_text = " ".join(calls)
        self.assertIn(f"v{__version__}", all_text, "Help dialog must display application version at the top")

    def test_show_results_dialog_esc(self):
        from pathlib import Path
        from mlx_commander.converter import ConversionResult
        res = ConversionResult(
            output_dir=Path("/tmp/out"),
            format_type=MLXFormat.PROMPT_COMPLETION,
            output_files={"train": Path("/tmp/out/train.jsonl")},
            record_counts={"train": 10},
            file_sizes={"train": 500},
            sample_records={"train": [{"prompt": "hi", "completion": "hello"}]},
            seed_used=42,
        )
        self.mock_win.getch.side_effect = [27]
        show_results_dialog(self.mock_win, res)
        self.assertTrue(self.mock_win.refresh.called)
        drawn_text = " ".join(str(call) for call in self.mock_win.addstr.call_args_list)
        self.assertIn("Conversion Successful!", drawn_text)
        self.assertNotIn("Fine-tuning Command", drawn_text)
        self.assertNotIn("mlx_lm.lora", drawn_text)

    def test_show_error_dialog_dismiss_enter(self):
        self.mock_win.getch.side_effect = [10]  # Enter key
        show_error_dialog(self.mock_win, "Test Error", "An unexpected failure occurred.")
        self.assertTrue(self.mock_win.refresh.called)
        # Verify double line borders were drawn
        has_double_border = any("╔" in str(call) for call in self.mock_win.addstr.call_args_list)
        self.assertTrue(has_double_border)

    def test_show_error_dialog_dismiss_esc(self):
        self.mock_win.getch.side_effect = [27]  # Esc key
        show_error_dialog(self.mock_win, "Test Error", ["Line 1", "Line 2"])
        self.assertTrue(self.mock_win.refresh.called)

    def test_show_results_dialog_resilience_to_legacy_call(self):
        # Verify calling with (stdscr, False, "error message") doesn't crash with TypeError
        self.mock_win.getch.side_effect = [10]
        show_results_dialog(self.mock_win, False, "Dataset Loading Failed")
        self.assertTrue(self.mock_win.refresh.called)

    def test_show_message_dialog_overlay(self):
        self.mock_win.getch.side_effect = [10]
        show_message_dialog(self.mock_win, "Notice", ["Everything is okay."])
        self.assertTrue(self.mock_win.refresh.called)

    def test_configure_escdelay(self):
        import os
        from mlx_commander.tui.widgets import configure_escdelay

        configure_escdelay(25)
        self.assertEqual(os.environ.get("ESCDELAY"), "25")
        if hasattr(curses, "get_escdelay"):
            # If curses supports get_escdelay
            try:
                self.assertEqual(curses.get_escdelay(), 25)
            except curses.error:
                pass


    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_run_commander_tui_wrapped_preview(self, mock_curs, mock_colors, mock_has_colors):
        state = CommanderState()
        long_text = "Word " * 50
        state.preview_cache = [
            f'{{"prompt": "{long_text}", "completion": "Answer 1"}}',
            f'{{"prompt": "Example 2", "completion": "{long_text}"}}',
            f'{{"prompt": "Example 3", "completion": "Answer 3"}}',
        ]
        self.mock_win.getmaxyx.return_value = (35, 90)
        self.mock_win.getch.side_effect = [ord("q")]
        res = run_commander_tui(self.mock_win, initial_state=state)
        self.assertIsNone(res)
        self.assertTrue(self.mock_win.refresh.called)

    def test_get_schema_mapping_targets_prompt_completion(self):
        from mlx_commander.formats import ColumnMapping, MLXFormat
        from mlx_commander.tui.widgets import get_schema_mapping_targets

        mapping = ColumnMapping(prompt_col="instruction + input", completion_col="output")
        columns = ["id", "instruction", "input", "output", "system_prompt"]
        targets, unmapped = get_schema_mapping_targets(MLXFormat.PROMPT_COMPLETION, mapping, columns)

        self.assertEqual(len(targets), 2)
        self.assertEqual(targets[0]["key"], "prompt")
        self.assertEqual(targets[0]["cols"], ["instruction", "input"])
        self.assertTrue(targets[0]["is_concat"])

        self.assertEqual(targets[1]["key"], "completion")
        self.assertEqual(targets[1]["cols"], ["output"])
        self.assertFalse(targets[1]["is_concat"])

        self.assertEqual(unmapped, ["id", "system_prompt"])

    def test_get_schema_mapping_targets_formats(self):
        from mlx_commander.formats import ColumnMapping, MLXFormat
        from mlx_commander.tui.widgets import get_schema_mapping_targets

        # Chat format
        chat_map = ColumnMapping(user_col="query", assistant_col="response", system_col="sys")
        cols = ["sys", "query", "response", "extra"]
        targets, unmapped = get_schema_mapping_targets(MLXFormat.CHAT, chat_map, cols)
        self.assertEqual([t["key"] for t in targets], ["user", "assistant", "system"])
        self.assertEqual(unmapped, ["extra"])

        # DPO format
        dpo_map = ColumnMapping(dpo_prompt_col="prompt_text", chosen_col="chosen_text", rejected_col="rejected_text")
        dpo_cols = ["prompt_text", "chosen_text", "rejected_text", "meta"]
        targets, unmapped = get_schema_mapping_targets(MLXFormat.DPO, dpo_map, dpo_cols)
        self.assertEqual([t["key"] for t in targets], ["prompt", "chosen", "rejected"])
        self.assertEqual(unmapped, ["meta"])

        # Text format
        text_map = ColumnMapping(text_col="raw_body")
        targets, unmapped = get_schema_mapping_targets(MLXFormat.TEXT, text_map, ["raw_body"])
        self.assertEqual([t["key"] for t in targets], ["text"])
        self.assertEqual(unmapped, [])

    def test_plan_mapping_panel_rows(self):
        from mlx_commander.tui.widgets import plan_mapping_panel_rows

        targets = [
            {"key": "prompt", "cols": ["instruction", "input"]},
            {"key": "completion", "cols": ["output"]},
        ]
        unmapped = ["id", "meta"]

        # Compact space: 5 rows (fits borders and 1 row per target)
        heights, show_unmapped = plan_mapping_panel_rows(targets, unmapped, avail_rows=5)
        self.assertEqual(heights, [1, 1])
        self.assertFalse(show_unmapped)

        # Space with room for unmapped: 7 rows
        heights, show_unmapped = plan_mapping_panel_rows(targets, unmapped, avail_rows=7)
        self.assertEqual(heights, [1, 1])
        self.assertTrue(show_unmapped)

        # Generous space: 9 rows (gives extra row to multi-column target)
        heights, show_unmapped = plan_mapping_panel_rows(targets, unmapped, avail_rows=9)
        self.assertEqual(heights, [2, 1])
        self.assertTrue(show_unmapped)

    def test_draw_mapping_pipeline_panel_no_emojis(self):
        from mlx_commander.formats import ColumnMapping, MLXFormat
        from mlx_commander.loader import LoadedDataset
        from mlx_commander.tui.widgets import draw_mapping_pipeline_panel

        state = CommanderState()
        state.target_format = MLXFormat.PROMPT_COMPLETION
        state.mapping = ColumnMapping(prompt_col="instruction + input", completion_col="output")
        state.loaded_dataset = LoadedDataset(
            source_path="/tmp/test.jsonl",
            is_split=False,
            split_names=["train"],
            split_counts={"train": 10},
            columns=["instruction", "input", "output", "id"],
            total_rows=10,
            sample_records=[{"instruction": "say hi", "input": "now", "output": "hello", "id": 1}],
        )

        printed_strings = []

        def fake_addstr(y, x, s, attr=0):
            printed_strings.append(s)

        self.mock_win.addstr.side_effect = fake_addstr
        self.mock_win.getmaxyx.return_value = (30, 90)

        # Test panel rendering across multiple heights
        for h in [6, 8, 12]:
            printed_strings.clear()
            draw_mapping_pipeline_panel(self.mock_win, y=10, x=0, h=h, w=90, state=state)
            self.assertTrue(len(printed_strings) > 0)

            # Assert ZERO emojis exist in any rendered string
            all_text = "".join(printed_strings)
            for ch in all_text:
                code = ord(ch)
                # Ensure no high Unicode emoji block characters
                is_emoji = (
                    0x1F300 <= code <= 0x1F9FF
                    or 0x2600 <= code <= 0x27BF
                    or 0x1F600 <= code <= 0x1F64F
                )
                self.assertFalse(
                    is_emoji,
                    f"Found forbidden emoji character '{ch}' (code {code:X}) in mapping panel output!",
                )

        # Verify that Target tags exist, but Source tags are removed from the left side
        self.assertFalse(any("Source: prompt" in s for s in printed_strings))
        self.assertFalse(any("Source: completion" in s for s in printed_strings))
        self.assertTrue(any("Target: prompt" in s for s in printed_strings))
        self.assertTrue(any("Target: completion" in s for s in printed_strings))
        self.assertTrue(any("instruction" in s for s in printed_strings))
        self.assertTrue(any("output" in s for s in printed_strings))
        self.assertTrue(any("Concat" in s for s in printed_strings))

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_run_commander_tui_multi_file_tip(self, mock_curs, mock_colors, mock_has_colors):
        printed_strings = []

        def fake_addstr(y, x, s, attr=0):
            printed_strings.append(s)

        self.mock_win.addstr.side_effect = fake_addstr
        self.mock_win.getmaxyx.return_value = (30, 100)
        self.mock_win.getch.side_effect = [ord("q")]

        run_commander_tui(self.mock_win, initial_state=CommanderState(active_tab=0))
        # Verify multi-file tip is drawn
        has_tip = any("Tip: You can load multiple files" in s for s in printed_strings)
        self.assertTrue(has_tip, "Multi-file loading tip was not found in TUI rendered output!")

    def test_commander_state_output_dir_default_and_custom(self):
        import json
        import tempfile
        from pathlib import Path
        from mlx_commander.tui.state import CommanderState

        temp_d = tempfile.mkdtemp()
        sub_d = Path(temp_d) / "my_hf_data"
        sub_d.mkdir(parents=True)
        sample_file = sub_d / "data.jsonl"
        with open(sample_file, "w") as f:
            f.write(json.dumps({"prompt": "p", "completion": "c"}) + "\n")

        state = CommanderState()
        # Loading dataset should set output_dir to <dataset_dir>/mlx_dataset
        state.load_dataset(str(sample_file))
        expected_default = str((sub_d / "mlx_dataset").resolve())
        self.assertEqual(state.output_dir, expected_default)
        self.assertFalse(state.has_custom_output_dir)

        # Customizing output_dir should be preserved on subsequent loads
        custom_out = str(Path(temp_d) / "my_custom_mlx")
        state.output_dir = custom_out
        state.has_custom_output_dir = True

        state.load_dataset(str(sample_file))
        self.assertEqual(state.output_dir, custom_out)

    def test_show_output_destination_dialog_esc(self):
        from mlx_commander.tui.widgets import show_output_destination_dialog
        self.mock_win.getch.side_effect = [27]  # Esc
        res = show_output_destination_dialog(self.mock_win, "/tmp/curr", "/tmp/default")
        self.assertIsNone(res)

    @patch("mlx_commander.gui_picker.is_macos", return_value=False)
    def test_show_output_destination_dialog_reset(self, mock_is_mac):
        from mlx_commander.tui.widgets import show_output_destination_dialog
        # On non-mac: option 0 = Manual, option 1 = Reset to default
        self.mock_win.getch.side_effect = [curses.KEY_DOWN, 10]
        res = show_output_destination_dialog(self.mock_win, "/tmp/curr", "/tmp/default")
        self.assertEqual(res, "/tmp/default")

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_vertical_navigation_left_panel_arrows(self, mock_curs, mock_colors, mock_has_colors):
        # Left panel starts at left_focus_idx = 0 (Dataset field).
        # With columns loaded, Down moves to column list (left_focus_idx = 1, selected_column_idx = 0)
        # Down again moves to next column (selected_column_idx = 1)
        # Up moves back to first column (selected_column_idx = 0)
        # Up again moves back to Dataset field (left_focus_idx = 0)
        from mlx_commander.loader import LoadedDataset
        state = CommanderState(active_tab=0)
        state.loaded_dataset = LoadedDataset(
            source_path="mock.jsonl",
            is_split=False,
            split_names=["train"],
            split_counts={"train": 10},
            columns=["col_a", "col_b", "col_c"],
            total_rows=10,
            sample_records=[{"col_a": "1", "col_b": "2", "col_c": "3"}],
        )
        state.active_panel = ActivePanel.LEFT
        state.left_focus_idx = 0

        self.mock_win.getch.side_effect = [
            curses.KEY_RIGHT,  # moves to column list (index 1, col 0)
            curses.KEY_RIGHT,  # moves to col 1
            curses.KEY_LEFT,   # moves back to col 0
            ord("q"),          # quit
        ]
        run_commander_tui(self.mock_win, initial_state=state)
        self.assertEqual(state.left_focus_idx, 1)
        self.assertEqual(state.selected_column_idx, 0)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_vertical_navigation_right_panel_and_format_toggle(self, mock_curs, mock_colors, mock_has_colors):
        # Right panel: format toggles are stacked vertically at indices 0, 1, 2, 3.
        # Starting at right_focus_idx = 0 (prompt_comp).
        # Right arrow moves down to index 1 (chat).
        # Enter selects format at index 1 -> target_format should become CHAT.
        # Right arrow moves down to index 2 (text).
        # Space selects format at index 2 -> target_format should become TEXT.
        state = CommanderState(active_tab=0)
        state.active_panel = ActivePanel.RIGHT
        state.right_focus_idx = 0

        self.mock_win.getch.side_effect = [
            curses.KEY_RIGHT,  # moves down to index 1 (chat)
            10,                # Enter -> selects CHAT
            curses.KEY_RIGHT,  # moves down to index 2 (text)
            32,                # Space -> selects TEXT
            curses.KEY_LEFT,   # moves up back to index 1 (chat)
            ord("q"),          # quit
        ]
        run_commander_tui(self.mock_win, initial_state=state)
        self.assertEqual(state.target_format, MLXFormat.TEXT)
        self.assertEqual(state.right_focus_idx, 1)

    def test_dialog_left_right_arrow_navigation(self):
        from mlx_commander.tui.widgets import run_menu, show_column_picker_dialog, show_output_destination_dialog

        # show_column_picker_dialog: KEY_RIGHT acts as DOWN (selects option 1)
        self.mock_win.getch.side_effect = [curses.KEY_RIGHT, 10]
        chosen = show_column_picker_dialog(self.mock_win, "Pick Column", ["col_a", "col_b", "col_c"])
        self.assertEqual(chosen, "col_b")

        # show_column_picker_dialog: KEY_RIGHT then KEY_LEFT acts as DOWN then UP (selects option 0)
        self.mock_win.getch.side_effect = [curses.KEY_RIGHT, curses.KEY_LEFT, 10]
        chosen = show_column_picker_dialog(self.mock_win, "Pick Column", ["col_a", "col_b", "col_c"])
        self.assertEqual(chosen, "col_a")

        # run_menu: KEY_RIGHT acts as DOWN, KEY_LEFT acts as UP
        self.mock_win.getch.side_effect = [curses.KEY_RIGHT, 10]
        res = run_menu(self.mock_win, "Title", "Sub", [("Opt 0", "d0"), ("Opt 1", "d1")])
        self.assertEqual(res, 1)

        self.mock_win.getch.side_effect = [curses.KEY_RIGHT, curses.KEY_LEFT, 10]
        res = run_menu(self.mock_win, "Title", "Sub", [("Opt 0", "d0"), ("Opt 1", "d1")])
        self.assertEqual(res, 0)


    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_classic_blue_color_scheme_toggle_f9_only(self, mock_curs, mock_colors, mock_has_colors):
        from mlx_commander.tui.state import ThemeMode
        state = CommanderState()
        self.assertEqual(state.theme_mode, ThemeMode.MODERN)

        # Press F9 -> switches to Classic Blue
        # Press 9 -> ignored (does not toggle)
        # Press t -> ignored (does not toggle)
        # Press T -> ignored (does not toggle)
        # Press F9 -> switches back to Modern
        # Press q -> quit
        self.mock_win.getch.side_effect = [
            curses.KEY_F9,
            ord("9"),
            ord("t"),
            ord("T"),
            curses.KEY_F9,
            ord("q"),
        ]
        run_commander_tui(self.mock_win, initial_state=state)
        # Verify init_colors was called with both modern and classic_blue
        mock_colors.assert_any_call(ThemeMode.MODERN)
        mock_colors.assert_any_call(ThemeMode.CLASSIC_BLUE)
        self.assertEqual(state.theme_mode, ThemeMode.MODERN)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.curses.can_change_color", return_value=True)
    @patch("mlx_commander.tui.app.curses.init_color")
    @patch("mlx_commander.tui.app.curses.start_color")
    @patch("mlx_commander.tui.app.curses.use_default_colors")
    @patch("mlx_commander.tui.app.curses.init_pair")
    def test_init_colors_modern_and_classic_blue(self, mock_init_pair, mock_default, mock_start, mock_init_color, mock_can_change, mock_has_colors):
        from mlx_commander.tui.app import init_colors
        from mlx_commander.tui.widgets import (
            COLOR_BANNER,
            COLOR_BORDER_JOINTS,
            COLOR_INPUT_FOCUSED,
            COLOR_INPUT_NORMAL,
            COLOR_LABEL_GRAY,
            COLOR_NORMAL_TEXT,
            COLOR_PANEL_BG,
            COLOR_TITLE_ACCENT,
        )

        init_colors("modern")
        mock_init_pair.assert_any_call(COLOR_BANNER, curses.COLOR_WHITE, curses.COLOR_BLUE)
        mock_init_pair.assert_any_call(COLOR_BORDER_JOINTS, curses.COLOR_CYAN, -1)
        mock_init_pair.assert_any_call(COLOR_NORMAL_TEXT, curses.COLOR_WHITE, -1)
        mock_init_pair.assert_any_call(COLOR_INPUT_NORMAL, curses.COLOR_CYAN, -1)

        mock_init_pair.reset_mock()
        init_colors("classic_blue")
        # Verify 6 exact VGA colors initialized
        mock_init_color.assert_any_call(20, 0, 0, 666)          # #0000AA (Classic VGA Blue)
        mock_init_color.assert_any_call(21, 0, 666, 666)        # #00AAAA (Cyan)
        mock_init_color.assert_any_call(22, 1000, 1000, 1000)   # #FFFFFF (Bright White)
        mock_init_color.assert_any_call(23, 820, 820, 820)      # #D1D1D1 (Off-White / Light Gray)
        mock_init_color.assert_any_call(24, 1000, 1000, 333)    # #FFFF55 (Bright Yellow)
        mock_init_color.assert_any_call(25, 0, 0, 0)            # #000000 (Black)
        mock_init_color.assert_any_call(26, 333, 1000, 1000)   # #55FFFF (Ice Blue)

        # Verify exact pair mappings
        mock_init_pair.assert_any_call(COLOR_BANNER, 25, 21)         # Black on Cyan
        mock_init_pair.assert_any_call(COLOR_BORDER_JOINTS, 21, 20)  # Cyan on Blue
        mock_init_pair.assert_any_call(COLOR_NORMAL_TEXT, 22, 20)    # White on Blue
        mock_init_pair.assert_any_call(COLOR_TITLE_ACCENT, 24, 20)   # Yellow on Blue
        mock_init_pair.assert_any_call(COLOR_LABEL_GRAY, 23, 20)     # Gray on Blue
        mock_init_pair.assert_any_call(COLOR_INPUT_NORMAL, 26, 20)   # Ice Blue on Deep Blue
        mock_init_pair.assert_any_call(COLOR_INPUT_FOCUSED, 25, 21)  # Black on Cyan
        mock_init_pair.assert_any_call(COLOR_PANEL_BG, 22, 20)       # White on Blue


    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_classic_blue_bottom_bar_rendering(self, mock_curs, mock_colors, mock_has_colors):
        from mlx_commander.tui.state import ThemeMode
        state = CommanderState(active_tab=0)
        state.theme_mode = ThemeMode.CLASSIC_BLUE

        self.mock_win.getch.side_effect = [ord("q")]
        run_commander_tui(self.mock_win, initial_state=state)

        # Verify that bottom bar was drawn with orthodox commander style Classic Blue buttons
        rendered_strings = [
            call_args[0][2]
            for call_args in self.mock_win.addstr.call_args_list
            if len(call_args[0]) >= 3 and isinstance(call_args[0][2], str)
        ]
        full_rendered = " ".join(rendered_strings)
        self.assertIn("Tab", full_rendered)
        self.assertIn("Switch", full_rendered)
        self.assertIn("Enter", full_rendered)
        self.assertIn("Select", full_rendered)
        self.assertIn("Help", full_rendered)
        self.assertNotIn("Open", full_rendered)
        self.assertIn("Mode", full_rendered)
        self.assertIn("Output", full_rendered)
        self.assertIn("Convert", full_rendered)
        self.assertIn("Theme", full_rendered)
        self.assertNotIn("Scheme", full_rendered)
        self.assertIn("F9", full_rendered)
        self.assertIn("Exit", full_rendered)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_classic_blue_bottom_bar_rendering_mode1_and_mode2(self, mock_curs, mock_colors, mock_has_colors):
        from mlx_commander.tui.state import ThemeMode

        # 1. Mode 1 (Single Run) in Classic Blue Theme
        state1 = CommanderState(active_tab=1)
        state1.theme_mode = ThemeMode.CLASSIC_BLUE
        self.mock_win.reset_mock()
        self.mock_win.getmaxyx.return_value = (30, 110)
        self.mock_win.getch.side_effect = [ord("q")]
        run_commander_tui(self.mock_win, initial_state=state1)

        rendered_mode1 = " ".join(
            call_args[0][2]
            for call_args in self.mock_win.addstr.call_args_list
            if len(call_args[0]) >= 3 and isinstance(call_args[0][2], str)
        )
        self.assertIn("Tab", rendered_mode1)
        self.assertIn("Switch", rendered_mode1)
        self.assertIn("Enter", rendered_mode1)
        self.assertIn("Select", rendered_mode1)
        self.assertIn("Clone", rendered_mode1)
        self.assertIn("Del", rendered_mode1)
        self.assertIn("Help", rendered_mode1)
        self.assertIn("Mode", rendered_mode1)
        self.assertIn("Run", rendered_mode1)
        self.assertIn("Add", rendered_mode1)
        self.assertIn("Theme", rendered_mode1)
        self.assertIn("Exit", rendered_mode1)

        # 2. Mode 2 (Multi-Run Matrix) in Classic Blue Theme
        state2 = CommanderState(active_tab=2)
        state2.theme_mode = ThemeMode.CLASSIC_BLUE
        self.mock_win.reset_mock()
        self.mock_win.getmaxyx.return_value = (30, 110)
        self.mock_win.getch.side_effect = [ord("q")]
        run_commander_tui(self.mock_win, initial_state=state2)

        rendered_mode2 = " ".join(
            call_args[0][2]
            for call_args in self.mock_win.addstr.call_args_list
            if len(call_args[0]) >= 3 and isinstance(call_args[0][2], str)
        )
        self.assertIn("Tab", rendered_mode2)
        self.assertIn("Switch", rendered_mode2)
        self.assertIn("Enter", rendered_mode2)
        self.assertIn("Amend", rendered_mode2)
        self.assertIn("Help", rendered_mode2)
        self.assertIn("Mode", rendered_mode2)
        self.assertIn("Run Sweep", rendered_mode2)
        self.assertIn("Add Sweep", rendered_mode2)
        self.assertIn("Theme", rendered_mode2)
        self.assertIn("Exit", rendered_mode2)

        # 3. Compact mode (< 75 cols, >= 70 min required) in Mode 1
        state_compact = CommanderState(active_tab=1)
        state_compact.theme_mode = ThemeMode.CLASSIC_BLUE
        self.mock_win.reset_mock()
        self.mock_win.getmaxyx.return_value = (30, 72)
        self.mock_win.getch.side_effect = [ord("q")]
        run_commander_tui(self.mock_win, initial_state=state_compact)

        rendered_compact = " ".join(
            call_args[0][2]
            for call_args in self.mock_win.addstr.call_args_list
            if len(call_args[0]) >= 3 and isinstance(call_args[0][2], str)
        )
        self.assertIn("Help", rendered_compact)
        self.assertIn("Mode", rendered_compact)
        self.assertIn("Run", rendered_compact)
        self.assertIn("Theme", rendered_compact)
        self.assertIn("Exit", rendered_compact)


    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_cross_tab_vertical_navigation(self, mock_curs, mock_colors, mock_has_colors):
        from mlx_commander.tui.state import ActivePanel
        state = CommanderState(active_tab=0)
        state.active_panel = ActivePanel.LEFT
        state.left_focus_idx = 0  # Unified Dataset field (no dataset loaded)

        # Press Down -> should transition to Tab 2 (right_focus_idx = 0)
        # Press Up -> should transition back to Tab 1 (left_focus_idx = 0)
        # Press q -> quit
        self.mock_win.getch.side_effect = [
            curses.KEY_DOWN,
            curses.KEY_UP,
            ord("q"),
        ]
        run_commander_tui(self.mock_win, initial_state=state)
        self.assertEqual(state.active_panel, ActivePanel.LEFT)
        self.assertEqual(state.left_focus_idx, 0)


    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_output_field_and_convert_button_no_row_collision(self, mock_curs, mock_colors, mock_has_colors):
        """Verify Output field and Convert Dataset button render on separate rows across formats."""
        for fmt in [MLXFormat.PROMPT_COMPLETION, MLXFormat.DPO, MLXFormat.TEXT, MLXFormat.CHAT]:
            self.mock_win.reset_mock()
            self.mock_win.getmaxyx.return_value = (24, 80)
            state = CommanderState(active_tab=0)
            state.target_format = fmt
            self.mock_win.getch.side_effect = [ord("q")]

            run_commander_tui(self.mock_win, initial_state=state)

            output_y = None
            convert_y = None
            for call_args in self.mock_win.addstr.call_args_list:
                args = call_args[0]
                if len(args) >= 3 and isinstance(args[2], str):
                    y, x, text = args[0], args[1], args[2]
                    if "Output:" in text:
                        output_y = y
                    elif "Convert Dataset" in text:
                        convert_y = y

            self.assertIsNotNone(output_y, f"Output label not found for format {fmt}")
            self.assertIsNotNone(convert_y, f"Convert button not found for format {fmt}")
            self.assertNotEqual(
                output_y,
                convert_y,
                f"Collision: Output field and Convert button share row {output_y} for format {fmt}"
            )
            self.assertEqual(
                convert_y,
                output_y + 1,
                f"Convert button row {convert_y} should be exactly output_y + 1 ({output_y + 1}) for format {fmt}"
            )


    def test_draw_field_path_truncation(self):
        """Verify draw_field truncates paths from the front so the directory/filename is visible without brackets."""
        from mlx_commander.tui.widgets import draw_field

        self.mock_win.reset_mock()
        long_path = "/Users/tomkubik/Downloads/Projects/toMLXfromHF – dataset converter/mlx_dataset"
        draw_field(self.mock_win, 5, 2, "Output", long_path, val_width=24, has_dropdown=True)

        box_str = None
        for call_args in self.mock_win.addstr.call_args_list:
            args = call_args[0]
            if len(args) >= 3 and isinstance(args[2], str) and "mlx_dataset" in args[2]:
                box_str = args[2]

        self.assertIsNotNone(box_str)
        self.assertIn("mlx_dataset", box_str)
        self.assertIn("…", box_str)
        self.assertNotIn("[", box_str)
        self.assertNotIn("]", box_str)


    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_splits_fields_right_brackets_visible(self, mock_curs, mock_colors, mock_has_colors):
        """Verify Train, Valid, and Test fields are right-aligned with no brackets."""
        for term_w in [130, 100]:
            self.mock_win.reset_mock()
            self.mock_win.getmaxyx.return_value = (30, term_w)
            state = CommanderState(active_tab=0)
            self.mock_win.getch.side_effect = [ord("q")]

            run_commander_tui(self.mock_win, initial_state=state)

            splits_y = 8 + len(state.get_mapping_fields_for_format())

            def get_row_str(target_y):
                row_chars = [" "] * term_w
                for call_args in self.mock_win.addstr.call_args_list:
                    args = call_args[0]
                    if len(args) >= 3 and isinstance(args[2], str):
                        y, x, text = args[0], args[1], args[2]
                        if y == target_y:
                            for i, ch in enumerate(text):
                                if x + i < term_w:
                                    row_chars[x + i] = ch
                return "".join(row_chars)

            train_row = get_row_str(splits_y)
            valid_row = get_row_str(splits_y + 1)
            test_row = get_row_str(splits_y + 2)

            self.assertIn("Dataset split:", train_row, f"Dataset split label missing on row {splits_y}: {train_row}")
            self.assertIn("Train:", train_row)
            self.assertIn("80%", train_row)
            self.assertNotIn("[", train_row[train_row.index("Train:"):])
            self.assertNotIn("]", train_row[train_row.index("Train:"):])
            self.assertIn("Valid:", valid_row)
            self.assertIn("10%", valid_row)
            self.assertNotIn("[", valid_row[valid_row.index("Valid:"):])
            self.assertNotIn("]", valid_row[valid_row.index("Valid:"):])
            self.assertIn("Test:", test_row)
            self.assertIn("10%", test_row)
            self.assertNotIn("[", test_row[test_row.index("Test:"):])
            self.assertNotIn("]", test_row[test_row.index("Test:"):])

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_splits_layout_vertical_alignment_and_white_labels(self, mock_curs, mock_colors, mock_has_colors):
        """Verify:
        1. 'Dataset split:' label is aligned to left (at left_w + 2).
        2. 'Train' starts on the same line as 'Dataset split'.
        3. 'Valid' and 'Test' are on consecutive lines below.
        4. Labels for 'Train', 'Valid', and 'Test' are in white font, unbolded.
        5. 'Train', 'Valid', and 'Test' input fields are aligned to the right.
        6. 'Train', 'Valid', and 'Test' labels are aligned to left with an indent.
        """
        term_w = 120
        self.mock_win.reset_mock()
        self.mock_win.getmaxyx.return_value = (30, term_w)
        left_w = term_w // 2  # 60
        right_w = term_w - left_w  # 60
        right_edge = left_w + right_w - 3  # 117

        state = CommanderState(active_tab=0)
        self.mock_win.getch.side_effect = [ord("q")]

        run_commander_tui(self.mock_win, initial_state=state)

        splits_y = 8 + len(state.get_mapping_fields_for_format())

        # Check 'Dataset split:' label position & attribute
        ds_split_found = False
        for call_args in self.mock_win.addstr.call_args_list:
            args = call_args[0]
            if len(args) >= 3 and isinstance(args[2], str):
                y, x, text = args[0], args[1], args[2]
                if y == splits_y and "Dataset split:" in text:
                    self.assertEqual(x, left_w + 2, "Dataset split label must be aligned to left (left_w + 2)")
                    ds_split_found = True
        self.assertTrue(ds_split_found, "Dataset split label not found")

        # Track rows, coordinates, attributes, and input box endpoints
        splits_info = {}
        for split_name, expected_y in [("Train", splits_y), ("Valid", splits_y + 1), ("Test", splits_y + 2)]:
            lbl_x = None
            lbl_attr = None
            box_end_x = None
            for call_args in self.mock_win.addstr.call_args_list:
                args = call_args[0]
                if len(args) >= 3 and isinstance(args[2], str):
                    y, x, text = args[0], args[1], args[2]
                    attr = args[3] if len(args) >= 4 else 0
                    if y == expected_y:
                        if f"{split_name}:" in text:
                            lbl_x = x
                            lbl_attr = attr
                        elif "%" in text:
                            box_end_x = x + len(text) - 1
            splits_info[split_name] = {"x": lbl_x, "attr": lbl_attr, "box_end": box_end_x}

        # Verify Train, Valid, Test labels are indented and aligned with each other
        train_x = splits_info["Train"]["x"]
        valid_x = splits_info["Valid"]["x"]
        test_x = splits_info["Test"]["x"]

        self.assertIsNotNone(train_x)
        self.assertGreater(train_x, left_w + 2, "Train label must be indented on the left side")
        self.assertEqual(train_x, right_edge - 14, "Train label must be indented right next to input fields")
        self.assertEqual(valid_x, train_x, "Valid label must align with Train label")
        self.assertEqual(test_x, train_x, "Test label must align with Train label")

        # Verify all three input fields are right-aligned to right_edge
        for name in ["Train", "Valid", "Test"]:
            self.assertEqual(
                splits_info[name]["box_end"],
                right_edge,
                f"{name} input field must end at right_edge ({right_edge})",
            )

        # Verify labels are unbolded (no curses.A_BOLD)
        for name in ["Train", "Valid", "Test"]:
            attr = splits_info[name]["attr"]
            self.assertIsNotNone(attr, f"{name} label attribute should be recorded")
            self.assertEqual(
                attr & curses.A_BOLD,
                0,
                f"{name} label must be unbolded",
            )



    def test_draw_field_right_edge_alignment(self):
        """Verify draw_field aligns the field to right_edge while keeping label at x."""
        from mlx_commander.tui.widgets import draw_field

        self.mock_win.reset_mock()
        draw_field(self.mock_win, 5, 10, "Prompt", "instruction", val_width=18, right_edge=75)

        lbl_x = None
        box_x = None
        box_str = None
        for call_args in self.mock_win.addstr.call_args_list:
            args = call_args[0]
            if len(args) >= 3 and isinstance(args[2], str):
                y, x, text = args[0], args[1], args[2]
                if y == 5 and "Prompt:" in text:
                    lbl_x = x
                elif y == 5 and "instruction" in text:
                    box_x = x
                    box_str = text

        self.assertEqual(lbl_x, 10, "Label should remain at x=10")
        self.assertIsNotNone(box_x)
        self.assertIsNotNone(box_str)
        self.assertEqual(box_x + len(box_str) - 1, 75, "Field should end exactly at right_edge=75")
        self.assertNotIn("[", box_str)
        self.assertNotIn("]", box_str)


    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_tui_fields_right_aligned_and_labels_left_aligned(self, mock_curs, mock_colors, mock_has_colors):
        """Verify all input fields in Tab 2 are right-aligned while labels remain left-aligned."""
        self.mock_win.reset_mock()
        term_w = 120
        self.mock_win.getmaxyx.return_value = (30, term_w)
        left_w = term_w // 2  # 60
        right_w = term_w - left_w  # 60
        tab2_right_edge = left_w + right_w - 3  # 117
        tab1_right_edge = left_w - 3  # 57

        state = CommanderState(active_tab=0)
        self.mock_win.getch.side_effect = [ord("q")]

        run_commander_tui(self.mock_win, initial_state=state)

        # 1. Verify Tab 1 Dataset label is at left (2) and field ends at tab1_right_edge (57)
        ds_lbl_found = False
        ds_val_end = None
        for call_args in self.mock_win.addstr.call_args_list:
            args = call_args[0]
            if len(args) >= 3 and isinstance(args[2], str):
                y, x, text = args[0], args[1], args[2]
                if y == 2 and "Dataset:" in text:
                    self.assertEqual(x, 2, "Tab 1 Dataset label must be aligned to left (x=2)")
                    ds_lbl_found = True
                elif y == 2 and 8 <= x < left_w and not text.startswith("┌") and not text.startswith("│"):
                    ds_val_end = x + len(text) - 1

        self.assertTrue(ds_lbl_found)
        self.assertEqual(ds_val_end, tab1_right_edge, f"Dataset display in Tab 1 should end at {tab1_right_edge}")

        # 2. Verify Tab 2 mapping labels start at left_w + 2 and boxes end at tab2_right_edge
        mapping_fields = state.get_mapping_fields_for_format()
        for idx in range(len(mapping_fields)):
            row_y = 8 + idx
            lbl_x = None
            box_x = None
            box_text = None
            for call_args in self.mock_win.addstr.call_args_list:
                args = call_args[0]
                if len(args) >= 3 and isinstance(args[2], str):
                    y, x, text = args[0], args[1], args[2]
                    if y == row_y:
                        if ":" in text:
                            lbl_x = x
                        else:
                            box_x = x
                            box_text = text

            self.assertEqual(lbl_x, left_w + 2, f"Mapping label row {row_y} should start at left_w + 2")
            self.assertIsNotNone(box_x, f"Mapping box row {row_y} not found")
            self.assertEqual(
                box_x + len(box_text) - 1,
                tab2_right_edge,
                f"Mapping box row {row_y} should end at right edge {tab2_right_edge}"
            )

        # 3. Verify Tab 2 Output label is at left_w + 2 and box ends at tab2_right_edge
        out_lbl_x = None
        out_box_x = None
        out_box_text = None
        for call_args in self.mock_win.addstr.call_args_list:
            args = call_args[0]
            if len(args) >= 3 and isinstance(args[2], str):
                y, x, text = args[0], args[1], args[2]
                if "Output:" in text:
                    out_lbl_x = x
                elif out_lbl_x is not None and "mlx_dataset" in text:
                    out_box_x = x
                    out_box_text = text

        self.assertEqual(out_lbl_x, left_w + 2, "Output label must be at left_w + 2")
        self.assertIsNotNone(out_box_x, "Output box must be rendered")
        self.assertEqual(
            out_box_x + len(out_box_text) - 1,
            tab2_right_edge,
            f"Output box should end at {tab2_right_edge}"
        )


    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_initial_tui_no_dataset_empty_preview(self, mock_curs, mock_colors, mock_has_colors):
        """Verify that starting MLX Commander without a dataset in Mode 1 shows (No dataset selected)."""
        self.mock_win.reset_mock()
        self.mock_win.getmaxyx.return_value = (30, 100)
        self.mock_win.getch.side_effect = [ord("q")]

        # Run with initial_state Mode 1
        run_commander_tui(self.mock_win, initial_state=CommanderState(active_tab=0))

        all_rendered_text = []
        no_ds_found = False
        for call_args in self.mock_win.addstr.call_args_list:
            args = call_args[0]
            if len(args) >= 3 and isinstance(args[2], str):
                text = args[2]
                all_rendered_text.append(text)
                if "(No dataset selected)" in text:
                    no_ds_found = True

        self.assertTrue(no_ds_found, "Live preview panel must show '(No dataset selected)' when no dataset is provided")

        full_dump = " ".join(all_rendered_text)
        self.assertNotIn("datasets>=2.14.0", full_dump, "Must not auto-load requirements.txt or current directory")
        self.assertNotIn("pyarrow>=12.0.0", full_dump, "Must not auto-load requirements.txt or current directory")

    @patch("mlx_commander.tui.widgets.safe_has_colors", return_value=True)
    def test_draw_field_decluttered_typography(self, mock_has_colors):
        """Verify draw_field renders values without [ ] brackets, using font color and focus highlight."""
        from mlx_commander.tui.widgets import draw_field, COLOR_INPUT_NORMAL, COLOR_INPUT_FOCUSED, get_color

        # 1. Unfocused field
        self.mock_win.reset_mock()
        draw_field(self.mock_win, 4, 2, "Batch Size", "4", is_focused=False, val_width=8, right_edge=50)

        unfoc_call = None
        for call_args in self.mock_win.addstr.call_args_list:
            args = call_args[0]
            if len(args) >= 3 and isinstance(args[2], str) and "4" in args[2]:
                unfoc_call = args

        self.assertIsNotNone(unfoc_call)
        y, x, text, attr = unfoc_call[0], unfoc_call[1], unfoc_call[2], unfoc_call[3]
        self.assertEqual(text, " 4 ")
        self.assertNotIn("[", text)
        self.assertNotIn("]", text)
        self.assertEqual(x + len(text) - 1, 50, "Right-edge alignment must be exact")
        self.assertEqual(attr, get_color(COLOR_INPUT_NORMAL))

        # 2. Focused field
        self.mock_win.reset_mock()
        draw_field(self.mock_win, 4, 2, "Iterations", "600", is_focused=True, val_width=10, right_edge=50)

        foc_call = None
        for call_args in self.mock_win.addstr.call_args_list:
            args = call_args[0]
            if len(args) >= 3 and isinstance(args[2], str) and "600" in args[2]:
                foc_call = args

        self.assertIsNotNone(foc_call)
        y, x, text, attr = foc_call[0], foc_call[1], foc_call[2], foc_call[3]
        self.assertEqual(text, " 600 ")
        self.assertNotIn("[", text)
        self.assertNotIn("]", text)
        self.assertEqual(x + len(text) - 1, 50, "Right-edge alignment must be exact")
        self.assertEqual(attr, get_color(COLOR_INPUT_FOCUSED) | curses.A_BOLD)

        # 3. Dropdown field
        self.mock_win.reset_mock()
        draw_field(self.mock_win, 4, 2, "Optimizer", "ADAMW", is_focused=False, has_dropdown=True, right_edge=50)

        drop_call = None
        for call_args in self.mock_win.addstr.call_args_list:
            args = call_args[0]
            if len(args) >= 3 and isinstance(args[2], str) and "ADAMW" in args[2]:
                drop_call = args

        self.assertIsNotNone(drop_call)
        y, x, text, attr = drop_call[0], drop_call[1], drop_call[2], drop_call[3]
        self.assertEqual(text, " ADAMW ▾ ")
        self.assertNotIn("[", text)
        self.assertNotIn("]", text)
        self.assertEqual(x + len(text) - 1, 50)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_top_menu_no_fn_labels_and_bottom_bar_has_all_fn_labels(self, mock_curs, mock_colors, mock_has_colors):
        self.mock_win.getmaxyx.return_value = (30, 100)
        self.mock_win.getch.side_effect = [ord("q")]
        state = CommanderState()
        run_commander_tui(self.mock_win, initial_state=state)

        # Inspect row 0 (top menu)
        row0_calls = [
            c[0][2] for c in self.mock_win.addstr.call_args_list
            if len(c[0]) >= 3 and c[0][0] == 0 and isinstance(c[0][2], str)
        ]
        row0_text = " ".join(row0_calls)
        self.assertNotIn("F1: Help", row0_text)
        self.assertNotIn("F2: Mode", row0_text)
        self.assertNotIn("F9: Theme", row0_text)
        self.assertNotIn("F10: Exit", row0_text)
        self.assertNotIn("(F2)", row0_text)
        self.assertIn("1: Dataset Converter", row0_text)
        self.assertIn("2: Single Run", row0_text)
        self.assertIn("3: Multi-Run Matrix", row0_text)

        # Inspect row 29 (bottom footer, max_y - 1)
        footer_calls = [
            c[0][2] for c in self.mock_win.addstr.call_args_list
            if len(c[0]) >= 3 and c[0][0] == 29 and isinstance(c[0][2], str)
        ]
        footer_text = " ".join(footer_calls)
        self.assertIn("[F1] Help", footer_text)
        self.assertIn("[F2] Mode", footer_text)
        self.assertIn("[F9] Theme", footer_text)
        self.assertIn("[F10] Exit", footer_text)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_footer_contains_all_fn_keys_in_mode2_narrow(self, mock_curs, mock_colors, mock_has_colors):
        self.mock_win.getmaxyx.return_value = (30, 80)
        self.mock_win.getch.side_effect = [ord("q")]
        state = CommanderState()
        state.active_tab = 1
        run_commander_tui(self.mock_win, initial_state=state)

        footer_calls = [
            c[0][2] for c in self.mock_win.addstr.call_args_list
            if len(c[0]) >= 3 and c[0][0] == 29 and isinstance(c[0][2], str)
        ]
        footer_text = " ".join(footer_calls)
        self.assertIn("[F1] Help", footer_text)
        self.assertIn("[F2] Mode", footer_text)
        self.assertIn("[F9] Theme", footer_text)
        self.assertIn("[F10] Exit", footer_text)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_interstitial_hides_convert_dataset_button(self, mock_curs, mock_colors, mock_has_colors):
        """Verify that when navigating to prompt-completion answer and displaying the interstitial,
        Convert Dataset button is not visible anywhere."""
        max_y, max_x = 30, 100
        self.mock_win.getmaxyx.return_value = (max_y, max_x)
        state = CommanderState()
        state.target_format = MLXFormat.PROMPT_COMPLETION
        from mlx_commander.tui.state import ActivePanel
        state.active_panel = ActivePanel.RIGHT
        # In prompt_completion: idx 0..3 are formats, idx 4 is prompt_col, idx 5 is completion_col (Answer)
        state.right_focus_idx = 5

        # Key sequence: Enter on completion_col (opens dialog), then 'q' in dialog, then 'q' in main loop
        self.mock_win.getch.side_effect = [10, ord("q"), ord("q")]

        # Directly verify what is rendered during show_column_picker_dialog
        grid_during_dialog = [[" " for _ in range(max_x)] for _ in range(max_y)]
        def fake_dialog_addstr(y, x, s, attr=0):
            for i, ch in enumerate(s):
                if 0 <= y < max_y and 0 <= x + i < max_x:
                    grid_during_dialog[y][x + i] = ch
        self.mock_win.addstr.side_effect = fake_dialog_addstr

        # Simulate state where dashboard was drawn
        from mlx_commander.tui.app import _draw_mode1_dashboard
        from mlx_commander.tui.widgets import safe_addstr
        left_w = max(34, max_x // 2)
        right_w = max_x - left_w
        _draw_mode1_dashboard(self.mock_win, state, max_y, max_x, left_w, right_w)
        btn_y = 8 + len(state.get_mapping_fields_for_format()) + 3 + 1 + 1
        safe_addstr(self.mock_win, btn_y, left_w + 2, " " * (right_w - 4), 0)

        self.mock_win.getch.side_effect = [ord("q")]
        show_column_picker_dialog(self.mock_win, "Select Column for 'Completion / Answer'", ["id", "output"], allow_none=True)

        full_screen = "\n".join("".join(row) for row in grid_during_dialog)
        self.assertNotIn("Convert Dataset", full_screen)
        # Verify footer row does not contain Convert
        self.assertNotIn("Convert", "".join(grid_during_dialog[max_y - 1]))

    @patch("mlx_commander.tui.app.show_dataset_source_dialog", return_value="/mock/path/data.jsonl")
    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_mode1_dataset_selector_opens_dialog_and_loads_data(self, mock_curs, mock_colors, mock_has_colors, mock_src_dialog):
        """Verify pressing Enter on Mode 1 Dataset field (index 0) invokes show_dataset_source_dialog."""
        state = CommanderState()
        state.active_tab = 0
        from mlx_commander.tui.state import ActivePanel
        state.active_panel = ActivePanel.LEFT
        state.left_focus_idx = 0

        with patch.object(state, "load_dataset", return_value=True) as mock_load:
            self.mock_win.getch.side_effect = [
                10,       # Enter on Dataset field
                ord("q")  # Exit
            ]
            run_commander_tui(self.mock_win, initial_state=state)
            mock_src_dialog.assert_called_once()
            mock_load.assert_called_once_with("/mock/path/data.jsonl")

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_mode1_dashboard_rendering_no_f2_or_buttons(self, mock_curs, mock_colors, mock_has_colors):
        """Verify Mode 1 dashboard renders unified Dataset field without Finder (F2) or Change Path buttons."""
        state = CommanderState()
        state.active_tab = 0
        self.mock_win.getch.side_effect = [ord("q")]
        run_commander_tui(self.mock_win, initial_state=state)

        calls = self.mock_win.addstr.call_args_list
        all_text = " ".join(c[0][2] for c in calls if len(c[0]) >= 3 and isinstance(c[0][2], str))
        self.assertIn("Dataset", all_text)
        self.assertNotIn("Finder (F2)", all_text)
        self.assertNotIn("Change Path", all_text)
        self.assertNotIn("[F2] Open", all_text)

    @patch("mlx_commander.gui_picker.pick_dataset_gui")
    @patch("mlx_commander.gui_picker.pick_model_gui")
    @patch("mlx_commander.gui_picker.pick_folder_gui")
    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_f2_key_switches_mode_and_does_not_open_picker(
        self, mock_curs, mock_colors, mock_has_colors, mock_pick_folder, mock_pick_model, mock_pick_data
    ):
        """Verify pressing F2 key switches between Mode 1 and Mode 2 and does not launch pickers."""
        # Mode 1 -> switches to Mode 2
        state1 = CommanderState()
        state1.active_tab = 0
        self.mock_win.getch.side_effect = [curses.KEY_F2, ord("q")]
        run_commander_tui(self.mock_win, initial_state=state1)
        self.assertEqual(state1.active_tab, 1)
        mock_pick_data.assert_not_called()
        mock_pick_folder.assert_not_called()
        mock_pick_model.assert_not_called()

        # Tab 1 -> switches to Tab 2 (Multi-Run)
        state2 = CommanderState()
        state2.active_tab = 1
        self.mock_win.getch.side_effect = [curses.KEY_F2, ord("q")]
        run_commander_tui(self.mock_win, initial_state=state2)
        self.assertEqual(state2.active_tab, 2)
        mock_pick_data.assert_not_called()
        mock_pick_folder.assert_not_called()
        mock_pick_model.assert_not_called()

        # Tab 2 -> cycles back to Tab 0 (Dataset Converter)
        state3 = CommanderState()
        state3.active_tab = 2
        self.mock_win.getch.side_effect = [curses.KEY_F2, ord("q")]
        run_commander_tui(self.mock_win, initial_state=state3)
        self.assertEqual(state3.active_tab, 0)
        mock_pick_data.assert_not_called()
        mock_pick_folder.assert_not_called()
        mock_pick_model.assert_not_called()

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_mode_switcher_navigation_and_confirm_enter(self, mock_curs, mock_colors, mock_has_colors):
        """
        Verify Mode Switcher behavior:
        - Pressing UP from top of left-hand pane takes user to Mode Switcher.
        - Navigating with LEFT/RIGHT arrows changes selected mode without changing active mode.
        - Pressing DOWN returns to top of left pane without switching mode.
        - Pressing Enter confirms switching mode and focuses top of left pane in new mode.
        """
        state = CommanderState()
        state.active_tab = 0
        state.active_panel = ActivePanel.LEFT
        state.left_focus_idx = 0

        # 1. UP -> into mode switcher
        # 2. RIGHT -> select Mode 2 (unconfirmed)
        # 3. DOWN -> returns to top of left pane without switching
        # 4. UP -> into mode switcher again
        # 5. RIGHT -> select Mode 2
        # 6. Enter (10) -> confirm mode switch to Mode 2!
        # 7. 'q' -> quit
        self.mock_win.getch.side_effect = [
            curses.KEY_UP,
            curses.KEY_RIGHT,
            curses.KEY_DOWN,
            curses.KEY_UP,
            curses.KEY_RIGHT,
            10,
            ord("q"),
        ]
        run_commander_tui(self.mock_win, initial_state=state)

        self.assertEqual(state.active_tab, 1, "Mode 2 should be active after Enter confirmation")
        self.assertFalse(state.mode_switcher_focused, "Mode switcher should no longer be focused after Enter")
        self.assertEqual(state.lora_active_panel, "left")
        self.assertEqual(state.lora_left_focus_idx, 0, "Top of left-hand pane in Mode 2 should be focused")

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_mode_switcher_from_mode2_to_mode1(self, mock_curs, mock_colors, mock_has_colors):
        """Verify pressing UP from top of Mode 2 left pane enters switcher, and Enter switches to Mode 1."""
        state = CommanderState()
        state.active_tab = 1
        state.lora_active_panel = "left"
        state.lora_left_focus_idx = 0

        # 1. UP -> enters mode switcher (index 1)
        # 2. LEFT -> select Mode 1
        # 3. Enter (10) -> confirms mode switch to Mode 1
        # 4. 'q' -> quit
        self.mock_win.getch.side_effect = [
            curses.KEY_UP,
            curses.KEY_LEFT,
            10,
            ord("q"),
        ]
        run_commander_tui(self.mock_win, initial_state=state)

        self.assertEqual(state.active_tab, 0, "Mode 1 should be active after Enter confirmation")
        self.assertFalse(state.mode_switcher_focused, "Mode switcher should not be focused")
        self.assertEqual(state.active_panel, ActivePanel.LEFT)
    @patch("mlx_commander.tui.app.show_choice_dialog", return_value="Prompt / Question")
    def test_left_panel_column_mapping_dialog(self, mock_choice):
        from mlx_commander.loader import LoadedDataset
        from mlx_commander.tui.app import _handle_mode1_input
        from mlx_commander.formats import MLXFormat

        state = CommanderState()
        state.loaded_dataset = LoadedDataset(
            source_path="mock.parquet",
            is_split=False,
            split_names=["default"],
            split_counts={"default": 10},
            columns=["col_x", "col_y"],
            total_rows=10,
            sample_records=[{"col_x": "Hello", "col_y": "World"}],
            base_dir=Path("/tmp"),
            _raw_splits={"default": [{"col_x": "Hello", "col_y": "World"}]},
        )
        state.active_panel = ActivePanel.LEFT
        state.left_focus_idx = 1
        state.selected_column_idx = 0  # col_x

        formats_list = [MLXFormat.PROMPT_COMPLETION, MLXFormat.CHAT, MLXFormat.TEXT, MLXFormat.DPO]
        _handle_mode1_input(self.mock_win, state, 10, formats_list)  # Enter

        mock_choice.assert_called_once()
        self.assertEqual(state.mapping.prompt_col, "col_x")
        self.assertIn("Mapped column 'col_x'", state.status_message)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_default_startup_mode_single_run(self, mock_curs, mock_colors, mock_has_colors):
        state = CommanderState()
        self.assertEqual(state.active_tab, 1, "Default active_tab must be 1 (Single Run)")
        self.assertEqual(state.mode_switcher_idx, 1, "Default mode switcher index must be 1")
        self.assertIn("Configure hyperparameters", state.status_message)

        # Run TUI with default state, press 'q' immediately
        self.mock_win.reset_mock()
        self.mock_win.getmaxyx.return_value = (45, 120)
        self.mock_win.getch.side_effect = [ord("q")]

        run_commander_tui(self.mock_win, initial_state=state)

        calls = self.mock_win.addstr.call_args_list
        all_text = " ".join(c[0][2] for c in calls if len(c[0]) >= 3 and isinstance(c[0][2], str))
        self.assertIn("2: Single Run", all_text)
        self.assertIn("LoRA Hyperparameters", all_text)
        self.assertIn("Runtime Estimates", all_text)
        self.assertIn("Fine-Tuning Queue", all_text)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_mode3_conditions_fully_visible_in_expanded_window(self, mock_curs, mock_colors, mock_has_colors):
        """Mode 3 (Multi-Run Matrix): All 15 hyperparameter conditions are fully visible without scrolling."""
        state = CommanderState()
        state.active_tab = 2
        state.mode_switcher_idx = 2

        self.mock_win.reset_mock()
        self.mock_win.getmaxyx.return_value = (45, 120)
        self.mock_win.getch.side_effect = [ord("q")]

        run_commander_tui(self.mock_win, initial_state=state)

        calls = self.mock_win.addstr.call_args_list
        all_text = " ".join(c[0][2] for c in calls if len(c[0]) >= 3 and isinstance(c[0][2], str))

        # Check that Mode 3 is active
        self.assertIn("3: Multi-Run Matrix", all_text)
        self.assertIn("Step 2: Conditions", all_text)

        # Check conditions in right panel are all rendered
        self.assertIn("Training Iterations", all_text)
        self.assertIn("Batch Size", all_text)
        self.assertIn("Gradient Accumulation Steps", all_text)
        self.assertIn("Learning Rate", all_text)
        self.assertIn("LoRA Rank (r)", all_text)
        self.assertIn("LoRA Alpha (α)", all_text)
        self.assertIn("LoRA Dropout", all_text)
        self.assertIn("Max Seq Length", all_text)
        self.assertIn("Fine-Tuned Layers", all_text)
        self.assertIn("Grad Checkpoint", all_text)
        self.assertIn("Mask Prompt", all_text)
        self.assertIn("Save Every", all_text)
        self.assertIn('Steps per "Eval"', all_text)
        self.assertIn("Run evals on test set", all_text)
        self.assertIn("+ Add Sweep to Queue (F6)", all_text)

        # Scroll offset must be 0 (no items scrolled out of view)
        self.assertEqual(state.multi_right_scroll_offset, 0)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_mode_switcher_up_arrow_wraps_to_bottom_mode0(self, mock_curs, mock_colors, mock_has_colors):
        """Mode 0: Pressing UP from mode switcher jumps to bottom-most element (Convert button)."""
        state = CommanderState()
        state.active_tab = 0
        state.active_panel = ActivePanel.LEFT
        state.left_focus_idx = 0

        # 1. UP -> enters mode switcher
        # 2. UP -> wraps to bottom-most functionality (Convert button in RIGHT panel)
        # 3. 'q' -> quit
        self.mock_win.getch.side_effect = [
            curses.KEY_UP,
            curses.KEY_UP,
            ord("q"),
        ]
        run_commander_tui(self.mock_win, initial_state=state)

        self.assertFalse(state.mode_switcher_focused)
        self.assertEqual(state.active_panel, ActivePanel.RIGHT)
        expected_btn_idx = 10 + len(state.get_mapping_fields_for_format())
        self.assertEqual(state.right_focus_idx, expected_btn_idx)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_mode_switcher_up_arrow_wraps_to_bottom_mode1(self, mock_curs, mock_colors, mock_has_colors):
        """Mode 1: Pressing UP from mode switcher jumps to bottom-most element (Queue panel)."""
        state = CommanderState()
        state.active_tab = 1
        state.lora_active_panel = "left"
        state.lora_left_focus_idx = 0

        # 1. UP -> enters mode switcher
        # 2. UP -> wraps to bottom-most functionality (Queue panel)
        # 3. 'q' -> quit
        self.mock_win.getch.side_effect = [
            curses.KEY_UP,
            curses.KEY_UP,
            ord("q"),
        ]
        run_commander_tui(self.mock_win, initial_state=state)

        self.assertFalse(state.mode_switcher_focused)
        self.assertEqual(state.lora_active_panel, "queue")
        self.assertEqual(state.selected_queue_idx, 0)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_mode_switcher_up_arrow_wraps_to_bottom_mode1_with_runs(self, mock_curs, mock_colors, mock_has_colors):
        """Mode 1: Pressing UP from mode switcher jumps to last run in queue when runs exist."""
        from mlx_commander.lora.queue import QueueManager
        from mlx_commander.lora.config import LoraRunConfig
        state = CommanderState()
        state.active_tab = 1
        state.lora_active_panel = "left"
        state.lora_left_focus_idx = 0
        state.queue_manager = QueueManager(Path("/tmp/mock_queue_test"))
        state.queue_manager.runs = [
            LoraRunConfig(id="r1", name="Run 1", model="m", data="d"),
            LoraRunConfig(id="r2", name="Run 2", model="m", data="d"),
            LoraRunConfig(id="r3", name="Run 3", model="m", data="d"),
        ]

        # 1. UP -> enters mode switcher
        # 2. UP -> wraps to bottom-most functionality (last item in Queue panel: index 2)
        # 3. 'q' -> quit
        self.mock_win.getch.side_effect = [
            curses.KEY_UP,
            curses.KEY_UP,
            ord("q"),
        ]
        run_commander_tui(self.mock_win, initial_state=state)

        self.assertFalse(state.mode_switcher_focused)
        self.assertEqual(state.lora_active_panel, "queue")
        self.assertEqual(state.selected_queue_idx, 2)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=True)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_mode_switcher_up_arrow_wraps_to_bottom_mode2(self, mock_curs, mock_colors, mock_has_colors):
        """Mode 2: Pressing UP (or 'k') from mode switcher jumps to bottom-most element (Sweep panel)."""
        state = CommanderState()
        state.active_tab = 2
        state.multi_active_panel = "left"
        state.multi_left_focus_idx = 0

        # 1. UP -> enters mode switcher
        # 2. ord('k') -> wraps to bottom-most functionality (Sweep panel)
        # 3. 'q' -> quit
        self.mock_win.getch.side_effect = [
            curses.KEY_UP,
            ord("k"),
            ord("q"),
        ]
        run_commander_tui(self.mock_win, initial_state=state)

        self.assertFalse(state.mode_switcher_focused)
        self.assertEqual(state.multi_active_panel, "sweep")


if __name__ == "__main__":
    unittest.main()




