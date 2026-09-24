import json
import tempfile
import unittest
from pathlib import Path

from mlx_commander.formats import MLXFormat
from mlx_commander.tui.state import ActivePanel, CommanderState


class TestCommanderState(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.src_file = Path(self.tmp_dir.name) / "test_data.jsonl"
        data = [
            {"id": i, "question": f"Question {i}", "answer": f"Answer {i}", "extra": "info"}
            for i in range(20)
        ]
        with open(self.src_file, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_initial_defaults(self):
        state = CommanderState()
        self.assertEqual(state.target_format, MLXFormat.PROMPT_COMPLETION)
        self.assertEqual(state.train_pct, 80.0)
        self.assertEqual(state.valid_pct, 10.0)
        self.assertEqual(state.test_pct, 10.0)
        self.assertEqual(state.active_panel, ActivePanel.LEFT)

    def test_load_dataset_and_auto_mapping(self):
        state = CommanderState()
        success = state.load_dataset(str(self.src_file))
        self.assertTrue(success)
        self.assertIsNotNone(state.loaded_dataset)
        self.assertEqual(state.loaded_dataset.total_rows, 20)
        self.assertIn("question", state.loaded_dataset.columns)
        self.assertIn("answer", state.loaded_dataset.columns)

        # Auto detection for prompt completion
        self.assertEqual(state.mapping.prompt_col, "question")
        self.assertEqual(state.mapping.completion_col, "answer")

        # Preview should have generated live
        self.assertTrue(len(state.preview_cache) > 0)
        rec0 = json.loads(state.preview_cache[0])
        self.assertEqual(rec0["prompt"], "Question 0")
        self.assertEqual(rec0["completion"], "Answer 0")

    def test_reactive_preview_on_format_change(self):
        state = CommanderState()
        state.load_dataset(str(self.src_file))

        # Change format to TEXT
        state.set_format(MLXFormat.TEXT)
        self.assertEqual(state.target_format, MLXFormat.TEXT)
        # Check mapping and preview
        state.set_mapping_field("text_col", "question")
        self.assertTrue(len(state.preview_cache) > 0)
        rec = json.loads(state.preview_cache[0])
        self.assertEqual(rec["text"], "Question 0")

    def test_split_counts_calculation(self):
        state = CommanderState()
        state.load_dataset(str(self.src_file))
        counts = state.get_split_counts()
        self.assertEqual(counts["train"] + counts["valid"] + counts["test"], 20)

    def test_randomize_seed(self):
        state = CommanderState()
        old_seed = state.seed
        # Calling randomize should set a valid 6-digit seed
        state.randomize_seed()
        self.assertGreaterEqual(state.seed, 100000)
        self.assertLessEqual(state.seed, 999999)


    def test_concatenated_preview_in_state(self):
        state = CommanderState()
        state.load_dataset(str(self.src_file))
        # Concatenate question and extra
        state.set_mapping_field("prompt_col", "question + extra")
        self.assertIsNone(state.preview_error)
        self.assertTrue(len(state.preview_cache) > 0)
        rec0 = json.loads(state.preview_cache[0])
        self.assertEqual(rec0["prompt"], "Question 0\n\ninfo")

    def test_theme_mode_toggle(self):
        from mlx_commander.tui.state import ThemeMode
        state = CommanderState()
        self.assertEqual(state.theme_mode, ThemeMode.MODERN)
        new_theme = state.toggle_theme()
        self.assertEqual(new_theme, ThemeMode.CLASSIC_BLUE)
        self.assertEqual(state.theme_mode, ThemeMode.CLASSIC_BLUE)
        toggled_back = state.toggle_theme()
        self.assertEqual(toggled_back, ThemeMode.MODERN)
        self.assertEqual(state.theme_mode, ThemeMode.MODERN)

    def test_prefill_theme(self):
        from mlx_commander.tui.state import ThemeMode
        state = CommanderState()
        state.apply_prefill({"theme": "classic_blue"})
        self.assertEqual(state.theme_mode, ThemeMode.CLASSIC_BLUE)
        state.apply_prefill({"theme_mode": "modern"})
        self.assertEqual(state.theme_mode, ThemeMode.MODERN)

    def test_state_auto_adapts_format_for_chat_and_text(self):
        # 1. Chat dataset with messages
        chat_file = Path(self.tmp_dir.name) / "chat.jsonl"
        with open(chat_file, "w") as f:
            f.write(json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}) + "\n")

        state = CommanderState()
        self.assertEqual(state.target_format, MLXFormat.PROMPT_COMPLETION)
        state.load_dataset(str(chat_file))
        self.assertEqual(state.target_format, MLXFormat.CHAT)
        self.assertEqual(state.mapping.messages_col, "messages")
        self.assertIsNone(state.preview_error)

        # 2. Text dataset with text + id
        text_file = Path(self.tmp_dir.name) / "text.jsonl"
        with open(text_file, "w") as f:
            f.write(json.dumps({"id": 1, "text": "Pre-training corpus content."}) + "\n")

        state2 = CommanderState()
        state2.load_dataset(str(text_file))
        self.assertEqual(state2.target_format, MLXFormat.TEXT)
        self.assertEqual(state2.mapping.text_col, "text")
        self.assertIsNone(state2.preview_error)

        # 3. Math dataset with problem / solution
        math_file = Path(self.tmp_dir.name) / "math.jsonl"
        with open(math_file, "w") as f:
            f.write(json.dumps({"problem": "What is 2+2?", "solution": "4"}) + "\n")

        state3 = CommanderState()
        state3.load_dataset(str(math_file))
        self.assertEqual(state3.target_format, MLXFormat.PROMPT_COMPLETION)
        self.assertEqual(state3.mapping.prompt_col, "problem")
        self.assertEqual(state3.mapping.completion_col, "solution")
        self.assertIsNone(state3.preview_error)

    def test_map_column_to_field(self):
        state = CommanderState()
        state.load_dataset(str(self.src_file))
        state.map_column_to_field("extra", "prompt_col")
        self.assertEqual(state.mapping.prompt_col, "extra")
        self.assertIsNone(state.preview_error)


if __name__ == "__main__":
    unittest.main()


