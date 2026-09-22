"""
Unit tests for generalized dataset transformations:
- Default System Prompt injection across all formats
- Label Mapping / Semantic Rewriting (e.g. 0/1/2 -> negative/neutral/positive)
- Discrete label detection and HF ClassLabel auto-detection
- Single canonical test.jsonl with optional auxiliary test_gold.jsonl
- CLI flags (--system-prompt, --label-map, --test-gold)
- TUI state prefill and auto-population
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlx_commander.cli import build_parser, parse_mapping_arg, run_direct_conversion
from mlx_commander.converter import convert_and_save
from mlx_commander.formats import (
    ColumnMapping,
    MLXFormat,
    format_record,
    parse_label_map,
)
from mlx_commander.loader import LoadedDataset, detect_column_discrete_labels
from mlx_commander.tui.state import CommanderState
from mlx_commander.tui.widgets import show_label_mapping_dialog


class TestDatasetTransformations(unittest.TestCase):
    def test_parse_label_map_dict(self):
        d = {"0": "negative", 1: "neutral", "2": "positive"}
        res = parse_label_map(d)
        self.assertEqual(res, {"0": "negative", "1": "neutral", "2": "positive"})

    def test_parse_label_map_json_str(self):
        s = '{"0": "negative", "1": "positive"}'
        res = parse_label_map(s)
        self.assertEqual(res, {"0": "negative", "1": "positive"})

    def test_parse_label_map_colon_delimited(self):
        s = "0:negative, 1:neutral, 2:positive"
        res = parse_label_map(s)
        self.assertEqual(res, {"0": "negative", "1": "neutral", "2": "positive"})

    def test_parse_label_map_equals_delimited(self):
        s = "0=negative, 1=neutral, 2=positive"
        res = parse_label_map(s)
        self.assertEqual(res, {"0": "negative", "1": "neutral", "2": "positive"})

    def test_parse_label_map_empty_or_none(self):
        self.assertIsNone(parse_label_map(None))
        self.assertIsNone(parse_label_map(""))
        self.assertIsNone(parse_label_map("   "))

    def test_default_system_prompt_chat_injection(self):
        mapping = ColumnMapping(
            user_col="question",
            assistant_col="answer",
            default_system_prompt="You are a helpful financial assistant.",
        )
        record = {"question": "What is EBITDA?", "answer": "Earnings before interest, taxes, depreciation, amortization."}
        formatted = format_record(record, MLXFormat.CHAT, mapping)
        msgs = formatted["messages"]
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "You are a helpful financial assistant.")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[2]["role"], "assistant")

    def test_default_system_prompt_chat_preserved(self):
        # When record already has a system message, default_system_prompt must not duplicate
        mapping = ColumnMapping(
            system_col="sys",
            user_col="question",
            assistant_col="answer",
            default_system_prompt="Default system prompt.",
        )
        record = {
            "sys": "Custom system prompt.",
            "question": "What is EBITDA?",
            "answer": "Earnings before interest...",
        }
        formatted = format_record(record, MLXFormat.CHAT, mapping)
        msgs = formatted["messages"]
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "Custom system prompt.")

    def test_default_system_prompt_prompt_completion(self):
        mapping = ColumnMapping(
            prompt_col="prompt",
            completion_col="completion",
            default_system_prompt="Classify sentiment as positive, neutral, or negative.",
        )
        record = {"prompt": "Sales rose 12%.", "completion": "positive"}
        formatted = format_record(record, MLXFormat.PROMPT_COMPLETION, mapping)
        expected_prompt = "Classify sentiment as positive, neutral, or negative.\n\nSales rose 12%."
        self.assertEqual(formatted["prompt"], expected_prompt)
        self.assertEqual(formatted["completion"], "positive")

    def test_default_system_prompt_text(self):
        mapping = ColumnMapping(
            text_col="raw",
            default_system_prompt="Pre-training prefix.",
        )
        record = {"raw": "Once upon a time..."}
        formatted = format_record(record, MLXFormat.TEXT, mapping)
        self.assertEqual(formatted["text"], "Pre-training prefix.\n\nOnce upon a time...")

    def test_label_map_prompt_completion(self):
        mapping = ColumnMapping(
            prompt_col="sentence",
            completion_col="label",
            label_map={"0": "negative", "1": "neutral", "2": "positive"},
        )
        record0 = {"sentence": "Profits tumbled.", "label": 0}
        record1 = {"sentence": "Results met expectations.", "label": "1"}
        record2 = {"sentence": "Revenue hit record high.", "label": 2}

        f0 = format_record(record0, MLXFormat.PROMPT_COMPLETION, mapping)
        f1 = format_record(record1, MLXFormat.PROMPT_COMPLETION, mapping)
        f2 = format_record(record2, MLXFormat.PROMPT_COMPLETION, mapping)

        self.assertEqual(f0["completion"], "negative")
        self.assertEqual(f1["completion"], "neutral")
        self.assertEqual(f2["completion"], "positive")

    def test_label_map_chat_multi_column(self):
        mapping = ColumnMapping(
            user_col="text",
            assistant_col="target",
            label_map={"0": "negative", "1": "neutral", "2": "positive"},
        )
        record = {"text": "Operating profit was down.", "target": 0}
        f = format_record(record, MLXFormat.CHAT, mapping)
        self.assertEqual(f["messages"][-1]["role"], "assistant")
        self.assertEqual(f["messages"][-1]["content"], "negative")

    def test_label_map_chat_messages_column(self):
        mapping = ColumnMapping(
            messages_col="conversation",
            label_map={"0": "negative", "1": "neutral", "2": "positive"},
        )
        record = {
            "conversation": [
                {"role": "user", "content": "How is the market today?"},
                {"role": "assistant", "content": "2"},
            ]
        }
        f = format_record(record, MLXFormat.CHAT, mapping)
        self.assertEqual(f["messages"][-1]["content"], "positive")

    def test_detect_column_discrete_labels_with_hf_classlabel(self):
        class MockClassLabel:
            names = ["negative", "neutral", "positive"]

        ds = LoadedDataset(
            source_path="mock_dataset",
            is_split=False,
            split_names=["default"],
            split_counts={"default": 10},
            columns=["text", "label"],
            total_rows=10,
            sample_records=[{"text": "Sample", "label": 0}],
            features={"label": MockClassLabel()},
            _raw_splits={"default": [{"text": "Sample", "label": 0}]},
        )

        disc = detect_column_discrete_labels(ds, "label")
        self.assertIsNotNone(disc)
        self.assertTrue(disc["discrete"])
        self.assertEqual(disc["unique_values"], ["0", "1", "2"])
        self.assertEqual(disc["hf_class_names"], ["negative", "neutral", "positive"])
        self.assertEqual(disc["suggested_map"], {"0": "negative", "1": "neutral", "2": "positive"})

    def test_detect_column_discrete_labels_from_sampled_records(self):
        records = [
            {"text": "A", "label": 0},
            {"text": "B", "label": 1},
            {"text": "C", "label": 2},
            {"text": "D", "label": 0},
            {"text": "E", "label": 1},
        ]
        ds = LoadedDataset(
            source_path="mock_dataset",
            is_split=False,
            split_names=["default"],
            split_counts={"default": len(records)},
            columns=["text", "label"],
            total_rows=len(records),
            sample_records=records[:3],
            features={},
            _raw_splits={"default": records},
        )

        disc = detect_column_discrete_labels(ds, "label")
        self.assertIsNotNone(disc)
        self.assertTrue(disc["discrete"])
        self.assertEqual(disc["unique_values"], ["0", "1", "2"])
        self.assertIsNone(disc["hf_class_names"])
        self.assertTrue(disc["is_integer_labels"])

    def test_detect_column_discrete_labels_continuous_text_returns_none(self):
        records = [{"text": f"Sentence number {i}", "label": f"Answer {i}"} for i in range(25)]
        ds = LoadedDataset(
            source_path="mock_dataset",
            is_split=False,
            split_names=["default"],
            split_counts={"default": len(records)},
            columns=["text", "label"],
            total_rows=len(records),
            sample_records=records[:5],
            features={},
            _raw_splits={"default": records},
        )

        disc = detect_column_discrete_labels(ds, "label")
        self.assertIsNone(disc)

    def test_parse_mapping_arg_with_system_prompt_and_label_map(self):
        s = "prompt=question,completion=target,system_prompt=You are a bot,label_map=0:bad,1:good"
        mapping = parse_mapping_arg(s)
        self.assertEqual(mapping.prompt_col, "question")
        self.assertEqual(mapping.completion_col, "target")
        self.assertEqual(mapping.default_system_prompt, "You are a bot")
        self.assertEqual(mapping.label_map, {"0": "bad", "1": "good"})

    def test_convert_and_save_with_export_test_gold(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            records = [
                {"sentence": f"Statement {i}", "sentiment": i % 3}
                for i in range(30)
            ]
            ds = LoadedDataset(
                source_path="mock_ds",
                is_split=False,
                split_names=["default"],
                split_counts={"default": len(records)},
                columns=["sentence", "sentiment"],
                total_rows=len(records),
                sample_records=records[:5],
                _raw_splits={"default": records},
            )

            mapping = ColumnMapping(
                user_col="sentence",
                assistant_col="sentiment",
                default_system_prompt="Classifier bot.",
                label_map={"0": "negative", "1": "neutral", "2": "positive"},
            )

            out_dir = tmp_path / "output"
            res = convert_and_save(
                dataset=ds,
                format_type=MLXFormat.CHAT,
                mapping=mapping,
                output_dir=out_dir,
                export_test_gold=True,
            )

            # Check canonical test.jsonl exists
            test_file = out_dir / "test.jsonl"
            self.assertTrue(test_file.exists())
            with open(test_file, "r") as tf:
                line = json.loads(tf.readline())
                self.assertIn("messages", line)
                self.assertEqual(line["messages"][0]["role"], "system")
                self.assertEqual(line["messages"][0]["content"], "Classifier bot.")
                self.assertIn(line["messages"][-1]["content"], ("negative", "neutral", "positive"))

            # Check auxiliary test_gold.jsonl exists
            gold_file = out_dir / "test_gold.jsonl"
            self.assertTrue(gold_file.exists())
            with open(gold_file, "r") as gf:
                gline = json.loads(gf.readline())
                self.assertIn("prompt", gline)
                self.assertIn("expected", gline)
                self.assertIn(gline["expected"], ("negative", "neutral", "positive"))

    def test_cli_direct_conversion_flags(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_file = tmp_path / "raw.jsonl"
            with open(data_file, "w") as f:
                for i in range(20):
                    f.write(json.dumps({"q": f"Question {i}", "a": i % 2}) + "\n")

            out_dir = tmp_path / "converted"
            parser = build_parser()
            args = parser.parse_args([
                "-d", str(data_file),
                "-f", "prompt_completion",
                "-o", str(out_dir),
                "--prompt-col", "q",
                "--completion-col", "a",
                "--system-prompt", "You are an assistant.",
                "--label-map", "0:no, 1:yes",
                "--test-gold",
            ])

            res = run_direct_conversion(args)
            self.assertTrue((out_dir / "train.jsonl").exists())
            self.assertTrue((out_dir / "test.jsonl").exists())
            self.assertTrue((out_dir / "test_gold.jsonl").exists())

            with open(out_dir / "train.jsonl", "r") as tr:
                first = json.loads(tr.readline())
                self.assertTrue(first["prompt"].startswith("You are an assistant."))
                self.assertIn(first["completion"], ("no", "yes"))

    def test_state_load_and_prefill_with_transformations(self):
        state = CommanderState()
        state.apply_prefill({
            "system_prompt": "Global instructions.",
            "label_map": "0:bad, 1:good",
        })
        self.assertEqual(state.mapping.default_system_prompt, "Global instructions.")
        self.assertEqual(state.mapping.label_map, {"0": "bad", "1": "good"})

        # Check mapping fields in right panel include System Prompt and Label Mapping
        fields = state.get_mapping_fields_for_format()
        field_keys = [f["key"] for f in fields]
        self.assertIn("default_system_prompt", field_keys)
        self.assertIn("label_map", field_keys)

    def test_show_label_mapping_dialog_save_hotkey(self):
        mock_win = MagicMock()
        mock_win.getmaxyx.return_value = (30, 80)
        # Press 's' (save) immediately
        mock_win.getch.side_effect = [ord("s")]

        disc_info = {
            "discrete": True,
            "unique_values": ["0", "1", "2"],
            "suggested_map": {"0": "negative", "1": "neutral", "2": "positive"},
        }
        res = show_label_mapping_dialog(mock_win, "sentiment", disc_info, None)
        self.assertEqual(res, {"0": "negative", "1": "neutral", "2": "positive"})

    def test_show_label_mapping_dialog_clear_hotkey(self):
        mock_win = MagicMock()
        mock_win.getmaxyx.return_value = (30, 80)
        # Press 'c' (clear) immediately
        mock_win.getch.side_effect = [ord("c")]

        current = {"0": "negative", "1": "positive"}
        res = show_label_mapping_dialog(mock_win, "sentiment", None, current)
        self.assertIsNone(res)

    def test_show_label_mapping_dialog_cancel_hotkey(self):
        mock_win = MagicMock()
        mock_win.getmaxyx.return_value = (30, 80)
        # Press 27 (Esc) immediately
        mock_win.getch.side_effect = [27]

        current = {"0": "negative", "1": "positive"}
        res = show_label_mapping_dialog(mock_win, "sentiment", None, current)
        self.assertEqual(res, current)


if __name__ == "__main__":
    unittest.main()
