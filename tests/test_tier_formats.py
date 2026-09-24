"""
Unit tests for Tier 1 and Tier 2 dataset formats, Unicode text loading,
and missing dependency interstitial error handling.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlx_commander.exceptions import MissingDependencyError
from mlx_commander.formats import MLXFormat
from mlx_commander.loader import (
    is_lance_dir,
    is_parquet_dir,
    is_parquet_file,
    load_from_arrow_file,
    load_from_duckdb,
    load_from_lance,
    load_from_parquet_dir,
    load_from_parquet_file,
    load_from_text,
    load_local_dataset,
)
from mlx_commander.tui.state import CommanderState
from mlx_commander.tui.widgets import show_missing_dependency_dialog


class TestTierFormats(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # --- Unicode Text Ingestion Tests ---

    def test_load_utf8_plain_text(self):
        file_path = Path(self.temp_dir) / "sample.txt"
        lines = [
            "First line of pre-training data.",
            "Second line with some punctuation and numbers: 12345.",
            "",  # empty line should be ignored
            "   ",  # whitespace line should be ignored
            "Third line of text content.",
        ]
        file_path.write_text("\n".join(lines), encoding="utf-8")

        records = load_from_text(file_path)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0], {"text": "First line of pre-training data."})
        self.assertEqual(records[1], {"text": "Second line with some punctuation and numbers: 12345."})
        self.assertEqual(records[2], {"text": "Third line of text content."})

    def test_load_utf8_bom_text(self):
        file_path = Path(self.temp_dir) / "bom_sample.txt"
        content = "Line with BOM prefix.\nSecond line after BOM."
        # Write with utf-8-sig which writes \xef\xbb\xbf BOM header
        file_path.write_bytes(content.encode("utf-8-sig"))

        records = load_from_text(file_path)
        self.assertEqual(len(records), 2)
        # Verify BOM character \ufeff does not contaminate text
        self.assertEqual(records[0]["text"], "Line with BOM prefix.")
        self.assertFalse(records[0]["text"].startswith("\ufeff"))
        self.assertEqual(records[1]["text"], "Second line after BOM.")

    def test_load_utf16_bom_text(self):
        file_path = Path(self.temp_dir) / "utf16_sample.txt"
        content = "UTF-16 line one.\nUTF-16 line two."
        file_path.write_bytes(content.encode("utf-16"))

        records = load_from_text(file_path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["text"], "UTF-16 line one.")
        self.assertEqual(records[1]["text"], "UTF-16 line two.")

    def test_load_multilingual_unicode_text(self):
        file_path = Path(self.temp_dir) / "multilingual.txt"
        lines = [
            "English text: The quick brown fox jumps over the lazy dog.",
            "Spanish text: El veloz murciélago hindú comía feliz cardillo y kiwi.",
            "French text: Voix ambiguë d'un cœur qui au zéphyr préfère les jattes de kiwis.",
            "German text: Victor jagt zwölf Boxkämpfer quer über den großen Sylter Deich.",
            "Japanese text: 本日は晴天なり。機械学習のデータセット。",
            "Cyrillic text: Съешь же ещё этих мягких французских булок, да выпей чаю.",
            "Greek text: Ξεσκεπάζω την ψυχοφθόρα βδελυγμία.",
            "Math symbols: ∀x ∈ ℝ, e^(iπ) + 1 = 0 ∧ ∑_{i=1}^n i = n(n+1)/2.",
        ]
        file_path.write_text("\n".join(lines), encoding="utf-8")

        records = load_from_text(file_path)
        self.assertEqual(len(records), len(lines))
        self.assertIn("murciélago", records[1]["text"])
        self.assertIn("機械学習", records[4]["text"])
        self.assertIn("∑_{i=1}^n", records[7]["text"])

    def test_load_text_via_load_local_dataset(self):
        file_path = Path(self.temp_dir) / "data.txt"
        file_path.write_text("Sentence 1\nSentence 2\nSentence 3\n", encoding="utf-8")

        ds = load_local_dataset(file_path)
        self.assertEqual(ds.total_rows, 3)
        self.assertEqual(ds.columns, ["text"])
        records = ds.get_all_records()
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0], {"text": "Sentence 1"})

    def test_load_text_directory_with_splits(self):
        dir_path = Path(self.temp_dir) / "text_dataset"
        dir_path.mkdir()
        (dir_path / "train.txt").write_text("Train line 1\nTrain line 2\n", encoding="utf-8")
        (dir_path / "valid.txt").write_text("Valid line 1\n", encoding="utf-8")
        (dir_path / "test.txt").write_text("Test line 1\n", encoding="utf-8")

        ds = load_local_dataset(dir_path)
        self.assertTrue(ds.is_split)
        self.assertEqual(ds.total_rows, 4)
        self.assertEqual(ds.split_counts.get("train"), 2)
        self.assertEqual(ds.split_counts.get("valid"), 1)
        self.assertEqual(ds.split_counts.get("test"), 1)

    def test_commander_state_auto_selects_text_format(self):
        file_path = Path(self.temp_dir) / "corpus.txt"
        file_path.write_text("Line 1\nLine 2\nLine 3\n", encoding="utf-8")

        state = CommanderState()
        self.assertEqual(state.target_format, MLXFormat.PROMPT_COMPLETION)

        success = state.load_dataset(str(file_path))
        self.assertTrue(success)
        self.assertEqual(state.target_format, MLXFormat.TEXT)
        self.assertEqual(state.mapping.text_col, "text")

    def test_load_latin1_text_no_chinese_characters(self):
        """
        Verify that Latin-1 / ISO-8859-1 encoded text with European/Finnish characters
        (such as Seppälä, Åland, etc.) does NOT get misdecoded as UTF-16 CJK ideographs
        (Chinese characters), and that delimited lines like '@positive' are correctly parsed.
        """
        file_path = Path(self.temp_dir) / "financial_phrasebank.txt"
        lines = [
            "Clothing retail chain Seppälä 's sales increased by 8 % to EUR 155.2 mn .@positive",
            "Finnish Bank of Åland reports its operating profit increased .@positive",
            "According to Gran , the company has no plans to move all production to Russia .@neutral",
            "Sales in Finland decreased by 10.5 % in January .@negative",
        ]
        content_latin1 = "\n".join(lines).encode("iso-8859-1")
        file_path.write_bytes(content_latin1)

        records = load_from_text(file_path)
        self.assertEqual(len(records), 4)

        # Verify no Chinese CJK characters anywhere in decoded records
        for r in records:
            for val in r.values():
                self.assertFalse(any(0x4E00 <= ord(c) <= 0x9FFF for c in str(val)))

        # Verify delimited record structure
        self.assertIn("prompt", records[0])
        self.assertIn("completion", records[0])
        self.assertIn("text", records[0])
        self.assertEqual(records[0]["completion"], "positive")
        self.assertEqual(records[2]["completion"], "neutral")
        self.assertEqual(records[3]["completion"], "negative")
        self.assertIn("Seppälä", records[0]["prompt"])
        self.assertIn("Åland", records[1]["prompt"])

    def test_load_cp1252_smart_quotes_and_euro(self):
        """Verify Windows-1252 specific characters (euro €, smart quotes “”, dashes –) decode cleanly."""
        file_path = Path(self.temp_dir) / "cp1252_sample.txt"
        text = '“Net sales rose to €150m – beating expectations!”@positive\n“Operating profit fell.”@negative'
        file_path.write_bytes(text.encode("cp1252"))

        records = load_from_text(file_path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["completion"], "positive")
        self.assertIn("€150m", records[0]["prompt"])
        self.assertTrue(records[0]["prompt"].startswith("“"))

    def test_load_utf16_without_bom(self):
        """Verify UTF-16 without BOM (detected via high null-byte ratio) decodes properly."""
        file_path = Path(self.temp_dir) / "utf16_no_bom.txt"
        text = "First UTF-16 line without BOM.\nSecond line content."
        file_path.write_bytes(text.encode("utf-16-le"))

        records = load_from_text(file_path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["text"], "First UTF-16 line without BOM.")
        self.assertEqual(records[1]["text"], "Second line content.")

    def test_load_tab_delimited_text_auto_detection(self):
        """Verify text files with tab delimiters auto-detect prompt and completion."""
        file_path = Path(self.temp_dir) / "tab_pairs.txt"
        lines = [
            "What is the capital of France?\tParis",
            "What is the capital of Germany?\tBerlin",
            "What is the capital of Italy?\tRome",
        ]
        file_path.write_text("\n".join(lines), encoding="utf-8")

        records = load_from_text(file_path)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["prompt"], "What is the capital of France?")
        self.assertEqual(records[0]["completion"], "Paris")

    # --- Missing Dependency Interstitial Tests ---

    def test_parquet_file_missing_pyarrow_raises_interstitial_error(self):
        file_path = Path(self.temp_dir) / "sample.parquet"
        file_path.write_bytes(b"PAR1mock")

        with patch("mlx_commander.loader.HAS_PYARROW", False):
            with self.assertRaises(MissingDependencyError) as ctx:
                load_from_parquet_file(file_path)

            exc = ctx.exception
            self.assertEqual(exc.format_name, "Parquet")
            self.assertEqual(exc.package_name, "pyarrow")
            self.assertEqual(exc.install_command, "pip install pyarrow")
            self.assertEqual(exc.pip_extra, "pip install 'mlx_commander[parquet]'")

    def test_parquet_dir_missing_pyarrow_raises_interstitial_error(self):
        dir_path = Path(self.temp_dir) / "parquet_shards"
        dir_path.mkdir()
        (dir_path / "train-00000.parquet").write_bytes(b"PAR1mock")

        with patch("mlx_commander.loader.HAS_PYARROW", False):
            with self.assertRaises(MissingDependencyError) as ctx:
                load_from_parquet_dir(dir_path)

            exc = ctx.exception
            self.assertEqual(exc.format_name, "Parquet")
            self.assertEqual(exc.package_name, "pyarrow")

    def test_arrow_file_missing_pyarrow_raises_interstitial_error(self):
        file_path = Path(self.temp_dir) / "sample.arrow"
        file_path.write_bytes(b"ARROW1mock")

        with patch("mlx_commander.loader.HAS_PYARROW", False):
            with self.assertRaises(MissingDependencyError) as ctx:
                load_from_arrow_file(file_path)

            exc = ctx.exception
            self.assertEqual(exc.format_name, "Apache Arrow")
            self.assertEqual(exc.package_name, "pyarrow")
            self.assertEqual(exc.install_command, "pip install pyarrow")

    def test_duckdb_missing_duckdb_raises_interstitial_error(self):
        file_path = Path(self.temp_dir) / "database.duckdb"
        file_path.write_bytes(b"DUCKDBmock")

        with patch("mlx_commander.loader.HAS_DUCKDB", False):
            with self.assertRaises(MissingDependencyError) as ctx:
                load_from_duckdb(file_path)

            exc = ctx.exception
            self.assertEqual(exc.format_name, "DuckDB")
            self.assertEqual(exc.package_name, "duckdb")
            self.assertEqual(exc.install_command, "pip install duckdb")
            self.assertEqual(exc.pip_extra, "pip install 'mlx_commander[duckdb]'")

    def test_lance_missing_pylance_raises_interstitial_error(self):
        dir_path = Path(self.temp_dir) / "dataset.lance"
        dir_path.mkdir()

        with patch("mlx_commander.loader.HAS_LANCE", False):
            with self.assertRaises(MissingDependencyError) as ctx:
                load_from_lance(dir_path)

            exc = ctx.exception
            self.assertEqual(exc.format_name, "Lance")
            self.assertEqual(exc.package_name, "pylance")
            self.assertEqual(exc.install_command, "pip install pylance")
            self.assertEqual(exc.pip_extra, "pip install 'mlx_commander[lance]'")

    def test_is_lance_dir_detection(self):
        lance_dir = Path(self.temp_dir) / "test_data.lance"
        lance_dir.mkdir()
        self.assertTrue(is_lance_dir(lance_dir))

        version_dir = Path(self.temp_dir) / "custom_dir"
        version_dir.mkdir()
        (version_dir / "_versions").mkdir()
        self.assertTrue(is_lance_dir(version_dir))

        normal_dir = Path(self.temp_dir) / "normal_folder"
        normal_dir.mkdir()
        self.assertFalse(is_lance_dir(normal_dir))

    def test_state_load_dataset_captures_missing_dependency(self):
        file_path = Path(self.temp_dir) / "records.parquet"
        file_path.write_bytes(b"PAR1")

        state = CommanderState()
        with patch("mlx_commander.loader.HAS_PYARROW", False):
            success = state.load_dataset(str(file_path))
            self.assertFalse(success)
            self.assertIsNotNone(state.last_missing_dependency)
            self.assertEqual(state.last_missing_dependency.package_name, "pyarrow")
            self.assertIn("Missing dependency: pyarrow", state.status_message)
            self.assertTrue(state.status_is_error)

    # --- TUI Modal Rendering Tests ---

    def test_show_missing_dependency_dialog(self):
        exc = MissingDependencyError(
            format_name="Parquet",
            package_name="pyarrow",
            install_command="pip install pyarrow",
            description="Parquet requires pyarrow.",
            extra_name="parquet",
        )

        # Mock curses stdscr
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (30, 100)
        # Return Enter key (10) on first getch to exit the loop
        mock_stdscr.getch.return_value = 10

        show_missing_dependency_dialog(mock_stdscr, exc)

        # Verify safe_addstr calls inside dialog
        addstr_calls = [c[0] for c in mock_stdscr.addstr.call_args_list]
        all_printed = " ".join(str(c[2]) for c in addstr_calls if len(c) > 2)

        self.assertIn("Dependency Required: Parquet", all_printed)
        self.assertIn("pip install pyarrow", all_printed)
        self.assertTrue("[Enter] OK / Dismiss" in all_printed or "[I] Install with pip" in all_printed)

    def test_parquet_sniffing_and_alternative_extensions(self):
        parquet_header = b"PAR1\x15\x00\x15\x92\x00\x00\x00PAR1"
        for fname in ("data.pq", "data.parq", "data.parquet.snappy", "data_no_ext"):
            fp = Path(self.temp_dir) / fname
            fp.write_bytes(parquet_header)
            self.assertTrue(is_parquet_file(fp), f"Failed is_parquet_file for {fname}")

            state = CommanderState()
            with patch("mlx_commander.loader.HAS_PYARROW", False):
                success = state.load_dataset(str(fp))
                self.assertFalse(success)
                self.assertIsNotNone(state.last_missing_dependency)
                self.assertEqual(state.last_missing_dependency.package_name, "pyarrow")
                self.assertNotIn("utf-8", state.status_message.lower())
                self.assertIn("Missing dependency: pyarrow", state.status_message)

    def test_parquet_directory_detection_with_subfolders(self):
        dir_path = Path(self.temp_dir) / "hf_style_repo"
        sub_data = dir_path / "data"
        sub_data.mkdir(parents=True)
        (sub_data / "train-00000-of-00001.parquet").write_bytes(b"PAR1\x15\x00\x15\x92\x00\x00PAR1")

        self.assertTrue(is_parquet_dir(dir_path))
        state = CommanderState()
        with patch("mlx_commander.loader.HAS_PYARROW", False):
            success = state.load_dataset(str(dir_path))
            self.assertFalse(success)
            self.assertIsNotNone(state.last_missing_dependency)
            self.assertEqual(state.last_missing_dependency.package_name, "pyarrow")
            self.assertNotIn("utf-8", state.status_message.lower())

    def test_dynamic_pyarrow_resolution_when_installed(self):
        from mlx_commander.loader import ensure_pyarrow
        res = ensure_pyarrow()
        self.assertIsInstance(res, bool)


if __name__ == "__main__":
    unittest.main()
