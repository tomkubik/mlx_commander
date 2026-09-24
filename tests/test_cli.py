"""
Unit tests for CLI parsing, arguments, and direct execution.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from mlx_commander.cli import build_parser, main, parse_mapping_arg
from tests.conftest import make_sample_qa_records


class TestCLI(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.src_file = Path(self.temp_dir) / "source.jsonl"
        self.out_dir = Path(self.temp_dir) / "mlx_cli_out"

        records = make_sample_qa_records(30)
        with open(self.src_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parse_mapping_arg_key_val(self):
        mapping = parse_mapping_arg("prompt=instruction,completion=output")
        self.assertEqual(mapping.prompt_col, "instruction")
        self.assertEqual(mapping.completion_col, "output")

    def test_parse_mapping_arg_json(self):
        mapping = parse_mapping_arg('{"text_col": "my_text"}')
        self.assertEqual(mapping.text_col, "my_text")

    def test_cli_direct_conversion(self):
        argv = [
            "-d", str(self.src_file),
            "-f", "prompt_completion",
            "-o", str(self.out_dir),
            "--prompt-col", "instruction",
            "--completion-col", "output",
            "--train", "70",
            "--valid", "20",
            "--test", "10",
            "--seed", "123456",
        ]
        ret = main(argv)
        self.assertEqual(ret, 0)

        train_file = self.out_dir / "train.jsonl"
        valid_file = self.out_dir / "valid.jsonl"
        test_file = self.out_dir / "test.jsonl"

        self.assertTrue(train_file.exists())
        self.assertTrue(valid_file.exists())
        self.assertTrue(test_file.exists())

        with open(train_file) as f:
            lines = [json.loads(l) for l in f]
            self.assertEqual(len(lines), 21)  # 70% of 30 is 21

    def test_cli_tsv_conversion(self):
        tsv_path = Path(self.temp_dir) / "source.tsv"
        import csv
        records = make_sample_qa_records(20)
        with open(tsv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()), delimiter="\t")
            writer.writeheader()
            for r in records:
                writer.writerow(r)

        out_dir = Path(self.temp_dir) / "tsv_out"
        argv = [
            "-d", str(tsv_path),
            "-f", "prompt_completion",
            "-o", str(out_dir),
            "--prompt-col", "instruction",
            "--completion-col", "output",
        ]
        ret = main(argv)
        self.assertEqual(ret, 0)
        self.assertTrue((out_dir / "train.jsonl").exists())

    def test_cli_sqlite_conversion(self):
        import sqlite3
        sqlite_path = Path(self.temp_dir) / "source.sqlite"
        records = make_sample_qa_records(20)
        conn = sqlite3.connect(str(sqlite_path))
        cur = conn.cursor()
        cur.execute("CREATE TABLE records (instruction TEXT, output TEXT);")
        for r in records:
            cur.execute("INSERT INTO records VALUES (?, ?)", (r["instruction"], r["output"]))
        conn.commit()
        conn.close()

        out_dir = Path(self.temp_dir) / "sqlite_out"
        argv = [
            "-d", f"{sqlite_path}::records",
            "-f", "prompt_completion",
            "-o", str(out_dir),
            "--prompt-col", "instruction",
            "--completion-col", "output",
        ]
        ret = main(argv)
        self.assertEqual(ret, 0)
        self.assertTrue((out_dir / "train.jsonl").exists())

    def test_cli_webdataset_conversion(self):
        import io
        import tarfile
        tar_path = Path(self.temp_dir) / "source.tar"
        with tarfile.open(tar_path, "w") as tar:
            for i in range(10):
                p_bytes = f"Prompt #{i}".encode("utf-8")
                c_bytes = f"Answer #{i}".encode("utf-8")
                ti_p = tarfile.TarInfo(f"{i:03d}.prompt.txt")
                ti_p.size = len(p_bytes)
                tar.addfile(ti_p, io.BytesIO(p_bytes))
                ti_c = tarfile.TarInfo(f"{i:03d}.completion.txt")
                ti_c.size = len(c_bytes)
                tar.addfile(ti_c, io.BytesIO(c_bytes))

        out_dir = Path(self.temp_dir) / "wds_out"
        argv = [
            "-d", str(tar_path),
            "-f", "prompt_completion",
            "-o", str(out_dir),
            "--prompt-col", "prompt",
            "--completion-col", "completion",
        ]
        ret = main(argv)
        self.assertEqual(ret, 0)
        self.assertTrue((out_dir / "train.jsonl").exists())

    def test_interactive_wizard(self):
        import io
        import sys
        from mlx_commander.tui.wizard_fallback import run_interactive_wizard

        inputs = [
            "3",    # Format: prompt_completion
            "1",    # Prompt col: instruction
            "2",    # Completion col: output
            "80",   # Train %
            "10",   # Valid %
            "10",   # Test %
            "999",  # Seed
            "Y",    # Confirm
        ]
        orig_stdin = sys.stdin
        try:
            sys.stdin = io.StringIO("\n".join(inputs) + "\n")
            res = run_interactive_wizard(dataset_path=str(self.src_file), output_dir_arg=str(self.out_dir))
            self.assertEqual(res.record_counts["train"], 24)
            self.assertEqual(res.record_counts["valid"], 3)
            self.assertEqual(res.record_counts["test"], 3)
        finally:
            sys.stdin = orig_stdin

    def test_entry_points(self):
        import subprocess
        root_dir = Path(__file__).resolve().parent.parent
        python_bin = sys.executable

        cmds = [
            [python_bin, "mlx_commander.py", "--version"],
            [python_bin, "run.py", "--version"],
            [python_bin, "-m", "mlx_commander", "--version"],
        ]
        for cmd in cmds:
            p = subprocess.run(cmd, cwd=str(root_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(p.returncode, 0, f"Command failed: {cmd}, stderr: {p.stderr}")
            self.assertIn("mlx_commander", p.stdout)

    def test_cli_no_fallback_to_wizard_on_tui_error(self):
        from unittest.mock import patch
        with patch("mlx_commander.cli.is_interactive_tty", return_value=True):
            with patch("mlx_commander.cli.launch_tui", side_effect=RuntimeError("curses failed")):
                with patch("mlx_commander.cli.run_interactive_wizard") as mock_wizard:
                    ret = main(["-d", str(self.src_file)])
                    self.assertEqual(ret, 1)
                    # Ensure the interactive wizard is NEVER called as a fallback
                    mock_wizard.assert_not_called()

    def test_cli_direct_conversion_default_output_folder(self):
        # When -o is omitted, direct conversion defaults to saving in <src_dir>/mlx_dataset
        argv = [
            "-d", str(self.src_file),
            "-f", "prompt_completion",
            "--prompt-col", "instruction",
            "--completion-col", "output",
        ]
        ret = main(argv)
        self.assertEqual(ret, 0)

        expected_dir = self.src_file.parent / "mlx_dataset"
        self.assertTrue((expected_dir / "train.jsonl").exists())
        self.assertTrue((expected_dir / "valid.jsonl").exists())
        self.assertTrue((expected_dir / "test.jsonl").exists())

    def test_cli_direct_conversion_custom_output_folder(self):
        custom_dir = Path(self.temp_dir) / "custom_destination"
        argv = [
            "-d", str(self.src_file),
            "-f", "prompt_completion",
            "-o", str(custom_dir),
            "--prompt-col", "instruction",
            "--completion-col", "output",
        ]
        ret = main(argv)
        self.assertEqual(ret, 0)
        self.assertTrue((custom_dir / "train.jsonl").exists())

    def test_cli_manifest_file_flag(self):
        custom_manifest = Path(self.temp_dir) / "agent_manifest.json"
        argv = [
            "-d", str(self.src_file),
            "-f", "prompt_completion",
            "-o", str(self.out_dir),
            "--prompt-col", "instruction",
            "--completion-col", "output",
            "--manifest-file", str(custom_manifest),
        ]
        ret = main(argv)
        self.assertEqual(ret, 0)
        self.assertTrue(custom_manifest.exists())
        with open(custom_manifest) as f:
            data = json.load(f)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["total_records"], 30)

    def test_cli_spawn_terminal_flag_delegates(self):
        from unittest.mock import patch
        with patch("mlx_commander.terminal_spawner.spawn_terminal_tui", return_value=0) as mock_spawn:
            argv = ["--spawn-terminal", "-d", str(self.src_file), "-f", "chat"]
            ret = main(argv)
            self.assertEqual(ret, 0)
            mock_spawn.assert_called_once()

    def test_cli_mcp_flag_delegates(self):
        from unittest.mock import patch
        with patch("mlx_commander.mcp_server.run_mcp_server", return_value=0) as mock_mcp:
            ret = main(["--mcp"])
            self.assertEqual(ret, 0)
            mock_mcp.assert_called_once()

    def test_cli_missing_dependency_clean_exit(self):
        parquet_path = Path(self.temp_dir) / "data.parquet"
        parquet_path.write_bytes(b"PAR1mock")
        import io
        from contextlib import redirect_stderr
        from unittest.mock import patch

        stderr_buf = io.StringIO()
        with patch("mlx_commander.loader.HAS_PYARROW", False):
            with redirect_stderr(stderr_buf):
                ret = main(["-d", str(parquet_path), "-f", "prompt_completion"])

        self.assertEqual(ret, 1)
        err_output = stderr_buf.getvalue()
        self.assertIn("Missing Dependency Error", err_output)
        self.assertIn("pip install pyarrow", err_output)

    def test_cli_mode_flags_routing(self):
        from unittest.mock import patch
        test_cases = [
            ([], 1),
            (["--single-run"], 1),
            (["--lora"], 1),
            (["--multi-run"], 2),
            (["--converter"], 0),
            (["--dataset-converter"], 0),
        ]
        for flags, expected_tab in test_cases:
            with patch("mlx_commander.cli.is_interactive_tty", return_value=True):
                with patch("mlx_commander.cli.launch_tui", return_value=None) as mock_launch:
                    main(flags)
                    mock_launch.assert_called_once()
                    prefill = mock_launch.call_args[1].get("prefill", {})
                    self.assertEqual(prefill.get("active_tab"), expected_tab, f"Failed for flags: {flags}")


if __name__ == "__main__":
    unittest.main()


