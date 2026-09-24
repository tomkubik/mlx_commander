"""
Unit tests for terminal_spawner helper functions.
"""

import json
import os
import tempfile
import unittest

from unittest.mock import MagicMock, patch

from mlx_commander.terminal_spawner import (
    build_terminal_script,
    ensure_adequate_terminal_size,
    is_macos,
)


class TestTerminalSpawner(unittest.TestCase):

    def test_is_macos_boolean(self):
        self.assertIsInstance(is_macos(), bool)

    def test_build_terminal_script(self):
        script = build_terminal_script(
            python_exe="/usr/bin/python3",
            args=["--dataset", "data.parquet", "--format", "chat"],
            status_file="/tmp/test_status.json",
            working_dir="/Users/test",
        )

        self.assertIn("cd /Users/test", script)
        self.assertIn("/usr/bin/python3 -m mlx_commander", script)
        self.assertIn("--tui", script)
        self.assertIn("--dataset data.parquet", script)
        self.assertIn("--format chat", script)
        self.assertIn("/tmp/test_status.json", script)
        self.assertIn("finished", script)
        self.assertIn(r"\e[8;45;120t", script)

    @patch("mlx_commander.terminal_spawner.is_terminal_interactive", return_value=False)
    @patch("shutil.get_terminal_size", return_value=(80, 24))
    def test_ensure_adequate_terminal_size_non_interactive_no_op(self, mock_size, mock_interactive):
        # In non-interactive or testing environments, ensure_adequate_terminal_size returns immediately
        cols, lines = ensure_adequate_terminal_size(min_cols=120, min_lines=45)
        self.assertEqual((cols, lines), (80, 24))

    @patch("mlx_commander.terminal_spawner.is_terminal_interactive", return_value=True)
    @patch("shutil.get_terminal_size", return_value=(150, 60))
    def test_ensure_adequate_terminal_size_already_adequate_no_shrink(self, mock_size, mock_interactive):
        # Windows already larger than 120x45 should never be shrunk
        cols, lines = ensure_adequate_terminal_size(min_cols=120, min_lines=45)
        self.assertEqual((cols, lines), (150, 60))

    @patch("mlx_commander.terminal_spawner.is_macos", return_value=False)
    @patch("mlx_commander.terminal_spawner.is_terminal_interactive", return_value=True)
    @patch("shutil.get_terminal_size", return_value=(80, 24))
    @patch("sys.stdout.write")
    def test_ensure_adequate_terminal_size_interactive_resizes_smaller_window(self, mock_write, mock_size, mock_interactive, mock_macos):
        cols, lines = ensure_adequate_terminal_size(min_cols=120, min_lines=45)
        self.assertEqual(cols, 120)
        self.assertEqual(lines, 45)
        mock_write.assert_any_call("\033[8;45;120t")


if __name__ == "__main__":
    unittest.main()
