"""
mlx_commander - Hugging Face to MLX Dataset Converter
Convert Hugging Face datasets into Apple MLX fine-tuning formats with interactive TUI and CLI.
"""

import os

# Set ncurses escape delay to 25ms (default is 1000ms) to ensure instantaneous
# exit/cancellation on ESC across all curses pickers and dialogs.
os.environ.setdefault("ESCDELAY", "25")

__version__ = "0.5.1"
