"""
mlx_commander - MLX Training Ops & Dataset Converter
Configure and run model training on Apple MLX via interactive TUI and CLI.
"""

import os
import warnings

# Suppress harmless upstream UserWarnings from transformers audio/mel processor initialization in multimodal models
warnings.filterwarnings("ignore", message=".*mel filter has all zero values.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*num_mel_filters.*", category=UserWarning)
warnings.filterwarnings("ignore", module="transformers.audio_utils")

# Set ncurses escape delay to 25ms (default is 1000ms) to ensure instantaneous
# exit/cancellation on ESC across all curses pickers and dialogs.
os.environ.setdefault("ESCDELAY", "25")

__version__ = "0.6.2"
