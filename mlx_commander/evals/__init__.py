"""
Generative Test Set Evaluation Engine for Apple MLX (mlx-lm).
"""

from .metrics import (
    build_confusion_matrix,
    build_migration_matrix,
    classify_transition,
    compute_exact_match,
    compute_substring_match,
    compute_word_metrics,
    format_cli_categorical_confusion_matrix,
    format_cli_eval_summary_matrices,
    format_cli_migration_matrix,
    normalize_answer,
)
from .runner import load_test_dataset, run_generative_eval
from .storage import (
    append_to_leaderboard_csv,
    generate_html_dashboard,
    save_run_predictions,
)

__all__ = [
    "compute_exact_match",
    "compute_substring_match",
    "compute_word_metrics",
    "classify_transition",
    "build_migration_matrix",
    "build_confusion_matrix",
    "format_cli_migration_matrix",
    "format_cli_categorical_confusion_matrix",
    "format_cli_eval_summary_matrices",
    "normalize_answer",
    "load_test_dataset",
    "run_generative_eval",
    "append_to_leaderboard_csv",
    "save_run_predictions",
    "generate_html_dashboard",
]
