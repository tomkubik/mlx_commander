"""
MLX Commander LoRA Fine-Tuning & Queue Orchestration Module.
"""

from .config import (
    FINE_TUNE_TYPES,
    OPTIMIZERS,
    POPULAR_MLX_MODELS,
    LoraRunConfig,
    format_learning_rate,
    format_adapter_filename,
    generate_deterministic_run_name,
    generate_hyperparameters_slug,
    sanitize_model_slug,
)
from .multi_config import MultiLoraRunConfig, SWEEP_FIELD_DEFS
from .estimator import (
    calculate_implied_epochs,
    estimate_duration,
    estimate_peak_memory,
    get_apple_silicon_chip,
    get_hardware_memory_bytes,
)
from .model_info import (
    ModelMetadata,
    detect_model_engine,
    inspect_local_model,
    is_engine_installed,
    is_local_path,
    is_model_directory,
    normalize_model_path,
    scan_local_models,
)
from .queue import QueueManager
from .runner import rename_saved_adapters, run_lora_queue
from .tracking import (
    WandbTracker,
    is_wandb_available,
    is_wandb_logged_in,
    parse_mlx_log_line,
)
from .vlm_safeguards import (
    calculate_vlm_batch_tweak,
    calculate_vlm_sweep_tweak,
    should_suggest_vlm_tweak,
    should_suggest_vlm_sweep_tweak,
)

__all__ = [
    "LoraRunConfig",
    "MultiLoraRunConfig",
    "SWEEP_FIELD_DEFS",
    "QueueManager",
    "rename_saved_adapters",
    "run_lora_queue",
    "calculate_implied_epochs",
    "estimate_peak_memory",
    "estimate_duration",
    "get_hardware_memory_bytes",
    "get_apple_silicon_chip",
    "POPULAR_MLX_MODELS",
    "FINE_TUNE_TYPES",
    "OPTIMIZERS",
    "format_adapter_filename",
    "generate_deterministic_run_name",
    "generate_hyperparameters_slug",
    "sanitize_model_slug",
    "format_learning_rate",
    "WandbTracker",
    "is_wandb_available",
    "is_wandb_logged_in",
    "parse_mlx_log_line",
    "ModelMetadata",
    "detect_model_engine",
    "is_engine_installed",
    "inspect_local_model",
    "is_local_path",
    "is_model_directory",
    "normalize_model_path",
    "scan_local_models",
    "calculate_vlm_batch_tweak",
    "calculate_vlm_sweep_tweak",
    "should_suggest_vlm_tweak",
    "should_suggest_vlm_sweep_tweak",
]
