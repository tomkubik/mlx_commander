"""
LoRA Fine-Tuning Configuration Model for Apple MLX (mlx-lm).
Defines explicit training arguments, validation, YAML serialization,
and CLI command generation for mlx_lm.lora.
"""

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

POPULAR_MLX_MODELS = [
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", "Llama-3.2-3B-Instruct (4-bit, fast & light)"),
    ("mlx-community/Llama-3.2-1B-Instruct-4bit", "Llama-3.2-1B-Instruct (4-bit, ultra-light)"),
    ("mlx-community/Qwen2.5-7B-Instruct-4bit", "Qwen2.5-7B-Instruct (4-bit, reasoning/code)"),
    ("mlx-community/Qwen2.5-3B-Instruct-4bit", "Qwen2.5-3B-Instruct (4-bit, balanced)"),
    ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", "Mistral-7B-Instruct-v0.3 (4-bit)"),
    ("mlx-community/Phi-3.5-mini-instruct-4bit", "Phi-3.5-mini-instruct (4-bit, 3.8B)"),
]

FINE_TUNE_TYPES = ["lora", "dora", "full"]
OPTIMIZERS = ["adamw", "adam", "muon", "sgd", "adafactor"]


def sanitize_model_slug(model_name: str) -> str:
    """
    Extract clean, filesystem-safe model slug from Hugging Face ID or local path.
    e.g. 'mlx-community/Llama-3.2-3B-Instruct-4bit' -> 'Llama-3.2-3B-Instruct-4bit'
    e.g. '/path/to/my_model/' -> 'my_model'
    """
    slug = model_name.strip().rstrip("/")
    if "/" in slug:
        slug = slug.split("/")[-1]
    slug = re.sub(r'[^a-zA-Z0-9.\-_]+', '-', slug).strip('-')
    return slug or "model"


def format_learning_rate(lr: float) -> str:
    """Format learning rate cleanly without decimal points for file names (e.g. 1e-5)."""
    s = f"{lr:g}"
    if "e" in s.lower():
        base, exp = re.split(r"[eE]", s)
        return f"{base}e{int(exp)}".replace(".", "p")
    return s.replace(".", "p")


def generate_hyperparameters_slug(
    config: Optional["LoraRunConfig"] = None,
    implied_epochs: Optional[float] = None,
    *,
    fine_tune_type: Optional[str] = None,
    rank: Optional[int] = None,
    alpha: Optional[float] = None,
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None,
    iters: Optional[int] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Generate the self-documenting hyperparameters slug without index prefix.
    Structure:
      {method}_r{rank}_a{alpha}_lr{lr}_b{batch}_i{iters}_{model_slug}
    Example:
      lora_r16_a32_lr1e-5_b4_i1000_Llama-3.2-3B-Instruct-4bit
    """
    if config is not None:
        method = config.fine_tune_type.lower()
        lr_val = config.learning_rate
        r_val = config.lora_rank
        a_val = config.lora_alpha
        b_val = config.batch_size
        i_val = config.iters
        m_name = config.model
    else:
        method = (fine_tune_type or "lora").lower()
        lr_val = learning_rate if learning_rate is not None else 1e-5
        r_val = rank if rank is not None else 8
        a_val = alpha if alpha is not None else 16.0
        b_val = batch_size if batch_size is not None else 4
        i_val = iters if iters is not None else 1000
        m_name = model_name or "model"

    lr_str = format_learning_rate(lr_val)
    model_slug = sanitize_model_slug(m_name)

    parts = [method]
    if method in ("lora", "dora"):
        parts.append(f"r{r_val}")
        alpha_val = int(a_val) if isinstance(a_val, (int, float)) and float(a_val).is_integer() else a_val
        parts.append(f"a{alpha_val}")
    parts.append(f"lr{lr_str}")
    parts.append(f"b{b_val}")
    parts.append(f"i{i_val}")
    if implied_epochs is not None and implied_epochs > 0:
        parts.append(f"ep{implied_epochs:.1f}".replace(".", "p"))
    parts.append(model_slug)
    return "_".join(parts)


def generate_deterministic_run_name(
    config: Optional["LoraRunConfig"] = None,
    index: int = 1,
    implied_epochs: Optional[float] = None,
    *,
    fine_tune_type: Optional[str] = None,
    rank: Optional[int] = None,
    alpha: Optional[float] = None,
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None,
    iters: Optional[int] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Generate a deterministic, self-documenting run name and directory slug.
    Structure:
      {index:02d}_{method}_r{rank}_a{alpha}_lr{lr}_b{batch}_i{iters}_{model_slug}
    Example:
      01_lora_r16_a32_lr1e-5_b4_i1000_Llama-3.2-3B-Instruct-4bit
    """
    slug = generate_hyperparameters_slug(
        config=config,
        implied_epochs=implied_epochs,
        fine_tune_type=fine_tune_type,
        rank=rank,
        alpha=alpha,
        learning_rate=learning_rate,
        batch_size=batch_size,
        iters=iters,
        model_name=model_name,
    )
    return f"{index:02d}_{slug}"


def format_adapter_filename(
    step: int,
    config: Optional["LoraRunConfig"] = None,
    *,
    implied_epochs: Optional[float] = None,
    prefix: str = "adapters",
    extension: str = ".safetensors",
    fine_tune_type: Optional[str] = None,
    rank: Optional[int] = None,
    alpha: Optional[float] = None,
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None,
    iters: Optional[int] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Format a saved adapter filename starting with the iteration count at which it was saved,
    with all training hyperparameters appended.
    Structure:
      {step:07d}_{prefix}_{hyperparameters_slug}{extension}
    Example:
      0000100_adapters_lora_r16_a32_lr1e-5_b4_i1000_Llama-3.2-3B-Instruct-4bit.safetensors
    """
    slug = generate_hyperparameters_slug(
        config=config,
        implied_epochs=implied_epochs,
        fine_tune_type=fine_tune_type,
        rank=rank,
        alpha=alpha,
        learning_rate=learning_rate,
        batch_size=batch_size,
        iters=iters,
        model_name=model_name,
    )
    step_str = f"{step:07d}" if isinstance(step, int) and step >= 0 else str(step)
    if prefix:
        return f"{step_str}_{prefix}_{slug}{extension}"
    return f"{step_str}_{slug}{extension}"


@dataclass
class LoraRunConfig:
    id: str = field(default_factory=lambda: f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}")
    name: str = ""
    model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    data: str = "mlx_dataset"
    train: bool = True
    test: bool = False
    fine_tune_type: str = "lora"
    optimizer: str = "adamw"
    iters: int = 1000
    batch_size: int = 4
    learning_rate: float = 1e-5
    num_layers: int = 16
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    max_seq_length: int = 2048
    grad_checkpoint: bool = True
    grad_accumulation_steps: int = 1
    mask_prompt: bool = True
    steps_per_report: int = 10
    steps_per_eval: int = 100
    val_batches: int = 25
    save_every: int = 100
    adapter_path: str = "adapters"
    seed: int = 0
    resume_adapter_file: Optional[str] = None
    run_eval: bool = False

    # Queue execution & tracking
    status: str = "queued"  # queued, running, completed, failed, cancelled
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    log_file: Optional[str] = None
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    sequence_index: Optional[int] = None
    is_custom_name: Optional[bool] = None
    wandb_url: Optional[str] = None
    wandb_project: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            self.name = generate_deterministic_run_name(self, index=1)
            if self.is_custom_name is None:
                self.is_custom_name = False
        else:
            if self.is_custom_name is None:
                self.is_custom_name = True
        self.is_custom_name = bool(self.is_custom_name)

        if not self.adapter_path or self.adapter_path == "adapters":
            self.adapter_path = f"adapters/{self.name}"

    def get_hyperparameters_slug(self, implied_epochs: Optional[float] = None) -> str:
        """Get the self-documenting hyperparameters slug for this run configuration."""
        return generate_hyperparameters_slug(self, implied_epochs=implied_epochs)

    def format_adapter_filename(
        self,
        step: int,
        prefix: str = "adapters",
        extension: str = ".safetensors",
        implied_epochs: Optional[float] = None,
    ) -> str:
        """Format a saved adapter filename starting with step count and with all hyperparameters appended."""
        return format_adapter_filename(
            step=step,
            config=self,
            prefix=prefix,
            extension=extension,
            implied_epochs=implied_epochs,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoraRunConfig":
        clean_data = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**clean_data)

    @classmethod
    def from_yaml(cls, source: Any) -> "LoraRunConfig":
        """Parse a LoraRunConfig from a YAML file path or YAML-formatted string."""
        content = ""
        try:
            p = Path(source)
            if p.is_file():
                content = p.read_text(encoding="utf-8")
            else:
                content = str(source)
        except Exception:
            content = str(source)

        data: Dict[str, Any] = {}
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("lora_parameters:"):
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k == "rank":
                    try:
                        data["lora_rank"] = int(v)
                    except ValueError:
                        pass
                elif k == "scale":
                    try:
                        data["lora_alpha"] = float(v)
                    except ValueError:
                        pass
                elif k == "dropout":
                    try:
                        data["lora_dropout"] = float(v)
                    except ValueError:
                        pass
                elif k in (
                    "batch_size",
                    "iters",
                    "val_batches",
                    "steps_per_report",
                    "steps_per_eval",
                    "save_every",
                    "max_seq_length",
                    "grad_accumulation_steps",
                    "seed",
                    "num_layers",
                ):
                    try:
                        data[k] = int(v)
                    except ValueError:
                        pass
                elif k == "learning_rate":
                    try:
                        data[k] = float(v)
                    except ValueError:
                        pass
                elif k in ("train", "test", "grad_checkpoint", "mask_prompt", "run_eval"):
                    data[k] = v.lower() == "true"
                else:
                    data[k] = v
        return cls.from_dict(data)

    def to_mlx_yaml(self) -> str:
        """Generate YAML configuration compliant with mlx_lm.lora --config schema."""
        lines = [
            f'model: "{self.model}"',
            f'train: {"true" if self.train else "false"}',
            f'data: "{self.data}"',
            f'fine_tune_type: "{self.fine_tune_type}"',
            f'optimizer: "{self.optimizer}"',
            f'batch_size: {self.batch_size}',
            f'iters: {self.iters}',
            f'val_batches: {self.val_batches}',
            f'learning_rate: {self.learning_rate:g}',
            f'steps_per_report: {self.steps_per_report}',
            f'steps_per_eval: {self.steps_per_eval}',
            f'save_every: {self.save_every}',
            f'adapter_path: "{self.adapter_path}"',
            f'max_seq_length: {self.max_seq_length}',
            f'grad_checkpoint: {"true" if self.grad_checkpoint else "false"}',
            f'grad_accumulation_steps: {self.grad_accumulation_steps}',
            f'mask_prompt: {"true" if self.mask_prompt else "false"}',
            f'seed: {self.seed}',
            f'num_layers: {self.num_layers}',
            'lora_parameters:',
            f'  rank: {self.lora_rank}',
            f'  dropout: {self.lora_dropout}',
            f'  scale: {self.lora_alpha}',
        ]
        if self.test:
            lines.append('test: true')
        return "\n".join(lines) + "\n"

    def to_cli_command(self, config_file: Optional[str] = None) -> str:
        """Generate the equivalent mlx_lm.lora command invocation."""
        if config_file:
            return f"mlx_lm.lora --config {config_file}"
        cmd = [
            "mlx_lm.lora",
            f"--model {self.model}",
            f"--data {self.data}",
        ]
        if self.train:
            cmd.append("--train")
        if self.test:
            cmd.append("--test")
        cmd.extend([
            f"--fine-tune-type {self.fine_tune_type}",
            f"--optimizer {self.optimizer}",
            f"--batch-size {self.batch_size}",
            f"--iters {self.iters}",
            f"--learning-rate {self.learning_rate:g}",
            f"--num-layers {self.num_layers}",
            f"--max-seq-length {self.max_seq_length}",
            f"--adapter-path {self.adapter_path}",
            f"--save-every {self.save_every}",
            f"--steps-per-eval {self.steps_per_eval}",
            f"--steps-per-report {self.steps_per_report}",
        ])
        if self.grad_checkpoint:
            cmd.append("--grad-checkpoint")
        if self.mask_prompt:
            cmd.append("--mask-prompt")
        if self.resume_adapter_file:
            cmd.append(f"--resume-adapter-file {self.resume_adapter_file}")
        return " ".join(cmd)

    def validate(self) -> List[str]:
        """Validate config parameters and return list of human-readable errors."""
        errors: List[str] = []
        if self.iters <= 0:
            errors.append("Iterations must be > 0.")
        if self.batch_size <= 0:
            errors.append("Batch size must be > 0.")
        if self.learning_rate <= 0:
            errors.append("Learning rate must be > 0.")
        if self.lora_rank <= 0:
            errors.append("LoRA rank must be > 0.")
        if self.num_layers <= 0:
            errors.append("Number of layers must be > 0.")
        if not (0.0 <= self.lora_dropout <= 0.5):
            errors.append("Dropout must be between 0.0 and 0.5.")
        if self.max_seq_length < 64:
            errors.append("Max sequence length must be at least 64.")
        return errors
