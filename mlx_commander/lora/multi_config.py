"""
Multi-Run LoRA Fine-Tuning Sweep Model for Apple MLX.
Defines multi-condition hyperparameter sweeps, Cartesian product run generation,
and aggregated runtime estimates (max peak RAM, total duration, min implied epochs).
"""

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import LoraRunConfig
from .estimator import (
    calculate_implied_epochs,
    estimate_duration,
    estimate_peak_memory,
)


SWEEP_FIELD_DEFS = [
    ("iters", "Iterations", int),
    ("batch_size", "Batch Size", int),
    ("grad_accumulation_steps", "Gradient Accumulation Steps", int),
    ("learning_rate", "Learning rate", float),
    ("lora_rank", "LoRA Rank", int),
    ("lora_alpha", "LoRA Alpha", float),
    ("lora_dropout", "LoRA Dropout", float),
    ("max_seq_length", "Max Seq Length", int),
    ("num_layers", "Fine-Tuned Layers", int),
    ("grad_checkpoint", "Grad Checkpoint", bool),
    ("mask_prompt", "Mask prompt", bool),
    ("save_every", "Save Every", int),
    ("steps_per_eval", 'Steps per "Eval" (validation loss)', int),
    ("run_eval", "Run evals on test set (experimental)", bool),
]


@dataclass
class MultiLoraRunConfig:
    """Configuration for a multi-condition hyperparameter sweep producing multiple runs."""

    # Base settings common across runs
    model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    data: str = "mlx_dataset"
    train: bool = True
    test: bool = False
    fine_tune_type: str = "lora"
    optimizer: str = "adamw"
    adapter_path: str = "adapters"
    seed: int = 0
    val_batches: int = 25
    engine: str = "mlx_lm"
    train_vision: bool = False
    train_on_completions: bool = True

    # Hyperparameter lists (each can hold 1 or more conditions)
    iters: List[int] = field(default_factory=lambda: [1000])
    batch_size: List[int] = field(default_factory=lambda: [4])
    grad_accumulation_steps: List[int] = field(default_factory=lambda: [1])
    learning_rate: List[float] = field(default_factory=lambda: [1e-5])
    lora_rank: List[int] = field(default_factory=lambda: [8])
    lora_alpha: List[float] = field(default_factory=lambda: [16.0])
    lora_dropout: List[float] = field(default_factory=lambda: [0.0])
    max_seq_length: List[int] = field(default_factory=lambda: [2048])
    num_layers: List[int] = field(default_factory=lambda: [16])
    grad_checkpoint: List[bool] = field(default_factory=lambda: [True])
    mask_prompt: List[bool] = field(default_factory=lambda: [True])
    save_every: List[int] = field(default_factory=lambda: [100])
    steps_per_eval: List[int] = field(default_factory=lambda: [100])
    run_eval: List[bool] = field(default_factory=lambda: [False])

    def get_field_values(self, field_name: str) -> List[Any]:
        """Get the list of values for a specific hyperparameter field."""
        val = getattr(self, field_name, None)
        if isinstance(val, list):
            return val
        return [val] if val is not None else []

    def set_field_values(self, field_name: str, values: List[Any]) -> None:
        """Set the list of values for a specific hyperparameter field, ensuring at least 1 value."""
        if not values:
            return
        setattr(self, field_name, list(values))

    @property
    def total_runs_count(self) -> int:
        """Total number of runs in the Cartesian product of all condition lists."""
        total = 1
        for field_name, _, _ in SWEEP_FIELD_DEFS:
            vals = self.get_field_values(field_name)
            total *= max(1, len(vals))
        return total

    def get_varying_hyperparameters(self) -> List[Tuple[str, str, List[Any]]]:
        """
        Return the list of (field_name, display_label, values) that vary across the sweep (len > 1).
        If no field has > 1 values, returns the primary fields (learning_rate, lora_rank, mask_prompt)
        with their single values so the sweep grid still renders cleanly.
        """
        varying = []
        for field_name, label, _ in SWEEP_FIELD_DEFS:
            vals = self.get_field_values(field_name)
            if len(vals) > 1:
                varying.append((field_name, label, vals))

        if not varying:
            # Fallback to display the 3 iconic primary fields from the screenshot
            primary = ["learning_rate", "lora_rank", "mask_prompt"]
            for field_name, label, _ in SWEEP_FIELD_DEFS:
                if field_name in primary:
                    varying.append((field_name, label, self.get_field_values(field_name)))

        return varying

    def generate_runs(self) -> List[LoraRunConfig]:
        """Generate the full Cartesian product list of concrete LoraRunConfig runs."""
        # Gather all field names and their lists
        field_keys = [f[0] for f in SWEEP_FIELD_DEFS]
        val_lists = [self.get_field_values(k) for k in field_keys]

        product_combs = list(itertools.product(*val_lists))
        runs: List[LoraRunConfig] = []

        for idx, comb in enumerate(product_combs, 1):
            comb_dict = dict(zip(field_keys, comb))
            cfg = LoraRunConfig(
                model=self.model,
                data=self.data,
                train=self.train,
                test=self.test,
                fine_tune_type=self.fine_tune_type,
                optimizer=self.optimizer,
                adapter_path=self.adapter_path,
                seed=self.seed,
                val_batches=self.val_batches,
                grad_accumulation_steps=comb_dict["grad_accumulation_steps"],
                iters=comb_dict["iters"],
                batch_size=comb_dict["batch_size"],
                learning_rate=comb_dict["learning_rate"],
                lora_rank=comb_dict["lora_rank"],
                lora_alpha=comb_dict["lora_alpha"],
                lora_dropout=comb_dict["lora_dropout"],
                max_seq_length=comb_dict["max_seq_length"],
                num_layers=comb_dict["num_layers"],
                grad_checkpoint=comb_dict["grad_checkpoint"],
                mask_prompt=comb_dict["mask_prompt"],
                save_every=comb_dict["save_every"],
                steps_per_eval=comb_dict["steps_per_eval"],
                run_eval=comb_dict["run_eval"],
                engine=self.engine,
                train_vision=self.train_vision,
                train_on_completions=self.train_on_completions,
            )
            runs.append(cfg)

        return runs

    def calculate_sweep_estimates(
        self,
        dataset_records: int,
        chip_name: Optional[str] = None,
        model_size_gb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate aggregated runtime estimates for the multi-run sweep:
          - Maximum peak unified RAM across all run conditions
          - Total estimated duration for all scheduled runs to complete
          - Minimum implied epochs across all runs
        """
        runs = self.generate_runs()
        if not runs:
            return {
                "total_runs": 0,
                "min_implied_epochs": 0.0,
                "max_peak_ram_gb": 0.0,
                "total_duration_seconds": 0.0,
                "total_duration_str": "0s",
            }

        min_epochs: Optional[float] = float("inf")
        max_ram = 0.0
        total_duration = 0.0
        max_ram_dict: Dict[str, Any] = {}
        for r in runs:
            # 1. Implied epochs: iters, batch_size, total_train_records, grad_accumulation_steps
            if dataset_records > 0:
                ep = calculate_implied_epochs(
                    r.iters,
                    r.batch_size,
                    dataset_records,
                    getattr(r, "grad_accumulation_steps", 1),
                )
                if ep is not None and ep < min_epochs:
                    min_epochs = ep

            # 2. Peak RAM: passes LoraRunConfig
            mem_est = estimate_peak_memory(r)
            peak_gb = mem_est.get("peak_gb", 0)
            if peak_gb >= max_ram:
                max_ram = peak_gb
                max_ram_dict = mem_est

            # 3. Duration: passes LoraRunConfig, chip_name
            dur_est = estimate_duration(r, chip_name=chip_name)
            total_duration += dur_est.get("seconds", 0.0)

        if min_epochs == float("inf"):
            min_epochs = None

        # Format total duration cleanly
        mins, secs = divmod(int(total_duration), 60)
        hrs, mins = divmod(mins, 60)
        if hrs > 0:
            dur_str = f"{hrs}h {mins:02d}m"
        elif mins > 0:
            dur_str = f"{mins}m {secs:02d}s"
        else:
            dur_str = f"{secs}s"

        return {
            "total_runs": len(runs),
            "min_implied_epochs": min_epochs,
            "max_peak_ram_gb": max_ram,
            "total_ram_gb": max_ram_dict.get("total_ram_gb", 16),
            "usage_ratio": max_ram_dict.get("usage_ratio", round(max_ram / 16.0, 2)),
            "safety_level": max_ram_dict.get("safety_level", "SAFE"),
            "safety_tier": max_ram_dict.get("safety_tier", "safe"),
            "total_duration_seconds": total_duration,
            "total_duration_str": dur_str,
        }
