"""
Queue Manager for MLX Commander LoRA Fine-Tuning Runs.
Maintains persistent FIFO queue state on disk in queue.json,
generates individual YAML configs, and supports run management (add, clone, delete, reorder).
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import LoraRunConfig, generate_deterministic_run_name


class QueueManager:
    """Manages a persistent FIFO queue of LoRA fine-tuning runs."""

    def __init__(self, queue_dir: Optional[Path] = None):
        if queue_dir is None:
            queue_dir = Path.cwd() / "mlx_runs"
        self.queue_dir = Path(queue_dir).resolve()
        self.configs_dir = self.queue_dir / "configs"
        self.logs_dir = self.queue_dir / "logs"
        self.queue_file = self.queue_dir / "queue.json"
        self.script_file = self.queue_dir / "run_queue.sh"
        self.runs: List[LoraRunConfig] = []

        self._ensure_dirs()
        self.load()

    def _ensure_dirs(self) -> None:
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        """Load queued runs from queue.json if it exists."""
        if not self.queue_file.exists():
            self.runs = []
            return
        try:
            with open(self.queue_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.runs = [LoraRunConfig.from_dict(item) for item in data.get("runs", [])]
        except Exception:
            self.runs = []

    def save(self) -> None:
        """Save all runs to queue.json and write individual YAML configs."""
        self._ensure_dirs()
        for r in self.runs:
            cfg_path = self.configs_dir / f"{r.id}.yaml"
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write(r.to_mlx_yaml())
            r.log_file = str(self.logs_dir / f"{r.id}.log")

        try:
            from mlx_commander import __version__
            ver = __version__
        except Exception:
            ver = "0.4.0"

        data = {
            "version": ver,
            "queue_dir": str(self.queue_dir),
            "runs": [r.to_dict() for r in self.runs],
        }
        tmp_file = self.queue_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_file, self.queue_file)

        self.generate_standalone_script()

    def add_run(self, config: LoraRunConfig) -> LoraRunConfig:
        """Add a new run to the end of the FIFO queue and persist."""
        next_idx = len(self.runs) + 1
        config.sequence_index = next_idx
        if not config.is_custom_name:
            config.name = generate_deterministic_run_name(config, index=next_idx)
            config.adapter_path = f"adapters/{config.name}"
        self.runs.append(config)
        self.save()
        return config

    def delete_run(self, run_id: str) -> bool:
        """Remove a run from the queue by ID."""
        initial_len = len(self.runs)
        self.runs = [r for r in self.runs if r.id != run_id]
        if len(self.runs) != initial_len:
            cfg_path = self.configs_dir / f"{run_id}.yaml"
            if cfg_path.exists():
                try:
                    cfg_path.unlink()
                except OSError:
                    pass
            self.save()
            return True
        return False

    def clone_run(self, run_id: str) -> Optional[LoraRunConfig]:
        """Clone an existing run with a new UUID and next sequential index."""
        target = self.get_run(run_id)
        if not target:
            return None

        import copy
        cloned = LoraRunConfig.from_dict(copy.deepcopy(target.to_dict()))
        cloned.id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        cloned.status = "queued"
        cloned.started_at = None
        cloned.finished_at = None
        cloned.exit_code = None
        cloned.error_message = None
        cloned.log_file = None
        cloned.wandb_url = None

        next_idx = len(self.runs) + 1
        cloned.sequence_index = next_idx
        if not target.is_custom_name:
            cloned.name = generate_deterministic_run_name(cloned, index=next_idx)
            cloned.is_custom_name = False
        else:
            cloned.name = f"Copy of {target.name}"
            cloned.is_custom_name = True
        cloned.adapter_path = f"adapters/{cloned.name}"
        self.runs.append(cloned)
        self.save()
        return cloned

    def clear_queue(self, only_pending: bool = False) -> None:
        """Clear all runs or only pending runs."""
        if only_pending:
            self.runs = [r for r in self.runs if r.status in ("running", "completed")]
        else:
            self.runs = []
            if self.configs_dir.exists():
                for f in self.configs_dir.glob("*.yaml"):
                    try:
                        f.unlink()
                    except OSError:
                        pass
        self.save()

    def generate_queue_script(self) -> Path:
        """Alias for generate_standalone_script."""
        return self.generate_standalone_script()

    def get_run(self, run_id: str) -> Optional[LoraRunConfig]:
        for r in self.runs:
            if r.id == run_id:
                return r
        return None

    def update_run_status(
        self,
        run_id: str,
        status: str,
        exit_code: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update state of a specific run and persist immediately."""
        r = self.get_run(run_id)
        if r:
            r.status = status
            if status == "running" and not r.started_at:
                r.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
            elif status in ("completed", "failed", "cancelled"):
                r.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
            if exit_code is not None:
                r.exit_code = exit_code
            if error_message is not None:
                r.error_message = error_message
            self.save()

    def get_pending_runs(self) -> List[LoraRunConfig]:
        return [r for r in self.runs if r.status == "queued"]

    def generate_standalone_script(self) -> Path:
        """Generate an executable bash script to run the entire queue sequentially."""
        lines = [
            "#!/usr/bin/env bash",
            "# Auto-generated by MLX Commander",
            "# Runs all queued LoRA fine-tuning jobs sequentially on Apple Silicon",
            "set -e",
            "",
            "echo '==================================================='",
            "echo '      MLX Commander :: Sequential Queue Runner     '",
            "echo '==================================================='",
            "",
        ]
        for idx, r in enumerate(self.runs, 1):
            cfg_file = self.configs_dir / f"{r.id}.yaml"
            log_file = self.logs_dir / f"{r.id}.log"
            lines.extend([
                "echo ''",
                f"echo '▶ [{idx}/{len(self.runs)}] Running: {r.name}'",
                f"echo '  Model:  {r.model}'",
                f"echo '  Data:   {r.data}'",
                f"echo '  Config: {cfg_file}'",
                f"echo '  Log:    {log_file}'",
                "echo '---------------------------------------------------'",
                f'mlx_lm.lora --config "{cfg_file}" 2>&1 | tee "{log_file}"',
                f'python3 -m mlx_commander.lora.runner --rename-adapters "{r.adapter_path}" --config "{cfg_file}" 2>&1 | tee -a "{log_file}"',
                f"echo '✔ [{idx}/{len(self.runs)}] Completed: {r.name}'",
            ])
        lines.extend([
            "",
            "echo ''",
            "echo '==================================================='",
            "echo '  All queued fine-tuning runs completed successfully!'",
            "echo '==================================================='",
        ])
        with open(self.script_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        self.script_file.chmod(0o755)
        return self.script_file
