"""
Weights & Biases (W&B) Experiment Tracking Integration for MLX Commander.
Automatically detects user authentication, initializes experiment tracking,
and streams real-time loss & throughput metrics from mlx_lm.lora runs.
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import LoraRunConfig, sanitize_model_slug
from .estimator import get_apple_silicon_chip, get_hardware_memory_bytes


def is_wandb_available() -> bool:
    """Return True if the 'wandb' package is installed."""
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False


import functools


@functools.lru_cache(maxsize=16)
def _check_wandb_logged_in_cached(
    env_key: Optional[str],
    env_entity: Optional[str],
    netrc_mtime: Optional[float],
) -> Tuple[bool, Optional[str]]:
    # 1. Environment variable
    if env_key:
        entity = env_entity
        if not entity:
            try:
                import wandb
                entity = getattr(wandb.setup().settings, "entity", None)
            except Exception:
                pass
        return True, entity or "env_user"

    # 2. Check ~/.netrc
    if netrc_mtime is not None:
        try:
            import netrc as py_netrc
            auths = py_netrc.netrc().authenticators("api.wandb.ai")
            if auths:
                login_user = auths[0]
                if login_user and login_user != "user":
                    return True, login_user
                try:
                    import wandb
                    entity = getattr(wandb.setup().settings, "entity", None)
                    if entity:
                        return True, entity
                except Exception:
                    pass
                return True, "wandb_user"
        except Exception:
            try:
                netrc_path = Path("~/.netrc").expanduser()
                with open(netrc_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if "api.wandb.ai" in content:
                    return True, "wandb_user"
            except Exception:
                pass

    # 3. Check wandb API credentials directly if library is installed
    if is_wandb_available():
        try:
            import wandb
            settings = wandb.setup().settings
            if getattr(settings, "api_key", None):
                return True, getattr(settings, "entity", None) or "wandb_user"
        except Exception:
            pass

    return False, None


def is_wandb_logged_in() -> Tuple[bool, Optional[str]]:
    """
    Check if the user is authenticated with Weights & Biases on this system.
    Returns (is_logged_in, entity_or_username).
    Completely eliminates synchronous network calls to prevent TUI rendering latency.
    """
    env_key = os.environ.get("WANDB_API_KEY")
    env_entity = os.environ.get("WANDB_ENTITY")
    netrc_mtime: Optional[float] = None
    try:
        netrc_p = Path("~/.netrc").expanduser()
        if netrc_p.exists():
            netrc_mtime = netrc_p.stat().st_mtime
    except Exception:
        pass

    return _check_wandb_logged_in_cached(env_key, env_entity, netrc_mtime)


def clear_wandb_login_cache() -> None:
    """Clear cached W&B login status check."""
    _check_wandb_logged_in_cached.cache_clear()


def parse_mlx_log_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse real-time training and validation stdout lines from mlx_lm.lora.
    Supported patterns:
      'Iter 100: Train loss 1.450, Learning Rate 1.000e-05, It/sec 1.150, Tokens/sec 1200.0'
      'Iter 200: Val loss 2.100, Val It/sec 2.000'
      'Iter 50: Train loss 0.982'
    """
    if not line or "Iter" not in line:
        return None

    # Match iteration index
    iter_match = re.search(r'Iter\s+(\d+):', line)
    if not iter_match:
        return None

    step = int(iter_match.group(1))
    metrics: Dict[str, Any] = {"iter": step}

    # Train loss
    train_loss_m = re.search(r'Train loss\s+([0-9.]+)', line)
    if train_loss_m:
        try:
            metrics["train/loss"] = float(train_loss_m.group(1))
        except ValueError:
            pass

    # Val loss
    val_loss_m = re.search(r'Val loss\s+([0-9.]+)', line)
    if val_loss_m:
        try:
            metrics["val/loss"] = float(val_loss_m.group(1))
        except ValueError:
            pass

    # Learning rate
    lr_m = re.search(r'Learning Rate\s+([0-9.eE+-]+)', line)
    if lr_m:
        try:
            metrics["train/learning_rate"] = float(lr_m.group(1))
        except ValueError:
            pass

    # Iterations per second (train or val)
    it_sec_m = re.search(r'(?<!Val\s)It/sec\s+([0-9.]+)', line)
    if it_sec_m:
        try:
            metrics["train/it_per_sec"] = float(it_sec_m.group(1))
        except ValueError:
            pass

    val_it_sec_m = re.search(r'Val It/sec\s+([0-9.]+)', line)
    if val_it_sec_m:
        try:
            metrics["val/it_per_sec"] = float(val_it_sec_m.group(1))
        except ValueError:
            pass

    # Tokens per second
    tok_sec_m = re.search(r'Tokens/sec\s+([0-9.]+)', line)
    if tok_sec_m:
        try:
            metrics["train/tokens_per_sec"] = float(tok_sec_m.group(1))
        except ValueError:
            pass

    if len(metrics) > 1:
        return metrics
    return None


class WandbTracker:
    """
    Safely manages a Weights & Biases tracking run.
    Guarantees zero-crash operation: any W&B API or network issues
    will never interrupt active MLX fine-tuning.
    """

    def __init__(self, project: Optional[str] = None, enabled: bool = True):
        self.project = project or "mlx-commander"
        self.enabled = enabled
        self.run = None
        self.run_url: Optional[str] = None
        self._is_active = False

    def start_run(
        self,
        config: LoraRunConfig,
        implied_epochs: Optional[float] = None,
    ) -> bool:
        """Initialize a Weights & Biases run for the given LoRA configuration."""
        if not self.enabled or not is_wandb_available():
            return False

        logged_in, entity = is_wandb_logged_in()
        if not logged_in:
            return False

        try:
            import wandb

            wandb_config = config.to_dict()
            wandb_config["hardware_chip"] = get_apple_silicon_chip()
            wandb_config["hardware_memory_gb"] = round(get_hardware_memory_bytes() / (1024 ** 3), 1)
            if implied_epochs is not None:
                wandb_config["implied_epochs"] = round(implied_epochs, 2)

            tags = [
                "mlx",
                "apple-silicon",
                config.fine_tune_type,
                sanitize_model_slug(config.model),
            ]

            proj_name = config.wandb_project or self.project or "mlx-commander"
            self.run = wandb.init(
                project=proj_name,
                name=config.name,
                config=wandb_config,
                tags=tags,
                reinit=True,
            )
            self._is_active = True
            try:
                self.run_url = self.run.get_url()
            except Exception:
                self.run_url = getattr(self.run, "url", None)

            print(f"[W&B] Tracking run at: {self.run_url}")
            return True
        except Exception as e:
            sys.stderr.write(f"[W&B Warning] Failed to initialize Weights & Biases tracking: {e}\n")
            self._is_active = False
            return False

    def log_line(self, line: str) -> None:
        """Parse stdout line and log metrics to W&B if available."""
        if not self._is_active or not self.run:
            return

        metrics = parse_mlx_log_line(line)
        if metrics:
            try:
                import wandb
                step = metrics.pop("iter", None)
                wandb.log(metrics, step=step)
            except Exception:
                pass

    def log_eval(self, eval_result: Dict[str, Any]) -> None:
        """Log generative test set evaluation metrics and prediction table to Weights & Biases."""
        if not self._is_active or not self.run or not eval_result:
            return

        try:
            import wandb
            summary = eval_result.get("summary", {})
            predictions = eval_result.get("predictions", [])

            # 1. Log scalar summary metrics
            eval_scalars = {
                "eval/exact_match_pct": summary.get("exact_match_pct", 0.0),
                "eval/substring_match_pct": summary.get("substring_match_pct", 0.0),
                "eval/word_f1": summary.get("avg_word_f1", 0.0),
                "eval/word_precision": summary.get("avg_word_prec", 0.0),
                "eval/word_recall": summary.get("avg_word_recall", 0.0),
                "eval/fixed_count": summary.get("fixed_count", 0),
                "eval/regressed_count": summary.get("regressed_count", 0),
                "eval/tokens_per_sec": summary.get("tokens_per_sec", 0.0),
            }
            wandb.log(eval_scalars)

            # Update run summary dictionary
            for k, v in eval_scalars.items():
                self.run.summary[k] = v

            # 2. Log sample predictions table
            if predictions:
                columns = ["id", "prompt", "golden", "baseline", "model_output", "status", "word_f1", "exact_match"]
                table_data = []
                for p in predictions:
                    table_data.append([
                        p.get("id", 0),
                        p.get("prompt", ""),
                        p.get("golden", ""),
                        p.get("baseline_output", ""),
                        p.get("model_output", ""),
                        p.get("status", ""),
                        p.get("word_f1", 0.0),
                        p.get("exact_match_norm", False),
                    ])
                table = wandb.Table(columns=columns, data=table_data)
                wandb.log({"eval/predictions_table": table})

            # 3. Log categorical confusion matrix if available
            cm_data = summary.get("confusion_matrix")
            if cm_data and cm_data.get("is_categorical") and predictions:
                y_true = [p.get("golden", "") for p in predictions]
                preds = [p.get("model_output", "") for p in predictions]
                cm = wandb.plot.confusion_matrix(
                    y_true=y_true,
                    preds=preds,
                    class_names=cm_data.get("classes", []),
                )
                wandb.log({"eval/confusion_matrix": cm})

        except Exception as e:
            sys.stderr.write(f"[W&B Warning] Failed to log evaluation to Weights & Biases: {e}\n")

    def finish_run(self, exit_code: int = 0) -> Optional[str]:
        """Finish the W&B run, recording exit code and returning the run URL."""
        if not self._is_active or not self.run:
            return None

        try:
            import wandb
            self.run.summary["exit_code"] = exit_code
            self.run.summary["status"] = "completed" if exit_code == 0 else "failed"
            url = self.run_url
            wandb.finish(exit_code=exit_code)
            self._is_active = False
            return url
        except Exception as e:
            sys.stderr.write(f"[W&B Warning] Failed to close run cleanly: {e}\n")
            self._is_active = False
            return self.run_url
