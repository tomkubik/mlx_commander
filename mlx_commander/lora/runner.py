"""
Sequential Execution Engine for MLX Commander LoRA Queue.
Executes queued fine-tuning runs one-by-one, streams real-time logs,
tracks progress in queue.json, and fires desktop notifications upon completion.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .config import LoraRunConfig, format_adapter_filename
from .queue import QueueManager
from .tracking import WandbTracker


def rename_saved_adapters(
    adapter_dir: Union[str, Path],
    config: Optional[LoraRunConfig] = None,
    final_iters: Optional[int] = None,
    create_symlinks: bool = True,
) -> List[Tuple[Path, Path]]:
    """
    Scan adapter_dir for saved MLX adapter checkpoint files and rename them
    so that each filename starts with the iteration count at which it was saved,
    and has all training hyperparameters appended to the name.

    Maintains relative symlinks (e.g. 'adapters.safetensors' and
    '0000100_adapters.safetensors') pointing to the renamed files so native
    MLX evaluation, generation, and resume workflows remain 100% compatible.

    Returns:
        List of (original_path, new_path) tuples that were renamed.
    """
    adir = Path(adapter_dir)
    if not adir.exists() or not adir.is_dir():
        return []

    # If config is None, attempt to load minimal config from adapter_config.json if present
    if config is None and (adir / "adapter_config.json").is_file():
        try:
            with open(adir / "adapter_config.json", "r", encoding="utf-8") as acf:
                ac_data = json.load(acf)
            lora_p = ac_data.get("lora_parameters", {})
            config = LoraRunConfig(
                model=ac_data.get("model", "model"),
                lora_rank=lora_p.get("rank", 8),
                lora_alpha=lora_p.get("scale", 16.0),
                lora_dropout=lora_p.get("dropout", 0.0),
            )
        except Exception:
            pass

    renamed: List[Tuple[Path, Path]] = []

    # 1. Step Checkpoints: e.g. 0000100_adapters.safetensors
    # Strictly match <digits>_adapters.safetensors (not already renamed with hyperparameters)
    step_pattern = re.compile(r"^(\d+)_adapters\.safetensors$")
    try:
        items = sorted(list(adir.iterdir()))
    except OSError:
        return []

    for item in items:
        if item.is_symlink():
            continue
        m = step_pattern.match(item.name)
        if m:
            step = int(m.group(1))
            target_name = format_adapter_filename(step=step, config=config)
            target_path = adir / target_name
            if target_path != item:
                if target_path.exists():
                    try:
                        target_path.unlink()
                    except OSError:
                        pass
                try:
                    item.rename(target_path)
                    renamed.append((item, target_path))

                    if create_symlinks:
                        try:
                            # Create relative symlink: item.name -> target_name
                            item.symlink_to(target_name)
                        except OSError:
                            pass
                except OSError:
                    pass

    # 2. Final Adapter Weights: adapters.safetensors
    final_file = adir / "adapters.safetensors"
    if final_file.exists() and not final_file.is_symlink():
        step = final_iters if final_iters is not None else (config.iters if config else 1000)
        target_name = format_adapter_filename(step=step, config=config)
        target_path = adir / target_name

        if target_path != final_file:
            try:
                if target_path.exists():
                    final_file.unlink()
                else:
                    final_file.rename(target_path)
                    renamed.append((final_file, target_path))

                if create_symlinks:
                    try:
                        final_file.symlink_to(target_name)
                    except OSError:
                        pass
            except OSError:
                pass

    return renamed


def notify_macos(title: str, message: str) -> None:
    """Fire a native macOS desktop notification via AppleScript."""
    try:
        script = f'display notification "{message}" with title "{title}"'
        subprocess.run(["osascript", "-e", script], capture_output=True, check=False)
    except Exception:
        pass


def execute_single_run(
    mgr: QueueManager,
    run_id: str,
    wandb_project: Optional[str] = None,
    enable_wandb: bool = True,
) -> int:
    """Execute a single LoRA run by ID, streaming output to console, log file, and Weights & Biases."""
    r = mgr.get_run(run_id)
    if not r:
        return 1

    cfg_file = mgr.configs_dir / f"{r.id}.yaml"
    log_file = mgr.logs_dir / f"{r.id}.log"

    # Initialize Weights & Biases tracking if enabled and authenticated
    tracker = WandbTracker(project=wandb_project or r.wandb_project, enabled=enable_wandb)
    tracker.start_run(r)

    mgr.update_run_status(run_id, "running")
    print(f"\n[MLX Commander] Starting run: {r.name}")
    print(f"  Model:      {r.model}")
    print(f"  Data:       {r.data}")
    print(f"  Config:     {cfg_file}")
    print(f"  Output Log: {log_file}")
    if tracker.run_url:
        print(f"  W&B Run:    {tracker.run_url}")
    print("-" * 60)

    cmd = [sys.executable, "-m", "mlx_lm.lora", "--config", str(cfg_file)]

    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"=== MLX Commander LoRA Run: {r.name} ===\n")
        lf.write(f"Command: {' '.join(cmd)}\n\n")
        lf.flush()

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in iter(proc.stdout.readline, ""):
                sys.stdout.write(line)
                sys.stdout.flush()
                lf.write(line)
                lf.flush()
                tracker.log_line(line)
                if "save" in line.lower() or "saved" in line.lower() or "iter" in line.lower():
                    try:
                        new_renamed = rename_saved_adapters(r.adapter_path, r)
                        for orig, dest in new_renamed:
                            msg = f"  [Adapter Checkpoint] Saved: {dest.name}\n"
                            sys.stdout.write(msg)
                            sys.stdout.flush()
                            lf.write(msg)
                            lf.flush()
                    except Exception:
                        pass
            proc.stdout.close()
            code = proc.wait()
        except FileNotFoundError:
            try:
                proc = subprocess.Popen(
                    ["mlx_lm.lora", "--config", str(cfg_file)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in iter(proc.stdout.readline, ""):
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    lf.write(line)
                    lf.flush()
                    tracker.log_line(line)
                    if "save" in line.lower() or "saved" in line.lower() or "iter" in line.lower():
                        try:
                            new_renamed = rename_saved_adapters(r.adapter_path, r)
                            for orig, dest in new_renamed:
                                msg = f"  [Adapter Checkpoint] Saved: {dest.name}\n"
                                sys.stdout.write(msg)
                                sys.stdout.flush()
                                lf.write(msg)
                                lf.flush()
                        except Exception:
                            pass
                proc.stdout.close()
                code = proc.wait()
            except Exception as e:
                err_str = f"Failed to spawn mlx_lm.lora: {e}"
                sys.stderr.write(err_str + "\n")
                lf.write(err_str + "\n")
                code = 127
        except Exception as e:
            err_str = f"Execution error: {e}"
            sys.stderr.write(err_str + "\n")
            lf.write(err_str + "\n")
            code = 1

    # Final pass to ensure all saved adapters (intermediate checkpoints & final weights) have hyperparameters
    try:
        final_renamed = rename_saved_adapters(r.adapter_path, r, final_iters=r.iters)
        for orig, dest in final_renamed:
            msg = f"  [Adapter Saved] {orig.name} -> {dest.name}\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(msg)
                lf.flush()
    except Exception:
        pass

    # Run Generative Evaluation on test split if enabled
    if code == 0 and getattr(r, "run_eval", True):
        ds_test = Path(r.data) / "test.jsonl" if Path(r.data).is_dir() else Path(r.data)
        if ds_test.exists():
            print("\n[MLX Commander] Running Generative Evaluation on test set...")
            try:
                from ..evals.runner import run_generative_eval
                eval_res = run_generative_eval(r)
                if eval_res and "summary" in eval_res:
                    summ = eval_res["summary"]
                    tracker.log_eval(eval_res)
                    print(f"  • Exact Match:      {summ.get('exact_match_pct', 0.0):.1f}%")
                    print(f"  • Substring Match:  {summ.get('substring_match_pct', 0.0):.1f}%")
                    print(f"  • Word F1 Score:    {summ.get('avg_word_f1', 0.0):.4f}")
                    print(f"  • Fixed / Regressed: {summ.get('fixed_count', 0)} / {summ.get('regressed_count', 0)}")
                    print(f"  • HTML Report:      {eval_res.get('html_dashboard', '')}")
            except Exception as e:
                print(f"  [Eval Warning] Generative eval failed: {e}")
        else:
            print(f"  [MLX Commander] Skipping generative eval: no test.jsonl found in '{r.data}'.")

    wandb_url = tracker.finish_run(exit_code=code)
    if wandb_url:
        r.wandb_url = wandb_url
        mgr.save()
        print(f"  [W&B] Run tracked at: {wandb_url}")

    if code == 0:
        mgr.update_run_status(run_id, "completed", exit_code=0)
        print(f"\n[OK] Run '{r.name}' completed successfully.")
    else:
        mgr.update_run_status(run_id, "failed", exit_code=code, error_message=f"Exited with code {code}")
        print(f"\n[FAIL] Run '{r.name}' failed with code {code}.")

    return code


def run_lora_queue(
    queue_dir: Optional[Path] = None,
    stop_on_failure: bool = False,
    wandb_project: Optional[str] = None,
    enable_wandb: bool = True,
) -> int:
    """
    Run all queued runs sequentially.
    Safe for Apple Silicon: one model trains at a time, preventing memory thrashing.
    """
    mgr = QueueManager(queue_dir)
    pending = mgr.get_pending_runs()

    if not pending:
        print("[MLX Commander] No queued runs found in", mgr.queue_dir)
        return 0

    total_runs = len(pending)
    print(f"\n===================================================")
    print(f"   MLX Commander :: Starting {total_runs} Queued LoRA Run(s)")
    print(f"   Directory: {mgr.queue_dir}")
    print(f"===================================================")

    completed_count = 0
    failed_count = 0

    for idx, r in enumerate(pending, 1):
        print(f"\n>>> Processing Job [{idx}/{total_runs}]: {r.name}")
        code = execute_single_run(mgr, r.id, wandb_project=wandb_project, enable_wandb=enable_wandb)
        if code == 0:
            completed_count += 1
        else:
            failed_count += 1
            if stop_on_failure:
                print(f"Stopping queue early due to failure in '{r.name}'.")
                break

    print(f"\n===================================================")
    print(f"  Queue Execution Finished!")
    print(f"  Total: {total_runs} | Completed: {completed_count} | Failed: {failed_count}")
    print(f"===================================================")

    notify_macos(
        "MLX Commander",
        f"Finished {total_runs} fine-tuning run(s): {completed_count} succeeded, {failed_count} failed.",
    )
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MLX Commander LoRA runner and adapter utility.")
    parser.add_argument("--rename-adapters", type=str, help="Directory containing saved adapters to rename.")
    parser.add_argument("--config", type=str, help="Path to run YAML config file.")
    parser.add_argument("--iters", type=int, default=None, help="Total training iterations.")
    cli_args = parser.parse_args()

    if cli_args.rename_adapters:
        cfg = None
        if cli_args.config and Path(cli_args.config).exists():
            cfg = LoraRunConfig.from_yaml(cli_args.config)
        res = rename_saved_adapters(cli_args.rename_adapters, config=cfg, final_iters=cli_args.iters)
        for orig, dest in res:
            print(f"[OK] Renamed: {orig.name} -> {dest.name}")
