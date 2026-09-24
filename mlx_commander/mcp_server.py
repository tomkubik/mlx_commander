"""
Model Context Protocol (MCP) Server for MLX Commander.
Exposes dataset inspection, pre-populated interactive TUI launching,
and headless conversion tools to AI agents (Claude Desktop, Cursor, Antigravity, Cline, Zed).
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlx_commander import __version__
from mlx_commander.converter import convert_and_save
from mlx_commander.evals.runner import run_generative_eval
from mlx_commander.formats import ColumnMapping, MLXFormat, auto_detect_mapping, validate_mapping
from mlx_commander.loader import load_local_dataset
from mlx_commander.lora.config import LoraRunConfig
from mlx_commander.lora.estimator import (
    calculate_implied_epochs,
    estimate_duration,
    estimate_eval_throughput,
    estimate_peak_memory,
)
from mlx_commander.lora.model_info import detect_model_engine
from mlx_commander.lora.multi_config import MultiLoraRunConfig
from mlx_commander.lora.queue import QueueManager
from mlx_commander.lora.vlm_safeguards import (
    calculate_vlm_batch_tweak,
    calculate_vlm_sweep_tweak,
)
from mlx_commander.splitter import SplitConfig, generate_random_seed
from mlx_commander.terminal_spawner import is_macos, spawn_terminal_tui


def inspect_dataset_tool(dataset_path: str) -> Dict[str, Any]:
    """Inspect a dataset file or folder and return schema, rows, splits, and suggested mapping."""
    try:
        ds = load_local_dataset(dataset_path)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load dataset: {e}"}

    sample_records = ds.get_all_records()[:3]

    # Auto-detect suggested mappings for each format
    suggested_mappings = {}
    for fmt in MLXFormat:
        m = auto_detect_mapping(fmt, ds.columns)
        errs = validate_mapping(fmt, m, ds.columns)
        suggested_mappings[fmt.value] = {
            "mapping": {
                k: v for k, v in m.__dict__.items() if v
            },
            "is_valid": len(errs) == 0,
            "validation_errors": errs,
        }

    return {
        "status": "success",
        "dataset_path": ds.source_path,
        "total_rows": ds.total_rows,
        "columns": ds.columns,
        "splits": ds.split_names,
        "split_counts": ds.split_counts,
        "sample_records": sample_records,
        "suggested_mappings": suggested_mappings,
    }


def launch_conversion_tui_tool(
    dataset_path: str,
    format: str = "prompt_completion",
    prompt_col: Optional[str] = None,
    completion_col: Optional[str] = None,
    messages_col: Optional[str] = None,
    text_col: Optional[str] = None,
    text_template: Optional[str] = None,
    user_col: Optional[str] = None,
    assistant_col: Optional[str] = None,
    system_col: Optional[str] = None,
    chosen_col: Optional[str] = None,
    rejected_col: Optional[str] = None,
    train_pct: float = 80.0,
    valid_pct: float = 10.0,
    test_pct: float = 10.0,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pre-populates and launches the MLX Commander interactive TUI in a macOS Terminal window.
    The user can review live JSONL preview, adjust settings with arrow keys, and press Convert.
    Synchronously awaits completion and returns the conversion manifest.
    """
    if not is_macos():
        return {
            "status": "error",
            "message": "Interactive TUI spawning is supported on macOS. For Linux or headless environments, use convert_dataset_headless.",
        }

    args = [
        "--tui",
        "--dataset", dataset_path,
        "--format", format,
        "--train", str(train_pct),
        "--valid", str(valid_pct),
        "--test", str(test_pct),
    ]
    if prompt_col:
        args.extend(["--prompt-col", prompt_col])
    if completion_col:
        args.extend(["--completion-col", completion_col])
    if messages_col:
        args.extend(["--messages-col", messages_col])
    if text_col:
        args.extend(["--text-col", text_col])
    if text_template:
        args.extend(["--text-template", text_template])
    if user_col:
        args.extend(["--user-col", user_col])
    if assistant_col:
        args.extend(["--assistant-col", assistant_col])
    if system_col:
        args.extend(["--system-col", system_col])
    if chosen_col:
        args.extend(["--chosen-col", chosen_col])
    if rejected_col:
        args.extend(["--rejected-col", rejected_col])
    if seed is not None:
        args.extend(["--seed", str(seed)])
    if output_dir:
        args.extend(["--output", output_dir])

    # Determine manifest destination
    p = Path(dataset_path).resolve()
    base_dir = p if p.is_dir() else p.parent
    target_out = Path(output_dir).resolve() if output_dir else (base_dir / "mlx_dataset")
    target_out.mkdir(parents=True, exist_ok=True)
    manifest_file = target_out / "mlx_manifest.json"

    ec = spawn_terminal_tui(args, manifest_path=str(manifest_file))
    if ec == 0 and manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"status": "error", "message": f"Failed reading manifest: {e}"}
    elif ec == 130:
        return {"status": "cancelled", "message": "User cancelled or closed the TUI session without converting."}
    else:
        return {"status": "error", "message": f"TUI session exited with status code {ec}."}


def convert_dataset_headless_tool(
    dataset_path: str,
    format: str = "prompt_completion",
    prompt_col: Optional[str] = None,
    completion_col: Optional[str] = None,
    messages_col: Optional[str] = None,
    text_col: Optional[str] = None,
    text_template: Optional[str] = None,
    user_col: Optional[str] = None,
    assistant_col: Optional[str] = None,
    system_col: Optional[str] = None,
    chosen_col: Optional[str] = None,
    rejected_col: Optional[str] = None,
    train_pct: float = 80.0,
    valid_pct: float = 10.0,
    test_pct: float = 10.0,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Perform automated headless conversion to MLX JSONL without opening the TUI."""
    try:
        ds = load_local_dataset(dataset_path)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load dataset: {e}"}

    try:
        target_fmt = MLXFormat(format.lower().strip())
    except ValueError:
        return {"status": "error", "message": f"Invalid format '{format}'. Must be one of: prompt_completion, chat, text, dpo"}

    mapping = auto_detect_mapping(target_fmt, ds.columns)

    if prompt_col:
        mapping.prompt_col = prompt_col
    if completion_col:
        mapping.completion_col = completion_col
    if messages_col:
        mapping.messages_col = messages_col
    if text_col:
        mapping.text_col = text_col
    if text_template:
        mapping.text_template = text_template
    if user_col:
        mapping.user_col = user_col
    if assistant_col:
        mapping.assistant_col = assistant_col
    if system_col:
        mapping.system_col = system_col
    if chosen_col:
        mapping.chosen_col = chosen_col
    if rejected_col:
        mapping.rejected_col = rejected_col

    split_cfg = SplitConfig(
        train_pct=train_pct,
        valid_pct=valid_pct,
        test_pct=test_pct,
        seed=seed if seed is not None else generate_random_seed(),
    )

    out = output_dir or str(ds.default_output_dir)
    try:
        res = convert_and_save(
            dataset=ds,
            format_type=target_fmt,
            mapping=mapping,
            output_dir=out,
            split_config=split_cfg,
        )
        return res.to_manifest_dict()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def estimate_fine_tuning_resources_tool(
    model: str,
    iters: int = 600,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    max_seq_length: int = 2048,
    lora_rank: int = 8,
    grad_checkpoint: bool = False,
    total_train_records: int = 0,
) -> Dict[str, Any]:
    """
    Estimate Unified Memory (RAM/VRAM) footprint, implied training epochs,
    and wall-clock training duration for an Apple MLX LoRA run.
    """
    try:
        cfg = LoraRunConfig(
            model=model,
            iters=iters,
            batch_size=batch_size,
            grad_accumulation_steps=gradient_accumulation_steps,
            max_seq_length=max_seq_length,
            lora_rank=lora_rank,
            grad_checkpoint=grad_checkpoint,
        )
        mem_info = estimate_peak_memory(cfg)
        dur_info = estimate_duration(cfg)
        epochs = (
            calculate_implied_epochs(
                iters=iters,
                batch_size=batch_size,
                total_train_records=total_train_records,
                grad_accumulation_steps=gradient_accumulation_steps,
            )
            if total_train_records > 0
            else None
        )

        return {
            "status": "success",
            "model": model,
            "peak_memory_gb": mem_info.get("peak_gb"),
            "total_ram_gb": mem_info.get("total_ram_gb"),
            "safety_tier": mem_info.get("safety_tier"),
            "safety_badge": mem_info.get("badge"),
            "recommendations": mem_info.get("recommendations", []),
            "implied_epochs": epochs,
            "estimated_duration_sec": dur_info.get("total_seconds"),
            "estimated_duration_str": dur_info.get("formatted_duration"),
            "estimated_clock_eta": dur_info.get("eta_str"),
            "chip": dur_info.get("chip"),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def queue_single_run_tool(
    model: str,
    data_path: str,
    name: Optional[str] = None,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1e-4,
    iters: int = 600,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    max_seq_length: int = 2048,
    num_layers: int = 16,
    grad_checkpoint: bool = False,
    mask_prompt: bool = True,
    run_eval: bool = False,
    engine: str = "auto",
    queue_dir: str = "mlx_runs",
) -> Dict[str, Any]:
    """
    Configure and enqueue a single Apple MLX fine-tuning run into the sequential execution queue.
    Automatically applies VLM attention mask safeguards if multimodal model is detected.
    """
    try:
        resolved_engine = engine.strip().lower()
        if resolved_engine == "auto":
            resolved_engine = detect_model_engine(model_name_or_path=model)

        eff_batch = batch_size * gradient_accumulation_steps
        vlm_tweak_applied = False
        if resolved_engine == "mlx_vlm" and batch_size > 1:
            batch_size, gradient_accumulation_steps = calculate_vlm_batch_tweak(batch_size, gradient_accumulation_steps)
            vlm_tweak_applied = True

        cfg = LoraRunConfig(
            model=model,
            data=data_path,
            name=name or "",
            batch_size=batch_size,
            grad_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            iters=iters,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            max_seq_length=max_seq_length,
            num_layers=num_layers,
            grad_checkpoint=grad_checkpoint,
            mask_prompt=mask_prompt,
            run_eval=run_eval,
            engine=resolved_engine,
        )
        if name:
            cfg.is_custom_name = True

        mgr = QueueManager(Path(queue_dir))
        added_cfg = mgr.add_run(cfg)

        mem_info = estimate_peak_memory(added_cfg)
        dur_info = estimate_duration(added_cfg)

        return {
            "status": "success",
            "run_id": added_cfg.id,
            "run_name": added_cfg.name,
            "sequence_index": added_cfg.sequence_index,
            "engine": resolved_engine,
            "vlm_safeguard_applied": vlm_tweak_applied,
            "batch_size": added_cfg.batch_size,
            "gradient_accumulation_steps": getattr(added_cfg, "grad_accumulation_steps", 1),
            "effective_batch_size": eff_batch,
            "adapter_path": added_cfg.adapter_path,
            "config_file": str(mgr.configs_dir / f"{added_cfg.id}.yaml"),
            "log_file": added_cfg.log_file,
            "run_eval": added_cfg.run_eval,
            "peak_ram_estimate_gb": mem_info.get("peak_gb"),
            "safety_badge": mem_info.get("badge"),
            "duration_estimate": dur_info.get("formatted_duration"),
            "queue_file": str(mgr.queue_file),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def queue_multi_run_sweep_tool(
    model: str,
    data_path: str,
    batch_size: Optional[List[int]] = None,
    gradient_accumulation_steps: Optional[List[int]] = None,
    learning_rate: Optional[List[float]] = None,
    iters: Optional[List[int]] = None,
    lora_rank: Optional[List[int]] = None,
    lora_alpha: Optional[List[int]] = None,
    max_seq_length: Optional[List[int]] = None,
    grad_checkpoint: Optional[List[bool]] = None,
    run_eval: Optional[List[bool]] = None,
    engine: str = "auto",
    queue_dir: str = "mlx_runs",
) -> Dict[str, Any]:
    """
    Expand a multi-value hyperparameter sweep grid (Cartesian product) and enqueue all
    resultant runs into the sequential execution queue.
    """
    try:
        resolved_engine = engine.strip().lower()
        if resolved_engine == "auto":
            resolved_engine = detect_model_engine(model_name_or_path=model)

        b_list = batch_size or [4]
        gas_list = gradient_accumulation_steps or [1]

        vlm_tweak_applied = False
        if resolved_engine == "mlx_vlm" and any(b > 1 for b in b_list):
            b_list, gas_list = calculate_vlm_sweep_tweak(b_list, gas_list)
            vlm_tweak_applied = True

        multi_cfg = MultiLoraRunConfig(
            model=model,
            data=data_path,
            batch_size=b_list,
            grad_accumulation_steps=gas_list,
            learning_rate=learning_rate or [1e-4],
            iters=iters or [600],
            lora_rank=lora_rank or [8],
            lora_alpha=lora_alpha or [16],
            max_seq_length=max_seq_length or [2048],
            grad_checkpoint=grad_checkpoint or [False],
            run_eval=run_eval or [False],
            engine=resolved_engine,
        )

        generated_runs = multi_cfg.generate_runs()
        mgr = QueueManager(Path(queue_dir))
        queued = []
        for r in generated_runs:
            added = mgr.add_run(r)
            queued.append({
                "id": added.id,
                "name": added.name,
                "batch_size": added.batch_size,
                "gradient_accumulation_steps": getattr(added, "grad_accumulation_steps", 1),
                "learning_rate": added.learning_rate,
                "lora_rank": added.lora_rank,
                "iters": added.iters,
                "run_eval": added.run_eval,
                "config_file": str(mgr.configs_dir / f"{added.id}.yaml"),
            })

        summary = multi_cfg.calculate_sweep_estimates(dataset_records=0)
        return {
            "status": "success",
            "total_runs": len(queued),
            "engine": resolved_engine,
            "vlm_safeguard_applied": vlm_tweak_applied,
            "peak_memory_gb": summary.get("max_peak_ram_gb"),
            "total_estimated_duration_sec": summary.get("total_duration_seconds"),
            "total_estimated_duration_str": summary.get("total_duration_str"),
            "queued_runs": queued,
            "queue_file": str(mgr.queue_file),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def inspect_queue_tool(queue_dir: str = "mlx_runs") -> Dict[str, Any]:
    """
    Inspect the status, configurations, and logs of all jobs in the sequential run queue.
    """
    try:
        mgr = QueueManager(Path(queue_dir))
        all_runs = mgr.runs
        pending = mgr.get_pending_runs()
        completed = [r for r in all_runs if r.status == "completed"]
        failed = [r for r in all_runs if r.status == "failed"]

        runs_info = []
        for r in all_runs:
            runs_info.append({
                "id": r.id,
                "name": r.name,
                "sequence_index": r.sequence_index,
                "status": r.status,
                "model": r.model,
                "engine": getattr(r, "engine", "mlx_lm"),
                "iters": r.iters,
                "batch_size": r.batch_size,
                "gradient_accumulation_steps": getattr(r, "grad_accumulation_steps", 1),
                "learning_rate": r.learning_rate,
                "lora_rank": r.lora_rank,
                "run_eval": getattr(r, "run_eval", False),
                "exit_code": r.exit_code,
                "config_file": str(mgr.configs_dir / f"{r.id}.yaml"),
                "log_file": r.log_file,
            })

        return {
            "status": "success",
            "queue_dir": str(mgr.queue_dir),
            "total_runs": len(all_runs),
            "pending_count": len(pending),
            "completed_count": len(completed),
            "failed_count": len(failed),
            "runs": runs_info,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def execute_queue_tool(
    queue_dir: str = "mlx_runs",
    spawn_terminal: bool = True,
) -> Dict[str, Any]:
    """
    Execute all pending runs in the queue sequentially.
    Safe for Apple Silicon: models train strictly one after another to prevent memory thrashing.
    """
    p = Path(queue_dir).resolve()
    if spawn_terminal:
        if not is_macos():
            return {
                "status": "error",
                "message": "spawn_terminal is only supported on macOS. Use spawn_terminal=False for headless execution.",
            }
        ec = spawn_terminal_tui(["--run-queue", str(p)])
        return {
            "status": "success" if ec == 0 else ("cancelled" if ec == 130 else "error"),
            "exit_code": ec,
            "message": "Queue execution completed in external macOS Terminal window.",
        }
    else:
        try:
            from mlx_commander.lora.runner import run_lora_queue
            ec = run_lora_queue(p)
            return {
                "status": "success" if ec == 0 else "failed",
                "exit_code": ec,
                "queue_dir": str(p),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


def run_test_evaluation_tool(
    model: str,
    adapter_path: str,
    data_path: str,
    max_tokens: int = 128,
    engine: str = "auto",
) -> Dict[str, Any]:
    """
    Execute real model inference evaluation on test.jsonl deterministically.
    Generates baseline and adapter predictions, calculates Exact Match, Substring,
    Word F1, categorical confusion matrices, and 2x2 migration/regression matrices.
    Persists leaderboard CSV, predictions JSONL, and offline HTML comparison dashboard.
    """
    try:
        resolved_engine = engine.strip().lower()
        if resolved_engine == "auto":
            resolved_engine = detect_model_engine(model_name_or_path=model)

        cfg = LoraRunConfig(
            model=model,
            adapter_path=adapter_path,
            data=data_path,
            engine=resolved_engine,
            run_eval=True,
        )
        res = run_generative_eval(
            config=cfg,
            data_path=Path(data_path),
            max_tokens=max_tokens,
        )
        return res
    except Exception as e:
        return {"status": "error", "message": str(e)}


def estimate_test_eval_throughput_tool(
    model_name: str,
    total_samples: int = 50,
    num_passes: int = 2,
    max_tokens: int = 128,
) -> Dict[str, Any]:
    """
    Calculate model-dependent generation throughput (tok/s), latency per sample,
    and wall-clock duration for generative evaluation on Apple Silicon based on
    parameter count, quantization, and hardware memory bandwidth physics.
    """
    try:
        res = estimate_eval_throughput(
            model_name=model_name,
            total_samples=total_samples,
            num_passes=num_passes,
            max_tokens=max_tokens,
        )
        return {"status": "success", **res}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def launch_lora_tui_tool(
    mode: int = 2,
    dataset_path: Optional[str] = None,
    model: Optional[str] = None,
    queue_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Launch the interactive MLX Commander TUI directly in Mode 2 (Single Run)
    or Mode 3 (Multi-Run Sweeps) in an external macOS Terminal.app window.
    """
    if not is_macos():
        return {
            "status": "error",
            "message": "Interactive TUI spawning is supported on macOS.",
        }

    args = ["--tui"]
    if mode == 3:
        args.append("--multi-run")
    else:
        args.append("--lora")

    if dataset_path:
        args.extend(["--dataset", dataset_path])
    if model:
        args.extend(["--model", model])
    if queue_dir:
        args.extend(["--queue-dir", queue_dir])

    ec = spawn_terminal_tui(args)
    if ec == 0:
        return {"status": "success", "message": "TUI session finished."}
    elif ec == 130:
        return {"status": "cancelled", "message": "User closed the TUI session."}
    else:
        return {"status": "error", "message": f"TUI exited with code {ec}."}


def create_mcp_server():
    """Build FastMCP server with registered tools and prompts."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(
        name="mlx_commander",
        instructions=(
            "MLX Commander: Comprehensive tool suite for inspecting and converting Hugging Face datasets "
            "and orchestrating Apple MLX fine-tuning runs (single runs, multi-run sweeps, resource estimation, "
            "sequential queue execution, and generative test set evaluations)."
        ),
    )

    @server.tool()
    def inspect_dataset(dataset_path: str) -> dict:
        """Inspect a local Hugging Face dataset (parquet, jsonl, arrow, csv, sqlite)
        and return column names, row counts, detected format, and sample records."""
        return inspect_dataset_tool(dataset_path)

    @server.tool()
    def launch_conversion_tui(
        dataset_path: str,
        format: str = "prompt_completion",
        prompt_col: Optional[str] = None,
        completion_col: Optional[str] = None,
        messages_col: Optional[str] = None,
        train_pct: float = 80.0,
        valid_pct: float = 10.0,
        test_pct: float = 10.0,
        output_dir: Optional[str] = None,
    ) -> dict:
        """Launch the MLX Commander persistent Norton Commander TUI dashboard in a macOS Terminal window
        with pre-populated settings for user visual confirmation and live JSONL preview. Returns conversion manifest."""
        return launch_conversion_tui_tool(
            dataset_path=dataset_path,
            format=format,
            prompt_col=prompt_col,
            completion_col=completion_col,
            messages_col=messages_col,
            train_pct=train_pct,
            valid_pct=valid_pct,
            test_pct=test_pct,
            output_dir=output_dir,
        )

    @server.tool()
    def convert_dataset_headless(
        dataset_path: str,
        format: str = "prompt_completion",
        prompt_col: Optional[str] = None,
        completion_col: Optional[str] = None,
        messages_col: Optional[str] = None,
        train_pct: float = 80.0,
        valid_pct: float = 10.0,
        test_pct: float = 10.0,
        output_dir: Optional[str] = None,
    ) -> dict:
        """Directly convert a dataset to MLX JSONL in the background without UI interaction."""
        return convert_dataset_headless_tool(
            dataset_path=dataset_path,
            format=format,
            prompt_col=prompt_col,
            completion_col=completion_col,
            messages_col=messages_col,
            train_pct=train_pct,
            valid_pct=valid_pct,
            test_pct=test_pct,
            output_dir=output_dir,
        )

    @server.tool()
    def estimate_fine_tuning_resources(
        model: str,
        iters: int = 600,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        max_seq_length: int = 2048,
        lora_rank: int = 8,
        grad_checkpoint: bool = False,
        total_train_records: int = 0,
    ) -> dict:
        """Estimate peak Unified Memory (RAM/VRAM) in GB, safety rating ([SAFE], [TIGHT], [OOM RISK]),
        implied training epochs, and wall-clock duration for an Apple Silicon fine-tuning configuration."""
        return estimate_fine_tuning_resources_tool(
            model=model,
            iters=iters,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_seq_length=max_seq_length,
            lora_rank=lora_rank,
            grad_checkpoint=grad_checkpoint,
            total_train_records=total_train_records,
        )

    @server.tool()
    def queue_single_run(
        model: str,
        data_path: str,
        name: Optional[str] = None,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 1e-4,
        iters: int = 600,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        max_seq_length: int = 2048,
        num_layers: int = 16,
        grad_checkpoint: bool = False,
        mask_prompt: bool = True,
        run_eval: bool = False,
        engine: str = "auto",
        queue_dir: str = "mlx_runs",
    ) -> dict:
        """Enqueue a single fine-tuning run (Mode 2) with explicit hyperparameters.
        Auto-detects multimodal models and safely adjusts micro-batch size and gradient accumulation
        to prevent attention mask crashes in mlx-vlm."""
        return queue_single_run_tool(
            model=model,
            data_path=data_path,
            name=name,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            iters=iters,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            max_seq_length=max_seq_length,
            num_layers=num_layers,
            grad_checkpoint=grad_checkpoint,
            mask_prompt=mask_prompt,
            run_eval=run_eval,
            engine=engine,
            queue_dir=queue_dir,
        )

    @server.tool()
    def queue_multi_run_sweep(
        model: str,
        data_path: str,
        batch_size: Optional[List[int]] = None,
        gradient_accumulation_steps: Optional[List[int]] = None,
        learning_rate: Optional[List[float]] = None,
        iters: Optional[List[int]] = None,
        lora_rank: Optional[List[int]] = None,
        lora_alpha: Optional[List[int]] = None,
        max_seq_length: Optional[List[int]] = None,
        grad_checkpoint: Optional[List[bool]] = None,
        run_eval: Optional[List[bool]] = None,
        engine: str = "auto",
        queue_dir: str = "mlx_runs",
    ) -> dict:
        """Enqueue a multi-run hyperparameter sweep (Mode 3). Expands the Cartesian product grid
        of all conditions, writes configs, and appends to the sequential FIFO queue."""
        return queue_multi_run_sweep_tool(
            model=model,
            data_path=data_path,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            iters=iters,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            max_seq_length=max_seq_length,
            grad_checkpoint=grad_checkpoint,
            run_eval=run_eval,
            engine=engine,
            queue_dir=queue_dir,
        )

    @server.tool()
    def inspect_queue(queue_dir: str = "mlx_runs") -> dict:
        """Inspect all queued, running, completed, and failed fine-tuning jobs in the queue directory."""
        return inspect_queue_tool(queue_dir=queue_dir)

    @server.tool()
    def execute_queue(queue_dir: str = "mlx_runs", spawn_terminal: bool = True) -> dict:
        """Execute all pending fine-tuning jobs in the queue sequentially to protect Unified Memory."""
        return execute_queue_tool(queue_dir=queue_dir, spawn_terminal=spawn_terminal)

    @server.tool()
    def run_test_evaluation(
        model: str,
        adapter_path: str,
        data_path: str,
        max_tokens: int = 128,
        engine: str = "auto",
    ) -> dict:
        """Execute generative evaluation on test.jsonl. Scores Exact Match, Substring, Word F1,
        and generates confusion/transition matrices and the offline HTML dashboard."""
        return run_test_evaluation_tool(
            model=model,
            adapter_path=adapter_path,
            data_path=data_path,
            max_tokens=max_tokens,
            engine=engine,
        )

    @server.tool()
    def estimate_test_eval_throughput(
        model_name: str,
        total_samples: int = 50,
        num_passes: int = 2,
        max_tokens: int = 128,
    ) -> dict:
        """Predict model-dependent evaluation throughput (tok/s) and wall-clock duration
        based on parameter count, quantization, and Apple Silicon memory bandwidth physics."""
        return estimate_test_eval_throughput_tool(
            model_name=model_name,
            total_samples=total_samples,
            num_passes=num_passes,
            max_tokens=max_tokens,
        )

    @server.tool()
    def launch_lora_tui(
        mode: int = 2,
        dataset_path: Optional[str] = None,
        model: Optional[str] = None,
        queue_dir: Optional[str] = None,
    ) -> dict:
        """Launch the MLX Commander interactive TUI directly in Mode 2 (Single Run) or Mode 3 (Multi-Run Sweeps)
        in a macOS Terminal window."""
        return launch_lora_tui_tool(
            mode=mode,
            dataset_path=dataset_path,
            model=model,
            queue_dir=queue_dir,
        )

    @server.prompt()
    def prepare_dataset_for_mlx(dataset_path: str) -> str:
        """Instructions and workflow prompt for preparing a dataset for MLX fine-tuning."""
        return (
            f"Please inspect the dataset at '{dataset_path}' using the inspect_dataset tool, "
            f"choose the appropriate MLX format (prompt_completion, chat, text, or dpo), "
            f"and launch the TUI with launch_conversion_tui so the user can verify the preview and convert."
        )

    @server.prompt()
    def orchestrate_fine_tuning(dataset_path: str, model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit") -> str:
        """Instructions and workflow prompt for end-to-end fine-tuning orchestration on Apple Silicon."""
        return (
            f"1. Estimate Unified Memory and duration using estimate_fine_tuning_resources with model '{model}'.\n"
            f"2. Enqueue fine-tuning run(s) using queue_single_run or queue_multi_run_sweep for dataset '{dataset_path}'.\n"
            f"3. Verify queued runs via inspect_queue, then execute sequentially using execute_queue.\n"
            f"4. If run_eval was enabled or after training completes, run generative evaluation using run_test_evaluation."
        )

    return server


def run_mcp_server() -> int:
    """Run MCP server over stdio transport."""
    try:
        server = create_mcp_server()
        server.run(transport="stdio")
        return 0
    except ImportError:
        sys.stderr.write(
            "Error: 'mcp' package is required to run the MCP server.\n"
            "Install it via:\n"
            "    pip install 'mlx_commander[mcp]'\n"
            "or run with uvx:\n"
            "    uvx --with mcp mlx_commander --mcp\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(run_mcp_server())
