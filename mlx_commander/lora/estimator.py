"""
Analytical Estimator for Apple MLX LoRA Fine-Tuning.
Computes:
1. Implied Epochs (iters * batch_size / dataset_records)
2. Peak Unified Memory (VRAM) with OOM Risk Level
3. Estimated Run Duration and Clock ETA
"""

import functools
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import LoraRunConfig


@functools.lru_cache(maxsize=1)
def get_hardware_memory_bytes() -> int:
    """Return total physical Unified Memory in bytes on macOS (or fallback). Cached."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
        return int(out)
    except Exception:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except Exception:
            return 16 * 1024 * 1024 * 1024  # 16 GB fallback


@functools.lru_cache(maxsize=1)
def get_apple_silicon_chip() -> str:
    """Detect Apple Silicon chip family string (e.g. 'Apple M3 Max'). Cached."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        if out:
            return out
    except Exception:
        pass
    return "Apple Silicon"


@functools.lru_cache(maxsize=64)
def _get_dir_safetensors_size_gb(dir_path_str: str) -> float:
    """Calculate total size in GB of model weight files in directory, cached."""
    try:
        p = Path(dir_path_str)
        if p.is_file():
            p = p.parent
        if p.is_dir():
            patterns = ("*.safetensors", "*.bin", "*.mlx", "*.pt", "*.npz")
            total = 0
            for pat in patterns:
                total += sum(f.stat().st_size for f in p.glob(pat))
            if total > 0:
                return total / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def clear_estimator_cache() -> None:
    """Clear cached hardware info and directory size lookups."""
    get_hardware_memory_bytes.cache_clear()
    get_apple_silicon_chip.cache_clear()
    get_chip_memory_bandwidth_gbps.cache_clear()
    _get_dir_safetensors_size_gb.cache_clear()


def calculate_implied_epochs(
    iters: int,
    batch_size: int,
    total_train_records: int,
    grad_accumulation_steps: int = 1,
) -> Optional[float]:
    """
    Calculate implied epochs: (iters * batch_size * grad_accumulation_steps) / total_train_records.
    Returns None if total_train_records is 0 or unknown.
    """
    if total_train_records <= 0 or iters <= 0 or batch_size <= 0:
        return None
    eff_batch = batch_size * max(1, grad_accumulation_steps)
    return (iters * eff_batch) / float(total_train_records)


def parse_model_param_billions(model_name: str) -> float:
    """Infer model parameter count in billions from model name or path."""
    lower = model_name.lower()
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)[bB]', lower)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    if "70b" in lower:
        return 70.0
    if "32b" in lower:
        return 32.0
    if "14b" in lower:
        return 14.0
    if "8b" in lower or "llama-3.1-8b" in lower:
        return 8.0
    if "7b" in lower or "mistral" in lower:
        return 7.0
    if "phi-4" in lower:
        return 14.0
    if "phi-3" in lower:
        return 3.8
    if "gemma-4" in lower or "gemma4" in lower:
        return 4.0
    if "3b" in lower or "llama-3.2-3b" in lower or "phi-3.5" in lower:
        return 3.2
    if "1b" in lower or "llama-3.2-1b" in lower:
        return 1.2
    return 7.0  # default assumption


def is_4bit_quantized(model_name: str) -> bool:
    lower = model_name.lower()
    return "4bit" in lower or "q4" in lower or "int4" in lower or "quantized" in lower


def estimate_peak_memory(config: LoraRunConfig, total_ram_bytes: Optional[int] = None) -> Dict[str, Any]:
    """
    Estimate peak Unified Memory (RAM/VRAM) consumption in GB during LoRA training.
    Accounts for base weights, LoRA optimizer states, gradient checkpointing, and activation buffers.
    """
    if total_ram_bytes is None:
        total_ram_bytes = get_hardware_memory_bytes()
    total_ram_gb = total_ram_bytes / (1024 ** 3)

    # 1. Base model weights (GB) & Parameter count
    param_b = parse_model_param_billions(config.model)
    local_p = Path(config.model).expanduser()
    if local_p.is_file():
        local_p = local_p.parent
    if local_p.exists() and local_p.is_dir():
        # Check actual disk size of model files (cached)
        dir_size_gb = _get_dir_safetensors_size_gb(str(local_p.resolve()))
        if dir_size_gb > 0:
            model_gb = dir_size_gb
            # If param_b was not specifically parsed from model name (fell back to 7.0),
            # refine param_b estimate from the actual weight files
            if "7b" not in config.model.lower() and param_b == 7.0:
                bytes_per_param = 0.55 if is_4bit_quantized(config.model) else 2.0
                param_b = max(0.5, round(dir_size_gb / bytes_per_param, 1))
        else:
            bytes_per_param = 0.55 if is_4bit_quantized(config.model) else 2.0
            model_gb = param_b * bytes_per_param
    else:
        bytes_per_param = 0.55 if is_4bit_quantized(config.model) else 2.0
        model_gb = param_b * bytes_per_param

    # 2. LoRA weights & Optimizer memory (GB)
    # Trainable params scale with rank and layers
    effective_layers = config.num_layers if config.num_layers > 0 else 32
    lora_params_m = 2 * config.lora_rank * 4096 * effective_layers * 4 / 1_000_000
    # AdamW stores 2 states (8 bytes) + gradients (4 bytes) in fp32
    lora_opt_gb = max(0.1, (lora_params_m * 12) / 1024)

    # 3. Activation memory (GB)
    # Scales with batch size * max_seq_length
    tokens_per_batch = config.batch_size * config.max_seq_length
    base_act_gb = (tokens_per_batch / 8192) * (param_b / 4.0) * 3.5

    if config.grad_checkpoint:
        # Gradient checkpointing drops activation footprint by ~80%
        act_gb = max(0.8, base_act_gb * 0.20)
    else:
        act_gb = max(2.5, base_act_gb)

    # 4. Runtime & Metal allocator overhead (GB)
    overhead_gb = 0.8

    peak_gb_raw = model_gb + lora_opt_gb + act_gb + overhead_gb
    usage_ratio = peak_gb_raw / total_ram_gb

    # Drop decimals and round everything up to the nearest gigabyte
    peak_gb = math.ceil(peak_gb_raw)
    total_ram_int = math.ceil(total_ram_gb)
    model_gb_int = math.ceil(model_gb)
    lora_opt_gb_int = math.ceil(lora_opt_gb)
    act_gb_int = math.ceil(act_gb)

    if usage_ratio < 0.70:
        safety_tier = "safe"
        safety_label = "SAFE"
    elif usage_ratio <= 0.85:
        safety_tier = "tight"
        safety_label = "TIGHT"
    else:
        safety_tier = "oom_risk"
        safety_label = "OOM RISK"

    recs = []
    if safety_tier == "oom_risk":
        if not config.grad_checkpoint:
            recs.append("Enable gradient checkpointing (grad_checkpoint=True) to save ~80% activation RAM.")
        if config.batch_size > 2:
            recs.append("Reduce batch_size to 2 or 1.")
        if config.max_seq_length > 2048:
            recs.append("Reduce max_seq_length (e.g. 2048).")

    badge = f"Est. Peak RAM: ~{peak_gb} GB / {total_ram_int} GB [{safety_label}]"
    return {
        "model_gb": model_gb_int,
        "base_model_gb": model_gb_int,
        "lora_opt_gb": lora_opt_gb_int,
        "optimizer_gb": lora_opt_gb_int,
        "act_gb": act_gb_int,
        "activations_gb": act_gb_int,
        "overhead_gb": math.ceil(overhead_gb),
        "peak_gb": peak_gb,
        "est_gb": peak_gb,
        "total_ram_gb": total_ram_int,
        "total_gb": total_ram_int,
        "hardware_gb": total_ram_int,
        "usage_ratio": round(usage_ratio, 2),
        "safety_tier": safety_tier,
        "safety_label": safety_label,
        "safety_level": safety_label,
        "badge": badge,
        "recommendations": recs,
    }


def estimate_duration(config: LoraRunConfig, chip_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Estimate training duration and completion ETA based on hardware and iterations.
    """
    if chip_name is None:
        chip_name = get_apple_silicon_chip()

    lower_chip = chip_name.lower()
    # Base step rate (seconds per step for ~3B model at batch 4, seq_len 2048)
    if "ultra" in lower_chip:
        step_base = 0.4
    elif "max" in lower_chip:
        step_base = 0.8
    elif "pro" in lower_chip:
        step_base = 1.4
    else:
        step_base = 2.4

    param_b = parse_model_param_billions(config.model)
    model_factor = max(0.6, param_b / 3.0)
    batch_factor = max(0.5, config.batch_size / 4.0)
    seq_factor = max(0.5, config.max_seq_length / 2048.0)

    sec_per_step = step_base * model_factor * batch_factor * seq_factor
    total_seconds = int(config.iters * sec_per_step)

    # Format human duration
    if total_seconds < 60:
        duration_str = f"{total_seconds}s"
    elif total_seconds < 3600:
        duration_str = f"{total_seconds // 60}m {total_seconds % 60}s"
    else:
        hours = total_seconds // 3600
        mins = (total_seconds % 3600) // 60
        duration_str = f"{hours}h {mins}m"

    # Format clock ETA
    eta_time = time.localtime(time.time() + total_seconds)
    eta_str = time.strftime("%I:%M %p", eta_time)

    return {
        "sec_per_step": round(sec_per_step, 2),
        "total_seconds": total_seconds,
        "seconds": total_seconds,
        "formatted_duration": duration_str,
        "duration_str": duration_str,
        "eta_str": eta_str,
        "eta_clock": eta_str,
        "chip": chip_name,
        "iters_per_sec": round(1.0 / sec_per_step, 2) if sec_per_step > 0 else 1.0,
    }


@functools.lru_cache(maxsize=16)
def get_chip_memory_bandwidth_gbps(chip_name: Optional[str] = None) -> float:
    """
    Return estimated Unified Memory bandwidth in GB/s for Apple Silicon chip families.
    Physics:
    - Apple M-series Ultra: ~800 GB/s (dual-die UltraFusion interconnect)
    - Apple M4/M5 Max: ~450 GB/s nominal
    - Apple M1/M2/M3 Max: ~350 GB/s nominal
    - Apple M-series Pro: ~200 GB/s nominal
    - Apple M3/M4/M5 Base: ~135 GB/s nominal
    - Apple M1/M2 Base: ~100 GB/s
    - Non-Apple Silicon / Fallback: ~135 GB/s
    """
    if chip_name is None:
        chip_name = get_apple_silicon_chip()
    lower = chip_name.lower()
    if "ultra" in lower:
        return 800.0
    elif "max" in lower:
        if "m5" in lower or "m4" in lower:
            return 450.0
        return 350.0
    elif "pro" in lower:
        return 200.0
    elif "m4" in lower or "m5" in lower:
        return 135.0
    elif "m3" in lower:
        return 150.0
    elif "m2" in lower or "m1" in lower:
        return 100.0
    return 135.0


def estimate_eval_throughput(
    model_name: str,
    total_samples: int = 1,
    num_passes: int = 2,
    max_tokens: int = 128,
    chip_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analytical throughput and duration estimator for generative evaluation on Apple Silicon.
    Autoregressive LLM generation is strictly memory-bandwidth bound:
    Throughput (tok/s) = (Memory_Bandwidth_GBps / Model_Weights_GB) * Efficiency_Factor.

    Returns dictionary containing:
    - est_tps: estimated generation speed in tokens/second
    - sec_per_sample: estimated latency per sample (prompt prefill + generation)
    - est_total_sec: estimated total duration for (total_samples * num_passes)
    - est_duration_str: formatted string (e.g. '~35s', '~1m 20s')
    - param_b: model size in billions of parameters
    - quant_label: '4-bit', '8-bit', or 'fp16'
    - model_gb: model size in Unified Memory (GB)
    - chip_name: hardware chip name
    - summary_label: human-readable badge for CLI/TUI display
    - total_generations: total samples across passes
    """
    if chip_name is None:
        chip_name = get_apple_silicon_chip()

    model_name_str = str(model_name or "").strip()
    lower = model_name_str.lower()
    param_b = parse_model_param_billions(model_name_str)

    is_4bit = is_4bit_quantized(model_name_str)
    is_8bit = "8bit" in lower or "q8" in lower or "int8" in lower

    if is_4bit:
        quant_label = "4-bit"
        bytes_per_param = 0.65
    elif is_8bit:
        quant_label = "8-bit"
        bytes_per_param = 1.15
    else:
        quant_label = "fp16"
        bytes_per_param = 2.0

    # Check if local model directory exists with safetensors/bin weights
    local_p = Path(model_name_str).expanduser() if model_name_str else Path(".")
    if local_p.is_file():
        local_p = local_p.parent
    if local_p.exists() and local_p.is_dir():
        dir_size_gb = _get_dir_safetensors_size_gb(str(local_p.resolve()))
        if dir_size_gb > 0:
            model_gb = dir_size_gb
            if param_b == 7.0 and "7b" not in lower:
                param_b = max(0.5, round(dir_size_gb / bytes_per_param, 1))
        else:
            model_gb = max(0.5, param_b * bytes_per_param)
    else:
        model_gb = max(0.5, param_b * bytes_per_param)

    bw = get_chip_memory_bandwidth_gbps(chip_name)

    # Sustained memory-bandwidth efficiency in MLX is typically ~60% - 70%
    # tok/s = (bw / model_gb) * 0.65, capped realistically at ~350 tok/s
    raw_tps = (bw / model_gb) * 0.65
    est_tps = max(1.0, min(350.0, round(raw_tps, 1)))

    # In generative evaluation, average response is ~35-45 tokens (or capped by max_tokens)
    expected_gen_tokens = min(40, max_tokens)
    gen_time_sec = expected_gen_tokens / est_tps
    prefill_overhead_sec = 0.08  # prompt processing & framework dispatch
    sec_per_sample = max(0.05, round(gen_time_sec + prefill_overhead_sec, 2))

    total_generations = max(1, total_samples * num_passes)
    est_total_sec = round(total_generations * sec_per_sample, 1)

    # Format duration string
    if est_total_sec < 60:
        est_duration_str = f"~{int(round(est_total_sec))}s"
    elif est_total_sec < 3600:
        mins = int(est_total_sec // 60)
        secs = int(est_total_sec % 60)
        est_duration_str = f"~{mins}m {secs:02d}s"
    else:
        hours = int(est_total_sec // 3600)
        mins = int((est_total_sec % 3600) // 60)
        est_duration_str = f"~{hours}h {mins:02d}m"

    param_str = f"{int(param_b)}B" if param_b.is_integer() else f"{param_b}B"
    summary_label = (
        f"model: ~{param_str} {quant_label} [~{model_gb:.1f} GB], "
        f"est. speed: ~{round(est_tps)} tok/s on {chip_name}"
    )

    return {
        "est_tps": est_tps,
        "sec_per_sample": sec_per_sample,
        "est_total_sec": est_total_sec,
        "est_duration_str": est_duration_str,
        "param_b": param_b,
        "quant_label": quant_label,
        "model_gb": round(model_gb, 1),
        "chip_name": chip_name,
        "summary_label": summary_label,
        "total_generations": total_generations,
    }

