"""
VLM Attention Mask Safeguards and Hyperparameter Substitutions.

In mlx-vlm (used for Vision-Language / Multimodal models like Gemma 4, Qwen2-VL,
PaliGemma, SmolVLM, etc.), training with micro-batch_size > 1 triggers attention mask
shape broadcasting errors (e.g. ValueError: [broadcast_shapes] (B, 96) and (B, 8, 96, 96))
due to unpadded 2D/4D multimodal attention masks across disparate sequences.

This module provides mathematical equivalence helpers to safely substitute:
    batch_size <- 1
    gradient_accumulation_steps <- batch_size * gradient_accumulation_steps
preserving identical effective batch sizes and gradient dynamics without crashing.
"""

from typing import List, Tuple


def calculate_vlm_batch_tweak(batch_size: int, grad_accumulation_steps: int = 1) -> Tuple[int, int]:
    """
    Calculate the equivalent single-sample batch configuration for mlx-vlm:
    Returns (tweaked_batch_size, tweaked_grad_accumulation_steps).
    
    Example:
        batch_size=4, grad_accumulation_steps=1 -> (1, 4)
        batch_size=2, grad_accumulation_steps=4 -> (1, 8)
        batch_size=1, grad_accumulation_steps=4 -> (1, 4)
    """
    b = max(1, int(batch_size))
    gas = max(1, int(grad_accumulation_steps))
    return 1, b * gas


def calculate_vlm_sweep_tweak(batch_sizes: List[int], grad_accums: List[int]) -> Tuple[List[int], List[int]]:
    """
    Calculate equivalent sweep values for mlx-vlm:
    Sets batch_size to [1] and maps all distinct effective batch sizes (b * g)
    into the gradient_accumulation_steps sweep list.
    
    Example:
        batch_sizes=[2, 4], grad_accums=[1] -> ([1], [2, 4])
        batch_sizes=[4], grad_accums=[1] -> ([1], [4])
        batch_sizes=[1, 2, 4], grad_accums=[1, 2] -> ([1], [1, 2, 4, 8])
    """
    cleaned_b = [max(1, int(b)) for b in batch_sizes] if batch_sizes else [1]
    cleaned_g = [max(1, int(g)) for g in grad_accums] if grad_accums else [1]
    
    effective_accums = sorted(list({b * g for b in cleaned_b for g in cleaned_g}))
    return [1], effective_accums


def should_suggest_vlm_tweak(engine: str, batch_size: int) -> bool:
    """Check if the current run configuration warrants an mlx-vlm batch size tweak."""
    return engine == "mlx_vlm" and batch_size > 1


def should_suggest_vlm_sweep_tweak(engine: str, batch_sizes: List[int]) -> bool:
    """Check if the current sweep configuration contains batch sizes > 1 for an mlx-vlm model."""
    if engine != "mlx_vlm":
        return False
    return any(int(b) > 1 for b in batch_sizes)
