"""
Enhanced MLX-VLM Fine-Tuning Launcher with Validation Dataset Support.

Fixes the upstream mlx_vlm.lora issue where val_dataset was hardcoded to None,
enabling automatic discovery and evaluation of valid.jsonl / validation.jsonl,
and applying micro-batch safeguards against attention mask broadcasting errors.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlx_commander.vlm_trainer")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MLX Commander Enhanced VLM Fine-Tuning with Validation Support"
    )

    # Model arguments
    parser.add_argument("--model-path", type=str, required=True, help="Model path or HF repo ID")
    parser.add_argument("--full-finetune", action="store_true", help="Fine-tune full model")
    parser.add_argument("--train-vision", action="store_true", help="Train vision encoder")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, required=True, help="Dataset directory or name")
    parser.add_argument("--split", type=str, default="train", help="Training split")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--image-resize-shape", type=int, nargs=2, default=None)
    parser.add_argument("--custom-prompt-format", type=str, default=None)

    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=200)
    parser.add_argument("--steps-per-save", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--train-on-completions", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--assistant-id", type=int, default=77091)

    # LoRA arguments
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.0)

    # Training mode
    parser.add_argument(
        "--train-mode",
        type=str,
        default="sft",
        choices=["sft", "orpo"],
        help="Training mode: 'sft' (default) or 'orpo'",
    )
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-8)

    # Output arguments
    parser.add_argument("--output-path", type=str, default="adapters.safetensors")
    parser.add_argument("--adapter-path", type=str, default=None)

    return parser


def run_vlm_training(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as e:
        raise ImportError("Fine-tuning requires 'datasets': pip install 'mlx-vlm[train]'") from e

    try:
        import mlx.optimizers as optim
        from mlx_vlm.lora import (
            not_supported_for_training,
            setup_model_for_training,
            transform_dataset_to_messages,
        )
        from mlx_vlm.trainer.datasets import PreferenceVisionDataset, VisionDataset
        from mlx_vlm.trainer.orpo_trainer import ORPOTrainingArgs, train_orpo
        from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
        from mlx_vlm.trainer.utils import Colors, print_trainable_parameters
        from mlx_vlm.utils import load
    except ModuleNotFoundError as e:
        raise ImportError("Fine-tuning requires 'mlx-vlm': pip install 'mlx-vlm[train]'") from e

    # Apply VLM attention mask safeguard: batch_size > 1 causes broadcast errors
    if args.batch_size > 1 and os.environ.get("MLX_DISABLE_VLM_SAFEGUARD") != "1":
        from mlx_commander.lora.vlm_safeguards import calculate_vlm_batch_tweak
        orig_b = args.batch_size
        orig_gas = args.gradient_accumulation_steps
        t_b, t_gas = calculate_vlm_batch_tweak(orig_b, orig_gas)
        args.batch_size = t_b
        args.gradient_accumulation_steps = t_gas
        print(
            f"  [MLX-VLM Safeguard] Detected batch_size={orig_b} > 1. Auto-adjusted to "
            f"batch_size={t_b} and gradient_accumulation_steps={t_gas} to avoid attention mask crashes."
        )

    # Output path resolution
    args.output_path = (
        args.output_path
        if args.output_path.endswith(".safetensors")
        else args.output_path + "/adapters.safetensors"
    )

    print(f"{Colors.HEADER}Loading model from {args.model_path}{Colors.ENDC}")
    model, processor = load(args.model_path, processor_config={"trust_remote_code": True})

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type in not_supported_for_training:
        raise ValueError(f"Model type {model_type} not supported for training")

    config = model.config.__dict__

    # 1. Load Training Dataset
    data_p = Path(args.dataset)
    train_file = data_p / "train.jsonl" if data_p.is_dir() else data_p
    if train_file.exists() and str(train_file).endswith(".jsonl"):
        print(f"{Colors.HEADER}Loading training split from {train_file}{Colors.ENDC}")
        dataset = load_dataset("json", data_files={"train": str(train_file)}, split="train")
    else:
        print(f"{Colors.HEADER}Loading dataset from {args.dataset}{Colors.ENDC}")
        dataset = load_dataset(
            args.dataset,
            args.dataset_config if args.dataset_config else None,
            split=args.split,
        )

    iters = (len(dataset) // args.batch_size) * args.epochs if args.epochs is not None else args.iters

    if args.train_mode != "orpo":
        dataset = transform_dataset_to_messages(dataset, model_type, args.custom_prompt_format)
        train_dataset = VisionDataset(
            dataset,
            config,
            processor,
            image_resize_shape=args.image_resize_shape,
            train_on_completions=args.train_on_completions,
        )
    else:
        train_dataset = PreferenceVisionDataset(
            dataset,
            config,
            processor,
            image_resize_shape=args.image_resize_shape,
        )

    # 2. Automatically Discover and Load Validation Dataset (valid.jsonl)
    val_dataset = None
    if data_p.is_dir():
        val_file = data_p / "valid.jsonl"
        if not val_file.exists():
            val_file = data_p / "validation.jsonl"

        if val_file.exists():
            try:
                print(f"{Colors.OKCYAN}[MLX Commander] Discovered validation split: {val_file}{Colors.ENDC}")
                val_hf = load_dataset("json", data_files={"valid": str(val_file)}, split="valid")
                if args.train_mode != "orpo":
                    val_hf = transform_dataset_to_messages(
                        val_hf, model_type, args.custom_prompt_format
                    )
                    val_dataset = VisionDataset(
                        val_hf,
                        config,
                        processor,
                        image_resize_shape=args.image_resize_shape,
                        train_on_completions=args.train_on_completions,
                    )
                else:
                    val_dataset = PreferenceVisionDataset(
                        val_hf,
                        config,
                        processor,
                        image_resize_shape=args.image_resize_shape,
                    )
                print(
                    f"{Colors.OKGREEN}[MLX Commander] Successfully loaded {len(val_dataset)} validation records. "
                    f"Validation loss will evaluate every {args.steps_per_eval} steps!{Colors.ENDC}"
                )
            except Exception as e:
                print(f"{Colors.WARNING}[MLX Commander Warning] Failed to load validation split: {e}{Colors.ENDC}")

    # 3. Setup Model & Optimizer
    model = setup_model_for_training(model, args, args.adapter_path)
    print_trainable_parameters(model)

    print(f"{Colors.HEADER}Setting up optimizer{Colors.ENDC}")
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    print(f"{Colors.HEADER}Training model ({args.train_mode}){Colors.ENDC}")
    if args.train_mode == "orpo":
        training_args = ORPOTrainingArgs(
            batch_size=args.batch_size,
            iters=iters,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.steps_per_save,
            val_batches=args.val_batches,
            max_seq_length=args.max_seq_length,
            adapter_file=args.output_path,
            grad_checkpoint=args.grad_checkpoint,
            learning_rate=args.learning_rate,
            grad_clip=args.grad_clip,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            full_finetune=args.full_finetune,
            beta=args.beta,
            eps=args.eps,
        )
        train_orpo(
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=training_args,
            train_on_completions=args.train_on_completions,
            assistant_id=args.assistant_id,
        )
    else:
        training_args = TrainingArgs(
            batch_size=args.batch_size,
            iters=iters,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.steps_per_save,
            val_batches=args.val_batches,
            max_seq_length=args.max_seq_length,
            adapter_file=args.output_path,
            grad_checkpoint=args.grad_checkpoint,
            learning_rate=args.learning_rate,
            grad_clip=args.grad_clip,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            full_finetune=args.full_finetune,
        )
        train(
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=training_args,
            train_on_completions=args.train_on_completions,
            assistant_id=args.assistant_id,
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_vlm_training(args)


if __name__ == "__main__":
    main()
