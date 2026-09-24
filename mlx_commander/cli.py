"""
Command Line Interface (CLI) for mlx_commander converter.
Dispatches between full-screen Curses TUI, interactive terminal wizard,
and automated headless conversion.
"""

import argparse
import json
import os
import sys

# Ensure ncurses escape delay is 25ms to make ESC instantaneous in all TUI pickers
os.environ.setdefault("ESCDELAY", "25")

import curses
from pathlib import Path
from typing import List, Optional

from mlx_commander import __version__
from mlx_commander.converter import ConversionResult, convert_and_save
from mlx_commander.formats import (
    ColumnMapping,
    MLXFormat,
    auto_detect_mapping,
    parse_label_map,
    validate_mapping,
)
from mlx_commander.exceptions import MissingDependencyError
from mlx_commander.loader import load_local_dataset
from mlx_commander.splitter import SplitConfig, generate_random_seed
from mlx_commander.tui.app import launch_tui
from mlx_commander.tui.wizard_fallback import run_interactive_wizard


def parse_mapping_arg(mapping_str: str) -> ColumnMapping:
    """Parse JSON or key=val,key=val string into ColumnMapping."""
    clean = mapping_str.strip()
    if clean.startswith("{"):
        d = json.loads(clean)
        if "label_map" in d and isinstance(d["label_map"], (dict, str)):
            d["label_map"] = parse_label_map(d["label_map"])
        return ColumnMapping(**d)

    mapping = ColumnMapping()
    last_key = None
    for pair in clean.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" in pair:
            k, v = pair.split("=", 1)
            k, v = k.strip(), v.strip()
            last_key = k
            if hasattr(mapping, k):
                if k == "label_map":
                    setattr(mapping, k, parse_label_map(v))
                else:
                    setattr(mapping, k, v)
            elif k in ("system_prompt", "default_system_prompt"):
                mapping.default_system_prompt = v
            elif k in ("label_map", "labels"):
                mapping.label_map = parse_label_map(v)
            elif k == "prompt":
                mapping.prompt_col = v
            elif k == "completion":
                mapping.completion_col = v
            elif k == "text":
                mapping.text_col = v
            elif k == "messages":
                mapping.messages_col = v
            elif k == "user":
                mapping.user_col = v
            elif k == "assistant":
                mapping.assistant_col = v
            elif k == "system":
                mapping.system_col = v
            elif k == "chosen":
                mapping.chosen_col = v
            elif k == "rejected":
                mapping.rejected_col = v
        elif last_key in ("label_map", "labels") and (":" in pair or "=" in pair):
            extra = parse_label_map(pair)
            if extra:
                if mapping.label_map is None:
                    mapping.label_map = {}
                mapping.label_map.update(extra)
    return mapping


def parse_prefill_state(val: str) -> dict:
    """Parse prefill state from a JSON string or file path."""
    clean = val.strip()
    if clean.startswith("{"):
        return json.loads(clean)
    p = Path(clean).expanduser()
    if p.exists() and p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        return json.loads(clean)
    except Exception as e:
        raise ValueError(f"Could not parse --prefill-state as JSON or file path: {e}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx_commander",
        description="Configure and run model training on Apple MLX via interactive TUI and CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch interactive TUI wizard:
  mlx_commander

  # Launch line-by-line CLI wizard:
  mlx_commander --no-tui

  # Direct conversion from CLI:
  mlx_commander -d ./my_hf_dataset -f prompt_completion -o ./mlx_out \
         --prompt-col question --completion-col answer \
         --train 80 --valid 10 --test 10 --seed 42

  # Combine multiple dataset files with schema verification & re-splitting:
  mlx_commander -d train.jsonl test.jsonl -f prompt_completion -o ./mlx_out \
         --prompt-col question --completion-col answer
        """,
    )

    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    # Core conversion flags
    parser.add_argument(
        "-d", "--dataset",
        nargs="+",
        type=str,
        help="Path(s) to local Hugging Face dataset folder or data file(s) (.parquet, .arrow, .jsonl, .csv, .tsv, .sqlite, .tar). Multiple files will be verified for schema consistency and merged.",
    )
    parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["text", "chat", "prompt_completion", "dpo"],
        help="Target MLX format: 'text', 'chat', 'prompt_completion', or 'dpo'.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Destination directory where train.jsonl, valid.jsonl, test.jsonl will be saved (default: 'mlx_dataset' subfolder in same folder as source dataset).",
    )

    # Split parameters
    parser.add_argument("--train", type=float, default=None, help="Train split percentage (e.g. 80.0).")
    parser.add_argument("--valid", type=float, default=None, help="Validation split percentage (e.g. 10.0).")
    parser.add_argument("--test", type=float, default=None, help="Test split percentage (e.g. 10.0, or 0 to omit).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible shuffling.")
    parser.add_argument("--keep-splits", action="store_true", help="Preserve existing splits without re-splitting.")

    # Column mapping flags
    parser.add_argument("--mapping", type=str, help="Column mapping as JSON string or key=val,key=val.")
    parser.add_argument("--text-col", type=str, help="Source column for 'text' format.")
    parser.add_argument("--text-template", type=str, help="Template string for 'text' format (e.g. '{instruction}\\n{output}').")
    parser.add_argument("--prompt-col", type=str, help="Source column for prompt / question.")
    parser.add_argument("--completion-col", type=str, help="Source column for completion / answer.")
    parser.add_argument("--messages-col", type=str, help="Source column containing chat messages list.")
    parser.add_argument("--user-col", type=str, help="Source column for user turn in chat format.")
    parser.add_argument("--assistant-col", type=str, help="Source column for assistant turn in chat format.")
    parser.add_argument("--system-col", type=str, help="Source column for system prompt in chat format.")
    parser.add_argument("--chosen-col", type=str, help="Source column for chosen response in DPO format.")
    parser.add_argument("--rejected-col", type=str, help="Source column for rejected response in DPO format.")
    parser.add_argument("--system-prompt", type=str, default=None, help="Default system prompt to inject into all records (especially useful when HF dataset lacks one).")
    parser.add_argument("--label-map", type=str, default=None, help="Semantic rewriting for target labels, e.g. '0:negative,1:neutral,2:positive'.")
    parser.add_argument("--test-gold", action="store_true", help="Also export auxiliary test_gold.jsonl with prompt/expected schema alongside canonical test.jsonl.")

    # Agent & Hand-off flags
    parser.add_argument(
        "--manifest-file",
        type=str,
        default=None,
        help="Custom file path where machine-readable mlx_manifest.json will be saved.",
    )
    parser.add_argument(
        "--prefill-state",
        type=str,
        default=None,
        help="Pre-populate TUI state from a JSON string or path to JSON file.",
    )
    parser.add_argument(
        "--spawn-terminal",
        action="store_true",
        help="Spawn interactive TUI in an external macOS Terminal window (ideal for AI agents & subshells).",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Launch Model Context Protocol (MCP) server over stdio for Claude Desktop, Cursor, etc.",
    )

    # UI mode flags
    parser.add_argument(
        "--commander", "--tui",
        action="store_true",
        dest="commander",
        help="Launch full-screen persistent MLX Commander TUI dashboard (default in interactive terminal).",
    )
    parser.add_argument(
        "--wizard", "--no-tui", "--cli",
        action="store_true",
        dest="wizard",
        help="Run sequential step-by-step terminal wizard instead of persistent MLX Commander dashboard.",
    )
    parser.add_argument(
        "--single-run", "--lora",
        action="store_true",
        dest="single_run",
        help="Launch TUI directly in Single Run mode (Mode 2: Fine-Tuning). This is the default startup mode.",
    )
    parser.add_argument(
        "--multi-run",
        action="store_true",
        dest="multi_run",
        help="Launch TUI directly in Multi-Run Matrix mode (Mode 3: Sweeps).",
    )
    parser.add_argument(
        "--converter", "--dataset-converter",
        action="store_true",
        dest="converter",
        help="Launch TUI directly in Dataset Converter mode (Mode 1).",
    )
    parser.add_argument(
        "--run-queue",
        nargs="?",
        const="mlx_runs",
        type=str,
        default=None,
        help="Execute LoRA fine-tuning runs sequentially from queue directory (default: 'mlx_runs').",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="mlx-commander",
        help="Weights & Biases project name for experiment tracking (default: 'mlx-commander').",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases experiment logging even if wandb is installed and logged in.",
    )

    return parser


def run_direct_conversion(args: argparse.Namespace) -> ConversionResult:
    """Perform headless conversion using command-line arguments."""
    dataset = load_local_dataset(args.dataset)
    target_format = MLXFormat(args.format.lower())

    # Build mapping
    if args.mapping:
        mapping = parse_mapping_arg(args.mapping)
    else:
        mapping = auto_detect_mapping(target_format, dataset.columns)
        if args.text_col:
            mapping.text_col = args.text_col
        if args.text_template:
            mapping.text_template = args.text_template
        if args.prompt_col:
            mapping.prompt_col = args.prompt_col
        if args.completion_col:
            mapping.completion_col = args.completion_col
        if args.messages_col:
            mapping.messages_col = args.messages_col
        if args.user_col:
            mapping.user_col = args.user_col
        if args.assistant_col:
            mapping.assistant_col = args.assistant_col
        if args.system_col:
            mapping.system_col = args.system_col
        if args.chosen_col:
            mapping.chosen_col = args.chosen_col
        if args.rejected_col:
            mapping.rejected_col = args.rejected_col

    if getattr(args, "system_prompt", None):
        mapping.default_system_prompt = args.system_prompt
    if getattr(args, "label_map", None):
        mapping.label_map = parse_label_map(args.label_map)

    # Validate mapping
    errs = validate_mapping(target_format, mapping, dataset.columns)
    if errs:
        raise ValueError("Mapping error: " + "; ".join(errs))

    # Split config
    split_config = None
    if not (args.keep_splits and dataset.is_split):
        train_p = args.train if args.train is not None else 80.0
        valid_p = args.valid if args.valid is not None else 10.0
        test_p = args.test if args.test is not None else (100.0 - train_p - valid_p)
        seed = args.seed if args.seed is not None else generate_random_seed()
        split_config = SplitConfig(train_pct=train_p, valid_pct=valid_p, test_pct=test_p, seed=seed)
        v_errs = split_config.validate()
        if v_errs:
            raise ValueError("Split error: " + "; ".join(v_errs))

    out_dir = args.output or str(dataset.default_output_dir)
    return convert_and_save(
        dataset=dataset,
        format_type=target_format,
        mapping=mapping,
        output_dir=out_dir,
        split_config=split_config,
        use_existing_splits=args.keep_splits,
        export_test_gold=getattr(args, "test_gold", False),
        manifest_file=getattr(args, "manifest_file", None),
    )


def is_interactive_tty() -> bool:
    """Check if stdout and stdin are interactive TTYs with adequate terminfo."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "")
    if not term or term == "dumb":
        return False
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mcp:
        from mlx_commander.mcp_server import run_mcp_server
        return run_mcp_server()

    # Case 0: Sequential LoRA queue execution
    if args.run_queue is not None:
        from mlx_commander.lora import run_lora_queue
        q_dir = Path(args.run_queue)
        ok = run_lora_queue(
            q_dir,
            wandb_project=args.wandb_project,
            enable_wandb=not args.no_wandb,
        )
        return 0 if ok else 1

    # Normalize dataset path argument
    dataset_input = None
    if args.dataset:
        if isinstance(args.dataset, list):
            dataset_input = "\n".join(args.dataset) if len(args.dataset) > 1 else args.dataset[0]
        else:
            dataset_input = args.dataset

    # Case 1: Direct Headless Run (dataset and format specified, without explicit UI flags)
    if args.dataset and args.format and not args.wizard and not args.commander and not args.spawn_terminal:
        try:
            result = run_direct_conversion(args)
            src_desc = f"{len(args.dataset)} files (merged)" if isinstance(args.dataset, list) and len(args.dataset) > 1 else (args.dataset[0] if isinstance(args.dataset, list) else str(args.dataset))
            print(f"[OK] Successfully converted {src_desc} to {args.format} format in {result.output_dir}")
            for s_name, path in result.output_files.items():
                cnt = result.record_counts.get(s_name, 0)
                print(f"  • {path.name}: {cnt:,} records")
            if result.manifest_path:
                print(f"  • Manifest: {result.manifest_path}")
            print(f"\nMLX Fine-tuning command:\n{result.generate_mlx_lora_command()}\n")
            return 0
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return 130
        except MissingDependencyError as e:
            print(f"\nMissing Dependency Error:\n{e.format_name} format requires package '{e.package_name}'.", file=sys.stderr)
            print(f"\nTo install, run:\n    {e.install_command}", file=sys.stderr)
            print(f"or install optional extra:\n    {e.pip_extra}\n", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # Case 2: Sequential wizard explicitly requested via --wizard / --no-tui / --cli
    if args.wizard:
        try:
            run_interactive_wizard(
                dataset_path=dataset_input,
                format_arg=args.format,
                output_dir_arg=args.output,
                train_pct_arg=args.train,
                valid_pct_arg=args.valid,
                test_pct_arg=args.test,
                seed_arg=args.seed,
            )
            return 0
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            return 130
        except MissingDependencyError as e:
            print(f"\nMissing Dependency Error:\n{e.format_name} format requires package '{e.package_name}'.", file=sys.stderr)
            print(f"\nTo install, run:\n    {e.install_command}", file=sys.stderr)
            print(f"or install optional extra:\n    {e.pip_extra}\n", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            return 1

    # Case 3: Interactive TUI Dashboard (with optional prefill and Terminal Spawner)
    prefill_dict: Dict[str, Any] = {}
    if args.prefill_state:
        try:
            prefill_dict.update(parse_prefill_state(args.prefill_state))
        except Exception as e:
            print(f"Error parsing --prefill-state: {e}", file=sys.stderr)
            return 1

    if args.format:
        prefill_dict["format"] = args.format
    if args.prompt_col:
        prefill_dict["prompt_col"] = args.prompt_col
    if args.completion_col:
        prefill_dict["completion_col"] = args.completion_col
    if args.text_col:
        prefill_dict["text_col"] = args.text_col
    if args.text_template:
        prefill_dict["text_template"] = args.text_template
    if args.messages_col:
        prefill_dict["messages_col"] = args.messages_col
    if args.user_col:
        prefill_dict["user_col"] = args.user_col
    if args.assistant_col:
        prefill_dict["assistant_col"] = args.assistant_col
    if args.system_col:
        prefill_dict["system_col"] = args.system_col
    if args.chosen_col:
        prefill_dict["chosen_col"] = args.chosen_col
    if args.rejected_col:
        prefill_dict["rejected_col"] = args.rejected_col
    if args.train is not None:
        prefill_dict["train"] = args.train
    if args.valid is not None:
        prefill_dict["valid"] = args.valid
    if args.test is not None:
        prefill_dict["test"] = args.test
    if args.seed is not None:
        prefill_dict["seed"] = args.seed
    if args.output:
        prefill_dict["output"] = args.output
    if dataset_input:
        prefill_dict["dataset"] = dataset_input
    if getattr(args, "converter", False) or (args.format and not getattr(args, "single_run", False) and not getattr(args, "multi_run", False)):
        prefill_dict["tab"] = 0
        prefill_dict["active_tab"] = 0
    elif getattr(args, "multi_run", False):
        prefill_dict["tab"] = 2
        prefill_dict["active_tab"] = 2
    elif getattr(args, "single_run", False):
        prefill_dict["tab"] = 1
        prefill_dict["active_tab"] = 1
    else:
        # Default startup view is Single Run (Mode 2)
        if "active_tab" not in prefill_dict and "tab" not in prefill_dict and "mode" not in prefill_dict:
            prefill_dict["tab"] = 1
            prefill_dict["active_tab"] = 1
    if args.wandb_project:
        prefill_dict["wandb_project"] = args.wandb_project
    if args.no_wandb:
        prefill_dict["wandb_enabled"] = False

    # Check if external terminal window should be spawned
    from mlx_commander.terminal_spawner import is_macos, spawn_terminal_tui
    should_spawn = args.spawn_terminal or (not is_interactive_tty() and is_macos())

    if should_spawn:
        raw_args = list(sys.argv[1:] if argv is None else argv)
        return spawn_terminal_tui(raw_args, manifest_path=args.manifest_file)

    if not is_interactive_tty():
        print(
            "Error: No interactive terminal (TTY) detected.\n"
            "On macOS, use --spawn-terminal to launch in Terminal.app, or specify --format for headless conversion.",
            file=sys.stderr,
        )
        return 1

    try:
        result = launch_tui(default_dataset_path=dataset_input, prefill=prefill_dict)
        if result is not None:
            if args.manifest_file:
                result.save_manifest(args.manifest_file)
            print(f"\n[OK] Successfully converted dataset to {result.format_type.value} format in {result.output_dir}")
            if result.manifest_path:
                print(f"  • Manifest: {result.manifest_path}")
            return 0
        else:
            if args.manifest_file:
                try:
                    Path(args.manifest_file).parent.mkdir(parents=True, exist_ok=True)
                    with open(args.manifest_file, "w", encoding="utf-8") as f:
                        json.dump({"status": "cancelled"}, f, indent=2)
                except Exception:
                    pass
            print("\nOperation cancelled.")
            return 130
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled.")
        return 130
    except curses.error as e:
        print(f"\nTerminal error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(130)
