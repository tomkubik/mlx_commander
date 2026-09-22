"""
Conversion Engine for writing MLX JSONL datasets.
Streams converted and mapped records to train.jsonl, valid.jsonl, and test.jsonl.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from mlx_commander.formats import ColumnMapping, MLXFormat, format_record, validate_mapping
from mlx_commander.loader import LoadedDataset
from mlx_commander.splitter import SplitConfig, split_records


@dataclass
class ConversionResult:
    output_dir: Path
    format_type: MLXFormat
    output_files: Dict[str, Path]
    record_counts: Dict[str, int]
    file_sizes: Dict[str, int]
    sample_records: Dict[str, List[Dict[str, Any]]]
    seed_used: Optional[int]
    manifest_path: Optional[Path] = None
    source_path: Optional[str] = None

    def generate_mlx_lora_command(self, model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit") -> str:
        """Generate a ready-to-use mlx_lm.lora command line string."""
        mask_arg = " --mask-prompt" if self.format_type in (MLXFormat.CHAT, MLXFormat.PROMPT_COMPLETION) else ""
        return (
            f"mlx_lm.lora \\\n"
            f"    --model {model_name} \\\n"
            f"    --train \\\n"
            f"    --data {self.output_dir}{mask_arg} \\\n"
            f"    --iters 600 \\\n"
            f"    --batch-size 4"
        )

    def to_manifest_dict(self) -> Dict[str, Any]:
        """Produce a structured, machine-readable dictionary representing conversion output."""
        return {
            "status": "success",
            "format": self.format_type.value,
            "source_path": self.source_path or "",
            "output_dir": str(self.output_dir),
            "files": {
                split: {
                    "path": str(p),
                    "filename": p.name,
                    "records": self.record_counts.get(split, 0),
                    "size_bytes": self.file_sizes.get(split, 0),
                }
                for split, p in self.output_files.items()
            },
            "splits": self.record_counts,
            "total_records": sum(self.record_counts.values()),
            "seed_used": self.seed_used,
            "mlx_lora_command": self.generate_mlx_lora_command(),
            "manifest_path": str(self.manifest_path) if self.manifest_path else str(self.output_dir / "mlx_manifest.json"),
        }

    def save_manifest(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Write manifest JSON to disk and record manifest_path."""
        target = Path(path).expanduser().resolve() if path else (self.output_dir / "mlx_manifest.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest_data = self.to_manifest_dict()
        manifest_data["manifest_path"] = str(target)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)
        self.manifest_path = target
        return target


def convert_and_save(
    dataset: LoadedDataset,
    format_type: MLXFormat,
    mapping: ColumnMapping,
    output_dir: Optional[Union[str, Path]] = None,
    split_config: Optional[SplitConfig] = None,
    use_existing_splits: bool = False,
    export_test_gold: bool = False,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    output_dir_str: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> ConversionResult:
    """
    Convert a loaded dataset to MLX JSONL format and write output files.
    Supports output_dir as str or Path, as well as output_dir_str for backward compatibility.
    """
    if kwargs.get("export_test_gold") or kwargs.get("test_gold"):
        export_test_gold = True

    target = output_dir if output_dir is not None else output_dir_str
    if target is None:
        target = kwargs.get("output_path")
    if target is None:
        target = getattr(dataset, "default_output_dir", None)
    if target is None:
        raise ValueError("output_dir must be specified for convert_and_save")

    output_dir = Path(target).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate mapping against dataset columns
    errors = validate_mapping(format_type, mapping, dataset.columns)
    if errors:
        raise ValueError("Invalid column mapping: " + "; ".join(errors))

    # Determine split dictionary
    splits_data: Dict[str, List[Dict[str, Any]]] = {}

    if use_existing_splits and dataset.is_split:
        for s_name in dataset.split_names:
            normalized_s = s_name.lower()
            if normalized_s in ("validation", "val", "dev"):
                key = "valid"
            elif normalized_s in ("train", "training"):
                key = "train"
            elif normalized_s in ("test", "testing", "eval"):
                key = "test"
            else:
                key = normalized_s
            splits_data[key] = list(dataset.iter_records(split=s_name))
    else:
        # Re-split all records according to SplitConfig
        all_records = dataset.get_all_records()
        if split_config is None:
            # Default 80/10/10 split
            split_config = SplitConfig(train_pct=80.0, valid_pct=10.0, test_pct=10.0, seed=42)
        splits_data = split_records(all_records, split_config)

    output_files: Dict[str, Path] = {}
    record_counts: Dict[str, int] = {}
    file_sizes: Dict[str, int] = {}
    samples: Dict[str, List[Dict[str, Any]]] = {}

    for split_name, records in splits_data.items():
        if not records:
            continue

        file_name = f"{split_name}.jsonl"
        target_path = output_dir / file_name
        output_files[split_name] = target_path

        written_count = 0
        split_samples: List[Dict[str, Any]] = []
        total_in_split = len(records)

        with open(target_path, "w", encoding="utf-8") as out_f:
            for idx, raw_record in enumerate(records, start=1):
                formatted = format_record(raw_record, format_type, mapping)
                line = json.dumps(formatted, ensure_ascii=False)
                out_f.write(line + "\n")
                written_count += 1

                if len(split_samples) < 3:
                    split_samples.append(formatted)

                if progress_callback and (idx % 200 == 0 or idx == total_in_split):
                    progress_callback(split_name, idx, total_in_split)

        record_counts[split_name] = written_count
        file_sizes[split_name] = target_path.stat().st_size
        samples[split_name] = split_samples

    # Optionally write auxiliary test_gold.jsonl for third-party eval tools
    if export_test_gold and "test" in splits_data and splits_data["test"]:
        gold_path = output_dir / "test_gold.jsonl"
        gold_written = 0
        with open(gold_path, "w", encoding="utf-8") as gf:
            for raw_record in splits_data["test"]:
                formatted = format_record(raw_record, format_type, mapping)
                if format_type == MLXFormat.CHAT and "messages" in formatted and formatted["messages"]:
                    msgs = formatted["messages"]
                    gold_row = {
                        "prompt": msgs[:-1],
                        "expected": msgs[-1].get("content", "") if msgs else "",
                    }
                elif format_type == MLXFormat.PROMPT_COMPLETION:
                    gold_row = {
                        "prompt": formatted.get("prompt", ""),
                        "expected": formatted.get("completion", ""),
                    }
                elif format_type == MLXFormat.DPO:
                    gold_row = {
                        "prompt": formatted.get("prompt", ""),
                        "expected": formatted.get("chosen", ""),
                    }
                else:
                    gold_row = {
                        "text": formatted.get("text", ""),
                    }
                gf.write(json.dumps(gold_row, ensure_ascii=False) + "\n")
                gold_written += 1

        output_files["test_gold"] = gold_path
        record_counts["test_gold"] = gold_written
        file_sizes["test_gold"] = gold_path.stat().st_size

    res = ConversionResult(
        output_dir=output_dir,
        format_type=format_type,
        output_files=output_files,
        record_counts=record_counts,
        file_sizes=file_sizes,
        sample_records=samples,
        seed_used=split_config.seed if split_config else None,
        source_path=getattr(dataset, "source_path", None),
    )

    manifest_file = kwargs.get("manifest_file")
    res.save_manifest(manifest_file if manifest_file else None)
    return res
