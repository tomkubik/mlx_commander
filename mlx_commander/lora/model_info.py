"""
Local Model Inspector for Apple MLX LoRA Fine-Tuning.
Reads model name, architecture, and key hyperparameters (layers, hidden size,
attention heads, context length, vocab size, quantization) directly from local
model files (config.json, safetensors) saved on drive.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelMetadata:
    """Detailed architecture and metadata for a base model saved on drive."""
    name: str
    path: str
    architecture: str = "Unknown"
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    context_length: Optional[int] = None
    quantization: Optional[str] = None
    file_size_gb: float = 0.0
    is_valid: bool = False
    raw_config: Dict[str, Any] = field(default_factory=dict)

    def format_hyperparameters_line(self) -> str:
        """Return a formatted string of key hyperparameters."""
        if not self.is_valid:
            return "(No local model config found on drive)"

        parts: List[str] = []
        if self.architecture and self.architecture != "Unknown":
            parts.append(f"arch={self.architecture}")
        if self.num_layers is not None:
            parts.append(f"{self.num_layers} layers")
        if self.hidden_size is not None:
            parts.append(f"{self.hidden_size} dim")
        if self.num_heads is not None:
            if self.num_kv_heads is not None and self.num_kv_heads != self.num_heads:
                parts.append(f"{self.num_heads} heads (KV: {self.num_kv_heads})")
            else:
                parts.append(f"{self.num_heads} heads")
        if self.context_length is not None:
            if self.context_length >= 1024:
                k_val = self.context_length // 1024
                parts.append(f"{k_val}k ctx ({self.context_length:,})")
            else:
                parts.append(f"{self.context_length} ctx")
        if self.vocab_size is not None:
            if self.vocab_size >= 1000:
                parts.append(f"{self.vocab_size // 1000}k vocab")
            else:
                parts.append(f"{self.vocab_size} vocab")
        if self.quantization:
            parts.append(self.quantization)

        return " │ ".join(parts) if parts else "arch=Transformer"


def _read_safetensors_header(file_path: Path) -> Optional[Dict[str, Any]]:
    """Read the JSON header metadata from a .safetensors file."""
    try:
        if not file_path.is_file() or file_path.stat().st_size < 8:
            return None
        with open(file_path, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                return None
            header_len = int.from_bytes(header_size_bytes, "little")
            if header_len <= 0 or header_len > 25 * 1024 * 1024:  # sanity check < 25MB header
                return None
            header_json = f.read(header_len).decode("utf-8", errors="ignore")
            return json.loads(header_json)
    except Exception:
        return None


def _calculate_dir_weights_size_gb(dir_path: Path) -> float:
    """Calculate total size in GB of model weight files in directory."""
    try:
        if not dir_path.is_dir():
            return 0.0
        patterns = ["*.safetensors", "*.bin", "*.mlx", "*.npz", "*.pt"]
        total_b = 0
        for pat in patterns:
            for f in dir_path.glob(pat):
                if f.is_file():
                    total_b += f.stat().st_size
        return total_b / (1024 ** 3)
    except Exception:
        return 0.0


def is_model_directory(path: Any) -> bool:
    """
    Check if a path (file or directory) corresponds to a local model directory or weights file.
    Returns True if it contains model configuration (config.json with model_type/architectures)
    or model weight files (*.safetensors, *.bin, *.mlx, etc.).
    """
    if not path:
        return False
    try:
        p = Path(path).expanduser().resolve()
        if p.is_file():
            if p.name == "config.json" or p.suffix.lower() in (".safetensors", ".bin", ".mlx", ".npz", ".pt"):
                return True
            p = p.parent
        if not p.is_dir():
            return False

        # Check for weight files
        patterns = ("*.safetensors", "*.bin", "*.mlx", "*.npz", "*.pt")
        for pat in patterns:
            if any(p.glob(pat)):
                return True
        if (p / "model.safetensors.index.json").exists() or (p / "pytorch_model.bin.index.json").exists():
            return True

        # Check config.json content
        cfg = p / "config.json"
        if cfg.is_file():
            try:
                with open(cfg, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, dict) and ("model_type" in d or "architectures" in d):
                        return True
            except Exception:
                pass

        # Check Hugging Face hub snapshots structure
        if (p / "snapshots").is_dir():
            return True

        return False
    except Exception:
        return False


def is_local_path(path_str: Optional[str]) -> bool:
    """
    Determine if a model string looks like a local filesystem path rather than a Hugging Face Hub ID.
    HF Hub IDs are in the format 'org/model' without leading slash, tilde, Windows drive letters,
    or filesystem path separators.
    """
    if not path_str or not str(path_str).strip():
        return False
    s = str(path_str).strip()
    if s.startswith(("/", "./", "../", "~")):
        return True
    if "\\" in s:
        return True
    if len(s) > 1 and s[1] == ":" and s[0].isalpha():  # Windows drive letter e.g. C:\
        return True
    if ":" in s and "/" in s:  # macOS POSIX path containing colon from Finder
        return True
    try:
        p = Path(s).expanduser().resolve()
        if p.exists():
            return True
    except Exception:
        pass
    return False


def normalize_model_path(path_str: Optional[str]) -> str:
    """
    Ensure model path resolves to the enclosing model directory.
    If the user selected an individual file inside the model directory (e.g.
    'model-00001-of-00004.safetensors', 'config.json', or 'tokenizer.json'),
    this automatically resolves and returns the model folder path.
    If pointing to a Hugging Face cache repo, resolves to the snapshot directory.
    If a macOS path has a colon/slash mismatch (e.g. 'mlx-community:Llama-3.2-3B-Instruct'
    when 'mlx-community/Llama-3.2-3B-Instruct' exists on disk, or vice versa), auto-heals it.
    """
    if not path_str or not str(path_str).strip():
        return ""

    clean = str(path_str).strip()

    def _resolve_candidate(cand_path: Path) -> Optional[str]:
        try:
            if not cand_path.exists():
                return None
            if cand_path.is_file():
                if is_model_directory(cand_path.parent) or cand_path.suffix.lower() in (".safetensors", ".bin", ".mlx", ".pt", ".npz", ".json"):
                    return str(cand_path.parent)
                return str(cand_path.parent)
            elif cand_path.is_dir():
                snaps_dir = cand_path / "snapshots"
                if snaps_dir.is_dir():
                    snaps = sorted([s for s in snaps_dir.iterdir() if s.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
                    for s in snaps:
                        if (s / "config.json").is_file():
                            return str(s)
                return str(cand_path)
        except Exception:
            pass
        return None

    try:
        p = Path(clean).expanduser().resolve()
        resolved = _resolve_candidate(p)
        if resolved:
            return resolved

        # Heuristic 1: Colon to slash
        # (e.g. /path/to/mlx-community:Llama -> /path/to/mlx-community/Llama)
        if ":" in clean:
            if len(clean) > 1 and clean[1] == ":" and clean[0].isalpha():
                drive_prefix = clean[:2]
                rest = clean[2:].replace(":", "/")
                alt_clean = drive_prefix + rest
            else:
                alt_clean = clean.replace(":", "/")
            cand = Path(alt_clean).expanduser().resolve()
            resolved = _resolve_candidate(cand)
            if resolved:
                return resolved

        # Heuristic 2: Slash to colon in last segment
        # (e.g. /path/to/mlx-community/Llama where on disk it was created as a single folder 'mlx-community:Llama')
        if "/" in clean:
            parent_p = p.parent
            if parent_p.parent.exists():
                colon_name = f"{parent_p.name}:{p.name}"
                cand = parent_p.parent / colon_name
                resolved = _resolve_candidate(cand)
                if resolved:
                    return resolved
    except Exception:
        pass

    return clean


def inspect_local_model(path_str: Optional[str]) -> ModelMetadata:
    """
    Inspect a local model saved on drive (directory or file).
    Extracts model name, architecture, and key hyperparameters from config.json
    or safetensors header.
    """
    if not path_str or not path_str.strip():
        return ModelMetadata(
            name="(No Model Selected)",
            path="",
            architecture="None",
            is_valid=False,
        )

    clean_path = path_str.strip()
    norm_path = normalize_model_path(clean_path)
    p = Path(norm_path).expanduser().resolve() if norm_path else Path(clean_path).expanduser().resolve()

    model_dir: Path
    config_file: Optional[Path] = None

    if p.is_file():
        if p.name == "config.json":
            config_file = p
            model_dir = p.parent
        else:
            # e.g. weights file selected
            model_dir = p.parent
            cand = model_dir / "config.json"
            if cand.is_file():
                config_file = cand
    elif p.is_dir():
        model_dir = p
        cand = model_dir / "config.json"
        if cand.is_file():
            config_file = cand
        else:
            # Check Hugging Face hub snapshots structure:
            # models--<org>--<name>/snapshots/<hash>/config.json
            snapshots_dir = model_dir / "snapshots"
            if snapshots_dir.is_dir():
                snaps = sorted([s for s in snapshots_dir.iterdir() if s.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
                for s in snaps:
                    c = s / "config.json"
                    if c.is_file():
                        config_file = c
                        model_dir = s
                        break
    else:
        # Path does not exist on disk
        slug = clean_path.split("/")[-1]
        return ModelMetadata(
            name=slug,
            path=clean_path,
            architecture="Not Found",
            is_valid=False,
        )

    # Compute weight size on disk
    size_gb = _calculate_dir_weights_size_gb(model_dir)
    if size_gb == 0.0 and p.is_file():
        size_gb = p.stat().st_size / (1024 ** 3)

    # Model name heuristic
    dir_name = model_dir.name
    if len(dir_name) == 40 and all(c in "0123456789abcdefABCDEF" for c in dir_name):
        # This was a snapshot commit hash; look at parent repo name
        parent_repo = model_dir.parent.parent.name
        if parent_repo.startswith("models--"):
            dir_name = parent_repo.replace("models--", "").replace("--", "/")
    model_name = dir_name

    raw_cfg: Dict[str, Any] = {}
    if config_file and config_file.is_file():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                raw_cfg = json.load(f)
        except Exception:
            raw_cfg = {}

    if raw_cfg:
        # Use _name_or_path if available and clean
        cfg_name = raw_cfg.get("_name_or_path")
        if cfg_name and isinstance(cfg_name, str) and not cfg_name.startswith("/"):
            model_name = cfg_name.split("/")[-1]

        # Architecture
        arch = raw_cfg.get("model_type")
        if not arch:
            arch_list = raw_cfg.get("architectures", [])
            if arch_list and isinstance(arch_list, list):
                arch = str(arch_list[0])
        if not arch:
            arch = "Transformer"

        # Num Layers
        layers = (
            raw_cfg.get("num_hidden_layers")
            or raw_cfg.get("n_layer")
            or raw_cfg.get("num_layers")
        )
        if layers is not None:
            try:
                layers = int(layers)
            except (ValueError, TypeError):
                layers = None

        # Hidden Dimension
        hidden = (
            raw_cfg.get("hidden_size")
            or raw_cfg.get("d_model")
            or raw_cfg.get("n_embd")
        )
        if hidden is not None:
            try:
                hidden = int(hidden)
            except (ValueError, TypeError):
                hidden = None

        # Attention Heads
        heads = (
            raw_cfg.get("num_attention_heads")
            or raw_cfg.get("n_head")
        )
        if heads is not None:
            try:
                heads = int(heads)
            except (ValueError, TypeError):
                heads = None

        # Key-Value Heads (for GQA / MQA)
        kv_heads = (
            raw_cfg.get("num_key_value_heads")
            or raw_cfg.get("n_head_kv")
        )
        if kv_heads is not None:
            try:
                kv_heads = int(kv_heads)
            except (ValueError, TypeError):
                kv_heads = None

        # Vocab Size
        vocab = raw_cfg.get("vocab_size")
        if vocab is not None:
            try:
                vocab = int(vocab)
            except (ValueError, TypeError):
                vocab = None

        # Context Length
        ctx = (
            raw_cfg.get("max_position_embeddings")
            or raw_cfg.get("max_sequence_length")
            or raw_cfg.get("context_length")
            or raw_cfg.get("seq_length")
        )
        if ctx is not None:
            try:
                ctx = int(ctx)
            except (ValueError, TypeError):
                ctx = None

        # Quantization
        quant_str: Optional[str] = None
        q_cfg = raw_cfg.get("quantization") or raw_cfg.get("quantization_config")
        if isinstance(q_cfg, dict):
            bits = q_cfg.get("bits") or q_cfg.get("quant_method")
            grp = q_cfg.get("group_size")
            if bits and grp:
                quant_str = f"{bits}-bit (group {grp})"
            elif bits:
                quant_str = f"{bits}-bit"
        elif "quantization" in raw_cfg and isinstance(raw_cfg["quantization"], str):
            quant_str = raw_cfg["quantization"]

        if not quant_str:
            # Check torch_dtype or path heuristics
            dtype = raw_cfg.get("torch_dtype")
            low_name = str(clean_path).lower()
            if "4bit" in low_name or "int4" in low_name or "q4" in low_name:
                quant_str = "4-bit"
            elif "8bit" in low_name or "int8" in low_name or "q8" in low_name:
                quant_str = "8-bit"
            elif dtype:
                quant_str = str(dtype)
            else:
                quant_str = "16-bit (unquantized)"

        return ModelMetadata(
            name=model_name,
            path=str(model_dir),
            architecture=str(arch),
            num_layers=layers,
            hidden_size=hidden,
            num_heads=heads,
            num_kv_heads=kv_heads,
            vocab_size=vocab,
            context_length=ctx,
            quantization=quant_str,
            file_size_gb=round(size_gb, 2) if size_gb >= 0.01 else (round(size_gb, 4) if size_gb > 0 else 0.0),
            is_valid=True,
            raw_config=raw_cfg,
        )

    # Fallback: Check for safetensors files in directory if no config.json
    sf_files = list(model_dir.glob("*.safetensors"))
    if sf_files:
        sf_header = _read_safetensors_header(sf_files[0])
        layers = None
        hidden = None
        vocab = None
        if sf_header:
            layer_indices = set()
            for k, v in sf_header.items():
                if not isinstance(v, dict):
                    continue
                # Match e.g. model.layers.27.
                m = re.search(r'layers\.(\d+)\.', k)
                if m:
                    layer_indices.add(int(m.group(1)))
                if "embed_tokens.weight" in k and "shape" in v:
                    shape = v["shape"]
                    if len(shape) == 2:
                        vocab, hidden = shape[0], shape[1]

            if layer_indices:
                layers = max(layer_indices) + 1

        low_name = str(clean_path).lower()
        quant = "4-bit" if ("4bit" in low_name or "int4" in low_name) else "Safetensors"

        return ModelMetadata(
            name=model_name,
            path=str(model_dir),
            architecture="MLX / Safetensors",
            num_layers=layers,
            hidden_size=hidden,
            vocab_size=vocab,
            quantization=quant,
            file_size_gb=round(size_gb, 2),
            is_valid=True,
        )

    # No config.json or safetensors found
    return ModelMetadata(
        name=model_name,
        path=str(model_dir),
        architecture="Local Directory",
        file_size_gb=round(size_gb, 2),
        is_valid=False,
    )


def scan_local_models(search_dirs: Optional[List[str]] = None) -> List[ModelMetadata]:
    """
    Scan common local directories on drive for downloaded base models.
    Checks ./models, ~/.cache/huggingface/hub, and current directory.
    """
    if search_dirs is None:
        search_dirs = [
            "./models",
            "../models",
            "./",
            "~/.cache/huggingface/hub",
        ]

    discovered: List[ModelMetadata] = []
    seen_paths = set()

    for d_str in search_dirs:
        try:
            d = Path(d_str).expanduser().resolve()
            if not d.is_dir():
                continue

            # Look for subdirectories containing config.json
            for item in d.iterdir():
                if not item.is_dir() or item.name.startswith("."):
                    continue

                if item.name.startswith("models--"):
                    # Hugging Face cache structure
                    snaps = item / "snapshots"
                    if snaps.is_dir():
                        for s in snaps.iterdir():
                            if s.is_dir() and (s / "config.json").is_file():
                                if str(s) not in seen_paths:
                                    seen_paths.add(str(s))
                                    meta = inspect_local_model(str(s))
                                    if meta.is_valid:
                                        discovered.append(meta)
                else:
                    if (item / "config.json").is_file() or list(item.glob("*.safetensors")):
                        if str(item) not in seen_paths:
                            seen_paths.add(str(item))
                            meta = inspect_local_model(str(item))
                            if meta.is_valid:
                                discovered.append(meta)
        except Exception:
            continue

    return discovered
