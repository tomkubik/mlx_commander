"""
Local Model Inspector for Apple MLX LoRA Fine-Tuning.
Reads model name, architecture, and key hyperparameters (layers, hidden size,
attention heads, context length, vocab size, quantization) directly from local
model files (config.json, safetensors) saved on drive.
"""

import importlib.util
import json
import os
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


KNOWN_VLM_MODEL_TYPES = {
    "gemma4",
    "gemma4_unified",
    "gemma3n",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_omni_moe",
    "llava",
    "llava_next",
    "llava_onevision",
    "paligemma",
    "paligemma2",
    "idefics",
    "idefics2",
    "idefics3",
    "pixtral",
    "molmo",
    "molmo2",
    "smolvlm",
    "phi3_v",
    "phi4mm",
    "mllama",
    "deepseek_vl",
    "deepseek_vl_v2",
    "internvl",
    "minicpmv",
    "minicpmo",
}


def detect_model_engine(
    raw_config: Optional[Dict[str, Any]] = None,
    model_name_or_path: Optional[str] = None,
) -> str:
    """
    Detect whether a model requires 'mlx_vlm' (Vision-Language / Multimodal Model)
    or standard 'mlx_lm' based on config.json parameters or model name/path.
    """
    if raw_config:
        # 1. Direct VLM configuration sub-dicts or multimodal token IDs
        vlm_keys = ("vision_config", "audio_config", "image_token_id", "video_token_id")
        for k in vlm_keys:
            if k in raw_config:
                return "mlx_vlm"

        # 2. Model type check
        model_type = str(raw_config.get("model_type", "")).lower()
        if model_type in KNOWN_VLM_MODEL_TYPES:
            return "mlx_vlm"
        for vlm_type in KNOWN_VLM_MODEL_TYPES:
            if vlm_type in model_type:
                return "mlx_vlm"

        # 3. Architectures check
        archs = raw_config.get("architectures", [])
        if isinstance(archs, list):
            for arch in archs:
                arch_str = str(arch).lower()
                if any(
                    term in arch_str
                    for term in (
                        "vision",
                        "vlm",
                        "vl",
                        "paligemma",
                        "llava",
                        "gemma4",
                        "gemma3n",
                        "mllama",
                        "smolvlm",
                        "pixtral",
                        "molmo",
                        "idefics",
                    )
                ):
                    return "mlx_vlm"

    if model_name_or_path:
        low_str = str(model_name_or_path).lower()
        vlm_name_indicators = (
            "gemma4",
            "gemma-4",
            "gemma_4",
            "gemma3n",
            "gemma-3n",
            "gemma_3n",
            "qwen2-vl",
            "qwen2_vl",
            "qwen2.5-vl",
            "qwen3-vl",
            "qwen3_vl",
            "paligemma",
            "llava",
            "smolvlm",
            "pixtral",
            "molmo",
            "idefics",
            "phi3-v",
            "phi3_v",
            "phi4mm",
            "mllama",
            "minicpm-v",
            "minicpmv",
            "internvl",
            "deepseek-vl",
            "deepseek_vl",
            "-vl-",
            "-vl",
            "_vl",
        )
        for ind in vlm_name_indicators:
            if ind in low_str:
                return "mlx_vlm"

    return "mlx_lm"


def is_engine_installed(engine: str) -> bool:
    """Check if the given engine ('mlx_lm' or 'mlx_vlm') is installed and importable."""
    try:
        return importlib.util.find_spec(engine) is not None
    except Exception:
        return False


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
    engine: str = "mlx_lm"  # "mlx_lm" or "mlx_vlm"
    is_vlm: bool = False

    def format_hyperparameters_line(self) -> str:
        """Return a formatted string of key hyperparameters."""
        if not self.is_valid:
            return "(No local model config found on drive)"

        parts: List[str] = []
        if self.is_vlm or self.engine == "mlx_vlm":
            parts.append("engine=mlx-vlm")
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
    If the user selected a parent container folder, auto-discovers the model subfolder.
    If a path has a colon/slash mismatch at any directory level (e.g. on macOS APFS
    where slashes in Finder folder names are stored as colons on disk, or vice versa),
    dynamically heals it across path segments.
    """
    if not path_str or not str(path_str).strip():
        return ""

    raw_val = str(path_str).strip()

    # 1. Clean input variations (quotes, file:// scheme, trailing slashes, URL encoding)
    def _clean_variants(s: str) -> List[str]:
        variants = []
        base = s.strip().strip("\"'\"")
        if base.startswith("file://"):
            base = base[7:]
            if base.startswith("localhost/"):
                base = base[9:]
        if len(base) > 1 and base.endswith(("/", "\\")):
            base = base.rstrip("/\\")
        variants.append(base)

        # Unquoted URL encoding (e.g. %20 -> space)
        if "%" in base:
            unquoted = urllib.parse.unquote(base)
            if unquoted not in variants:
                variants.append(unquoted)

        # Unescaped shell backslash escapes (e.g. \ )
        if "\\" in base and not (len(base) > 2 and base[1] == ":"):
            unescaped = base.replace(r"\ ", " ").replace(r"\:", ":")
            if unescaped not in variants:
                variants.append(unescaped)

        # Dash normalization (Unicode en-dash \u2013 and em-dash \u2014 to ASCII -)
        for v in list(variants):
            if "–" in v or "—" in v:
                norm_dash = v.replace("–", "-").replace("—", "-")
                if norm_dash not in variants:
                    variants.append(norm_dash)

        return variants

    def _resolve_candidate(cand_path: Path) -> Optional[str]:
        try:
            if not cand_path.exists():
                return None
            cand_res = cand_path.resolve()
            if cand_res.is_file():
                if is_model_directory(cand_res.parent) or cand_res.suffix.lower() in (
                    ".safetensors", ".bin", ".mlx", ".pt", ".npz", ".json"
                ):
                    return str(cand_res.parent.resolve())
                return str(cand_res.parent.resolve())
            elif cand_res.is_dir():
                # Check HF hub snapshots
                snaps_dir = cand_res / "snapshots"
                if snaps_dir.is_dir():
                    snaps = sorted(
                        [s for s in snaps_dir.iterdir() if s.is_dir()],
                        key=lambda x: x.stat().st_mtime,
                        reverse=True,
                    )
                    for s in snaps:
                        if (s / "config.json").is_file():
                            return str(s.resolve())

                if is_model_directory(cand_res):
                    return str(cand_res.resolve())

                # Auto-dive into immediate children if user selected parent container
                try:
                    children = list(cand_res.iterdir())
                    model_subdirs = [c for c in children if c.is_dir() and is_model_directory(c)]
                    if len(model_subdirs) == 1:
                        return str(model_subdirs[0].resolve())
                    elif len(model_subdirs) > 1:
                        # Prioritize child with hyphen or model indicators
                        hyphen_subs = [c for c in model_subdirs if "-" in c.name]
                        if hyphen_subs:
                            return str(hyphen_subs[0].resolve())
                        return str(model_subdirs[0].resolve())

                    # Check 2 levels deep (e.g. Models/org/model-name)
                    for c in children:
                        if c.is_dir():
                            deep_subs = [d for d in c.iterdir() if d.is_dir() and is_model_directory(d)]
                            if len(deep_subs) == 1:
                                return str(deep_subs[0].resolve())
                            elif len(deep_subs) > 1:
                                hyphen_deep = [d for d in deep_subs if "-" in d.name]
                                if hyphen_deep:
                                    return str(hyphen_deep[0].resolve())
                                return str(deep_subs[0].resolve())
                except Exception:
                    pass

                return str(cand_res.resolve())
        except Exception:
            pass
        return None

    def _heal_path_segments(target_path: str) -> Optional[str]:
        try:
            target = Path(target_path).expanduser()
            parts = list(target.parts)
            if not parts:
                return None

            # Find deepest existing ancestor root
            idx = len(parts)
            while idx > 0:
                cand = Path(*parts[:idx])
                if cand.exists():
                    break
                idx -= 1

            if idx == 0:
                return None

            current = Path(*parts[:idx])
            remaining = parts[idx:]

            def match_sub(curr_dir: Path, rem_parts: List[str]) -> Optional[Path]:
                if not rem_parts:
                    return curr_dir
                if not curr_dir.is_dir():
                    return None

                try:
                    entries = list(curr_dir.iterdir())
                except Exception:
                    return None

                def norm_name(name: str) -> str:
                    return name.replace("–", "-").replace("—", "-")

                p0 = rem_parts[0]
                p0_norm = norm_name(p0)

                # 1. Direct match for rem_parts[0]
                for e in entries:
                    if e.name == p0 or norm_name(e.name) == p0_norm:
                        res = match_sub(e, rem_parts[1:])
                        if res:
                            return res

                # 2. Multi-part colon join: e.g. rem_parts[0]:rem_parts[1]
                for k in range(2, len(rem_parts) + 1):
                    combined = ":".join(rem_parts[:k])
                    comb_norm = norm_name(combined)
                    for e in entries:
                        if e.name == combined or norm_name(e.name) == comb_norm:
                            res = match_sub(e, rem_parts[k:])
                            if res:
                                return res

                # 3. If rem_parts[0] has colons, split and try nested match
                if ":" in p0:
                    sub_parts = [p for p in p0.split(":") if p] + list(rem_parts[1:])
                    res = match_sub(curr_dir, sub_parts)
                    if res:
                        return res

                # 4. If remaining parts is a parent prefix (e.g. user selected mlx-community)
                # Look for directory entries starting with prefix + ":"
                if len(rem_parts) == 1:
                    prefix_colon = p0 + ":"
                    prefix_colon_lower = p0.lower() + ":"
                    for e in entries:
                        if e.name.startswith(prefix_colon) or e.name.lower().startswith(prefix_colon_lower):
                            return e

                # 5. Case-insensitive matching fallback
                for e in entries:
                    if e.name.lower() == p0.lower() or norm_name(e.name).lower() == p0_norm.lower():
                        res = match_sub(e, rem_parts[1:])
                        if res:
                            return res

                return None

            matched = match_sub(current, remaining)
            if matched and matched.exists():
                return _resolve_candidate(matched)
        except Exception:
            pass
        return None

    variants = _clean_variants(raw_val)

    # Pass 1: Direct resolution of variants
    for v in variants:
        try:
            p = Path(v).expanduser().resolve()
            resolved = _resolve_candidate(p)
            if resolved:
                return resolved
        except Exception:
            pass

    # Pass 2: Deepest-ancestor colon/slash healing
    for v in variants:
        healed = _heal_path_segments(v)
        if healed:
            return healed

    # Pass 3: Global colon-to-slash replacement
    for v in variants:
        if ":" in v:
            if len(v) > 1 and v[1] == ":" and v[0].isalpha():
                drive_prefix = v[:2]
                rest = v[2:].replace(":", "/")
                alt_clean = drive_prefix + rest
            else:
                alt_clean = v.replace(":", "/")
            try:
                cand = Path(alt_clean).expanduser().resolve()
                resolved = _resolve_candidate(cand)
                if resolved:
                    return resolved
                healed = _heal_path_segments(alt_clean)
                if healed:
                    return healed
            except Exception:
                pass

    # Pass 4: Check relative to standard local model locations if relative
    if not is_local_path(variants[0]):
        for loc in (Path.cwd(), Path.home() / "Models", Path.home() / "Downloads"):
            for v in variants:
                loc_cand = loc / v
                try:
                    if loc_cand.exists():
                        res = _resolve_candidate(loc_cand)
                        if res:
                            return res
                    healed = _heal_path_segments(str(loc_cand))
                    if healed:
                        return healed
                except Exception:
                    pass

    return variants[0]



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
        # Path does not exist on disk (could be a Hugging Face model ID or missing folder)
        slug = clean_path.split("/")[-1]
        engine = detect_model_engine(model_name_or_path=clean_path)
        is_vlm = (engine == "mlx_vlm")
        return ModelMetadata(
            name=slug,
            path=clean_path,
            architecture="HuggingFace Hub / Remote" if not is_local_path(clean_path) else "Not Found",
            is_valid=False,
            engine=engine,
            is_vlm=is_vlm,
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

        engine = detect_model_engine(raw_config=raw_cfg, model_name_or_path=clean_path)
        is_vlm = (engine == "mlx_vlm")
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
            engine=engine,
            is_vlm=is_vlm,
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

        engine = detect_model_engine(model_name_or_path=clean_path)
        is_vlm = (engine == "mlx_vlm")
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
            engine=engine,
            is_vlm=is_vlm,
        )

    # No config.json or safetensors found
    engine = detect_model_engine(model_name_or_path=clean_path)
    is_vlm = (engine == "mlx_vlm")
    return ModelMetadata(
        name=model_name,
        path=str(model_dir),
        architecture="Local Directory",
        file_size_gb=round(size_gb, 2),
        is_valid=False,
        engine=engine,
        is_vlm=is_vlm,
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
