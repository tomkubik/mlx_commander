"""
Dataset Inspection and Loading for Hugging Face datasets stored on disk.
Supports:
- Hugging Face datasets saved with `dataset.save_to_disk()`
- Hugging Face DatasetDict saved with `dataset_dict.save_to_disk()`
- Local Parquet, Arrow, JSON, JSONL, and CSV files/folders
- Hugging Face cache directories
- Fallback loaders when `datasets` library is not yet installed
"""

import csv
import io
import json
import os
import sqlite3
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from mlx_commander.exceptions import MissingDependencyError

try:
    import datasets
    from datasets import Dataset, DatasetDict, load_from_disk
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    datasets = None
    Dataset = None
    DatasetDict = None
    load_from_disk = None

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.ipc as pa_ipc
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    pa = None
    ds = None
    pa_ipc = None
    pq = None

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    duckdb = None

try:
    import lance
    HAS_LANCE = True
except ImportError:
    HAS_LANCE = False
    lance = None


def ensure_pyarrow() -> bool:
    """
    Ensure pyarrow and its submodules are imported and ready.
    If pyarrow was installed while the application was running, this re-attempts
    the import and updates global references dynamically so datasets open seamlessly.
    Also probes standard user/system site-packages if running in an isolated environment.
    """
    global HAS_PYARROW, pa, ds, pa_ipc, pq
    if HAS_PYARROW and pq is not None:
        return True
    try:
        import pyarrow as _pa
        import pyarrow.dataset as _ds
        import pyarrow.ipc as _pa_ipc
        import pyarrow.parquet as _pq
        pa = _pa
        ds = _ds
        pa_ipc = _pa_ipc
        pq = _pq
        HAS_PYARROW = True
        return True
    except ImportError:
        pass

    # Probe standard site-packages in case pyarrow is installed in user/system packages
    try:
        import site
        candidate_dirs: List[Path] = []
        try:
            user_site = site.getusersitepackages()
            if user_site:
                candidate_dirs.append(Path(user_site))
        except Exception:
            pass

        home = Path.home()
        for ver in ("3.9", "3.10", "3.11", "3.12", "3.13"):
            candidate_dirs.append(home / f"Library/Python/{ver}/lib/python/site-packages")
            candidate_dirs.append(home / f".local/lib/python{ver}/site-packages")
            candidate_dirs.append(Path(f"/opt/homebrew/lib/python{ver}/site-packages"))
            candidate_dirs.append(Path(f"/usr/local/lib/python{ver}/site-packages"))

        # Also search uv wheel/archive caches
        uv_archive = home / ".cache/uv/archive-v0"
        if uv_archive.is_dir():
            for pth in uv_archive.glob("*/lib/python*/site-packages"):
                if (pth / "pyarrow").is_dir():
                    candidate_dirs.append(pth)

        for c_dir in candidate_dirs:
            if (c_dir / "pyarrow").is_dir() and str(c_dir) not in sys.path:
                sys.path.append(str(c_dir))
                try:
                    import pyarrow as _pa
                    import pyarrow.dataset as _ds
                    import pyarrow.ipc as _pa_ipc
                    import pyarrow.parquet as _pq
                    pa = _pa
                    ds = _ds
                    pa_ipc = _pa_ipc
                    pq = _pq
                    HAS_PYARROW = True
                    return True
                except Exception:
                    continue
    except Exception:
        pass

    HAS_PYARROW = False
    return False


def ensure_duckdb() -> bool:
    """Ensure duckdb is imported and ready dynamically."""
    global HAS_DUCKDB, duckdb
    if HAS_DUCKDB and duckdb is not None:
        return True
    try:
        import duckdb as _duckdb
        duckdb = _duckdb
        HAS_DUCKDB = True
        return True
    except ImportError:
        HAS_DUCKDB = False
        return False


def ensure_lance() -> bool:
    """Ensure lance is imported and ready dynamically."""
    global HAS_LANCE, lance
    if HAS_LANCE and lance is not None:
        return True
    try:
        import lance as _lance
        lance = _lance
        HAS_LANCE = True
        return True
    except ImportError:
        HAS_LANCE = False
        return False


def ensure_datasets() -> bool:
    """Ensure datasets is imported and ready dynamically."""
    global HAS_DATASETS, datasets, Dataset, DatasetDict, load_from_disk
    if HAS_DATASETS and datasets is not None:
        return True
    try:
        import datasets as _datasets
        from datasets import Dataset as _Dataset, DatasetDict as _DatasetDict, load_from_disk as _load_from_disk
        datasets = _datasets
        Dataset = _Dataset
        DatasetDict = _DatasetDict
        load_from_disk = _load_from_disk
        HAS_DATASETS = True
        return True
    except ImportError:
        HAS_DATASETS = False
        return False


# Magic byte signatures for binary formats
PARQUET_MAGIC = b"PAR1"
ARROW_IPC_FILE_MAGIC = b"ARROW1"
SQLITE_MAGIC = b"SQLite format 3\x00"
PARQUET_EXTS = (".parquet", ".pq", ".parq")


def is_parquet_file(file_path: Path) -> bool:
    """
    Check if a file is a Parquet file by extension or magic bytes (PAR1).
    Guarantees reliable detection even if file has an alternative extension
    (.pq, .parq, .parquet.snappy, .snappy) or no extension at all.
    """
    if not file_path.is_file():
        return False
    name_lower = file_path.name.lower()
    if any(name_lower.endswith(ext) for ext in PARQUET_EXTS) or ".parquet." in name_lower:
        return True
    try:
        with open(file_path, "rb") as f:
            header = f.read(4)
            if header == PARQUET_MAGIC:
                return True
    except (OSError, PermissionError):
        pass
    return False


def is_arrow_file(file_path: Path) -> bool:
    """Check if file is an Arrow IPC file by extension or magic bytes."""
    if not file_path.is_file():
        return False
    if file_path.name.lower().endswith(".arrow"):
        return True
    try:
        with open(file_path, "rb") as f:
            header = f.read(6)
            if header == ARROW_IPC_FILE_MAGIC:
                return True
    except (OSError, PermissionError):
        pass
    return False


def is_sqlite_file(file_path: Path) -> bool:
    """Check if file is a SQLite database file by extension or magic bytes."""
    if not file_path.is_file():
        return False
    if any(file_path.name.lower().endswith(ext) for ext in (".sqlite", ".db", ".sqlite3")):
        return True
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)
            if header == SQLITE_MAGIC:
                return True
    except (OSError, PermissionError):
        pass
    return False


def is_parquet_dir(dir_path: Path) -> bool:
    """
    Check if directory contains Parquet dataset files (root level or common subfolders).
    """
    if not dir_path.is_dir():
        return False
    try:
        # Check root level
        for f in dir_path.iterdir():
            if f.is_file():
                name_l = f.name.lower()
                if any(name_l.endswith(ext) for ext in PARQUET_EXTS) or ".parquet." in name_l:
                    return True
                if f.stat().st_size >= 4:
                    with open(f, "rb") as bf:
                        if bf.read(4) == PARQUET_MAGIC:
                            return True

        # Check common subdirectories like data/, train/, test/
        for sub in ("data", "train", "test", "valid", "validation", "dev"):
            sub_dir = dir_path / sub
            if sub_dir.is_dir():
                for f in sub_dir.iterdir():
                    if f.is_file():
                        name_l = f.name.lower()
                        if any(name_l.endswith(ext) for ext in PARQUET_EXTS) or ".parquet." in name_l:
                            return True
                        if f.stat().st_size >= 4:
                            with open(f, "rb") as bf:
                                if bf.read(4) == PARQUET_MAGIC:
                                    return True
    except (OSError, PermissionError):
        pass
    return False


@dataclass
class LoadedDataset:
    source_path: str
    is_split: bool
    split_names: List[str]
    split_counts: Dict[str, int]
    columns: List[str]
    total_rows: int
    sample_records: List[Dict[str, Any]]
    base_dir: Optional[Path] = None
    features: Dict[str, Any] = field(default_factory=dict)
    _raw_splits: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def default_output_dir(self) -> Path:
        """
        Default destination folder: a subfolder named 'mlx_dataset' in the
        exact folder where the Hugging Face dataset was loaded from.
        """
        if self.base_dir is not None:
            return self.base_dir / "mlx_dataset"
        try:
            raw = self.source_path.split("::")[0].strip()
            p = Path(raw).resolve()
            if p.is_dir():
                return p / "mlx_dataset"
            return p.parent / "mlx_dataset"
        except Exception:
            return Path.cwd() / "mlx_dataset"


    def iter_records(self, split: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """Yield records as standard Python dicts for the specified split or all records."""
        target_splits = [split] if split else self.split_names
        for s_name in target_splits:
            data = self._raw_splits.get(s_name)
            if data is None:
                continue

            if HAS_DATASETS and isinstance(data, (Dataset, datasets.Dataset)):
                for row in data:
                    yield dict(row)
            elif HAS_PYARROW and isinstance(data, pa.Table):
                for batch in data.to_batches():
                    pydicts = batch.to_pylist()
                    for r in pydicts:
                        yield r
            elif isinstance(data, list):
                for item in data:
                    yield item
            elif callable(data):
                for item in data():
                    yield item

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Return all records across all splits as a list."""
        return list(self.iter_records())


def detect_column_discrete_labels(
    dataset: LoadedDataset,
    column: Optional[str],
    max_classes: int = 12,
    max_sample_rows: int = 200,
) -> Optional[Dict[str, Any]]:
    """
    Detect if a target/label column in a dataset contains discrete categories or class indices.
    Extracts Hugging Face ClassLabel names if present in dataset.features, or detects
    small integer sets (e.g. 0, 1, 2) from sample records.

    Returns a dict with:
      - 'discrete': True
      - 'unique_values': List of unique stringified labels found, e.g. ['0', '1', '2']
      - 'suggested_map': Optional Dict[str, str] mapping indices to semantic names
      - 'hf_class_names': Optional List[str] of canonical HF class names
      - 'is_integer_labels': bool indicating if raw values are numeric codes
    Or None if the column is continuous text or has more than max_classes unique values.
    """
    if not dataset or not column or column not in dataset.columns:
        return None

    # 1. Check dataset.features (Hugging Face ClassLabel or Feature dict)
    feat = dataset.features.get(column) if dataset.features else None
    if feat is not None:
        names = getattr(feat, "names", None)
        if names is None and isinstance(feat, dict) and "names" in feat:
            names = feat["names"]
        if names and isinstance(names, (list, tuple)):
            names_list = [str(n) for n in names]
            indices = [str(i) for i in range(len(names_list))]
            return {
                "discrete": True,
                "unique_values": indices,
                "suggested_map": {str(i): names_list[i] for i in range(len(names_list))},
                "hf_class_names": names_list,
                "is_integer_labels": True,
            }

    # 2. Inspect sample records from LoadedDataset
    sampled_vals: List[Any] = []
    for r in dataset.sample_records:
        if column in r and r[column] is not None:
            sampled_vals.append(r[column])

    # If sample_records is small (< max_sample_rows), pull up to max_sample_rows using iter_records
    if len(sampled_vals) < max_sample_rows:
        count = len(sampled_vals)
        for r in dataset.iter_records():
            if column in r and r[column] is not None:
                sampled_vals.append(r[column])
                count += 1
                if count >= max_sample_rows:
                    break

    if not sampled_vals:
        return None

    unique_set = set()
    is_all_int_like = True
    for v in sampled_vals:
        if isinstance(v, bool):
            unique_set.add(int(v))
        elif isinstance(v, int):
            unique_set.add(v)
        elif isinstance(v, float) and v.is_integer():
            unique_set.add(int(v))
        elif isinstance(v, str) and v.strip().lstrip("-").isdigit():
            unique_set.add(int(v.strip()))
        else:
            is_all_int_like = False
            unique_set.add(str(v).strip())

    if len(unique_set) < 2 or len(unique_set) > max_classes:
        return None

    sorted_vals = sorted(list(unique_set), key=lambda x: (isinstance(x, str), x))
    str_vals = [str(x) for x in sorted_vals]

    return {
        "discrete": True,
        "unique_values": str_vals,
        "suggested_map": None,
        "hf_class_names": None,
        "is_integer_labels": is_all_int_like,
    }


def _normalize_path_list(path_input: Union[str, Path, List[Union[str, Path]]]) -> List[str]:
    """Normalize input into a clean list of individual path strings."""
    if isinstance(path_input, list):
        out = []
        for item in path_input:
            out.extend(_normalize_path_list(item))
        return out
    if isinstance(path_input, Path):
        return [str(path_input)]
    if isinstance(path_input, str):
        lines = [l.strip() for l in path_input.splitlines() if l.strip()]
        if len(lines) > 1:
            return lines
        if "," in path_input and not Path(path_input).exists():
            parts = [p.strip() for p in path_input.split(",") if p.strip()]
            if len(parts) > 1:
                return parts
        if path_input.strip():
            return [path_input.strip()]
    return []


def _split_file_and_table(path_str: str) -> Tuple[Path, Optional[str]]:
    """Parse potential table specifier in file path, e.g. 'data.db::my_table'."""
    if "::" in path_str:
        f_part, _, t_part = path_str.partition("::")
        f_path = Path(f_part.strip()).expanduser().resolve()
        return f_path, t_part.strip() or None
    return Path(path_str).expanduser().resolve(), None


def inspect_dataset_path(path_input: Union[str, Path, List[Union[str, Path]]]) -> Tuple[bool, str]:
    """
    Validate path(s) and return (is_valid, error_or_info_message).
    Supports single file/folder, newline-separated, or comma-separated list of files,
    as well as SQLite table syntax ('file.sqlite::table_name').
    """
    raw_paths = _normalize_path_list(path_input)
    if not raw_paths:
        return False, "No dataset path provided."

    if len(raw_paths) == 1:
        f_path, t_name = _split_file_and_table(raw_paths[0])
        if not f_path.exists():
            return False, f"Path does not exist: {f_path}"
        if t_name:
            return True, f"Found: {f_path} (Table: {t_name})"
        return True, f"Found: {f_path}"

    missing = []
    for rp in raw_paths:
        f_path, _ = _split_file_and_table(rp)
        if not f_path.exists():
            missing.append(str(f_path))
    if missing:
        return False, f"Missing file(s): {', '.join(missing)}"
    return True, f"Found {len(raw_paths)} files to merge."


def load_from_json_or_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSON lines or standard JSON array from file."""
    if is_parquet_file(file_path):
        return load_from_parquet_file(file_path)
    if is_arrow_file(file_path):
        return load_from_arrow_file(file_path)

    records: List[Dict[str, Any]] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Check first non-whitespace char
            pos = f.tell()
            first_char = f.read(1)
            while first_char and first_char.isspace():
                first_char = f.read(1)
            f.seek(pos)

            if first_char == "[":
                # JSON array
                data = json.load(f)
                if isinstance(data, list):
                    records = [d for d in data if isinstance(d, dict)]
            else:
                # JSONL
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        try:
                            obj = json.loads(line_str)
                            if isinstance(obj, dict):
                                records.append(obj)
                        except json.JSONDecodeError:
                            continue
        return records
    except UnicodeDecodeError as ue:
        if is_parquet_file(file_path):
            return load_from_parquet_file(file_path)
        if is_arrow_file(file_path):
            return load_from_arrow_file(file_path)
        raise ValueError(
            f"File '{file_path.name}' contains binary or non-UTF-8 data ({ue}) and cannot be loaded as JSON/JSONL."
        )


def load_from_csv(file_path: Path) -> List[Dict[str, Any]]:
    """Load records from CSV file."""
    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def load_from_tsv(file_path: Path) -> List[Dict[str, Any]]:
    """Load records from Tab-Separated Values (TSV) file."""
    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records.append(dict(row))
    return records


def load_from_text(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load records from a plain text file (.txt, .text).
    Supports multiple encodings with automatic fallbacks:
    1. utf-8-sig (detects and cleanly removes UTF-8 BOM)
    2. utf-8
    3. utf-16 (handles UTF-16-LE / UTF-16-BE with BOM)
    4. latin-1 / cp1252 (fallback that never fails decoding)

    Converts each non-empty line into a record: {"text": line}
    """
    raw_bytes = file_path.read_bytes()
    decoded_text: Optional[str] = None
    for enc in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            decoded_text = raw_bytes.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if decoded_text is None:
        decoded_text = raw_bytes.decode("utf-8", errors="replace")

    records: List[Dict[str, Any]] = []
    for line in decoded_text.splitlines():
        cleaned = line.strip()
        if cleaned:
            records.append({"text": cleaned})
    return records


def load_from_sqlite(
    file_path: Path,
    table_name: Optional[str] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], bool]:
    """
    Load records from a SQLite database (.sqlite, .db, .sqlite3).
    Returns (raw_splits, is_split).
    Supports:
    - Explicit table selection via table_name or 'file.db::table_name'.
    - Automatic mapping of split tables ('train', 'test', 'valid', 'validation', 'val', 'dev').
    - Single table database auto-loading.
    - Multi-table schema matching auto-splitting.
    """
    db_path = str(file_path.resolve())
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except Exception:
        conn = sqlite3.connect(db_path)

    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        if not tables:
            raise ValueError(f"No user tables found in SQLite database '{file_path}'.")

        if table_name:
            if table_name not in tables:
                raise ValueError(
                    f"Table '{table_name}' not found in SQLite database '{file_path.name}'. Available tables: {', '.join(tables)}"
                )
            cursor.execute(f'SELECT * FROM "{table_name}"')
            rows = [dict(r) for r in cursor.fetchall()]
            return {"default": rows}, False

        # Check if table names match standard split names
        split_map: Dict[str, str] = {}
        for t in tables:
            t_lower = t.lower()
            if t_lower in ("train", "training"):
                split_map["train"] = t
            elif t_lower in ("valid", "validation", "val", "dev"):
                split_map["valid"] = t
            elif t_lower in ("test", "testing", "eval", "evaluation"):
                split_map["test"] = t

        if split_map:
            raw_splits: Dict[str, List[Dict[str, Any]]] = {}
            for s_name, t_name in split_map.items():
                cursor.execute(f'SELECT * FROM "{t_name}"')
                raw_splits[s_name] = [dict(r) for r in cursor.fetchall()]
            return raw_splits, len(raw_splits) > 1

        # If only 1 table
        if len(tables) == 1:
            t = tables[0]
            cursor.execute(f'SELECT * FROM "{t}"')
            rows = [dict(r) for r in cursor.fetchall()]
            return {"default": rows}, False

        # If multiple tables, check for primary table names
        primary_candidates = [
            t for t in tables
            if t.lower() in ("data", "dataset", "records", "items", "samples", "main")
        ]
        if primary_candidates:
            primary_table = primary_candidates[0]
            cursor.execute(f'SELECT * FROM "{primary_table}"')
            rows = [dict(r) for r in cursor.fetchall()]
            return {"default": rows}, False

        # Check if table schemas match
        table_cols = {}
        for t in tables:
            cursor.execute(f'PRAGMA table_info("{t}")')
            table_cols[t] = [r["name"] for r in cursor.fetchall()]

        first_cols = table_cols[tables[0]]
        if all(table_cols[t] == first_cols for t in tables[1:]):
            raw_splits = {}
            for t in tables:
                cursor.execute(f'SELECT * FROM "{t}"')
                raw_splits[t] = [dict(r) for r in cursor.fetchall()]
            return raw_splits, True

        # Otherwise, default to first table
        first_table = tables[0]
        cursor.execute(f'SELECT * FROM "{first_table}"')
        rows = [dict(r) for r in cursor.fetchall()]
        return {"default": rows}, False

    finally:
        conn.close()


def load_from_duckdb(
    file_path: Path,
    table_name: Optional[str] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], bool]:
    """
    Load records from a DuckDB database (.duckdb, .ddb).
    Returns (raw_splits, is_split).
    Supports:
    - Explicit table selection via table_name or 'file.duckdb::table_name'.
    - Automatic mapping of split tables ('train', 'test', 'valid', 'validation', 'val', 'dev').
    - Single table database auto-loading.
    - Multi-table schema matching auto-splitting.
    """
    if not ensure_duckdb() or not HAS_DUCKDB:
        raise MissingDependencyError(
            format_name="DuckDB",
            package_name="duckdb",
            install_command="pip install duckdb",
            description="DuckDB database files require the 'duckdb' library.",
            extra_name="duckdb",
        )

    db_path = str(file_path.resolve())
    conn = duckdb.connect(db_path, read_only=True)
    try:
        res = conn.execute("SHOW TABLES").fetchall()
        tables = [row[0] for row in res]
        if not tables:
            raise ValueError(f"No user tables found in DuckDB database '{file_path}'.")

        def _fetch_records(tbl: str) -> List[Dict[str, Any]]:
            query = f'SELECT * FROM "{tbl}"'
            rel = conn.execute(query)
            try:
                return rel.arrow().to_pylist()
            except Exception:
                cols = [desc[0] for desc in rel.description]
                rows = rel.fetchall()
                return [dict(zip(cols, r)) for r in rows]

        if table_name:
            if table_name not in tables:
                raise ValueError(
                    f"Table '{table_name}' not found in DuckDB database '{file_path.name}'. Available tables: {', '.join(tables)}"
                )
            return {"default": _fetch_records(table_name)}, False

        # Check if table names match standard split names
        split_map: Dict[str, str] = {}
        for t in tables:
            t_lower = t.lower()
            if t_lower in ("train", "training"):
                split_map["train"] = t
            elif t_lower in ("valid", "validation", "val", "dev"):
                split_map["valid"] = t
            elif t_lower in ("test", "testing", "eval", "evaluation"):
                split_map["test"] = t

        if split_map:
            raw_splits: Dict[str, List[Dict[str, Any]]] = {}
            for s_name, t_name in split_map.items():
                raw_splits[s_name] = _fetch_records(t_name)
            return raw_splits, len(raw_splits) > 1

        if len(tables) == 1:
            return {"default": _fetch_records(tables[0])}, False

        primary_candidates = [
            t for t in tables
            if t.lower() in ("data", "dataset", "records", "items", "samples", "main")
        ]
        if primary_candidates:
            return {"default": _fetch_records(primary_candidates[0])}, False

        # Check if table schemas match
        table_cols = {}
        for t in tables:
            schema_rows = conn.execute(f'DESCRIBE "{t}"').fetchall()
            table_cols[t] = [r[0] for r in schema_rows]

        first_cols = table_cols[tables[0]]
        if all(table_cols[t] == first_cols for t in tables[1:]):
            raw_splits = {t: _fetch_records(t) for t in tables}
            return raw_splits, True

        return {"default": _fetch_records(tables[0])}, False
    finally:
        conn.close()


def load_from_webdataset(
    file_path: Path,
) -> Tuple[Dict[str, List[Dict[str, Any]]], bool]:
    """
    Load records from a WebDataset archive (.tar, .tar.gz, .tgz, .tar.bz2, .tar.xz).
    Returns (raw_splits, is_split).
    Supports:
    - Standard WebDataset format (samples grouped by key/stem, e.g., key.json, key.txt, key.prompt.txt).
    - Tar archives containing full dataset files (e.g. train.jsonl, data.csv, records.tsv).
    - Automatic split detection based on archive directory structure (e.g. train/, test/, valid/).
    """
    raw_splits: Dict[str, List[Dict[str, Any]]] = {}

    with tarfile.open(str(file_path), "r:*") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and not Path(m.name).name.startswith((".", "__"))]
        if not members:
            raise ValueError(f"No valid data files found in WebDataset archive '{file_path}'.")

        # Check if archive contains standalone dataset files (e.g. train.jsonl, test.csv)
        standalone_extensions = (".jsonl", ".csv", ".tsv", ".tab")
        standalone_members = [m for m in members if any(m.name.lower().endswith(ext) for ext in standalone_extensions)]

        if standalone_members and len(standalone_members) <= len(members) and all(m.size > 0 for m in standalone_members):
            for m in standalone_members:
                f = tar.extractfile(m)
                if f is None:
                    continue
                m_name = Path(m.name).stem.lower()
                split_name = "default"
                for s in ("train", "valid", "validation", "val", "test", "dev"):
                    if s in m_name:
                        split_name = s
                        break

                m_lower = m.name.lower()
                records: List[Dict[str, Any]] = []
                if m_lower.endswith(".jsonl"):
                    text_io = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                    for line in text_io:
                        line_str = line.strip()
                        if line_str:
                            try:
                                obj = json.loads(line_str)
                                if isinstance(obj, dict):
                                    records.append(obj)
                            except json.JSONDecodeError:
                                pass
                elif m_lower.endswith(".csv"):
                    text_io = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                    reader = csv.DictReader(text_io)
                    for row in reader:
                        records.append(dict(row))
                elif m_lower.endswith((".tsv", ".tab")):
                    text_io = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                    reader = csv.DictReader(text_io, delimiter="\t")
                    for row in reader:
                        records.append(dict(row))

                if records:
                    if split_name not in raw_splits:
                        raw_splits[split_name] = []
                    raw_splits[split_name].extend(records)

            if raw_splits:
                is_split = len(raw_splits) > 1 or (len(raw_splits) == 1 and "default" not in raw_splits)
                return raw_splits, is_split

        # Standard WebDataset sample grouping
        samples_by_split: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for m in members:
            m_path = Path(m.name)
            parts = m_path.parts

            # Detect split from parent directories
            split_name = "default"
            for p in parts[:-1]:
                p_lower = p.lower()
                if p_lower in ("train", "training"):
                    split_name = "train"
                    break
                elif p_lower in ("valid", "validation", "val", "dev"):
                    split_name = "valid"
                    break
                elif p_lower in ("test", "testing", "eval", "evaluation"):
                    split_name = "test"
                    break

            filename = m_path.name
            dots = filename.split(".")
            if len(dots) == 1:
                key = dots[0]
                field_name = "text"
                ext = ""
            elif len(dots) == 2:
                key = dots[0]
                ext = dots[1].lower()
                field_name = "text" if ext in ("txt", "text") else ext
            else:
                key = dots[0]
                field_candidate = dots[1].lower()
                ext = dots[-1].lower()
                field_name = field_candidate if field_candidate in (
                    "prompt", "completion", "response", "instruction",
                    "input", "output", "chosen", "rejected", "label",
                    "query", "answer", "system", "text", "context"
                ) else ("text" if ext in ("txt", "text") else ext)

            if split_name not in samples_by_split:
                samples_by_split[split_name] = {}

            if key not in samples_by_split[split_name]:
                samples_by_split[split_name][key] = {"__key__": key}

            record = samples_by_split[split_name][key]

            f = tar.extractfile(m)
            if f is None:
                continue

            content_bytes = f.read()
            if ext == "json":
                try:
                    data = json.loads(content_bytes.decode("utf-8", errors="replace"))
                    if isinstance(data, dict):
                        record.update(data)
                    else:
                        record["json"] = data
                except Exception:
                    record["json"] = content_bytes.decode("utf-8", errors="replace")
            else:
                text_val = content_bytes.decode("utf-8", errors="replace").strip()
                record[field_name] = text_val

        for s_name, samples_dict in samples_by_split.items():
            records_list = list(samples_dict.values())
            if records_list:
                raw_splits[s_name] = records_list

    if not raw_splits or all(len(v) == 0 for v in raw_splits.values()):
        raise ValueError(f"No valid records found in WebDataset archive '{file_path}'.")

    is_split = len(raw_splits) > 1 or (len(raw_splits) == 1 and "default" not in raw_splits)
    return raw_splits, is_split


def load_from_arrow_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load records from an Apache Arrow IPC stream/file."""
    if not ensure_pyarrow() or not HAS_PYARROW:
        raise MissingDependencyError(
            format_name="Apache Arrow",
            package_name="pyarrow",
            install_command="pip install pyarrow",
            description="Arrow IPC files require the 'pyarrow' library to decode.",
            extra_name="arrow",
        )
    try:
        with pa.memory_map(str(file_path), "r") as source:
            try:
                reader = pa_ipc.open_file(source)
                table = reader.read_all()
            except Exception:
                reader = pa_ipc.open_stream(source)
                table = reader.read_all()
        return table.to_pylist()
    except Exception as e:
        raise RuntimeError(f"Failed to read Arrow file '{file_path}': {e}")


def load_from_parquet_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load records from a Parquet file using pyarrow (with duckdb/fastparquet fallbacks)."""
    if ensure_pyarrow() and HAS_PYARROW:
        try:
            table = pq.read_table(str(file_path))
            return table.to_pylist()
        except Exception as e:
            raise RuntimeError(f"Failed to read Parquet file '{file_path}': {e}")

    # Fallback to DuckDB if available
    if ensure_duckdb() and HAS_DUCKDB:
        try:
            rel = duckdb.query(f"SELECT * FROM read_parquet('{file_path}')")
            return [dict(r) for r in rel.df().to_dict(orient="records")]
        except Exception:
            pass

    # Fallback to fastparquet if available
    try:
        import fastparquet
        pf = fastparquet.ParquetFile(str(file_path))
        return [dict(r) for r in pf.to_pandas().to_dict(orient="records")]
    except Exception:
        pass

    raise MissingDependencyError(
        format_name="Parquet",
        package_name="pyarrow",
        install_command="pip install pyarrow",
        description="Parquet files use columnar compression (snappy, zstd, dictionary encoding) requiring the high-performance 'pyarrow' library.",
        extra_name="parquet",
    )


def load_from_parquet_dir(dir_path: Path) -> List[Dict[str, Any]]:
    """Load records from a directory of Parquet files (sharded dataset) using pyarrow.dataset."""
    if ensure_pyarrow() and HAS_PYARROW:
        try:
            target_dir = dir_path
            # If no parquet files in root but exists in data/ subdirectory
            if not any(f.name.lower().endswith(PARQUET_EXTS) for f in dir_path.iterdir() if f.is_file()):
                data_sub = dir_path / "data"
                if data_sub.is_dir():
                    target_dir = data_sub
            dataset = ds.dataset(str(target_dir), format="parquet")
            table = dataset.to_table()
            return table.to_pylist()
        except Exception as e:
            raise RuntimeError(f"Failed to read Parquet directory '{dir_path}': {e}")

    # Fallback to DuckDB if available
    if ensure_duckdb() and HAS_DUCKDB:
        try:
            target_dir = dir_path
            if not any(f.name.lower().endswith(PARQUET_EXTS) for f in dir_path.iterdir() if f.is_file()):
                data_sub = dir_path / "data"
                if data_sub.is_dir():
                    target_dir = data_sub
            glob_path = str(target_dir / "*.parquet")
            rel = duckdb.query(f"SELECT * FROM read_parquet('{glob_path}')")
            return [dict(r) for r in rel.df().to_dict(orient="records")]
        except Exception:
            pass

    raise MissingDependencyError(
        format_name="Parquet",
        package_name="pyarrow",
        install_command="pip install pyarrow",
        description="Parquet files use columnar compression (snappy, zstd, dictionary encoding) requiring the high-performance 'pyarrow' library.",
        extra_name="parquet",
    )


def is_lance_dir(path: Path) -> bool:
    """Check if directory has the signature of a Lance dataset."""
    if not path.is_dir():
        return False
    if path.suffix.lower() == ".lance" or path.name.lower().endswith(".lance"):
        return True
    if (path / "_versions").is_dir() or (path / "_indices").is_dir():
        return True
    return False


def load_from_lance(dir_path: Path) -> List[Dict[str, Any]]:
    """Load records from a Lance dataset directory (.lance)."""
    if not ensure_lance() or not HAS_LANCE:
        raise MissingDependencyError(
            format_name="Lance",
            package_name="pylance",
            install_command="pip install pylance",
            description="Lance columnar dataset folders require the 'pylance' library.",
            extra_name="lance",
        )
    try:
        ds_lance = lance.dataset(str(dir_path))
        table = ds_lance.to_table()
        return table.to_pylist()
    except Exception as e:
        raise RuntimeError(f"Failed to read Lance dataset '{dir_path}': {e}")


def is_hf_save_to_disk_dir(path: Path) -> bool:
    """Check if directory has the signature of datasets.save_to_disk()."""
    if not path.is_dir():
        return False
    # Check for dataset_dict.json or dataset_info.json or state.json
    has_dict = (path / "dataset_dict.json").exists()
    has_info = (path / "dataset_info.json").exists()
    has_state = (path / "state.json").exists()
    has_arrow = bool(list(path.glob("*.arrow"))) or bool(list(path.glob("*/*.arrow")))
    return (has_dict or (has_info and has_arrow) or (has_state and has_arrow))


def load_single_local_dataset(path: Union[Path, str]) -> LoadedDataset:
    """Load a single dataset path (file or folder)."""
    path_str = str(path)
    file_path, table_name = _split_file_and_table(path_str)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {file_path}")

    # Case 1: HF datasets library is available and directory is a saved dataset
    if HAS_DATASETS and file_path.is_dir():
        if is_hf_save_to_disk_dir(file_path):
            try:
                loaded = load_from_disk(str(file_path))
                if isinstance(loaded, DatasetDict):
                    split_names = list(loaded.keys())
                    split_counts = {k: len(loaded[k]) for k in split_names}
                    first_split = loaded[split_names[0]]
                    cols = first_split.column_names
                    feats = getattr(first_split, "features", {})
                    total = sum(split_counts.values())
                    samples = [dict(first_split[i]) for i in range(min(5, len(first_split)))]
                    return LoadedDataset(
                        source_path=str(file_path),
                        is_split=True,
                        split_names=split_names,
                        split_counts=split_counts,
                        columns=cols,
                        total_rows=total,
                        sample_records=samples,
                        base_dir=file_path.resolve(),
                        features=feats if isinstance(feats, dict) else {},
                        _raw_splits=dict(loaded),
                    )
                elif isinstance(loaded, Dataset):
                    cols = loaded.column_names
                    feats = getattr(loaded, "features", {})
                    total = len(loaded)
                    samples = [dict(loaded[i]) for i in range(min(5, len(loaded)))]
                    return LoadedDataset(
                        source_path=str(file_path),
                        is_split=False,
                        split_names=["default"],
                        split_counts={"default": total},
                        columns=cols,
                        total_rows=total,
                        sample_records=samples,
                        base_dir=file_path.resolve(),
                        features=feats if isinstance(feats, dict) else {},
                        _raw_splits={"default": loaded},
                    )
            except Exception:
                pass

    # Case 2: Direct file or directory of data files
    raw_splits: Dict[str, List[Dict[str, Any]]] = {}
    is_split = False

    if file_path.is_file() or table_name is not None:
        if file_path.name == "config.json":
            from mlx_commander.lora.model_info import is_model_directory
            if is_model_directory(file_path.parent):
                raise ValueError(
                    f"'{file_path.name}' is a model architecture configuration file inside '{file_path.parent.name}', not a training dataset. "
                    f"To fine-tune this model, switch to Mode 2: Fine-Tuning Single Run (press F2 or Tab 2) and select '{file_path.parent.name}' under 'Base Model'."
                )
        if file_path.suffix.lower() in (".safetensors", ".bin", ".mlx", ".pt"):
            raise ValueError(
                f"'{file_path.name}' is a machine learning model weights file, not a training dataset. "
                f"To fine-tune this model, switch to Mode 2: Fine-Tuning Single Run (press F2 or Tab 2) and select the model folder under 'Base Model'."
            )

        ext = file_path.suffix.lower()
        full_ext = "".join(file_path.suffixes).lower()

        if is_parquet_file(file_path):
            raw_splits["default"] = load_from_parquet_file(file_path)
        elif is_arrow_file(file_path):
            raw_splits["default"] = load_from_arrow_file(file_path)
        elif ext in (".json", ".jsonl"):
            raw_splits["default"] = load_from_json_or_jsonl(file_path)
        elif ext == ".csv":
            raw_splits["default"] = load_from_csv(file_path)
        elif ext in (".tsv", ".tab"):
            raw_splits["default"] = load_from_tsv(file_path)
        elif ext in (".txt", ".text"):
            raw_splits["default"] = load_from_text(file_path)
        elif ext in (".duckdb", ".ddb"):
            raw_splits, is_split = load_from_duckdb(file_path, table_name=table_name)
        elif is_sqlite_file(file_path) or table_name is not None:
            raw_splits, is_split = load_from_sqlite(file_path, table_name=table_name)
        elif any(full_ext.endswith(tar_ext) for tar_ext in (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
            raw_splits, is_split = load_from_webdataset(file_path)
        else:
            raw_splits["default"] = load_from_json_or_jsonl(file_path)

    elif file_path.is_dir():
        from mlx_commander.lora.model_info import is_model_directory, inspect_local_model
        if is_model_directory(file_path):
            meta = inspect_local_model(str(file_path))
            size_str = f" ({meta.architecture}, {meta.file_size_gb:.1f} GB)" if meta.file_size_gb > 0 else ""
            raise ValueError(
                f"'{file_path.name}' is a machine learning base model directory{size_str}, not a training dataset. "
                f"To fine-tune this model, switch to Mode 2: Fine-Tuning Single Run (press F2 or Tab 2) "
                f"and select it under 'Base Model'. Datasets should contain training samples (e.g. .parquet, .jsonl, .csv, or train.jsonl)."
            )

        if is_lance_dir(file_path):
            raw_splits["default"] = load_from_lance(file_path)
        else:
            # Check if subdirectories correspond to splits (e.g. train, test, validation)
            split_dirs = [d for d in file_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
            # If HF save_to_disk with dataset_dict.json
            dict_json_file = file_path / "dataset_dict.json"
            if dict_json_file.exists():
                try:
                    with open(dict_json_file, "r") as f:
                        dict_info = json.load(f)
                        expected_splits = dict_info.get("splits", [])
                        for s in expected_splits:
                            s_dir = file_path / s
                            arrow_files = list(s_dir.glob("*.arrow"))
                            s_records: List[Dict[str, Any]] = []
                            for af in arrow_files:
                                s_records.extend(load_from_arrow_file(af))
                            raw_splits[s] = s_records
                except Exception:
                    pass

            if not raw_splits:
                # Check for standard split files in the root dir: train.jsonl, test.jsonl, etc.
                found_split_files = False
                split_exts = [
                    ".jsonl", ".parquet", ".pq", ".parq", ".arrow", ".json", ".csv",
                    ".tsv", ".tab", ".txt", ".text", ".duckdb", ".ddb",
                    ".sqlite", ".db", ".sqlite3",
                    ".tar", ".tar.gz", ".tgz",
                ]
                for s in ["train", "valid", "validation", "val", "test", "dev"]:
                    for ext in split_exts:
                        candidate = file_path / f"{s}{ext}"
                        if candidate.exists():
                            found_split_files = True
                            if ext in (".parquet", ".pq", ".parq"):
                                raw_splits[s] = load_from_parquet_file(candidate)
                            elif ext == ".arrow":
                                raw_splits[s] = load_from_arrow_file(candidate)
                            elif ext in (".json", ".jsonl"):
                                raw_splits[s] = load_from_json_or_jsonl(candidate)
                            elif ext == ".csv":
                                raw_splits[s] = load_from_csv(candidate)
                            elif ext in (".tsv", ".tab"):
                                raw_splits[s] = load_from_tsv(candidate)
                            elif ext in (".txt", ".text"):
                                raw_splits[s] = load_from_text(candidate)
                            elif ext in (".duckdb", ".ddb"):
                                db_splits, _ = load_from_duckdb(candidate)
                                raw_splits[s] = db_splits.get("default", list(db_splits.values())[0] if db_splits else [])
                            elif ext in (".sqlite", ".db", ".sqlite3"):
                                db_splits, _ = load_from_sqlite(candidate)
                                raw_splits[s] = db_splits.get("default", list(db_splits.values())[0] if db_splits else [])
                            elif ext in (".tar", ".tar.gz", ".tgz") or ext.endswith(".tar"):
                                tar_splits, _ = load_from_webdataset(candidate)
                                raw_splits[s] = tar_splits.get("default", list(tar_splits.values())[0] if tar_splits else [])
                            break

                if not found_split_files:
                    # Check for sharded Parquet dataset directory (in root or subfolder)
                    if is_parquet_dir(file_path):
                        raw_splits["default"] = load_from_parquet_dir(file_path)
                    else:
                        # Aggregate all data files in the directory
                        all_records: List[Dict[str, Any]] = []
                        supported_exts = (
                            ".jsonl", ".json", ".parquet", ".pq", ".parq", ".arrow", ".csv",
                            ".tsv", ".tab", ".txt", ".text", ".duckdb", ".ddb",
                            ".sqlite", ".db", ".sqlite3",
                            ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz",
                        )
                        data_files = sorted([
                            f for f in file_path.glob("*")
                            if any(f.name.lower().endswith(ext) for ext in supported_exts)
                        ])
                        for df in data_files:
                            fname = df.name.lower()
                            if is_parquet_file(df):
                                all_records.extend(load_from_parquet_file(df))
                            elif is_arrow_file(df):
                                all_records.extend(load_from_arrow_file(df))
                            elif fname.endswith((".json", ".jsonl")):
                                all_records.extend(load_from_json_or_jsonl(df))
                            elif fname.endswith(".csv"):
                                all_records.extend(load_from_csv(df))
                            elif fname.endswith((".tsv", ".tab")):
                                all_records.extend(load_from_tsv(df))
                            elif fname.endswith((".txt", ".text")):
                                all_records.extend(load_from_text(df))
                            elif fname.endswith((".duckdb", ".ddb")):
                                db_splits, _ = load_from_duckdb(df)
                                for rows in db_splits.values():
                                    all_records.extend(rows)
                            elif fname.endswith((".sqlite", ".db", ".sqlite3")):
                                db_splits, _ = load_from_sqlite(df)
                                for rows in db_splits.values():
                                    all_records.extend(rows)
                            elif any(fname.endswith(ext) for ext in (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
                                tar_splits, _ = load_from_webdataset(df)
                                for rows in tar_splits.values():
                                    all_records.extend(rows)
                        raw_splits["default"] = all_records

    if not raw_splits or all(len(v) == 0 for v in raw_splits.values()):
        raise ValueError(
            f"No dataset records found in '{path}'. Supported formats include Hugging Face save_to_disk folders, "
            f"Parquet (.parquet), Arrow (.arrow), JSON Lines (.jsonl), JSON (.json), CSV (.csv), "
            f"TSV (.tsv, .tab), Plain Text (.txt), SQLite (.sqlite, .db), DuckDB (.duckdb), "
            f"Lance (.lance), or WebDataset (.tar, .tar.gz)."
        )

    split_names = [k for k in raw_splits.keys() if len(raw_splits[k]) > 0]
    split_counts = {k: len(raw_splits[k]) for k in split_names}
    total_rows = sum(split_counts.values())

    # Extract columns from the first available record
    columns: List[str] = []
    sample_records: List[Dict[str, Any]] = []
    for s_name in split_names:
        for r in raw_splits[s_name]:
            if not columns:
                columns = list(r.keys())
            if len(sample_records) < 5:
                sample_records.append(r)
            if len(sample_records) >= 5:
                break
        if len(sample_records) >= 5:
            break

    if not is_split:
        is_split = len(split_names) > 1 or (len(split_names) == 1 and split_names[0] != "default")

    source_desc = f"{file_path}::{table_name}" if table_name else str(file_path)
    base_dir = file_path.resolve() if file_path.is_dir() else file_path.resolve().parent

    return LoadedDataset(
        source_path=source_desc,
        is_split=is_split,
        split_names=split_names,
        split_counts=split_counts,
        columns=columns,
        total_rows=total_rows,
        sample_records=sample_records,
        base_dir=base_dir,
        _raw_splits=raw_splits,
    )


def load_local_dataset(path_input: Union[str, Path, List[Union[str, Path]]]) -> LoadedDataset:
    """
    Load a Hugging Face dataset or data files from local drive.
    Supports:
      - Single file or folder (str or Path)
      - SQLite database with table specifier (e.g. 'data.sqlite::my_table')
      - List of files (List[str] or List[Path])
      - Newline-separated or comma-separated file paths in a single string

    When multiple files are provided:
      1. Verifies that all files exist.
      2. Validates schema consistency across all files (matching columns).
         Raises ValueError with detailed differences if schemas do not match.
      3. Merges records into a single consolidated dataset ready for
         randomized train/validation/test re-splitting from scratch with a custom seed.
    """
    raw_paths = _normalize_path_list(path_input)
    if not raw_paths:
        raise ValueError("No dataset path provided.")

    if len(raw_paths) == 1:
        rp = raw_paths[0]
        f_path, _ = _split_file_and_table(rp)
        if not f_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {f_path}")
        return load_single_local_dataset(rp)

    # Multi-file loading & merging with schema validation
    loaded_datasets: List[LoadedDataset] = []
    for rp in raw_paths:
        f_path, _ = _split_file_and_table(rp)
        if not f_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {f_path}")
        loaded = load_single_local_dataset(rp)
        loaded_datasets.append(loaded)

    # 1. Schema Consistency Validation
    base_ds = loaded_datasets[0]
    base_cols = base_ds.columns
    base_cols_set = set(base_cols)
    base_name = Path(base_ds.source_path).name or base_ds.source_path

    for other_ds in loaded_datasets[1:]:
        other_cols_set = set(other_ds.columns)
        other_name = Path(other_ds.source_path).name or other_ds.source_path
        if base_cols_set != other_cols_set:
            missing_in_other = base_cols_set - other_cols_set
            extra_in_other = other_cols_set - base_cols_set
            diff_parts = []
            if missing_in_other:
                diff_parts.append(f"Missing in '{other_name}': {sorted(missing_in_other)}")
            if extra_in_other:
                diff_parts.append(f"Extra in '{other_name}': {sorted(extra_in_other)}")
            diff_str = "; ".join(diff_parts)
            raise ValueError(
                f"Schema mismatch detected across dataset files!\n"
                f"  • '{base_name}' columns ({len(base_cols)}): {base_cols}\n"
                f"  • '{other_name}' columns ({len(other_ds.columns)}): {other_ds.columns}\n"
                f"Difference: {diff_str}\n"
                f"Cannot merge datasets with incompatible columns."
            )

    # 2. Merge records across all files
    merged_records: List[Dict[str, Any]] = []
    for ds in loaded_datasets:
        merged_records.extend(ds.get_all_records())

    total_rows = len(merged_records)
    filenames = [Path(d.source_path).name for d in loaded_datasets]
    summary_path = f"Merged ({len(loaded_datasets)} files: {', '.join(filenames)})"

    return LoadedDataset(
        source_path=summary_path,
        is_split=False,
        split_names=["default"],
        split_counts={"default": total_rows},
        columns=base_cols,
        total_rows=total_rows,
        sample_records=merged_records[:5],
        base_dir=base_ds.base_dir,
        features=getattr(base_ds, "features", {}),
        _raw_splits={"default": merged_records},
    )

