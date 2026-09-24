"""
Central Application State for MLX Commander persistent TUI.
Maintains dataset info, format configuration, column mappings,
active panel focus, and reactive preview cache.
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mlx_commander.exceptions import MissingDependencyError
from mlx_commander.converter import ConversionResult, convert_and_save
from mlx_commander.formats import (
    ColumnMapping,
    MLXFormat,
    auto_detect_format_and_mapping,
    auto_detect_mapping,
    format_record,
    parse_label_map,
    validate_mapping,
)
from mlx_commander.loader import LoadedDataset, detect_column_discrete_labels, load_local_dataset
from mlx_commander.splitter import (
    SplitConfig,
    calculate_split_counts,
    generate_random_seed,
)
from mlx_commander.lora import (
    LoraRunConfig,
    MultiLoraRunConfig,
    ModelMetadata,
    QueueManager,
    calculate_implied_epochs,
    estimate_duration,
    estimate_peak_memory,
    generate_deterministic_run_name,
    inspect_local_model,
    is_wandb_available,
    is_wandb_logged_in,
)


class ActivePanel(Enum):
    LEFT = "left"    # Dataset source, schema, and columns
    RIGHT = "right"  # MLX format, column mappings, splits, output
    PREVIEW = "preview"


class ThemeMode(str, Enum):
    MODERN = "modern"
    NORTON = "norton"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def is_norton(cls, mode: Any) -> bool:
        if isinstance(mode, cls):
            return mode == cls.NORTON
        val = getattr(mode, "value", str(mode))
        return "norton" in str(val).lower() or str(val).lower().strip() in ("nc", "blue", "classic")


@dataclass
class CommanderState:
    dataset_path: str = ""
    loaded_dataset: Optional[LoadedDataset] = None
    last_missing_dependency: Optional[MissingDependencyError] = None
    target_format: MLXFormat = MLXFormat.PROMPT_COMPLETION
    mapping: ColumnMapping = field(default_factory=ColumnMapping)
    train_pct: float = 80.0
    valid_pct: float = 10.0
    test_pct: float = 10.0
    seed: int = field(default_factory=generate_random_seed)
    output_dir: str = ""
    has_custom_output_dir: bool = False

    # Theme
    theme_mode: str = ThemeMode.MODERN

    # UI navigation state
    active_panel: ActivePanel = ActivePanel.LEFT
    left_focus_idx: int = 0   # 0: Dataset path field, 1: Columns list
    right_focus_idx: int = 0  # 0: Format, 1-3: Mappings, 4-6: Splits, 7: Seed, 8: Output, 9: Convert button
    column_scroll_offset: int = 0
    selected_column_idx: int = 0

    # Status / notification
    status_message: str = "Ready. Configure hyperparameters, [F6] Add to Queue, or [F5] Run."
    status_is_error: bool = False

    # Reactive preview cache
    preview_cache: List[str] = field(default_factory=list)
    preview_error: Optional[str] = None

    # Screen / Mode Navigation
    active_tab: int = 1  # 0: Dataset Conversion, 1: Single Run, 2: Multi-Run Matrix
    mode_switcher_focused: bool = False
    mode_switcher_idx: int = 1

    # LoRA Fine-Tuning Single Run State
    lora_config: LoraRunConfig = field(default_factory=LoraRunConfig)
    queue_manager: Optional[QueueManager] = None
    selected_queue_idx: int = 0
    lora_active_panel: str = "left"  # "left", "right", "queue"
    lora_left_focus_idx: int = 0   # 0: Model, 1: Dataset, 2: Method, 3: Optim, 4: Mode, 5: Run Name
    lora_right_focus_idx: int = 0  # 0 to 13 (sequential single column)
    lora_right_scroll_offset: int = 0
    lora_queue_scroll_offset: int = 0
    current_model_metadata: Optional[ModelMetadata] = None

    # LoRA Fine-Tuning Multi-Run Sweep State
    multi_lora_config: MultiLoraRunConfig = field(default_factory=MultiLoraRunConfig)
    multi_active_panel: str = "left"  # "left", "right", "sweep"
    multi_left_focus_idx: int = 0   # 0: Model, 1: Dataset, 2: Method, 3: Optim, 4: Mode
    multi_right_focus_idx: int = 0  # 0 to 12
    multi_right_scroll_offset: int = 0

    # Weights & Biases Experiment Tracking State
    wandb_enabled: bool = True
    wandb_project: str = "mlx-commander"

    # Performance & Caching
    _cached_wandb_status: Optional[Dict[str, Any]] = None
    _inspected_model_target: Optional[str] = None
    _cached_train_counts: Dict[str, Tuple[float, int]] = field(default_factory=dict)
    _cached_mem_estimate: Optional[Dict[str, Any]] = None
    _cached_mem_key: Optional[Tuple] = None
    _cached_dur_estimate: Optional[Dict[str, Any]] = None
    _cached_dur_key: Optional[Tuple] = None

    def __post_init__(self) -> None:
        if self.output_dir:
            self.has_custom_output_dir = True
        else:
            if self.loaded_dataset:
                self.output_dir = str(self.loaded_dataset.default_output_dir)
            elif self.dataset_path:
                try:
                    p = Path(self.dataset_path.split("::")[0]).resolve()
                    base = p if p.is_dir() else p.parent
                    self.output_dir = str(base / "mlx_dataset")
                except Exception:
                    self.output_dir = str(Path.cwd() / "mlx_dataset")
            else:
                self.output_dir = str(Path.cwd() / "mlx_dataset")

        if self.queue_manager is None:
            self.queue_manager = QueueManager(Path.cwd() / "mlx_runs")
        self.sync_dataset_to_lora()
        self.inspect_current_model()
        self.update_deterministic_lora_name()

    def inspect_current_model(self, force: bool = False) -> ModelMetadata:
        """Inspect and cache base model architecture metadata."""
        if not force and self.current_model_metadata is not None:
            if (
                self._inspected_model_target == self.lora_config.model
                or self.current_model_metadata.path == self.lora_config.model
                or self.current_model_metadata.name == self.lora_config.model
            ):
                self._inspected_model_target = self.lora_config.model
                if hasattr(self.current_model_metadata, "engine"):
                    self.lora_config.engine = self.current_model_metadata.engine
                    self.multi_lora_config.engine = self.current_model_metadata.engine
                return self.current_model_metadata
        meta = inspect_local_model(self.lora_config.model)
        self.current_model_metadata = meta
        self._inspected_model_target = self.lora_config.model
        if hasattr(meta, "engine"):
            self.lora_config.engine = meta.engine
            self.multi_lora_config.engine = meta.engine
        return meta

    def get_wandb_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Return W&B availability and authentication status (cached)."""
        if self._cached_wandb_status is not None and not force_refresh:
            return self._cached_wandb_status
        avail = is_wandb_available()
        logged_in, entity = is_wandb_logged_in() if avail else (False, None)
        self._cached_wandb_status = {
            "available": avail,
            "logged_in": logged_in,
            "entity": entity,
            "enabled": self.wandb_enabled and avail and logged_in,
            "project": self.wandb_project,
        }
        return self._cached_wandb_status

    def update_deterministic_lora_name(self) -> None:
        """Update current form lora_config run name and adapter path deterministically if not custom."""
        if not self.lora_config.is_custom_name:
            next_idx = (len(self.queue_manager.runs) + 1) if self.queue_manager else 1
            new_name = generate_deterministic_run_name(
                index=next_idx,
                fine_tune_type=self.lora_config.fine_tune_type,
                rank=self.lora_config.lora_rank,
                alpha=self.lora_config.lora_alpha,
                learning_rate=self.lora_config.learning_rate,
                batch_size=self.lora_config.batch_size,
                iters=self.lora_config.iters,
                model_name=self.lora_config.model,
            )
            self.lora_config.name = new_name
            self.lora_config.adapter_path = f"adapters/{new_name}"

    def get_train_record_count(self) -> int:
        """Return the number of train records, cached by file mtime to eliminate disk I/O during render."""
        if self.loaded_dataset:
            split_counts = self.get_split_counts()
            return split_counts.get("train", 0)
        if self.lora_config.data:
            train_f = Path(self.lora_config.data) / "train.jsonl"
            if train_f.exists():
                try:
                    mtime = train_f.stat().st_mtime
                    cache_key = str(train_f.resolve())
                    if cache_key in self._cached_train_counts:
                        cached_mtime, cached_cnt = self._cached_train_counts[cache_key]
                        if cached_mtime == mtime:
                            return cached_cnt
                    with open(train_f, "rb") as f:
                        cnt = sum(1 for _ in f)
                    self._cached_train_counts[cache_key] = (mtime, cnt)
                    return cnt
                except Exception:
                    pass
        return 0

    def get_implied_epochs(self) -> Optional[float]:
        """Calculate implied epochs: (iters * batch_size * grad_accumulation_steps) / train_records."""
        train_count = self.get_train_record_count()
        return calculate_implied_epochs(
            self.lora_config.iters,
            self.lora_config.batch_size,
            train_count,
            getattr(self.lora_config, "grad_accumulation_steps", 1),
        )

    def get_memory_estimate(self) -> Dict[str, Any]:
        """Return peak memory estimate (cached by configuration parameters)."""
        cfg = self.lora_config
        key = (cfg.model, cfg.batch_size, cfg.max_seq_length, cfg.lora_rank, cfg.num_layers, cfg.grad_checkpoint)
        if self._cached_mem_key == key and self._cached_mem_estimate is not None:
            return self._cached_mem_estimate
        est = estimate_peak_memory(cfg)
        self._cached_mem_key = key
        self._cached_mem_estimate = est
        return est

    def get_duration_estimate(self) -> Dict[str, Any]:
        """Return training duration estimate (cached by configuration parameters)."""
        cfg = self.lora_config
        key = (cfg.model, cfg.iters, cfg.batch_size, cfg.max_seq_length)
        if self._cached_dur_key == key and self._cached_dur_estimate is not None:
            return self._cached_dur_estimate
        est = estimate_duration(cfg)
        self._cached_dur_key = key
        self._cached_dur_estimate = est
        return est

    def clear_estimates_cache(self) -> None:
        """Clear memory, duration, and train count estimate caches."""
        self._cached_mem_key = None
        self._cached_mem_estimate = None
        self._cached_dur_key = None
        self._cached_dur_estimate = None
        self._cached_train_counts.clear()

    def get_multi_sweep_estimates(self) -> Dict[str, Any]:
        """Calculate aggregated runtime estimates for the multi-run sweep."""
        train_count = self.get_train_record_count()
        return self.multi_lora_config.calculate_sweep_estimates(
            dataset_records=train_count,
        )

    def sync_dataset_to_lora(self) -> None:
        """Pre-populate the LoRA dataset directory from the active conversion target or source."""
        target = None
        if self.output_dir and (Path(self.output_dir) / "train.jsonl").exists():
            target = str(self.output_dir)
        elif self.output_dir:
            target = str(self.output_dir)
        elif self.dataset_path:
            target = str(self.dataset_path)

        if target:
            self.lora_config.data = target
            self.multi_lora_config.data = target

    def switch_mode(self, target_tab: int) -> None:
        """Switch active mode tab and synchronize dependent state."""
        if target_tab != self.active_tab:
            self.active_tab = target_tab
            if self.active_tab == 1:
                self.sync_dataset_to_lora()
                self.status_message = "Switched to Single Run Mode."
            elif self.active_tab == 2:
                self.sync_dataset_to_lora()
                self.multi_lora_config.model = self.lora_config.model
                self.multi_lora_config.fine_tune_type = self.lora_config.fine_tune_type
                self.multi_lora_config.optimizer = self.lora_config.optimizer
                self.status_message = "Switched to Multi-Run Matrix Mode."
            else:
                self.status_message = "Switched to Dataset Conversion Mode."
            self.status_is_error = False

    def add_multi_lora_runs_to_queue(self) -> List[LoraRunConfig]:
        """Generate all combinations from multi_lora_config and add to queue."""
        if not self.queue_manager:
            self.queue_manager = QueueManager(Path.cwd() / "mlx_runs")
        runs = self.multi_lora_config.generate_runs()
        added_runs = []
        for r in runs:
            r.wandb_project = self.wandb_project
            added = self.queue_manager.add_run(r)
            added_runs.append(added)
        if self.queue_manager.runs:
            self.selected_queue_idx = len(self.queue_manager.runs) - 1
        return added_runs

    def add_current_lora_to_queue(self) -> LoraRunConfig:
        """Add current form config as a new run in the queue."""
        if not self.queue_manager:
            self.queue_manager = QueueManager(Path.cwd() / "mlx_runs")
        import time, uuid
        run = LoraRunConfig.from_dict(self.lora_config.to_dict())
        run.id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        run.status = "queued"
        run.started_at = None
        run.finished_at = None
        run.exit_code = None
        run.error_message = None
        run.wandb_project = self.wandb_project
        self.queue_manager.add_run(run)
        self.selected_queue_idx = len(self.queue_manager.runs) - 1
        self.update_deterministic_lora_name()
        return run

    def load_queue_run_into_form(self, run_id: str) -> bool:
        """Load a selected run from the queue back into the form fields."""
        if not self.queue_manager:
            return False
        r = self.queue_manager.get_run(run_id)
        if r:
            self.lora_config = LoraRunConfig.from_dict(r.to_dict())
            self.inspect_current_model()
            return True
        return False

    def load_dataset(self, path_input: Any, custom_mapping: Optional[ColumnMapping] = None) -> bool:
        """Load dataset from disk (single file/folder or multiple merged files) and update state."""
        if not path_input:
            self.status_message = "Path cannot be empty."
            self.status_is_error = True
            return False

        self.last_missing_dependency = None
        try:
            self.status_message = "Loading dataset..."
            self.status_is_error = False
            ds = load_local_dataset(path_input)
            self.loaded_dataset = ds
            self.dataset_path = ds.source_path
            self.selected_column_idx = 0
            self.column_scroll_offset = 0

            # Intelligently detect the most suitable MLX format based on dataset columns
            detected_format, detected_map = auto_detect_format_and_mapping(ds.columns)
            if self.target_format == MLXFormat.PROMPT_COMPLETION:
                self.target_format = detected_format
                auto = detected_map
            else:
                auto = auto_detect_mapping(self.target_format, ds.columns)

            if custom_mapping is not None:
                self.mapping = custom_mapping
                # Backfill unset fields from auto detection
                for attr in (
                    "prompt_col", "completion_col", "text_col", "text_template",
                    "messages_col", "user_col", "assistant_col", "system_col",
                    "chosen_col", "rejected_col"
                ):
                    if not getattr(self.mapping, attr, None) and getattr(auto, attr, None):
                        setattr(self.mapping, attr, getattr(auto, attr))
            else:
                self.mapping = auto

            # Auto-detect discrete labels and pre-populate label_map if suggested by HF ClassLabel
            target_col = self.mapping.completion_col or self.mapping.assistant_col
            if target_col and not self.mapping.label_map:
                disc = detect_column_discrete_labels(ds, target_col)
                if disc and disc.get("suggested_map"):
                    self.mapping.label_map = disc["suggested_map"]

            if not self.has_custom_output_dir:
                self.output_dir = str(ds.default_output_dir)
            if "Merged (" in ds.source_path:
                self.status_message = f"[OK] {ds.source_path}: {ds.total_rows:,} records merged. Schemas verified."
            else:
                self.status_message = f"Loaded {ds.total_rows:,} records with {len(ds.columns)} columns."
            self.status_is_error = False
            self.update_preview()
            return True
        except MissingDependencyError as e:
            self.last_missing_dependency = e
            self.status_message = f"Missing dependency: {e.package_name} ({e.install_command})"
            self.status_is_error = True
            return False
        except Exception as e:
            self.status_message = f"Failed to load: {e}"
            self.status_is_error = True
            return False

    def apply_prefill(self, config: Dict[str, Any]) -> None:
        """
        Apply pre-populated configuration (from CLI flags, JSON file, or agent skill).
        Enables agents to launch the TUI with schema, format, mapping, and splits pre-configured.
        """
        if not config:
            return

        # Target format
        fmt_val = config.get("format") or config.get("target_format")
        if fmt_val:
            if isinstance(fmt_val, MLXFormat):
                self.target_format = fmt_val
            elif isinstance(fmt_val, str):
                try:
                    self.target_format = MLXFormat(fmt_val.lower().strip())
                except ValueError:
                    pass

        # Splits & Seed
        if "train" in config and config["train"] is not None:
            self.train_pct = float(config["train"])
        elif "train_pct" in config and config["train_pct"] is not None:
            self.train_pct = float(config["train_pct"])

        if "valid" in config and config["valid"] is not None:
            self.valid_pct = float(config["valid"])
        elif "valid_pct" in config and config["valid_pct"] is not None:
            self.valid_pct = float(config["valid_pct"])

        if "test" in config and config["test"] is not None:
            self.test_pct = float(config["test"])
        elif "test_pct" in config and config["test_pct"] is not None:
            self.test_pct = float(config["test_pct"])

        if "seed" in config and config["seed"] is not None:
            self.seed = int(config["seed"])

        # Output directory
        out_val = config.get("output") or config.get("output_dir")
        if out_val:
            self.output_dir = str(out_val).strip()
            self.has_custom_output_dir = True

        # Custom Mapping
        custom_map = config.get("mapping")
        mapping_obj: Optional[ColumnMapping] = None
        if isinstance(custom_map, ColumnMapping):
            mapping_obj = custom_map
        elif isinstance(custom_map, dict):
            mapping_obj = ColumnMapping(**custom_map)
        else:
            mapping_obj = ColumnMapping()

        map_keys = {
            "prompt_col": ["prompt_col", "prompt"],
            "completion_col": ["completion_col", "completion"],
            "text_col": ["text_col", "text"],
            "text_template": ["text_template", "template"],
            "messages_col": ["messages_col", "messages"],
            "user_col": ["user_col", "user"],
            "assistant_col": ["assistant_col", "assistant"],
            "system_col": ["system_col", "system"],
            "chosen_col": ["chosen_col", "chosen"],
            "rejected_col": ["rejected_col", "rejected"],
            "default_system_prompt": ["default_system_prompt", "system_prompt"],
        }
        has_custom_field = False
        for attr, aliases in map_keys.items():
            for alias in aliases:
                if alias in config and config[alias]:
                    setattr(mapping_obj, attr, str(config[alias]).strip())
                    has_custom_field = True
                    break

        if "label_map" in config and config["label_map"]:
            mapping_obj.label_map = parse_label_map(config["label_map"])
            has_custom_field = True
        elif "labels" in config and config["labels"]:
            mapping_obj.label_map = parse_label_map(config["labels"])
            has_custom_field = True

        # Theme prefill
        theme_val = config.get("theme") or config.get("theme_mode")
        if theme_val:
            if str(theme_val).lower().strip() in ("norton", "nc", "blue", "classic"):
                self.theme_mode = ThemeMode.NORTON
            else:
                self.theme_mode = ThemeMode.MODERN

        # Dataset loading
        ds_path = config.get("dataset") or config.get("dataset_path") or self.dataset_path
        if ds_path:
            self.load_dataset(ds_path, custom_mapping=mapping_obj if (custom_map or has_custom_field) else None)
            # Switch active panel to RIGHT so user immediately sees mappings, splits, and preview
            self.active_panel = ActivePanel.RIGHT
            self.right_focus_idx = 0
        elif custom_map or has_custom_field:
            self.mapping = mapping_obj
            self.update_preview()
        # Mode / Active Tab
        tab_val = config.get("active_tab")
        if tab_val is None:
            tab_val = config.get("tab")
        if tab_val is None:
            tab_val = config.get("mode")

        if tab_val is not None:
            val_str = str(tab_val).lower().strip()
            if val_str in ("2", "multi", "multi_run", "multirun", "sweep", "multi-run"):
                self.switch_mode(2)
            elif val_str in ("1", "lora", "fine_tune", "finetune", "single", "single_run", "single-run"):
                self.switch_mode(1)
            elif val_str in ("0", "dataset", "convert", "converter", "dataset_converter", "dataset-converter"):
                self.switch_mode(0)
        elif config.get("format"):
            self.switch_mode(0)

        # Weights & Biases Tracking
        if "wandb_enabled" in config:
            self.wandb_enabled = bool(config["wandb_enabled"])
        if "wandb_project" in config and config["wandb_project"]:
            self.wandb_project = str(config["wandb_project"]).strip()
        if "wandb_project" in config:
            self.lora_config.wandb_project = self.wandb_project

    def toggle_theme(self) -> str:
        """Toggle between Modern and Norton Commander color schemes."""
        if ThemeMode.is_norton(self.theme_mode):
            self.theme_mode = ThemeMode.MODERN
        else:
            self.theme_mode = ThemeMode.NORTON
        return self.theme_mode


    def set_format(self, fmt: MLXFormat) -> None:
        """Change MLX target format and re-detect mappings if appropriate."""
        self.target_format = fmt
        if self.loaded_dataset and self.loaded_dataset.columns:
            self.mapping = auto_detect_mapping(fmt, self.loaded_dataset.columns)
        self.update_preview()

    def update_preview(self) -> None:
        """Generate live preview records based on current mapping and format."""
        self.preview_cache.clear()
        self.preview_error = None

        if not self.loaded_dataset or not self.loaded_dataset.sample_records:
            self.preview_cache = []
            return

        samples = self.loaded_dataset.sample_records[:3]
        errors = validate_mapping(self.target_format, self.mapping, self.loaded_dataset.columns)
        if errors:
            self.preview_error = f"Mapping incomplete: {', '.join(errors)}"
            return

        formatted = []
        for s in samples:
            try:
                rec = format_record(s, self.target_format, self.mapping)
                formatted.append(json.dumps(rec, ensure_ascii=False))
            except Exception as e:
                self.preview_error = f"Format error: {e}"
                return

        self.preview_cache = formatted

    def randomize_seed(self) -> None:
        """Generate a new random seed."""
        self.seed = generate_random_seed()
        self.status_message = f"Random seed generated: {self.seed}"
        self.status_is_error = False

    def get_split_counts(self) -> Dict[str, int]:
        """Calculate record counts for train / valid / test splits."""
        if not self.loaded_dataset:
            return {"train": 0, "valid": 0, "test": 0}
        n_train, n_valid, n_test = calculate_split_counts(
            self.loaded_dataset.total_rows,
            self.train_pct,
            self.valid_pct,
            self.test_pct,
        )
        return {"train": n_train, "valid": n_valid, "test": n_test}

    def get_discrete_labels_for_target(self) -> Optional[Dict[str, Any]]:
        """Return discrete label info for the current completion or assistant column."""
        if not self.loaded_dataset:
            return None
        target_col = self.mapping.completion_col or self.mapping.assistant_col
        if not target_col:
            return None
        return detect_column_discrete_labels(self.loaded_dataset, target_col)

    def get_mapping_fields_for_format(self) -> List[Dict[str, Any]]:
        """Return the column fields relevant to the current format."""
        cols = self.loaded_dataset.columns if self.loaded_dataset else []
        fields: List[Dict[str, Any]] = []

        if self.target_format == MLXFormat.TEXT:
            fields = [
                {
                    "key": "text_col",
                    "label": "Text Column",
                    "current": self.mapping.text_col,
                    "help": "Column containing causal text",
                },
                {
                    "key": "text_template",
                    "label": "Text Template",
                    "current": self.mapping.text_template,
                    "help": "Optional template e.g. {input}\n{output}",
                },
            ]
        elif self.target_format == MLXFormat.PROMPT_COMPLETION:
            fields = [
                {
                    "key": "prompt_col",
                    "label": "Prompt / Question",
                    "current": self.mapping.prompt_col,
                    "help": "Column containing prompt or instruction",
                },
                {
                    "key": "completion_col",
                    "label": "Completion / Answer",
                    "current": self.mapping.completion_col,
                    "help": "Column containing response or completion",
                },
            ]
        elif self.target_format == MLXFormat.CHAT:
            if self.mapping.messages_col:
                fields = [
                    {
                        "key": "messages_col",
                        "label": "Messages List Col",
                        "current": self.mapping.messages_col,
                        "help": "List of role/content dicts",
                    },
                ]
            else:
                fields = [
                    {
                        "key": "user_col",
                        "label": "User Turn Col",
                        "current": self.mapping.user_col,
                        "help": "Column containing user message",
                    },
                    {
                        "key": "assistant_col",
                        "label": "Assistant Col",
                        "current": self.mapping.assistant_col,
                        "help": "Column containing assistant message",
                    },
                    {
                        "key": "system_col",
                        "label": "System Prompt Col",
                        "current": self.mapping.system_col,
                        "help": "Optional system prompt column",
                    },
                ]
        elif self.target_format == MLXFormat.DPO:
            fields = [
                {
                    "key": "prompt_col",
                    "label": "Prompt Col",
                    "current": self.mapping.dpo_prompt_col or self.mapping.prompt_col,
                    "help": "Column containing prompt",
                },
                {
                    "key": "chosen_col",
                    "label": "Chosen Response",
                    "current": self.mapping.chosen_col,
                    "help": "Preferred response",
                },
                {
                    "key": "rejected_col",
                    "label": "Rejected Response",
                    "current": self.mapping.rejected_col,
                    "help": "Dispreferred response",
                },
            ]

        # System Prompt Field (available for all formats)
        sys_p = self.mapping.default_system_prompt
        sys_disp = (sys_p[:14] + "…") if sys_p and len(sys_p) > 15 else (sys_p or "<None>")
        fields.append({
            "key": "default_system_prompt",
            "label": "System Prompt",
            "current": sys_disp,
            "full_value": sys_p,
            "help": "Optional system prompt prepended/injected into all records",
            "is_system_prompt": True,
        })

        # Label Mapping Field (for formats with target/completion/assistant)
        if self.target_format in (MLXFormat.PROMPT_COMPLETION, MLXFormat.CHAT, MLXFormat.DPO):
            label_count = len(self.mapping.label_map) if self.mapping.label_map else 0
            if label_count > 0:
                label_disp = f"{label_count} mapped"
            else:
                disc = self.get_discrete_labels_for_target()
                if disc and disc.get("unique_values"):
                    label_disp = f"{len(disc['unique_values'])} discrete"
                else:
                    label_disp = "<None>"

            fields.append({
                "key": "label_map",
                "label": "Label Mapping",
                "current": label_disp,
                "help": "Map discrete labels (e.g. 0->negative, 1->neutral, 2->positive)",
                "is_label_map": True,
            })

        return fields

    def set_mapping_field(self, key: str, value: Optional[str]) -> None:
        """Update a specific mapping field and refresh preview."""
        if hasattr(self.mapping, key):
            setattr(self.mapping, key, value)
            if key == "prompt_col" and self.target_format == MLXFormat.DPO:
                self.mapping.dpo_prompt_col = value
            elif key == "dpo_prompt_col":
                self.mapping.prompt_col = value
            self.update_preview()

    def map_column_to_field(self, col_name: str, field_key: str) -> None:
        """Map a dataset column to a specific mapping field and update preview."""
        self.set_mapping_field(field_key, col_name)
