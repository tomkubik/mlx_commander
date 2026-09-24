"""
MLX Commander: Persistent Full-Screen Curses TUI Dashboard.
Dual-panel Norton Commander-style interface with real-time reactive JSONL preview.

Layout:
┌─ Dataset & Schema (Left Panel) ─────────┐┌─ MLX Format & Mappings (Right Panel) ─────┐
│ Path, Stats, and Scrollable Column List   ││ Format Radio, Column Dropdowns, Splits     │
└───────────────────────────────────────────┘└────────────────────────────────────────────┘
┌─ Live Converted Record Preview (Updates instantaneously as you edit fields) ──────────┐
│ {"prompt": "...", "completion": "..."}                                                │
└───────────────────────────────────────────────────────────────────────────────────────┘
 [Tab] Switch Pane   [↑/↓] Navigate   [Enter] Edit/Select   [F5] Convert   [F10] Exit
"""

import curses
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mlx_commander.converter import ConversionResult, convert_and_save
from mlx_commander.formats import (
    ColumnMapping,
    MLXFormat,
    auto_detect_mapping,
    format_record,
    validate_mapping,
)
from mlx_commander.loader import LoadedDataset, load_local_dataset
from mlx_commander.splitter import (
    SplitConfig,
    calculate_split_counts,
    generate_random_seed,
)
from mlx_commander.lora import (
    FINE_TUNE_TYPES,
    OPTIMIZERS,
    POPULAR_MLX_MODELS,
    LoraRunConfig,
    MultiLoraRunConfig,
    QueueManager,
    SWEEP_FIELD_DEFS,
)
from mlx_commander.terminal_spawner import (
    ensure_adequate_terminal_size,
    is_macos,
    spawn_lora_queue_terminal,
)
from mlx_commander.tui.state import ActivePanel, CommanderState, ThemeMode
from mlx_commander.tui.widgets import (
    COLOR_BANNER,
    COLOR_BORDER_JOINTS,
    COLOR_SUCCESS,
    COLOR_NORMAL_TEXT,
    COLOR_ERROR,
    COLOR_TITLE_ACCENT,
    COLOR_LABEL_GRAY,
    COLOR_INPUT_NORMAL,
    COLOR_INPUT_FOCUSED,
    COLOR_FN_NUMBER,
    COLOR_FN_LABEL,
    COLOR_PANEL_BG,
    configure_escdelay,
    draw_box_panel,
    draw_button,
    draw_field,
    draw_footer,
    draw_header,
    draw_mapping_pipeline_panel,
    draw_multi_field,
    draw_queue_table,
    draw_radio,
    draw_sweep_grid,
    get_color,
    safe_addstr,
    show_choice_dialog,
    show_column_picker_dialog,
    show_dataset_picker_dialog,
    show_dataset_source_dialog,
    show_error_dialog,
    show_help_dialog,
    show_label_mapping_dialog,
    show_message_dialog,
    show_missing_dependency_dialog,
    show_model_picker_dialog,
    show_multi_value_edit_dialog,
    show_output_destination_dialog,
    show_results_dialog,
    show_text_edit_dialog,
)


def init_nc_palette() -> Tuple[int, int, int, int, int, int, int]:
    """
    Configure and return Norton Commander VGA color indices:
      - Background Blue: #0000AA (Classic VGA Blue)
      - Highlight / Cyan: #00AAAA (Cyan)
      - Main Text / White: #FFFFFF (Bright White)
      - Label / Light Gray: #D1D1D1 (Off-White / Light Gray for high contrast on blue)
      - Cursor / Yellow: #FFFF55 (Bright Yellow)
      - Prompt / Black: #000000 (Black)
      - Inactive Input Ice Blue: #55FFFF (Bright Cyan / Ice Blue)
    """
    if curses.can_change_color():
        # Exact RGB on 0..1000 curses scale
        curses.init_color(20, 0, 0, 666)         # #0000AA (Classic VGA Blue)
        curses.init_color(21, 0, 666, 666)       # #00AAAA (Cyan)
        curses.init_color(22, 1000, 1000, 1000)  # #FFFFFF (Bright White)
        curses.init_color(23, 820, 820, 820)     # #D1D1D1 (Off-White / Light Gray for high contrast on blue)
        curses.init_color(24, 1000, 1000, 333)   # #FFFF55 (Bright Yellow)
        curses.init_color(25, 0, 0, 0)           # #000000 (Black)
        curses.init_color(26, 333, 1000, 1000)   # #55FFFF (Ice Blue / Bright Cyan)
        return 20, 21, 22, 23, 24, 25, 26
    elif curses.COLORS >= 256:
        # Closest 256-color palette slots:
        # 19: #0000af (Blue), 37: #00afaf (Cyan), 15: #ffffff (White),
        # 252: #d0d0d0 (Light Gray / Off-White), 227: #ffff5f (Yellow), 0: #000000 (Black), 51: #00ffff (Ice Blue)
        return 19, 37, 15, 252, 227, 0, 51
    else:
        # 16-color standard ANSI fallback
        return (
            curses.COLOR_BLUE,
            curses.COLOR_CYAN,
            curses.COLOR_WHITE,
            curses.COLOR_WHITE,
            curses.COLOR_YELLOW,
            curses.COLOR_BLACK,
            curses.COLOR_CYAN,
        )


def init_colors(theme_mode: Any = "modern") -> None:
    """Initialize curses color pairs for Modern or Norton Commander color schemes."""
    if not curses.has_colors():
        return
    curses.start_color()
    is_norton = ThemeMode.is_norton(theme_mode)

    if is_norton:
        c_blue, c_cyan, c_white, c_gray, c_yellow, c_black, c_ice_blue = init_nc_palette()
        # Norton Commander Classic EGA/VGA Palette:
        # Pair 1: Top Header Banner (Prompt / Black on Highlight / Cyan)
        curses.init_pair(COLOR_BANNER, c_black, c_cyan)
        # Pair 2: Highlight / Cyan Joints & Borders on Background Blue
        curses.init_pair(COLOR_BORDER_JOINTS, c_cyan, c_blue)
        # Pair 3: Success / Checked on Background Blue
        curses.init_pair(COLOR_SUCCESS, curses.COLOR_GREEN, c_blue)
        # Pair 4: Main Text / White (Standard file names and UI text) on Background Blue
        curses.init_pair(COLOR_NORMAL_TEXT, c_white, c_blue)
        # Pair 5: Error / Alerts (Main Text White on Red)
        curses.init_pair(COLOR_ERROR, c_white, curses.COLOR_RED)
        # Pair 6: Cursor / Yellow (Panel titles & accents) on Background Blue
        curses.init_pair(COLOR_TITLE_ACCENT, c_yellow, c_blue)
        # Pair 7: Label / Light Gray (Inactive elements, background text) on Background Blue
        curses.init_pair(COLOR_LABEL_GRAY, c_gray, c_blue)
        # Pair 8: User Inputs / Inactive Fields (Ice Blue on Deep Blue background)
        curses.init_pair(COLOR_INPUT_NORMAL, c_ice_blue, c_blue)
        # Pair 9: Active / Focused Field (Prompt / Black on Highlight / Cyan background)
        curses.init_pair(COLOR_INPUT_FOCUSED, c_black, c_cyan)
        # Pair 10: Function Key Number (White on Black)
        curses.init_pair(COLOR_FN_NUMBER, c_white, c_black)
        # Pair 11: Function Key Label (Black on Cyan)
        curses.init_pair(COLOR_FN_LABEL, c_black, c_cyan)
        # Pair 12: Background Blue (Panel/Window Fill: #0000AA)
        curses.init_pair(COLOR_PANEL_BG, c_white, c_blue)
    else:
        # Modern Dark Theme
        curses.use_default_colors()
        curses.init_pair(COLOR_BANNER, curses.COLOR_WHITE, curses.COLOR_BLUE)
        curses.init_pair(COLOR_BORDER_JOINTS, curses.COLOR_CYAN, -1)
        curses.init_pair(COLOR_SUCCESS, curses.COLOR_GREEN, -1)
        curses.init_pair(COLOR_NORMAL_TEXT, curses.COLOR_WHITE, -1)
        curses.init_pair(COLOR_ERROR, curses.COLOR_RED, -1)
        curses.init_pair(COLOR_TITLE_ACCENT, curses.COLOR_WHITE, -1)
        curses.init_pair(COLOR_LABEL_GRAY, curses.COLOR_WHITE, -1)
        curses.init_pair(COLOR_INPUT_NORMAL, curses.COLOR_CYAN, -1)
        curses.init_pair(COLOR_INPUT_FOCUSED, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(COLOR_FN_NUMBER, curses.COLOR_WHITE, -1)
        curses.init_pair(COLOR_FN_LABEL, curses.COLOR_CYAN, -1)
        curses.init_pair(COLOR_PANEL_BG, curses.COLOR_WHITE, -1)



def execute_conversion(stdscr: curses.window, state: CommanderState) -> Optional[ConversionResult]:
    """Validate mappings, execute conversion, and display the result modal."""
    if not state.loaded_dataset:
        state.status_message = "Please load a dataset before converting."
        state.status_is_error = True
        show_error_dialog(stdscr, "Conversion Error", "Please load a dataset before converting.")
        return None

    errors = validate_mapping(state.target_format, state.mapping, state.loaded_dataset.columns)
    if errors:
        state.status_message = f"Cannot convert: {errors[0]}"
        state.status_is_error = True
        show_error_dialog(stdscr, "Cannot Convert", errors)
        return None

    state.status_message = "Converting dataset and writing JSONL files..."
    state.status_is_error = False
    stdscr.refresh()

    split_cfg = SplitConfig(
        train_pct=state.train_pct,
        valid_pct=state.valid_pct,
        test_pct=state.test_pct,
        seed=state.seed,
    )
    try:
        res = convert_and_save(
            dataset=state.loaded_dataset,
            format_type=state.target_format,
            mapping=state.mapping,
            output_dir=Path(state.output_dir),
            split_config=split_cfg,
        )
        show_results_dialog(stdscr, res)
        state.status_message = f"[OK] Conversion complete! Saved to {res.output_dir}"
        state.status_is_error = False
        return res
    except Exception as e:
        show_error_dialog(stdscr, "Conversion Error", str(e))
        state.status_message = f"Conversion error: {e}"
        state.status_is_error = True
        return None


def prompt_vlm_batch_tweak(stdscr: Optional[curses.window], config: Any) -> bool:
    """
    Prompt user with an interactive suggestion dialog to tweak batch_size -> 1
    and gradient_accumulation_steps -> effective_batch to avoid mlx-vlm attention mask crashes.
    Returns True if user chooses Auto-tweak or Keep, or False if Cancel.
    """
    from mlx_commander.lora.vlm_safeguards import calculate_vlm_batch_tweak, should_suggest_vlm_tweak
    b = getattr(config, "batch_size", 1)
    engine = getattr(config, "engine", "mlx_lm")
    if not should_suggest_vlm_tweak(engine, b):
        return True

    t_batch, t_gas = calculate_vlm_batch_tweak(b, getattr(config, "grad_accumulation_steps", 1))
    opt_tweak = f"Auto-tweak: batch=1, grad_accum={t_gas} (Recommended)"
    opt_keep = f"Keep batch_size={b} (Experimental)"
    opt_cancel = "Cancel"

    if stdscr is None:
        config.batch_size = t_batch
        config.grad_accumulation_steps = t_gas
        return True

    choice = show_choice_dialog(
        stdscr,
        "MLX-VLM Attention Mask Notice",
        f"In mlx-vlm, batch_size={b} causes attention mask shape broadcast errors.",
        [opt_tweak, opt_keep, opt_cancel],
        current_val=opt_tweak,
    )
    if choice == opt_tweak:
        config.batch_size = t_batch
        config.grad_accumulation_steps = t_gas
        return True
    elif choice == opt_keep:
        return True
    return False


def prompt_vlm_sweep_batch_tweak(stdscr: Optional[curses.window], multi_config: Any) -> bool:
    """
    Prompt user with an interactive suggestion dialog to tweak multi-run batch_size
    values > 1 into gradient_accumulation_steps for mlx-vlm models.
    Returns True if user chooses Auto-tweak or Keep, or False if Cancel.
    """
    from mlx_commander.lora.vlm_safeguards import calculate_vlm_sweep_tweak, should_suggest_vlm_sweep_tweak
    b_list = getattr(multi_config, "batch_size", [1])
    engine = getattr(multi_config, "engine", "mlx_lm")
    if not should_suggest_vlm_sweep_tweak(engine, b_list):
        return True

    t_batches, t_gasses = calculate_vlm_sweep_tweak(b_list, getattr(multi_config, "grad_accumulation_steps", [1]))
    opt_tweak = f"Auto-tweak: batch=[1], grad_accum={t_gasses} (Recommended)"
    opt_keep = f"Keep batch_size={b_list} (Experimental)"
    opt_cancel = "Cancel"

    if stdscr is None:
        multi_config.batch_size = t_batches
        multi_config.grad_accumulation_steps = t_gasses
        return True

    choice = show_choice_dialog(
        stdscr,
        "MLX-VLM Attention Mask Notice",
        "In mlx-vlm, sweep batch_size > 1 causes attention mask broadcast errors.",
        [opt_tweak, opt_keep, opt_cancel],
        current_val=opt_tweak,
    )
    if choice == opt_tweak:
        multi_config.batch_size = t_batches
        multi_config.grad_accumulation_steps = t_gasses
        return True
    elif choice == opt_keep:
        return True
    return False


def validate_lora_queue_preconditions(
    stdscr: curses.window,
    model_path: str,
    data_path: str,
    config: Optional[Any] = None,
    multi_config: Optional[Any] = None,
) -> bool:
    """
    Validate base model and dataset before queuing or executing runs.
    Returns True if valid, or False after displaying an informative error dialog.
    """
    from mlx_commander.lora.model_info import is_local_path, normalize_model_path, is_model_directory

    # 1. Model validation
    if not model_path or model_path.strip() in ("", "None", "(No Model Selected)"):
        show_error_dialog(
            stdscr,
            "Base Model Required",
            "Please select a Base Model (Row 0) before queuing fine-tuning runs.",
        )
        return False

    healed_model = normalize_model_path(model_path.strip())
    if is_local_path(model_path) or is_local_path(healed_model):
        model_p = Path(healed_model).expanduser().resolve()
        if not model_p.exists():
            show_error_dialog(
                stdscr,
                "Model Not Found",
                f"The selected base model directory does not exist on disk:\n\n"
                f"  {model_path.strip()}\n\n"
                f"Please select a valid local model directory or Hugging Face model ID.",
            )
            return False

        if not is_model_directory(model_p):
            show_error_dialog(
                stdscr,
                "Invalid Model Directory",
                f"The directory does not appear to contain valid model files:\n\n"
                f"  {model_path.strip()}\n\n"
                f"Missing config.json or weights (.safetensors) files.",
            )
            return False

    # 2. Dataset validation
    if not data_path or data_path.strip() in ("", "None"):
        show_error_dialog(
            stdscr,
            "Dataset Required",
            "Please select a Dataset directory containing 'train.jsonl' (Row 1).",
        )
        return False

    data_p = Path(data_path.strip()).expanduser().resolve()
    if not data_p.exists():
        show_error_dialog(
            stdscr,
            "Dataset Not Found",
            f"The selected dataset path does not exist on disk:\n\n"
            f"  {data_path.strip()}\n\n"
            f"Please convert a dataset in Mode 1 or select an existing folder.",
        )
        return False

    train_file = data_p / "train.jsonl" if data_p.is_dir() else data_p
    if not train_file.exists():
        show_error_dialog(
            stdscr,
            "Missing train.jsonl",
            f"The dataset directory does not contain 'train.jsonl':\n\n"
            f"  {data_path.strip()}\n\n"
            f"Apple MLX fine-tuning requires 'train.jsonl'.",
        )
        return False

    # 3. Engine dependency validation
    from mlx_commander.lora.model_info import (
        detect_model_engine,
        is_engine_installed,
        inspect_local_model,
        validate_dataset_for_engine,
    )
    meta = inspect_local_model(model_path.strip())
    engine = getattr(meta, "engine", None) or detect_model_engine(raw_config=meta.raw_config if meta else None, model_name_or_path=model_path.strip())
    if engine == "mlx_vlm" and not is_engine_installed("mlx_vlm"):
        show_error_dialog(
            stdscr,
            "Missing Dependency: mlx-vlm",
            f"The selected model requires 'mlx-vlm' (Vision-Language / Multimodal model):\n\n"
            f"  Model: {model_path.strip()}\n\n"
            f"Please install it in your Python environment by running:\n\n"
            f"  pip install \"mlx-vlm[train]\"\n\n"
            f"Note: mlx-vlm requires Python 3.10+.",
        )
        return False

    # 4. Dataset engine compatibility and schema consistency
    is_valid_ds, ds_err = validate_dataset_for_engine(data_p, engine)
    if not is_valid_ds:
        show_error_dialog(
            stdscr,
            "Dataset Incompatible",
            ds_err or "The dataset is incompatible with the selected fine-tuning engine.",
        )
        return False

    # 5. MLX-VLM attention mask safeguard & gradient accumulation tweak
    if engine == "mlx_vlm":
        if config is not None:
            config.engine = "mlx_vlm"
            if not prompt_vlm_batch_tweak(stdscr, config):
                return False
        if multi_config is not None:
            multi_config.engine = "mlx_vlm"
            if not prompt_vlm_sweep_batch_tweak(stdscr, multi_config):
                return False

    return True


def execute_lora_queue_action(stdscr: curses.window, state: CommanderState) -> None:
    """Execute queued LoRA fine-tuning runs sequentially via spawned macOS Terminal window."""
    if not state.queue_manager or not state.queue_manager.runs:
        show_error_dialog(
            stdscr,
            "Queue Empty",
            "The LoRA queue is empty. Configure parameters and press [F6] to add runs before executing.",
        )
        return

    pending = state.queue_manager.get_pending_runs()
    if not pending:
        show_error_dialog(
            stdscr,
            "No Pending Runs",
            "All runs in the queue have already completed. Add a new run [F6] or clone an existing run [c].",
        )
        return

    # Pre-flight check on pending runs before spawning terminal
    from mlx_commander.lora.model_info import is_local_path, normalize_model_path, is_model_directory, is_engine_installed
    for r in pending:
        engine = getattr(r, "engine", "mlx_lm")
        if engine == "mlx_vlm" and not is_engine_installed("mlx_vlm"):
            show_error_dialog(
                stdscr,
                "Missing Dependency: mlx-vlm",
                f"Run '{r.name}' requires 'mlx-vlm', which is not installed:\n\n"
                f"Please run:\n"
                f"  pip install \"mlx-vlm[train]\"\n\n"
                f"before executing the queue.",
            )
            return

        healed = normalize_model_path(r.model)
        if is_local_path(r.model) or is_local_path(healed):
            if healed != r.model and Path(healed).exists():
                r.model = healed
            model_p = Path(r.model).expanduser().resolve()
            if not model_p.exists():
                show_error_dialog(
                    stdscr,
                    "Model Not Found in Run",
                    f"Run '{r.name}' specifies a local model that does not exist:\n\n"
                    f"  {r.model}\n\n"
                    f"Please update or remove this run before executing.",
                )
                return
            if not is_model_directory(model_p):
                show_error_dialog(
                    stdscr,
                    "Invalid Model Directory in Run",
                    f"Run '{r.name}' points to an invalid model directory:\n\n"
                    f"  {r.model}\n\n"
                    f"Missing config.json or weights (.safetensors) files.",
                )
                return

        data_p = Path(r.data).expanduser().resolve()
        if not data_p.exists() or not (data_p / "train.jsonl" if data_p.is_dir() else data_p).exists():
            show_error_dialog(
                stdscr,
                "Dataset Not Found in Run",
                f"Run '{r.name}' specifies a dataset without 'train.jsonl':\n\n"
                f"  {r.data}\n\n"
                f"Please update or remove this run before executing.",
            )
            return

    state.queue_manager.save()

    wb_stat = state.get_wandb_status()
    wb_desc = f"Connected ({wb_stat['project']})" if wb_stat["enabled"] else ("Offline" if not wb_stat["available"] else "Disabled/Not logged in")

    if is_macos():
        spawned = spawn_lora_queue_terminal(
            str(state.queue_manager.queue_dir),
            wandb_project=state.wandb_project,
            enable_wandb=state.wandb_enabled,
        )
        if spawned:
            show_message_dialog(
                stdscr,
                "LoRA Queue Launched",
                [
                    f"Sequential execution of {len(pending)} pending run(s) launched in macOS Terminal.",
                    "",
                    "• Runs execute sequentially to protect Apple Silicon Unified Memory.",
                    "• Real-time loss, throughput, and progress stream in the Terminal window.",
                    f"• Weights & Biases: {wb_desc}",
                    "• You may safely close MLX Commander without interrupting training.",
                ],
            )
            state.status_message = f"Queue running in Terminal window ({len(pending)} pending runs)."
            state.status_is_error = False
        else:
            show_error_dialog(stdscr, "Execution Error", "Failed to spawn external macOS Terminal.app window.")
    else:
        cmd = f"python3 -m mlx_commander --run-queue {state.queue_manager.queue_dir}"
        show_message_dialog(
            stdscr,
            "Execute LoRA Queue",
            [
                f"Queue saved to: {state.queue_manager.queue_dir}",
                "Run the following command in another terminal window:",
                "",
                cmd,
            ],
        )
        state.status_message = f"Queue saved. Run: {cmd}"




def _draw_mode1_dashboard(
    stdscr: curses.window,
    state: CommanderState,
    max_y: int,
    max_x: int,
    left_w: int,
    right_w: int,
) -> None:
    """Draw Mode 1: Dataset Converter dual-panel view with pipeline and preview."""
    mapping_fields = state.get_mapping_fields_for_format()
    min_top_h = 14 + len(mapping_fields)
    needed_top_h = max(16, min_top_h)
    max_possible_top = max(min_top_h, max_y - 10)
    panel_h = min(needed_top_h, max_possible_top)

    remaining_y = max(8, max_y - 1 - (panel_h + 1))
    vis_h = max(4, min(10, remaining_y // 2))
    vis_y = panel_h + 1

    preview_y = vis_y + vis_h
    preview_h = max(4, max_y - 1 - preview_y)

    is_left = (state.active_panel == ActivePanel.LEFT and not state.mode_switcher_focused)
    is_right = (state.active_panel == ActivePanel.RIGHT and not state.mode_switcher_focused)

    # 1. Left Panel: Dataset Source & Schema
    draw_box_panel(
        stdscr,
        1,
        0,
        panel_h,
        left_w,
        "Dataset & Schema",
        is_focused=is_left,
        subtitle="Tab 1",
    )

    d_focus = is_left and state.left_focus_idx == 0
    data_disp = state.dataset_path or "None"
    tab1_right_edge = left_w - 3
    if len(data_disp) > left_w - 18:
        data_disp = "…" + data_disp[-(left_w - 19):]
    draw_field(stdscr, 2, 2, "Dataset", data_disp, is_focused=d_focus, val_width=max(14, left_w - 18), has_dropdown=True, right_edge=tab1_right_edge)

    safe_addstr(stdscr, 3, 2, "Tip: You can load multiple files (select multiple or use commas)"[:left_w - 4], get_color(COLOR_LABEL_GRAY) if curses.has_colors() else curses.A_DIM)

    if state.loaded_dataset:
        ds = state.loaded_dataset
        safe_addstr(stdscr, 5, 2, "Rows: ", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, 5, 8, f"{ds.total_rows:,}  ", get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else 0)
        splits_x = 8 + len(f"{ds.total_rows:,}  ")
        safe_addstr(stdscr, 5, splits_x, "Splits: ", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, 5, splits_x + 8, f"{len(ds.split_names)}", get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else 0)
        split_summary = ", ".join(f"{s}: {ds.split_counts.get(s, 0):,}" for s in ds.split_names[:3])
        safe_addstr(stdscr, 6, 2, f"Found: [{split_summary}]", get_color(COLOR_LABEL_GRAY) if curses.has_colors() else curses.A_DIM)
        cols = ds.columns
    else:
        safe_addstr(stdscr, 5, 2, "No dataset loaded.", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, 6, 2, "Press Enter on Dataset to browse or enter path.", get_color(COLOR_LABEL_GRAY) if curses.has_colors() else curses.A_DIM)
        cols = []

    safe_addstr(stdscr, 7, 2, f"Columns ({len(cols)}):", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
    col_list_focus = is_left and state.left_focus_idx == 1
    col_list_start_y = 8
    col_list_rows = max(1, panel_h - 10)

    if cols:
        if state.selected_column_idx < state.column_scroll_offset:
            state.column_scroll_offset = state.selected_column_idx
        elif state.selected_column_idx >= state.column_scroll_offset + col_list_rows:
            state.column_scroll_offset = state.selected_column_idx - col_list_rows + 1

        for r in range(col_list_rows):
            c_idx = state.column_scroll_offset + r
            row_y = col_list_start_y + r
            if c_idx < len(cols):
                c_name = cols[c_idx]
                is_col_sel = col_list_focus and (c_idx == state.selected_column_idx)
                prefix = " ▶ " if is_col_sel else "   "
                text = f"{prefix}{c_idx + 1}. {c_name}"[:left_w - 4]
                if is_col_sel:
                    c_attr = (get_color(COLOR_INPUT_FOCUSED) | curses.A_BOLD) if curses.has_colors() else curses.A_STANDOUT
                elif col_list_focus:
                    c_attr = (get_color(COLOR_BORDER_JOINTS) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD
                else:
                    c_attr = get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else 0
                safe_addstr(stdscr, row_y, 2, text, c_attr)

        if 0 <= state.selected_column_idx < len(cols) and state.loaded_dataset and state.loaded_dataset.sample_records:
            col_name = cols[state.selected_column_idx]
            sample_val = str(state.loaded_dataset.sample_records[0].get(col_name, ""))
            if len(sample_val) > left_w - 12:
                sample_val = sample_val[:left_w - 13] + "…"
            safe_addstr(stdscr, panel_h - 1, 2, "Sample: ", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
            safe_addstr(stdscr, panel_h - 1, 10, sample_val, get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else curses.A_DIM)
    else:
        safe_addstr(stdscr, col_list_start_y, 4, "(Load dataset to view schema)", get_color(COLOR_LABEL_GRAY) if curses.has_colors() else curses.A_DIM)

    # 2. Right Panel: MLX Format, Mappings & Splits
    draw_box_panel(
        stdscr,
        1,
        left_w,
        panel_h,
        right_w,
        "MLX Format & Mappings",
        is_focused=is_right,
        subtitle="Tab 2",
    )

    total_right_fields = 11 + len(mapping_fields)
    if state.right_focus_idx >= total_right_fields:
        state.right_focus_idx = total_right_fields - 1

    safe_addstr(stdscr, 2, left_w + 2, "Format (MLX Target):", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
    format_options = [
        (MLXFormat.PROMPT_COMPLETION, "prompt_comp (Q&A)", "prompt_comp"),
        (MLXFormat.CHAT, "chat (Dialogue / Messages)", "chat"),
        (MLXFormat.TEXT, "text (Causal LM)", "text"),
        (MLXFormat.DPO, "dpo (Preference)", "dpo"),
    ]
    for f_i, (fmt_val, fmt_lbl, fmt_short) in enumerate(format_options):
        row_y = 3 + f_i
        lbl_text = fmt_short if right_w < 35 else fmt_lbl
        is_chk = (state.target_format == fmt_val)
        is_foc = (is_right and state.right_focus_idx == f_i)
        draw_radio(stdscr, row_y, left_w + 4, lbl_text, is_checked=is_chk, is_focused=is_foc)

    right_edge = left_w + right_w - 3
    safe_addstr(stdscr, 7, left_w + 2, "Column Mappings [Press Enter]:", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
    mapping_val_w = min(22, max(12, right_w - 24))

    for idx, f_info in enumerate(mapping_fields):
        row_y = 8 + idx
        is_f_focused = is_right and (state.right_focus_idx == 4 + idx)
        draw_field(
            stdscr,
            row_y,
            left_w + 2,
            f_info["label"],
            str(f_info["current"] or "<None>"),
            is_focused=is_f_focused,
            val_width=mapping_val_w,
            has_dropdown=True,
            right_edge=right_edge,
        )

    splits_start_y = 8 + len(mapping_fields)
    train_focus = is_right and state.right_focus_idx == 4 + len(mapping_fields)
    valid_focus = is_right and state.right_focus_idx == 5 + len(mapping_fields)
    test_focus = is_right and state.right_focus_idx == 6 + len(mapping_fields)

    safe_addstr(
        stdscr,
        splits_start_y,
        left_w + 2,
        "Dataset split:",
        (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD,
    )
    split_label_x = max(left_w + 18, right_edge - 14)
    white_unbold = get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else 0

    draw_field(stdscr, splits_start_y, split_label_x, "Train", f"{state.train_pct:.0f}%", is_focused=train_focus, val_width=6, right_edge=right_edge, lbl_attr=white_unbold)
    draw_field(stdscr, splits_start_y + 1, split_label_x, "Valid", f"{state.valid_pct:.0f}%", is_focused=valid_focus, val_width=6, right_edge=right_edge, lbl_attr=white_unbold)
    draw_field(stdscr, splits_start_y + 2, split_label_x, "Test", f"{state.test_pct:.0f}%", is_focused=test_focus, val_width=6, right_edge=right_edge, lbl_attr=white_unbold)

    seed_y = splits_start_y + 3
    seed_focus = is_right and state.right_focus_idx == 7 + len(mapping_fields)
    rand_focus = is_right and state.right_focus_idx == 8 + len(mapping_fields)

    btn_w = 17
    seed_box_w = 10
    btn_x = max(left_w + 18, right_edge - btn_w + 1)
    seed_field_x = max(left_w + 8, btn_x - 2 - seed_box_w)

    draw_field(stdscr, seed_y, left_w + 2, "Seed", str(state.seed), is_focused=seed_focus, val_width=8, field_x=seed_field_x)
    draw_button(stdscr, seed_y, btn_x, "Randomize (r)", is_focused=rand_focus)

    out_y = seed_y + 1
    out_focus = is_right and state.right_focus_idx == 9 + len(mapping_fields)
    draw_field(
        stdscr,
        out_y,
        left_w + 2,
        "Output",
        state.output_dir,
        is_focused=out_focus,
        val_width=mapping_val_w,
        has_dropdown=True,
        right_edge=right_edge,
    )

    conv_focus = is_right and state.right_focus_idx == 10 + len(mapping_fields)
    btn_y = out_y + 1
    draw_button(stdscr, btn_y, left_w + 4, "Convert Dataset (F5)", is_focused=conv_focus)

    # 3. Middle Panel: Visual Column Mapping Pipeline
    draw_mapping_pipeline_panel(
        stdscr,
        vis_y,
        0,
        vis_h,
        max_x,
        state,
    )

    # 4. Bottom Panel: Live Converted Record Preview
    draw_box_panel(
        stdscr,
        preview_y,
        0,
        preview_h,
        max_x,
        "Live Converted Record Preview (MLX JSONL Format)",
        is_focused=False,
        subtitle=f"{state.target_format.value.upper()}",
    )

    if not state.loaded_dataset:
        safe_addstr(stdscr, preview_y + 1, 3, "(No dataset selected)", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, preview_y + 2, 3, "Select a dataset in Tab 1 to preview records.", get_color(COLOR_LABEL_GRAY) if curses.has_colors() else curses.A_DIM)
    elif state.preview_error:
        safe_addstr(stdscr, preview_y + 1, 3, f"[!] {state.preview_error}", get_color(5) | curses.A_BOLD)
        safe_addstr(stdscr, preview_y + 2, 3, "Adjust column mappings in the Right Panel (Tab 2) to preview records.", get_color(COLOR_LABEL_GRAY) if curses.has_colors() else curses.A_DIM)
    elif not state.preview_cache:
        safe_addstr(stdscr, preview_y + 1, 3, "(No records to preview)", get_color(COLOR_LABEL_GRAY) if curses.has_colors() else curses.A_DIM)
    else:
        avail_w = max(20, max_x - 10)
        avail_rows = max(1, preview_h - 2)

        all_wrapped: List[List[str]] = []
        for line in state.preview_cache:
            wrapped = textwrap.wrap(
                line,
                width=avail_w,
                break_long_words=True,
                break_on_hyphens=False,
            )
            all_wrapped.append(wrapped if wrapped else [""])

        n_examples = len(all_wrapped)
        lengths = [len(w) for w in all_wrapped]
        allocated = [0] * n_examples
        rem_rows = avail_rows

        for i in range(n_examples):
            if rem_rows > 0:
                allocated[i] = 1
                rem_rows -= 1

        while rem_rows > 0:
            candidates = [i for i in range(n_examples) if allocated[i] < lengths[i]]
            if not candidates:
                break
            for i in candidates:
                if rem_rows > 0 and allocated[i] < lengths[i]:
                    allocated[i] += 1
                    rem_rows -= 1

        curr_row = preview_y + 1
        for i, wrapped_lines in enumerate(all_wrapped):
            num_lines = allocated[i]
            for line_idx in range(num_lines):
                if curr_row >= preview_y + preview_h - 1:
                    break
                line_text = wrapped_lines[line_idx]
                if line_idx == num_lines - 1 and num_lines < len(wrapped_lines):
                    if len(line_text) > avail_w - 5:
                        line_text = line_text[:avail_w - 5].rstrip() + " ...}"
                    else:
                        line_text = line_text + " ...}"

                if line_idx == 0:
                    safe_addstr(stdscr, curr_row, 3, f"{i + 1}: ", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
                    safe_addstr(stdscr, curr_row, 6, line_text, (get_color(COLOR_NORMAL_TEXT) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
                else:
                    safe_addstr(stdscr, curr_row, 6, line_text, get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else 0)
                curr_row += 1


def _draw_mode2_dashboard(
    stdscr: curses.window,
    state: CommanderState,
    max_y: int,
    max_x: int,
    left_w: int,
    right_w: int,
) -> None:
    """Draw Mode 2: Apple MLX LoRA Fine-Tuning Dashboard & Queue."""
    # Height allocations across 3 tiers:
    # Top tier (Left & Right panels): prioritized to panel_h >= 19 so the hyperparameter pane
    # always displays the full list of 14 hyperparameters without scrolling.
    # Middle tier (Resource & Training Estimates): 5-6 rows.
    # Bottom tier (Fine-Tuning Queue): shrunk to save vertical space; supports full scrolling.
    if max_y >= 30:
        panel_h = 19
        vis_h = 6
    elif max_y >= 28:
        vis_h = 5
        panel_h = min(19, max_y - 5 - vis_h)
    elif max_y >= 25:
        vis_h = 4
        panel_h = max(12, max_y - 5 - vis_h)
    else:
        vis_h = 4
        panel_h = max(10, max_y - 9)

    vis_y = panel_h + 1
    queue_y = vis_y + vis_h
    queue_h = max(3, max_y - 1 - queue_y)

    is_lora_left = (state.lora_active_panel == "left" and not state.mode_switcher_focused)
    is_lora_right = (state.lora_active_panel == "right" and not state.mode_switcher_focused)
    is_lora_queue = (state.lora_active_panel == "queue" and not state.mode_switcher_focused)

    # 1. Left Panel: Model & Dataset Selector (All input fields right-aligned to left_right_edge)
    left_right_edge = left_w - 3
    draw_box_panel(
        stdscr,
        1,
        0,
        panel_h,
        left_w,
        "Model & Dataset Selector (LoRA Base)",
        is_focused=is_lora_left,
        subtitle="Step 1: Setup",
    )

    # Field 0: Base Model
    m_focus = is_lora_left and state.lora_left_focus_idx == 0
    model_disp = state.lora_config.model or "None"
    if len(model_disp) > left_w - 18:
        model_disp = "…" + model_disp[-(left_w - 19):]
    draw_field(stdscr, 2, 2, "Base Model", model_disp, is_focused=m_focus, val_width=max(14, left_w - 18), has_dropdown=True, right_edge=left_right_edge)

    # Engine badge & Architecture specs
    meta = state.inspect_current_model()
    is_vlm = getattr(meta, "is_vlm", False) or getattr(state.lora_config, "engine", "mlx_lm") == "mlx_vlm"
    badge_str = "[MLX-VLM]" if is_vlm else "[MLX-LM]"
    gray_dim = (get_color(COLOR_LABEL_GRAY) | curses.A_DIM) if curses.has_colors() else curses.A_DIM
    badge_attr = (get_color(COLOR_TITLE_ACCENT) | curses.A_BOLD) if (curses.has_colors() and is_vlm) else gray_dim
    safe_addstr(stdscr, 3, 2, badge_str, badge_attr)
    specs = meta.format_hyperparameters_line() if meta and meta.is_valid else ""
    if specs:
        safe_addstr(stdscr, 3, 2 + len(badge_str) + 1, specs[:left_w - 5 - len(badge_str)], gray_dim)

    # Field 1: Dataset Directory
    d_focus = is_lora_left and state.lora_left_focus_idx == 1
    data_disp = state.lora_config.data or "None"
    if len(data_disp) > left_w - 18:
        data_disp = "…" + data_disp[-(left_w - 19):]
    draw_field(stdscr, 4, 2, "Dataset", data_disp, is_focused=d_focus, val_width=max(14, left_w - 18), has_dropdown=True, right_edge=left_right_edge)

    # Field 2: Technique
    type_focus = is_lora_left and state.lora_left_focus_idx == 2
    draw_field(stdscr, 6, 2, "Method", state.lora_config.fine_tune_type.upper(), is_focused=type_focus, val_width=10, has_dropdown=True, right_edge=left_right_edge)

    # Field 3: Optimizer
    opt_focus = is_lora_left and state.lora_left_focus_idx == 3
    draw_field(stdscr, 7, 2, "Optimizer", state.lora_config.optimizer, is_focused=opt_focus, val_width=12, has_dropdown=True, right_edge=left_right_edge)

    # Field 4: Mode (Train / Test)
    mode_focus = is_lora_left and state.lora_left_focus_idx == 4
    mode_str = f"Train: {'Yes' if state.lora_config.train else 'No'}  Test: {'Yes' if state.lora_config.test else 'No'}"
    draw_field(stdscr, 8, 2, "Mode", mode_str, is_focused=mode_focus, val_width=22, right_edge=left_right_edge)

    # Field 5: Run Name
    name_focus = is_lora_left and state.lora_left_focus_idx == 5
    name_disp = state.lora_config.name or "<auto>"
    if len(name_disp) > left_w - 18:
        name_disp = "…" + name_disp[-(left_w - 19):]
    draw_field(stdscr, 9, 2, "Run Name", name_disp, is_focused=name_focus, val_width=max(14, left_w - 18), right_edge=left_right_edge)

    # 2. Right Panel: Hyperparameters & Base Model Architecture (White non-bold specs)
    wb = state.get_wandb_status()
    if wb["enabled"]:
        wb_badge = f"W&B: @{wb.get('entity') or 'active'} ({wb.get('project')})"
    elif wb["available"]:
        wb_badge = "W&B: Not Logged In"
    else:
        wb_badge = "W&B: Offline"

    draw_box_panel(
        stdscr,
        1,
        left_w,
        panel_h,
        right_w,
        "LoRA Hyperparameters",
        is_focused=is_lora_right,
        subtitle="Step 2: Config",
    )

    right_edge = max_x - 3
    cfg = state.lora_config

    white_unbold = get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else 0
    field_start_y = 2
    inner_h = max(1, panel_h - 2)

    fields_def = [
        (0, "Training Iterations", str(cfg.iters), 10, False),
        (1, "Batch Size", str(cfg.batch_size), 8, False),
        (2, "Gradient Accumulation Steps", str(getattr(cfg, "grad_accumulation_steps", 1)), 8, False),
        (3, "Learning Rate", f"{cfg.learning_rate:g}", 12, False),
        (4, "LoRA Rank (r)", str(cfg.lora_rank), 8, False),
        (5, "LoRA Alpha (α)", f"{cfg.lora_alpha:g}", 8, False),
        (6, "LoRA Dropout", f"{cfg.lora_dropout:g}", 8, False),
        (7, "Max Seq Length", str(cfg.max_seq_length), 10, False),
        (8, "Fine-Tuned Layers", str(cfg.num_layers), 8, False),
        (9, "Grad Checkpoint", "True" if cfg.grad_checkpoint else "False", 10, False),
        (10, "Mask Prompt", "True" if cfg.mask_prompt else "False", 10, False),
        (11, "Save Every", str(cfg.save_every), 8, False),
        (12, 'Steps per "Eval" (validation loss)', str(cfg.steps_per_eval), 8, False),
        (13, "Run evals on test set (experimental)", "Yes" if getattr(cfg, "run_eval", False) else "No", 8, False),
        (14, "Adapter Path", cfg.adapter_path, max(14, right_w - 20), False),
        (15, "+ Add to Queue (F6)", "", 0, True),
    ]

    # Handle smooth scrolling in single-column hyperparameter list
    if state.lora_right_focus_idx < state.lora_right_scroll_offset:
        state.lora_right_scroll_offset = state.lora_right_focus_idx
    elif state.lora_right_focus_idx >= state.lora_right_scroll_offset + inner_h:
        state.lora_right_scroll_offset = state.lora_right_focus_idx - inner_h + 1
    scroll_off = state.lora_right_scroll_offset

    for idx_in_view in range(min(inner_h, len(fields_def) - scroll_off)):
        f_idx = scroll_off + idx_in_view
        f_num, f_label, f_val, f_val_w, is_btn = fields_def[f_idx]
        f_y = field_start_y + idx_in_view
        is_foc = is_lora_right and (state.lora_right_focus_idx == f_num)

        if is_btn:
            btn_str = "+ Add to Queue (F6)"
            btn_w = len(btn_str) + 4
            draw_button(stdscr, f_y, right_edge - btn_w + 1, btn_str, is_focused=is_foc)
        else:
            draw_field(stdscr, f_y, left_w + 2, f_label, f_val, is_focused=is_foc, val_width=f_val_w, right_edge=right_edge)

    # Visual scroll indicators for Right Panel
    if scroll_off > 0:
        safe_addstr(stdscr, field_start_y - 1, max_x - 4, "▲", (get_color(COLOR_TITLE_ACCENT) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
    if scroll_off + inner_h < len(fields_def):
        safe_addstr(stdscr, panel_h, max_x - 4, "▼", (get_color(COLOR_TITLE_ACCENT) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)

    # 3. Middle Panel: Runtime Estimates (Implied Epochs, Peak RAM, Duration)
    draw_box_panel(
        stdscr,
        vis_y,
        0,
        vis_h,
        max_x,
        "Runtime Estimates",
        is_focused=False,
        subtitle=wb_badge,
    )

    # Implied Epochs (Row 1)
    epochs = state.get_implied_epochs()
    train_count = state.get_train_record_count()

    gray_unbold = (get_color(COLOR_LABEL_GRAY) | curses.A_DIM) if curses.has_colors() else curses.A_DIM

    if epochs is not None:
        lbl_part = f"• Implied Epochs:   {epochs:.2f} epochs"
        if getattr(cfg, "grad_accumulation_steps", 1) > 1:
            calc_part = f"  [(iters: {cfg.iters:,} × batch: {cfg.batch_size} × accum: {cfg.grad_accumulation_steps}) / {train_count:,} train records]"
        else:
            calc_part = f"  [(iters: {cfg.iters:,} × batch: {cfg.batch_size}) / {train_count:,} train records]"
        epoch_attr = (get_color(COLOR_ERROR) if curses.has_colors() else 0) if epochs < 1.0 else white_unbold
        safe_addstr(stdscr, vis_y + 1, 2, lbl_part[:max_x - 4], epoch_attr)
        if len(lbl_part) + 2 < max_x - 4:
            safe_addstr(stdscr, vis_y + 1, 2 + len(lbl_part), calc_part[:max_x - 4 - len(lbl_part)], gray_unbold)
    else:
        lbl_part = "• Implied Epochs:   "
        calc_part = "(Select or specify dataset containing train.jsonl to calculate implied epochs)"
        safe_addstr(stdscr, vis_y + 1, 2, lbl_part[:max_x - 4], white_unbold)
        if len(lbl_part) + 2 < max_x - 4:
            safe_addstr(stdscr, vis_y + 1, 2 + len(lbl_part), calc_part[:max_x - 4 - len(lbl_part)], gray_unbold)

    # Peak Unified RAM & Safety Rating (Row 2)
    mem_est = state.get_memory_estimate()
    lvl = mem_est.get("safety_level", "SAFE")
    if lvl == "SAFE":
        lvl_attr = get_color(COLOR_SUCCESS) if curses.has_colors() else 0
        desc = "Safe fine-tuning headroom"
    elif lvl == "TIGHT":
        lvl_attr = get_color(COLOR_TITLE_ACCENT) if curses.has_colors() else 0
        desc = "Close background applications"
    else:
        lvl_attr = get_color(COLOR_ERROR) if curses.has_colors() else 0
        desc = "Reduce batch size or enable grad checkpoint"

    peak_gb = mem_est.get('peak_gb', 0)
    total_ram = mem_est.get('total_ram_gb', 16)
    pct = int(round(mem_est.get('usage_ratio', 0) * 100))
    ram_main = f"• Peak Unified RAM: {peak_gb} GB / {total_ram} GB ({pct}%)  [{lvl}]"
    ram_desc = f"  ({desc})"
    safe_addstr(stdscr, vis_y + 2, 2, ram_main[:max_x - 4], lvl_attr)
    if len(ram_main) + 2 < max_x - 4:
        safe_addstr(stdscr, vis_y + 2, 2 + len(ram_main), ram_desc[:max_x - 4 - len(ram_main)], gray_unbold)

    # Memory Breakdown (Row 3) & Duration (Row 4)
    m_base = mem_est.get('base_model_gb', mem_est.get('model_gb', 0))
    m_act = mem_est.get('activations_gb', mem_est.get('act_gb', 0))
    m_opt = mem_est.get('optimizer_gb', mem_est.get('lora_opt_gb', 0))
    breakdown_line = f"  RAM Breakdown:    Base Model: {m_base} GB │ Activations: {m_act} GB │ LoRA Optimizer (fp32): {m_opt} GB"
    dur_est = state.get_duration_estimate()
    dur_line = f"• Est. Duration:     {dur_est.get('duration_str', '1m')} (ETA: {dur_est.get('eta_clock', 'N/A')})"

    if vis_h >= 6:
        safe_addstr(stdscr, vis_y + 3, 2, breakdown_line[:max_x - 4], gray_unbold)
        safe_addstr(stdscr, vis_y + 4, 2, dur_line[:max_x - 4], white_unbold)
    else:
        safe_addstr(stdscr, vis_y + 3, 2, dur_line[:max_x - 4], white_unbold)
        bd_compact = f" │ Base: {m_base} GB │ Act: {m_act} GB │ Opt: {m_opt} GB"
        if len(dur_line) + 2 < max_x - 4:
            safe_addstr(stdscr, vis_y + 3, 2 + len(dur_line), bd_compact[:max_x - 4 - len(dur_line)], gray_unbold)

    # 4. Bottom Panel: Queued Runs (with Config Spec displayed underneath each run)
    runs_list = state.queue_manager.runs if state.queue_manager else []
    draw_box_panel(
        stdscr,
        queue_y,
        0,
        queue_h,
        max_x,
        "Fine-Tuning Queue (Sequential Execution)",
        is_focused=is_lora_queue,
        subtitle=f"{len(runs_list)} run(s) queued (↑/↓ to scroll) │ F5: Run Queue │ c: Clone │ d: Delete │ x: Clear │ Enter: Load",
    )
    state.selected_queue_idx = draw_queue_table(
        stdscr,
        queue_y + 1,
        2,
        queue_h - 2,
        max_x - 4,
        runs_list,
        selected_idx=state.selected_queue_idx,
        is_focused=is_lora_queue,
        scroll_offset=state.lora_queue_scroll_offset,
    )


def _draw_mode3_dashboard(
    stdscr: curses.window,
    state: CommanderState,
    max_y: int,
    max_x: int,
    left_w: int,
    right_w: int,
) -> None:
    """Draw Mode 3: Apple MLX LoRA Fine-Tuning Multi-Run Dashboard & Sweep Grid."""
    if max_y >= 30:
        panel_h = 19
        vis_h = 6
    elif max_y >= 28:
        vis_h = 5
        panel_h = min(19, max_y - 5 - vis_h)
    elif max_y >= 25:
        vis_h = 4
        panel_h = max(12, max_y - 5 - vis_h)
    else:
        vis_h = 4
        panel_h = max(10, max_y - 9)

    vis_y = panel_h + 1
    sweep_y = vis_y + vis_h
    sweep_h = max(3, max_y - 1 - sweep_y)

    is_multi_left = (state.multi_active_panel == "left" and not state.mode_switcher_focused)
    is_multi_right = (state.multi_active_panel == "right" and not state.mode_switcher_focused)
    is_multi_sweep = (state.multi_active_panel == "sweep" and not state.mode_switcher_focused)

    # 1. Left Panel: Model & Dataset Selector (identical setup to Mode 2)
    left_right_edge = left_w - 3
    draw_box_panel(
        stdscr,
        1,
        0,
        panel_h,
        left_w,
        "Model & Dataset Selector (LoRA Base)",
        is_focused=is_multi_left,
        subtitle="Step 1: Setup",
    )

    # Field 0: Base Model
    m_focus = is_multi_left and state.multi_left_focus_idx == 0
    model_disp = state.multi_lora_config.model or "None"
    if len(model_disp) > left_w - 18:
        model_disp = "…" + model_disp[-(left_w - 19):]
    draw_field(stdscr, 2, 2, "Base Model", model_disp, is_focused=m_focus, val_width=max(14, left_w - 18), has_dropdown=True, right_edge=left_right_edge)

    # Engine badge & Architecture specs
    meta = state.inspect_current_model()
    is_vlm = getattr(meta, "is_vlm", False) or getattr(state.multi_lora_config, "engine", "mlx_lm") == "mlx_vlm"
    badge_str = "[MLX-VLM]" if is_vlm else "[MLX-LM]"
    gray_dim = (get_color(COLOR_LABEL_GRAY) | curses.A_DIM) if curses.has_colors() else curses.A_DIM
    badge_attr = (get_color(COLOR_TITLE_ACCENT) | curses.A_BOLD) if (curses.has_colors() and is_vlm) else gray_dim
    safe_addstr(stdscr, 3, 2, badge_str, badge_attr)
    specs = meta.format_hyperparameters_line() if meta and meta.is_valid else ""
    if specs:
        safe_addstr(stdscr, 3, 2 + len(badge_str) + 1, specs[:left_w - 5 - len(badge_str)], gray_dim)

    # Field 1: Dataset Directory
    d_focus = is_multi_left and state.multi_left_focus_idx == 1
    data_disp = state.multi_lora_config.data or "None"
    if len(data_disp) > left_w - 18:
        data_disp = "…" + data_disp[-(left_w - 19):]
    draw_field(stdscr, 4, 2, "Dataset", data_disp, is_focused=d_focus, val_width=max(14, left_w - 18), has_dropdown=True, right_edge=left_right_edge)

    # Field 2: Technique
    type_focus = is_multi_left and state.multi_left_focus_idx == 2
    draw_field(stdscr, 6, 2, "Method", state.multi_lora_config.fine_tune_type.upper(), is_focused=type_focus, val_width=10, has_dropdown=True, right_edge=left_right_edge)

    # Field 3: Optimizer
    opt_focus = is_multi_left and state.multi_left_focus_idx == 3
    draw_field(stdscr, 7, 2, "Optimizer", state.multi_lora_config.optimizer, is_focused=opt_focus, val_width=12, has_dropdown=True, right_edge=left_right_edge)

    # Field 4: Mode (Train / Test)
    mode_focus = is_multi_left and state.multi_left_focus_idx == 4
    mode_str = f"Train: {'Yes' if state.multi_lora_config.train else 'No'}  Test: {'Yes' if state.multi_lora_config.test else 'No'}"
    draw_field(stdscr, 8, 2, "Mode", mode_str, is_focused=mode_focus, val_width=22, right_edge=left_right_edge)

    # Field 5: Adapter Base Path
    adp_focus = is_multi_left and state.multi_left_focus_idx == 5
    adp_disp = state.multi_lora_config.adapter_path or "adapters"
    if len(adp_disp) > left_w - 18:
        adp_disp = "…" + adp_disp[-(left_w - 19):]
    draw_field(stdscr, 9, 2, "Adapters Dir", adp_disp, is_focused=adp_focus, val_width=max(14, left_w - 18), right_edge=left_right_edge)

    # 2. Right Panel: Multi-Run LoRA Hyperparameters
    wb = state.get_wandb_status()
    if wb["enabled"]:
        wb_badge = f"W&B: @{wb.get('entity') or 'active'} ({wb.get('project')})"
    elif wb["available"]:
        wb_badge = "W&B: Not Logged In"
    else:
        wb_badge = "W&B: Offline"

    draw_box_panel(
        stdscr,
        1,
        left_w,
        panel_h,
        right_w,
        "LoRA Hyperparameters (Multi-Run)",
        is_focused=is_multi_right,
        subtitle="Step 2: Conditions",
    )

    right_edge = max_x - 3
    cfg = state.multi_lora_config

    white_unbold = get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else 0
    field_start_y = 2
    inner_h = max(1, panel_h - 2)

    # Multi-field definitions: (field_num, label, values_list, val_type, is_btn)
    multi_fields_def = [
        (0, "Training Iterations", cfg.iters, int, False),
        (1, "Batch Size", cfg.batch_size, int, False),
        (2, "Gradient Accumulation Steps", cfg.grad_accumulation_steps, int, False),
        (3, "Learning Rate", cfg.learning_rate, float, False),
        (4, "LoRA Rank (r)", cfg.lora_rank, int, False),
        (5, "LoRA Alpha (α)", cfg.lora_alpha, float, False),
        (6, "LoRA Dropout", cfg.lora_dropout, float, False),
        (7, "Max Seq Length", cfg.max_seq_length, int, False),
        (8, "Fine-Tuned Layers", cfg.num_layers, int, False),
        (9, "Grad Checkpoint", cfg.grad_checkpoint, bool, False),
        (10, "Mask Prompt", cfg.mask_prompt, bool, False),
        (11, "Save Every", cfg.save_every, int, False),
        (12, 'Steps per "Eval" (validation loss)', cfg.steps_per_eval, int, False),
        (13, "Run evals on test set (experimental)", getattr(cfg, "run_eval", [False]), bool, False),
        (14, "+ Add Sweep to Queue (F6)", [], None, True),
    ]

    # Handle smooth scrolling in right panel
    if state.multi_right_focus_idx < state.multi_right_scroll_offset:
        state.multi_right_scroll_offset = state.multi_right_focus_idx
    elif state.multi_right_focus_idx >= state.multi_right_scroll_offset + inner_h:
        state.multi_right_scroll_offset = state.multi_right_focus_idx - inner_h + 1
    scroll_off = state.multi_right_scroll_offset

    for idx_in_view in range(min(inner_h, len(multi_fields_def) - scroll_off)):
        f_idx = scroll_off + idx_in_view
        f_num, f_label, f_vals, f_type, is_btn = multi_fields_def[f_idx]
        f_y = field_start_y + idx_in_view
        is_foc = is_multi_right and (state.multi_right_focus_idx == f_num)

        if is_btn:
            btn_str = "+ Add Sweep to Queue (F6)"
            btn_w = len(btn_str) + 4
            draw_button(stdscr, f_y, right_edge - btn_w + 1, btn_str, is_focused=is_foc)
        else:
            draw_multi_field(stdscr, f_y, left_w + 2, f_label, f_vals, is_focused=is_foc, right_edge=right_edge)

    # Visual scroll indicators
    if scroll_off > 0:
        safe_addstr(stdscr, field_start_y - 1, max_x - 4, "▲", (get_color(COLOR_TITLE_ACCENT) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
    if scroll_off + inner_h < len(multi_fields_def):
        safe_addstr(stdscr, panel_h, max_x - 4, "▼", (get_color(COLOR_TITLE_ACCENT) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)

    # 3. Middle Panel: Runtime Estimates (Aggregated across all runs in sweep)
    draw_box_panel(
        stdscr,
        vis_y,
        0,
        vis_h,
        max_x,
        "Runtime Estimates (Sweep Aggregated)",
        is_focused=False,
        subtitle=wb_badge,
    )

    sweep_est = state.get_multi_sweep_estimates()
    total_runs = sweep_est.get("total_runs", cfg.total_runs_count)
    min_epochs = sweep_est.get("min_implied_epochs")
    max_ram_gb = sweep_est.get("max_peak_ram_gb", 0)
    total_ram = sweep_est.get("total_ram_gb", 16)
    safety_lvl = sweep_est.get("safety_level", "SAFE")
    dur_str = sweep_est.get("total_duration_str", "0s")

    gray_unbold = (get_color(COLOR_LABEL_GRAY) | curses.A_DIM) if curses.has_colors() else curses.A_DIM

    # Row 1: Implied Epochs (Minimum across all sweep conditions, highlighted in dark red if < 1.0)
    if min_epochs is not None:
        lbl_part = f"• Min Implied Epochs: {min_epochs:.2f} epochs"
        calc_part = f"  (minimum across all {total_runs} scheduled runs; < 1.0 indicates incomplete dataset pass)"
        epoch_attr = (get_color(COLOR_ERROR) if curses.has_colors() else 0) if min_epochs < 1.0 else white_unbold
        safe_addstr(stdscr, vis_y + 1, 2, lbl_part[:max_x - 4], epoch_attr)
        if len(lbl_part) + 2 < max_x - 4:
            safe_addstr(stdscr, vis_y + 1, 2 + len(lbl_part), calc_part[:max_x - 4 - len(lbl_part)], gray_unbold)
    else:
        lbl_part = "• Min Implied Epochs: "
        calc_part = "(Select dataset to calculate implied epochs across sweep runs)"
        safe_addstr(stdscr, vis_y + 1, 2, lbl_part[:max_x - 4], white_unbold)
        if len(lbl_part) + 2 < max_x - 4:
            safe_addstr(stdscr, vis_y + 1, 2 + len(lbl_part), calc_part[:max_x - 4 - len(lbl_part)], gray_unbold)

    # Row 2: Maximum Peak Unified RAM across sweep
    if safety_lvl == "SAFE":
        lvl_attr = get_color(COLOR_SUCCESS) if curses.has_colors() else 0
        desc = "Safe fine-tuning headroom for all runs"
    elif safety_lvl == "TIGHT":
        lvl_attr = get_color(COLOR_TITLE_ACCENT) if curses.has_colors() else 0
        desc = "Close background apps; high RAM condition present"
    else:
        lvl_attr = get_color(COLOR_ERROR) if curses.has_colors() else 0
        desc = "Reduce batch size or enable grad checkpoint in large-rank conditions"

    pct = int(round((max_ram_gb / max(1, total_ram)) * 100))
    ram_main = f"• Max Peak Unified RAM: {max_ram_gb} GB / {total_ram} GB ({pct}%)  [{safety_lvl}]"
    ram_desc = f"  ({desc})"
    safe_addstr(stdscr, vis_y + 2, 2, ram_main[:max_x - 4], lvl_attr)
    if len(ram_main) + 2 < max_x - 4:
        safe_addstr(stdscr, vis_y + 2, 2 + len(ram_main), ram_desc[:max_x - 4 - len(ram_main)], gray_unbold)

    # Row 3: Total Sweep Duration
    dur_line = f"• Total Sweep Duration: {dur_str}  ({total_runs} runs scheduled sequentially)"
    safe_addstr(stdscr, vis_y + 3, 2, dur_line[:max_x - 4], white_unbold)

    # 4. Bottom Panel: Cartesian Multi-Run Sweep Grid
    varying = cfg.get_varying_hyperparameters()
    draw_box_panel(
        stdscr,
        sweep_y,
        0,
        sweep_h,
        max_x,
        "Multi-Run Parameter Sweep",
        is_focused=is_multi_sweep,
        subtitle=f"{total_runs} run(s) scheduled │ F5: Run Sweep │ F6: Add to Queue │ Tab: Switch Panels",
    )
    draw_sweep_grid(
        stdscr,
        sweep_y + 1,
        2,
        sweep_h - 2,
        max_x - 4,
        varying,
        total_runs,
    )


def _handle_mode1_input(
    stdscr: curses.window,
    state: CommanderState,
    key: int,
    formats_list: List[MLXFormat],
) -> Optional[ConversionResult]:
    """Handle keyboard input specific to Mode 1 (Dataset Conversion)."""
    mapping_fields = state.get_mapping_fields_for_format()
    total_right_fields = 11 + len(mapping_fields)
    is_left = (state.active_panel == ActivePanel.LEFT)
    is_right = (state.active_panel == ActivePanel.RIGHT)

    if key in (curses.KEY_F3,):  # F3: Destination Folder
        curses.def_prog_mode()
        curses.endwin()
        from mlx_commander.gui_picker import is_macos, pick_folder_gui
        if is_macos():
            start_dir = state.output_dir or (str(state.loaded_dataset.default_output_dir) if state.loaded_dataset else os.getcwd())
            chosen = pick_folder_gui("Select Destination Folder", default_dir=start_dir)
        else:
            chosen = None
        curses.reset_prog_mode()
        stdscr.refresh()
        if chosen:
            state.output_dir = chosen.strip()
            default_out = str(state.loaded_dataset.default_output_dir) if state.loaded_dataset else ""
            state.has_custom_output_dir = (state.output_dir != default_out)
            state.status_message = f"Output folder set to: {state.output_dir}"
            state.status_is_error = False
        elif not is_macos():
            default_out = str(state.loaded_dataset.default_output_dir) if state.loaded_dataset else state.output_dir
            val = show_text_edit_dialog(
                stdscr,
                "Output Directory",
                "Enter folder to save MLX JSONL datasets:",
                default_val=state.output_dir or default_out,
            )
            if val:
                state.output_dir = val.strip()
                state.has_custom_output_dir = (state.output_dir != default_out)
                state.status_message = f"Output folder set to: {state.output_dir}"
                state.status_is_error = False

    elif key in (ord("r"), ord("R")):
        state.randomize_seed()

    elif key in (9, curses.KEY_BTAB):  # Tab / Shift-Tab
        state.active_panel = ActivePanel.RIGHT if state.active_panel == ActivePanel.LEFT else ActivePanel.LEFT

    # Navigation in Left Panel (Tab 1)
    elif is_left:
        num_cols = len(state.loaded_dataset.columns) if (state.loaded_dataset and state.loaded_dataset.columns) else 0
        if key in (curses.KEY_UP, ord("k")):
            if state.left_focus_idx == 1 and state.selected_column_idx > 0:
                state.selected_column_idx -= 1
            elif state.left_focus_idx == 1:
                state.left_focus_idx = 0
            elif state.left_focus_idx == 0:
                state.mode_switcher_focused = True
                state.mode_switcher_idx = state.active_tab
                state.status_message = "Mode Switcher: Use [←/→] to select mode, [Enter] to switch, [↓] to return to pane."
                state.status_is_error = False
        elif key in (curses.KEY_LEFT, ord("h")):
            if state.left_focus_idx == 1 and state.selected_column_idx > 0:
                state.selected_column_idx -= 1
            elif state.left_focus_idx == 1:
                state.left_focus_idx = 0
            elif state.left_focus_idx == 0:
                state.active_panel = ActivePanel.RIGHT
                state.right_focus_idx = total_right_fields - 1
        elif key in (curses.KEY_DOWN, curses.KEY_RIGHT, ord("j"), ord("l")):
            if state.left_focus_idx == 0:
                if num_cols > 0:
                    state.left_focus_idx = 1
                    state.selected_column_idx = 0
                else:
                    state.active_panel = ActivePanel.RIGHT
                    state.right_focus_idx = 0
            elif state.left_focus_idx == 1:
                if state.selected_column_idx < num_cols - 1:
                    state.selected_column_idx += 1
                else:
                    state.active_panel = ActivePanel.RIGHT
                    state.right_focus_idx = 0
        elif key in (10, 13, curses.KEY_ENTER, 32):  # Enter or Space
            if state.left_focus_idx == 0:  # Dataset field
                chosen = show_dataset_source_dialog(stdscr, state.dataset_path)
                if chosen:
                    from mlx_commander.lora.model_info import is_model_directory, inspect_local_model
                    chosen_p = Path(chosen.strip()).expanduser()
                    if is_model_directory(chosen_p):
                        meta = inspect_local_model(str(chosen_p))
                        size_txt = f" ({meta.architecture}, {meta.file_size_gb:.1f} GB)" if meta.file_size_gb > 0 else ""
                        switch_opt = "Switch to Mode 2 (Single Run) & set as Base Model"
                        cancel_opt = "Cancel (Stay in Dataset Converter)"
                        choice = show_choice_dialog(
                            stdscr,
                            "Model Directory Selected",
                            f"'{meta.name}' is a Base Model{size_txt}, not a training dataset.",
                            [switch_opt, cancel_opt],
                            default_choice=switch_opt,
                        )
                        if choice == switch_opt:
                            state.lora_config.model = meta.path
                            state.multi_lora_config.model = meta.path
                            state.inspect_current_model(force=True)
                            state.update_deterministic_lora_name()
                            state.switch_mode(1)
                            state.status_message = f"Base model set to: {meta.name}. Now select your training dataset below."
                            state.status_is_error = False
                            if getattr(meta, "engine", "mlx_lm") == "mlx_vlm":
                                from mlx_commander.lora.model_info import is_engine_installed
                                if not is_engine_installed("mlx_vlm"):
                                    show_error_dialog(
                                        stdscr,
                                        "Dependency Required: mlx-vlm",
                                        f"Model '{meta.name}' requires 'mlx-vlm' (Vision-Language / Multimodal model).\n\n"
                                        f"'mlx-vlm' is not currently installed in this Python environment.\n"
                                        f"Please install it to run fine-tuning:\n\n"
                                        f"  pip install \"mlx-vlm[train]\"\n\n"
                                        f"Note: mlx-vlm requires Python 3.10+.",
                                    )
                    else:
                        if not state.load_dataset(chosen):
                            if state.last_missing_dependency:
                                if show_missing_dependency_dialog(stdscr, state.last_missing_dependency):
                                    state.load_dataset(chosen)
                            else:
                                show_error_dialog(stdscr, "Dataset Loading Failed", state.status_message)
            elif state.left_focus_idx == 1 and state.loaded_dataset and state.loaded_dataset.columns:
                col_idx = state.selected_column_idx
                if 0 <= col_idx < len(state.loaded_dataset.columns):
                    c_name = state.loaded_dataset.columns[col_idx]
                    mapping_fields = state.get_mapping_fields_for_format()
                    labels = [f["label"] for f in mapping_fields]
                    chosen_label = show_choice_dialog(
                        stdscr,
                        f"Map Column '{c_name}'",
                        f"Assign '{c_name}' to MLX target field:",
                        labels,
                    )
                    if chosen_label:
                        for f in mapping_fields:
                            if f["label"] == chosen_label:
                                state.set_mapping_field(f["key"], c_name)
                                state.status_message = f"Mapped column '{c_name}' to {chosen_label}."
                                state.status_is_error = False
                                break

    # Navigation in Right Panel (Tab 2)
    elif is_right:
        num_cols = len(state.loaded_dataset.columns) if (state.loaded_dataset and state.loaded_dataset.columns) else 0
        if key in (curses.KEY_UP, curses.KEY_LEFT, ord("k"), ord("h")):
            if state.right_focus_idx > 0:
                state.right_focus_idx -= 1
            else:
                state.active_panel = ActivePanel.LEFT
                if num_cols > 0:
                    state.left_focus_idx = 1
                    state.selected_column_idx = num_cols - 1
                else:
                    state.left_focus_idx = 0
        elif key in (curses.KEY_DOWN, curses.KEY_RIGHT, ord("j"), ord("l")):
            if state.right_focus_idx < total_right_fields - 1:
                state.right_focus_idx += 1
            else:
                state.active_panel = ActivePanel.LEFT
                state.left_focus_idx = 0
        elif key in (10, 13, curses.KEY_ENTER, 32):  # Enter or Space
            idx = state.right_focus_idx
            if 0 <= idx <= 3:
                state.set_format(formats_list[idx])
            elif 4 <= idx <= 3 + len(mapping_fields):
                f_info = mapping_fields[idx - 4]
                if f_info.get("is_system_prompt"):
                    val = show_text_edit_dialog(
                        stdscr,
                        "Default System Prompt",
                        "Enter system prompt for all records (leave blank for none):",
                        state.mapping.default_system_prompt or "",
                    )
                    if val is not None:
                        state.mapping.default_system_prompt = val.strip() if val.strip() else None
                        state.update_preview()
                elif f_info.get("is_label_map"):
                    target_col = state.mapping.completion_col or state.mapping.assistant_col or "label"
                    disc_info = state.get_discrete_labels_for_target()
                    new_map = show_label_mapping_dialog(
                        stdscr,
                        target_col,
                        disc_info,
                        state.mapping.label_map,
                    )
                    state.mapping.label_map = new_map
                    state.update_preview()
                else:
                    cols = state.loaded_dataset.columns if state.loaded_dataset else []
                    btn_y = 8 + len(mapping_fields) + 3 + 1 + 1
                    safe_addstr(stdscr, btn_y, left_w + 2, " " * (right_w - 4), get_color(COLOR_PANEL_BG))
                    stdscr.refresh()
                    # Required fields must not allow selecting (None / Skip)
                    is_optional = f_info["key"] in ("system_col", "text_template")
                    chosen = show_column_picker_dialog(
                        stdscr,
                        f"Select Column for '{f_info['label']}'",
                        cols,
                        current_val=f_info["current"],
                        allow_none=is_optional,
                    )
                    state.set_mapping_field(f_info["key"], chosen)
            elif idx == 4 + len(mapping_fields):  # Train %
                val = show_text_edit_dialog(stdscr, "Train Split %", "Enter Train percentage (0-100):", f"{state.train_pct:.0f}", is_number=True)
                if val:
                    try: state.train_pct = float(val)
                    except ValueError: pass
            elif idx == 5 + len(mapping_fields):  # Valid %
                val = show_text_edit_dialog(stdscr, "Valid Split %", "Enter Valid percentage (0-100):", f"{state.valid_pct:.0f}", is_number=True)
                if val:
                    try: state.valid_pct = float(val)
                    except ValueError: pass
            elif idx == 6 + len(mapping_fields):  # Test %
                val = show_text_edit_dialog(stdscr, "Test Split %", "Enter Test percentage (0-100):", f"{state.test_pct:.0f}", is_number=True)
                if val:
                    try: state.test_pct = float(val)
                    except ValueError: pass
            elif idx == 7 + len(mapping_fields):  # Seed
                val = show_text_edit_dialog(stdscr, "Random Seed", "Enter random seed integer:", str(state.seed), is_number=True)
                if val:
                    try: state.seed = int(val)
                    except ValueError: pass
            elif idx == 8 + len(mapping_fields):  # Randomize
                state.randomize_seed()
            elif idx == 9 + len(mapping_fields):  # Output Dir
                default_out = str(state.loaded_dataset.default_output_dir) if state.loaded_dataset else state.output_dir
                chosen = show_output_destination_dialog(
                    stdscr,
                    current_output=state.output_dir,
                    default_output=default_out,
                )
                if chosen is not None:
                    state.output_dir = chosen.strip()
                    state.has_custom_output_dir = (state.output_dir != default_out)
                    state.status_message = f"Output folder set to: {state.output_dir}"
                    state.status_is_error = False
            elif idx == 10 + len(mapping_fields):  # Convert button
                res = execute_conversion(stdscr, state)
                if res is not None:
                    return res

    return None


def _handle_mode2_input(
    stdscr: curses.window,
    state: CommanderState,
    key: int,
) -> None:
    """Handle keyboard input specific to Mode 2 (LoRA Fine-Tuning & Queue)."""
    if key in (9,):  # Tab
        if state.lora_active_panel == "left":
            state.lora_active_panel = "right"
        elif state.lora_active_panel == "right":
            state.lora_active_panel = "queue"
        else:
            state.lora_active_panel = "left"
    elif key in (curses.KEY_BTAB,):  # Shift-Tab
        if state.lora_active_panel == "left":
            state.lora_active_panel = "queue"
        elif state.lora_active_panel == "queue":
            state.lora_active_panel = "right"
        else:
            state.lora_active_panel = "left"

    # Navigation in Left Panel (Setup)
    elif state.lora_active_panel == "left":
        if key in (curses.KEY_UP, ord("k")):
            if state.lora_left_focus_idx == 0:
                state.mode_switcher_focused = True
                state.mode_switcher_idx = state.active_tab
                state.status_message = "Mode Switcher: Use [←/→] to select mode, [Enter] to switch, [↓] to return to pane."
                state.status_is_error = False
            else:
                state.lora_left_focus_idx -= 1
        elif key in (curses.KEY_DOWN, ord("j")):
            if state.lora_left_focus_idx < 5:
                state.lora_left_focus_idx += 1
            else:
                state.lora_active_panel = "right"
                state.lora_right_focus_idx = 0
        elif key in (curses.KEY_RIGHT, ord("l")):
            state.lora_active_panel = "right"
        elif key in (10, 13, curses.KEY_ENTER, 32):  # Enter or Space
            idx = state.lora_left_focus_idx
            if idx == 0:  # Base Model
                chosen = show_model_picker_dialog(stdscr, state.lora_config.model)
                if chosen:
                    from mlx_commander.lora.model_info import normalize_model_path
                    norm = normalize_model_path(chosen.strip())
                    state.lora_config.model = norm
                    meta = state.inspect_current_model(force=True)
                    if meta.is_valid and meta.path:
                        state.lora_config.model = meta.path
                    state.update_deterministic_lora_name()
                    state.clear_estimates_cache()
                    state.status_message = f"Base model set to: {state.lora_config.model}"
                    state.status_is_error = False
                    if getattr(meta, "engine", "mlx_lm") == "mlx_vlm":
                        from mlx_commander.lora.model_info import is_engine_installed
                        if not is_engine_installed("mlx_vlm"):
                            show_error_dialog(
                                stdscr,
                                "Dependency Required: mlx-vlm",
                                f"Model '{meta.name}' requires 'mlx-vlm' (Vision-Language / Multimodal model).\n\n"
                                f"'mlx-vlm' is not currently installed in this Python environment.\n"
                                f"Please install it to run fine-tuning:\n\n"
                                f"  pip install \"mlx-vlm[train]\"\n\n"
                                f"Note: mlx-vlm requires Python 3.10+.",
                            )
                        elif getattr(state.lora_config, "batch_size", 1) > 1:
                            if prompt_vlm_batch_tweak(stdscr, state.lora_config):
                                state.update_deterministic_lora_name()
                                state.clear_estimates_cache()
            elif idx == 1:  # Dataset
                chosen = show_dataset_picker_dialog(stdscr, state.lora_config.data)
                if chosen:
                    from mlx_commander.lora.model_info import is_model_directory
                    chosen_p = Path(chosen.strip()).expanduser()
                    if is_model_directory(chosen_p):
                        show_error_dialog(
                            stdscr,
                            "Invalid Dataset Folder",
                            [
                                f"'{chosen_p.name}' is a Base Model directory (weights/config.json), not a training dataset.",
                                "",
                                "• To set this as your model, select 'Base Model' (Row 0) above.",
                                "• Training datasets must contain 'train.jsonl' (or convert one in Mode 1).",
                            ],
                        )
                    else:
                        state.lora_config.data = chosen.strip()
                        state.clear_estimates_cache()
                        state.status_message = f"Dataset folder set to: {chosen.strip()}"
                        state.status_is_error = False
            elif idx == 2:  # Method
                chosen = show_choice_dialog(stdscr, "Fine-Tune Method", "Select technique:", FINE_TUNE_TYPES, state.lora_config.fine_tune_type)
                if chosen:
                    state.lora_config.fine_tune_type = chosen
                    state.update_deterministic_lora_name()
            elif idx == 3:  # Optimizer
                chosen = show_choice_dialog(stdscr, "Optimizer", "Select training optimizer:", OPTIMIZERS, state.lora_config.optimizer)
                if chosen:
                    state.lora_config.optimizer = chosen
            elif idx == 4:  # Mode (toggle train/test)
                state.lora_config.train = not state.lora_config.train
            elif idx == 5:  # Run Name
                val = show_text_edit_dialog(stdscr, "Run Name", "Enter descriptive label for this run:", default_val=state.lora_config.name)
                if val:
                    state.lora_config.name = val.strip()
                    state.lora_config.is_custom_name = True
                    state.lora_config.adapter_path = f"adapters/{state.lora_config.name}"

    # Navigation in Right Panel (Hyperparameters)
    elif state.lora_active_panel == "right":
        if key in (curses.KEY_UP, ord("k")):
            if state.lora_right_focus_idx > 0:
                state.lora_right_focus_idx -= 1
            else:
                state.lora_active_panel = "left"
                state.lora_left_focus_idx = 5
        elif key in (curses.KEY_DOWN, ord("j")):
            if state.lora_right_focus_idx < 15:
                state.lora_right_focus_idx += 1
            else:
                state.lora_active_panel = "queue"
                state.selected_queue_idx = 0
        elif key in (curses.KEY_LEFT, ord("h")):
            state.lora_active_panel = "left"
        elif key in (10, 13, curses.KEY_ENTER, 32):  # Enter or Space
            idx = state.lora_right_focus_idx
            if idx == 0:  # Iters
                val = show_text_edit_dialog(stdscr, "Training Iterations", "Enter number of iterations (e.g. 1000):", str(state.lora_config.iters), is_number=True)
                if val:
                    try:
                        state.lora_config.iters = max(1, int(val))
                        state.update_deterministic_lora_name()
                    except ValueError: pass
            elif idx == 1:  # Batch Size
                val = show_text_edit_dialog(stdscr, "Batch Size", "Enter micro-batch size (e.g. 4):", str(state.lora_config.batch_size), is_number=True)
                if val:
                    try:
                        new_batch = max(1, int(val))
                        state.lora_config.batch_size = new_batch
                        if getattr(state.lora_config, "engine", "mlx_lm") == "mlx_vlm" and new_batch > 1:
                            prompt_vlm_batch_tweak(stdscr, state.lora_config)
                        state.update_deterministic_lora_name()
                        state.clear_estimates_cache()
                    except ValueError: pass
            elif idx == 2:  # Gradient Accumulation Steps
                val = show_text_edit_dialog(stdscr, "Gradient Accumulation Steps", "Enter gradient accumulation steps (e.g. 1, 4):", str(getattr(state.lora_config, "grad_accumulation_steps", 1)), is_number=True)
                if val:
                    try:
                        state.lora_config.grad_accumulation_steps = max(1, int(val))
                        state.update_deterministic_lora_name()
                    except ValueError: pass
            elif idx == 3:  # Learning Rate
                val = show_text_edit_dialog(stdscr, "Learning Rate", "Enter learning rate (e.g. 1e-5):", f"{state.lora_config.learning_rate:g}")
                if val:
                    try:
                        state.lora_config.learning_rate = float(val)
                        state.update_deterministic_lora_name()
                    except ValueError: pass
            elif idx == 4:  # LoRA Rank
                val = show_text_edit_dialog(stdscr, "LoRA Rank", "Enter rank dimension r (e.g. 8, 16):", str(state.lora_config.lora_rank), is_number=True)
                if val:
                    try:
                        state.lora_config.lora_rank = max(1, int(val))
                        state.update_deterministic_lora_name()
                    except ValueError: pass
            elif idx == 5:  # LoRA Alpha
                val = show_text_edit_dialog(stdscr, "LoRA Scale / Alpha", "Enter alpha scaling factor (e.g. 16.0):", f"{state.lora_config.lora_alpha:g}")
                if val:
                    try:
                        state.lora_config.lora_alpha = float(val)
                        state.update_deterministic_lora_name()
                    except ValueError: pass
            elif idx == 6:  # LoRA Dropout
                val = show_text_edit_dialog(stdscr, "LoRA Dropout", "Enter dropout probability (0.0 to 0.5):", f"{state.lora_config.lora_dropout:g}")
                if val:
                    try: state.lora_config.lora_dropout = float(val)
                    except ValueError: pass
            elif idx == 7:  # Max Seq Len
                val = show_text_edit_dialog(stdscr, "Max Sequence Length", "Enter context window limit (e.g. 2048):", str(state.lora_config.max_seq_length), is_number=True)
                if val:
                    try: state.lora_config.max_seq_length = max(64, int(val))
                    except ValueError: pass
            elif idx == 8:  # Num Layers
                val = show_text_edit_dialog(stdscr, "LoRA Fine-Tuned Layers", "Enter number of top layers to adapt (e.g. 16):", str(state.lora_config.num_layers), is_number=True)
                if val:
                    try: state.lora_config.num_layers = max(1, int(val))
                    except ValueError: pass
            elif idx == 9:  # Grad Checkpoint
                state.lora_config.grad_checkpoint = not state.lora_config.grad_checkpoint
            elif idx == 10:  # Mask Prompt
                state.lora_config.mask_prompt = not state.lora_config.mask_prompt
            elif idx == 11:  # Save Every
                val = show_text_edit_dialog(stdscr, "Save Every", "Save adapter checkpoint every N steps:", str(state.lora_config.save_every), is_number=True)
                if val:
                    try: state.lora_config.save_every = max(1, int(val))
                    except ValueError: pass
            elif idx == 12:  # Steps per eval
                val = show_text_edit_dialog(stdscr, 'Steps per "Eval" (validation loss)', "Evaluate on validation split every N steps:", str(state.lora_config.steps_per_eval), is_number=True)
                if val:
                    try: state.lora_config.steps_per_eval = max(1, int(val))
                    except ValueError: pass
            elif idx == 13:  # Run evals on test set (experimental)
                state.lora_config.run_eval = not getattr(state.lora_config, "run_eval", False)
                state.status_message = f"Run evals on test set (experimental): {'Yes' if state.lora_config.run_eval else 'No'}"
                state.status_is_error = False
            elif idx == 14:  # Adapter Out
                val = show_text_edit_dialog(stdscr, "Adapter Path", "Directory to store fine-tuned LoRA weights:", state.lora_config.adapter_path)
                if val:
                    state.lora_config.adapter_path = val.strip()
                    state.lora_config.is_custom_name = True
            elif idx == 15:  # Add to Queue
                if validate_lora_queue_preconditions(stdscr, state.lora_config.model, state.lora_config.data, config=state.lora_config):
                    state.update_deterministic_lora_name()
                    state.clear_estimates_cache()
                    from mlx_commander.lora.model_info import normalize_model_path
                    healed = normalize_model_path(state.lora_config.model)
                    if healed and Path(healed).exists():
                        state.lora_config.model = healed
                    added = state.add_current_lora_to_queue()
                    state.status_message = f"Added '{added.name}' to queue ({len(state.queue_manager.runs)} run(s) queued)."
                    state.status_is_error = False

    # Navigation in Queue Panel
    elif state.lora_active_panel == "queue":
        num_runs = len(state.queue_manager.runs) if state.queue_manager else 0
        if key in (curses.KEY_UP, ord("k")):
            if state.selected_queue_idx > 0:
                state.selected_queue_idx -= 1
            else:
                state.lora_active_panel = "right"
                state.lora_right_focus_idx = 15
        elif key in (curses.KEY_DOWN, ord("j")):
            if num_runs > 0 and state.selected_queue_idx < num_runs - 1:
                state.selected_queue_idx += 1
            else:
                state.lora_active_panel = "left"
                state.lora_left_focus_idx = 0
        elif key in (curses.KEY_HOME,):
            state.selected_queue_idx = 0
        elif key in (curses.KEY_END,):
            state.selected_queue_idx = max(0, num_runs - 1)
        elif key in (ord("c"), ord("C")):  # Clone
            if state.queue_manager and state.queue_manager.runs:
                curr_r = state.queue_manager.runs[state.selected_queue_idx]
                cloned = state.queue_manager.clone_run(curr_r.id)
                if cloned:
                    state.selected_queue_idx = len(state.queue_manager.runs) - 1
                    state.status_message = f"Cloned run '{cloned.name}'."
                    state.status_is_error = False
        elif key in (ord("d"), ord("D")):  # Delete
            if state.queue_manager and state.queue_manager.runs:
                curr_r = state.queue_manager.runs[state.selected_queue_idx]
                state.queue_manager.delete_run(curr_r.id)
                if state.selected_queue_idx >= len(state.queue_manager.runs):
                    state.selected_queue_idx = max(0, len(state.queue_manager.runs) - 1)
                state.status_message = f"Deleted run '{curr_r.name}'."
                state.status_is_error = False
        elif key in (ord("x"), ord("X")):  # Clear
            if state.queue_manager and state.queue_manager.runs:
                state.queue_manager.clear_queue()
                state.selected_queue_idx = 0
                state.status_message = "LoRA queue cleared."
                state.status_is_error = False
        elif key in (10, 13, curses.KEY_ENTER):  # Load run to form
            if state.queue_manager and state.queue_manager.runs:
                curr_r = state.queue_manager.runs[state.selected_queue_idx]
                state.load_queue_run_into_form(curr_r.id)
                state.status_message = f"Loaded run '{curr_r.name}' into form editor."
                state.status_is_error = False


def _handle_mode3_input(
    stdscr: curses.window,
    state: CommanderState,
    key: int,
) -> None:
    """Handle keyboard input specific to Mode 3 (Multi-Run LoRA Fine-Tuning Sweep)."""
    if key in (9,):  # Tab
        if state.multi_active_panel == "left":
            state.multi_active_panel = "right"
        elif state.multi_active_panel == "right":
            state.multi_active_panel = "sweep"
        else:
            state.multi_active_panel = "left"
    elif key in (curses.KEY_BTAB,):  # Shift-Tab
        if state.multi_active_panel == "left":
            state.multi_active_panel = "sweep"
        elif state.multi_active_panel == "sweep":
            state.multi_active_panel = "right"
        else:
            state.multi_active_panel = "left"

    # Navigation in Left Panel (Setup)
    elif state.multi_active_panel == "left":
        if key in (curses.KEY_UP, ord("k")):
            if state.multi_left_focus_idx == 0:
                state.mode_switcher_focused = True
                state.mode_switcher_idx = state.active_tab
                state.status_message = "Mode Switcher: Use [←/→] to select mode, [Enter] to switch, [↓] to return to pane."
                state.status_is_error = False
            else:
                state.multi_left_focus_idx -= 1
        elif key in (curses.KEY_DOWN, ord("j")):
            if state.multi_left_focus_idx < 5:
                state.multi_left_focus_idx += 1
            else:
                state.multi_active_panel = "right"
                state.multi_right_focus_idx = 0
        elif key in (curses.KEY_RIGHT, ord("l")):
            state.multi_active_panel = "right"
        elif key in (10, 13, curses.KEY_ENTER, 32):  # Enter or Space
            idx = state.multi_left_focus_idx
            if idx == 0:  # Base Model
                chosen = show_model_picker_dialog(stdscr, state.multi_lora_config.model)
                if chosen:
                    from mlx_commander.lora.model_info import normalize_model_path
                    norm = normalize_model_path(chosen.strip())
                    state.multi_lora_config.model = norm
                    state.lora_config.model = norm
                    meta = state.inspect_current_model(force=True)
                    if meta.is_valid and meta.path:
                        state.multi_lora_config.model = meta.path
                        state.lora_config.model = meta.path
                    state.update_deterministic_lora_name()
                    state.clear_estimates_cache()
                    state.status_message = f"Base model set to: {state.multi_lora_config.model}"
                    state.status_is_error = False
                    if getattr(meta, "engine", "mlx_lm") == "mlx_vlm":
                        from mlx_commander.lora.model_info import is_engine_installed
                        if not is_engine_installed("mlx_vlm"):
                            show_error_dialog(
                                stdscr,
                                "Dependency Required: mlx-vlm",
                                f"Model '{meta.name}' requires 'mlx-vlm' (Vision-Language / Multimodal model).\n\n"
                                f"'mlx-vlm' is not currently installed in this Python environment.\n"
                                f"Please install it to run fine-tuning:\n\n"
                                f"  pip install \"mlx-vlm[train]\"\n\n"
                                f"Note: mlx-vlm requires Python 3.10+.",
                            )
                        elif any(int(b) > 1 for b in state.multi_lora_config.batch_size):
                            if prompt_vlm_sweep_batch_tweak(stdscr, state.multi_lora_config):
                                state.update_deterministic_lora_name()
                                state.clear_estimates_cache()
            elif idx == 1:  # Dataset
                chosen = show_dataset_picker_dialog(stdscr, state.multi_lora_config.data)
                if chosen:
                    from mlx_commander.lora.model_info import is_model_directory
                    chosen_p = Path(chosen.strip()).expanduser()
                    if is_model_directory(chosen_p):
                        show_error_dialog(
                            stdscr,
                            "Invalid Dataset Folder",
                            [
                                f"'{chosen_p.name}' is a Base Model directory (weights/config.json), not a training dataset.",
                                "",
                                "• To set this as your model, select 'Base Model' (Row 0) above.",
                                "• Training datasets must contain 'train.jsonl' (or convert one in Mode 1).",
                            ],
                        )
                    else:
                        state.multi_lora_config.data = chosen.strip()
                        state.lora_config.data = chosen.strip()
                        state.clear_estimates_cache()
                        state.status_message = f"Dataset folder set to: {chosen.strip()}"
                        state.status_is_error = False
            elif idx == 2:  # Method
                chosen = show_choice_dialog(stdscr, "Fine-Tune Method", "Select technique:", FINE_TUNE_TYPES, state.multi_lora_config.fine_tune_type)
                if chosen:
                    state.multi_lora_config.fine_tune_type = chosen
                    state.lora_config.fine_tune_type = chosen
            elif idx == 3:  # Optimizer
                chosen = show_choice_dialog(stdscr, "Optimizer", "Select training optimizer:", OPTIMIZERS, state.multi_lora_config.optimizer)
                if chosen:
                    state.multi_lora_config.optimizer = chosen
                    state.lora_config.optimizer = chosen
            elif idx == 4:  # Mode (toggle train/test)
                state.multi_lora_config.train = not state.multi_lora_config.train
            elif idx == 5:  # Adapters dir
                val = show_text_edit_dialog(stdscr, "Adapters Directory", "Base directory for saved adapter weights:", default_val=state.multi_lora_config.adapter_path)
                if val:
                    state.multi_lora_config.adapter_path = val.strip()

    # Navigation in Right Panel (Multi-Run Hyperparameters)
    elif state.multi_active_panel == "right":
        if key in (curses.KEY_UP, ord("k")):
            if state.multi_right_focus_idx > 0:
                state.multi_right_focus_idx -= 1
            else:
                state.multi_active_panel = "left"
                state.multi_left_focus_idx = 5
        elif key in (curses.KEY_DOWN, ord("j")):
            if state.multi_right_focus_idx < 14:
                state.multi_right_focus_idx += 1
            else:
                state.multi_active_panel = "sweep"
        elif key in (curses.KEY_LEFT, ord("h")):
            state.multi_active_panel = "left"
        elif key in (10, 13, curses.KEY_ENTER, 32):  # Enter or Space
            idx = state.multi_right_focus_idx
            if idx < 14:
                field_name, field_label, field_type = SWEEP_FIELD_DEFS[idx]
                curr_vals = state.multi_lora_config.get_field_values(field_name)
                chosen_vals = show_multi_value_edit_dialog(
                    stdscr,
                    title=f"Edit {field_label}",
                    prompt=f"Enter conditions for {field_label}:",
                    current_values=curr_vals,
                    val_type=field_type,
                )
                if chosen_vals:
                    state.multi_lora_config.set_field_values(field_name, chosen_vals)
                    if field_name == "batch_size" and getattr(state.multi_lora_config, "engine", "mlx_lm") == "mlx_vlm":
                        if any(int(b) > 1 for b in chosen_vals):
                            prompt_vlm_sweep_batch_tweak(stdscr, state.multi_lora_config)
                    state.status_message = f"Updated {field_label} conditions: {state.multi_lora_config.get_field_values(field_name)}"
                    state.status_is_error = False
            elif idx == 14:  # Add sweep to queue
                if validate_lora_queue_preconditions(stdscr, state.multi_lora_config.model, state.multi_lora_config.data, multi_config=state.multi_lora_config):
                    from mlx_commander.lora.model_info import normalize_model_path
                    healed = normalize_model_path(state.multi_lora_config.model)
                    if healed and Path(healed).exists():
                        state.multi_lora_config.model = healed
                    added = state.add_multi_lora_runs_to_queue()
                    state.status_message = f"Added {len(added)} sweep run(s) to queue ({len(state.queue_manager.runs)} total queued)."
                    state.status_is_error = False

    # Navigation in Bottom Sweep Panel
    elif state.multi_active_panel == "sweep":
        if key in (curses.KEY_UP, ord("k")):
            state.multi_active_panel = "right"
            state.multi_right_focus_idx = 14
        elif key in (curses.KEY_DOWN, ord("j")):
            state.multi_active_panel = "left"
            state.multi_left_focus_idx = 0
        elif key in (10, 13, curses.KEY_ENTER, 32):  # Add sweep to queue
            if validate_lora_queue_preconditions(stdscr, state.multi_lora_config.model, state.multi_lora_config.data, multi_config=state.multi_lora_config):
                from mlx_commander.lora.model_info import normalize_model_path
                healed = normalize_model_path(state.multi_lora_config.model)
                if healed and Path(healed).exists():
                    state.multi_lora_config.model = healed
                added = state.add_multi_lora_runs_to_queue()
                state.status_message = f"Added {len(added)} sweep run(s) to queue ({len(state.queue_manager.runs)} total queued)."
                state.status_is_error = False


def run_commander_tui(
    stdscr: curses.window,
    default_dataset_path: Optional[str] = None,
    initial_state: Optional[CommanderState] = None,
    prefill: Optional[Dict[str, Any]] = None,
) -> Optional[ConversionResult]:
    """Main event loop for the persistent MLX Commander dashboard."""
    configure_escdelay(25)
    state = initial_state if initial_state is not None else CommanderState()

    if prefill:
        state.apply_prefill(prefill)

    init_colors(state.theme_mode)
    if ThemeMode.is_norton(state.theme_mode):
        try:
            stdscr.bkgd(" ", get_color(COLOR_PANEL_BG))
        except curses.error:
            pass

    curses.curs_set(0)
    stdscr.keypad(True)

    target_dataset_path = default_dataset_path or state.dataset_path
    if not state.loaded_dataset and target_dataset_path:
        try:
            p = Path(target_dataset_path).resolve()
            if p.exists():
                loaded = state.load_dataset(str(p))
                if not loaded and state.last_missing_dependency:
                    if show_missing_dependency_dialog(stdscr, state.last_missing_dependency):
                        state.load_dataset(str(p))
            else:
                state.dataset_path = str(p)
                if not state.has_custom_output_dir:
                    base = p if p.is_dir() else p.parent
                    state.output_dir = str(base / "mlx_dataset")
        except Exception:
            state.dataset_path = target_dataset_path
            if not state.has_custom_output_dir:
                try:
                    p = Path(target_dataset_path).resolve()
                    base = p if p.is_dir() else p.parent
                    state.output_dir = str(base / "mlx_dataset")
                except Exception:
                    state.output_dir = str(Path.cwd() / "mlx_dataset")
    elif not state.loaded_dataset:
        if not state.has_custom_output_dir and not state.output_dir:
            state.output_dir = str(Path.cwd() / "mlx_dataset")

    conversion_result: Optional[ConversionResult] = None
    formats_list = [
        MLXFormat.PROMPT_COMPLETION,
        MLXFormat.CHAT,
        MLXFormat.TEXT,
        MLXFormat.DPO,
    ]

    try:
        cur_y, cur_x = stdscr.getmaxyx()
        if cur_y < 38 or cur_x < 120:
            ensure_adequate_terminal_size(min_cols=120, min_lines=38)
    except Exception:
        pass

    while True:
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()

        # Check for minimum terminal dimension
        if max_y < 16 or max_x < 70:
            ensure_adequate_terminal_size(min_cols=120, min_lines=38)
            max_y, max_x = stdscr.getmaxyx()
            if max_y < 16 or max_x < 70:
                safe_addstr(stdscr, 1, 2, "Terminal window too small for MLX Commander.", curses.A_BOLD)
                safe_addstr(stdscr, 2, 2, f"Current: {max_x}x{max_y} (Minimum required: 70x16)", curses.A_DIM)
                safe_addstr(stdscr, 4, 2, "Please resize your terminal window or press [q] to exit.", curses.A_DIM)
                stdscr.refresh()
                k = stdscr.getch()
                if k in (ord("q"), ord("Q"), 27):
                    return None
                continue

        # ----------------------------------------------------
        # 1. Header Banner with Mode Switcher Tabs
        # ----------------------------------------------------
        hdr_attr = (get_color(COLOR_BANNER) | curses.A_BOLD) if curses.has_colors() else curses.A_STANDOUT
        safe_addstr(stdscr, 0, 0, " " * max_x, hdr_attr)

        title_prefix = "MLX Commander  ::  "
        safe_addstr(stdscr, 0, 2, title_prefix, hdr_attr)
        x_m1 = 2 + len(title_prefix)

        mode1_title = "[ 1: Dataset Converter ]"
        mode2_title = "[ 2: Single Run ]"
        mode3_title = "[ 3: Multi-Run Matrix ]"
        if state.mode_switcher_focused:
            focused_attr = (get_color(COLOR_INPUT_FOCUSED) | curses.A_BOLD) if curses.has_colors() else (curses.A_STANDOUT | curses.A_BOLD)
            active_attr = hdr_attr | curses.A_STANDOUT | curses.A_BOLD
            dim_attr = hdr_attr | curses.A_DIM

            m1_attr = focused_attr if state.mode_switcher_idx == 0 else (active_attr if state.active_tab == 0 else dim_attr)
            m2_attr = focused_attr if state.mode_switcher_idx == 1 else (active_attr if state.active_tab == 1 else dim_attr)
            m3_attr = focused_attr if state.mode_switcher_idx == 2 else (active_attr if state.active_tab == 2 else dim_attr)
        else:
            m1_attr = (hdr_attr | curses.A_STANDOUT | curses.A_BOLD) if state.active_tab == 0 else (hdr_attr | curses.A_DIM)
            m2_attr = (hdr_attr | curses.A_STANDOUT | curses.A_BOLD) if state.active_tab == 1 else (hdr_attr | curses.A_DIM)
            m3_attr = (hdr_attr | curses.A_STANDOUT | curses.A_BOLD) if state.active_tab == 2 else (hdr_attr | curses.A_DIM)

        safe_addstr(stdscr, 0, x_m1, mode1_title, m1_attr)
        x_m2 = x_m1 + len(mode1_title) + 2
        safe_addstr(stdscr, 0, x_m2, mode2_title, m2_attr)
        x_m3 = x_m2 + len(mode2_title) + 2
        safe_addstr(stdscr, 0, x_m3, mode3_title, m3_attr)

        if state.mode_switcher_focused:
            hint = "Navigate: [←/→]  Confirm: [Enter]  Return: [↓]"
            if max_x >= x_m3 + len(mode3_title) + len(hint) + 3:
                safe_addstr(stdscr, 0, max_x - len(hint) - 2, hint, (get_color(COLOR_TITLE_ACCENT) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)

        # ----------------------------------------------------
        # 2. Dimensions & Coordinates (3-Tier Responsive Layout)
        # ----------------------------------------------------
        left_w = max(34, max_x // 2)
        right_w = max_x - left_w

        if state.active_tab == 0:
            _draw_mode1_dashboard(stdscr, state, max_y, max_x, left_w, right_w)
        elif state.active_tab == 1:
            _draw_mode2_dashboard(stdscr, state, max_y, max_x, left_w, right_w)
        else:
            _draw_mode3_dashboard(stdscr, state, max_y, max_x, left_w, right_w)

        # ----------------------------------------------------
        # 3. Bottom Status / Hotkey Bar
        # ----------------------------------------------------
        footer_y = max_y - 1
        if ThemeMode.is_norton(state.theme_mode):
            if state.active_tab == 0:
                fn_items = [
                    ("1", "Help"),
                    ("2", "Mode"),
                    ("3", "Output"),
                    ("5", "Convert"),
                    ("F9", "Theme"),
                    ("10", "Exit"),
                ]
            elif state.active_tab == 1:
                fn_items = [
                    ("1", "Help"),
                    ("2", "Mode"),
                    ("5", "RunQueue"),
                    ("6", "AddRun"),
                    ("F9", "Theme"),
                    ("10", "Exit"),
                ]
            else:
                fn_items = [
                    ("1", "Help"),
                    ("2", "Mode"),
                    ("5", "RunSweep"),
                    ("6", "AddSweep"),
                    ("F9", "Theme"),
                    ("10", "Exit"),
                ]
            safe_addstr(stdscr, footer_y, 0, " " * max_x, get_color(COLOR_PANEL_BG))
            total_fn_width = sum(len(num_str) + len(lbl_str) + 3 for num_str, lbl_str in fn_items)

            cur_x = 0
            avail_status = max_x - total_fn_width - 2
            if max_x >= 90 and state.status_message and avail_status >= 10:
                prefix = "[OK] " if not state.status_is_error else "[ERR] "
                short_status = f"{prefix}{state.status_message}"[:avail_status]
                stat_attr = (get_color(COLOR_TITLE_ACCENT) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD
                safe_addstr(stdscr, footer_y, 0, short_status, stat_attr)
                cur_x = len(short_status) + 1

            start_fn_x = max(cur_x, max_x - total_fn_width)
            if start_fn_x + total_fn_width > max_x:
                start_fn_x = max(0, max_x - total_fn_width)

            pos_x = start_fn_x
            for num_str, lbl_str in fn_items:
                if pos_x + len(num_str) + len(lbl_str) + 2 >= max_x:
                    break
                num_attr = (get_color(COLOR_FN_NUMBER) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD
                lbl_attr = get_color(COLOR_FN_LABEL) if curses.has_colors() else curses.A_STANDOUT
                safe_addstr(stdscr, footer_y, pos_x, f" {num_str} ", num_attr)
                pos_x += len(num_str) + 2
                safe_addstr(stdscr, footer_y, pos_x, f"{lbl_str} ", lbl_attr)
                pos_x += len(lbl_str) + 1
        else:
            safe_addstr(stdscr, footer_y, 0, " " * max_x, hdr_attr)
            unbold_attr = get_color(COLOR_BANNER) if curses.has_colors() else curses.A_NORMAL

            if state.active_tab == 0:
                if max_x >= 120:
                    bar_shortcuts = "[Tab] Switch  [Enter] Select  [F1] Help  [F2] Mode  [F3] Output  [F5] Convert  [F9] Theme  [F10] Exit"
                elif max_x >= 100:
                    bar_shortcuts = "[Tab] Switch  [F1] Help  [F2] Mode  [F3] Output  [F5] Convert  [F9] Theme  [F10] Exit"
                elif max_x >= 80:
                    bar_shortcuts = "[Tab] Switch  [F1] Help  [F2] Mode  [F5] Convert  [F9] Theme  [F10] Exit"
                elif max_x >= 65:
                    bar_shortcuts = "[F1] Help  [F2] Mode  [F5] Convert  [F9] Theme  [F10] Exit"
                elif max_x >= 57:
                    bar_shortcuts = "[F1] Help [F2] Mode [F5] Convert [F9] Theme [F10] Exit"
                else:
                    bar_shortcuts = "F1:Help F2:Mode F9:Theme F10:Exit"
            elif state.active_tab == 1:
                if max_x >= 120:
                    bar_shortcuts = "[Tab] Switch  [Enter] Select  [c] Clone  [d] Del  [F1] Help  [F2] Mode  [F5] Run  [F6] Add  [F9] Theme  [F10] Exit"
                elif max_x >= 102:
                    bar_shortcuts = "[Tab] Switch  [c] Clone  [F1] Help  [F2] Mode  [F5] Run  [F6] Add  [F9] Theme  [F10] Exit"
                elif max_x >= 85:
                    bar_shortcuts = "[Tab] Switch  [F1] Help  [F2] Mode  [F5] Run  [F6] Add  [F9] Theme  [F10] Exit"
                elif max_x >= 76:
                    bar_shortcuts = "[Tab] Switch [F1] Help [F2] Mode [F5] Run [F6] Add [F9] Theme [F10] Exit"
                elif max_x >= 60:
                    bar_shortcuts = "[F1] Help  [F2] Mode  [F5] Run  [F9] Theme  [F10] Exit"
                elif max_x >= 53:
                    bar_shortcuts = "[F1] Help [F2] Mode [F5] Run [F9] Theme [F10] Exit"
                else:
                    bar_shortcuts = "F1:Help F2:Mode F9:Theme F10:Exit"
            else:
                if max_x >= 120:
                    bar_shortcuts = "[Tab] Switch  [Enter] Amend  [F1] Help  [F2] Mode  [F5] Run Sweep  [F6] Add Sweep  [F9] Theme  [F10] Exit"
                elif max_x >= 95:
                    bar_shortcuts = "[Tab] Switch  [F1] Help  [F2] Mode  [F5] Run Sweep  [F6] Add Sweep  [F9] Theme  [F10] Exit"
                elif max_x >= 75:
                    bar_shortcuts = "[Tab] Switch [F2] Mode [F5] Run [F6] Add [F9] Theme [F10] Exit"
                else:
                    bar_shortcuts = "F1:Help F2:Mode F5:Run F6:Add F10:Exit"

            shortcuts_len = len(bar_shortcuts)
            shortcuts_x = max(1, max_x - shortcuts_len - 1)
            avail_status = max(0, shortcuts_x - 3)

            if avail_status >= 6 and state.status_message:
                status_prefix = "[OK] " if not state.status_is_error else "[ERR] "
                status_text = f"{status_prefix}{state.status_message}"[:avail_status]
                safe_addstr(stdscr, footer_y, 2, status_text, hdr_attr | curses.A_BOLD)

            safe_addstr(stdscr, footer_y, shortcuts_x, bar_shortcuts, unbold_attr)

        stdscr.refresh()

        # ----------------------------------------------------
        # 4. Input & Key Event Handling
        # ----------------------------------------------------
        key = stdscr.getch()

        if key == curses.KEY_RESIZE:
            stdscr.clear()
            continue

        if key == 27:  # ESC
            if state.mode_switcher_focused:
                state.mode_switcher_focused = False
                state.mode_switcher_idx = state.active_tab
                if state.active_tab == 0:
                    state.active_panel = ActivePanel.LEFT
                    state.left_focus_idx = 0
                elif state.active_tab == 1:
                    state.lora_active_panel = "left"
                    state.lora_left_focus_idx = 0
                else:
                    state.multi_active_panel = "left"
                    state.multi_left_focus_idx = 0
                continue
            break

        elif key in (ord("q"), ord("Q"), curses.KEY_F10):
            break

        try:
            if key in (ord("?"), curses.KEY_F1):
                show_help_dialog(stdscr)

            elif key in (curses.KEY_F2, 20):  # F2 or Ctrl+T (Mode switcher)
                state.mode_switcher_focused = False
                state.switch_mode((state.active_tab + 1) % 3)
                state.mode_switcher_idx = state.active_tab

            elif state.mode_switcher_focused:
                if key in (curses.KEY_LEFT, ord("h")):
                    state.mode_switcher_idx = max(0, state.mode_switcher_idx - 1)
                elif key in (curses.KEY_RIGHT, ord("l")):
                    state.mode_switcher_idx = min(2, state.mode_switcher_idx + 1)
                elif key in (9,):  # Tab cycles between modes
                    state.mode_switcher_idx = (state.mode_switcher_idx + 1) % 3
                elif key in (curses.KEY_DOWN, ord("j")):  # Down: return to top of left-hand pane
                    state.mode_switcher_focused = False
                    state.mode_switcher_idx = state.active_tab
                    if state.active_tab == 0:
                        state.active_panel = ActivePanel.LEFT
                        state.left_focus_idx = 0
                    elif state.active_tab == 1:
                        state.lora_active_panel = "left"
                        state.lora_left_focus_idx = 0
                    else:
                        state.multi_active_panel = "left"
                        state.multi_left_focus_idx = 0
                elif key in (10, 13, curses.KEY_ENTER, 32):  # Enter: confirm switching modes
                    state.switch_mode(state.mode_switcher_idx)
                    state.mode_switcher_focused = False
                    if state.active_tab == 0:
                        state.active_panel = ActivePanel.LEFT
                        state.left_focus_idx = 0
                    elif state.active_tab == 1:
                        state.lora_active_panel = "left"
                        state.lora_left_focus_idx = 0
                    else:
                        state.multi_active_panel = "left"
                        state.multi_left_focus_idx = 0

            elif key == curses.KEY_F6:
                if state.active_tab == 2:
                    if validate_lora_queue_preconditions(stdscr, state.multi_lora_config.model, state.multi_lora_config.data, multi_config=state.multi_lora_config):
                        from mlx_commander.lora.model_info import normalize_model_path
                        healed = normalize_model_path(state.multi_lora_config.model)
                        if healed and Path(healed).exists():
                            state.multi_lora_config.model = healed
                        added = state.add_multi_lora_runs_to_queue()
                        state.status_message = f"Added {len(added)} sweep run(s) to queue ({len(state.queue_manager.runs)} total queued)."
                        state.status_is_error = False
                elif state.active_tab == 1:
                    if validate_lora_queue_preconditions(stdscr, state.lora_config.model, state.lora_config.data, config=state.lora_config):
                        state.update_deterministic_lora_name()
                        state.clear_estimates_cache()
                        from mlx_commander.lora.model_info import normalize_model_path
                        healed = normalize_model_path(state.lora_config.model)
                        if healed and Path(healed).exists():
                            state.lora_config.model = healed
                        added = state.add_current_lora_to_queue()
                        state.status_message = f"Added '{added.name}' to queue ({len(state.queue_manager.runs)} run(s) queued)."
                        state.status_is_error = False

            elif key == curses.KEY_F5:
                if state.active_tab == 0:
                    res = execute_conversion(stdscr, state)
                    if res is not None:
                        conversion_result = res
                elif state.active_tab == 1:
                    execute_lora_queue_action(stdscr, state)
                else:
                    if not state.queue_manager or not state.queue_manager.get_pending_runs():
                        if not validate_lora_queue_preconditions(stdscr, state.multi_lora_config.model, state.multi_lora_config.data, multi_config=state.multi_lora_config):
                            continue
                        from mlx_commander.lora.model_info import normalize_model_path
                        healed = normalize_model_path(state.multi_lora_config.model)
                        if healed and Path(healed).exists():
                            state.multi_lora_config.model = healed
                        state.add_multi_lora_runs_to_queue()
                    execute_lora_queue_action(stdscr, state)

            elif key == curses.KEY_F9:
                state.toggle_theme()
                init_colors(state.theme_mode)
                if ThemeMode.is_norton(state.theme_mode):
                    try:
                        stdscr.bkgd(" ", get_color(COLOR_PANEL_BG))
                    except curses.error:
                        pass
                    state.status_message = "Theme: Norton Commander (Classic Blue)"
                else:
                    try:
                        stdscr.bkgd(" ", curses.color_pair(0))
                    except curses.error:
                        pass
                    state.status_message = "Theme: Modern Terminal"
                state.status_is_error = False
                try:
                    stdscr.touchwin()
                except curses.error:
                    pass
                stdscr.clear()

            elif state.active_tab == 0:
                res = _handle_mode1_input(stdscr, state, key, formats_list)
                if res is not None:
                    conversion_result = res
            elif state.active_tab == 1:
                _handle_mode2_input(stdscr, state, key)
            else:
                _handle_mode3_input(stdscr, state, key)

        except curses.error:
            pass
        except Exception as e:
            state.status_message = f"Error: {e}"
            state.status_is_error = True
            show_error_dialog(stdscr, "Error Encountered", str(e))

    return conversion_result


def launch_tui(
    default_dataset_path: Optional[str] = None,
    initial_state: Optional[CommanderState] = None,
    prefill: Optional[Dict[str, Any]] = None,
) -> Optional[ConversionResult]:
    """Launch the MLX Commander full-screen curses dashboard with optional prefill."""
    ensure_adequate_terminal_size(min_cols=120, min_lines=38)
    configure_escdelay(25)
    try:
        return curses.wrapper(run_commander_tui, default_dataset_path, initial_state, prefill)
    except KeyboardInterrupt:
        return None
