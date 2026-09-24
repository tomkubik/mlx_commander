"""
Curses TUI Widgets and UI Helpers.
Includes menus, text input fields, panels, and styled dialogs.
"""

import curses
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Semantic color pair constants
COLOR_BANNER = 1         # Top Header Banner
COLOR_BORDER_JOINTS = 2  # Borders, Frame, and Routing Joints (Highlight / Cyan #00AAAA)
COLOR_SUCCESS = 3        # Success, Checks
COLOR_NORMAL_TEXT = 4    # Main Text / White #FFFFFF (Standard file names and UI text)
COLOR_ERROR = 5          # Error, Alerts
COLOR_TITLE_ACCENT = 6   # Panel Titles / Accents (Cursor / Yellow #FFFF55)
COLOR_LABEL_GRAY = 7     # Label / Light Gray #AAAAAA (Inactive elements, background text)
COLOR_INPUT_NORMAL = 8   # User Inputs / Inactive Fields (Ice Blue #55FFFF on Background Blue #0000AA)
COLOR_INPUT_FOCUSED = 9  # Active / Focused Field (Prompt / Black on Highlight / Cyan)
COLOR_FN_NUMBER = 10     # Norton Hotkey Bar: Key Number (White on Black)
COLOR_FN_LABEL = 11      # Norton Hotkey Bar: Key Label (Black on Cyan)
COLOR_PANEL_BG = 12      # Background Blue Fill #0000AA (Classic VGA Blue)


def safe_addstr(win: curses.window, y: int, x: int, text: str, attr: int = 0) -> None:
    """Safely print text inside a window without raising on border clip."""
    max_y, max_x = win.getmaxyx()
    if y < 0 or y >= max_y or x < 0 or x >= max_x:
        return
    avail = max_x - x
    if avail <= 0:
        return
    to_print = text[:avail]
    try:
        win.addstr(y, x, to_print, attr)
    except curses.error:
        pass


def safe_has_colors() -> bool:
    """Safely check if curses has colors initialized."""
    try:
        return curses.has_colors()
    except curses.error:
        return False


def get_color(pair_idx: int) -> int:
    """Safely return color pair attribute if colors are available."""
    try:
        if safe_has_colors():
            return curses.color_pair(pair_idx)
    except curses.error:
        pass
    return 0


def safe_curs_set(visibility: int) -> None:
    """Safely change cursor visibility without raising on uninitialized curses."""
    try:
        curses.curs_set(visibility)
    except curses.error:
        pass


def configure_escdelay(delay_ms: int = 25) -> None:
    """
    Configure ncurses ESC key delay to make exiting/canceling dialogs instantaneous.
    By default, ncurses waits 1000ms (1 full second) on an ESC keypress to disambiguate
    a standalone ESC from multi-byte escape sequences (such as arrow keys).
    Setting this to 25ms makes ESC exit instantaneous while still allowing
    arrow and function keys to be cleanly processed.
    """
    os.environ["ESCDELAY"] = str(delay_ms)
    try:
        curses.set_escdelay(delay_ms)
    except (AttributeError, curses.error):
        pass


def draw_header(stdscr: curses.window, title: str, step_info: str) -> None:
    """Draw top header banner with step indicator."""
    max_y, max_x = stdscr.getmaxyx()
    header_attr = get_color(1) | curses.A_BOLD if safe_has_colors() else curses.A_STANDOUT
    sub_attr = get_color(2) if safe_has_colors() else 0

    stdscr.attron(header_attr)
    safe_addstr(stdscr, 0, 0, " " * max_x, header_attr)
    safe_addstr(stdscr, 0, 2, f"MLX Commander  ::  {title}", header_attr)
    stdscr.attroff(header_attr)

    if step_info:
        safe_addstr(stdscr, 1, 2, f"Step: {step_info}", sub_attr | curses.A_DIM)


def draw_footer(stdscr: curses.window, shortcuts: str) -> None:
    """Draw bottom footer bar with available keyboard shortcuts."""
    max_y, max_x = stdscr.getmaxyx()
    footer_attr = get_color(1) if safe_has_colors() else curses.A_STANDOUT
    y = max_y - 1
    safe_addstr(stdscr, y, 0, " " * max_x, footer_attr)
    safe_addstr(stdscr, y, 2, shortcuts, footer_attr)


def run_menu(
    stdscr: curses.window,
    title: str,
    subtitle: str,
    options: List[Tuple[str, str]],
    default_idx: int = 0,
) -> Optional[int]:
    """
    Display a navigable menu of options with [Title, Description].
    Returns selected index or None if canceled (ESC or 'q').
    """
    configure_escdelay(25)
    safe_curs_set(0)
    current_idx = default_idx
    max_idx = len(options) - 1

    while True:
        stdscr.erase()
        draw_header(stdscr, title, subtitle)
        max_y, max_x = stdscr.getmaxyx()

        y_start = 3
        safe_addstr(stdscr, y_start, 2, "Use [↑/↓] to Navigate, [Enter] to Select, [q] to Quit", curses.A_DIM)

        list_y = y_start + 2
        for idx, (label, desc) in enumerate(options):
            if list_y + idx * 2 >= max_y - 2:
                break
            is_selected = (idx == current_idx)
            indicator = " ▶ " if is_selected else "   "
            line = f"{indicator}{label}"

            if is_selected:
                attr = (get_color(3) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
            else:
                attr = get_color(4) if safe_has_colors() else 0

            safe_addstr(stdscr, list_y + idx * 2, 2, line, attr)
            if desc:
                safe_addstr(stdscr, list_y + idx * 2 + 1, 6, desc, curses.A_DIM)

        draw_footer(stdscr, "[↑/↓] Navigate  |  [Enter] Select  |  [Esc/q] Cancel")
        stdscr.refresh()

        key = stdscr.getch()
        if key in (curses.KEY_UP, curses.KEY_LEFT, ord("k"), ord("h")):
            current_idx = max(0, current_idx - 1)
        elif key in (curses.KEY_DOWN, curses.KEY_RIGHT, ord("j"), ord("l")):
            current_idx = min(max_idx, current_idx + 1)
        elif key in (10, 13, curses.KEY_ENTER):
            return current_idx
        elif key in (27, ord("q"), ord("Q")):
            return None


def run_text_input(
    stdscr: curses.window,
    title: str,
    prompt: str,
    default_value: str = "",
    help_text: str = "",
    history: Optional[List[str]] = None,
    gui_picker_type: Optional[str] = None,  # "dataset" or "folder"
) -> Optional[str]:
    """
    Display a text input screen with cursor editing, auto-completion, and default value.
    Supports triggering a native GUI file/folder picker via [Ctrl+O].
    """
    from mlx_commander.gui_picker import pick_dataset_gui, pick_folder_gui

    configure_escdelay(25)
    safe_curs_set(1)
    text = list(default_value)
    cursor_pos = len(text)

    while True:
        stdscr.erase()
        draw_header(stdscr, title, prompt)
        max_y, max_x = stdscr.getmaxyx()

        safe_addstr(stdscr, 3, 2, prompt, curses.A_BOLD)
        if help_text:
            safe_addstr(stdscr, 4, 2, help_text, curses.A_DIM)

        # Draw input box
        box_y = 6
        box_width = min(max_x - 6, 80)
        safe_addstr(stdscr, box_y - 1, 2, "┌" + "─" * box_width + "┐", curses.A_DIM)
        safe_addstr(stdscr, box_y, 2, "│" + " " * box_width + "│", curses.A_DIM)
        safe_addstr(stdscr, box_y + 1, 2, "└" + "─" * box_width + "┘", curses.A_DIM)

        # Draw text inside box
        current_str = "".join(text)
        visible_text = current_str
        if len(visible_text) > box_width - 4:
            visible_text = visible_text[-(box_width - 4):]

        safe_addstr(stdscr, box_y, 4, current_str[:box_width - 4], get_color(2) if safe_has_colors() else 0)

        # Draw hints / autocomplete suggestions if path-like
        if "/" in current_str or "~" in current_str or "." in current_str:
            expanded = Path(current_str).expanduser()
            parent = expanded.parent if not current_str.endswith("/") else expanded
            if parent.exists() and parent.is_dir():
                try:
                    candidates = [p.name for p in parent.iterdir() if p.name.startswith(expanded.name or "")][:4]
                    if candidates:
                        safe_addstr(stdscr, box_y + 3, 2, "Suggestions: " + ", ".join(candidates), curses.A_DIM)
                except Exception:
                    pass

        footer_text = "[Enter] Confirm  |  [Tab] Auto-complete  |  [Esc] Cancel"
        if gui_picker_type:
            footer_text = "[Enter] Confirm  |  [Ctrl+O] Open Finder GUI  |  [Tab] Complete  |  [Esc] Cancel"
        draw_footer(stdscr, footer_text)

        # Position cursor
        cursor_x = min(4 + cursor_pos, box_width)
        stdscr.move(box_y, cursor_x)
        stdscr.refresh()

        key = stdscr.getch()
        if key in (10, 13, curses.KEY_ENTER):
            return "".join(text)
        elif key == 27:  # ESC
            return None
        elif gui_picker_type and key in (15, 6):  # Ctrl+O or Ctrl+F
            curses.def_prog_mode()
            curses.endwin()
            curr_path = "".join(text).strip() or os.getcwd()
            from mlx_commander.gui_picker import pick_dataset_gui, pick_folder_gui
            if gui_picker_type == "folder":
                chosen = pick_folder_gui("Select Destination Folder", default_dir=curr_path)
            else:
                chosen = pick_dataset_gui(default_dir=curr_path)
            curses.reset_prog_mode()
            stdscr.refresh()
            if chosen:
                text = list(chosen)
                cursor_pos = len(text)
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            if cursor_pos > 0:
                text.pop(cursor_pos - 1)
                cursor_pos -= 1
        elif key == curses.KEY_DC:  # Delete
            if cursor_pos < len(text):
                text.pop(cursor_pos)
        elif key == curses.KEY_LEFT:
            cursor_pos = max(0, cursor_pos - 1)
        elif key == curses.KEY_RIGHT:
            cursor_pos = min(len(text), cursor_pos + 1)
        elif key == curses.KEY_HOME or key == 1:  # Ctrl+A
            cursor_pos = 0
        elif key == curses.KEY_END or key == 5:  # Ctrl+E
            cursor_pos = len(text)
        elif key == 9:  # TAB: simple path completion
            current_str = "".join(text)
            expanded = Path(current_str).expanduser()
            parent = expanded.parent if not current_str.endswith("/") else expanded
            if parent.exists() and parent.is_dir():
                try:
                    prefix = "" if current_str.endswith("/") else expanded.name
                    matches = [p for p in parent.iterdir() if p.name.startswith(prefix)]
                    if len(matches) == 1:
                        completed = str(matches[0]) + ("/" if matches[0].is_dir() else "")
                        text = list(completed)
                        cursor_pos = len(text)
                except Exception:
                    pass
        elif 32 <= key <= 126:
            text.insert(cursor_pos, chr(key))
            cursor_pos += 1


def show_error_dialog(
    stdscr: curses.window,
    title: str,
    message: Any,
) -> None:
    """
    Display a centered modal error dialog overlay on top of MLX Commander.
    Preserves the background dashboard without erasing the screen.
    """
    import textwrap
    configure_escdelay(25)
    safe_curs_set(0)

    max_y, max_x = stdscr.getmaxyx()

    if isinstance(message, str):
        raw_lines = [l for l in message.split("\n")]
    elif isinstance(message, (list, tuple)):
        raw_lines = [str(m) for m in message]
    else:
        raw_lines = [str(message)]

    w = min(max_x - 6, 76)
    w = max(40, w)
    inner_w = w - 6

    wrapped_lines: List[str] = []
    for line in raw_lines:
        if not line.strip():
            wrapped_lines.append("")
            continue
        wrapped = textwrap.wrap(line, width=inner_w, break_long_words=True)
        wrapped_lines.extend(wrapped if wrapped else [""])

    max_text_lines = max(1, min(len(wrapped_lines), max_y - 8))
    h = max_text_lines + 5

    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    border_attr = (get_color(5) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
    title_attr = (get_color(5) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    text_attr = (get_color(4) | curses.A_BOLD) if safe_has_colors() else 0

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", border_attr)
        title_str = f" [!] {title} "[: w - 4]
        safe_addstr(stdscr, start_y, start_x + 2, title_str, title_attr)

        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", border_attr)

        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", border_attr)

        for idx in range(max_text_lines):
            safe_addstr(stdscr, start_y + 2 + idx, start_x + 3, wrapped_lines[idx][:inner_w], text_attr)

        hint = "[Enter] OK   [Esc] Dismiss"
        safe_addstr(stdscr, start_y + h - 2, start_x + 3, hint, (get_color(2) | curses.A_STANDOUT | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT)

        stdscr.refresh()

        k = stdscr.getch()
        if k in (10, 13, 32, 27, ord("q"), ord("Q")):
            break


def show_missing_dependency_dialog(
    stdscr: curses.window,
    exc: Any,
) -> None:
    """
    Display a centered modal interstitial dialog when an unsupported dataset format
    requires an uninstalled optional dependency (such as pyarrow, duckdb, or pylance).
    Preserves the background dashboard without erasing the screen.
    Inactive labels and titles are displayed in white; active interactive elements in teal.
    """
    import textwrap
    configure_escdelay(25)
    safe_curs_set(0)

    max_y, max_x = stdscr.getmaxyx()

    format_name = getattr(exc, "format_name", "Required Format")
    package_name = getattr(exc, "package_name", "package")
    install_command = getattr(exc, "install_command", f"pip install {package_name}")
    extra_name = getattr(exc, "extra_name", package_name.replace("py", ""))
    description = getattr(exc, "description", "")

    w = min(max_x - 4, 76)
    w = max(48, w)
    inner_w = w - 6

    raw_lines = [
        f"The '{format_name}' format requires the '{package_name}' package.",
    ]
    if description:
        raw_lines.append(description)
    raw_lines.append("")
    raw_lines.append("To enable support for this format, install the package:")

    wrapped_lines: List[str] = []
    for line in raw_lines:
        if not line.strip():
            wrapped_lines.append("")
            continue
        wrapped = textwrap.wrap(line, width=inner_w, break_long_words=True)
        wrapped_lines.extend(wrapped if wrapped else [""])

    extra_cmd = f"pip install 'mlx_commander[{extra_name}]'"
    cmd_box_width = min(inner_w, max(len(install_command) + 6, len(extra_cmd) + 6, 38))

    h = len(wrapped_lines) + 3 + 4 + 3
    h = min(h, max_y - 4)

    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    border_attr = (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
    title_attr = (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    label_attr = get_color(4) if safe_has_colors() else 0
    active_teal = (get_color(2) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
    action_attr = (get_color(2) | curses.A_STANDOUT | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", border_attr)
        title_str = f" [!] Dependency Required: {format_name} "[: w - 4]
        safe_addstr(stdscr, start_y, start_x + 2, title_str, title_attr)

        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", border_attr)

        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", border_attr)

        curr_y = start_y + 2
        for l in wrapped_lines:
            if curr_y >= start_y + h - 6:
                break
            safe_addstr(stdscr, curr_y, start_x + 3, l[:inner_w], label_attr)
            curr_y += 1

        curr_y += 1
        if curr_y < start_y + h - 4:
            safe_addstr(stdscr, curr_y, start_x + 4, "┌" + "─" * (cmd_box_width - 2) + "┐", active_teal)
            cmd_text = f"  {install_command}".ljust(cmd_box_width - 2)
            safe_addstr(stdscr, curr_y + 1, start_x + 4, f"│{cmd_text}│", active_teal)
            safe_addstr(stdscr, curr_y + 2, start_x + 4, "└" + "─" * (cmd_box_width - 2) + "┘", active_teal)
            curr_y += 3

        if curr_y < start_y + h - 2:
            hint_str = f"Or install optional extra: {extra_cmd}"
            safe_addstr(stdscr, curr_y, start_x + 4, hint_str[:inner_w], label_attr)

        hint = "[I] Install with pip   [Enter] OK / Dismiss"
        safe_addstr(stdscr, start_y + h - 2, start_x + 3, hint, action_attr)

        stdscr.refresh()

        k = stdscr.getch()
        if k in (ord("i"), ord("I")):
            installing_str = f"Installing {package_name}...".ljust(w - 6)
            safe_addstr(stdscr, start_y + h - 2, start_x + 3, installing_str[: w - 6], title_attr)
            stdscr.refresh()
            import os
            import shutil
            import subprocess
            import sys
            from pathlib import Path
            try:
                uv_bin = shutil.which("uv")
                if not uv_bin:
                    user_uv = Path.home() / ".local/bin/uv"
                    if user_uv.exists() and os.access(user_uv, os.X_OK):
                        uv_bin = str(user_uv)

                res = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package_name],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if res.returncode != 0 and uv_bin and ("No module named pip" in (res.stderr or "") or "No module named pip" in (res.stdout or "")):
                    res = subprocess.run(
                        [uv_bin, "pip", "install", "--python", sys.executable, package_name],
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                if res.returncode == 0:
                    ok_str = f"Installed {package_name} successfully!".ljust(w - 6)
                    safe_addstr(stdscr, start_y + h - 2, start_x + 3, ok_str[: w - 6], active_teal)
                    stdscr.refresh()
                    curses.napms(800)
                    return True
                else:
                    err_hint = f"install failed (code {res.returncode}). Press any key."
                    safe_addstr(stdscr, start_y + h - 2, start_x + 3, err_hint[: w - 6], action_attr)
                    stdscr.refresh()
                    stdscr.getch()
            except Exception as pe:
                err_hint = f"Error: {pe}"[: w - 6]
                safe_addstr(stdscr, start_y + h - 2, start_x + 3, err_hint, action_attr)
                stdscr.refresh()
                stdscr.getch()
            return False
        elif k in (10, 13, 32, 27, ord("q"), ord("Q")):
            return False
    return False


def show_message_dialog(
    stdscr: curses.window,
    title: str,
    message_lines: List[str],
    is_error: bool = False,
) -> None:
    """Display an informational or error modal dialog overlay on top of the screen."""
    if is_error:
        show_error_dialog(stdscr, title, message_lines)
        return

    import textwrap
    configure_escdelay(25)
    safe_curs_set(0)

    max_y, max_x = stdscr.getmaxyx()
    w = min(max_x - 6, 76)
    w = max(40, w)
    inner_w = w - 6

    wrapped_lines: List[str] = []
    for line in message_lines:
        wrapped = textwrap.wrap(str(line), width=inner_w, break_long_words=True)
        wrapped_lines.extend(wrapped if wrapped else [""])

    max_text_lines = max(1, min(len(wrapped_lines), max_y - 8))
    h = max_text_lines + 5

    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    border_attr = (get_color(2) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
    title_attr = (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    text_attr = get_color(4) if safe_has_colors() else 0

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", border_attr)
        title_str = f" {title} "[: w - 4]
        safe_addstr(stdscr, start_y, start_x + 2, title_str, title_attr)

        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", border_attr)

        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", border_attr)

        for idx in range(max_text_lines):
            safe_addstr(stdscr, start_y + 2 + idx, start_x + 3, wrapped_lines[idx][:inner_w], text_attr)

        hint = "[Enter] OK   [Esc] Dismiss"
        safe_addstr(stdscr, start_y + h - 2, start_x + 3, hint, (get_color(2) | curses.A_STANDOUT | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT)

        stdscr.refresh()

        k = stdscr.getch()
        if k in (10, 13, 32, 27, ord("q"), ord("Q")):
            break


def draw_box_panel(
    win: curses.window,
    y: int,
    x: int,
    h: int,
    w: int,
    title: str,
    is_focused: bool = False,
    subtitle: str = "",
) -> None:
    """Draw a styled box panel with title and active focus border."""
    if h <= 2 or w <= 4:
        return

    border_attr = (get_color(COLOR_BORDER_JOINTS) | curses.A_BOLD) if (is_focused and safe_has_colors()) else get_color(COLOR_BORDER_JOINTS)
    title_attr = (get_color(COLOR_TITLE_ACCENT) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD

    safe_addstr(win, y, x, "┌─ ", border_attr)
    title_text = f"{title} "
    safe_addstr(win, y, x + 3, title_text, title_attr)
    curr_x = x + 3 + len(title_text)
    if subtitle:
        sub_text = f"[{subtitle}] "
        safe_addstr(win, y, curr_x, sub_text, (get_color(COLOR_LABEL_GRAY) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)
        curr_x += len(sub_text)
    avail_line = max(0, (x + w - 1) - curr_x)
    if avail_line > 0:
        safe_addstr(win, y, curr_x, "─" * avail_line + "┐", border_attr)
    else:
        safe_addstr(win, y, x + w - 1, "┐", border_attr)

    for row in range(y + 1, y + h - 1):
        safe_addstr(win, row, x, "│", border_attr)
        safe_addstr(win, row, x + w - 1, "│", border_attr)

    safe_addstr(win, y + h - 1, x, "└" + "─" * (w - 2) + "┘", border_attr)


def draw_button(win: curses.window, y: int, x: int, label: str, is_focused: bool = False) -> None:
    """Draw an interactive button with deep blue background and active white font."""
    btn_str = f"[ {label} ]"
    if is_focused:
        attr = (get_color(COLOR_INPUT_FOCUSED) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
    else:
        attr = get_color(COLOR_INPUT_NORMAL) if safe_has_colors() else 0
    safe_addstr(win, y, x, btn_str, attr)


def draw_radio(
    win: curses.window,
    y: int,
    x: int,
    label: str,
    is_checked: bool = False,
    is_focused: bool = False,
) -> None:
    """Draw an interactive radio button option with input field color or active focus."""
    mark = "●" if is_checked else " "
    radio_str = f"({mark}) {label}"
    if is_focused:
        attr = (get_color(COLOR_INPUT_FOCUSED) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
    elif is_checked:
        attr = (get_color(COLOR_INPUT_NORMAL) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    else:
        attr = get_color(COLOR_INPUT_NORMAL) if safe_has_colors() else 0
    safe_addstr(win, y, x, radio_str, attr)


def draw_field(
    win: curses.window,
    y: int,
    x: int,
    label: str,
    val_str: str,
    is_focused: bool = False,
    val_width: int = 24,
    has_dropdown: bool = False,
    right_edge: Optional[int] = None,
    field_x: Optional[int] = None,
    lbl_attr: Optional[int] = None,
) -> None:
    """Draw a labeled form field with deep blue background and active bright white font.

    If right_edge is specified, aligns the field box so its right bracket is at right_edge,
    while keeping the label aligned at x (the left side).
    If field_x is specified, renders the box starting explicitly at field_x.
    If lbl_attr is specified, overrides the default bold gray label styling.
    """
    lbl = f"{label}: "
    used_lbl_attr = (
        lbl_attr
        if lbl_attr is not None
        else ((get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
    )
    safe_addstr(win, y, x, lbl, used_lbl_attr)

    if right_edge is not None and field_x is None:
        avail_for_box = right_edge - (x + len(lbl)) + 1
        if avail_for_box > 4 and val_width + 2 > avail_for_box:
            val_width = max(4, avail_for_box - 2)

    disp_val = val_str or "<none>"
    arrow = " ▾" if has_dropdown else ""
    max_len = max(4, val_width - len(arrow) - 2)
    if len(disp_val) > max_len:
        if "/" in disp_val or "\\" in disp_val:
            disp_val = "…" + disp_val[-(max_len - 1):]
        else:
            disp_val = disp_val[: max_len - 1] + "…"

    field_str = f" {disp_val}{arrow} "

    if field_x is not None:
        val_x = field_x
    elif right_edge is not None:
        field_len = len(field_str)
        min_val_x = x + len(lbl)
        val_x = max(min_val_x, right_edge - field_len + 1)
    else:
        val_x = x + len(lbl)

    if is_focused:
        attr = (get_color(COLOR_INPUT_FOCUSED) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
    else:
        attr = get_color(COLOR_INPUT_NORMAL) if safe_has_colors() else 0
    safe_addstr(win, y, val_x, field_str, attr)


def draw_multi_field(
    win: curses.window,
    y: int,
    x: int,
    label: str,
    values: List[Any],
    is_focused: bool = False,
    right_edge: Optional[int] = None,
    lbl_attr: Optional[int] = None,
) -> None:
    """Draw a form field that can contain multiple condition values on a single line.

    All added conditions are rendered as individual boxes [ val ].
    The rightmost box represents the active clickable target; when is_focused is True,
    only the rightmost box is styled with COLOR_INPUT_FOCUSED while earlier boxes
    remain in COLOR_INPUT_NORMAL.
    """
    lbl = f"{label}: "
    used_lbl_attr = (
        lbl_attr
        if lbl_attr is not None
        else ((get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
    )
    safe_addstr(win, y, x, lbl, used_lbl_attr)

    box_strs: List[str] = []
    for v in values:
        if isinstance(v, float):
            disp = f"{v:g}"
        elif isinstance(v, bool):
            disp = ("Yes" if v else "No") if "eval" in label.lower() else ("True" if v else "False")
        else:
            disp = str(v)
        box_strs.append(f" {disp} ")

    if not box_strs:
        box_strs = [" <none> "]

    total_w = sum(len(b) for b in box_strs) + (len(box_strs) - 1)
    if right_edge is not None:
        min_x = x + len(lbl)
        val_x = max(min_x, right_edge - total_w + 1)
    else:
        val_x = x + len(lbl)

    cur_x = val_x
    for i, b_str in enumerate(box_strs):
        is_last = (i == len(box_strs) - 1)
        if is_focused and is_last:
            attr = (get_color(COLOR_INPUT_FOCUSED) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
        else:
            attr = get_color(COLOR_INPUT_NORMAL) if safe_has_colors() else 0
        safe_addstr(win, y, cur_x, b_str, attr)
        cur_x += len(b_str) + 1


def show_column_picker_dialog(
    stdscr: curses.window,
    title: str,
    columns: List[str],
    current_val: Optional[str] = None,
    allow_none: bool = False,
) -> Optional[str]:
    """
    Modal dialog overlay to pick or concatenate columns from a scrollable list.
    Supports:
    - [Space]: Toggle column selection. Multiple selections are ordered in the sequence
               they were pressed, displayed with [1], [2], etc., and concatenated with ' + '.
    - [Enter]: Confirm selection.
    - [Esc] / [q]: Cancel and return previous value without delay.
    """
    from mlx_commander.formats import parse_column_list

    configure_escdelay(25)
    options = []
    if allow_none:
        options.append("(None / Skip)")
    options.extend(columns)

    if not options:
        return None

    # Track ordered selected columns
    initial_cols = [c for c in parse_column_list(current_val, columns) if c in columns]
    selected_cols: List[str] = list(initial_cols)
    space_used = False

    sel_idx = 1 if (allow_none and len(options) > 1) else 0
    if selected_cols:
        for i, opt in enumerate(options):
            if opt == selected_cols[0]:
                sel_idx = i
                break
    elif current_val:
        for i, opt in enumerate(options):
            if opt == current_val:
                sel_idx = i
                break

    initial_sel_idx = sel_idx

    max_y, max_x = stdscr.getmaxyx()
    h = min(max_y - 4, max(len(options) + 7, 14), 22)
    longest_col = max((len(c) for c in options), default=10)
    w = min(max_x - 4, max(longest_col + 24, 76))
    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    scroll_offset = max(0, sel_idx - (h - 7) // 2)
    safe_curs_set(0)

    while True:
        # Update bottom hotkey bar for modal dialog so Convert is not visible at bottom
        safe_addstr(stdscr, max_y - 1, 0, " " * max_x, get_color(COLOR_BANNER))
        footer_keys = "[Space] Toggle/Order  [Enter] Confirm Selection  [Esc/q] Cancel"
        safe_addstr(stdscr, max_y - 1, 2, footer_keys, get_color(COLOR_BANNER))

        # Draw box and clear all internal rows
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", get_color(2) | curses.A_BOLD)
        safe_addstr(stdscr, start_y, start_x + 2, f" {title} ", (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", get_color(2))
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", get_color(2) | curses.A_BOLD)

        visible_rows = h - 5
        if sel_idx < scroll_offset:
            scroll_offset = sel_idx
        elif sel_idx >= scroll_offset + visible_rows:
            scroll_offset = sel_idx - visible_rows + 1

        for r in range(visible_rows):
            row_y = start_y + 1 + r
            opt_idx = scroll_offset + r
            if opt_idx < len(options):
                opt_name = options[opt_idx]
                is_focused = (opt_idx == sel_idx)
                prefix = " ▶ " if is_focused else "   "

                if opt_name == "(None / Skip)":
                    box = "   "
                elif opt_name in selected_cols:
                    if len(selected_cols) > 1:
                        order_num = selected_cols.index(opt_name) + 1
                        box = f"[{order_num}]"
                    else:
                        box = "[*]"
                else:
                    box = "[ ]"

                line_text = f"{prefix}{box} {opt_name}"[: w - 4]
                if is_focused:
                    attr = (get_color(2) | curses.A_STANDOUT | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
                elif opt_name in selected_cols:
                    attr = (get_color(2) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
                else:
                    attr = get_color(2) if safe_has_colors() else 0

                safe_addstr(stdscr, row_y, start_x + 1, line_text, attr)

        # Divider above summary & footer
        safe_addstr(stdscr, start_y + h - 4, start_x, "╟" + "─" * (w - 2) + "╢", get_color(2))

        # Concatenation selection summary
        safe_addstr(stdscr, start_y + h - 3, start_x + 2, "Selection: ", (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        if selected_cols:
            summary_txt = " + ".join(selected_cols)
            safe_addstr(stdscr, start_y + h - 3, start_x + 13, summary_txt[: w - 15], (get_color(2) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        else:
            safe_addstr(stdscr, start_y + h - 3, start_x + 13, "<none>", (get_color(4) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)

        # Footer inside dialog
        safe_addstr(stdscr, start_y + h - 2, start_x + 2, "[Space] Toggle/Order  [Enter] OK  [Esc] Cancel"[: w - 4], (get_color(4) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (curses.KEY_UP, curses.KEY_LEFT, ord("k"), ord("h")):
            sel_idx = max(0, sel_idx - 1)
        elif k in (curses.KEY_DOWN, curses.KEY_RIGHT, ord("j"), ord("l")):
            sel_idx = min(len(options) - 1, sel_idx + 1)
        elif k == 32:  # Spacebar: toggle selection
            space_used = True
            chosen = options[sel_idx]
            if chosen == "(None / Skip)":
                selected_cols.clear()
            else:
                if chosen in selected_cols:
                    selected_cols.remove(chosen)
                else:
                    selected_cols.append(chosen)
        elif k in (10, 13, curses.KEY_ENTER):
            if space_used:
                if not selected_cols:
                    if options[sel_idx] == "(None / Skip)":
                        return None
                    return None if allow_none else options[sel_idx]
                elif len(selected_cols) == 1:
                    return selected_cols[0]
                else:
                    return " + ".join(selected_cols)
            else:
                # Space was not pressed: single-select navigation
                chosen = options[sel_idx]
                if allow_none and chosen == "(None / Skip)":
                    return None
                if len(selected_cols) > 1 and sel_idx == initial_sel_idx:
                    return " + ".join(selected_cols)
                return chosen
        elif k in (27, ord("q")):
            return current_val


def show_text_edit_dialog(
    stdscr: curses.window,
    title: str,
    prompt: str,
    default_val: str = "",
    is_number: bool = False,
) -> Optional[str]:
    """Modal dialog overlay to edit a single value in-place with horizontal scrolling."""
    configure_escdelay(25)
    max_y, max_x = stdscr.getmaxyx()
    h = 7
    w = min(max_x - 4, max(50, min(max_x - 8, 86)))
    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    safe_curs_set(1)
    text_chars = list(default_val)
    cursor_pos = len(text_chars)
    box_w = max(10, w - 6)
    scroll_offset = max(0, cursor_pos - box_w + 1) if len(text_chars) >= box_w else 0

    while True:
        cursor_pos = max(0, min(len(text_chars), cursor_pos))
        if len(text_chars) < box_w:
            scroll_offset = 0
        else:
            if cursor_pos < scroll_offset:
                scroll_offset = cursor_pos
            elif cursor_pos >= scroll_offset + box_w:
                scroll_offset = cursor_pos - box_w + 1
            scroll_offset = max(0, min(scroll_offset, len(text_chars) - box_w + 1))
            scroll_offset = max(0, min(scroll_offset, cursor_pos))

        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", get_color(2) | curses.A_BOLD)
        safe_addstr(stdscr, start_y, start_x + 2, f" {title} ", (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", get_color(2))
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", get_color(2) | curses.A_BOLD)

        safe_addstr(stdscr, start_y + 1, start_x + 2, prompt[: w - 4], (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)

        disp_slice = text_chars[scroll_offset : scroll_offset + box_w]
        disp = "".join(disp_slice)
        pad = " " * (box_w - len(disp))
        safe_addstr(stdscr, start_y + 3, start_x + 3, disp + pad, (get_color(2) | curses.A_STANDOUT) if safe_has_colors() else curses.A_STANDOUT)

        left_ind = "«" if scroll_offset > 0 else " "
        right_ind = "»" if (scroll_offset + box_w < len(text_chars)) else " "
        safe_addstr(stdscr, start_y + 3, start_x + 2, left_ind, (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, start_y + 3, start_x + 3 + box_w, right_ind, (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)

        hint = "[Enter] OK   [Esc] Cancel   [←/→] Scroll   [Home/End]"
        safe_addstr(stdscr, start_y + 5, start_x + 2, hint[: w - 4], (get_color(4) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)

        cursor_col = cursor_pos - scroll_offset
        cursor_x = start_x + 3 + cursor_col
        stdscr.move(start_y + 3, cursor_x)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (10, 13, curses.KEY_ENTER):
            safe_curs_set(0)
            return "".join(text_chars)
        elif k == 27:  # ESC
            safe_curs_set(0)
            return None
        elif k in (curses.KEY_BACKSPACE, 127, 8):
            if cursor_pos > 0:
                text_chars.pop(cursor_pos - 1)
                cursor_pos -= 1
        elif k in (curses.KEY_DC, 4):  # Delete or Ctrl+D
            if cursor_pos < len(text_chars):
                text_chars.pop(cursor_pos)
        elif k == curses.KEY_LEFT:
            cursor_pos = max(0, cursor_pos - 1)
        elif k == curses.KEY_RIGHT:
            cursor_pos = min(len(text_chars), cursor_pos + 1)
        elif k in (curses.KEY_HOME, 1):  # Home or Ctrl+A
            cursor_pos = 0
            scroll_offset = 0
        elif k in (curses.KEY_END, 5):  # End or Ctrl+E
            cursor_pos = len(text_chars)
            scroll_offset = max(0, cursor_pos - box_w + 1)
        elif k == 21:  # Ctrl+U
            text_chars = text_chars[cursor_pos:]
            cursor_pos = 0
            scroll_offset = 0
        elif k == 11:  # Ctrl+K
            text_chars = text_chars[:cursor_pos]
        elif 32 <= k <= 126:
            char = chr(k)
            if is_number and char not in "0123456789.":
                continue
            text_chars.insert(cursor_pos, char)
            cursor_pos += 1


def show_multi_value_edit_dialog(
    stdscr: curses.window,
    title: str,
    prompt: str,
    current_values: List[Any],
    val_type: type = int,
) -> Optional[List[Any]]:
    """Modal dialog overlay to edit multi-condition values as comma-separated entries with horizontal scrolling."""
    configure_escdelay(25)
    max_y, max_x = stdscr.getmaxyx()
    h = 8
    w = min(max_x - 4, max(50, min(max_x - 8, 86)))
    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    safe_curs_set(1)
    if val_type is float:
        default_str = ", ".join(f"{v:g}" for v in current_values)
    elif val_type is bool:
        if "eval" in title.lower():
            default_str = ", ".join("Yes" if v else "No" for v in current_values)
        else:
            default_str = ", ".join("True" if v else "False" for v in current_values)
    else:
        default_str = ", ".join(str(v) for v in current_values)

    text_chars = list(default_str)
    cursor_pos = len(text_chars)
    box_w = max(10, w - 6)
    scroll_offset = max(0, cursor_pos - box_w + 1) if len(text_chars) >= box_w else 0
    err_msg = ""

    while True:
        cursor_pos = max(0, min(len(text_chars), cursor_pos))
        if len(text_chars) < box_w:
            scroll_offset = 0
        else:
            if cursor_pos < scroll_offset:
                scroll_offset = cursor_pos
            elif cursor_pos >= scroll_offset + box_w:
                scroll_offset = cursor_pos - box_w + 1
            scroll_offset = max(0, min(scroll_offset, len(text_chars) - box_w + 1))
            scroll_offset = max(0, min(scroll_offset, cursor_pos))

        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", get_color(2) | curses.A_BOLD)
        safe_addstr(stdscr, start_y, start_x + 2, f" {title} ", (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", get_color(2))
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", get_color(2) | curses.A_BOLD)

        safe_addstr(stdscr, start_y + 1, start_x + 2, prompt[: w - 4], (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        if err_msg:
            safe_addstr(stdscr, start_y + 2, start_x + 2, err_msg[:w - 4], (get_color(COLOR_ERROR) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        else:
            hint = "Comma-separated list (e.g. 2, 4, 8) — each forms a sweep run"
            safe_addstr(stdscr, start_y + 2, start_x + 2, hint[:w - 4], (get_color(COLOR_LABEL_GRAY) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)

        disp_slice = text_chars[scroll_offset : scroll_offset + box_w]
        disp = "".join(disp_slice)
        pad = " " * (box_w - len(disp))
        safe_addstr(stdscr, start_y + 4, start_x + 3, disp + pad, (get_color(2) | curses.A_STANDOUT) if safe_has_colors() else curses.A_STANDOUT)

        left_ind = "«" if scroll_offset > 0 else " "
        right_ind = "»" if (scroll_offset + box_w < len(text_chars)) else " "
        safe_addstr(stdscr, start_y + 4, start_x + 2, left_ind, (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, start_y + 4, start_x + 3 + box_w, right_ind, (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)

        footer_hint = "[Enter] OK   [Esc] Cancel   [←/→] Scroll   [Home/End]"
        safe_addstr(stdscr, start_y + 6, start_x + 2, footer_hint[: w - 4], (get_color(4) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)

        cursor_col = cursor_pos - scroll_offset
        cursor_x = start_x + 3 + cursor_col
        stdscr.move(start_y + 4, cursor_x)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (10, 13, curses.KEY_ENTER):
            tokens = [t.strip() for t in "".join(text_chars).split(",") if t.strip()]
            if not tokens:
                err_msg = "Please enter at least one valid value."
                continue
            parsed = []
            parse_err = False
            tok_err = ""
            for tok in tokens:
                try:
                    if val_type is int:
                        parsed.append(int(tok))
                    elif val_type is float:
                        parsed.append(float(tok))
                    elif val_type is bool:
                        tok_lower = tok.lower()
                        if tok_lower in ("true", "1", "yes", "t", "y"):
                            parsed.append(True)
                        elif tok_lower in ("false", "0", "no", "f", "n"):
                            parsed.append(False)
                        else:
                            parse_err = True
                            tok_err = tok
                            break
                    else:
                        parsed.append(tok)
                except ValueError:
                    parse_err = True
                    tok_err = tok
                    break

            if parse_err:
                err_msg = f"Invalid value '{tok_err}'. Expected: {val_type.__name__}."
                continue

            safe_curs_set(0)
            return list(dict.fromkeys(parsed))
        elif k == 27:  # ESC
            safe_curs_set(0)
            return None
        elif k in (curses.KEY_BACKSPACE, 127, 8):
            if cursor_pos > 0:
                text_chars.pop(cursor_pos - 1)
                cursor_pos -= 1
                err_msg = ""
        elif k in (curses.KEY_DC, 4):  # Delete or Ctrl+D
            if cursor_pos < len(text_chars):
                text_chars.pop(cursor_pos)
                err_msg = ""
        elif k == curses.KEY_LEFT:
            cursor_pos = max(0, cursor_pos - 1)
        elif k == curses.KEY_RIGHT:
            cursor_pos = min(len(text_chars), cursor_pos + 1)
        elif k in (curses.KEY_HOME, 1):  # Home or Ctrl+A
            cursor_pos = 0
            scroll_offset = 0
        elif k in (curses.KEY_END, 5):  # End or Ctrl+E
            cursor_pos = len(text_chars)
            scroll_offset = max(0, cursor_pos - box_w + 1)
        elif k == 21:  # Ctrl+U
            text_chars = text_chars[cursor_pos:]
            cursor_pos = 0
            scroll_offset = 0
            err_msg = ""
        elif k == 11:  # Ctrl+K
            text_chars = text_chars[:cursor_pos]
            err_msg = ""
        elif 32 <= k <= 126:
            char = chr(k)
            text_chars.insert(cursor_pos, char)
            cursor_pos += 1
            err_msg = ""


def show_output_destination_dialog(
    stdscr: curses.window,
    current_output: str,
    default_output: str,
) -> Optional[str]:
    """
    Dialog allowing user to choose how to specify the destination output folder:
      1) Open macOS Finder (GUI Folder Picker) [if macOS]
      2) Enter folder path manually
      3) Reset to default (<dataset_dir>/mlx_dataset)
    Returns:
      Selected path string, or None if cancelled with Esc.
    """
    from mlx_commander.gui_picker import is_macos, pick_folder_gui

    options: List[Tuple[str, str]] = []
    if is_macos():
        options.append(("Finder (GUI Folder Picker)", "Open native macOS Finder to select destination folder"))
    options.append(("Manual Path Entry", "Type or paste custom destination directory path"))
    options.append(("Reset to Default", f"Use default: {default_output}"))

    sel_idx = 0
    max_y, max_x = stdscr.getmaxyx()
    h = min(12, max_y - 4)
    w = min(74, max_x - 4)
    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    safe_curs_set(0)

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", (get_color(2) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, start_y, start_x + 2, " Destination Output Folder ", (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", get_color(2) if safe_has_colors() else 0)
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", (get_color(2) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)

        # Summary of current and default paths
        curr_disp = current_output or "<none>"
        if len(curr_disp) > w - 16:
            curr_disp = "…" + curr_disp[-(w - 17):]
        def_disp = default_output or "<none>"
        if len(def_disp) > w - 16:
            def_disp = "…" + def_disp[-(w - 17):]

        safe_addstr(stdscr, start_y + 1, start_x + 3, "Current: ", (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, start_y + 1, start_x + 12, curr_disp, (get_color(2) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, start_y + 2, start_x + 3, "Default: ", (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, start_y + 2, start_x + 12, def_disp, (get_color(4) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)

        safe_addstr(stdscr, start_y + 3, start_x + 2, "╟" + "─" * (w - 4) + "╢", (get_color(2) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)

        # Draw option rows
        for i, (label, desc) in enumerate(options):
            row_y = start_y + 4 + i
            is_focused = (i == sel_idx)
            prefix = " ▶ " if is_focused else "   "
            line_str = f"{prefix}{label:<28} {desc}"[: w - 6]
            if is_focused:
                attr = (get_color(2) | curses.A_STANDOUT | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
            else:
                attr = get_color(2) if safe_has_colors() else 0
            safe_addstr(stdscr, row_y, start_x + 2, line_str, attr)

        safe_addstr(stdscr, start_y + h - 2, start_x + 3, "[Enter/Space] Select   [Esc] Cancel", (get_color(4) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (curses.KEY_UP, curses.KEY_LEFT, ord("k"), ord("h")):
            sel_idx = (sel_idx - 1) % len(options)
        elif k in (curses.KEY_DOWN, curses.KEY_RIGHT, ord("j"), ord("l")):
            sel_idx = (sel_idx + 1) % len(options)
        elif k in (27, ord("q"), ord("Q")):
            return None
        elif k in (10, 13, 32):  # Enter or Space
            chosen_label = options[sel_idx][0]
            if "Finder" in chosen_label:
                curses.def_prog_mode()
                curses.endwin()
                start_dir = current_output or default_output or os.getcwd()
                res = pick_folder_gui("Select Destination Folder", default_dir=start_dir)
                curses.reset_prog_mode()
                stdscr.refresh()
                return res if res else None
            elif "Manual" in chosen_label:
                val = show_text_edit_dialog(
                    stdscr,
                    "Output Directory",
                    "Enter folder to save MLX JSONL datasets:",
                    default_val=current_output or default_output,
                )
                return val.strip() if val else None
            elif "Reset" in chosen_label:
                return default_output


def show_results_dialog(stdscr: curses.window, result: Any, *args: Any) -> None:
    """Modal dialog displaying conversion results."""
    if not hasattr(result, "output_dir") or result is False:
        msg = args[0] if args else (str(result) if result else "Operation failed.")
        show_error_dialog(stdscr, "Operation Error", msg)
        return

    configure_escdelay(25)
    safe_curs_set(0)

    max_y, max_x = stdscr.getmaxyx()
    num_files = len(getattr(result, "output_files", {}))
    h = min(max(9, 7 + num_files), max_y - 2)
    w = min(74, max_x - 4)
    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", get_color(3) | curses.A_BOLD)
        safe_addstr(stdscr, start_y, start_x + 2, " [OK] Conversion Successful! ", (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", get_color(3))
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", get_color(3) | curses.A_BOLD)

        safe_addstr(stdscr, start_y + 2, start_x + 3, f"Saved dataset to: {result.output_dir}"[: w - 6], (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        row = start_y + 4
        for split_name, file_path in getattr(result, "output_files", {}).items():
            if row < start_y + h - 2:
                cnt = getattr(result, "record_counts", {}).get(split_name, 0)
                sz = getattr(result, "file_sizes", {}).get(split_name, 0) / 1024.0
                safe_addstr(stdscr, row, start_x + 3, f"• {file_path.name}: {cnt:,} records ({sz:.1f} KB)"[: w - 6], (get_color(4) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)
                row += 1

        safe_addstr(stdscr, start_y + h - 2, start_x + 3, "Press [Enter], [Space], or [Esc] to return", (get_color(2) | curses.A_STANDOUT | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (10, 13, 32, 27, ord("q"), ord("Q")):
            break


def show_help_dialog(stdscr: curses.window) -> None:
    """Modal dialog displaying all keyboard shortcuts."""
    from mlx_commander import __version__

    max_y, max_x = stdscr.getmaxyx()
    h = min(22, max_y - 2)
    w = min(74, max_x - 4)
    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    title_str = f" MLX Commander v{__version__} — Keyboard Shortcuts "
    if len(title_str) > w - 4:
        title_str = f" MLX Commander v{__version__} "
    if len(title_str) > w - 4:
        title_str = f" v{__version__} "

    shortcuts = [
        ("F2", "Cycle Modes: Converter -> Single-Run -> Multi-Run Sweep"),
        ("Tab / Shift-Tab", "Cycle focus between active screen panels"),
        ("↑ / ↓ (or k / j)", "Navigate vertically through fields, options, and queue"),
        ("← / → (or h / l)", "Navigate horizontally between columns"),
        ("Enter / Space", "Edit field, toggle value, or load queued run"),
        ("F3", "Open macOS Finder to choose output destination folder"),
        ("F5", "Run immediately with current config or Execute Queue (Mode 2 & 3)"),
        ("F6", "Schedule current configuration to Queue (Mode 2 & 3)"),
        ("c", "Clone selected LoRA run in Queue (Mode 2)"),
        ("d", "Delete selected LoRA run from Queue (Mode 2)"),
        ("x", "Clear LoRA Queue (Mode 2)"),
        ("F9", "Toggle theme (Norton Commander <-> Modern)"),
        ("r / R", "Randomize split seed"),
        ("? / F1", "Show this help screen"),
        ("F10 / q / Esc", "Exit MLX Commander"),
    ]

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", get_color(2) | curses.A_BOLD)
        safe_addstr(stdscr, start_y, start_x + 2, title_str, (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", get_color(2))
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", get_color(2) | curses.A_BOLD)

        for i, (key, desc) in enumerate(shortcuts):
            row = start_y + 2 + i
            if row < start_y + h - 2:
                safe_addstr(stdscr, row, start_x + 3, f"{key:<20}", (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
                safe_addstr(stdscr, row, start_x + 24, desc[: w - 27], (get_color(4) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)

        safe_addstr(stdscr, start_y + h - 2, start_x + 3, "Press [Enter], [Space], or [Esc] to close", (get_color(2) | curses.A_STANDOUT | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (10, 13, 32, 27, ord("q"), ord("Q")):
            break


def get_schema_mapping_targets(
    target_format: Any,
    mapping: Any,
    available_columns: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Extract structured target fields and unmapped source columns.
    Returns:
      (targets, unmapped_columns)
    Each target is:
      {
        "key": str,          # e.g. "prompt", "completion", "chosen", etc.
        "label": str,        # display label
        "type_str": str,     # e.g. "string", "string/dict", "list[dict]"
        "required": bool,    # whether this field is required
        "cols": List[str],   # mapped source columns (parsed from col_spec)
        "is_concat": bool,   # True if 2 or more columns are combined
      }
    """
    from mlx_commander.formats import MLXFormat, parse_column_list

    avail = available_columns or []
    targets: List[Dict[str, Any]] = []

    if target_format == MLXFormat.PROMPT_COMPLETION:
        prompt_spec = getattr(mapping, "prompt_col", None)
        comp_spec = getattr(mapping, "completion_col", None)
        p_cols = parse_column_list(prompt_spec, avail) if prompt_spec else []
        c_cols = parse_column_list(comp_spec, avail) if comp_spec else []
        targets.append({
            "key": "prompt",
            "label": "prompt",
            "type_str": "string",
            "required": True,
            "cols": p_cols,
            "is_concat": len(p_cols) > 1,
        })
        targets.append({
            "key": "completion",
            "label": "completion",
            "type_str": "string",
            "required": True,
            "cols": c_cols,
            "is_concat": len(c_cols) > 1,
        })

    elif target_format == MLXFormat.CHAT:
        messages_col = getattr(mapping, "messages_col", None)
        if messages_col:
            m_cols = parse_column_list(messages_col, avail) if messages_col else []
            targets.append({
                "key": "messages",
                "label": "messages",
                "type_str": "list[dict]",
                "required": True,
                "cols": m_cols,
                "is_concat": len(m_cols) > 1,
            })
        else:
            u_cols = parse_column_list(getattr(mapping, "user_col", None), avail)
            a_cols = parse_column_list(getattr(mapping, "assistant_col", None), avail)
            s_cols = parse_column_list(getattr(mapping, "system_col", None), avail)
            targets.append({
                "key": "user",
                "label": "user (turn)",
                "type_str": "string",
                "required": True,
                "cols": u_cols,
                "is_concat": len(u_cols) > 1,
            })
            targets.append({
                "key": "assistant",
                "label": "assistant (turn)",
                "type_str": "string",
                "required": True,
                "cols": a_cols,
                "is_concat": len(a_cols) > 1,
            })
            targets.append({
                "key": "system",
                "label": "system (prompt)",
                "type_str": "string (opt)",
                "required": False,
                "cols": s_cols,
                "is_concat": len(s_cols) > 1,
            })

    elif target_format == MLXFormat.DPO:
        p_raw = getattr(mapping, "dpo_prompt_col", None) or getattr(mapping, "prompt_col", None)
        p_cols = parse_column_list(p_raw, avail) if p_raw else []
        c_cols = parse_column_list(getattr(mapping, "chosen_col", None), avail)
        r_cols = parse_column_list(getattr(mapping, "rejected_col", None), avail)
        targets.append({
            "key": "prompt",
            "label": "prompt",
            "type_str": "string/dict",
            "required": True,
            "cols": p_cols,
            "is_concat": len(p_cols) > 1,
        })
        targets.append({
            "key": "chosen",
            "label": "chosen (pref)",
            "type_str": "string/dict",
            "required": True,
            "cols": c_cols,
            "is_concat": len(c_cols) > 1,
        })
        targets.append({
            "key": "rejected",
            "label": "rejected (disp)",
            "type_str": "string/dict",
            "required": True,
            "cols": r_cols,
            "is_concat": len(r_cols) > 1,
        })

    elif target_format == MLXFormat.TEXT:
        t_raw = getattr(mapping, "text_col", None)
        t_cols = parse_column_list(t_raw, avail) if t_raw else []
        targets.append({
            "key": "text",
            "label": "text",
            "type_str": "string",
            "required": True,
            "cols": t_cols,
            "is_concat": len(t_cols) > 1,
        })

    # Unmapped columns: original dataset columns that are not assigned to any target
    mapped_set = set()
    for t in targets:
        for c in t["cols"]:
            mapped_set.add(c)
    unmapped = [c for c in avail if c not in mapped_set]

    return targets, unmapped


def plan_mapping_panel_rows(
    targets: List[Dict[str, Any]],
    unmapped: List[str],
    avail_rows: int,
) -> Tuple[List[int], bool]:
    """
    Determine row allocation for each target and whether to show unmapped columns.
    Guarantees total rendered lines fits within avail_rows.
    """
    n = len(targets)
    if n == 0 or avail_rows < 3:
        return ([1] * max(1, n), False)

    min_needed = 2 + (n - 1) + n  # 1 top border + 1 bottom border + (n - 1) dividers + n target rows
    if avail_rows < min_needed:
        return ([1] * n, False)

    show_unmapped = False
    if unmapped and avail_rows >= min_needed + 2:
        show_unmapped = True

    used = min_needed + (2 if show_unmapped else 0)
    extra = avail_rows - used
    heights = [1] * n

    for i, t in enumerate(targets):
        if extra > 0 and len(t["cols"]) > 1:
            heights[i] += 1
            extra -= 1

    return heights, show_unmapped


def draw_mapping_pipeline_panel(
    win: curses.window,
    y: int,
    x: int,
    h: int,
    w: int,
    state: Any,
) -> None:
    """
    Draw the visual column mapping pipeline panel with:
    - Bundled original columns reordered per MLX target field
    - Matching jointed divider blocks on both source and target sides
    - Clear flow routing joints connecting source bundles to MLX target blocks
    - Concat badges and bracket tree joints for multi-column mappings
    - Clean typography with zero emojis
    """
    if h < 4 or w < 50:
        return

    from mlx_commander.formats import MLXFormat

    target_fmt = getattr(state, "target_format", MLXFormat.PROMPT_COMPLETION)
    mapping = getattr(state, "mapping", None)
    loaded_ds = getattr(state, "loaded_dataset", None)
    avail_cols = loaded_ds.columns if loaded_ds else []

    targets, unmapped = get_schema_mapping_targets(target_fmt, mapping, avail_cols)

    # 1. Outer Box Panel
    draw_box_panel(
        win,
        y,
        x,
        h,
        w,
        "Column Mapping Pipeline (Source -> Target MLX Schema)",
        is_focused=False,
        subtitle=target_fmt.value.upper(),
    )

    # 2. Dimensions and Header Row
    box_w = max(22, min(28, (w - 24) // 2))
    left_x = x + 2
    right_x = x + w - box_w - 2
    flow_x = left_x + box_w
    flow_w = right_x - flow_x

    header_y = y + 1
    safe_addstr(win, header_y, left_x + 1, "ORIGINAL COLUMNS", curses.A_BOLD | get_color(4))
    flow_title = "FLOW JOINTS"
    safe_addstr(win, header_y, flow_x + max(0, (flow_w - len(flow_title)) // 2), flow_title, curses.A_DIM | get_color(2))
    safe_addstr(win, header_y, right_x + 1, "TARGET MLX SCHEMA", curses.A_BOLD | get_color(4))

    # 3. Row Allocation
    start_y = y + 2
    avail_rows = h - 3
    if avail_rows < 2:
        return

    n = len(targets)
    heights, show_unmapped = plan_mapping_panel_rows(targets, unmapped, avail_rows)

    border_attr = get_color(2) if safe_has_colors() else curses.A_DIM
    header_box_attr = (get_color(2) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    green_attr = (get_color(3) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    cyan_attr = (get_color(2) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    yellow_attr = (get_color(6) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    red_attr = (get_color(5) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    dim_attr = curses.A_DIM

    curr_y = start_y

    def fmt_border(left_char: str, right_char: str, title: str, width: int) -> str:
        if title:
            prefix = f"{left_char}─ {title} "
            rem = width - len(prefix) - 1
            if rem >= 0:
                return prefix + "─" * rem + right_char
        return left_char + "─" * (width - 2) + right_char

    # Top Border of Boxes
    if targets and curr_y < y + h - 1:
        t0 = targets[0]
        left_top = fmt_border("┌", "┐", "", box_w)
        right_top = fmt_border("┌", "┐", f"Target: {t0['key']}", box_w)
        safe_addstr(win, curr_y, left_x, left_top, header_box_attr)
        safe_addstr(win, curr_y, right_x, right_top, header_box_attr)
        curr_y += 1

    # Render each target block
    for i, t in enumerate(targets):
        cols = t["cols"]
        b_h = heights[i]

        for r in range(b_h):
            if curr_y >= y + h - 1:
                break

            # Left Box Content
            if not loaded_ds:
                l_text = " (No dataset loaded)" if r == 0 else ""
                l_attr = dim_attr
            elif not cols:
                l_text = " ! (none selected)" if r == 0 else ""
                l_attr = red_attr if t["required"] else dim_attr
            elif len(cols) == 1:
                l_text = f" • {cols[0]}" if r == 0 else "   [1 column]"
                l_attr = green_attr if r == 0 else dim_attr
            elif len(cols) >= 2:
                if b_h >= 2:
                    c_name = cols[r] if r < len(cols) else cols[-1]
                    l_text = f" • {c_name}"
                    l_attr = cyan_attr
                else:
                    c_joined = " + ".join(cols)
                    l_text = f" • {c_joined}"
                    l_attr = cyan_attr

            l_padded = f"│ {l_text:<{box_w - 4}} │"
            safe_addstr(win, curr_y, left_x, l_padded, border_attr)
            if l_text:
                safe_addstr(win, curr_y, left_x + 2, l_text[:box_w - 4], l_attr)

            # Right Box Content
            if not loaded_ds:
                r_text = f" ○ {t['key']} (awaiting)" if r == 0 else ""
                r_attr = dim_attr
            elif not cols:
                r_status = "missing" if t["required"] else "optional"
                r_text = f" ○ {t['key']} ({r_status})" if r == 0 else ""
                r_attr = red_attr if t["required"] else dim_attr
            else:
                if r == 0:
                    r_text = f" ● {t['key']}"
                    r_attr = green_attr
                else:
                    r_text = f"   Type: {t['type_str']}"
                    r_attr = dim_attr

            r_padded = f"│ {r_text:<{box_w - 4}} │"
            safe_addstr(win, curr_y, right_x, r_padded, border_attr)
            if r_text:
                safe_addstr(win, curr_y, right_x + 2, r_text[:box_w - 4], r_attr)

            # Center Flow Joints
            if not loaded_ds:
                msg = "(awaiting dataset)"
                rem_flow = flow_w - len(msg) - 8
                dash1 = max(1, rem_flow // 2)
                dash2 = max(1, rem_flow - dash1)
                c_flow = f" {'- ' * (dash1 // 2)}{msg}{' -' * (dash2 // 2)}► "
                safe_addstr(win, curr_y, flow_x, c_flow[:flow_w], dim_attr)
            elif not cols:
                c_flow = f" {'- ' * max(1, flow_w // 2 - 2)}► "
                safe_addstr(win, curr_y, flow_x, c_flow[:flow_w], red_attr if t["required"] else dim_attr)
            elif len(cols) == 1:
                bar_len = max(2, flow_w - 4)
                c_flow = f" {'─' * bar_len}► "
                safe_addstr(win, curr_y, flow_x, c_flow[:flow_w], green_attr)
            elif len(cols) >= 2:
                badge = "[ + Concat ]"
                lead = " ──┴──► "
                lead_len = len(lead)
                badge_len = len(badge)
                bar_len = max(2, flow_w - lead_len - badge_len - 3)
                tail = f" {'─' * bar_len}► "

                if b_h >= 2 and r == 0:
                    c_flow = " ──╮"
                    safe_addstr(win, curr_y, flow_x, c_flow, cyan_attr)
                else:
                    safe_addstr(win, curr_y, flow_x, lead, cyan_attr)
                    safe_addstr(win, curr_y, flow_x + lead_len, badge, cyan_attr | curses.A_STANDOUT)
                    safe_addstr(win, curr_y, flow_x + lead_len + badge_len, tail, green_attr)

            curr_y += 1

        # Divider joint between target blocks
        if curr_y < y + h - 1:
            if i < n - 1:
                next_t = targets[i + 1]
                left_div = fmt_border("├", "┤", "", box_w)
                right_div = fmt_border("├", "┤", f"Target: {next_t['key']}", box_w)
                safe_addstr(win, curr_y, left_x, left_div, header_box_attr)
                safe_addstr(win, curr_y, right_x, right_div, header_box_attr)
                curr_y += 1

    # Close Right Box
    if curr_y < y + h - 1:
        right_bot = fmt_border("└", "┘", "", box_w)
        safe_addstr(win, curr_y, right_x, right_bot, border_attr)

    # Unmapped columns section
    if show_unmapped and curr_y < y + h - 2:
        left_unmap_div = fmt_border("├", "┤", "Unmapped Columns", box_w)
        safe_addstr(win, curr_y, left_x, left_unmap_div, header_box_attr)
        curr_y += 1

        if curr_y < y + h - 1:
            unmapped_str = " · " + ", ".join(unmapped)
            avail_unmap = box_w - 4
            if len(unmapped_str) > avail_unmap:
                unmapped_str = unmapped_str[:avail_unmap - 1] + "…"
            l_unmap_padded = f"│ {unmapped_str:<{box_w - 4}} │"
            safe_addstr(win, curr_y, left_x, l_unmap_padded, border_attr)
            safe_addstr(win, curr_y, left_x + 2, unmapped_str, yellow_attr | dim_attr)

            unmap_flow = " - - - (ignored / excluded)"
            safe_addstr(win, curr_y, flow_x, unmap_flow[:flow_w], dim_attr)
            curr_y += 1

    # Close Left Box
    if curr_y < y + h:
        left_bot = fmt_border("└", "┘", "", box_w)
        safe_addstr(win, curr_y, left_x, left_bot, border_attr)


def draw_queue_table(
    win: curses.window,
    y: int,
    x: int,
    h: int,
    w: int,
    runs: List[Any],
    selected_idx: int = 0,
    is_focused: bool = False,
    scroll_offset: int = 0,
) -> int:
    """
    Draw a scrollable table of queued LoRA runs where every run displays
    its full hyperparameter configuration specification directly underneath.
    Returns selected_idx.
    """
    if h < 2 or w < 30:
        return selected_idx

    header_y = y
    avail_rows = h - 1

    if not runs:
        safe_addstr(win, header_y, x + 2, "(Queue is empty. Press [F5] to run current configuration, or [F6] to schedule)", (get_color(COLOR_LABEL_GRAY) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)
        return selected_idx

    # Use 2-row stride so every run displays its config spec underneath
    row_stride = 2 if avail_rows >= 4 else 1
    avail_items = max(1, avail_rows // row_stride)

    # Adjust scroll offset
    if selected_idx < scroll_offset:
        scroll_offset = selected_idx
    elif selected_idx >= scroll_offset + avail_items:
        scroll_offset = selected_idx - avail_items + 1

    for r_i in range(avail_items):
        idx = scroll_offset + r_i
        row_y = header_y + r_i * row_stride
        if idx >= len(runs):
            break

        run = runs[idx]
        is_sel = (idx == selected_idx)
        prefix = "▶ " if is_sel else "  "

        idx_str = f"{idx + 1}."
        name_str = getattr(run, "name", "") or getattr(run, "model", "").split("/")[-1]
        
        status_raw = getattr(run, "status", "queued").lower()
        if status_raw == "running":
            st_text = "[RUNNING]"
            st_attr = (get_color(COLOR_TITLE_ACCENT) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
        elif status_raw == "completed":
            st_text = "[DONE]"
            st_attr = (get_color(COLOR_SUCCESS) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
        elif status_raw == "failed":
            st_text = "[FAILED]"
            st_attr = (get_color(COLOR_ERROR) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
        else:
            st_text = "[QUEUED]"
            st_attr = (get_color(COLOR_BORDER_JOINTS) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD

        # Header line for this run: Index + Name + Status
        status_len = len(st_text)
        max_name_w = max(10, w - len(prefix) - len(idx_str) - status_len - 4)
        if len(name_str) > max_name_w:
            name_disp = name_str[:max_name_w - 1] + "…"
        else:
            name_disp = name_str

        pad_w = max(1, w - len(prefix) - len(idx_str) - 1 - len(name_disp) - status_len - 2)
        row_line = f"{prefix}{idx_str} {name_disp}{' ' * pad_w}{st_text}"

        if is_sel and is_focused:
            attr_line1 = (get_color(COLOR_INPUT_FOCUSED) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
            safe_addstr(win, row_y, x, row_line[:w], attr_line1)
        elif is_sel:
            attr_line1 = (get_color(COLOR_BORDER_JOINTS) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
            safe_addstr(win, row_y, x, row_line[:w], attr_line1)
            # Accentuate status badge
            st_pos = x + len(row_line) - status_len
            if st_pos < x + w:
                safe_addstr(win, row_y, st_pos, st_text[:x + w - st_pos], st_attr)
        else:
            attr_line1 = get_color(COLOR_NORMAL_TEXT) if safe_has_colors() else 0
            safe_addstr(win, row_y, x, row_line[:w], attr_line1)
            st_pos = x + len(row_line) - status_len
            if st_pos < x + w:
                safe_addstr(win, row_y, st_pos, st_text[:x + w - st_pos], st_attr)

        # Underneath line: Full Config Specification
        if row_stride == 2 and (row_y + 1) < (header_y + avail_rows):
            model_slug = getattr(run, "model", "").split("/")[-1]
            cfg_parts = []
            if getattr(run, "engine", "mlx_lm") == "mlx_vlm":
                cfg_parts.append("engine=mlx-vlm")
            cfg_parts.extend([
                f"model={model_slug}",
                f"iters={getattr(run, 'iters', 1000)}",
                f"batch={getattr(run, 'batch_size', 4)}",
                f"lr={getattr(run, 'learning_rate', 1e-5):g}",
                f"rank={getattr(run, 'lora_rank', 8)}",
                f"alpha={getattr(run, 'lora_alpha', 16.0):g}",
                f"layers={getattr(run, 'num_layers', 16)}",
                f"grad_chk={getattr(run, 'grad_checkpoint', True)}",
            ])
            wandb_u = getattr(run, "wandb_url", None)
            if wandb_u:
                cfg_parts.append(f"W&B={wandb_u}")

            spec_line = f"     Config: {' │ '.join(cfg_parts)}"

            if is_sel and is_focused:
                attr_line2 = (get_color(COLOR_TITLE_ACCENT) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
            elif is_sel:
                attr_line2 = (get_color(COLOR_NORMAL_TEXT) | curses.A_DIM) if safe_has_colors() else curses.A_DIM
            else:
                attr_line2 = (get_color(COLOR_LABEL_GRAY) | curses.A_DIM) if safe_has_colors() else curses.A_DIM

            safe_addstr(win, row_y + 1, x, spec_line[:w], attr_line2)

    # Visual scroll indicators for Queue panel
    if scroll_offset > 0:
        safe_addstr(win, header_y, x + w - 2, "▲", (get_color(COLOR_TITLE_ACCENT) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
    if scroll_offset + avail_items < len(runs):
        safe_addstr(win, header_y + h - 1, x + w - 2, "▼", (get_color(COLOR_TITLE_ACCENT) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)

    return selected_idx


def draw_sweep_grid(
    win: curses.window,
    y: int,
    x: int,
    h: int,
    w: int,
    varying_params: List[Tuple[str, str, List[Any]]],
    total_runs_count: int,
) -> None:
    """
    Draw Cartesian multi-run sweep grid matching user visual specification:
    - Column headers for each parameter
    - Vertically stacked box cells with internal dividers
    - Centered multiplication cross '✖' between columns
    - Centered total training runs footer at bottom
    """
    if h < 3 or w < 20 or not varying_params:
        return

    border_attr = get_color(COLOR_BORDER_JOINTS) if safe_has_colors() else 0
    hdr_attr = (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    val_attr = (get_color(COLOR_NORMAL_TEXT) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    cross_attr = (get_color(COLOR_TITLE_ACCENT) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    footer_attr = (get_color(COLOR_NORMAL_TEXT) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD

    # Format values for each column
    formatted_cols: List[Tuple[str, List[str]]] = []
    for field_name, label, vals in varying_params:
        val_strs: List[str] = []
        for v in vals:
            if isinstance(v, float):
                val_strs.append(f"{v:g}")
            elif isinstance(v, bool):
                val_strs.append("True" if v else "False")
            else:
                val_strs.append(str(v))
        formatted_cols.append((label, val_strs))

    # Calculate column widths
    col_widths: List[int] = []
    for label, val_strs in formatted_cols:
        max_str_len = max([len(label)] + [len(s) for s in val_strs]) if val_strs else len(label)
        cw = max(14, max_str_len + 4)
        col_widths.append(cw)

    # Gap between columns (containing the multiplication cross '✖')
    gap = 6
    total_grid_w = sum(col_widths) + (len(col_widths) - 1) * gap

    # Center horizontally within available width w
    if total_grid_w < w - 2:
        start_grid_x = x + (w - total_grid_w) // 2
    else:
        start_grid_x = x + 2

    # Draw columns and crosses
    curr_col_x = start_grid_x
    max_bottom_y = y

    for c_idx, (label, val_strs) in enumerate(formatted_cols):
        cw = col_widths[c_idx]

        # 1. Header centered over column
        hdr_pad = max(0, (cw - len(label)) // 2)
        safe_addstr(win, y, curr_col_x + hdr_pad, label, hdr_attr)

        # 2. Box Top border
        box_top_y = y + 1
        safe_addstr(win, box_top_y, curr_col_x, "┌" + "─" * (cw - 2) + "┐", border_attr)

        # 3. Stacked value cells
        cur_y = box_top_y + 1
        for v_idx, vs in enumerate(val_strs):
            if cur_y >= y + h - 2:
                break
            v_pad_l = max(0, (cw - 2 - len(vs)) // 2)
            v_pad_r = max(0, cw - 2 - len(vs) - v_pad_l)
            safe_addstr(win, cur_y, curr_col_x, "│", border_attr)
            safe_addstr(win, cur_y, curr_col_x + 1, " " * v_pad_l + vs + " " * v_pad_r, val_attr)
            safe_addstr(win, cur_y, curr_col_x + cw - 1, "│", border_attr)
            cur_y += 1

            # Cell divider (between cells)
            if v_idx < len(val_strs) - 1 and cur_y < y + h - 2:
                safe_addstr(win, cur_y, curr_col_x, "├" + "─" * (cw - 2) + "┤", border_attr)
                cur_y += 1

        # 4. Box Bottom border
        if cur_y < y + h - 1:
            safe_addstr(win, cur_y, curr_col_x, "└" + "─" * (cw - 2) + "┘", border_attr)
            cur_y += 1

        if cur_y > max_bottom_y:
            max_bottom_y = cur_y

        # 5. Multiplication cross '✖' between columns
        if c_idx < len(formatted_cols) - 1:
            cross_x = curr_col_x + cw + (gap - 1) // 2
            cross_y = box_top_y + 1
            safe_addstr(win, cross_y, cross_x, "✖", cross_attr)

        curr_col_x += cw + gap

    # 6. Centered footer: "<N> training runs"
    runs_txt = f"{total_runs_count} training run" if total_runs_count == 1 else f"{total_runs_count} training runs"
    footer_y = min(y + h - 1, max(max_bottom_y + 1, y + 4))
    footer_x = x + max(0, (w - len(runs_txt)) // 2)
    safe_addstr(win, footer_y, footer_x, runs_txt, footer_attr)


def show_model_picker_dialog(
    stdscr: curses.window,
    current_model: str = "",
) -> Optional[str]:
    """
    Modal dialog to select a base model already saved on drive.
    Allows browsing via native macOS Finder or entering a local model path.
    """
    from mlx_commander.lora.model_info import scan_local_models
    from mlx_commander.gui_picker import pick_model_gui

    configure_escdelay(25)
    options: List[Tuple[str, str]] = [
        ("[ 📁 Browse Drive with Finder... ]", "Select model folder (or any file inside it) via Finder"),
        ("[ ✍ Enter Local Model Path Manually... ]", "Enter path to model directory or weights"),
    ]

    discovered = scan_local_models()
    for m in discovered:
        m_label = f"• {m.name}"
        m_desc = f"{m.architecture} ({m.file_size_gb:.1f} GB) - {m.path}"
        options.append((m_label, m_desc))

    sel_idx = 0
    if current_model:
        for i, (m_id, _) in enumerate(options):
            if current_model in m_id or m_id.endswith(current_model):
                sel_idx = i
                break

    max_y, max_x = stdscr.getmaxyx()
    h = min(len(options) + 6, max_y - 4, 16)
    w = min(max_x - 4, 76)
    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    safe_curs_set(0)

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", (get_color(COLOR_BORDER_JOINTS) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, start_y, start_x + 2, " Select Local Base Model on Drive ", (get_color(COLOR_NORMAL_TEXT) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", get_color(COLOR_BORDER_JOINTS) if safe_has_colors() else 0)
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", (get_color(COLOR_BORDER_JOINTS) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)

        visible_rows = h - 4
        scroll_offset = max(0, sel_idx - visible_rows // 2)
        if sel_idx < scroll_offset:
            scroll_offset = sel_idx
        elif sel_idx >= scroll_offset + visible_rows:
            scroll_offset = sel_idx - visible_rows + 1

        for r in range(visible_rows):
            opt_idx = scroll_offset + r
            row_y = start_y + 1 + r
            if opt_idx < len(options):
                m_id, desc = options[opt_idx]
                is_focused = (opt_idx == sel_idx)
                prefix = " ▶ " if is_focused else "   "
                short_m = m_id if len(m_id) <= 42 else m_id[:39] + "…"
                line_str = f"{prefix}{short_m:<42} {desc}"[: w - 4]
                if is_focused:
                    attr = (get_color(COLOR_INPUT_FOCUSED) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
                else:
                    attr = get_color(COLOR_BORDER_JOINTS) if safe_has_colors() else 0
                safe_addstr(stdscr, row_y, start_x + 1, line_str, attr)

        safe_addstr(stdscr, start_y + h - 2, start_x + 3, "[Enter/Space] Select   [Esc] Cancel", (get_color(COLOR_LABEL_GRAY) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (curses.KEY_UP, curses.KEY_LEFT, ord("k"), ord("h")):
            sel_idx = max(0, sel_idx - 1)
        elif k in (curses.KEY_DOWN, curses.KEY_RIGHT, ord("j"), ord("l")):
            sel_idx = min(len(options) - 1, sel_idx + 1)
        elif k in (27, ord("q"), ord("Q")):
            return None
        elif k in (10, 13, 32):  # Enter or Space
            from mlx_commander.lora.model_info import normalize_model_path
            if sel_idx == 0:  # Browse Drive with Finder
                curses.def_prog_mode()
                curses.endwin()
                chosen = pick_model_gui(prompt="Select Local Base Model Folder (or any file inside it)", default_dir=current_model or os.getcwd())
                curses.reset_prog_mode()
                stdscr.refresh()
                if chosen and chosen.strip():
                    return normalize_model_path(chosen.strip())
                # If Finder was cancelled or not supported, continue dialog
                continue
            elif sel_idx == 1:  # Enter Local Model Path Manually
                val = show_text_edit_dialog(
                    stdscr,
                    "Local Model Path",
                    "Enter path to local model folder or config.json:",
                    default_val=current_model,
                )
                if val and val.strip():
                    return normalize_model_path(val.strip())
                return None
            else:
                # Discovered model selected
                disc_idx = sel_idx - 2
                if 0 <= disc_idx < len(discovered):
                    return normalize_model_path(discovered[disc_idx].path)
                return normalize_model_path(options[sel_idx][0])


def show_dataset_picker_dialog(
    stdscr: curses.window,
    current_dataset: str = "",
) -> Optional[str]:
    """
    Modal dialog to select a local dataset directory on drive.
    Allows browsing via native macOS Finder or entering a local dataset path.
    """
    from mlx_commander.gui_picker import pick_folder_gui

    configure_escdelay(25)
    options: List[Tuple[str, str]] = [
        ("[ 📁 Browse Drive with Finder... ]", "Select dataset folder containing train.jsonl via Finder"),
        ("[ ✍ Enter Local Dataset Path Manually... ]", "Enter path to directory with train.jsonl / valid.jsonl"),
    ]

    # Discover candidate local datasets
    candidates: List[Path] = []
    cwd = Path.cwd()
    for p in [cwd / "mlx_dataset", cwd]:
        if p.exists() and (p / "train.jsonl").exists() and p not in candidates:
            candidates.append(p)
    try:
        for sub in sorted(cwd.iterdir()):
            if sub.is_dir() and sub not in candidates and (sub / "train.jsonl").exists():
                candidates.append(sub)
    except Exception:
        pass

    for c in candidates:
        rec_cnt = 0
        try:
            with open(c / "train.jsonl", "rb") as f:
                rec_cnt = sum(1 for _ in f)
        except Exception:
            pass
        options.append((f"• {c.name}", f"{rec_cnt:,} train records - {c}"))

    sel_idx = 0
    if current_dataset:
        for i, (d_id, d_desc) in enumerate(options):
            if current_dataset in d_id or current_dataset in d_desc or d_id.endswith(current_dataset):
                sel_idx = i
                break

    max_y, max_x = stdscr.getmaxyx()
    h = min(len(options) + 6, max_y - 4, 16)
    w = min(max_x - 4, 76)
    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    safe_curs_set(0)

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", (get_color(COLOR_BORDER_JOINTS) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, start_y, start_x + 2, " Select Local Dataset Directory on Drive ", (get_color(COLOR_NORMAL_TEXT) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", get_color(COLOR_BORDER_JOINTS) if safe_has_colors() else 0)
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", (get_color(COLOR_BORDER_JOINTS) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)

        visible_rows = h - 4
        scroll_offset = max(0, sel_idx - visible_rows // 2)
        if sel_idx < scroll_offset:
            scroll_offset = sel_idx
        elif sel_idx >= scroll_offset + visible_rows:
            scroll_offset = sel_idx - visible_rows + 1

        for r in range(visible_rows):
            opt_idx = scroll_offset + r
            row_y = start_y + 1 + r
            if opt_idx < len(options):
                d_id, desc = options[opt_idx]
                is_focused = (opt_idx == sel_idx)
                prefix = " ▶ " if is_focused else "   "
                short_d = d_id if len(d_id) <= 42 else d_id[:39] + "…"
                line_str = f"{prefix}{short_d:<42} {desc}"[: w - 4]
                if is_focused:
                    attr = (get_color(COLOR_INPUT_FOCUSED) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
                else:
                    attr = get_color(COLOR_BORDER_JOINTS) if safe_has_colors() else 0
                safe_addstr(stdscr, row_y, start_x + 1, line_str, attr)

        safe_addstr(stdscr, start_y + h - 2, start_x + 3, "[Enter/Space] Select   [Esc] Cancel", (get_color(COLOR_LABEL_GRAY) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (curses.KEY_UP, curses.KEY_LEFT, ord("k"), ord("h")):
            sel_idx = max(0, sel_idx - 1)
        elif k in (curses.KEY_DOWN, curses.KEY_RIGHT, ord("j"), ord("l")):
            sel_idx = min(len(options) - 1, sel_idx + 1)
        elif k in (27, ord("q"), ord("Q")):
            return None
        elif k in (10, 13, 32):  # Enter or Space
            if sel_idx == 0:  # Browse Drive with Finder
                curses.def_prog_mode()
                curses.endwin()
                chosen = pick_folder_gui(prompt="Select Dataset Folder", default_dir=current_dataset or os.getcwd())
                curses.reset_prog_mode()
                stdscr.refresh()
                if chosen and chosen.strip():
                    return chosen.strip()
                continue
            elif sel_idx == 1:  # Enter Local Dataset Path Manually
                val = show_text_edit_dialog(
                    stdscr,
                    "Dataset Path",
                    "Enter directory containing train.jsonl / valid.jsonl:",
                    default_val=current_dataset,
                )
                if val and val.strip():
                    return val.strip()
                return None
            else:
                disc_idx = sel_idx - 2
                if 0 <= disc_idx < len(candidates):
                    return str(candidates[disc_idx])
                return options[sel_idx][0]


def show_dataset_source_dialog(
    stdscr: curses.window,
    current_dataset: str = "",
) -> Optional[str]:
    """
    Modal dialog to select a dataset source for conversion (Mode 1).
    Allows browsing via native macOS Finder, entering a local path or Hugging Face ID manually,
    or selecting a discovered local dataset file/directory.
    """
    from mlx_commander.gui_picker import pick_dataset_gui

    configure_escdelay(25)
    options: List[Tuple[str, str]] = [
        ("[ 📁 Browse Drive with Finder... ]", "Select dataset file(s) or folder via Finder"),
        ("[ ✍ Enter Dataset Path or HF ID Manually... ]", "Enter local path(s) or Hugging Face repo ID"),
    ]

    # Discover candidate local datasets in cwd
    candidates: List[Path] = []
    cwd = Path.cwd()
    data_exts = {".parquet", ".jsonl", ".arrow", ".csv", ".tsv", ".json", ".sqlite", ".tar"}
    try:
        for p in sorted(cwd.iterdir()):
            if p.name.startswith("."):
                continue
            if p.is_file() and p.suffix.lower() in data_exts:
                candidates.append(p)
            elif p.is_dir() and p.name in ("mlx_dataset", "data", "dataset", "datasets"):
                candidates.append(p)
    except Exception:
        pass

    for c in candidates[:8]:
        if c.is_file():
            sz_mb = c.stat().st_size / (1024 * 1024)
            options.append((f"• {c.name}", f"Local file ({sz_mb:.1f} MB) - {c}"))
        else:
            options.append((f"• {c.name}/", f"Local directory - {c}"))

    sel_idx = 0
    if current_dataset:
        for i, (d_id, d_desc) in enumerate(options):
            if current_dataset in d_id or current_dataset in d_desc or d_id.endswith(current_dataset):
                sel_idx = i
                break

    max_y, max_x = stdscr.getmaxyx()
    h = min(len(options) + 6, max_y - 4, 16)
    w = min(max_x - 4, 76)
    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    safe_curs_set(0)

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", (get_color(COLOR_BORDER_JOINTS) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, start_y, start_x + 2, " Select Dataset Source ", (get_color(COLOR_NORMAL_TEXT) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", get_color(COLOR_BORDER_JOINTS) if safe_has_colors() else 0)
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", (get_color(COLOR_BORDER_JOINTS) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)

        visible_rows = h - 4
        scroll_offset = max(0, sel_idx - visible_rows // 2)
        if sel_idx < scroll_offset:
            scroll_offset = sel_idx
        elif sel_idx >= scroll_offset + visible_rows:
            scroll_offset = sel_idx - visible_rows + 1

        for r in range(visible_rows):
            opt_idx = scroll_offset + r
            row_y = start_y + 1 + r
            if opt_idx < len(options):
                d_id, desc = options[opt_idx]
                is_focused = (opt_idx == sel_idx)
                prefix = " ▶ " if is_focused else "   "
                short_d = d_id if len(d_id) <= 42 else d_id[:39] + "…"
                line_str = f"{prefix}{short_d:<42} {desc}"[: w - 4]
                if is_focused:
                    attr = (get_color(COLOR_INPUT_FOCUSED) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
                else:
                    attr = get_color(COLOR_BORDER_JOINTS) if safe_has_colors() else 0
                safe_addstr(stdscr, row_y, start_x + 1, line_str, attr)

        safe_addstr(stdscr, start_y + h - 2, start_x + 3, "[Enter/Space] Select   [Esc] Cancel", (get_color(COLOR_LABEL_GRAY) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (curses.KEY_UP, curses.KEY_LEFT, ord("k"), ord("h")):
            sel_idx = max(0, sel_idx - 1)
        elif k in (curses.KEY_DOWN, curses.KEY_RIGHT, ord("j"), ord("l")):
            sel_idx = min(len(options) - 1, sel_idx + 1)
        elif k in (27, ord("q"), ord("Q")):
            return None
        elif k in (10, 13, 32):  # Enter or Space
            if sel_idx == 0:  # Browse Drive with Finder
                curses.def_prog_mode()
                curses.endwin()
                chosen = pick_dataset_gui(prompt="Select Dataset (File, Files, or Folder)", default_dir=current_dataset or os.getcwd())
                curses.reset_prog_mode()
                stdscr.refresh()
                if chosen and chosen.strip():
                    return chosen.strip()
                continue
            elif sel_idx == 1:  # Enter Dataset Path or HF ID Manually
                val = show_text_edit_dialog(
                    stdscr,
                    "Dataset Path or HF ID",
                    "Enter local dataset path(s) or Hugging Face repo ID:",
                    default_val=current_dataset or os.getcwd(),
                )
                if val and val.strip():
                    return val.strip()
                return None
            else:
                disc_idx = sel_idx - 2
                if 0 <= disc_idx < len(candidates):
                    return str(candidates[disc_idx])
                return options[sel_idx][0]


def show_choice_dialog(
    stdscr: curses.window,
    title: str,
    prompt: str,
    choices: List[str],
    current_val: Optional[str] = None,
) -> Optional[str]:
    """
    Modal dialog to select a single choice from a list (e.g. optimizer, fine_tune_type).
    """
    configure_escdelay(25)
    sel_idx = 0
    if current_val and current_val in choices:
        sel_idx = choices.index(current_val)

    max_y, max_x = stdscr.getmaxyx()
    h = min(len(choices) + 6, max_y - 4, 18)
    longest_choice = max((len(c) for c in choices), default=30)
    w = min(max_x - 4, max(56, longest_choice + 12))
    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    safe_curs_set(0)

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", (get_color(COLOR_BORDER_JOINTS) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, start_y, start_x + 2, f" {title} ", (get_color(COLOR_NORMAL_TEXT) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", get_color(COLOR_BORDER_JOINTS) if safe_has_colors() else 0)
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", (get_color(COLOR_BORDER_JOINTS) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)

        safe_addstr(stdscr, start_y + 1, start_x + 3, prompt[:w - 6], (get_color(COLOR_LABEL_GRAY) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)

        for i, c in enumerate(choices):
            row_y = start_y + 3 + i
            if row_y >= start_y + h - 2:
                break
            is_focused = (i == sel_idx)
            is_curr = (c == current_val)
            prefix = " ▶ " if is_focused else "   "
            bullet = "(*)" if is_curr else "( )"
            line_str = f"{prefix}{bullet} {c}"[: w - 4]
            if is_focused:
                attr = (get_color(COLOR_INPUT_FOCUSED) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
            else:
                attr = get_color(COLOR_BORDER_JOINTS) if safe_has_colors() else 0
            safe_addstr(stdscr, row_y, start_x + 2, line_str, attr)

        safe_addstr(stdscr, start_y + h - 2, start_x + 3, "[Enter] Select   [Esc] Cancel", (get_color(COLOR_LABEL_GRAY) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (curses.KEY_UP, curses.KEY_LEFT, ord("k"), ord("h")):
            sel_idx = (sel_idx - 1) % len(choices)
        elif k in (curses.KEY_DOWN, curses.KEY_RIGHT, ord("j"), ord("l")):
            sel_idx = (sel_idx + 1) % len(choices)
        elif k in (27, ord("q"), ord("Q")):
            return None
        elif k in (10, 13, 32):  # Enter or Space
            return choices[sel_idx]


def show_label_mapping_dialog(
    stdscr: curses.window,
    col_name: str,
    discrete_info: Optional[Dict[str, Any]],
    current_map: Optional[Dict[str, str]],
) -> Optional[Dict[str, str]]:
    """
    Modal dialog allowing users to define semantic rewriting for discrete class labels.
    Displays discrete values (e.g. 0, 1, 2) alongside their mapped semantic strings
    (e.g. negative, neutral, positive).
    Allows navigating rows, editing values, applying, or clearing to raw.
    """
    from mlx_commander.formats import parse_label_map

    configure_escdelay(25)
    working_map: Dict[str, str] = dict(current_map) if current_map else {}

    raw_keys: List[str] = []
    suggested = None
    if discrete_info:
        if discrete_info.get("unique_values"):
            raw_keys.extend(str(v) for v in discrete_info["unique_values"])
        suggested = discrete_info.get("suggested_map")

    for k in working_map.keys():
        if k not in raw_keys:
            raw_keys.append(k)

    if not raw_keys:
        raw_keys = ["0", "1", "2"]

    if not working_map and suggested:
        working_map = dict(suggested)

    total_actions = len(raw_keys) + 3
    sel_idx = 0

    max_y, max_x = stdscr.getmaxyx()
    h = min(len(raw_keys) + 8, max_y - 4, 18)
    w = min(max_x - 4, 62)
    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    safe_curs_set(0)

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", (get_color(COLOR_BORDER_JOINTS) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, start_y, start_x + 2, f" Semantic Label Mapping: '{col_name}' ", (get_color(COLOR_NORMAL_TEXT) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", get_color(COLOR_BORDER_JOINTS) if safe_has_colors() else 0)
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", (get_color(COLOR_BORDER_JOINTS) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)

        safe_addstr(stdscr, start_y + 1, start_x + 3, "Map discrete values to semantic labels (Enter to edit):"[:w - 6], (get_color(COLOR_LABEL_GRAY) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)

        list_start_y = start_y + 2
        for i, k in enumerate(raw_keys):
            row_y = list_start_y + i
            if row_y >= start_y + h - 3:
                break
            is_foc = (sel_idx == i)
            val = working_map.get(k, "")
            val_disp = f"'{val}'" if val else "(keep raw)"
            prefix = " ▶ " if is_foc else "   "
            row_str = f"{prefix}[{k}] ──▶ {val_disp}"[: w - 6]
            attr = (get_color(COLOR_INPUT_FOCUSED) | curses.A_BOLD) if (is_foc and safe_has_colors()) else (curses.A_STANDOUT if is_foc else (get_color(COLOR_NORMAL_TEXT) if safe_has_colors() else 0))
            safe_addstr(stdscr, row_y, start_x + 3, row_str, attr)

        btn_y = start_y + h - 3
        save_foc = (sel_idx == len(raw_keys))
        clear_foc = (sel_idx == len(raw_keys) + 1)
        cancel_foc = (sel_idx == len(raw_keys) + 2)

        save_attr = (get_color(COLOR_SUCCESS) | curses.A_STANDOUT) if save_foc else ((get_color(COLOR_SUCCESS) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        clear_attr = (get_color(COLOR_ERROR) | curses.A_STANDOUT) if clear_foc else ((get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if safe_has_colors() else curses.A_DIM)
        cancel_attr = (get_color(COLOR_NORMAL_TEXT) | curses.A_STANDOUT) if cancel_foc else (get_color(COLOR_NORMAL_TEXT) if safe_has_colors() else 0)

        safe_addstr(stdscr, btn_y, start_x + 4, "[ Save & Apply (s) ]", save_attr)
        safe_addstr(stdscr, btn_y, start_x + 27, "[ Clear / Raw (c) ]", clear_attr)
        safe_addstr(stdscr, btn_y, start_x + 49, "[ Cancel ]", cancel_attr)

        safe_addstr(stdscr, start_y + h - 2, start_x + 3, "[Enter] Edit/Select   [s] Save   [c] Clear   [Esc] Cancel"[:w - 6], (get_color(COLOR_LABEL_GRAY) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)
        stdscr.refresh()

        ch = stdscr.getch()
        if ch in (curses.KEY_UP, ord("k")):
            sel_idx = (sel_idx - 1) % total_actions
        elif ch in (curses.KEY_DOWN, ord("j")):
            sel_idx = (sel_idx + 1) % total_actions
        elif ch in (27, ord("q"), ord("Q")):
            return current_map
        elif ch in (ord("s"), ord("S")):
            filtered = {k: v.strip() for k, v in working_map.items() if v and v.strip()}
            return filtered if filtered else None
        elif ch in (ord("c"), ord("C")):
            return None
        elif ch in (10, 13, 32):  # Enter or Space
            if sel_idx < len(raw_keys):
                target_key = raw_keys[sel_idx]
                curr_val = working_map.get(target_key, "")
                new_val = show_text_edit_dialog(
                    stdscr,
                    f"Label Mapping for '{target_key}'",
                    f"Enter semantic replacement text for '{target_key}' (leave blank to keep raw):",
                    curr_val,
                )
                if new_val is not None:
                    trimmed = new_val.strip()
                    if trimmed:
                        working_map[target_key] = trimmed
                    elif target_key in working_map:
                        del working_map[target_key]
            elif sel_idx == len(raw_keys):  # Save & Apply
                filtered = {k: v.strip() for k, v in working_map.items() if v and v.strip()}
                return filtered if filtered else None
            elif sel_idx == len(raw_keys) + 1:  # Clear / Raw
                return None
            elif sel_idx == len(raw_keys) + 2:  # Cancel
                return current_map

