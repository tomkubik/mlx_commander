"""
Terminal Spawner for MLX Commander.
Enables AI agents (Antigravity, Claude Desktop, Cursor, MCP clients) and non-interactive
subshells to spawn an interactive macOS Terminal window running the curses TUI,
awaiting completion via a file-based status handshake.
"""

import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple


def ensure_adequate_terminal_size(min_cols: int = 120, min_lines: int = 38) -> Tuple[int, int]:
    """
    Ensure the terminal window is at least min_cols wide and min_lines tall so that
    all interface elements (hyperparameters, runtime estimates, and queued runs)
    are fully visible without vertical clipping or horizontal truncation.

    Only increases size if current window is smaller than minimums; never shrinks.
    Uses:
      1. ANSI/VT100 escape sequence: \\033[8;{lines};{cols}t
      2. macOS AppleScript (Terminal.app and iTerm2)
    """
    if not is_terminal_interactive():
        try:
            return shutil.get_terminal_size((min_cols, min_lines))
        except Exception:
            return min_cols, min_lines
    try:
        cur_cols, cur_lines = shutil.get_terminal_size((80, 24))
    except Exception:
        cur_cols, cur_lines = 80, 24

    target_cols = max(cur_cols, min_cols)
    target_lines = max(cur_lines, min_lines)

    if target_cols <= cur_cols and target_lines <= cur_lines:
        return cur_cols, cur_lines

    # 1. Output ANSI escape code to stdout and /dev/tty
    try:
        sys.stdout.write(f"\033[8;{target_lines};{target_cols}t")
        sys.stdout.flush()
    except Exception:
        pass

    try:
        if os.path.exists("/dev/tty"):
            with open("/dev/tty", "w") as tty:
                tty.write(f"\033[8;{target_lines};{target_cols}t")
                tty.flush()
    except Exception:
        pass

    # 2. On macOS, actively resize Terminal.app or iTerm2 via AppleScript
    if is_macos():
        term_prog = os.environ.get("TERM_PROGRAM", "")
        script = None
        if term_prog == "Apple_Terminal" or not term_prog:
            script = f'''
            tell application "Terminal"
                if (count of windows) > 0 then
                    set w to front window
                    if number of columns of w < {target_cols} then
                        set number of columns of w to {target_cols}
                    end if
                    if number of rows of w < {target_lines} then
                        set number of rows of w to {target_lines}
                    end if
                end if
            end tell
            '''
        elif term_prog == "iTerm.app":
            script = f'''
            tell application "iTerm"
                if (count of windows) > 0 then
                    tell current session of current window
                        if columns < {target_cols} then
                            set columns to {target_cols}
                        end if
                        if rows < {target_lines} then
                            set rows to {target_lines}
                        end if
                    end tell
                end if
            end tell
            '''

        if script:
            try:
                subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=0.6,
                )
            except Exception:
                pass

    try:
        time.sleep(0.06)
    except Exception:
        pass

    return target_cols, target_lines


def is_macos() -> bool:
    """Return True if running on macOS (Darwin)."""
    return platform.system() == "Darwin"


def is_terminal_interactive() -> bool:
    """Check if current process has an interactive TTY with adequate terminfo."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "")
    if not term or term == "dumb":
        return False
    return True


def build_terminal_script(
    python_exe: str,
    args: List[str],
    status_file: str,
    working_dir: str,
) -> str:
    """
    Construct the shell script executed inside the spawned macOS Terminal window.
    Ensures exit status and completion are safely written to status_file.
    """
    clean_args = [a for a in args if a not in ("--spawn-terminal", "--external-terminal")]
    if "--tui" not in clean_args and "--commander" not in clean_args:
        clean_args.insert(0, "--tui")

    cmd_parts = [shlex.quote(python_exe), "-m", "mlx_commander"] + [shlex.quote(a) for a in clean_args]
    exec_cmd = " ".join(cmd_parts)
    escaped_cwd = shlex.quote(working_dir)

    script_lines = [
        "#!/usr/bin/env bash",
        "printf '\\e[8;38;120t' 2>/dev/null || true",
        f"cd {escaped_cwd}",
        exec_cmd,
        "EC=$?",
        f'python3 -c "import json; json.dump({{\\"exit_code\\": $EC, \\"finished\\": True}}, open({json.dumps(status_file)}, \\"w\\"))"',
        "if [ $EC -eq 0 ]; then",
        '  echo "\n[OK] Conversion completed. Closing window in 2s..."; sleep 2; exit 0;',
        "else",
        '  echo "\nSession ended (code $EC). Closing window..."; sleep 1; exit $EC;',
        "fi",
    ]
    return "\n".join(script_lines) + "\n"


def spawn_terminal_tui(
    args: List[str],
    manifest_path: Optional[str] = None,
    timeout: Optional[float] = None,
    working_dir: Optional[str] = None,
) -> int:
    """
    Spawn the MLX Commander TUI in a dedicated macOS Terminal.app window
    and block synchronously until user completes conversion or exits.

    Returns:
        Exit code: 0 on successful conversion, 130 on cancel, 1 on error.
    """
    if not is_macos():
        raise RuntimeError("Terminal spawning via AppleScript is only supported on macOS.")

    cwd = working_dir or os.getcwd()
    run_id = uuid.uuid4().hex[:10]
    status_file = os.path.join(tempfile.gettempdir(), f"mlx_commander_status_{run_id}.json")
    script_file = os.path.join(tempfile.gettempdir(), f"mlx_commander_run_{run_id}.sh")

    # Forward manifest path if provided
    forward_args = list(args)
    if manifest_path and "--manifest-file" not in forward_args:
        forward_args.extend(["--manifest-file", str(manifest_path)])

    script_content = build_terminal_script(
        python_exe=sys.executable,
        args=forward_args,
        status_file=status_file,
        working_dir=cwd,
    )

    with open(script_file, "w", encoding="utf-8") as f:
        f.write(script_content)
    os.chmod(script_file, 0o755)

    applescript = (
        f'tell application "Terminal"\n'
        f'    activate\n'
        f'    do script "{script_file}"\n'
        f'    try\n'
        f'        set number of columns of front window to 120\n'
        f'        set number of rows of front window to 38\n'
        f'    end try\n'
        f'end tell\n'
    )

    try:
        proc = subprocess.run(
            ["osascript", "-e", applescript],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            sys.stderr.write(f"Failed to spawn Terminal via AppleScript: {proc.stderr}\n")
            return 1

        # Poll status file
        start_time = time.time()
        while True:
            if timeout and (time.time() - start_time) > timeout:
                sys.stderr.write("Timed out waiting for MLX Commander TUI session.\n")
                return 1

            if os.path.exists(status_file):
                try:
                    with open(status_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("finished"):
                        return int(data.get("exit_code", 0))
                except (json.JSONDecodeError, OSError):
                    pass

            time.sleep(0.25)
    except KeyboardInterrupt:
        sys.stderr.write("\nProcess interrupted.\n")
        return 130
    finally:
        for fpath in (status_file, script_file):
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass


def spawn_lora_queue_terminal(
    queue_dir: str,
    working_dir: Optional[str] = None,
    wandb_project: Optional[str] = None,
    enable_wandb: bool = True,
) -> bool:
    """
    Launch the LoRA queue runner in a dedicated detached macOS Terminal.app window.
    Runs python3 -m mlx_commander --run-queue <queue_dir> sequentially.
    Returns True if successfully spawned.
    """
    if not is_macos():
        return False

    cwd = working_dir or os.getcwd()
    run_id = uuid.uuid4().hex[:10]
    script_file = os.path.join(tempfile.gettempdir(), f"mlx_lora_runner_{run_id}.sh")
    escaped_cwd = shlex.quote(cwd)
    python_exe = sys.executable
    cmd_parts = [shlex.quote(python_exe), "-m", "mlx_commander", "--run-queue", shlex.quote(str(queue_dir))]
    if wandb_project:
        cmd_parts.extend(["--wandb-project", shlex.quote(wandb_project)])
    if not enable_wandb:
        cmd_parts.append("--no-wandb")
    exec_cmd = " ".join(cmd_parts)

    script_lines = [
        "#!/usr/bin/env bash",
        f"cd {escaped_cwd}",
        'echo "=========================================================="',
        'echo " Apple MLX LoRA Sequential Queue Runner"',
        'echo " Running detached in dedicated Terminal window"',
        'echo " You may close MLX Commander without interrupting runs"',
        'echo "=========================================================="',
        'echo ""',
        exec_cmd,
        'EC=$?',
        'echo ""',
        'if [ $EC -eq 0 ]; then',
        '  echo "[OK] All queue runs finished successfully."',
        'else',
        '  echo "[!] Queue finished with errors (code $EC)."',
        'fi',
        'echo "Press any key to close this terminal window..."',
        'read -n 1 -s',
        'exit $EC',
    ]
    with open(script_file, "w", encoding="utf-8") as f:
        f.write("\n".join(script_lines) + "\n")
    os.chmod(script_file, 0o755)

    applescript = (
        f'tell application "Terminal"\n'
        f'    activate\n'
        f'    do script "{script_file}"\n'
        f'end tell\n'
    )
    proc = subprocess.run(["osascript", "-e", applescript], capture_output=True, text=True, check=False)
    return proc.returncode == 0

