"""
GUI File and Folder Picker for macOS.
Compiles the native Cocoa NSOpenPanel helper on-the-fly at runtime into /tmp,
ensuring zero pre-compiled binaries exist in the project repository.
"""

import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


def is_macos() -> bool:
    return platform.system() == "Darwin"


def _clean_path(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return None
    p = path_str.strip()
    if p and os.path.exists(p):
        return str(Path(p).resolve())
    return p if p else None


def _compile_picker_at_runtime() -> Optional[str]:
    """
    Compile the native Cocoa picker at runtime into a temporary file.
    Guarantees zero pre-compiled binaries are shipped or stored in the repository.
    """
    pkg_dir = Path(__file__).resolve().parent
    src_path = pkg_dir / "bin" / "mac_picker.m"
    if not src_path.exists():
        return None

    uid = os.getuid() if hasattr(os, "getuid") else "user"
    tmp_bin = Path(tempfile.gettempdir()) / f"mlx_commander_picker_{uid}"

    # Use cached runtime build if source hasn't changed
    if tmp_bin.exists() and os.access(tmp_bin, os.X_OK):
        if tmp_bin.stat().st_mtime >= src_path.stat().st_mtime:
            return str(tmp_bin)

    try:
        res = subprocess.run(
            ["clang", "-O2", "-framework", "Cocoa", str(src_path), "-o", str(tmp_bin)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode == 0 and tmp_bin.exists():
            tmp_bin.chmod(0o755)
            return str(tmp_bin)
    except Exception:
        pass

    return None


def run_runtime_compiled_picker(mode: str = "both", default_dir: Optional[str] = None, prompt: Optional[str] = None) -> Optional[str]:
    """
    Execute the runtime-compiled Cocoa picker.
    Returns:
      - Selected path if user picked a file/folder and clicked Select.
      - None if user cancelled or closed the window (never opens a second window).
    """
    picker_bin = _compile_picker_at_runtime()
    if not picker_bin:
        return None

    dir_posix = str(Path(default_dir or os.getcwd()).resolve())
    cmd = [picker_bin, mode, dir_posix]
    if prompt:
        cmd.append(prompt)

    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            out = res.stdout.strip()
            if not out:
                return None
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            if not lines:
                return None
            if len(lines) == 1:
                return _clean_path(lines[0])
            cleaned = [_clean_path(l) for l in lines]
            valid = [c for c in cleaned if c]
            return "\n".join(valid) if valid else None
    except Exception:
        pass

    return None


def pick_dataset_gui(default_dir: Optional[str] = None, prompt: str = "Select Dataset (File, Files, or Folder)") -> Optional[str]:
    """
    Unified GUI dataset picker.
    Allows selecting a folder OR one or more dataset files (.parquet, .jsonl, .arrow, .csv, .tsv, .sqlite, .tar).
    Compiled dynamically at runtime.
    """
    if is_macos():
        # Primary: runtime-compiled native Cocoa open panel
        res = run_runtime_compiled_picker(mode="both", default_dir=default_dir, prompt=prompt)
        if res is not None:
            return res

        # Fallback to direct AppleScript only if runtime compilation failed
        if not _compile_picker_at_runtime():
            dir_posix = str(Path(default_dir or os.getcwd()).resolve())
            script = f'''
            try
                set thePaths to choose file with prompt "{prompt}" default location (POSIX file "{dir_posix}") with multiple selections allowed
                set posixList to ""
                repeat with aFile in thePaths
                    set posixList to posixList & (POSIX path of aFile) & linefeed
                end repeat
                return posixList
            on error
                try
                    set thePath to choose folder with prompt "{prompt}" default location (POSIX file "{dir_posix}")
                    return POSIX path of thePath
                on error
                    return ""
                end try
            end try
            '''
            try:
                sub_res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
                if sub_res.returncode == 0:
                    out = sub_res.stdout.strip()
                    lines = [l.strip() for l in out.splitlines() if l.strip()]
                    if len(lines) == 1:
                        return _clean_path(lines[0])
                    cleaned = [_clean_path(l) for l in lines]
                    valid = [c for c in cleaned if c]
                    return "\n".join(valid) if valid else None
            except Exception:
                pass

    return None


def pick_folder_gui(prompt: str = "Select Destination Folder", default_dir: Optional[str] = None) -> Optional[str]:
    """
    Unified GUI folder picker (e.g. for destination directories).
    Compiled dynamically at runtime.
    """
    if is_macos():
        res = run_runtime_compiled_picker(mode="folder", default_dir=default_dir, prompt=prompt)
        if res is not None:
            return res

        if not _compile_picker_at_runtime():
            dir_posix = str(Path(default_dir or os.getcwd()).resolve())
            script = f'''
            try
                set thePath to choose folder with prompt "{prompt}" default location (POSIX file "{dir_posix}")
                return POSIX path of thePath
            on error
                return ""
            end try
            '''
            try:
                sub_res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
                if sub_res.returncode == 0:
                    out = sub_res.stdout.strip()
                    return _clean_path(out) if out else None
            except Exception:
                pass

    return None


def pick_file_gui(prompt: str = "Select Dataset File", default_dir: Optional[str] = None) -> Optional[str]:
    """
    Unified GUI file picker (specifically for data files).
    Compiled dynamically at runtime.
    """
    if is_macos():
        res = run_runtime_compiled_picker(mode="file", default_dir=default_dir, prompt=prompt)
        if res is not None:
            return res

        if not _compile_picker_at_runtime():
            dir_posix = str(Path(default_dir or os.getcwd()).resolve())
            script = f'''
            try
                set thePath to choose file with prompt "{prompt}" default location (POSIX file "{dir_posix}")
                return POSIX path of thePath
            on error
                return ""
            end try
            '''
            try:
                sub_res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
                if sub_res.returncode == 0:
                    out = sub_res.stdout.strip()
                    return _clean_path(out) if out else None
            except Exception:
                pass

    return None


def pick_model_gui(prompt: str = "Select Local Base Model Folder (or any file inside it)", default_dir: Optional[str] = None) -> Optional[str]:
    """
    Unified GUI model picker.
    Allows selecting either a local model folder or a model weights/config file on drive.
    Automatically normalizes to the containing model directory.
    Compiled dynamically at runtime.
    """
    if is_macos():
        res = run_runtime_compiled_picker(mode="both", default_dir=default_dir, prompt=prompt)
        if res is not None:
            # If multiple files selected, return the first
            first = res.splitlines()[0] if "\n" in res else res
            cleaned = _clean_path(first)
            if cleaned:
                from mlx_commander.lora.model_info import normalize_model_path
                return normalize_model_path(cleaned)
            return None

        if not _compile_picker_at_runtime():
            dir_posix = str(Path(default_dir or os.getcwd()).resolve())
            script = f'''
            try
                set thePath to choose folder with prompt "{prompt}" default location (POSIX file "{dir_posix}")
                return POSIX path of thePath
            on error
                try
                    set thePath to choose file with prompt "{prompt}" default location (POSIX file "{dir_posix}")
                    return POSIX path of thePath
                on error
                    return ""
                end try
            end try
            '''
            try:
                sub_res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
                if sub_res.returncode == 0:
                    out = sub_res.stdout.strip()
                    cleaned = _clean_path(out) if out else None
                    if cleaned:
                        from mlx_commander.lora.model_info import normalize_model_path
                        return normalize_model_path(cleaned)
                    return None
            except Exception:
                pass

    return None

