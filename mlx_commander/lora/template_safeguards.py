"""
Behavioral Chat Template Probing and Auto-Healing Safeguards.

Probes model chat templates using synthetic `[system, user]` (inference) and
`[system, user, assistant]` (training) canary conversations before fine-tuning
or evaluation. Detects and repairs two classes of silent template failures without
hardcoding model names:

1. Train/Eval System-Role Asymmetry (e.g. Ministral 2410):
   Templates that guard system-prompt injection with `if loop.last and system_message is defined`
   include the system prompt during 2-turn inference (`[system, user]`, where `user` is
   `loop.last`), but silently drop it during 3-turn training (`[system, user, assistant]`,
   where `assistant` is `loop.last`). With `mask_prompt=True`, the 2-turn prompt offset is
   longer than the full 3-turn sequence (`offset > len(tokens)`), masking out 100% of tokens
   and producing `loss = NaN`.

2. Unmasked `<think>` Injection / Default Thinking Mode on Non-Thinking Datasets (e.g. Qwen 3):
   Templates that unconditionally inject `<think>\\n\\n</think>\\n\\n` before the assistant turn
   during training (`add_generation_prompt=False`), while omitting it during inference/prompt-masking
   (`add_generation_prompt=True`) unless `enable_thinking=False` is passed. On standard non-thinking
   datasets, this breaks prefix alignment (`P_train != P_eval`), forces LoRA to grade 4 boilerplate
   thinking tokens per example during training, and leaves thinking mode active during evaluation.
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

SYS_CANARY = "__MLX_SYS_CANARY_9f3a__"
USR_CANARY = "__MLX_USR_CANARY_7b2c__"
AST_CANARY = "__MLX_AST_CANARY_4e1d__"


def dataset_has_thinking_traces(
    data_path: Optional[Union[str, Path]],
    max_rows: int = 50,
) -> bool:
    """
    Inspect up to `max_rows` of a dataset to determine whether assistant targets
    intentionally contain reasoning blocks (`<think>...</think>`, `<|channel>thought`,
    or explicit `reasoning_content` / `reasoning` fields).
    """
    if not data_path:
        return False

    p = Path(data_path).expanduser()
    if not p.exists():
        return False

    candidates: List[Path] = []
    if p.is_file():
        candidates.append(p)
    elif p.is_dir():
        for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
            f = p / name
            if f.is_file():
                candidates.append(f)

    think_markers = ("<think>", "</think>", "<|channel>thought", "<channel|>")

    for file_path in candidates:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx >= max_rows:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue

                    if "messages" in row and isinstance(row["messages"], list):
                        for m in row["messages"]:
                            if not isinstance(m, dict):
                                continue
                            if m.get("reasoning_content") or m.get("reasoning"):
                                return True
                            if m.get("role") == "assistant":
                                content = str(m.get("content", ""))
                                if any(marker in content for marker in think_markers):
                                    return True
                    else:
                        for key in ("completion", "chosen", "text"):
                            val = str(row.get(key, ""))
                            if any(marker in val for marker in think_markers):
                                return True
        except OSError:
            continue

    return False


def _render_jinja_template(
    chat_template: str,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool,
    bos_token: str = "<s>",
    eos_token: str = "</s>",
    **extra_kwargs: Any,
) -> str:
    """Render a Hugging Face Jinja2 `chat_template` string in a sandboxed environment."""
    from jinja2 import Environment

    extensions = []
    try:
        from jinja2.ext import loopcontrols
        extensions.append(loopcontrols)
    except Exception:
        pass

    env = Environment(extensions=extensions)

    def _raise_exception(msg: str) -> None:
        raise ValueError(msg)

    env.globals["raise_exception"] = _raise_exception
    tmpl = env.from_string(chat_template)
    return tmpl.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        bos_token=bos_token,
        eos_token=eos_token,
        **extra_kwargs,
    )


def probe_chat_template(
    template_or_callable: Any,
    bos_token: str = "<s>",
    eos_token: str = "</s>",
) -> Dict[str, Any]:
    """
    Behaviorally probe a chat template (either a Jinja string, a tokenizer/processor
    with `apply_chat_template`, or a callable) using synthetic `[system, user]` and
    `[system, user, assistant]` canary messages.

    Returns a dictionary describing:
      - `status`: "ok", "broken", "raises", or "unavailable"
      - `raises_on_system`: True if the template raises on `role == "system"` (e.g. Gemma 2/3)
      - `sys_in_eval`: True if the system prompt appears in 2-turn eval (`[system, user]`)
      - `sys_in_train`: True if the system prompt appears in 3-turn train (`[system, user, assistant]`)
      - `prefix_match`: True if `P_train` (everything before the assistant answer in 3-turn train)
        matches `P_eval` (2-turn eval with `add_generation_prompt=True`) up to whitespace
      - `mask_offset_overflow`: True if `len(P_eval) > len(P_train)` (causes `loss=NaN` with `--mask-prompt`)
      - `unmasked_extra`: Substring injected between `P_eval` and the assistant answer in `P_train`
      - `injects_empty_think_in_train`: True if `<think>...</think>` is injected in `unmasked_extra`
      - `nothink_fixes_prefix`: True if `enable_thinking=False` restores `prefix_match`
    """
    eval_msgs = [
        {"role": "system", "content": SYS_CANARY},
        {"role": "user", "content": USR_CANARY},
    ]
    train_msgs = eval_msgs + [
        {"role": "assistant", "content": AST_CANARY},
    ]

    render_fn: Optional[Callable[..., str]] = None

    if isinstance(template_or_callable, str):
        try:
            import jinja2  # noqa: F401
        except ImportError:
            return {"status": "unavailable", "reason": "jinja2 not installed"}

        def _render_str(msgs: List[Dict[str, str]], add_gen: bool, **kw: Any) -> str:
            return _render_jinja_template(
                template_or_callable,
                messages=msgs,
                add_generation_prompt=add_gen,
                bos_token=bos_token,
                eos_token=eos_token,
                **kw,
            )

        render_fn = _render_str
    else:
        tok = getattr(template_or_callable, "tokenizer", None) or template_or_callable
        apply_fn = (
            getattr(template_or_callable, "apply_chat_template", None)
            or getattr(tok, "apply_chat_template", None)
            or (template_or_callable if callable(template_or_callable) else None)
        )
        if not callable(apply_fn):
            return {"status": "unavailable", "reason": "no apply_chat_template callable"}

        def _render_tok(msgs: List[Dict[str, str]], add_gen: bool, **kw: Any) -> str:
            try:
                return str(apply_fn(msgs, tokenize=False, add_generation_prompt=add_gen, **kw))
            except TypeError:
                return str(apply_fn(msgs, tokenize=False, add_generation_prompt=add_gen))

        render_fn = _render_tok

    try:
        p_eval = render_fn(eval_msgs, True)
        s_train = render_fn(train_msgs, False)
    except Exception as e:
        return {
            "status": "raises",
            "raises_on_system": True,
            "sys_in_eval": False,
            "sys_in_train": False,
            "prefix_match": False,
            "mask_offset_overflow": False,
            "unmasked_extra": "",
            "injects_empty_think_in_train": False,
            "nothink_fixes_prefix": False,
            "error": str(e),
        }

    ast_idx = s_train.find(AST_CANARY)
    p_train = s_train[:ast_idx] if ast_idx != -1 else s_train

    sys_in_eval = SYS_CANARY in p_eval
    sys_in_train = SYS_CANARY in p_train
    mask_offset_overflow = len(p_eval) > len(p_train)

    if p_train.startswith(p_eval):
        unmasked_extra = p_train[len(p_eval):]
        prefix_match = unmasked_extra.strip() == ""
    elif p_eval.rstrip(" ") == p_train.rstrip(" "):
        unmasked_extra = ""
        prefix_match = True
    else:
        unmasked_extra = "MISMATCH"
        prefix_match = False

    injects_empty_think = "<think>" in unmasked_extra and "</think>" in unmasked_extra

    nothink_fixes_prefix = False
    if not prefix_match:
        try:
            p_eval_nothink = render_fn(eval_msgs, True, enable_thinking=False)
            if p_train.startswith(p_eval_nothink) and p_train[len(p_eval_nothink):].strip() == "":
                nothink_fixes_prefix = True
        except Exception:
            pass

    is_ok = sys_in_eval and sys_in_train and prefix_match and not mask_offset_overflow

    return {
        "status": "ok" if is_ok else "broken",
        "raises_on_system": False,
        "sys_in_eval": sys_in_eval,
        "sys_in_train": sys_in_train,
        "prefix_match": prefix_match,
        "mask_offset_overflow": mask_offset_overflow,
        "unmasked_extra": unmasked_extra,
        "injects_empty_think_in_train": injects_empty_think,
        "nothink_fixes_prefix": nothink_fixes_prefix,
    }


def heal_chat_template_string(
    chat_template: str,
    disable_thinking: bool = True,
    bos_token: str = "<s>",
    eos_token: str = "</s>",
) -> Tuple[str, List[str]]:
    """
    Inspect a Jinja `chat_template` string via `probe_chat_template` and apply
    verified deterministic repairs for known behavioral failures.

    Every candidate edit is re-probed before acceptance: if the edit does not
    actually resolve the probed failure mode, it is discarded.

    Returns:
        (healed_template_string, list_of_human_readable_fix_descriptions)
    """
    if not chat_template or not isinstance(chat_template, str):
        return chat_template, []

    probe = probe_chat_template(chat_template, bos_token=bos_token, eos_token=eos_token)
    if probe.get("status") in ("ok", "unavailable"):
        return chat_template, []

    current = chat_template
    applied_fixes: List[str] = []

    # 1. Fix Train/Eval System-Role Asymmetry (e.g. Ministral 2410 `loop.last and system_message`)
    if probe.get("sys_in_eval") and not probe.get("sys_in_train"):
        candidate = re.sub(
            r"\bloop\.last(\s+and\s+system_message\b)",
            r"loop.first\1",
            current,
        )
        candidate = re.sub(
            r"(\bsystem_message\b[^%]*?\band\s+)loop\.last\b",
            r"\1loop.first",
            candidate,
        )
        if candidate != current:
            after = probe_chat_template(candidate, bos_token=bos_token, eos_token=eos_token)
            if after.get("sys_in_eval") and after.get("sys_in_train") and not after.get("mask_offset_overflow"):
                current = candidate
                probe = after
                applied_fixes.append(
                    "Fixed system-role loop guard (`loop.last` -> `loop.first`) so the system "
                    "prompt is preserved during 3-turn training and `--mask-prompt` offset calculation."
                )

    # 2. Fix Unmasked `<think>` Injection / Default Thinking Mode on Non-Thinking Datasets (e.g. Qwen 3)
    if disable_thinking and (
        probe.get("injects_empty_think_in_train") or probe.get("nothink_fixes_prefix")
    ):
        candidate = re.sub(
            r"\benable_thinking\s+is\s+defined\s+and\s+enable_thinking\s+is\s+false\b",
            "enable_thinking is not defined or enable_thinking is false",
            current,
        )
        if candidate == current:
            candidate = re.sub(
                r"\benable_thinking\s*==\s*false\b",
                "(enable_thinking is not defined or enable_thinking == false)",
                current,
            )
        if candidate != current:
            after = probe_chat_template(candidate, bos_token=bos_token, eos_token=eos_token)
            if after.get("prefix_match") and after.get("sys_in_eval") and after.get("sys_in_train"):
                current = candidate
                probe = after
                applied_fixes.append(
                    "Defaulted `enable_thinking` to `false` in `add_generation_prompt` so `<think>\\n\\n</think>\\n\\n` "
                    "is prefilled in the prompt prefix during both `--mask-prompt` training and evaluation."
                )

    return current, applied_fixes


def _resolve_token_str(val: Any, default: str) -> str:
    if isinstance(val, str) and val:
        return val
    if isinstance(val, dict) and isinstance(val.get("content"), str):
        return val["content"]
    return default


def heal_model_chat_template(
    model_path: Union[str, Path],
    data_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Probe and auto-heal the local `tokenizer_config.json` (and/or `chat_template.jinja`)
    inside `model_path` before launching training or evaluation.

    Creates a `.bak` backup of the original file before modifying it.
    Respects `MLX_DISABLE_TEMPLATE_SAFEGUARD=1` to opt out.
    """
    result: Dict[str, Any] = {
        "healed": False,
        "fixes": [],
        "files_modified": [],
        "probe_before": None,
        "probe_after": None,
    }

    if os.environ.get("MLX_DISABLE_TEMPLATE_SAFEGUARD") == "1":
        return result

    if not model_path:
        return result

    p = Path(str(model_path)).expanduser()
    if not p.exists() or not p.is_dir():
        return result

    disable_thinking = not dataset_has_thinking_traces(data_path)

    tok_cfg_path = p / "tokenizer_config.json"
    jinja_path = p / "chat_template.jinja"

    bos_token = "<s>"
    eos_token = "</s>"
    tok_cfg_data: Optional[Dict[str, Any]] = None

    if tok_cfg_path.is_file():
        try:
            with open(tok_cfg_path, "r", encoding="utf-8") as f:
                tok_cfg_data = json.load(f)
            if isinstance(tok_cfg_data, dict):
                bos_token = _resolve_token_str(tok_cfg_data.get("bos_token"), "<s>")
                eos_token = _resolve_token_str(tok_cfg_data.get("eos_token"), "</s>")
        except Exception:
            tok_cfg_data = None

    # 1. Check and heal `tokenizer_config.json` if it embeds a `chat_template` string
    if isinstance(tok_cfg_data, dict) and isinstance(tok_cfg_data.get("chat_template"), str):
        orig_ct = tok_cfg_data["chat_template"]
        result["probe_before"] = probe_chat_template(orig_ct, bos_token=bos_token, eos_token=eos_token)
        healed_ct, fixes = heal_chat_template_string(
            orig_ct,
            disable_thinking=disable_thinking,
            bos_token=bos_token,
            eos_token=eos_token,
        )
        if fixes and healed_ct != orig_ct:
            bak_path = tok_cfg_path.with_suffix(".json.bak")
            try:
                if not bak_path.exists():
                    shutil.copy2(tok_cfg_path, bak_path)
                tok_cfg_data["chat_template"] = healed_ct
                tmp_path = tok_cfg_path.with_suffix(".json.tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(tok_cfg_data, f, indent=2, ensure_ascii=False)
                    f.write("\n")
                os.replace(tmp_path, tok_cfg_path)
                result["healed"] = True
                result["fixes"].extend(fixes)
                result["files_modified"].append(str(tok_cfg_path))
                result["probe_after"] = probe_chat_template(healed_ct, bos_token=bos_token, eos_token=eos_token)
            except OSError:
                pass

    # 2. Check and heal sidecar `chat_template.jinja` if present
    if jinja_path.is_file():
        try:
            orig_jinja = jinja_path.read_text(encoding="utf-8")
            if result["probe_before"] is None:
                result["probe_before"] = probe_chat_template(orig_jinja, bos_token=bos_token, eos_token=eos_token)
            healed_jinja, fixes = heal_chat_template_string(
                orig_jinja,
                disable_thinking=disable_thinking,
                bos_token=bos_token,
                eos_token=eos_token,
            )
            if fixes and healed_jinja != orig_jinja:
                bak_jinja = jinja_path.with_suffix(".jinja.bak")
                if not bak_jinja.exists():
                    shutil.copy2(jinja_path, bak_jinja)
                tmp_jinja = jinja_path.with_suffix(".jinja.tmp")
                tmp_jinja.write_text(healed_jinja, encoding="utf-8")
                os.replace(tmp_jinja, jinja_path)
                result["healed"] = True
                for fix in fixes:
                    if fix not in result["fixes"]:
                        result["fixes"].append(fix)
                result["files_modified"].append(str(jinja_path))
                result["probe_after"] = probe_chat_template(healed_jinja, bos_token=bos_token, eos_token=eos_token)
        except OSError:
            pass

    if result["probe_after"] is None:
        result["probe_after"] = result["probe_before"]

    return result
