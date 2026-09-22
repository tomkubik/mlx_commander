"""
MLX Dataset Format Definitions and Record Formatting Logic.
Supports Apple MLX (mlx-lm) fine-tuning formats:
- Text format: {"text": "..."}
- Chat / Messages format: {"messages": [{"role": "...", "content": "..."}]}
- Prompt & Completion format: {"prompt": "...", "completion": "..."}
- DPO / Preference format: {"prompt": "...", "chosen": "...", "rejected": "..."}
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class MLXFormat(str, Enum):
    TEXT = "text"
    CHAT = "chat"
    PROMPT_COMPLETION = "prompt_completion"
    DPO = "dpo"

    @classmethod
    def display_names(cls) -> Dict["MLXFormat", str]:
        return {
            cls.TEXT: "Text Format (Causal LM / Pre-training)",
            cls.CHAT: "Chat / Messages Format (Multi-turn conversations)",
            cls.PROMPT_COMPLETION: "Prompt & Completion Format (Instruction / Q&A)",
            cls.DPO: "DPO / Preference Format (Direct Preference Optimization)",
        }

    @classmethod
    def descriptions(cls) -> Dict["MLXFormat", str]:
        return {
            cls.TEXT: 'Schema: {"text": "..."} — Standard causal language modeling.',
            cls.CHAT: 'Schema: {"messages": [{"role": "system|user|assistant", "content": "..."}]} — Full dialogue fine-tuning.',
            cls.PROMPT_COMPLETION: 'Schema: {"prompt": "...", "completion": "..."} — Fine-tune model to complete specific prompts (supports --mask-prompt).',
            cls.DPO: 'Schema: {"prompt": "...", "chosen": "...", "rejected": "..."} — Preference alignment fine-tuning.',
        }


@dataclass
class ColumnMapping:
    # Text format
    text_col: Optional[str] = None
    text_template: Optional[str] = None  # e.g., "{instruction}\n\n{output}"

    # Chat format: Option A (pre-structured messages column)
    messages_col: Optional[str] = None
    role_key: str = "role"
    content_key: str = "content"
    role_map: Dict[str, str] = field(default_factory=lambda: {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
        "bot": "assistant",
    })

    # Chat format: Option B (separate role columns)
    system_col: Optional[str] = None
    user_col: Optional[str] = None
    assistant_col: Optional[str] = None

    # Prompt & Completion format
    prompt_col: Optional[str] = None
    completion_col: Optional[str] = None
    prompt_template: Optional[str] = None

    # DPO format
    dpo_prompt_col: Optional[str] = None
    chosen_col: Optional[str] = None
    rejected_col: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def extract_template_vars(template: str) -> List[str]:
    """Find all {variable_name} references in a format string."""
    return re.findall(r"\{([a-zA-Z0-9_]+)\}", template)


def parse_column_list(
    col_spec: Optional[str],
    available_columns: Optional[List[str]] = None,
) -> List[str]:
    """
    Parse a single column name, plus-separated ('a + b'), or comma-separated ('a, b') list.
    If the full col_spec exists directly in available_columns, it is treated as a single column name.
    """
    if not col_spec:
        return []
    cleaned = col_spec.strip()
    if not cleaned:
        return []
    if available_columns and cleaned in available_columns:
        return [cleaned]
    if "+" in cleaned:
        parts = [c.strip() for c in cleaned.split("+")]
        return [c for c in parts if c]
    if "," in cleaned:
        parts = [c.strip() for c in cleaned.split(",")]
        return [c for c in parts if c]
    return [cleaned]


def get_column_value(
    record: Dict[str, Any],
    col_spec: Optional[str],
    separator: str = "\n\n",
) -> str:
    """
    Extract and concatenate values from one or more columns in record.
    Joins non-empty string values using separator (default double newline \n\n).
    """
    if not col_spec:
        return ""
    cols = parse_column_list(col_spec)
    parts: List[str] = []
    for c in cols:
        if c in record and record[c] is not None:
            val = str(record[c]).strip()
            if val:
                parts.append(val)
    return separator.join(parts)


def format_template(template: str, record: Dict[str, Any]) -> str:
    """Format template string safely, filling missing keys with empty string."""
    class SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return ""

    safe_record = SafeDict({k: str(v) if v is not None else "" for k, v in record.items()})
    try:
        return template.format_map(safe_record)
    except Exception:
        # Fallback simple replacement if braces in text cause issues
        res = template
        for k, v in record.items():
            res = res.replace(f"{{{k}}}", str(v) if v is not None else "")
        return res


def validate_mapping(
    format_type: MLXFormat,
    mapping: ColumnMapping,
    available_columns: List[str],
) -> List[str]:
    """Validate that the required columns or templates exist for the selected format."""
    errors = []
    col_set = set(available_columns)

    def check_cols(spec: Optional[str], field_name: str, required: bool = True) -> None:
        if not spec:
            if required:
                errors.append(f"{field_name} column must be specified.")
            return
        cols = parse_column_list(spec, available_columns)
        if not cols and required:
            errors.append(f"{field_name} column must be specified.")
            return
        for c in cols:
            if c not in col_set:
                errors.append(f"{field_name} column '{c}' not found in dataset columns.")

    if format_type == MLXFormat.TEXT:
        if mapping.text_template:
            vars_needed = extract_template_vars(mapping.text_template)
            missing = [v for v in vars_needed if v not in col_set]
            if missing:
                errors.append(f"Template references missing column(s): {', '.join(missing)}")
        elif mapping.text_col:
            check_cols(mapping.text_col, "Text", required=True)
        else:
            errors.append("Either a text column or a text template must be specified.")

    elif format_type == MLXFormat.CHAT:
        if mapping.messages_col:
            if mapping.messages_col not in col_set:
                errors.append(f"Messages column '{mapping.messages_col}' not found.")
        elif mapping.user_col and mapping.assistant_col:
            check_cols(mapping.user_col, "User", required=True)
            check_cols(mapping.assistant_col, "Assistant", required=True)
            if mapping.system_col:
                check_cols(mapping.system_col, "System", required=False)
        else:
            errors.append("Chat format requires either a messages list column OR user and assistant columns.")

    elif format_type == MLXFormat.PROMPT_COMPLETION:
        if mapping.prompt_template:
            vars_needed = extract_template_vars(mapping.prompt_template)
            missing = [v for v in vars_needed if v not in col_set]
            if missing:
                errors.append(f"Template references missing column(s): {', '.join(missing)}")
        else:
            check_cols(mapping.prompt_col, "Prompt", required=True)

        check_cols(mapping.completion_col, "Completion", required=True)

    elif format_type == MLXFormat.DPO:
        check_cols(mapping.dpo_prompt_col or mapping.prompt_col, "Prompt", required=True)
        check_cols(mapping.chosen_col, "Chosen", required=True)
        check_cols(mapping.rejected_col, "Rejected", required=True)

    return errors


def normalize_role(role_raw: Any, role_map: Dict[str, str]) -> str:
    """Normalize role names to standard MLX roles (system, user, assistant)."""
    val = str(role_raw).lower().strip()
    return role_map.get(val, val)


def format_record(
    record: Dict[str, Any],
    format_type: MLXFormat,
    mapping: ColumnMapping,
) -> Dict[str, Any]:
    """
    Format an individual dataset record into the selected MLX JSON schema.
    Returns a dictionary suitable for JSONL serialization.
    """
    if format_type == MLXFormat.TEXT:
        if mapping.text_template:
            text = format_template(mapping.text_template, record)
        elif mapping.text_col:
            text = get_column_value(record, mapping.text_col)
        else:
            text = ""
        return {"text": text}

    elif format_type == MLXFormat.CHAT:
        messages: List[Dict[str, str]] = []

        if mapping.messages_col and mapping.messages_col in record:
            raw_msgs = record[mapping.messages_col]
            if isinstance(raw_msgs, list):
                for item in raw_msgs:
                    if isinstance(item, dict):
                        raw_role = item.get(mapping.role_key, "user")
                        role = normalize_role(raw_role, mapping.role_map)
                        content = str(item.get(mapping.content_key, ""))
                        messages.append({"role": role, "content": content})
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        role = normalize_role(item[0], mapping.role_map)
                        content = str(item[1])
                        messages.append({"role": role, "content": content})
        else:
            # Multi-column mapping
            if mapping.system_col:
                sys_content = get_column_value(record, mapping.system_col)
                if sys_content:
                    messages.append({
                        "role": "system",
                        "content": sys_content,
                    })
            if mapping.user_col:
                user_content = get_column_value(record, mapping.user_col)
                messages.append({
                    "role": "user",
                    "content": user_content,
                })
            if mapping.assistant_col:
                asst_content = get_column_value(record, mapping.assistant_col)
                messages.append({
                    "role": "assistant",
                    "content": asst_content,
                })

        return {"messages": messages}

    elif format_type == MLXFormat.PROMPT_COMPLETION:
        if mapping.prompt_template:
            prompt = format_template(mapping.prompt_template, record)
        elif mapping.prompt_col:
            prompt = get_column_value(record, mapping.prompt_col)
        else:
            prompt = ""

        completion = get_column_value(record, mapping.completion_col) if mapping.completion_col else ""

        return {"prompt": prompt, "completion": completion}

    elif format_type == MLXFormat.DPO:
        p_col = mapping.dpo_prompt_col or mapping.prompt_col
        if p_col and p_col in record and isinstance(record[p_col], (dict, list)):
            p_val = record[p_col]
        else:
            p_val = get_column_value(record, p_col)

        if mapping.chosen_col and mapping.chosen_col in record and isinstance(record[mapping.chosen_col], (dict, list)):
            c_val = record[mapping.chosen_col]
        else:
            c_val = get_column_value(record, mapping.chosen_col)

        if mapping.rejected_col and mapping.rejected_col in record and isinstance(record[mapping.rejected_col], (dict, list)):
            r_val = record[mapping.rejected_col]
        else:
            r_val = get_column_value(record, mapping.rejected_col)

        return {
            "prompt": p_val,
            "chosen": c_val,
            "rejected": r_val,
        }

    raise ValueError(f"Unknown MLX format: {format_type}")


def _normalize_col_key(k: str) -> str:
    """Normalize a column name for matching (lowercase, stripped, replace '-' with '_')."""
    return k.strip().lower().replace("-", "_")


def auto_detect_mapping(format_type: MLXFormat, columns: List[str]) -> ColumnMapping:
    """Attempt to intelligently auto-detect column mappings based on common field names."""
    norm_map = {_normalize_col_key(col): col for col in columns}
    mapping = ColumnMapping()

    def find_match(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            norm_cand = _normalize_col_key(cand)
            if norm_cand in norm_map:
                return norm_map[norm_cand]
        return None

    if format_type == MLXFormat.TEXT:
        text_candidates = [
            "text", "content", "body", "document", "raw_text", "sentence",
            "sentences", "article", "passage", "data", "raw", "tokens",
        ]
        match = find_match(text_candidates)
        if match:
            mapping.text_col = match
        elif columns:
            mapping.text_col = columns[0]

    elif format_type == MLXFormat.CHAT:
        messages_candidates = [
            "messages", "conversations", "dialog", "dialogue", "chat",
            "history", "turns", "conversation",
        ]
        match = find_match(messages_candidates)
        if match:
            mapping.messages_col = match
            if _normalize_col_key(match) == "conversations":
                mapping.role_key = "from"
                mapping.content_key = "value"
            return mapping

        system_candidates = [
            "system", "system_prompt", "instruction_system", "system_message", "developer", "context",
        ]
        user_candidates = [
            "user", "human", "prompt", "instruction", "input", "query", "question",
            "user_message", "client", "problem",
        ]
        assistant_candidates = [
            "assistant", "gpt", "bot", "response", "output", "completion", "answer",
            "solution", "bot_message", "model", "model_response",
        ]
        mapping.system_col = find_match(system_candidates)
        mapping.user_col = find_match(user_candidates)
        mapping.assistant_col = find_match(assistant_candidates)

    elif format_type == MLXFormat.PROMPT_COMPLETION:
        prompt_candidates = [
            "prompt", "instruction", "input", "query", "question", "problem",
            "task", "task_description", "context", "document", "article",
            "premise", "source", "src", "text", "user_query", "input_text",
            "goal", "prompts", "instructions", "inputs", "queries",
            "questions", "problems",
        ]
        completion_candidates = [
            "completion", "output", "response", "answer", "solution",
            "target", "reference", "reference_answer", "ground_truth",
            "ground_truth_answer", "gt", "result", "summary", "highlights",
            "hypothesis", "label", "labels", "reply", "generation",
            "canonical_solution", "code", "output_text", "model_response",
            "target_text", "completions", "outputs", "responses", "answers",
            "solutions", "results",
        ]

        p_match = find_match(prompt_candidates)
        c_match = find_match(completion_candidates)

        # Disallow mapping both prompt and completion to the exact same column if multiple columns exist
        if p_match and c_match and p_match == c_match and len(columns) > 1:
            sub_map = {k: v for k, v in norm_map.items() if v != p_match}
            c_match = None
            for cand in completion_candidates:
                if _normalize_col_key(cand) in sub_map:
                    c_match = sub_map[_normalize_col_key(cand)]
                    break

        # Fallback 1: Exactly 2 columns available
        if len(columns) == 2:
            if p_match and not c_match:
                c_match = [c for c in columns if c != p_match][0]
            elif c_match and not p_match:
                p_match = [c for c in columns if c != c_match][0]
            elif not p_match and not c_match:
                p_match = columns[0]
                c_match = columns[1]

        # Fallback 2: Prompt matched, but completion unmatched from 3+ columns
        elif len(columns) >= 3 and p_match and not c_match:
            meta_keys = {"id", "idx", "index", "uid", "split", "metadata", "meta", "timestamp", "created_at", "source"}
            rem = [c for c in columns if c != p_match and _normalize_col_key(c) not in meta_keys]
            if len(rem) == 1:
                c_match = rem[0]

        mapping.prompt_col = p_match
        mapping.completion_col = c_match

    elif format_type == MLXFormat.DPO:
        dpo_prompt_candidates = [
            "prompt", "instruction", "input", "question", "query", "problem", "context",
        ]
        chosen_candidates = [
            "chosen", "preferred", "accepted", "positive", "better", "winner",
            "chosen_response", "selected",
        ]
        rejected_candidates = [
            "rejected", "dispreferred", "negative", "worse", "loser",
            "rejected_response", "unselected",
        ]
        mapping.dpo_prompt_col = find_match(dpo_prompt_candidates)
        mapping.chosen_col = find_match(chosen_candidates)
        mapping.rejected_col = find_match(rejected_candidates)

    return mapping


def auto_detect_format_and_mapping(columns: List[str]) -> Tuple[MLXFormat, ColumnMapping]:
    """
    Intelligently determine the most appropriate MLX target format and its column mapping
    based on the available columns in a dataset.
    """
    if not columns:
        return MLXFormat.PROMPT_COMPLETION, ColumnMapping()

    norm_set = {_normalize_col_key(c) for c in columns}

    # 1. Check for Chat format
    chat_keys = {"messages", "conversations", "dialog", "dialogue", "chat", "turns"}
    if norm_set & chat_keys:
        return MLXFormat.CHAT, auto_detect_mapping(MLXFormat.CHAT, columns)

    if ({"user", "human", "prompt"} & norm_set) and ({"assistant", "gpt", "bot"} & norm_set):
        return MLXFormat.CHAT, auto_detect_mapping(MLXFormat.CHAT, columns)

    # 2. Check for DPO format
    if ({"chosen", "preferred", "better", "winner"} & norm_set) and ({"rejected", "dispreferred", "worse", "loser"} & norm_set):
        return MLXFormat.DPO, auto_detect_mapping(MLXFormat.DPO, columns)

    # 3. Check for 1-column dataset (always Text format)
    if len(columns) == 1:
        return MLXFormat.TEXT, auto_detect_mapping(MLXFormat.TEXT, columns)

    # 4. Check for Text format (e.g. text/content without paired completion/answer)
    text_primary = {"text", "raw_text", "tokens", "passage", "document"}
    has_text = bool(norm_set & text_primary)
    completion_keys = {
        "completion", "output", "response", "answer", "solution", "target",
        "ground_truth", "gt", "result", "summary", "label", "reply",
    }
    has_completion = bool(norm_set & completion_keys)

    if has_text and not has_completion:
        # If there's a text column with no separate completion column, causal text is best
        return MLXFormat.TEXT, auto_detect_mapping(MLXFormat.TEXT, columns)

    # 5. Default to Prompt & Completion (Q&A / Instruction)
    return MLXFormat.PROMPT_COMPLETION, auto_detect_mapping(MLXFormat.PROMPT_COMPLETION, columns)
