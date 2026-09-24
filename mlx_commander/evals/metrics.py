"""
Evaluation Metrics & Error Analysis for Generative Model Inference.
Implements:
  - Exact Match (strict & normalized)
  - Substring Contains Match
  - Word-Level Precision, Recall, and F1 (SQuAD standard)
  - 2x2 Migration & Regression Matrix (PRESERVED, FIXED, REGRESSED, PERSISTENT_FAIL)
  - Categorical Confusion Matrix for structured classification tasks
"""

import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def normalize_answer(text: str) -> str:
    """
    Standard NLP normalization:
    - Lowercase
    - Remove punctuation
    - Remove English articles (a, an, the)
    - Collapse extra whitespace
    """
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse multiple whitespaces
    return " ".join(text.split())


def compute_exact_match(prediction: str, golden: str, normalize: bool = True) -> bool:
    """Check if prediction equals golden answer (optionally normalized)."""
    if normalize:
        return normalize_answer(prediction) == normalize_answer(golden)
    return prediction == golden


def compute_substring_match(prediction: str, golden: str) -> bool:
    """
    Check if the golden answer is contained within the prediction string.
    Evaluates both normalized and raw case-insensitive inclusion.
    """
    if not golden or not prediction:
        return False
    norm_gold = normalize_answer(golden)
    norm_pred = normalize_answer(prediction)
    if norm_gold and norm_gold in norm_pred:
        return True
    return golden.strip().lower() in prediction.strip().lower()


def compute_word_metrics(prediction: str, golden: str) -> Dict[str, float]:
    """
    Compute Word-Level Precision, Recall, and F1 score.
    Tokens are normalized words from the actual generated output.
    """
    pred_words = normalize_answer(prediction).split()
    gold_words = normalize_answer(golden).split()

    if not pred_words or not gold_words:
        # If both empty, perfect match; if only one empty, zero match
        is_empty_match = (len(pred_words) == 0 and len(gold_words) == 0)
        score = 1.0 if is_empty_match else 0.0
        return {"precision": score, "recall": score, "f1": score}

    common = Counter(pred_words) & Counter(gold_words)
    num_same = sum(common.values())

    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = 1.0 * num_same / len(pred_words)
    recall = 1.0 * num_same / len(gold_words)
    f1 = (2 * precision * recall) / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def classify_transition(baseline_correct: bool, finetuned_correct: bool) -> str:
    """
    Classify the sample transition state from baseline to fine-tuned:
      - PRESERVED:       Baseline Correct  ->  Fine-Tuned Correct
      - FIXED:           Baseline Wrong    ->  Fine-Tuned Correct (LoRA healed it)
      - REGRESSED:       Baseline Correct  ->  Fine-Tuned Wrong   (Catastrophic forgetting)
      - PERSISTENT_FAIL: Baseline Wrong    ->  Fine-Tuned Wrong   (Persistent hard example)
    """
    if baseline_correct and finetuned_correct:
        return "PRESERVED"
    elif not baseline_correct and finetuned_correct:
        return "FIXED"
    elif baseline_correct and not finetuned_correct:
        return "REGRESSED"
    else:
        return "PERSISTENT_FAIL"


def build_migration_matrix(transitions: List[str]) -> Dict[str, Any]:
    """
    Build 2x2 Model Migration & Regression Matrix from sample transition statuses.
    """
    total = max(1, len(transitions))
    counts = {
        "PRESERVED": transitions.count("PRESERVED"),
        "FIXED": transitions.count("FIXED"),
        "REGRESSED": transitions.count("REGRESSED"),
        "PERSISTENT_FAIL": transitions.count("PERSISTENT_FAIL"),
    }
    percentages = {k: round((v / total) * 100.0, 1) for k, v in counts.items()}

    return {
        "total": len(transitions),
        "counts": counts,
        "percentages": percentages,
        "fixed_count": counts["FIXED"],
        "regressed_count": counts["REGRESSED"],
    }


def build_confusion_matrix(
    ground_truths: List[str],
    predictions: List[str],
    max_classes: int = 15,
) -> Optional[Dict[str, Any]]:
    """
    Build a categorical Confusion Matrix if target answers belong to a finite label set.
    Returns None if output space is open-ended text.
    """
    if not ground_truths or len(ground_truths) != len(predictions):
        return None

    # Normalize labels
    norm_gt = [normalize_answer(g) for g in ground_truths]
    norm_pred = [normalize_answer(p) for p in predictions]

    unique_classes = sorted(list(set(norm_gt)))
    if not (2 <= len(unique_classes) <= max_classes):
        return None

    # Matrix: matrix[actual_class][predicted_class] = count
    matrix: Dict[str, Dict[str, int]] = {
        c: {c2: 0 for c2 in unique_classes + ["other"]} for c in unique_classes
    }

    correct_count = 0
    for gt, pred in zip(norm_gt, norm_pred):
        if pred in unique_classes:
            matrix[gt][pred] += 1
            if gt == pred:
                correct_count += 1
        else:
            # Check if gt is a substring in pred
            matched_c = None
            for c in unique_classes:
                if c and c in pred:
                    matched_c = c
                    break
            if matched_c:
                matrix[gt][matched_c] += 1
                if gt == matched_c:
                    correct_count += 1
            else:
                matrix[gt]["other"] += 1

    accuracy = round((correct_count / len(ground_truths)) * 100.0, 2)

    return {
        "is_categorical": True,
        "classes": unique_classes,
        "matrix": matrix,
        "accuracy": accuracy,
        "total_samples": len(ground_truths),
    }


def format_cli_migration_matrix(migration_matrix: Dict[str, Any]) -> str:
    """
    Format 2x2 Model Migration & Regression Matrix into a clear, aligned ASCII box.
    """
    if not migration_matrix:
        return ""
    counts = migration_matrix.get("counts", {})
    pcts = migration_matrix.get("percentages", {})
    total = migration_matrix.get("total", 0)

    pres_c = counts.get("PRESERVED", 0)
    pres_p = pcts.get("PRESERVED", 0.0)
    regr_c = counts.get("REGRESSED", 0)
    regr_p = pcts.get("REGRESSED", 0.0)
    fix_c = counts.get("FIXED", 0)
    fix_p = pcts.get("FIXED", 0.0)
    fail_c = counts.get("PERSISTENT_FAIL", 0)
    fail_p = pcts.get("PERSISTENT_FAIL", 0.0)

    net_gain = fix_c - regr_c
    net_gain_pct = round((net_gain / max(1, total)) * 100.0, 1)
    sign = "+" if net_gain >= 0 else ""

    lines = [
        "======================================================================",
        "                 MODEL CONFUSION & MIGRATION MATRIX                   ",
        "               (Pre-Trained Baseline vs Post-Tuning LoRA)             ",
        "======================================================================",
        "                              POST-TUNING (LoRA)                      ",
        "                           CORRECT             WRONG                  ",
        "                     ┌──────────────────┬──────────────────┐          ",
        f"           CORRECT   │  PRESERVED: {pres_c:>4} │  REGRESSED: {regr_c:>4} │  ◄ Catastrophic forgetting",
        f"BASELINE             │     ({pres_p:>5.1f}%)    │     ({regr_p:>5.1f}%)    │          ",
        "                     ├──────────────────┼──────────────────┤          ",
        f"           WRONG     │  FIXED:     {fix_c:>4} │  PERSISTENT:{fail_c:>4} │  ◄ Where LoRA healed model",
        f"                     │     ({fix_p:>5.1f}%)    │     ({fail_p:>5.1f}%)    │          ",
        "                     └──────────────────┴──────────────────┘          ",
        f"  • Preserved (both correct):    {pres_c:>4} ({pres_p:.1f}%)",
        f"  • Fixed (LoRA healed):         {fix_c:>4} ({fix_p:.1f}%)",
        f"  • Regressed (model broke):      {regr_c:>4} ({regr_p:.1f}%)",
        f"  • Persistent Fail (both fail): {fail_c:>4} ({fail_p:.1f}%)",
        f"  • Net Model Accuracy Gain:     {sign}{net_gain} ({sign}{net_gain_pct:.1f}%)",
        "======================================================================",
    ]
    return "\n".join(lines)


def format_cli_categorical_confusion_matrix(cm: Optional[Dict[str, Any]]) -> str:
    """
    Format Categorical Confusion Matrix ([Actual] x [Predicted]) into an ASCII table.
    """
    if not cm or not cm.get("is_categorical"):
        return ""
    classes = cm.get("classes", [])
    matrix = cm.get("matrix", {})
    if not classes or not matrix:
        return ""

    has_other = any(matrix.get(c, {}).get("other", 0) > 0 for c in classes)
    pred_cols = list(classes) + (["other"] if has_other else [])

    max_cls_len = max(len(str(c)) for c in classes)
    col_w = max(9, max_cls_len + 2)
    label_w = max(16, max_cls_len + 2)

    lines = [
        "----------------------------------------------------------------------",
        "           CATEGORICAL CONFUSION MATRIX (ACTUAL × PREDICTED)          ",
        "----------------------------------------------------------------------",
    ]

    header_cols = "".join(f"{c:>{col_w}}" for c in pred_cols)
    actual_pred_hdr = "Actual \\ Pred"
    lines.append(f"{actual_pred_hdr:<{label_w}}{header_cols}{'Recall':>{col_w}}")
    lines.append("-" * (label_w + (len(pred_cols) + 1) * col_w))

    for c in classes:
        row_counts = matrix.get(c, {})
        c_total = sum(row_counts.values())
        c_correct = row_counts.get(c, 0)
        c_recall = (c_correct / c_total * 100.0) if c_total > 0 else 0.0
        cells = "".join(f"{row_counts.get(p, 0):>{col_w}}" for p in pred_cols)
        lines.append(f"{c:<{label_w}}{cells}{f'{c_recall:.1f}%':>{col_w}}")

    lines.append("-" * (label_w + (len(pred_cols) + 1) * col_w))
    accuracy = cm.get("accuracy", 0.0)
    total_samples = cm.get("total_samples", 0)
    lines.append(f"  • Categorical Accuracy: {accuracy:.1f}% ({total_samples} samples across {len(classes)} classes)")
    lines.append("======================================================================")
    return "\n".join(lines)


def format_cli_eval_summary_matrices(summary: Dict[str, Any]) -> str:
    """Format all available evaluation matrices for CLI display."""
    chunks: List[str] = []
    mig = summary.get("migration_matrix")
    if mig:
        chunks.append(format_cli_migration_matrix(mig))
    cm = summary.get("confusion_matrix")
    if cm and cm.get("is_categorical"):
        chunks.append(format_cli_categorical_confusion_matrix(cm))
    return "\n\n".join(chunks)
