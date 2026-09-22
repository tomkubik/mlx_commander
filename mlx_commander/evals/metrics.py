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
