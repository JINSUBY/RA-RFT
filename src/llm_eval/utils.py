from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import re

_ANSWER_RE = re.compile(r"<\s*answer\b[^>]*>(.*?)<\s*/\s*answer\s*>",
                        flags=re.IGNORECASE | re.DOTALL)
_THINK_RE  = re.compile(r"<\s*think\b[^>]*>.*?<\s*/\s*think\s*>",
                        flags=re.IGNORECASE | re.DOTALL)
_CORRECTION_RE = re.compile(r"<\s*correction\b[^>]*>(.*?)<\s*/\s*correction\s*>",
                            flags=re.IGNORECASE | re.DOTALL)
_ALL_TAGS_RE = re.compile(r"<\s*/?[a-zA-Z_][a-zA-Z0-9_]*\b[^>]*>", flags=re.IGNORECASE)

def extract_answer_content(text: Optional[str]) -> str:
    """
    Extract only the <answer> block from model response (for step1 classification):
    - Remove all <think>...</think> blocks
    - Extract content inside <answer>...</answer> if present
    - Remove "\nNIL\n" pattern
    - Return original text if no answer block found
    - Apply strip() to the result
    """
    if not text:
        return ""
    s = str(text)

    # 1) Remove all <think>...</think> blocks
    s = _THINK_RE.sub("", s)

    # 2) Extract content from first <answer>...</answer> (use original if not found)
    m = _ANSWER_RE.search(s)
    if m:
        answer_content = m.group(1)
        # Remove "\nNIL\n" pattern inside <answer> block
        answer_content = answer_content.replace("\nNIL\n", "")
        return answer_content.strip()
    return s.strip()


def extract_reasoning_text(text: Optional[str]) -> str:
    """
    Clean model output without using correction:
    - Keep content of all tags but remove tag markers
    - If <correction>...</correction> block exists:
      1) Exclude correction block content
    - Remove "\nNIL\n" pattern
    - Apply strip() to the result
    """
    if not text:
        return ""
    s = str(text)

    # 1) Extract and remove <correction>...</correction> block
    m_correction = _CORRECTION_RE.search(s)
    if m_correction:
        # Remove entire correction block (including tags)
        s = _CORRECTION_RE.sub("", s)

    # 2) Remove all tags (<>, </>) while keeping content
    s = _ALL_TAGS_RE.sub("", s)

    # 3) Remove "\nNIL\n" pattern
    s = s.replace("\nNIL\n", "")

    return s.strip()


_ANSWER_TAG_RE = re.compile(r"<\s*/?answer\b[^>]*>", flags=re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"<\s*/?think\b[^>]*>", flags=re.IGNORECASE)


def strip_format_tags(text: Optional[str]) -> str:
    """
    Remove only special tags from model response while keeping all content:
    - Remove <think> and </think> tags only (keep content)
    - Remove <answer> and </answer> tags only (keep content)
    - Use all content from the full response
    - Apply strip() to the result
    """
    if not text:
        return ""
    s = str(text)

    # 1) Remove <think> and </think> tags only (keep content)
    s = _THINK_TAG_RE.sub("", s)

    # 2) Remove <answer> and </answer> tags only (keep content)
    s = _ANSWER_TAG_RE.sub("", s)

    return s.strip()

# -------------------------------
# Safe division
# -------------------------------
def _safe_div(n: float, d: float) -> float:
    return 0.0 if d == 0 else n / d


# -------------------------------
# Temporal IoU for intervals
# -------------------------------
def calc_temporal_iou(interval1: List[float], interval2: List[float]) -> float:
    """
    IoU of two intervals [s1, e1] and [s2, e2] in seconds.
    - Returns 0.0 if intervals don't overlap
    - Returns 0.0 if union is 0
    """
    if not interval1 or not interval2:
        return 0.0
    s1, e1 = interval1
    s2, e2 = interval2
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(0.0, (e1 - s1) + (e2 - s2) - inter)
    return 0.0 if union == 0.0 else inter / union


# -------------------------------
# Set-based Jaccard (IoU for labels)
# -------------------------------
def jaccard_index_sets(
    pred_labels: Iterable[str],
    gt_labels: Iterable[str],
    empty_equals_one: bool = True,
) -> float:
    """
    Jaccard Index for multi-label sets = |A∩B| / |A∪B|
    - If (A∪B) is empty: returns 1.0 if empty_equals_one=True, else 0.0
    """
    p, g = set(pred_labels), set(gt_labels)
    union = p | g
    if len(union) == 0:
        return 1.0 if empty_equals_one else 0.0
    inter = p & g
    return len(inter) / len(union)


# -------------------------------
# Label space management + F1 aggregation utilities
# -------------------------------
@dataclass(frozen=True)
class VTGLabelSpace:
    """
    Helper for managing a fixed label space.
    - labels: Sorted list of label names (for stable indexing)
    - index:  Label-to-index mapping
    """
    labels: Tuple[str, ...]
    index: Dict[str, int]

    @staticmethod
    def from_paths(valid_paths: Iterable[str]) -> "VTGLabelSpace":
        labels = tuple(sorted(valid_paths))
        index = {lab: i for i, lab in enumerate(labels)}
        return VTGLabelSpace(labels=labels, index=index)

    # -------- Encoding/Decoding --------
    def encode(self, labs: Iterable[str]) -> List[int]:
        """Convert label list to fixed-size 0/1 vector"""
        vec = [0] * len(self.labels)
        for lab in labs:
            idx = self.index.get(lab)
            if idx is not None:
                vec[idx] = 1
        return vec

    # -------- Count dictionary --------
    def new_counts(self) -> Dict[str, Dict[str, int]]:
        """
        Initialize per_label_counts
        {label: {'tp':0,'fp':0,'fn':0}, ...}
        """
        return {lab: {"tp": 0, "fp": 0, "fn": 0} for lab in self.labels}

    def update_counts(
        self,
        counts: Dict[str, Dict[str, int]],
        y_true_vec: List[int],
        y_pred_vec: List[int],
    ) -> None:
        """
        Accumulate tp/fp/fn counts per label
        """
        m = len(self.labels)
        if len(y_true_vec) != m or len(y_pred_vec) != m:
            raise ValueError("Vector length mismatch with label space.")
        for i, lab in enumerate(self.labels):
            yt, yp = y_true_vec[i], y_pred_vec[i]
            c = counts[lab]
            if yp == 1 and yt == 1:
                c["tp"] += 1
            elif yp == 1 and yt == 0:
                c["fp"] += 1
            elif yp == 0 and yt == 1:
                c["fn"] += 1
            # (yt==0 & yp==0) doesn't affect F1

    # -------- Metric computation --------
    @staticmethod
    def f1_from_counts(tp: int, fp: int, fn: int) -> float:
        # F1 = 2TP / (2TP + FP + FN)
        denom = 2 * tp + fp + fn
        return 0.0 if denom == 0 else (2 * tp) / denom

    @staticmethod
    def precision_from_counts(tp: int, fp: int) -> float:
        return _safe_div(tp, tp + fp)

    @staticmethod
    def recall_from_counts(tp: int, fn: int) -> float:
        return _safe_div(tp, tp + fn)

    def label_f1(self, counts: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        """Return per-label F1 dictionary"""
        return {
            lab: self.f1_from_counts(c["tp"], c["fp"], c["fn"])
            for lab, c in counts.items()
        }

    def macro_f1(self, counts: Dict[str, Dict[str, int]]) -> float:
        """Arithmetic mean of per-label F1 scores (Macro-F1)"""
        f1s = self.label_f1(counts).values()
        return 0.0 if not f1s else sum(f1s) / len(self.labels)

    # -------- Count merging --------
    def merge_counts(
        self, many: Iterable[Dict[str, Dict[str, int]]]
    ) -> Dict[str, Dict[str, int]]:
        """Merge multiple per_label_counts dicts by summing them."""
        out = self.new_counts()
        for counts in many:
            for lab in self.labels:
                c = counts.get(lab)
                if not c:
                    continue
                out[lab]["tp"] += int(c.get("tp", 0))
                out[lab]["fp"] += int(c.get("fp", 0))
                out[lab]["fn"] += int(c.get("fn", 0))
        return out


def compute_binary_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and F1 from confusion matrix values."""
    tot = tp + fp + fn + tn
    acc = (tp + tn) / tot if tot else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def compute_label_score_stats(
    score_data: Dict[str, Dict[str, List[float]]],
    labels: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    """Compute simple_mean, weighted_mean, count, total_weight for each label."""
    result = {}
    for label in labels:
        data = score_data.get(label, {"scores": [], "weights": []})
        scores = data["scores"]
        weights = data["weights"]
        if scores:
            simple_mean = sum(scores) / len(scores)
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            total_weight = sum(weights)
            weighted_mean = weighted_sum / total_weight if total_weight > 0 else 0.0
            result[label] = {
                "simple_mean": simple_mean,
                "weighted_mean": weighted_mean,
                "count": len(scores),
                "total_weight": total_weight,
            }
        else:
            result[label] = {"simple_mean": 0.0, "weighted_mean": 0.0, "count": 0, "total_weight": 0.0}
    return result
