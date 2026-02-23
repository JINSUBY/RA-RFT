"""
Core processing and merging logic for parallel evaluation.

Handles per-sample evaluation, metric accumulation, and merging
results from multiple parallel splits.
"""

import multiprocessing as mp
from typing import Any, Dict, List, Optional

from src.llm_eval.llm_evaluator import VTG_VALID_PATHS
from src.llm_eval.utils import VTGLabelSpace, compute_binary_metrics, compute_label_score_stats

# Label space (shared across all workers via fork)
label_space = VTGLabelSpace.from_paths(VTG_VALID_PATHS)


# -----------------------------------------------------------------------
# Worker pool helpers
# -----------------------------------------------------------------------

_progress_counter = None


def _init_worker(counter):
    """Initialize worker process with shared progress counter."""
    global _progress_counter
    _progress_counter = counter


# -----------------------------------------------------------------------
# Per-split processing
# -----------------------------------------------------------------------

def _new_hardness_data() -> Dict[str, Any]:
    """Create a fresh per-hardness-level metric accumulator."""
    return {
        "iou_all": [],
        "jaccard_list": [],
        "reasoning_scores": [],
        "tn_reasoning_scores": [],
        "sbert_scores": [],
        "tn_sbert_scores": [],
        "counts": label_space.new_counts(),
        "r_at_k_counts": {"0.3": 0, "0.5": 0, "0.7": 0},
    }


def _new_label_score_data() -> Dict[str, Dict[str, List[float]]]:
    """Create a fresh per-label score accumulator."""
    return {label: {"scores": [], "weights": []} for label in label_space.labels}


def process_split_with_progress(
    split_data: List[Dict[str, Any]],
    split_idx: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Process a single data split and update the shared progress counter."""
    from src.llm_eval.llm_evaluator import evaluate_three_stage

    n = len(split_data)
    TP = FP = FN = TN = 0
    iou_all: List[float] = []
    mean_cat_jacc_list: List[float] = []
    global_counts = label_space.new_counts()

    updated_items: List[Dict[str, Any]] = []

    # Score tracking for irrelevant and TN samples
    irrelevant_scores: List[float] = []
    label_score_data = _new_label_score_data()
    tn_scores: List[float] = []
    tn_label_score_data = _new_label_score_data()

    # SBERT similarity tracking for irrelevant and TN samples
    irrelevant_sbert_scores: List[float] = []
    label_sbert_data = _new_label_score_data()
    tn_sbert_scores: List[float] = []
    tn_label_sbert_data = _new_label_score_data()

    # R@k counts
    r_at_k_counts = {"0.3": 0, "0.5": 0, "0.7": 0}

    # Hardness level accumulators
    hardness_level_stats: Dict[str, Dict[str, int]] = {
        level: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for level in ("original", "weak", "moderate", "strong")
    }
    hardness_level_metrics_data: Dict[str, Dict[str, Any]] = {
        level: _new_hardness_data()
        for level in ("original", "weak", "moderate", "strong")
    }

    # Category-based irrelevant classification
    category_classification: Dict[str, Dict[str, int]] = {
        label: {"correct": 0, "incorrect": 0} for label in label_space.labels
    }

    def bool_to_rel(b: bool) -> str:
        return "relevant" if b else "irrelevant"

    def first_span(gt_timestamp: Optional[List]) -> Optional[List[float]]:
        if not gt_timestamp:
            return None
        if len(gt_timestamp) == 2 and isinstance(gt_timestamp[0], (int, float)):
            return [float(gt_timestamp[0]), float(gt_timestamp[1])]
        return None

    for ex in split_data:
        item_copy = ex.copy()

        gt_relevance = bool_to_rel(bool(ex.get("gt_relevant", False)))
        gt_span = first_span(ex.get("gt_timestamp", []))
        raw_gt_response = ex.get("gt_output", "") or ""
        gt_categories = ex.get("gt_categories", []) or []
        raw_model_output = ex.get("model_output", "") or ""
        hardness_level = ex.get("hardness_level", "")

        out = evaluate_three_stage(
            generated_response=raw_model_output,
            gt_response=raw_gt_response,
            gt_relevance=gt_relevance,
            gt_span=gt_span,
            gt_categories=gt_categories,
            cached_classification_correct=ex.get("classification_correct"),
            cached_iou=ex.get("iou"),
            cached_pred_categories=ex.get("pred_categories"),
        )

        # Determine predicted relevance from classification result
        if out.get("classification_result"):
            pred_relevance = gt_relevance
        else:
            pred_relevance = "irrelevant" if gt_relevance == "relevant" else "relevant"

        # Cache computed results back onto item
        item_copy["classification_correct"] = (gt_relevance == pred_relevance)
        item_copy["iou"] = float(out.get("iou", 0.0))
        item_copy["pred_categories"] = out.get("pred_categories", [])
        if "sbert_similarity" in out:
            item_copy["sbert_similarity"] = float(out["sbert_similarity"])

        # Update confusion matrix
        if gt_relevance == "relevant" and pred_relevance == "relevant":
            TP += 1; conf_key = "TP"
        elif gt_relevance == "relevant" and pred_relevance == "irrelevant":
            FN += 1; conf_key = "FN"
        elif gt_relevance == "irrelevant" and pred_relevance == "relevant":
            FP += 1; conf_key = "FP"
        else:
            TN += 1; conf_key = "TN"

        if hardness_level in hardness_level_stats:
            hardness_level_stats[hardness_level][conf_key] += 1

        # Category-level irrelevant classification tracking
        if gt_relevance == "irrelevant" and gt_categories:
            is_correct = (pred_relevance == "irrelevant")
            for label in gt_categories:
                if label in category_classification:
                    key = "correct" if is_correct else "incorrect"
                    category_classification[label][key] += 1

        # IoU and R@k
        iou_val = float(out.get("iou", 0.0))
        iou_all.append(iou_val)
        for thr in ("0.3", "0.5", "0.7"):
            if iou_val >= float(thr):
                r_at_k_counts[thr] += 1

        # Category Jaccard
        jacc = out.get("jaccard_categories")
        if jacc is not None:
            mean_cat_jacc_list.append(float(jacc))
            per_label_counts = out.get("per_label_counts")
            if isinstance(per_label_counts, dict):
                for lab in label_space.labels:
                    c = per_label_counts.get(lab)
                    if not c:
                        continue
                    global_counts[lab]["tp"] += int(c.get("tp", 0))
                    global_counts[lab]["fp"] += int(c.get("fp", 0))
                    global_counts[lab]["fn"] += int(c.get("fn", 0))

        # Reasoning / SBERT scores for irrelevant samples
        if gt_relevance == "irrelevant":
            score = out.get("score", 0.0)
            irrelevant_scores.append(float(score))
            if gt_categories:
                weight = 1.0 / len(gt_categories)
                for label in gt_categories:
                    if label in label_score_data:
                        label_score_data[label]["scores"].append(float(score))
                        label_score_data[label]["weights"].append(weight)

            if pred_relevance == "irrelevant":
                tn_scores.append(float(score))
                if gt_categories:
                    weight = 1.0 / len(gt_categories)
                    for label in gt_categories:
                        if label in tn_label_score_data:
                            tn_label_score_data[label]["scores"].append(float(score))
                            tn_label_score_data[label]["weights"].append(weight)

            sbert_score = out.get("sbert_similarity", ex.get("sbert_similarity"))
            if sbert_score is not None:
                irrelevant_sbert_scores.append(float(sbert_score))
                if gt_categories:
                    weight = 1.0 / len(gt_categories)
                    for label in gt_categories:
                        if label in label_sbert_data:
                            label_sbert_data[label]["scores"].append(float(sbert_score))
                            label_sbert_data[label]["weights"].append(weight)

                if pred_relevance == "irrelevant":
                    tn_sbert_scores.append(float(sbert_score))
                    if gt_categories:
                        weight = 1.0 / len(gt_categories)
                        for label in gt_categories:
                            if label in tn_label_sbert_data:
                                tn_label_sbert_data[label]["scores"].append(float(sbert_score))
                                tn_label_sbert_data[label]["weights"].append(weight)

        # Hardness-level detailed tracking
        if hardness_level in hardness_level_metrics_data:
            diff_data = hardness_level_metrics_data[hardness_level]
            diff_data["iou_all"].append(iou_val)
            for thr in ("0.3", "0.5", "0.7"):
                if iou_val >= float(thr):
                    diff_data["r_at_k_counts"][thr] += 1

            if jacc is not None:
                diff_data["jaccard_list"].append(float(jacc))
                per_label_counts = out.get("per_label_counts")
                if isinstance(per_label_counts, dict):
                    for lab in label_space.labels:
                        c = per_label_counts.get(lab)
                        if not c:
                            continue
                        diff_data["counts"][lab]["tp"] += int(c.get("tp", 0))
                        diff_data["counts"][lab]["fp"] += int(c.get("fp", 0))
                        diff_data["counts"][lab]["fn"] += int(c.get("fn", 0))

            if gt_relevance == "irrelevant":
                score = out.get("score", 0.0)
                diff_data["reasoning_scores"].append(float(score))
                if pred_relevance == "irrelevant":
                    diff_data["tn_reasoning_scores"].append(float(score))

                sbert_score = out.get("sbert_similarity", ex.get("sbert_similarity"))
                if sbert_score is not None:
                    diff_data["sbert_scores"].append(float(sbert_score))
                    if pred_relevance == "irrelevant":
                        diff_data["tn_sbert_scores"].append(float(sbert_score))

        updated_items.append(item_copy)

        # Update shared progress counter
        global _progress_counter
        if _progress_counter is not None:
            with _progress_counter.get_lock():
                _progress_counter.value += 1

    # --- Compute final metrics for this split ---

    overall_cls = compute_binary_metrics(TP, FP, FN, TN)
    rel_metrics = overall_cls
    irrel_metrics = compute_binary_metrics(TN, FN, FP, TP)

    acc_by_gt = {
        "relevant": (TP / (TP + FN)) if (TP + FN) else 0.0,
        "irrelevant": (TN / (TN + FP)) if (TN + FP) else 0.0,
    }

    mean_iou_all = sum(iou_all) / len(iou_all) if iou_all else 0.0
    mean_cat_jacc = sum(mean_cat_jacc_list) / len(mean_cat_jacc_list) if mean_cat_jacc_list else 0.0

    r_at_k_metrics = {
        f"R@{thr}": (r_at_k_counts[thr] / len(iou_all) * 100) if iou_all else 0.0
        for thr in ("0.3", "0.5", "0.7")
    }

    mean_tn_score = sum(tn_scores) / len(tn_scores) if tn_scores else 0.0
    mean_irrelevant_score = (
        mean_tn_score * len(tn_scores) / len(irrelevant_scores)
        if irrelevant_scores else 0.0
    )

    mean_tn_sbert = sum(tn_sbert_scores) / len(tn_sbert_scores) if tn_sbert_scores else 0.0
    mean_irrelevant_sbert = (
        sum(tn_sbert_scores) / len(irrelevant_sbert_scores)
        if irrelevant_sbert_scores else 0.0
    )

    label_scores = compute_label_score_stats(label_score_data, label_space.labels)
    tn_label_scores = compute_label_score_stats(tn_label_score_data, label_space.labels)
    label_sbert_scores = compute_label_score_stats(label_sbert_data, label_space.labels)
    tn_label_sbert_scores = compute_label_score_stats(tn_label_sbert_data, label_space.labels)

    # Hardness-level metrics
    hardness_level_metrics = _compute_hardness_metrics(
        hardness_level_stats, hardness_level_metrics_data
    )

    # Category-based irrelevant classification
    category_irrelevant_metrics = {
        label: {
            "correct": counts["correct"],
            "incorrect": counts["incorrect"],
            "total": counts["correct"] + counts["incorrect"],
            "accuracy": (
                counts["correct"] / (counts["correct"] + counts["incorrect"])
                if (counts["correct"] + counts["incorrect"]) > 0 else 0.0
            ),
        }
        for label, counts in category_classification.items()
    }

    return {
        "num_items": n,
        "relevance": {
            "overall_accuracy": overall_cls["accuracy"],
            "per_class": {
                "relevant": {
                    "precision": rel_metrics["precision"],
                    "recall": rel_metrics["recall"],
                    "f1": rel_metrics["f1"],
                    "accuracy_within_gt_relevance": acc_by_gt["relevant"],
                },
                "irrelevant": {
                    "precision": irrel_metrics["precision"],
                    "recall": irrel_metrics["recall"],
                    "f1": irrel_metrics["f1"],
                    "accuracy_within_gt_relevance": acc_by_gt["irrelevant"],
                },
            },
            "confusion": {"TP": TP, "FP": FP, "FN": FN, "TN": TN},
        },
        "RA-IoU": {"mean_all": mean_iou_all, **r_at_k_metrics},
        "RT-IoU": mean_cat_jacc,
        "irrelevant_samples": {
            "mean_reasoning_score": mean_irrelevant_score,
            "count": len(irrelevant_scores),
            "label_reasoning_scores": label_scores,
            "raw_scores": irrelevant_scores,
            "raw_label_score_data": label_score_data,
            "mean_sbert_similarity": mean_irrelevant_sbert,
            "sbert_count": len(irrelevant_sbert_scores),
            "label_sbert_scores": label_sbert_scores,
            "raw_sbert_scores": irrelevant_sbert_scores,
            "raw_label_sbert_data": label_sbert_data,
        },
        "true_negative_samples": {
            "mean_reasoning_score": mean_tn_score,
            "count": len(tn_scores),
            "label_reasoning_scores": tn_label_scores,
            "raw_scores": tn_scores,
            "raw_label_score_data": tn_label_score_data,
            "mean_sbert_similarity": mean_tn_sbert,
            "sbert_count": len(tn_sbert_scores),
            "label_sbert_scores": tn_label_sbert_scores,
            "raw_sbert_scores": tn_sbert_scores,
            "raw_label_sbert_data": tn_label_sbert_data,
        },
        "hardness_level_analysis": {
            "metrics_by_hardness_level": hardness_level_metrics,
            "raw_hardness_level_stats": hardness_level_stats,
            "raw_hardness_level_metrics_data": hardness_level_metrics_data,
        },
        "raw_temporal_data": {"r_at_k_counts": r_at_k_counts},
        "category_irrelevant_classification": {
            "metrics_by_category": category_irrelevant_metrics,
            "raw_category_stats": category_classification,
        },
        "updated_items": updated_items,
    }


def process_split(
    split_data: List[Dict[str, Any]],
    split_idx: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Process a single data split without progress tracking."""
    return process_split_with_progress(split_data, split_idx, verbose=verbose)


# -----------------------------------------------------------------------
# Hardness-level metric computation (shared between split and merge)
# -----------------------------------------------------------------------

def _compute_hardness_metrics(
    hardness_level_stats: Dict[str, Dict[str, int]],
    hardness_level_metrics_data: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute per-hardness-level metrics from raw counts and detailed data."""
    result = {}
    for diff, counts in hardness_level_stats.items():
        tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
        total = tp + fp + fn + tn
        diff_data = hardness_level_metrics_data[diff]

        if total == 0:
            result[diff] = {
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "confusion": counts, "total": 0,
                "RA-IoU": {"mean_all": 0.0, "R@0.3": 0.0, "R@0.5": 0.0, "R@0.7": 0.0},
                "RT-IoU": 0.0,
                "irrelevant_samples": {
                    "mean_reasoning_score": 0.0, "count": 0,
                    "mean_sbert_similarity": 0.0, "sbert_count": 0,
                },
            }
            continue

        if diff == "original":
            metrics = compute_binary_metrics(tp, fp, fn, tn)
        else:
            # For irrelevant levels: TP_irrel = TN_relevance
            metrics = compute_binary_metrics(tn, fn, fp, tp)

        iou_all_diff = diff_data["iou_all"]
        mean_iou_all_diff = sum(iou_all_diff) / len(iou_all_diff) if iou_all_diff else 0.0

        r_at_k_diff = {
            f"R@{thr}": (diff_data["r_at_k_counts"][thr] / total * 100) if total > 0 else 0.0
            for thr in ("0.3", "0.5", "0.7")
        }

        jaccard_list = diff_data["jaccard_list"]
        mean_jaccard = sum(jaccard_list) / len(jaccard_list) if jaccard_list else 0.0

        reasoning_scores = diff_data["reasoning_scores"]
        tn_reasoning_scores = diff_data["tn_reasoning_scores"]
        sbert_scores = diff_data["sbert_scores"]
        tn_sbert_scores = diff_data["tn_sbert_scores"]

        mean_reasoning = (
            sum(tn_reasoning_scores) / len(reasoning_scores)
            if reasoning_scores else 0.0
        )
        mean_sbert = (
            sum(tn_sbert_scores) / len(sbert_scores)
            if sbert_scores else 0.0
        )

        result[diff] = {
            **metrics,
            "confusion": counts,
            "total": total,
            "RA-IoU": {"mean_all": mean_iou_all_diff, **r_at_k_diff},
            "RT-IoU": mean_jaccard,
            "irrelevant_samples": {
                "mean_reasoning_score": mean_reasoning,
                "count": len(reasoning_scores),
                "mean_sbert_similarity": mean_sbert,
                "sbert_count": len(sbert_scores),
            },
        }
    return result


# -----------------------------------------------------------------------
# Merge metrics from multiple splits
# -----------------------------------------------------------------------

def merge_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge metrics dicts from multiple parallel splits into a single result."""
    if not metrics_list:
        raise ValueError("No metrics to merge")
    if len(metrics_list) == 1:
        return metrics_list[0]

    # Collect updated items from all splits
    all_updated_items: List[Dict[str, Any]] = []
    for m in metrics_list:
        all_updated_items.extend(m.get("updated_items", []))

    # Aggregate confusion counts
    total_tp = sum(m["relevance"]["confusion"]["TP"] for m in metrics_list)
    total_fp = sum(m["relevance"]["confusion"]["FP"] for m in metrics_list)
    total_fn = sum(m["relevance"]["confusion"]["FN"] for m in metrics_list)
    total_tn = sum(m["relevance"]["confusion"]["TN"] for m in metrics_list)
    total_items = sum(m["num_items"] for m in metrics_list)

    # Relevance metrics
    total = total_tp + total_fp + total_fn + total_tn
    overall_acc = (total_tp + total_tn) / total if total else 0.0

    rel_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    rel_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    rel_f1 = 2 * rel_prec * rel_rec / (rel_prec + rel_rec) if (rel_prec + rel_rec) else 0.0

    irrel_prec = total_tn / (total_tn + total_fn) if (total_tn + total_fn) else 0.0
    irrel_rec = total_tn / (total_tn + total_fp) if (total_tn + total_fp) else 0.0
    irrel_f1 = 2 * irrel_prec * irrel_rec / (irrel_prec + irrel_rec) if (irrel_prec + irrel_rec) else 0.0

    acc_rel = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    acc_irrel = total_tn / (total_tn + total_fp) if (total_tn + total_fp) else 0.0

    # RA-IoU (weighted by num_items)
    total_iou_all = sum(m["RA-IoU"]["mean_all"] * m["num_items"] for m in metrics_list)
    mean_iou_all = total_iou_all / total_items if total_items else 0.0

    merged_r_at_k_counts = {"0.3": 0, "0.5": 0, "0.7": 0}
    for m in metrics_list:
        raw_r = m.get("raw_temporal_data", {}).get("r_at_k_counts", {})
        for thr in ("0.3", "0.5", "0.7"):
            merged_r_at_k_counts[thr] += raw_r.get(thr, 0)
    merged_r_at_k_metrics = {
        f"R@{thr}": (merged_r_at_k_counts[thr] / total_items * 100) if total_items else 0.0
        for thr in ("0.3", "0.5", "0.7")
    }

    # RT-IoU (weighted average)
    mean_jaccard = sum(m["RT-IoU"] * m["num_items"] for m in metrics_list) / total_items if total_items else 0.0

    # --- Merge irrelevant sample raw scores ---
    all_irrelevant_scores: List[float] = []
    merged_label_score_data = _new_label_score_data()
    all_irrelevant_sbert_scores: List[float] = []
    merged_label_sbert_data = _new_label_score_data()

    for m in metrics_list:
        irrel = m.get("irrelevant_samples", {})
        all_irrelevant_scores.extend(irrel.get("raw_scores", []))
        _extend_label_data(merged_label_score_data, irrel.get("raw_label_score_data", {}))
        all_irrelevant_sbert_scores.extend(irrel.get("raw_sbert_scores", []))
        _extend_label_data(merged_label_sbert_data, irrel.get("raw_label_sbert_data", {}))

    # --- Merge TN sample raw scores ---
    all_tn_scores: List[float] = []
    merged_tn_label_score_data = _new_label_score_data()
    all_tn_sbert_scores: List[float] = []
    merged_tn_label_sbert_data = _new_label_score_data()

    for m in metrics_list:
        tn = m.get("true_negative_samples", {})
        all_tn_scores.extend(tn.get("raw_scores", []))
        _extend_label_data(merged_tn_label_score_data, tn.get("raw_label_score_data", {}))
        all_tn_sbert_scores.extend(tn.get("raw_sbert_scores", []))
        _extend_label_data(merged_tn_label_sbert_data, tn.get("raw_label_sbert_data", {}))

    # Compute aggregate scores
    mean_tn_score = sum(all_tn_scores) / len(all_tn_scores) if all_tn_scores else 0.0
    mean_irrelevant_score = (
        mean_tn_score * len(all_tn_scores) / len(all_irrelevant_scores)
        if all_irrelevant_scores else 0.0
    )

    mean_tn_sbert = sum(all_tn_sbert_scores) / len(all_tn_sbert_scores) if all_tn_sbert_scores else 0.0
    mean_irrelevant_sbert = (
        sum(all_tn_sbert_scores) / len(all_irrelevant_sbert_scores)
        if all_irrelevant_sbert_scores else 0.0
    )

    merged_label_scores = compute_label_score_stats(merged_label_score_data, label_space.labels)
    merged_tn_label_scores = compute_label_score_stats(merged_tn_label_score_data, label_space.labels)
    merged_label_sbert_scores = compute_label_score_stats(merged_label_sbert_data, label_space.labels)
    merged_tn_label_sbert_scores = compute_label_score_stats(merged_tn_label_sbert_data, label_space.labels)

    # --- Merge hardness level stats ---
    merged_hardness_stats: Dict[str, Dict[str, int]] = {
        level: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for level in ("original", "weak", "moderate", "strong")
    }
    merged_hardness_metrics_data: Dict[str, Dict[str, Any]] = {
        level: _new_hardness_data()
        for level in ("original", "weak", "moderate", "strong")
    }

    for m in metrics_list:
        diff_analysis = m.get("hardness_level_analysis", {})

        for diff, stats in diff_analysis.get("raw_hardness_level_stats", {}).items():
            for key in ("TP", "FP", "FN", "TN"):
                merged_hardness_stats[diff][key] += stats.get(key, 0)

        for diff, src in diff_analysis.get("raw_hardness_level_metrics_data", {}).items():
            tgt = merged_hardness_metrics_data[diff]
            tgt["iou_all"].extend(src.get("iou_all", []))
            tgt["jaccard_list"].extend(src.get("jaccard_list", []))
            tgt["reasoning_scores"].extend(src.get("reasoning_scores", []))
            tgt["tn_reasoning_scores"].extend(src.get("tn_reasoning_scores", []))
            tgt["sbert_scores"].extend(src.get("sbert_scores", []))
            tgt["tn_sbert_scores"].extend(src.get("tn_sbert_scores", []))
            for thr in ("0.3", "0.5", "0.7"):
                tgt["r_at_k_counts"][thr] += src.get("r_at_k_counts", {}).get(thr, 0)
            for lab in label_space.labels:
                src_c = src.get("counts", {}).get(lab)
                if src_c:
                    tgt["counts"][lab]["tp"] += src_c.get("tp", 0)
                    tgt["counts"][lab]["fp"] += src_c.get("fp", 0)
                    tgt["counts"][lab]["fn"] += src_c.get("fn", 0)

    merged_hardness_level_metrics = _compute_hardness_metrics(
        merged_hardness_stats, merged_hardness_metrics_data
    )

    # --- Merge category irrelevant classification ---
    merged_cat_cls: Dict[str, Dict[str, int]] = {
        label: {"correct": 0, "incorrect": 0} for label in label_space.labels
    }
    for m in metrics_list:
        for label, counts in m.get("category_irrelevant_classification", {}).get("raw_category_stats", {}).items():
            if label in merged_cat_cls:
                merged_cat_cls[label]["correct"] += counts.get("correct", 0)
                merged_cat_cls[label]["incorrect"] += counts.get("incorrect", 0)

    merged_category_metrics = {
        label: {
            "correct": c["correct"],
            "incorrect": c["incorrect"],
            "total": c["correct"] + c["incorrect"],
            "accuracy": (
                c["correct"] / (c["correct"] + c["incorrect"])
                if (c["correct"] + c["incorrect"]) > 0 else 0.0
            ),
        }
        for label, c in merged_cat_cls.items()
    }

    return {
        "num_items": total_items,
        "relevance": {
            "overall_accuracy": overall_acc,
            "per_class": {
                "relevant": {
                    "precision": rel_prec,
                    "recall": rel_rec,
                    "f1": rel_f1,
                    "accuracy_within_gt_relevance": acc_rel,
                },
                "irrelevant": {
                    "precision": irrel_prec,
                    "recall": irrel_rec,
                    "f1": irrel_f1,
                    "accuracy_within_gt_relevance": acc_irrel,
                },
            },
            "confusion": {"TP": total_tp, "FP": total_fp, "FN": total_fn, "TN": total_tn},
        },
        "RA-IoU": {"mean_all": mean_iou_all, **merged_r_at_k_metrics},
        "RT-IoU": mean_jaccard,
        "irrelevant_samples": {
            "mean_reasoning_score": mean_irrelevant_score,
            "count": len(all_irrelevant_scores),
            "label_reasoning_scores": merged_label_scores,
            "mean_sbert_similarity": mean_irrelevant_sbert,
            "sbert_count": len(all_irrelevant_sbert_scores),
            "label_sbert_scores": merged_label_sbert_scores,
        },
        "true_negative_samples": {
            "mean_reasoning_score": mean_tn_score,
            "count": len(all_tn_scores),
            "label_reasoning_scores": merged_tn_label_scores,
            "mean_sbert_similarity": mean_tn_sbert,
            "sbert_count": len(all_tn_sbert_scores),
            "label_sbert_scores": merged_tn_label_sbert_scores,
        },
        "hardness_level_analysis": {
            "metrics_by_hardness_level": merged_hardness_level_metrics,
        },
        "category_irrelevant_classification": {
            "metrics_by_category": merged_category_metrics,
        },
        "updated_items": all_updated_items,
    }


# -----------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------

def _extend_label_data(
    target: Dict[str, Dict[str, List[float]]],
    source: Dict[str, Dict[str, List[float]]],
) -> None:
    """Extend target label score data with entries from source."""
    for label in label_space.labels:
        src = source.get(label)
        if src:
            target[label]["scores"].extend(src.get("scores", []))
            target[label]["weights"].extend(src.get("weights", []))


def split_data(
    data: List[Dict[str, Any]], num_splits: int
) -> List[List[Dict[str, Any]]]:
    """Split data into roughly equal chunks for parallel processing."""
    n = len(data)
    chunk_size = (n + num_splits - 1) // num_splits
    return [data[i:i + chunk_size] for i in range(0, n, chunk_size)]
