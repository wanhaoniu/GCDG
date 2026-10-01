"""Metrics for edge-level and grasp-level dependency evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


LABEL_NAMES = ["dep_progress_any", "dep_sufficient", "dep_approach", "dep_lift"]


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Tie-aware AP, exactly as used by the camera-ready experiment wrapper."""
    from sklearn.metrics import average_precision_score
    if not len(y_true) or not np.any(np.asarray(y_true) > 0):
        return 0.0
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks for ties.
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = ranks[order[start:end]].mean()
        start = end
    pos_ranks = ranks[: len(pos)]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, *, threshold: float) -> dict[str, float]:
    y_true = y_true.astype(np.int64)
    y_pred = (y_score >= threshold).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(y_true), 1)
    specificity = tn / max(tn + fp, 1)
    balanced_accuracy = 0.5 * (recall + specificity)
    positive_rate = int(y_true.sum()) / max(len(y_true), 1)
    best = best_f1_threshold(y_true, y_score)
    return {
        "accuracy": float(accuracy),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "positive_rate": float(positive_rate),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "best_threshold": best["threshold"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "best_f1": best["f1"],
        "ap": average_precision(y_true, y_score),
        "auc": roc_auc(y_true, y_score),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "positives": float(y_true.sum()),
        "total": float(len(y_true)),
    }


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Return the best F1 operating point on a validation/test vector.

    Dependency positives are very sparse, so a fixed 0.5 threshold can hide
    improvements in ranking quality. This sweep is used for model selection and
    reporting; deployment thresholds should still be calibrated on validation
    data, not on test data.
    """

    y_true = y_true.astype(np.int64)
    if len(y_true) == 0:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    score_thresholds = np.quantile(y_score, np.linspace(0.0, 1.0, 201, dtype=np.float32))
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99, dtype=np.float32),
                score_thresholds.astype(np.float32),
            ]
        )
    )
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for threshold in thresholds:
        y_pred = (y_score >= float(threshold)).astype(np.int64)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
    return best


def edge_level_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    threshold: float,
) -> dict[str, Any]:
    if mask is None:
        keep = np.ones(y_true.shape[0], dtype=bool)
    else:
        keep = mask.astype(bool)
    if keep.sum() == 0:
        return {"labels": {}, "overall": {}}
    yt = y_true[keep]
    ys = y_score[keep]
    num_labels = min(len(LABEL_NAMES), yt.shape[1], ys.shape[1])
    labels = {
        name: binary_metrics(yt[:, idx], ys[:, idx], threshold=threshold)
        for idx, name in enumerate(LABEL_NAMES[:num_labels])
    }
    overall = binary_metrics(yt[:, :num_labels].reshape(-1), ys[:, :num_labels].reshape(-1), threshold=threshold)
    return {"labels": labels, "overall": overall}


def blocker_set_metrics(
    edge_sample_ids: list[str],
    edge_object_ids: list[str],
    edge_grasp_ids: list[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    threshold: float,
) -> dict[str, Any]:
    groups: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: {"pred": set(), "gt": set()})
    keep = np.ones(y_true.shape[0], dtype=bool) if mask is None else mask.astype(bool)
    if y_true.shape[1] == 0 or y_score.shape[1] == 0:
        return {"count": 0, "mean_precision": 0.0, "mean_recall": 0.0, "mean_set_iou": 0.0, "per_grasp": []}
    for idx, valid in enumerate(keep.tolist()):
        if not valid:
            continue
        key = (edge_sample_ids[idx], edge_grasp_ids[idx])
        if y_score[idx, 0] >= threshold:
            groups[key]["pred"].add(edge_object_ids[idx])
        if y_true[idx, 0] >= 0.5:
            groups[key]["gt"].add(edge_object_ids[idx])

    per_grasp = []
    for (sample_id, grasp_id), sets in sorted(groups.items()):
        pred = sets["pred"]
        gt = sets["gt"]
        inter = len(pred & gt)
        union = len(pred | gt)
        precision = inter / max(len(pred), 1)
        recall = inter / max(len(gt), 1)
        iou = inter / union if union > 0 else 1.0
        per_grasp.append(
            {
                "sample_id": sample_id,
                "grasp_id": grasp_id,
                "predicted_blockers": sorted(pred),
                "gt_blockers": sorted(gt),
                "precision": float(precision),
                "recall": float(recall),
                "set_iou": float(iou),
            }
        )
    if not per_grasp:
        return {"count": 0, "mean_precision": 0.0, "mean_recall": 0.0, "mean_set_iou": 0.0, "per_grasp": []}
    return {
        "count": len(per_grasp),
        "mean_precision": float(np.mean([item["precision"] for item in per_grasp])),
        "mean_recall": float(np.mean([item["recall"] for item in per_grasp])),
        "mean_set_iou": float(np.mean([item["set_iou"] for item in per_grasp])),
        "per_grasp": per_grasp,
    }
