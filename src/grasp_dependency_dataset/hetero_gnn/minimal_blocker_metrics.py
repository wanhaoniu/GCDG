"""Metrics for oracle minimal-blocker-set recovery.

These metrics use `planning_labels[].minimal_blocker_set` as ground truth. They
are intentionally separate from edge-derived blocker-set metrics, whose ground
truth is single-object dependency labels.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


_SOLVED_STATUSES = {"already_feasible", "solved_within_depth"}
_EPS = 1e-9


@dataclass(frozen=True)
class MinimalBlockerRecord:
    """Oracle minimal blocker set for one sample/grasp pair."""

    sample_id: str
    grasp_id: str
    minimal_blocker_set: frozenset[str]
    status: str


def minimal_blocker_lookup(
    planning_labels_by_sample: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[tuple[str, str], MinimalBlockerRecord]:
    """Return solved per-grasp oracle MBS labels keyed by `(sample_id, grasp_id)`."""

    lookup: dict[tuple[str, str], MinimalBlockerRecord] = {}
    for sample_id, planning_labels in planning_labels_by_sample.items():
        for label in planning_labels:
            grasp_id = str(label.get("grasp_id") or "")
            status = str(label.get("status") or "")
            if not grasp_id or status not in _SOLVED_STATUSES:
                continue
            blockers = frozenset(str(item) for item in label.get("minimal_blocker_set", []) if item is not None)
            lookup[(str(sample_id), grasp_id)] = MinimalBlockerRecord(
                sample_id=str(sample_id),
                grasp_id=grasp_id,
                minimal_blocker_set=blockers,
                status=status,
            )
    return lookup


def ranked_objects_by_score(object_scores: Mapping[str, float]) -> list[str]:
    """Rank objects by descending dependency score with stable id tie-breaking."""

    return [
        object_id
        for object_id, _score in sorted(
            object_scores.items(),
            key=lambda item: (-float(item[1]), str(item[0])),
        )
    ]


def minimal_blocker_set_metrics(
    *,
    edge_sample_ids: list[str],
    edge_object_ids: list[str],
    edge_grasp_ids: list[str],
    y_score: np.ndarray,
    planning_labels_by_sample: Mapping[str, Iterable[Mapping[str, Any]]],
    mask: np.ndarray | None = None,
    threshold: float = 0.5,
    score_label_index: int = 0,
    recall_ks: Iterable[int] = (1, 2, 3, 5),
    y_true: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare thresholded/ranked edge scores against oracle minimal blocker sets."""

    scores = _score_vector(y_score, score_label_index)
    keep = np.ones((len(edge_sample_ids),), dtype=bool) if mask is None else np.asarray(mask).astype(bool)
    recall_ks = tuple(sorted({int(k) for k in recall_ks if int(k) > 0}))
    if not recall_ks:
        recall_ks = (1,)

    score_groups: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    edge_label_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    true_labels = _score_vector(y_true, score_label_index) if y_true is not None else None
    for idx, valid in enumerate(keep.tolist()):
        if not valid:
            continue
        key = (str(edge_sample_ids[idx]), str(edge_grasp_ids[idx]))
        object_id = str(edge_object_ids[idx])
        score_groups[key][object_id] = float(scores[idx])
        if true_labels is not None and float(true_labels[idx]) >= 0.5:
            edge_label_groups[key].add(object_id)

    oracle_lookup = minimal_blocker_lookup(planning_labels_by_sample)
    per_grasp: list[dict[str, Any]] = []
    edge_label_disagreement_count = 0
    for key, record in sorted(oracle_lookup.items()):
        object_scores = score_groups.get(key, {})
        ranked = ranked_objects_by_score(object_scores)
        pred = frozenset(object_id for object_id, score in object_scores.items() if float(score) >= threshold)
        oracle = record.minimal_blocker_set
        comparison = _set_comparison(pred, oracle)
        recall_at = {
            f"recall_at_{k}": _oracle_recall_at_k(ranked, oracle, k)
            for k in recall_ks
        }
        if true_labels is not None and frozenset(edge_label_groups.get(key, set())) != oracle:
            edge_label_disagreement_count += 1
        per_grasp.append(
            {
                "sample_id": record.sample_id,
                "grasp_id": record.grasp_id,
                "status": record.status,
                "oracle_minimal_blocker_set": sorted(oracle),
                "predicted_blocker_set": sorted(pred),
                "ranked_objects": ranked,
                **comparison,
                **recall_at,
            }
        )

    return {
        "all_solved": _aggregate_section(per_grasp, recall_ks),
        "nontrivial": _aggregate_section(
            [item for item in per_grasp if int(item["oracle_size"]) > 0],
            recall_ks,
        ),
        "zero_blocker": _aggregate_section(
            [item for item in per_grasp if int(item["oracle_size"]) == 0],
            recall_ks,
        ),
        "per_grasp": per_grasp,
        "recall_ks": list(recall_ks),
        "num_oracle_records": len(oracle_lookup),
        "num_scored_groups": len(score_groups),
        "edge_label_disagreement_count": edge_label_disagreement_count,
    }


def _score_vector(values: np.ndarray | None, label_index: int) -> np.ndarray:
    if values is None:
        return np.zeros((0,), dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.shape[1] == 0:
        return np.zeros((arr.shape[0],), dtype=np.float32)
    idx = max(0, min(int(label_index), arr.shape[1] - 1))
    return arr[:, idx]


def _oracle_recall_at_k(ranked_objects: list[str], oracle: frozenset[str], k: int) -> float:
    if not oracle:
        return 1.0
    top_k = set(ranked_objects[: max(0, int(k))])
    return float(oracle.issubset(top_k))


def _set_comparison(predicted: frozenset[str], oracle: frozenset[str]) -> dict[str, float | int]:
    inter = len(predicted & oracle)
    union = len(predicted | oracle)
    if not predicted and not oracle:
        precision = recall = f1 = iou = 1.0
    else:
        precision = inter / max(len(predicted), 1)
        recall = inter / max(len(oracle), 1) if oracle else 1.0
        f1 = 2.0 * precision * recall / max(precision + recall, _EPS)
        iou = inter / union if union else 1.0
    return {
        "exact_match": float(predicted == oracle),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "pred_size": int(len(predicted)),
        "oracle_size": int(len(oracle)),
        "size_mae": float(abs(len(predicted) - len(oracle))),
        "excess": float(len(predicted - oracle)),
        "missing": float(len(oracle - predicted)),
    }


def _aggregate_section(items: list[dict[str, Any]], recall_ks: tuple[int, ...]) -> dict[str, float | int]:
    fields = [
        "exact_match",
        "precision",
        "recall",
        "f1",
        "iou",
        "size_mae",
        "excess",
        "missing",
        "pred_size",
        "oracle_size",
    ]
    out: dict[str, float | int] = {"count": len(items)}
    for field in fields:
        out[f"mean_{field}"] = _mean(float(item[field]) for item in items)
    for k in recall_ks:
        out[f"mean_recall_at_{k}"] = _mean(float(item[f"recall_at_{k}"]) for item in items)
    out["mean_recall_at_k"] = out[f"mean_recall_at_{recall_ks[-1]}"]
    return out


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))
