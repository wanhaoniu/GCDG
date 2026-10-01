"""Utility functions for dependency-guided minimal intervention planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


EPS = 1e-9


@dataclass(frozen=True)
class PlannerParams:
    tau_any: float = 0.5
    tau_sufficient: float = 0.5
    tau_app: float = 0.5
    tau_lift: float = 0.5
    alpha: float = 0.25
    beta: float = 0.10
    gamma: float = 0.75
    eta: float = 0.10
    max_steps: int = 5
    budget_residual_weight: float = 1.0
    budget_residual_ratio_weight: float = 0.0
    budget_removal_weight: float = 0.10
    budget_removed_reward: float = 0.0
    budget_coverage_reward: float = 0.0
    mbs_residual_threshold: float = 0.5
    topk_min_strength: float = 0.0
    candidate_grasp_min_score_ratio: float = 0.0
    candidate_grasp_top_k: int = 0
    use_typed_dependency: bool = True
    use_sufficient_dependency: bool = False
    use_grasp_score: bool = True
    mechanical_search_target_score_thresh: float = 0.2
    xray_target_score_thresh: float = 0.2
    xray_overlap_weight: float = 2.0
    xray_near_weight: float = 0.75
    xray_front_weight: float = 0.75
    xray_area_weight: float = 0.50
    swept_volume_iou_threshold: float = 0.01
    swept_volume_clearance_threshold: float = 0.0
    swept_volume_approach_weight: float = 1.0
    swept_volume_lift_weight: float = 1.0
    swept_volume_clearance_weight: float = 0.25
    swept_volume_distance_weight: float = 0.05

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PlannerParams":
        payload = payload or {}
        return cls(
            tau_any=float(payload.get("tau_any", 0.5)),
            tau_sufficient=float(payload.get("tau_sufficient", payload.get("tau_any", 0.5))),
            tau_app=float(payload.get("tau_app", 0.5)),
            tau_lift=float(payload.get("tau_lift", 0.5)),
            alpha=float(payload.get("alpha", 0.25)),
            beta=float(payload.get("beta", 0.10)),
            gamma=float(payload.get("gamma", 0.75)),
            eta=float(payload.get("eta", 0.10)),
            max_steps=int(payload.get("max_steps", 5)),
            budget_residual_weight=float(payload.get("budget_residual_weight", 1.0)),
            budget_residual_ratio_weight=float(payload.get("budget_residual_ratio_weight", 0.0)),
            budget_removal_weight=float(payload.get("budget_removal_weight", 0.10)),
            budget_removed_reward=float(payload.get("budget_removed_reward", 0.0)),
            budget_coverage_reward=float(payload.get("budget_coverage_reward", 0.0)),
            mbs_residual_threshold=float(payload.get("mbs_residual_threshold", 0.5)),
            topk_min_strength=float(payload.get("topk_min_strength", 0.0)),
            candidate_grasp_min_score_ratio=float(payload.get("candidate_grasp_min_score_ratio", 0.0)),
            candidate_grasp_top_k=int(payload.get("candidate_grasp_top_k", 0)),
            use_typed_dependency=bool(payload.get("use_typed_dependency", True)),
            use_sufficient_dependency=bool(payload.get("use_sufficient_dependency", False)),
            use_grasp_score=bool(payload.get("use_grasp_score", True)),
            mechanical_search_target_score_thresh=float(
                payload.get("mechanical_search_target_score_thresh", 0.2)
            ),
            xray_target_score_thresh=float(payload.get("xray_target_score_thresh", 0.2)),
            xray_overlap_weight=float(payload.get("xray_overlap_weight", 2.0)),
            xray_near_weight=float(payload.get("xray_near_weight", 0.75)),
            xray_front_weight=float(payload.get("xray_front_weight", 0.75)),
            xray_area_weight=float(payload.get("xray_area_weight", 0.50)),
            swept_volume_iou_threshold=float(payload.get("swept_volume_iou_threshold", 0.01)),
            swept_volume_clearance_threshold=float(payload.get("swept_volume_clearance_threshold", 0.0)),
            swept_volume_approach_weight=float(payload.get("swept_volume_approach_weight", 1.0)),
            swept_volume_lift_weight=float(payload.get("swept_volume_lift_weight", 1.0)),
            swept_volume_clearance_weight=float(payload.get("swept_volume_clearance_weight", 0.25)),
            swept_volume_distance_weight=float(payload.get("swept_volume_distance_weight", 0.05)),
        )


def edge_probability(edge: dict[str, float], key: str) -> float:
    aliases = {
        "any": ("any", "dep_any", "p_any"),
        "sufficient": ("sufficient", "dep_sufficient", "p_sufficient"),
        "app": ("app", "approach", "dep_approach", "dep_app", "p_app"),
        "lift": ("lift", "dep_lift", "p_lift"),
    }
    for name in aliases[key]:
        if name in edge:
            return float(edge[name])
    return 0.0


def dependency_strength(
    edge: dict[str, float],
    *,
    use_typed_dependency: bool,
    use_sufficient_dependency: bool = False,
) -> float:
    if use_sufficient_dependency:
        return edge_probability(edge, "sufficient")
    if use_typed_dependency:
        return edge_probability(edge, "app") + edge_probability(edge, "lift")
    return edge_probability(edge, "any")


def compute_blocker_set(
    grasp_edges: dict[str, dict[str, float]],
    remaining_objects: Iterable[str],
    params: PlannerParams,
) -> list[str]:
    remaining = set(remaining_objects)
    blockers: list[str] = []
    for object_id, edge in grasp_edges.items():
        if object_id not in remaining:
            continue
        if params.use_sufficient_dependency:
            is_blocker = edge_probability(edge, "sufficient") > params.tau_sufficient
        elif params.use_typed_dependency:
            is_blocker = (
                edge_probability(edge, "app") > params.tau_app
                or edge_probability(edge, "lift") > params.tau_lift
            )
        else:
            is_blocker = edge_probability(edge, "any") > params.tau_any
        if is_blocker:
            blockers.append(object_id)
    return sorted(blockers)


def risk_score(
    grasp_edges: dict[str, dict[str, float]],
    remaining_objects: Iterable[str],
    *,
    use_typed_dependency: bool,
    use_sufficient_dependency: bool = False,
) -> float:
    remaining = set(remaining_objects)
    total = 0.0
    for object_id, edge in grasp_edges.items():
        if object_id not in remaining:
            continue
        total += dependency_strength(
            edge,
            use_typed_dependency=use_typed_dependency,
            use_sufficient_dependency=use_sufficient_dependency,
        )
    return float(total)


def grasp_selection_score(
    *,
    num_blockers: int,
    grasp_score: float,
    risk: float,
    params: PlannerParams,
) -> float:
    score_bonus = params.alpha * float(grasp_score) if params.use_grasp_score else 0.0
    return float(num_blockers - score_bonus + params.beta * risk)


def blocker_ranking_score(
    *,
    dependency_strength_value: float,
    blocker_graspability: float = 1.0,
    removal_cost: float = 0.0,
    params: PlannerParams,
) -> float:
    return float(
        params.gamma * dependency_strength_value
        + (1.0 - params.gamma) * blocker_graspability
        - params.eta * removal_cost
    )


def set_comparison(predicted: Iterable[str], oracle: Iterable[str]) -> dict[str, float]:
    pred = set(predicted)
    gt = set(oracle)
    inter = len(pred & gt)
    union = len(pred | gt)
    precision = inter / max(len(pred), 1)
    recall = inter / max(len(gt), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    iou = inter / union if union > 0 else 1.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "intersection": float(inter),
        "pred_size": float(len(pred)),
        "oracle_size": float(len(gt)),
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))
