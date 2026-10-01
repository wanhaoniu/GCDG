"""Closed-loop dependency-guided minimal intervention planner."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from planners.planner_utils import (
    EPS,
    PlannerParams,
    blocker_ranking_score,
    compute_blocker_set,
    dependency_strength,
    grasp_selection_score,
    risk_score,
)


@dataclass
class PlannerSample:
    scene_id: str
    target_id: str
    object_ids: list[str]
    grasps: dict[str, dict[str, Any]]
    predicted_dependencies: dict[str, dict[str, dict[str, float]]]
    oracle_dependencies: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    object_costs: dict[str, float] = field(default_factory=dict)
    object_graspability: dict[str, float] = field(default_factory=dict)
    object_visible_area: dict[str, float] = field(default_factory=dict)
    object_target_iou: dict[str, float] = field(default_factory=dict)
    object_depth_order: dict[str, float] = field(default_factory=dict)
    object_grasp_geometry: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    oracle_min_by_grasp: dict[str, dict[str, Any]] = field(default_factory=dict)
    grasp_feasibility: dict[str, dict[str, Any]] = field(default_factory=dict)
    planning_summary: dict[str, Any] = field(default_factory=dict)


class DependencyGuidedPlanner:
    """Select target grasps and minimal removal actions from dependency graphs."""

    def __init__(self, params: PlannerParams, *, rng_seed: int = 7, verbose: bool = False) -> None:
        self.params = params
        self.rng = random.Random(rng_seed)
        self.verbose = verbose

    def plan(
        self,
        sample: PlannerSample,
        *,
        planner_type: str = "predicted_dependency",
        closed_loop: bool = True,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        planner_type = str(planner_type)
        if planner_type == "direct_target_grasp":
            result = self._plan_direct(sample)
        elif planner_type == "random_removal":
            result = self._plan_ranked_removals(sample, planner_type=planner_type, closed_loop=closed_loop)
        elif planner_type == "nearest_to_target":
            result = self._plan_ranked_removals(sample, planner_type=planner_type, closed_loop=closed_loop)
        elif planner_type == "highest_dependency":
            result = self._plan_highest_global_dependency(sample, closed_loop=closed_loop)
        elif planner_type in {"mechanical_search_largest_first", "mechanical_search_preempted_random"}:
            result = self._plan_mechanical_search(sample, planner_type=planner_type, closed_loop=closed_loop)
        elif planner_type == "xray_support_reduction":
            result = self._plan_xray_support_reduction(sample, closed_loop=closed_loop)
        elif planner_type == "swept_volume_clearing":
            result = self._plan_swept_volume_clearing(sample, closed_loop=closed_loop)
        elif planner_type == "oracle_dependency":
            result = self._plan_dependency(sample, dependency_source="oracle", closed_loop=closed_loop)
        elif planner_type == "oracle_minimal_blocker":
            result = self._plan_oracle_minimal(sample)
        elif planner_type == "topk_dependency":
            result = self._plan_topk_dependency(sample, closed_loop=closed_loop)
        elif planner_type == "minimal_blocker_set_planner":
            result = self._plan_minimal_blocker_set(sample, closed_loop=closed_loop)
        elif planner_type == "budgeted_dependency":
            result = self._plan_budgeted_dependency(sample, closed_loop=closed_loop)
        elif planner_type == "predicted_dependency":
            result = self._plan_dependency(sample, dependency_source="predicted", closed_loop=closed_loop)
        else:
            raise ValueError(f"Unknown planner_type: {planner_type}")

        result["planning_time_sec"] = float(time.perf_counter() - start)
        result["planner_type"] = planner_type
        result["mode"] = "closed_loop" if closed_loop else "open_loop"
        result.setdefault("scene_id", sample.scene_id)
        result.setdefault("target_id", sample.target_id)
        if self.verbose:
            print(
                f"[planner] {sample.scene_id}/{sample.target_id} {planner_type} "
                f"grasp={result.get('selected_grasp_id')} blockers={result.get('predicted_blocker_set')} "
                f"seq={result.get('removal_sequence')}"
            )
        return result

    def _dependency_graph(self, sample: PlannerSample, dependency_source: str) -> dict[str, dict[str, dict[str, float]]]:
        if dependency_source == "oracle":
            return sample.oracle_dependencies
        return sample.predicted_dependencies

    def _plan_direct(self, sample: PlannerSample) -> dict[str, Any]:
        grasp_id = self._best_grasp_by_score(sample)
        return {
            "scene_id": sample.scene_id,
            "target_id": sample.target_id,
            "selected_grasp_id": grasp_id,
            "selected_grasp_score": self._grasp_score(sample, grasp_id),
            "predicted_blocker_set": [],
            "initial_predicted_blocker_set": [],
            "removal_sequence": [],
            "num_removals": 0,
            "action": "target_grasp",
            "steps": [
                {
                    "step": 0,
                    "action": "target_grasp",
                    "selected_grasp_id": grasp_id,
                    "reason": "direct baseline does not remove blockers",
                }
            ],
        }

    def _plan_dependency(
        self,
        sample: PlannerSample,
        *,
        dependency_source: str,
        closed_loop: bool,
    ) -> dict[str, Any]:
        graph = self._dependency_graph(sample, dependency_source)
        remaining = set(sample.object_ids)
        selected = self._select_grasp(sample, graph, remaining)
        initial_blockers = selected["blocker_set"]

        if not closed_loop:
            removal_sequence = self._rank_blockers(sample, graph, selected["grasp_id"], initial_blockers)
            removal_sequence = removal_sequence[: self.params.max_steps]
            action = "target_grasp" if not initial_blockers else "remove_blockers_then_target"
            return {
                "scene_id": sample.scene_id,
                "target_id": sample.target_id,
                "selected_grasp_id": selected["grasp_id"],
                "selected_grasp_score": self._grasp_score(sample, selected["grasp_id"]),
                "predicted_blocker_set": initial_blockers,
                "initial_predicted_blocker_set": initial_blockers,
                "removal_sequence": removal_sequence,
                "num_removals": len(removal_sequence),
                "action": action,
                "candidate_scores": selected["candidate_scores"],
                "steps": [
                    {
                        "step": 0,
                        "action": action,
                        "selected_grasp_id": selected["grasp_id"],
                        "predicted_blocker_set": initial_blockers,
                        "removal_sequence": removal_sequence,
                    }
                ],
            }

        steps = []
        removal_sequence: list[str] = []
        current_selected = selected
        for step_idx in range(self.params.max_steps + 1):
            current_selected = self._select_grasp(sample, graph, remaining)
            grasp_id = current_selected["grasp_id"]
            blockers = current_selected["blocker_set"]
            if not blockers:
                steps.append(
                    {
                        "step": step_idx,
                        "action": "target_grasp",
                        "selected_grasp_id": grasp_id,
                        "predicted_blocker_set": [],
                        "remaining_objects": sorted(remaining),
                    }
                )
                return {
                    "scene_id": sample.scene_id,
                    "target_id": sample.target_id,
                    "selected_grasp_id": grasp_id,
                    "selected_grasp_score": self._grasp_score(sample, grasp_id),
                    "predicted_blocker_set": [],
                    "initial_predicted_blocker_set": initial_blockers,
                    "removal_sequence": removal_sequence,
                    "num_removals": len(removal_sequence),
                    "action": "target_grasp",
                    "candidate_scores": current_selected["candidate_scores"],
                    "steps": steps,
                }
            if step_idx >= self.params.max_steps:
                break
            next_object = self._select_next_blocker(sample, graph, grasp_id, blockers)
            if next_object is None:
                break
            removal_sequence.append(next_object)
            remaining.discard(next_object)
            steps.append(
                {
                    "step": step_idx,
                    "action": "remove_blocker",
                    "selected_grasp_id": grasp_id,
                    "predicted_blocker_set": blockers,
                    "remove_object_id": next_object,
                    "remaining_objects": sorted(remaining),
                }
            )

        return {
            "scene_id": sample.scene_id,
            "target_id": sample.target_id,
            "selected_grasp_id": current_selected["grasp_id"],
            "selected_grasp_score": self._grasp_score(sample, current_selected["grasp_id"]),
            "predicted_blocker_set": current_selected["blocker_set"],
            "initial_predicted_blocker_set": initial_blockers,
            "removal_sequence": removal_sequence,
            "num_removals": len(removal_sequence),
            "action": "max_steps_reached",
            "candidate_scores": current_selected["candidate_scores"],
            "steps": steps,
        }

    def _plan_ranked_removals(
        self,
        sample: PlannerSample,
        *,
        planner_type: str,
        closed_loop: bool,
    ) -> dict[str, Any]:
        grasp_id = self._best_grasp_by_score(sample)
        order = list(sample.object_ids)
        if planner_type == "random_removal":
            self.rng.shuffle(order)
        elif planner_type == "nearest_to_target":
            order.sort(key=lambda oid: float(sample.object_costs.get(oid, 0.0)))
        else:
            raise ValueError(planner_type)

        removal_sequence = order[: self.params.max_steps]
        if not closed_loop:
            steps = [
                {
                    "step": 0,
                    "action": "baseline_removal_sequence",
                    "selected_grasp_id": grasp_id,
                    "removal_sequence": removal_sequence,
                }
            ]
        else:
            steps = [
                {
                    "step": idx,
                    "action": "remove_blocker",
                    "selected_grasp_id": grasp_id,
                    "remove_object_id": object_id,
                }
                for idx, object_id in enumerate(removal_sequence)
            ]
        return {
            "scene_id": sample.scene_id,
            "target_id": sample.target_id,
            "selected_grasp_id": grasp_id,
            "selected_grasp_score": self._grasp_score(sample, grasp_id),
            "predicted_blocker_set": [],
            "initial_predicted_blocker_set": [],
            "removal_sequence": removal_sequence,
            "num_removals": len(removal_sequence),
            "action": "baseline_removal_sequence",
            "steps": steps,
        }

    def _plan_highest_global_dependency(self, sample: PlannerSample, *, closed_loop: bool) -> dict[str, Any]:
        scores = {object_id: 0.0 for object_id in sample.object_ids}
        for grasp_edges in sample.predicted_dependencies.values():
            for object_id, edge in grasp_edges.items():
                if object_id in scores:
                    scores[object_id] += dependency_strength(
                        edge,
                        use_typed_dependency=self.params.use_typed_dependency,
                        use_sufficient_dependency=self.params.use_sufficient_dependency,
                    )
        order = sorted(sample.object_ids, key=lambda oid: (-scores.get(oid, 0.0), oid))
        grasp_id = self._best_grasp_by_score(sample)
        removal_sequence = order[: self.params.max_steps]
        return {
            "scene_id": sample.scene_id,
            "target_id": sample.target_id,
            "selected_grasp_id": grasp_id,
            "selected_grasp_score": self._grasp_score(sample, grasp_id),
            "predicted_blocker_set": [],
            "initial_predicted_blocker_set": [],
            "removal_sequence": removal_sequence,
            "num_removals": len(removal_sequence),
            "action": "global_dependency_removal",
            "global_dependency_scores": scores,
            "steps": [
                {
                    "step": idx,
                    "action": "remove_blocker",
                    "selected_grasp_id": grasp_id,
                    "remove_object_id": object_id,
                    "global_dependency_score": scores.get(object_id, 0.0),
                }
                for idx, object_id in enumerate(removal_sequence if closed_loop else removal_sequence[:1])
            ],
        }

    def _plan_mechanical_search(
        self,
        sample: PlannerSample,
        *,
        planner_type: str,
        closed_loop: bool,
    ) -> dict[str, Any]:
        """Mechanical Search-style target-preempted removal policy.

        This is a dataset-level adaptation of published Mechanical Search
        policies. The original system combines target-preempted grasp/suction
        with physical pushes and real re-observation. Here we reproduce the
        high-level object prioritization using only available target grasp
        proposal scores and observed object visible area proxies.
        """

        grasp_id = self._best_grasp_by_score(sample)
        grasp_score = self._grasp_score(sample, grasp_id)
        threshold = self.params.mechanical_search_target_score_thresh
        target_is_confident = grasp_score >= threshold

        if target_is_confident:
            return {
                "scene_id": sample.scene_id,
                "target_id": sample.target_id,
                "selected_grasp_id": grasp_id,
                "selected_grasp_score": grasp_score,
                "predicted_blocker_set": [],
                "initial_predicted_blocker_set": [],
                "removal_sequence": [],
                "num_removals": 0,
                "action": "target_grasp",
                "mechanical_search_policy": planner_type,
                "mechanical_search_target_score_threshold": threshold,
                "mechanical_search_target_preempted": True,
                "steps": [
                    {
                        "step": 0,
                        "action": "target_grasp",
                        "selected_grasp_id": grasp_id,
                        "target_grasp_score": grasp_score,
                        "target_score_threshold": threshold,
                        "reason": "target grasp score exceeds Mechanical Search confidence threshold",
                    }
                ],
            }

        if planner_type == "mechanical_search_largest_first":
            order = sorted(
                sample.object_ids,
                key=lambda oid: (-float(sample.object_visible_area.get(oid, 0.0)), oid),
            )
            priority_scores = {oid: float(sample.object_visible_area.get(oid, 0.0)) for oid in sample.object_ids}
            priority_name = "visible_bbox_area"
        elif planner_type == "mechanical_search_preempted_random":
            order = list(sample.object_ids)
            self.rng.shuffle(order)
            priority_scores = {oid: 0.0 for oid in sample.object_ids}
            priority_name = "random"
        else:
            raise ValueError(planner_type)

        removal_sequence = order[: self.params.max_steps]
        if closed_loop:
            steps = [
                {
                    "step": idx,
                    "action": "mechanical_search_remove",
                    "selected_grasp_id": grasp_id,
                    "remove_object_id": object_id,
                    "priority_score": priority_scores.get(object_id, 0.0),
                    "priority_name": priority_name,
                }
                for idx, object_id in enumerate(removal_sequence)
            ]
            steps.append(
                {
                    "step": len(removal_sequence),
                    "action": "target_grasp",
                    "selected_grasp_id": grasp_id,
                    "target_grasp_score": grasp_score,
                    "target_score_threshold": threshold,
                }
            )
        else:
            steps = [
                {
                    "step": 0,
                    "action": "mechanical_search_removal_sequence",
                    "selected_grasp_id": grasp_id,
                    "target_grasp_score": grasp_score,
                    "target_score_threshold": threshold,
                    "priority_name": priority_name,
                    "removal_sequence": removal_sequence,
                }
            ]

        return {
            "scene_id": sample.scene_id,
            "target_id": sample.target_id,
            "selected_grasp_id": grasp_id,
            "selected_grasp_score": grasp_score,
            "predicted_blocker_set": [],
            "initial_predicted_blocker_set": [],
            "removal_sequence": removal_sequence,
            "num_removals": len(removal_sequence),
            "action": "mechanical_search_target_preempted_removal",
            "mechanical_search_policy": planner_type,
            "mechanical_search_priority_name": priority_name,
            "mechanical_search_priority_scores": priority_scores,
            "mechanical_search_target_score_threshold": threshold,
            "mechanical_search_target_preempted": False,
            "steps": steps,
        }

    def _plan_xray_support_reduction(self, sample: PlannerSample, *, closed_loop: bool) -> dict[str, Any]:
        """X-Ray-style target support reduction baseline.

        The published X-Ray/LAX-RAY systems select actions that reduce the
        possible target-support occupancy. This dataset-level adapter uses the
        same target-centric idea without learned dependency labels: objects are
        scored by overlap with the target bbox, proximity to the target, whether
        they appear in front of the target in depth, and visible area.
        """

        grasp_id = self._best_grasp_by_score(sample)
        grasp_score = self._grasp_score(sample, grasp_id)
        threshold = self.params.xray_target_score_thresh
        if grasp_score >= threshold:
            return {
                "scene_id": sample.scene_id,
                "target_id": sample.target_id,
                "selected_grasp_id": grasp_id,
                "selected_grasp_score": grasp_score,
                "predicted_blocker_set": [],
                "initial_predicted_blocker_set": [],
                "removal_sequence": [],
                "num_removals": 0,
                "action": "target_grasp",
                "xray_target_score_threshold": threshold,
                "xray_target_preempted": True,
                "steps": [
                    {
                        "step": 0,
                        "action": "target_grasp",
                        "selected_grasp_id": grasp_id,
                        "target_grasp_score": grasp_score,
                        "target_score_threshold": threshold,
                        "reason": "target grasp score exceeds X-Ray-style confidence threshold",
                    }
                ],
            }

        priority_scores = {
            object_id: self._xray_support_score(sample, object_id)
            for object_id in sample.object_ids
        }
        order = sorted(sample.object_ids, key=lambda oid: (-priority_scores.get(oid, 0.0), oid))
        removal_sequence = order[: self.params.max_steps]
        if closed_loop:
            steps = [
                {
                    "step": idx,
                    "action": "xray_support_reduction_remove",
                    "selected_grasp_id": grasp_id,
                    "remove_object_id": object_id,
                    "priority_score": priority_scores.get(object_id, 0.0),
                }
                for idx, object_id in enumerate(removal_sequence)
            ]
            steps.append(
                {
                    "step": len(removal_sequence),
                    "action": "target_grasp",
                    "selected_grasp_id": grasp_id,
                    "target_grasp_score": grasp_score,
                    "target_score_threshold": threshold,
                }
            )
        else:
            steps = [
                {
                    "step": 0,
                    "action": "xray_support_reduction_sequence",
                    "selected_grasp_id": grasp_id,
                    "target_grasp_score": grasp_score,
                    "target_score_threshold": threshold,
                    "removal_sequence": removal_sequence,
                }
            ]

        return {
            "scene_id": sample.scene_id,
            "target_id": sample.target_id,
            "selected_grasp_id": grasp_id,
            "selected_grasp_score": grasp_score,
            "predicted_blocker_set": [],
            "initial_predicted_blocker_set": [],
            "removal_sequence": removal_sequence,
            "num_removals": len(removal_sequence),
            "action": "xray_support_reduction_removal",
            "xray_priority_scores": priority_scores,
            "xray_target_score_threshold": threshold,
            "xray_target_preempted": False,
            "steps": steps,
        }

    def _plan_swept_volume_clearing(self, sample: PlannerSample, *, closed_loop: bool) -> dict[str, Any]:
        """Sense-plan-act geometry baseline using target-grasp swept volumes."""

        selected = self._select_swept_volume_plan(sample, set(sample.object_ids))
        grasp_id = str(selected["grasp_id"])
        blockers = list(selected["blocker_set"])
        removal_sequence = blockers[: self.params.max_steps]

        if not grasp_id:
            return {
                "scene_id": sample.scene_id,
                "target_id": sample.target_id,
                "selected_grasp_id": "",
                "selected_grasp_score": 0.0,
                "predicted_blocker_set": [],
                "initial_predicted_blocker_set": [],
                "removal_sequence": [],
                "num_removals": 0,
                "action": "no_valid_swept_volume_plan",
                "candidate_scores": [],
                "steps": [],
            }

        if not removal_sequence:
            return {
                "scene_id": sample.scene_id,
                "target_id": sample.target_id,
                "selected_grasp_id": grasp_id,
                "selected_grasp_score": self._grasp_score(sample, grasp_id),
                "predicted_blocker_set": [],
                "initial_predicted_blocker_set": blockers,
                "removal_sequence": [],
                "num_removals": 0,
                "action": "target_grasp",
                "candidate_scores": selected["candidate_scores"],
                "steps": [
                    {
                        "step": 0,
                        "action": "target_grasp",
                        "selected_grasp_id": grasp_id,
                        "predicted_blocker_set": [],
                        "reason": "no swept-volume geometry blockers",
                    }
                ],
            }

        if closed_loop:
            steps = [
                {
                    "step": 0,
                    "action": "swept_volume_remove",
                    "selected_grasp_id": grasp_id,
                    "predicted_blocker_set": blockers,
                    "remove_object_id": removal_sequence[0],
                    "swept_volume_risk": selected["blocker_scores"].get(removal_sequence[0], 0.0),
                }
            ]
        else:
            steps = [
                {
                    "step": 0,
                    "action": "swept_volume_removal_sequence",
                    "selected_grasp_id": grasp_id,
                    "predicted_blocker_set": blockers,
                    "removal_sequence": removal_sequence,
                }
            ]

        return {
            "scene_id": sample.scene_id,
            "target_id": sample.target_id,
            "selected_grasp_id": grasp_id,
            "selected_grasp_score": self._grasp_score(sample, grasp_id),
            "predicted_blocker_set": removal_sequence,
            "initial_predicted_blocker_set": blockers,
            "removal_sequence": removal_sequence,
            "num_removals": len(removal_sequence),
            "action": "swept_volume_clearing_removal",
            "candidate_scores": selected["candidate_scores"],
            "swept_volume_blocker_scores": selected["blocker_scores"],
            "steps": steps,
        }

    def _plan_oracle_minimal(self, sample: PlannerSample) -> dict[str, Any]:
        """Oracle upper bound that consumes Stage-1 minimal-blocker labels."""

        valid_statuses = {"already_feasible", "solved_within_depth"}
        candidates = []
        for grasp_id in sorted(sample.grasps):
            blocker_set, status = self._oracle_minimal_set(sample, grasp_id)
            if status not in valid_statuses:
                continue
            candidates.append(
                {
                    "grasp_id": grasp_id,
                    "blocker_set": blocker_set,
                    "status": status,
                    "score": len(blocker_set) - self.params.alpha * self._grasp_score(sample, grasp_id),
                    "num_blockers": len(blocker_set),
                    "grasp_score": self._grasp_score(sample, grasp_id),
                }
            )
        if not candidates:
            return {
                "scene_id": sample.scene_id,
                "target_id": sample.target_id,
                "selected_grasp_id": "",
                "selected_grasp_score": 0.0,
                "predicted_blocker_set": [],
                "initial_predicted_blocker_set": [],
                "removal_sequence": [],
                "num_removals": 0,
                "action": "no_valid_oracle_minimal_plan",
                "oracle_grasp_status": "no_valid_oracle_minimal_plan",
                "candidate_scores": [],
                "steps": [],
            }
        candidates.sort(key=lambda item: (item["score"], item["num_blockers"], -item["grasp_score"], item["grasp_id"]))
        selected = candidates[0]
        removal_sequence = list(selected["blocker_set"])[: self.params.max_steps]
        return {
            "scene_id": sample.scene_id,
            "target_id": sample.target_id,
            "selected_grasp_id": selected["grasp_id"],
            "selected_grasp_score": self._grasp_score(sample, selected["grasp_id"]),
            "predicted_blocker_set": list(selected["blocker_set"]),
            "initial_predicted_blocker_set": list(selected["blocker_set"]),
            "removal_sequence": removal_sequence,
            "num_removals": len(removal_sequence),
            "action": "oracle_minimal_blocker_removal",
            "oracle_grasp_status": selected["status"],
            "candidate_scores": candidates[:10],
            "steps": [
                {
                    "step": idx,
                    "action": "remove_oracle_blocker",
                    "selected_grasp_id": selected["grasp_id"],
                    "remove_object_id": object_id,
                }
                for idx, object_id in enumerate(removal_sequence)
            ],
        }

    def _plan_topk_dependency(self, sample: PlannerSample, *, closed_loop: bool) -> dict[str, Any]:
        """Per-grasp top-k dependency baseline for the highest-scoring target grasp."""

        graph = sample.predicted_dependencies
        grasp_id = self._best_grasp_by_score(sample)
        remaining = set(sample.object_ids)
        ranked = self._rank_all_dependencies(sample, graph, grasp_id, remaining)
        removal_sequence = [object_id for object_id, _score in ranked[: self.params.max_steps]]
        if not closed_loop:
            steps = [
                {
                    "step": 0,
                    "action": "topk_dependency_sequence",
                    "selected_grasp_id": grasp_id,
                    "removal_sequence": removal_sequence,
                }
            ]
        else:
            steps = [
                {
                    "step": idx,
                    "action": "remove_topk_dependency",
                    "selected_grasp_id": grasp_id,
                    "remove_object_id": object_id,
                    "dependency_strength": dict(ranked).get(object_id, 0.0),
                }
                for idx, object_id in enumerate(removal_sequence)
            ]
        return {
            "scene_id": sample.scene_id,
            "target_id": sample.target_id,
            "selected_grasp_id": grasp_id,
            "selected_grasp_score": self._grasp_score(sample, grasp_id),
            "predicted_blocker_set": removal_sequence,
            "initial_predicted_blocker_set": removal_sequence,
            "removal_sequence": removal_sequence,
            "num_removals": len(removal_sequence),
            "action": "topk_dependency_removal",
            "candidate_scores": [
                {"object_id": object_id, "dependency_strength": score}
                for object_id, score in ranked[:10]
            ],
            "steps": steps,
        }

    def _plan_budgeted_dependency(self, sample: PlannerSample, *, closed_loop: bool) -> dict[str, Any]:
        """Set-level dependency planner that enumerates budgeted blocker prefixes."""

        graph = sample.predicted_dependencies
        remaining = set(sample.object_ids)
        selected = self._select_budgeted_plan(sample, graph, remaining)
        initial_blockers = list(selected["blocker_set"])

        if not closed_loop:
            removal_sequence = initial_blockers[: self.params.max_steps]
            action = "target_grasp" if not removal_sequence else "remove_budgeted_blockers_then_target"
            return {
                "scene_id": sample.scene_id,
                "target_id": sample.target_id,
                "selected_grasp_id": selected["grasp_id"],
                "selected_grasp_score": self._grasp_score(sample, selected["grasp_id"]),
                "predicted_blocker_set": removal_sequence,
                "initial_predicted_blocker_set": initial_blockers,
                "removal_sequence": removal_sequence,
                "num_removals": len(removal_sequence),
                "action": action,
                "candidate_scores": selected["candidate_scores"],
                "steps": [
                    {
                        "step": 0,
                        "action": action,
                        "selected_grasp_id": selected["grasp_id"],
                        "predicted_blocker_set": initial_blockers,
                        "removal_sequence": removal_sequence,
                    }
                ],
            }

        steps = []
        removal_sequence: list[str] = []
        current_selected = selected
        for step_idx in range(self.params.max_steps + 1):
            current_selected = self._select_budgeted_plan(sample, graph, remaining)
            grasp_id = current_selected["grasp_id"]
            blockers = list(current_selected["blocker_set"])
            if not blockers:
                steps.append(
                    {
                        "step": step_idx,
                        "action": "target_grasp",
                        "selected_grasp_id": grasp_id,
                        "predicted_blocker_set": [],
                        "remaining_objects": sorted(remaining),
                    }
                )
                return {
                    "scene_id": sample.scene_id,
                    "target_id": sample.target_id,
                    "selected_grasp_id": grasp_id,
                    "selected_grasp_score": self._grasp_score(sample, grasp_id),
                    "predicted_blocker_set": [],
                    "initial_predicted_blocker_set": initial_blockers,
                    "removal_sequence": removal_sequence,
                    "num_removals": len(removal_sequence),
                    "action": "target_grasp",
                    "candidate_scores": current_selected["candidate_scores"],
                    "steps": steps,
                }
            if step_idx >= self.params.max_steps:
                break
            next_object = blockers[0]
            removal_sequence.append(next_object)
            remaining.discard(next_object)
            steps.append(
                {
                    "step": step_idx,
                    "action": "remove_budgeted_blocker",
                    "selected_grasp_id": grasp_id,
                    "predicted_blocker_set": blockers,
                    "remove_object_id": next_object,
                    "remaining_objects": sorted(remaining),
                }
            )

        return {
            "scene_id": sample.scene_id,
            "target_id": sample.target_id,
            "selected_grasp_id": current_selected["grasp_id"],
            "selected_grasp_score": self._grasp_score(sample, current_selected["grasp_id"]),
            "predicted_blocker_set": list(current_selected["blocker_set"]),
            "initial_predicted_blocker_set": initial_blockers,
            "removal_sequence": removal_sequence,
            "num_removals": len(removal_sequence),
            "action": "max_steps_reached",
            "candidate_scores": current_selected["candidate_scores"],
            "steps": steps,
        }

    def _plan_minimal_blocker_set(self, sample: PlannerSample, *, closed_loop: bool) -> dict[str, Any]:
        """Choose the smallest dependency prefix whose residual risk is low."""

        graph = sample.predicted_dependencies
        remaining = set(sample.object_ids)
        selected = self._select_minimal_blocker_prefix(sample, graph, remaining)
        initial_blockers = list(selected["blocker_set"])

        if not closed_loop:
            removal_sequence = initial_blockers[: self.params.max_steps]
            action = "target_grasp" if not removal_sequence else "remove_mbs_prefix_then_target"
            return {
                "scene_id": sample.scene_id,
                "target_id": sample.target_id,
                "selected_grasp_id": selected["grasp_id"],
                "selected_grasp_score": self._grasp_score(sample, selected["grasp_id"]),
                "predicted_blocker_set": removal_sequence,
                "initial_predicted_blocker_set": initial_blockers,
                "removal_sequence": removal_sequence,
                "num_removals": len(removal_sequence),
                "action": action,
                "candidate_scores": selected["candidate_scores"],
                "steps": [
                    {
                        "step": 0,
                        "action": action,
                        "selected_grasp_id": selected["grasp_id"],
                        "predicted_blocker_set": initial_blockers,
                        "residual_risk": selected["residual_risk"],
                        "sufficient_prefix": selected["sufficient_prefix"],
                    }
                ],
            }

        steps = []
        removal_sequence: list[str] = []
        current_selected = selected
        for step_idx in range(self.params.max_steps + 1):
            current_selected = self._select_minimal_blocker_prefix(sample, graph, remaining)
            grasp_id = current_selected["grasp_id"]
            blockers = list(current_selected["blocker_set"])
            if not blockers:
                steps.append(
                    {
                        "step": step_idx,
                        "action": "target_grasp",
                        "selected_grasp_id": grasp_id,
                        "predicted_blocker_set": [],
                        "remaining_objects": sorted(remaining),
                    }
                )
                return {
                    "scene_id": sample.scene_id,
                    "target_id": sample.target_id,
                    "selected_grasp_id": grasp_id,
                    "selected_grasp_score": self._grasp_score(sample, grasp_id),
                    "predicted_blocker_set": [],
                    "initial_predicted_blocker_set": initial_blockers,
                    "removal_sequence": removal_sequence,
                    "num_removals": len(removal_sequence),
                    "action": "target_grasp",
                    "candidate_scores": current_selected["candidate_scores"],
                    "steps": steps,
                }
            if step_idx >= self.params.max_steps:
                break
            next_object = blockers[0]
            removal_sequence.append(next_object)
            remaining.discard(next_object)
            steps.append(
                {
                    "step": step_idx,
                    "action": "remove_mbs_prefix_blocker",
                    "selected_grasp_id": grasp_id,
                    "predicted_blocker_set": blockers,
                    "remove_object_id": next_object,
                    "remaining_objects": sorted(remaining),
                    "residual_risk": current_selected["residual_risk"],
                    "sufficient_prefix": current_selected["sufficient_prefix"],
                }
            )

        return {
            "scene_id": sample.scene_id,
            "target_id": sample.target_id,
            "selected_grasp_id": current_selected["grasp_id"],
            "selected_grasp_score": self._grasp_score(sample, current_selected["grasp_id"]),
            "predicted_blocker_set": list(current_selected["blocker_set"]),
            "initial_predicted_blocker_set": initial_blockers,
            "removal_sequence": removal_sequence,
            "num_removals": len(removal_sequence),
            "action": "max_steps_reached",
            "candidate_scores": current_selected["candidate_scores"],
            "steps": steps,
        }

    def _select_grasp(
        self,
        sample: PlannerSample,
        graph: dict[str, dict[str, dict[str, float]]],
        remaining_objects: set[str],
    ) -> dict[str, Any]:
        if not sample.grasps:
            return {"grasp_id": "", "blocker_set": [], "candidate_scores": []}

        candidates = []
        for grasp_id in sorted(sample.grasps):
            edges = graph.get(grasp_id, {})
            blockers = compute_blocker_set(edges, remaining_objects, self.params)
            risk = risk_score(
                edges,
                remaining_objects,
                use_typed_dependency=self.params.use_typed_dependency,
                use_sufficient_dependency=self.params.use_sufficient_dependency,
            )
            grasp_score = self._grasp_score(sample, grasp_id)
            selection_score = grasp_selection_score(
                num_blockers=len(blockers),
                grasp_score=grasp_score,
                risk=risk,
                params=self.params,
            )
            candidates.append(
                {
                    "grasp_id": grasp_id,
                    "score": selection_score,
                    "num_blockers": len(blockers),
                    "grasp_score": grasp_score,
                    "risk": risk,
                    "blocker_set": blockers,
                }
            )
        candidates.sort(key=lambda item: (item["score"], item["num_blockers"], -item["grasp_score"], item["grasp_id"]))
        best = candidates[0]
        return {
            "grasp_id": best["grasp_id"],
            "blocker_set": best["blocker_set"],
            "candidate_scores": candidates[:10],
        }

    def _select_budgeted_plan(
        self,
        sample: PlannerSample,
        graph: dict[str, dict[str, dict[str, float]]],
        remaining_objects: set[str],
    ) -> dict[str, Any]:
        if not sample.grasps:
            return {"grasp_id": "", "blocker_set": [], "candidate_scores": []}

        candidates = []
        max_steps = max(0, int(self.params.max_steps))
        for grasp_id in self._candidate_grasp_ids(sample):
            ranked = self._rank_all_dependencies(sample, graph, grasp_id, remaining_objects)
            strengths = {object_id: score for object_id, score in ranked}
            total_risk = float(sum(strengths.values()))
            max_k = min(max_steps, len(ranked))
            grasp_score = self._grasp_score(sample, grasp_id)
            for k in range(max_k + 1):
                removed = [object_id for object_id, _score in ranked[:k]]
                removed_strength = float(sum(strengths.get(object_id, 0.0) for object_id in removed))
                residual_risk = max(0.0, total_risk - removed_strength)
                coverage = removed_strength / max(total_risk, EPS) if total_risk > EPS else 0.0
                residual_ratio = residual_risk / max(total_risk, EPS) if total_risk > EPS else 0.0
                score_bonus = self.params.alpha * grasp_score if self.params.use_grasp_score else 0.0
                selection_score = (
                    self.params.budget_residual_weight * residual_risk
                    + self.params.budget_residual_ratio_weight * residual_ratio
                    + self.params.budget_removal_weight * k
                    - score_bonus
                    - self.params.budget_removed_reward * removed_strength
                    - self.params.budget_coverage_reward * coverage
                )
                candidates.append(
                    {
                        "grasp_id": grasp_id,
                        "score": float(selection_score),
                        "num_blockers": k,
                        "blocker_set": removed,
                        "grasp_score": grasp_score,
                        "total_risk": total_risk,
                        "removed_strength": removed_strength,
                        "residual_risk": residual_risk,
                        "coverage": coverage,
                        "residual_ratio": residual_ratio,
                    }
                )
        candidates.sort(
            key=lambda item: (
                item["score"],
                item["residual_risk"],
                item["num_blockers"],
                -item["grasp_score"],
                item["grasp_id"],
            )
        )
        best = candidates[0]
        return {
            "grasp_id": best["grasp_id"],
            "blocker_set": list(best["blocker_set"]),
            "candidate_scores": candidates[:10],
        }

    def _select_minimal_blocker_prefix(
        self,
        sample: PlannerSample,
        graph: dict[str, dict[str, dict[str, float]]],
        remaining_objects: set[str],
    ) -> dict[str, Any]:
        if not sample.grasps:
            return {
                "grasp_id": "",
                "blocker_set": [],
                "candidate_scores": [],
                "residual_risk": 0.0,
                "sufficient_prefix": False,
            }

        candidates = []
        max_steps = max(0, int(self.params.max_steps))
        threshold = float(self.params.mbs_residual_threshold)
        for grasp_id in self._candidate_grasp_ids(sample):
            ranked = self._rank_all_dependencies(sample, graph, grasp_id, remaining_objects)
            strengths = {object_id: score for object_id, score in ranked}
            total_risk = float(sum(strengths.values()))
            max_k = min(max_steps, len(ranked))
            grasp_score = self._grasp_score(sample, grasp_id)
            for k in range(max_k + 1):
                removed = [object_id for object_id, _score in ranked[:k]]
                removed_strength = float(sum(strengths.get(object_id, 0.0) for object_id in removed))
                residual_risk = max(0.0, total_risk - removed_strength)
                coverage = removed_strength / max(total_risk, EPS) if total_risk > EPS else 0.0
                residual_ratio = residual_risk / max(total_risk, EPS) if total_risk > EPS else 0.0
                sufficient = residual_risk <= threshold
                score_bonus = self.params.alpha * grasp_score if self.params.use_grasp_score else 0.0
                selection_score = (
                    self.params.budget_residual_weight * residual_risk
                    + self.params.budget_residual_ratio_weight * residual_ratio
                    + self.params.budget_removal_weight * k
                    - score_bonus
                    - self.params.budget_removed_reward * removed_strength
                    - self.params.budget_coverage_reward * coverage
                )
                candidates.append(
                    {
                        "grasp_id": grasp_id,
                        "score": float(selection_score),
                        "num_blockers": k,
                        "blocker_set": removed,
                        "grasp_score": grasp_score,
                        "total_risk": total_risk,
                        "removed_strength": removed_strength,
                        "residual_risk": residual_risk,
                        "coverage": coverage,
                        "residual_ratio": residual_ratio,
                        "sufficient_prefix": bool(sufficient),
                    }
                )
        candidates.sort(
            key=lambda item: (
                0 if item["sufficient_prefix"] else 1,
                item["num_blockers"] if item["sufficient_prefix"] else item["residual_risk"],
                item["score"],
                item["residual_risk"],
                -item["grasp_score"],
                item["grasp_id"],
            )
        )
        best = candidates[0]
        return {
            "grasp_id": best["grasp_id"],
            "blocker_set": list(best["blocker_set"]),
            "candidate_scores": candidates[:10],
            "residual_risk": float(best["residual_risk"]),
            "sufficient_prefix": bool(best["sufficient_prefix"]),
        }

    def _select_swept_volume_plan(self, sample: PlannerSample, remaining_objects: set[str]) -> dict[str, Any]:
        if not sample.grasps:
            return {"grasp_id": "", "blocker_set": [], "candidate_scores": [], "blocker_scores": {}}

        candidates: list[dict[str, Any]] = []
        for grasp_id in self._candidate_grasp_ids(sample):
            ranked = self._swept_geometry_blockers(sample, grasp_id, remaining_objects)
            blockers = [object_id for object_id, _score in ranked]
            blocker_scores = {object_id: score for object_id, score in ranked}
            total_risk = float(sum(blocker_scores.values()))
            grasp_score = self._grasp_score(sample, grasp_id)
            candidates.append(
                {
                    "grasp_id": grasp_id,
                    "score": float(len(blockers) + 0.1 * total_risk - self.params.alpha * grasp_score),
                    "num_blockers": len(blockers),
                    "blocker_set": blockers,
                    "blocker_scores": blocker_scores,
                    "grasp_score": grasp_score,
                    "total_swept_risk": total_risk,
                }
            )
        candidates.sort(
            key=lambda item: (
                item["num_blockers"],
                item["total_swept_risk"],
                item["score"],
                -item["grasp_score"],
                item["grasp_id"],
            )
        )
        best = candidates[0]
        return {
            "grasp_id": best["grasp_id"],
            "blocker_set": list(best["blocker_set"]),
            "blocker_scores": dict(best["blocker_scores"]),
            "candidate_scores": candidates[:10],
        }

    def _swept_geometry_blockers(
        self,
        sample: PlannerSample,
        grasp_id: str,
        remaining_objects: set[str],
    ) -> list[tuple[str, float]]:
        geometry_by_object = sample.object_grasp_geometry.get(grasp_id, {})
        ranked: list[tuple[str, float]] = []
        for object_id in sorted(remaining_objects):
            geometry = geometry_by_object.get(object_id, {})
            if not self._is_swept_geometry_blocker(geometry):
                continue
            ranked.append((object_id, self._swept_geometry_risk(geometry)))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def _is_swept_geometry_blocker(self, geometry: dict[str, float]) -> bool:
        if not geometry:
            return False
        approach_iou = float(geometry.get("bbox_approach_swept_iou", 0.0))
        lift_iou = float(geometry.get("bbox_lift_swept_iou", 0.0))
        approach_clearance = float(geometry.get("approach_clearance", 1.0))
        lift_clearance = float(geometry.get("lift_clearance", 1.0))
        return (
            approach_iou > self.params.swept_volume_iou_threshold
            or lift_iou > self.params.swept_volume_iou_threshold
            or approach_clearance < self.params.swept_volume_clearance_threshold
            or lift_clearance < self.params.swept_volume_clearance_threshold
        )

    def _swept_geometry_risk(self, geometry: dict[str, float]) -> float:
        approach_iou = max(0.0, float(geometry.get("bbox_approach_swept_iou", 0.0)))
        lift_iou = max(0.0, float(geometry.get("bbox_lift_swept_iou", 0.0)))
        approach_clearance = float(geometry.get("approach_clearance", 1.0))
        lift_clearance = float(geometry.get("lift_clearance", 1.0))
        threshold = float(self.params.swept_volume_clearance_threshold)
        clearance_penalty = max(0.0, threshold - approach_clearance) + max(0.0, threshold - lift_clearance)
        distance = max(0.0, float(geometry.get("center_to_grasp_dist", 0.0)))
        return float(
            self.params.swept_volume_approach_weight * approach_iou
            + self.params.swept_volume_lift_weight * lift_iou
            + self.params.swept_volume_clearance_weight * clearance_penalty
            - self.params.swept_volume_distance_weight * distance
        )

    def _rank_blockers(
        self,
        sample: PlannerSample,
        graph: dict[str, dict[str, dict[str, float]]],
        grasp_id: str,
        blockers: list[str],
    ) -> list[str]:
        scored = []
        edges = graph.get(grasp_id, {})
        for object_id in blockers:
            edge = edges.get(object_id, {})
            score = blocker_ranking_score(
                dependency_strength_value=dependency_strength(
                    edge,
                    use_typed_dependency=self.params.use_typed_dependency,
                    use_sufficient_dependency=self.params.use_sufficient_dependency,
                ),
                blocker_graspability=float(sample.object_graspability.get(object_id, 1.0)),
                removal_cost=float(sample.object_costs.get(object_id, 0.0)),
                params=self.params,
            )
            scored.append((score, object_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [object_id for _score, object_id in scored]

    def _select_next_blocker(
        self,
        sample: PlannerSample,
        graph: dict[str, dict[str, dict[str, float]]],
        grasp_id: str,
        blockers: list[str],
    ) -> str | None:
        ranked = self._rank_blockers(sample, graph, grasp_id, blockers)
        return ranked[0] if ranked else None

    def _rank_all_dependencies(
        self,
        sample: PlannerSample,
        graph: dict[str, dict[str, dict[str, float]]],
        grasp_id: str,
        remaining_objects: set[str],
    ) -> list[tuple[str, float]]:
        edges = graph.get(grasp_id, {})
        ranked: list[tuple[str, float]] = []
        for object_id in sorted(remaining_objects):
            edge = edges.get(object_id, {})
            score = dependency_strength(
                edge,
                use_typed_dependency=self.params.use_typed_dependency,
                use_sufficient_dependency=self.params.use_sufficient_dependency,
            )
            if score >= self.params.topk_min_strength:
                ranked.append((object_id, float(score)))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def _best_grasp_by_score(self, sample: PlannerSample) -> str:
        if not sample.grasps:
            return ""
        return sorted(sample.grasps, key=lambda gid: (-self._grasp_score(sample, gid), gid))[0]

    def _candidate_grasp_ids(self, sample: PlannerSample) -> list[str]:
        """Return grasp ids allowed for set-level blocker planning."""

        if not sample.grasps:
            return []
        ranked = sorted(sample.grasps, key=lambda gid: (-self._grasp_score(sample, gid), gid))
        top_k = int(self.params.candidate_grasp_top_k)
        if top_k > 0:
            ranked = ranked[:top_k]

        ratio = max(0.0, float(self.params.candidate_grasp_min_score_ratio))
        if ratio <= 0.0:
            return ranked
        best_score = self._grasp_score(sample, ranked[0])
        if best_score <= 0.0:
            return ranked
        threshold = best_score * ratio
        filtered = [grasp_id for grasp_id in ranked if self._grasp_score(sample, grasp_id) >= threshold]
        return filtered or [ranked[0]]

    def _xray_support_score(self, sample: PlannerSample, object_id: str) -> float:
        overlap = max(0.0, float(sample.object_target_iou.get(object_id, 0.0)))
        distance = max(0.0, float(sample.object_costs.get(object_id, 0.0)))
        near = 1.0 / (1.0 + distance)
        depth_order = float(sample.object_depth_order.get(object_id, 0.0))
        in_front = max(0.0, -depth_order)
        area = max(0.0, float(sample.object_visible_area.get(object_id, 0.0)))
        return float(
            self.params.xray_overlap_weight * overlap
            + self.params.xray_near_weight * near
            + self.params.xray_front_weight * in_front
            + self.params.xray_area_weight * area
        )

    def _oracle_minimal_set(self, sample: PlannerSample, grasp_id: str) -> tuple[list[str], str]:
        label = sample.oracle_min_by_grasp.get(grasp_id, {})
        status = str(label.get("status", ""))
        minimal = sorted(str(x) for x in (label.get("minimal_blocker_set") or []))
        if status in {"already_feasible", "solved_within_depth"}:
            return minimal, status

        dep_edges = sample.oracle_dependencies.get(grasp_id, {})
        dep_set = sorted(object_id for object_id, edge in dep_edges.items() if float(edge.get("sufficient", 0.0)) >= 0.5)
        if dep_set:
            return dep_set, status or "oracle_sufficient_fallback"
        return minimal, status

    @staticmethod
    def _grasp_score(sample: PlannerSample, grasp_id: str) -> float:
        if not grasp_id:
            return 0.0
        return float(sample.grasps.get(grasp_id, {}).get("score", 0.0))
