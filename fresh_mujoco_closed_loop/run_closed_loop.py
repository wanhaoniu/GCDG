"""Run fresh-observation MuJoCo closed-loop target-retrieval episodes.

This module deliberately separates the closed-loop control logic from the heavy
MuJoCo/proposal/model adapters. Unit tests exercise the injected core loop; the
CLI wires that loop to the existing dataset runner, Stage-2 predictor, and
Stage-3 planner.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from safe_placement import PlacementRejected
import yaml

from fresh_mujoco_closed_loop.proposal_protocol import (
    FreshProposalProtocolConfig,
    generate_stage1_aligned_proposals,
)
from grasp_dependency_dataset.common.types import (
    GraspProposal,
    ObjectSpec,
    Pose,
    SceneObjectState,
    StableScene,
    ValidationResult,
)
from grasp_dependency_dataset.hetero_gnn.graph_dataset import (
    HeteroGraphDataset,
    SampleRef,
    collate_samples,
    discover_sample_refs,
    read_json,
)
from grasp_dependency_dataset.hetero_gnn.hetero_gnn import extract_edge_logits
from grasp_dependency_dataset.hetero_gnn.eval_hetero_gnn import choose_backend
from grasp_dependency_dataset.hetero_gnn.train_hetero_gnn import (
    adapt_batch_feature_dims,
    batch_to_torch,
    iter_sample_batches,
    load_config,
    resolve_torch_device,
)
from grasp_dependency_dataset.pipeline.runner import DatasetPipelineRunner
from planners.dependency_planner import DependencyGuidedPlanner
from planners.planner_utils import PlannerParams
from run_planner import (
    apply_calibration_to_planner_cfg,
    checkpoint_class_map,
    load_torch_model,
    planner_sample_from_graph,
    predict_dependency_maps,
    resolve_calibration_thresholds,
)


JsonDict = dict[str, Any]
ProposeFn = Callable[[StableScene, str], list[GraspProposal]]
PlanFn = Callable[[StableScene, str, list[GraspProposal], int], JsonDict]
ValidateFn = Callable[[StableScene, str, GraspProposal], ValidationResult]
ResettleFn = Callable[[StableScene, str, int], StableScene]
RefreshFn = Callable[[StableScene, str], StableScene]
ProposalTraceFn = Callable[[StableScene, str, list[GraspProposal]], JsonDict]
InterventionFn = Callable[[StableScene, str, str, int], StableScene]


@dataclass(frozen=True)
class FreshLoopConfig:
    planner_type: str = "budgeted_dependency"
    max_steps: int = 5
    resettle_after_removal: bool = True
    resettle_steps: int = 1200
    refresh_visibility: bool = True
    lock_planned_removal_set: bool = False
    force_target_after_locked_set: bool = True
    max_locked_removal_phases: int = 1
    validation_guided_recovery: bool = False
    recover_after_any_target_failure: bool = False
    retry_bin_wall_target_grasps: bool = False
    max_target_grasp_retries: int = 3
    bin_wall_safe_target_selection: bool = False
    max_bin_wall_safe_candidates: int = 3
    target_feasibility_selection: bool = False
    max_target_feasibility_candidates: int = 8
    target_feasibility_prefer_recoverable_failure: bool = True
    target_feasibility_recoverable_max_planner_score: float | None = None
    pre_removal_target_feasibility_guard: bool = False
    enable_shared_visibility_fallback: bool = False
    max_shared_fallback_removals: int | None = None
    shared_fallback_min_visible_pixels: int = 1
    terminate_on_invalid_resettle: bool = False


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_yaml(path: Path) -> JsonDict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def summarize_proposal_trace(
    proposals: list[GraspProposal],
    protocol_stats: JsonDict | None = None,
) -> JsonDict:
    """Build compact per-step proposal diagnostics for episode traces."""

    stats = dict(protocol_stats or {})
    source_counts = Counter(str(proposal.source) for proposal in proposals)
    grasp_type_counts = Counter(str(proposal.grasp_type.value) for proposal in proposals)
    isolated_raw_count = int(stats.get("parallel_raw_count", 0) or 0) + int(stats.get("suction_raw_count", 0) or 0)
    base_scene_count = int(
        stats.get(
            "base_existing_count",
            stats.get("total_final_kept_count", len(proposals)),
        )
        or 0
    )
    return {
        "source_mode": str(stats.get("source_mode", "")),
        "total_count": len(proposals),
        "source_counts": dict(sorted(source_counts.items())),
        "grasp_type_counts": dict(sorted(grasp_type_counts.items())),
        "base_scene_count": base_scene_count,
        "isolated_raw_count": isolated_raw_count,
        "isolated_added_count": int(stats.get("augmented_added_count", 0) or 0),
        "isolated_skipped_counts": dict(stats.get("augmented_skipped_counts") or {}),
        "final_count": int(stats.get("final_count", len(proposals)) or 0),
        "protocol_stats": stats,
    }


def visibility_rows_by_object_id(scene: StableScene) -> dict[str, JsonDict]:
    """Return the most recent per-object visibility rows stored on a scene."""

    ranking = scene.metadata.get("target_selection", {}).get("visibility_ranking", [])
    rows: dict[str, JsonDict] = {}
    for item in ranking:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "")
        if object_id:
            rows[object_id] = dict(item)
    return rows


def rank_visible_objects_by_physical_size(
    scene: StableScene,
    *,
    target_id: str,
    excluded_object_ids: Iterable[str] = (),
    min_visible_pixels: int = 1,
) -> list[JsonDict]:
    """Rank visible non-target objects by scaled physical proxy-AABB volume.

    Visibility is used only as a gate.  The ordering itself is based on the
    object's metric physical dimensions in the scene, rather than its apparent
    image area, so the same fallback can be used by every downstream planner.
    """

    visibility = visibility_rows_by_object_id(scene)
    excluded = {str(object_id) for object_id in excluded_object_ids}
    threshold = max(1, int(min_visible_pixels))
    ranked: list[JsonDict] = []
    for scene_object in scene.objects:
        object_id = str(scene_object.object_id)
        if object_id == str(target_id) or object_id in excluded:
            continue
        visible_pixels = int(visibility.get(object_id, {}).get("visible_pixels", 0) or 0)
        if visible_pixels < threshold:
            continue
        half_extents = tuple(float(value) for value in scene_object.scaled_proxy_half_extents)
        dimensions = tuple(2.0 * value for value in half_extents)
        volume = float(dimensions[0] * dimensions[1] * dimensions[2])
        ranked.append(
            {
                "object_id": object_id,
                "asset_name": str(scene_object.asset_name),
                "visible_pixels": visible_pixels,
                "physical_dimensions_m": [float(value) for value in dimensions],
                "physical_proxy_aabb_volume_m3": volume,
            }
        )
    ranked.sort(
        key=lambda row: (
            -float(row["physical_proxy_aabb_volume_m3"]),
            -int(row["visible_pixels"]),
            str(row["object_id"]),
        )
    )
    return ranked


def invalid_resettle_terminal_reason(resettle: JsonDict | None) -> str:
    """Return a terminal failure reason for an invalid post-removal state."""

    record = dict(resettle or {})
    if not record:
        return "resettle_audit_missing"
    if list(record.get("out_of_bin_object_ids") or []):
        return "resettle_out_of_bin"
    if not bool(record.get("stable", False)) or not bool(record.get("accepted", False)):
        return "resettle_unstable"
    return ""


def with_fresh_scene_id(
    scene: StableScene,
    *,
    original_scene_id: str,
    step_index: int,
    removed_object_ids: Iterable[str],
) -> StableScene:
    """Return the same scene with a per-step id so render/proposal caches miss."""

    metadata = dict(scene.metadata)
    metadata["fresh_closed_loop"] = {
        "original_scene_id": str(original_scene_id),
        "step_index": int(step_index),
        "removed_object_ids": sorted(str(item) for item in removed_object_ids),
    }
    return replace(
        scene,
        scene_id=f"{original_scene_id}__fresh_step_{int(step_index):02d}",
        metadata=metadata,
    )


def extract_next_action(
    plan_result: JsonDict,
    unavailable_object_ids: Iterable[str] | None = None,
) -> JsonDict:
    """Convert a Stage-3 planner result into one environment action."""

    selected_grasp_id = str(plan_result.get("selected_grasp_id") or "")
    unavailable = {str(object_id) for object_id in (unavailable_object_ids or [])}
    for step in plan_result.get("steps") or []:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "")
        if action == "target_grasp":
            return {
                "action": "target_grasp",
                "selected_grasp_id": str(step.get("selected_grasp_id") or selected_grasp_id),
            }
        remove_object_id = step.get("remove_object_id")
        if remove_object_id and str(remove_object_id) not in unavailable:
            return {
                "action": "remove_object",
                "selected_grasp_id": str(step.get("selected_grasp_id") or selected_grasp_id),
                "remove_object_id": str(remove_object_id),
            }

    for object_id in plan_result.get("removal_sequence") or []:
        if str(object_id) in unavailable:
            continue
        return {
            "action": "remove_object",
            "selected_grasp_id": selected_grasp_id,
            "remove_object_id": str(object_id),
        }

    return {
        "action": "target_grasp",
        "selected_grasp_id": selected_grasp_id,
    }


def planned_removal_sequence(plan_result: JsonDict, next_action: JsonDict) -> list[str]:
    """Return the ordered blocker set selected by the planner for this phase."""

    sequence: list[str] = []
    seen: set[str] = set()
    for object_id in plan_result.get("removal_sequence") or []:
        value = str(object_id)
        if value and value not in seen:
            sequence.append(value)
            seen.add(value)
    if not sequence and next_action.get("action") == "remove_object":
        value = str(next_action.get("remove_object_id") or "")
        if value:
            sequence.append(value)
    return sequence


def validation_recovery_blockers(
    validation_payload: JsonDict,
    *,
    target_id: str,
    removed_object_ids: Iterable[str],
    removable_object_ids: Iterable[str],
) -> list[str]:
    """Extract object blockers observed by the validator in typed priority order."""

    stage_blockers = validation_payload.get("stage_blockers") or {}
    if not isinstance(stage_blockers, dict):
        return []
    removable = {str(object_id) for object_id in removable_object_ids}
    ignored = {str(target_id), *(str(object_id) for object_id in removed_object_ids)}
    ordered_stages = (
        "approach_collision",
        "lift_collision",
        "closing_or_seal_failure",
        "unreachable",
    )
    stage_names = list(ordered_stages) + sorted(str(name) for name in stage_blockers if str(name) not in ordered_stages)
    blockers: list[str] = []
    seen: set[str] = set()
    for stage_name in stage_names:
        values = stage_blockers.get(stage_name) or []
        if isinstance(values, str):
            values = [values]
        for object_id in values:
            value = str(object_id)
            if not value or value in seen or value in ignored or value not in removable:
                continue
            blockers.append(value)
            seen.add(value)
    return blockers


def is_bin_wall_only_failure(validation_payload: JsonDict) -> bool:
    """Return true if the failed validation only reports bin-wall blockers."""

    if bool(validation_payload.get("feasible", False)):
        return False
    stage_blockers = validation_payload.get("stage_blockers") or {}
    if not isinstance(stage_blockers, dict):
        return False
    blockers: list[str] = []
    for values in stage_blockers.values():
        if isinstance(values, str):
            values = [values]
        blockers.extend(str(value) for value in (values or []) if str(value))
    return bool(blockers) and all(value == "__bin_wall__" for value in blockers)


def ordered_target_grasp_candidates(
    plan_result: JsonDict,
    proposals: list[GraspProposal],
    *,
    selected_grasp_id: str,
    rejected_grasp_ids: Iterable[str],
) -> list[str]:
    """Return unique target-grasp candidates for retry in planner/proposal order."""

    rejected = {str(grasp_id) for grasp_id in rejected_grasp_ids}
    proposal_ids = {proposal.grasp_id for proposal in proposals}
    ordered: list[str] = []
    seen: set[str] = set()

    def add(grasp_id: str) -> None:
        value = str(grasp_id or "")
        if not value or value in seen or value in rejected or value not in proposal_ids:
            return
        ordered.append(value)
        seen.add(value)

    add(selected_grasp_id)
    for row in plan_result.get("candidate_scores") or []:
        if isinstance(row, dict):
            add(str(row.get("grasp_id") or ""))
    for proposal in sorted(proposals, key=lambda item: (-float(item.proposal_score), item.grasp_id)):
        add(proposal.grasp_id)
    return ordered


def planner_candidate_score(plan_result: JsonDict, grasp_id: str) -> float | None:
    """Return the planner objective score for a candidate grasp, if logged."""

    for row in plan_result.get("candidate_scores") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("grasp_id") or "") != str(grasp_id):
            continue
        try:
            return float(row.get("score"))
        except (TypeError, ValueError):
            return None
    return None


class FreshClosedLoopEpisodeRunner:
    """Dependency-injected fresh closed-loop executor."""

    def __init__(
        self,
        *,
        propose: ProposeFn,
        plan: PlanFn,
        validate_target_grasp: ValidateFn,
        resettle_after_removal: ResettleFn | None = None,
        refresh_observation_metadata: RefreshFn | None = None,
        proposal_trace: ProposalTraceFn | None = None,
        apply_non_target_intervention: InterventionFn | None = None,
    ) -> None:
        self._propose = propose
        self._plan = plan
        self._validate = validate_target_grasp
        self._resettle = resettle_after_removal or (lambda scene, _object_id, _step: scene)
        self._refresh = refresh_observation_metadata or (lambda scene, _target_id: scene)
        self._proposal_trace = proposal_trace or (
            lambda _scene, _target_id, proposals: summarize_proposal_trace(proposals, {})
        )
        self._apply_non_target_intervention = apply_non_target_intervention

    def run_episode(self, scene: StableScene, target_id: str, config: FreshLoopConfig) -> JsonDict:
        original_scene_id = scene.scene_id
        current_scene = scene
        removed: list[str] = []
        fallback_removed: list[str] = []
        planner_removed: list[str] = []
        steps: list[JsonDict] = []
        fallback_actions: list[JsonDict] = []
        availability_observations: list[JsonDict] = []
        first_proposal_recovery: JsonDict | None = None
        planner_invocation_count = 0
        locked_removal_sequence: list[str] = []
        force_target_attempt_next = False
        locked_phase_count = 0
        pending_validation_recovery_blockers: list[str] = []
        max_locked_phases = max(1, int(config.max_locked_removal_phases))
        max_loop_iterations = int(config.max_steps) + (max_locked_phases if config.lock_planned_removal_set else 1)
        fallback_limit = (
            max(0, int(config.max_steps))
            if config.max_shared_fallback_removals is None
            else max(0, int(config.max_shared_fallback_removals))
        )

        def finish(*, success: bool, terminal_reason: str) -> JsonDict:
            result = self._episode_result(
                scene=scene,
                target_id=target_id,
                config=config,
                success=success,
                terminal_reason=terminal_reason,
                removed=removed,
                steps=steps,
            )
            reobservations = [
                dict(record)
                for record in availability_observations
                if bool(record.get("is_reobservation"))
            ]
            recovery_payload: JsonDict = {
                "required": bool(fallback_removed),
                "recovered": first_proposal_recovery is not None,
                "first_recovered_observation_index": None,
                "first_recovered_reobservation_index": None,
                "fallback_actions_before_recovery": None,
                "terminal_without_recovery": (
                    terminal_reason
                    if fallback_removed and first_proposal_recovery is None
                    else None
                ),
            }
            if first_proposal_recovery is not None:
                recovery_payload.update(
                    {
                        "first_recovered_observation_index": int(
                            first_proposal_recovery["observation_index"]
                        ),
                        "first_recovered_reobservation_index": int(
                            first_proposal_recovery["reobservation_index"]
                        ),
                        "fallback_actions_before_recovery": int(
                            first_proposal_recovery["fallback_actions_completed"]
                        ),
                        "target_visible_pixels_at_recovery": first_proposal_recovery.get(
                            "target_visible_pixels"
                        ),
                        "current_scene_proposal_count_at_recovery": int(
                            first_proposal_recovery["current_scene_proposal_count"]
                        ),
                        "planner_proposal_count_at_recovery": int(
                            first_proposal_recovery["planner_proposal_count"]
                        ),
                    }
                )
            result.update(
                {
                    "num_total_removals": len(removed),
                    "num_fallback_removals": len(fallback_removed),
                    "num_planner_removals": len(planner_removed),
                    "fallback_removal_sequence": list(fallback_removed),
                    "planner_removal_sequence": list(planner_removed),
                    "num_planner_invocations": int(planner_invocation_count),
                    "shared_fallback": {
                        "enabled": bool(config.enable_shared_visibility_fallback),
                        "policy_id": "largest_visible_physical_proxy_aabb_v1",
                        "visibility_role": "gate_only",
                        "size_metric": "scaled_physical_proxy_aabb_volume_m3",
                        "min_visible_pixels": max(
                            1, int(config.shared_fallback_min_visible_pixels)
                        ),
                        "max_fallback_removals": int(fallback_limit),
                        "counts_toward_shared_removal_budget": True,
                        "num_actions": len(fallback_actions),
                        "actions": [dict(action) for action in fallback_actions],
                        "num_preplanner_observations": len(availability_observations),
                        "preplanner_observations": [
                            dict(record) for record in availability_observations
                        ],
                        "num_reobservations": len(reobservations),
                        "reobservations": reobservations,
                        "proposal_recovery": recovery_payload,
                    },
                }
            )
            return result

        for step_index in range(max_loop_iterations):
            step_scene = with_fresh_scene_id(
                current_scene,
                original_scene_id=original_scene_id,
                step_index=step_index,
                removed_object_ids=removed,
            )
            if config.refresh_visibility or config.enable_shared_visibility_fallback:
                step_scene = self._refresh(step_scene, target_id)

            # 'removed' remains an action history: repeats still spend budget.
            # Relocation does not make an object absent from the current state.
            present_ids = {obj.object_id for obj in step_scene.objects}
            unavailable_ids = [oid for oid in removed if oid not in present_ids]
            proposals = self._propose(step_scene, target_id)
            proposal_by_id = {proposal.grasp_id: proposal for proposal in proposals}
            remaining_budget = max(0, int(config.max_steps) - len(removed))
            step_record: JsonDict = {
                "step": step_index,
                "scene_id": step_scene.scene_id,
                "target_id": target_id,
                "remaining_budget": remaining_budget,
                "remaining_objects": [obj.object_id for obj in step_scene.objects if obj.object_id != target_id],
                "num_proposals": len(proposals),
            }
            step_record["proposal_trace"] = self._proposal_trace(step_scene, target_id, proposals)

            if config.enable_shared_visibility_fallback:
                visibility_rows = visibility_rows_by_object_id(step_scene)
                target_visibility = visibility_rows.get(str(target_id))
                target_visible_pixels = (
                    None
                    if target_visibility is None
                    else int(target_visibility.get("visible_pixels", 0) or 0)
                )
                min_visible_pixels = max(1, int(config.shared_fallback_min_visible_pixels))
                current_scene_proposal_count = int(
                    step_record["proposal_trace"].get("base_scene_count", len(proposals))
                    or 0
                )
                trigger_reasons: list[str] = []
                if target_visible_pixels == 0:
                    trigger_reasons.append("target_fully_hidden")
                elif (
                    target_visible_pixels is not None
                    and target_visible_pixels < min_visible_pixels
                ):
                    trigger_reasons.append("target_below_min_visible_pixels")
                # Blocked augmented proposals are valid planner inputs when the
                # target is visible. Their collision is what the model reasons about.
                if not proposals:
                    trigger_reasons.append("planner_proposals_empty")

                is_reobservation = bool(fallback_removed)
                reobservation_index = None
                if is_reobservation:
                    reobservation_index = 1 + sum(
                        1
                        for record in availability_observations
                        if bool(record.get("is_reobservation"))
                    )
                availability_record: JsonDict = {
                    "observation_index": len(availability_observations),
                    "step": int(step_index),
                    "scene_id": step_scene.scene_id,
                    "is_reobservation": is_reobservation,
                    "reobservation_index": reobservation_index,
                    "fallback_actions_completed": len(fallback_removed),
                    "target_visibility_available": target_visibility is not None,
                    "target_visible_pixels": target_visible_pixels,
                    "current_scene_proposal_count": current_scene_proposal_count,
                    "planner_proposal_count": len(proposals),
                    "proposal_protocol_source_mode": str(
                        step_record["proposal_trace"].get("source_mode", "")
                    ),
                    "trigger_reasons": list(trigger_reasons),
                    "availability_state": "unavailable" if trigger_reasons else "available",
                    "proposal_recovered": bool(fallback_removed) and not trigger_reasons,
                    "planner_invoked": False,
                    "resettle_after_previous_removal": (
                        dict(step_scene.metadata.get("fresh_resettle") or {}) or None
                    ),
                }
                availability_observations.append(availability_record)
                step_record["shared_fallback_observation"] = availability_record

                if not trigger_reasons and fallback_removed:
                    availability_record["availability_state"] = "recovered"
                    if first_proposal_recovery is None:
                        first_proposal_recovery = dict(availability_record)

                if trigger_reasons:
                    step_record["action_source"] = "shared_preplanner_visibility_fallback"
                    if remaining_budget <= 0:
                        step_record["action"] = "shared_fallback_removal_budget_exhausted"
                        steps.append(step_record)
                        return finish(
                            success=False,
                            terminal_reason="shared_fallback_removal_budget_exhausted",
                        )
                    if len(fallback_removed) >= fallback_limit:
                        step_record["action"] = "shared_fallback_limit_exhausted"
                        steps.append(step_record)
                        return finish(
                            success=False,
                            terminal_reason="shared_fallback_limit_exhausted",
                        )

                    visible_candidates = rank_visible_objects_by_physical_size(
                        step_scene,
                        target_id=target_id,
                        excluded_object_ids=removed,
                        min_visible_pixels=min_visible_pixels,
                    )
                    availability_record["visible_non_target_candidates"] = visible_candidates
                    if not visible_candidates:
                        step_record["action"] = "shared_fallback_no_visible_non_target"
                        steps.append(step_record)
                        return finish(
                            success=False,
                            terminal_reason="shared_fallback_no_visible_non_target",
                        )

                    selected = visible_candidates[0]
                    remove_object_id = str(selected["object_id"])
                    planner_state_reset = bool(
                        locked_removal_sequence
                        or force_target_attempt_next
                        or pending_validation_recovery_blockers
                    )
                    locked_removal_sequence = []
                    force_target_attempt_next = False
                    pending_validation_recovery_blockers = []
                    fallback_action: JsonDict = {
                        "fallback_action_index": len(fallback_actions),
                        "observation_index": int(availability_record["observation_index"]),
                        "step": int(step_index),
                        "action": "remove_largest_visible_object",
                        "action_source": "shared_preplanner_visibility_fallback",
                        "trigger_reasons": list(trigger_reasons),
                        "remove_object_id": remove_object_id,
                        "selected_asset_name": str(selected["asset_name"]),
                        "selected_visible_pixels": int(selected["visible_pixels"]),
                        "selected_physical_dimensions_m": list(
                            selected["physical_dimensions_m"]
                        ),
                        "selected_physical_proxy_aabb_volume_m3": float(
                            selected["physical_proxy_aabb_volume_m3"]
                        ),
                        "candidate_rank": 1,
                        "num_visible_non_target_candidates": len(visible_candidates),
                        "planner_state_reset": planner_state_reset,
                        "reobservation_requested": True,
                    }
                    availability_record["fallback_action"] = fallback_action
                    step_record["action"] = "shared_fallback_remove_object"
                    step_record["remove_object_id"] = remove_object_id
                    step_record["fallback_action"] = fallback_action
                    fallback_actions.append(dict(fallback_action))
                    fallback_removed.append(remove_object_id)
                    removed.append(remove_object_id)
                    steps.append(step_record)

                    if self._apply_non_target_intervention is None:
                        next_scene = step_scene.without_objects({remove_object_id})
                        if config.resettle_after_removal:
                            next_scene = self._resettle(next_scene, remove_object_id, step_index)
                    else:
                        try:
                            next_scene = self._apply_non_target_intervention(
                                step_scene,
                                target_id,
                                remove_object_id,
                                step_index,
                            )
                        except PlacementRejected as exc:
                            removed.pop()
                            fallback_removed.pop()
                            step_record['placement_rejected'] = exc.record
                            step_record['intervention_executed'] = False
                            fallback_actions[-1]['intervention_executed'] = False
                            return finish(success=False, terminal_reason='placement_no_feasible_pose')
                    resettle_record = dict(next_scene.metadata.get("fresh_resettle") or {})
                    step_record["resettle"] = resettle_record or None
                    fallback_action["resettle"] = resettle_record or None
                    fallback_actions[-1] = dict(fallback_action)
                    if config.resettle_after_removal and config.terminate_on_invalid_resettle:
                        invalid_reason = invalid_resettle_terminal_reason(resettle_record)
                        if invalid_reason:
                            step_record["resettle_terminal_failure"] = invalid_reason
                            fallback_action["resettle_terminal_failure"] = invalid_reason
                            fallback_actions[-1] = dict(fallback_action)
                            return finish(success=False, terminal_reason=invalid_reason)
                    current_scene = next_scene
                    continue

            if not proposals:
                step_record["action"] = "no_proposals"
                steps.append(step_record)
                return finish(
                    success=False,
                    terminal_reason="no_proposals",
                )

            force_target_attempt = (
                bool(config.lock_planned_removal_set)
                and bool(config.force_target_after_locked_set)
                and bool(force_target_attempt_next)
            )
            plan_budget = 0 if force_target_attempt else remaining_budget
            planner_invocation_count += 1
            if config.enable_shared_visibility_fallback:
                availability_observations[-1]["planner_invoked"] = True
            plan_result = self._plan(step_scene, target_id, proposals, plan_budget)
            next_action = extract_next_action(plan_result, unavailable_object_ids=unavailable_ids)
            step_record["action_source"] = "dependency_planner"
            step_record["planner_result"] = plan_result
            step_record["next_action"] = next_action
            if config.lock_planned_removal_set:
                step_record["stop_policy"] = {
                    "lock_planned_removal_set": True,
                    "force_target_after_locked_set": bool(config.force_target_after_locked_set),
                    "max_locked_removal_phases": int(max_locked_phases),
                    "validation_guided_recovery": bool(config.validation_guided_recovery),
                    "recover_after_any_target_failure": bool(config.recover_after_any_target_failure),
                    "bin_wall_safe_target_selection": bool(config.bin_wall_safe_target_selection),
                    "target_feasibility_selection": bool(config.target_feasibility_selection),
                    "pre_removal_target_feasibility_guard": bool(config.pre_removal_target_feasibility_guard),
                    "locked_phase_count": int(locked_phase_count),
                    "forced_target_attempt": bool(force_target_attempt),
                    "locked_removal_sequence_before": list(locked_removal_sequence),
                    "pending_validation_recovery_blockers": list(pending_validation_recovery_blockers),
                    "plan_budget": int(plan_budget),
                }
            force_target_attempt_next = False

            removable = {
                obj.object_id
                for obj in step_scene.objects
                if obj.object_id != target_id and obj.object_id not in unavailable_ids
            }
            if config.lock_planned_removal_set and locked_removal_sequence and not force_target_attempt:
                while locked_removal_sequence and locked_removal_sequence[0] not in removable:
                    locked_removal_sequence.pop(0)
                if locked_removal_sequence:
                    next_action = {
                        "action": "remove_object",
                        "selected_grasp_id": str(plan_result.get("selected_grasp_id") or next_action.get("selected_grasp_id") or ""),
                        "remove_object_id": locked_removal_sequence[0],
                    }
                    step_record["next_action"] = next_action
                    step_record["stop_policy"]["using_locked_removal_sequence"] = True
                elif config.force_target_after_locked_set:
                    force_target_attempt_next = True
            elif (
                config.lock_planned_removal_set
                and (
                    next_action.get("action") == "remove_object"
                    or (config.validation_guided_recovery and bool(pending_validation_recovery_blockers))
                )
                and not force_target_attempt
            ):
                learned_sequence = planned_removal_sequence(plan_result, next_action)
                locked_removal_sequence = list(learned_sequence)
                if config.validation_guided_recovery and pending_validation_recovery_blockers:
                    observed = [
                        object_id
                        for object_id in pending_validation_recovery_blockers
                        if object_id in removable
                    ]
                    merged: list[str] = []
                    seen: set[str] = set()
                    for object_id in [*observed, *learned_sequence]:
                        if object_id and object_id not in seen and object_id in removable:
                            merged.append(object_id)
                            seen.add(object_id)
                    locked_removal_sequence = merged
                    step_record["stop_policy"]["validation_guided_recovery_blockers"] = list(observed)
                    step_record["stop_policy"]["learned_locked_removal_sequence"] = list(learned_sequence)
                    pending_validation_recovery_blockers = []
                locked_removal_sequence = [
                    object_id for object_id in locked_removal_sequence if object_id in removable
                ]
                locked_removal_sequence = locked_removal_sequence[:remaining_budget]
                step_record["stop_policy"]["locked_removal_sequence_initialized"] = list(locked_removal_sequence)
                if locked_removal_sequence:
                    locked_phase_count += 1
                    step_record["stop_policy"]["locked_phase_count"] = int(locked_phase_count)
                    next_action = {
                        "action": "remove_object",
                        "selected_grasp_id": str(plan_result.get("selected_grasp_id") or next_action.get("selected_grasp_id") or ""),
                        "remove_object_id": locked_removal_sequence[0],
                    }
                    step_record["next_action"] = next_action

            prevalidated_target: tuple[str, ValidationResult, JsonDict, list[JsonDict]] | None = None
            if bool(config.pre_removal_target_feasibility_guard) and next_action.get("action") == "remove_object":
                max_guard_candidates = max(1, int(config.max_target_feasibility_candidates))
                guard_record: JsonDict = {
                    "enabled": True,
                    "planned_remove_object_id": str(next_action.get("remove_object_id") or ""),
                    "max_candidates": int(max_guard_candidates),
                    "evaluated_grasp_ids": [],
                    "attempts": [],
                    "intervened": False,
                    "selected_grasp_id": "",
                }
                guard_attempts: list[JsonDict] = []
                guard_candidates = ordered_target_grasp_candidates(
                    plan_result,
                    proposals,
                    selected_grasp_id=str(plan_result.get("selected_grasp_id") or next_action.get("selected_grasp_id") or ""),
                    rejected_grasp_ids=set(),
                )[:max_guard_candidates]
                for candidate_id in guard_candidates:
                    proposal = proposal_by_id.get(candidate_id)
                    if proposal is None:
                        continue
                    candidate_validation = self._validate(step_scene, target_id, proposal)
                    candidate_payload = candidate_validation.to_dict()
                    attempt_record = {
                        "grasp_id": candidate_id,
                        "feasible": bool(candidate_validation.feasible),
                        "failure_stage": candidate_validation.failure_stage.value,
                        "stage_blockers": candidate_payload.get("stage_blockers", {}),
                    }
                    guard_record["evaluated_grasp_ids"].append(candidate_id)
                    guard_attempts.append(attempt_record)
                    guard_record["attempts"].append(dict(attempt_record))
                    if not candidate_validation.feasible:
                        continue
                    guard_record["intervened"] = True
                    guard_record["selected_grasp_id"] = candidate_id
                    next_action = {
                        "action": "target_grasp",
                        "selected_grasp_id": candidate_id,
                    }
                    step_record["next_action"] = next_action
                    prevalidated_target = (
                        candidate_id,
                        candidate_validation,
                        candidate_payload,
                        list(guard_attempts),
                    )
                    break
                step_record["pre_removal_target_feasibility_guard"] = guard_record

            if next_action["action"] == "target_grasp":
                grasp_id = str(next_action.get("selected_grasp_id") or "")
                original_grasp_id = grasp_id
                rejected_target_grasps: set[str] = set()
                target_attempts: list[JsonDict] = []
                validation: ValidationResult | None = None
                validation_payload: JsonDict = {}
                max_target_attempts = max(1, int(config.max_target_grasp_retries))

                def validate_candidate(candidate_grasp_id: str) -> tuple[ValidationResult, JsonDict]:
                    proposal = proposal_by_id.get(candidate_grasp_id)
                    if proposal is None:
                        step_record["action"] = "target_grasp_missing_proposal"
                        step_record["selected_grasp_id"] = candidate_grasp_id
                        steps.append(step_record)
                        raise KeyError(candidate_grasp_id)
                    candidate_validation = self._validate(step_scene, target_id, proposal)
                    candidate_payload = candidate_validation.to_dict()
                    target_attempts.append(
                        {
                            "grasp_id": candidate_grasp_id,
                            "feasible": bool(candidate_validation.feasible),
                            "failure_stage": candidate_validation.failure_stage.value,
                            "stage_blockers": candidate_payload.get("stage_blockers", {}),
                        }
                    )
                    return candidate_validation, candidate_payload

                try:
                    if prevalidated_target is not None and str(prevalidated_target[0]) == grasp_id:
                        grasp_id, validation, validation_payload, prevalidated_attempts = prevalidated_target
                        target_attempts.extend(prevalidated_attempts)
                    elif config.target_feasibility_selection:
                        max_feasibility_candidates = max(1, int(config.max_target_feasibility_candidates))
                        candidate_ids = ordered_target_grasp_candidates(
                            plan_result,
                            proposals,
                            selected_grasp_id=original_grasp_id,
                            rejected_grasp_ids=set(),
                        )[:max_feasibility_candidates]
                        selection_record: JsonDict = {
                            "enabled": True,
                            "original_grasp_id": original_grasp_id,
                            "max_candidates": int(max_feasibility_candidates),
                            "evaluated_grasp_ids": [],
                            "selected_grasp_id": "",
                            "selected_reason": "",
                            "feasible_candidate_found": False,
                            "recoverable_candidate_found": False,
                            "recoverable_candidate_accepted": False,
                            "recoverable_candidate_grasp_id": "",
                            "recoverable_candidate_score": None,
                            "recoverable_candidate_blockers": [],
                            "recoverable_rejection_reason": "",
                            "recoverable_score_gate": config.target_feasibility_recoverable_max_planner_score,
                        }
                        original_validation: ValidationResult | None = None
                        original_payload: JsonDict = {}
                        recoverable_choice: tuple[str, ValidationResult, JsonDict] | None = None
                        for candidate_id in candidate_ids:
                            candidate_validation, candidate_payload = validate_candidate(candidate_id)
                            selection_record["evaluated_grasp_ids"].append(candidate_id)
                            if candidate_id == original_grasp_id:
                                original_validation = candidate_validation
                                original_payload = candidate_payload
                            if candidate_validation.feasible:
                                grasp_id = candidate_id
                                validation = candidate_validation
                                validation_payload = candidate_payload
                                selection_record["selected_grasp_id"] = candidate_id
                                selection_record["selected_reason"] = "feasible"
                                selection_record["feasible_candidate_found"] = True
                                break
                            if recoverable_choice is not None:
                                continue
                            if not (
                                bool(config.target_feasibility_prefer_recoverable_failure)
                                and bool(config.validation_guided_recovery)
                                and remaining_budget > 0
                            ):
                                continue
                            recovered_blockers = validation_recovery_blockers(
                                candidate_payload,
                                target_id=target_id,
                                removed_object_ids=unavailable_ids,
                                removable_object_ids=removable,
                            )
                            if not recovered_blockers:
                                continue
                            candidate_score = planner_candidate_score(plan_result, candidate_id)
                            selection_record["recoverable_candidate_found"] = True
                            selection_record["recoverable_candidate_grasp_id"] = candidate_id
                            selection_record["recoverable_candidate_score"] = candidate_score
                            selection_record["recoverable_candidate_blockers"] = list(recovered_blockers)
                            max_recoverable_score = config.target_feasibility_recoverable_max_planner_score
                            if max_recoverable_score is not None:
                                if candidate_score is None:
                                    selection_record["recoverable_rejection_reason"] = "missing_planner_score"
                                    continue
                                if float(candidate_score) > float(max_recoverable_score):
                                    selection_record["recoverable_rejection_reason"] = "planner_score_above_threshold"
                                    continue
                            recoverable_choice = (candidate_id, candidate_validation, candidate_payload)
                            selection_record["recoverable_candidate_accepted"] = True
                            selection_record["recoverable_rejection_reason"] = ""
                        if validation is None:
                            if recoverable_choice is not None:
                                grasp_id, validation, validation_payload = recoverable_choice
                                selection_record["selected_grasp_id"] = grasp_id
                                selection_record["selected_reason"] = "recoverable_failure"
                            else:
                                if original_validation is None:
                                    original_validation, original_payload = validate_candidate(original_grasp_id)
                                    selection_record["evaluated_grasp_ids"].append(original_grasp_id)
                                grasp_id = original_grasp_id
                                validation = original_validation
                                validation_payload = original_payload
                                selection_record["selected_grasp_id"] = grasp_id
                                selection_record["selected_reason"] = "original_failure"
                        step_record["target_feasibility_selection"] = selection_record
                    elif config.bin_wall_safe_target_selection:
                        max_safe_candidates = max(1, int(config.max_bin_wall_safe_candidates))
                        candidate_ids = ordered_target_grasp_candidates(
                            plan_result,
                            proposals,
                            selected_grasp_id=original_grasp_id,
                            rejected_grasp_ids=set(),
                        )[:max_safe_candidates]
                        safe_selection_record: JsonDict = {
                            "enabled": True,
                            "original_grasp_id": original_grasp_id,
                            "max_candidates": int(max_safe_candidates),
                            "evaluated_grasp_ids": [],
                            "switched_to_grasp_id": "",
                            "kept_original_failure": False,
                        }
                        original_validation: ValidationResult | None = None
                        original_payload: JsonDict = {}
                        for candidate_id in candidate_ids:
                            candidate_validation, candidate_payload = validate_candidate(candidate_id)
                            safe_selection_record["evaluated_grasp_ids"].append(candidate_id)
                            if candidate_id == original_grasp_id:
                                original_validation = candidate_validation
                                original_payload = candidate_payload
                                validation = candidate_validation
                                validation_payload = candidate_payload
                                if candidate_validation.feasible or not is_bin_wall_only_failure(candidate_payload):
                                    break
                                continue
                            if candidate_validation.feasible:
                                grasp_id = candidate_id
                                validation = candidate_validation
                                validation_payload = candidate_payload
                                safe_selection_record["switched_to_grasp_id"] = candidate_id
                                break
                        if not safe_selection_record["switched_to_grasp_id"]:
                            if original_validation is None:
                                original_validation, original_payload = validate_candidate(original_grasp_id)
                                safe_selection_record["evaluated_grasp_ids"].append(original_grasp_id)
                            grasp_id = original_grasp_id
                            validation = original_validation
                            validation_payload = original_payload
                            safe_selection_record["kept_original_failure"] = bool(
                                validation is not None and not validation.feasible
                            )
                        step_record["bin_wall_safe_selection"] = safe_selection_record
                    else:
                        while True:
                            validation, validation_payload = validate_candidate(grasp_id)
                            if (
                                not config.retry_bin_wall_target_grasps
                                or validation.feasible
                                or not is_bin_wall_only_failure(validation_payload)
                                or len(target_attempts) >= max_target_attempts
                            ):
                                break
                            rejected_target_grasps.add(grasp_id)
                            candidates = ordered_target_grasp_candidates(
                                plan_result,
                                proposals,
                                selected_grasp_id=str(plan_result.get("selected_grasp_id") or grasp_id),
                                rejected_grasp_ids=rejected_target_grasps,
                            )
                            if not candidates:
                                break
                            grasp_id = candidates[0]
                except KeyError:
                    return finish(
                        success=False,
                        terminal_reason="selected_grasp_not_in_fresh_proposals",
                    )

                step_record["action"] = "target_grasp"
                step_record["selected_grasp_id"] = grasp_id
                step_record["validation"] = validation_payload
                step_record["target_grasp_attempts"] = target_attempts
                steps.append(step_record)
                if validation is None:
                    raise RuntimeError("Target grasp validation was not executed.")
                if (
                    config.lock_planned_removal_set
                    and not validation.feasible
                    and remaining_budget > 0
                    and locked_phase_count < max_locked_phases
                    and (
                        force_target_attempt
                        or (
                            config.validation_guided_recovery
                            and config.recover_after_any_target_failure
                        )
                    )
                ):
                    recovered_blockers: list[str] = []
                    if config.validation_guided_recovery:
                        recovered_blockers = validation_recovery_blockers(
                            validation_payload,
                            target_id=target_id,
                            removed_object_ids=unavailable_ids,
                            removable_object_ids=removable,
                        )
                    if not force_target_attempt and not recovered_blockers:
                        return finish(
                            success=False,
                            terminal_reason=validation.failure_stage.value,
                        )
                    step_record["stop_policy"]["recover_after_failed_target_attempt"] = True
                    step_record["stop_policy"]["locked_phase_count"] = int(locked_phase_count)
                    if config.validation_guided_recovery:
                        pending_validation_recovery_blockers = list(recovered_blockers)
                        step_record["stop_policy"]["validation_recovery_blockers_observed"] = list(
                            pending_validation_recovery_blockers
                        )
                    locked_removal_sequence = []
                    continue
                return finish(
                    success=bool(validation.feasible),
                    terminal_reason="target_grasp_success" if validation.feasible else validation.failure_stage.value,
                )

            remove_object_id = str(next_action.get("remove_object_id") or "")
            if remaining_budget <= 0:
                step_record["action"] = "removal_budget_exhausted"
                step_record["remove_object_id"] = remove_object_id
                steps.append(step_record)
                return finish(
                    success=False,
                    terminal_reason="removal_budget_exhausted",
                )

            if remove_object_id not in removable:
                step_record["action"] = "invalid_removal"
                step_record["remove_object_id"] = remove_object_id
                steps.append(step_record)
                return finish(
                    success=False,
                    terminal_reason="invalid_removal_action",
                )

            step_record["action"] = "remove_object"
            step_record["remove_object_id"] = remove_object_id
            planner_removed.append(remove_object_id)
            removed.append(remove_object_id)
            if config.lock_planned_removal_set:
                if locked_removal_sequence and locked_removal_sequence[0] == remove_object_id:
                    locked_removal_sequence.pop(0)
                else:
                    locked_removal_sequence = [item for item in locked_removal_sequence if item != remove_object_id]
                step_record["stop_policy"]["locked_removal_sequence_after"] = list(locked_removal_sequence)
                if not locked_removal_sequence and config.force_target_after_locked_set:
                    force_target_attempt_next = True
            steps.append(step_record)

            if self._apply_non_target_intervention is None:
                next_scene = step_scene.without_objects({remove_object_id})
                if config.resettle_after_removal:
                    next_scene = self._resettle(next_scene, remove_object_id, step_index)
            else:
                try:
                    next_scene = self._apply_non_target_intervention(
                        step_scene,
                        target_id,
                        remove_object_id,
                        step_index,
                    )
                except PlacementRejected as exc:
                    removed.pop()
                    planner_removed.pop()
                    step_record['placement_rejected'] = exc.record
                    step_record['intervention_executed'] = False
                    return finish(success=False, terminal_reason='placement_no_feasible_pose')
            step_record["resettle"] = (
                dict(next_scene.metadata.get("fresh_resettle") or {}) or None
            )
            if config.resettle_after_removal and config.terminate_on_invalid_resettle:
                invalid_reason = invalid_resettle_terminal_reason(step_record["resettle"])
                if invalid_reason:
                    step_record["resettle_terminal_failure"] = invalid_reason
                    return finish(success=False, terminal_reason=invalid_reason)
            current_scene = next_scene

        return finish(
            success=False,
            terminal_reason="max_steps_reached_without_target_attempt",
        )

    @staticmethod
    def _episode_result(
        *,
        scene: StableScene,
        target_id: str,
        config: FreshLoopConfig,
        success: bool,
        terminal_reason: str,
        removed: list[str],
        steps: list[JsonDict],
    ) -> JsonDict:
        return {
            "scene_id": scene.scene_id,
            "target_id": target_id,
            "planner_type": config.planner_type,
            "success": bool(success),
            "terminal_reason": terminal_reason,
            "num_removals": len(removed),
            "removal_sequence": list(removed),
            "steps": steps,
        }


def _spec_from_dict(payload: JsonDict) -> ObjectSpec:
    return ObjectSpec(
        name=str(payload["name"]),
        primitive_type=str(payload["primitive_type"]),
        size=tuple(float(v) for v in payload["size"]),
        mass=float(payload["mass"]),
        rgba=tuple(float(v) for v in payload["rgba"]),
        mesh_path=payload.get("mesh_path"),
        collision_mesh_paths=tuple(str(v) for v in payload["collision_mesh_paths"])
        if payload.get("collision_mesh_paths")
        else None,
        texture_path=payload.get("texture_path"),
        mesh_scale=tuple(float(v) for v in payload["mesh_scale"])
        if payload.get("mesh_scale")
        else None,
        mesh_offset=tuple(float(v) for v in payload["mesh_offset"])
        if payload.get("mesh_offset")
        else None,
        proxy_quaternion_wxyz=tuple(float(v) for v in payload["proxy_quaternion_wxyz"])
        if payload.get("proxy_quaternion_wxyz")
        else None,
        notes=payload.get("notes"),
    )


def load_scene_json(path: Path) -> StableScene:
    payload = json.loads(path.read_text(encoding="utf-8"))
    objects = [
        SceneObjectState(
            object_id=str(obj["object_id"]),
            asset_name=str(obj["asset_name"]),
            scale=float(obj["scale"]),
            spec=_spec_from_dict(obj["spec"]),
            pose=Pose(
                position=tuple(float(v) for v in obj["pose"]["position"]),
                quaternion_wxyz=tuple(float(v) for v in obj["pose"]["quaternion_wxyz"]),
            ),
        )
        for obj in payload["objects"]
    ]
    return StableScene(
        scene_id=str(payload["scene_id"]),
        objects=tuple(objects),
        target_ids=tuple(str(v) for v in payload.get("target_ids", [])),
        bin_size=tuple(float(v) for v in payload["bin_size"]),
        wall_thickness=float(payload["wall_thickness"]),
        metadata=dict(payload.get("metadata", {})),
    )


def clear_provider_caches(runner: DatasetPipelineRunner) -> None:
    for provider in (runner.parallel_provider, runner.suction_provider):
        for attr in ("_observation_cache", "_generation_observation_cache", "_renderer_pool"):
            cache = getattr(provider, attr, None)
            if hasattr(cache, "clear"):
                cache.clear()


def scene_observation_cache_key(
    scene: StableScene,
    *,
    target_id: str,
    namespace: str,
) -> str:
    """Hash the exact physical observation state for cross-method reuse."""

    payload = {
        "cache_protocol": "fresh_scene_observation_v1",
        "namespace": namespace,
        "scene_id": scene.scene_id,
        "target_id": target_id,
        "bin_size": list(scene.bin_size),
        "wall_thickness": float(scene.wall_thickness),
        "objects": [
            {
                "object_id": obj.object_id,
                "asset_name": obj.asset_name,
                "scale": float(obj.scale),
                "position": list(obj.pose.position),
                "quaternion_wxyz": list(obj.pose.quaternion_wxyz),
            }
            for obj in scene.objects
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_or_compute_locked_pickle(
    path: Path,
    compute: Callable[[], Any],
) -> tuple[Any, bool]:
    """Atomically share deterministic observation work between CLI processes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if path.exists():
            with path.open("rb") as handle:
                return pickle.load(handle), True
        value = compute()
        temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary_path.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        return value, False


def scene_with_refreshed_visibility(runner: DatasetPipelineRunner, scene: StableScene, target_id: str) -> StableScene:
    clear_provider_caches(runner)
    ranking = runner.rank_targets_by_visibility(scene)
    metadata = dict(scene.metadata)
    target_selection = dict(metadata.get("target_selection") or {})
    target_selection["visibility_ranking"] = ranking
    target_selection["fresh_visibility_target_id"] = target_id
    metadata["target_selection"] = target_selection
    return replace(scene, metadata=metadata)


def _empty_labels_payload(proposals: list[GraspProposal]) -> JsonDict:
    return {
        "grasp_feasibility": [
            {
                "grasp_id": proposal.grasp_id,
                "target_id": proposal.target_id,
                "grasp_type": proposal.grasp_type.value,
                "proposal_score": float(proposal.proposal_score),
                "position": [float(value) for value in proposal.pose.position],
                "orientation": [float(value) for value in proposal.pose.quaternion_wxyz],
                "approach_dir": [float(value) for value in proposal.approach_vector],
                "lift_dir": [float(value) for value in proposal.lift_vector],
                "feasible": False,
                "failure_stage": "unknown",
                "stage_blockers": {},
                "notes": "Fresh closed-loop planning sample; labels intentionally withheld.",
            }
            for proposal in proposals
        ],
        "dependencies": [],
        "planning_labels": [],
        "planning_summary": {},
    }


def _edge_payload(values: Any, idx: int) -> dict[str, float]:
    width = values.shape[1] if values.ndim == 2 else 0
    progress_any = float(values[idx, 0]) if width > 0 else 0.0
    sufficient = float(values[idx, 1]) if width > 1 else progress_any
    approach = float(values[idx, 2]) if width > 2 else 0.0
    lift = float(values[idx, 3]) if width > 3 else 0.0
    return {
        "any": progress_any,
        "sufficient": sufficient,
        "app": approach,
        "lift": lift,
    }


class CachedDependencyMapPredictor:
    """Stage-2 dependency predictor that avoids reloading torch checkpoints per step."""

    def __init__(
        self,
        *,
        checkpoint: str | Path | None,
        backend: str,
        device: str,
        batch_size: int,
        heuristic_cfg: JsonDict | None,
        torch_loader: Callable[[Path, str], tuple[Any, dict[str, Any]]] = load_torch_model,
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.backend = str(backend)
        self.device = str(device)
        self.batch_size = int(batch_size)
        self.heuristic_cfg = dict(heuristic_cfg or {})
        self._torch_model: Any | None = None
        self._torch_dims: dict[str, Any] | None = None
        if self.backend == "torch":
            if self.checkpoint is None:
                raise FileNotFoundError("Torch backend requires a checkpoint.")
            self._torch_model, self._torch_dims = torch_loader(self.checkpoint, self.device)

    def predict(
        self,
        dataset: HeteroGraphDataset,
    ) -> tuple[
        dict[str, dict[str, dict[str, dict[str, float]]]],
        dict[str, dict[str, dict[str, dict[str, float]]]],
    ]:
        if self.backend != "torch":
            return predict_dependency_maps(
                dataset,
                checkpoint=self.checkpoint,
                backend=self.backend,
                device=self.device,
                batch_size=self.batch_size,
                heuristic_cfg=self.heuristic_cfg,
            )
        return self._predict_torch(dataset)

    def _predict_torch(
        self,
        dataset: HeteroGraphDataset,
    ) -> tuple[
        dict[str, dict[str, dict[str, dict[str, float]]]],
        dict[str, dict[str, dict[str, dict[str, float]]]],
    ]:
        if self._torch_model is None:
            raise RuntimeError("Torch predictor was not initialized.")
        import torch

        predicted: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        oracle: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        for samples in iter_sample_batches(dataset, batch_size=self.batch_size, shuffle=False, seed=0):
            batch_np = collate_samples(samples)
            batch_np_model = adapt_batch_feature_dims(batch_np, self._torch_dims or {})
            batch = batch_to_torch(batch_np_model, self.device)
            with torch.no_grad():
                outputs = self._torch_model(batch)
                probs = torch.sigmoid(extract_edge_logits(outputs)).detach().cpu().numpy()

            labels = batch_np["edge_label_og"].astype("float32")
            for idx, sample_id in enumerate(batch_np["edge_sample_ids"]):
                object_id = batch_np["edge_object_ids"][idx]
                grasp_id = batch_np["edge_grasp_ids"][idx]
                predicted.setdefault(sample_id, {}).setdefault(grasp_id, {})[object_id] = _edge_payload(probs, idx)
                oracle.setdefault(sample_id, {}).setdefault(grasp_id, {})[object_id] = _edge_payload(labels, idx)
        return predicted, oracle


def write_transient_graph_sample(
    *,
    runner: DatasetPipelineRunner,
    scene: StableScene,
    target_id: str,
    proposals: list[GraspProposal],
    output_root: Path,
) -> SampleRef:
    scene_dir = output_root / "scenes" / scene.scene_id
    target_dir = scene_dir / "targets" / target_id
    scene_path = scene_dir / "scene.json"
    proposals_path = target_dir / "proposals.json"
    labels_path = target_dir / "labels.json"
    manifest_path = target_dir / "manifest.json"

    write_json(scene_path, scene.to_dict())
    write_json(proposals_path, runner._proposal_export_payload(target_id, proposals))
    write_json(labels_path, _empty_labels_payload(proposals))
    shared_observations = write_transient_shared_observations(
        runner=runner,
        scene=scene,
        target_id=target_id,
        scene_dir=scene_dir,
    )

    manifest = runner._build_manifest(
        scene=scene,
        target_id=target_id,
        proposals=proposals,
        results={},
        dependencies=[],
        planning_summary=_preview_planning_summary_stub(),
        scene_path=str(scene_path),
        proposal_path=str(proposals_path),
        label_path=str(labels_path),
        shared_observations=shared_observations,
    )
    manifest["statistics"]["fresh_closed_loop_labels_withheld"] = True
    write_json(manifest_path, manifest)

    return SampleRef(
        sample_id=f"{scene.scene_id}__{target_id}",
        scene_id=scene.scene_id,
        target_id=target_id,
        scene_dir=scene_dir,
        target_dir=target_dir,
        scene_path=scene_path,
        proposals_path=proposals_path,
        labels_path=labels_path,
        manifest_path=manifest_path,
    )


def _preview_planning_summary_stub() -> Any:
    from grasp_dependency_dataset.common.types import PlanningStatus, PlanningSummary

    return PlanningSummary(
        best_grasp_id=None,
        minimal_blocker_set=tuple(),
        oracle_removal_sequence=tuple(),
        status=PlanningStatus.UNSOLVED_WITHIN_DEPTH,
        status_counts={},
        terminal_reason_counts={},
    )


def _depth_to_rgb_uint8(depth_m: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth_m)
    if not np.any(finite):
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    low = float(np.percentile(depth_m[finite], 5))
    high = float(np.percentile(depth_m[finite], 95))
    if high <= low:
        high = low + 1e-3
    normalized = np.clip((depth_m - low) / (high - low), 0.0, 1.0)
    gray = ((1.0 - normalized) * 255.0).astype(np.uint8)
    gray[~finite] = 0
    return np.repeat(gray[..., None], 3, axis=2)


def write_transient_shared_observations(
    *,
    runner: DatasetPipelineRunner,
    scene: StableScene,
    target_id: str,
    scene_dir: Path,
) -> JsonDict:
    """Render shared RGB-D assets used by graph features for a transient scene."""

    from PIL import Image

    observation = runner._get_target_observation(scene, target_id)
    shared_dir = scene_dir / "observations" / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = shared_dir / "rgb.png"
    depth_path = shared_dir / "depth.png"

    color = np.asarray(observation.color)
    if color.dtype != np.uint8:
        color = np.clip(color, 0, 255).astype(np.uint8)
    Image.fromarray(color).save(rgb_path)
    Image.fromarray(_depth_to_rgb_uint8(np.asarray(observation.depth_m, dtype=np.float32))).save(depth_path)
    return {
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
    }


class RealFreshMujocoRuntime:
    """Adapter from the injected fresh loop to the existing project pipeline."""

    def __init__(
        self,
        *,
        runner: DatasetPipelineRunner,
        output_dir: Path,
        graph_cfg: JsonDict,
        checkpoint: Path | None,
        prediction_backend: str,
        device: str,
        feature_config: JsonDict,
        class_map: dict[str, int] | None,
        planner_params: PlannerParams,
        planner_seed: int,
        planner_type: str,
        proposal_protocol_config: FreshProposalProtocolConfig | None = None,
        dependency_predictor: CachedDependencyMapPredictor | None = None,
        shared_observation_cache_dir: Path | None = None,
        shared_observation_cache_namespace: str = "",
    ) -> None:
        self.runner = runner
        self.output_dir = output_dir
        self.graph_cfg = graph_cfg
        self.checkpoint = checkpoint
        self.prediction_backend = prediction_backend
        self.device = device
        self.feature_config = feature_config
        self.class_map = class_map
        self.planner_params = planner_params
        self.planner_seed = planner_seed
        self.planner_type = planner_type
        self.proposal_protocol_config = proposal_protocol_config or FreshProposalProtocolConfig()
        self.shared_observation_cache_dir = shared_observation_cache_dir
        self.shared_observation_cache_namespace = shared_observation_cache_namespace
        self.last_proposal_stats: JsonDict = {}
        self.dependency_predictor = dependency_predictor or CachedDependencyMapPredictor(
            checkpoint=checkpoint,
            backend=prediction_backend,
            device=device,
            batch_size=1,
            heuristic_cfg=graph_cfg.get("geometry_heuristic", {}),
        )

    def proposals(self, scene: StableScene, target_id: str) -> list[GraspProposal]:
        protocol = getattr(self, "proposal_protocol_config", FreshProposalProtocolConfig())

        def compute() -> tuple[list[GraspProposal], JsonDict]:
            clear_provider_caches(self.runner)
            if protocol.mode == "stage1_aligned_isolated":
                proposals, stats = generate_stage1_aligned_proposals(
                    runner=self.runner,
                    scene=scene,
                    target_id=target_id,
                    config=protocol,
                )
                return proposals, dict(stats)
            if protocol.mode == "scene":
                proposals, stats = self.runner.generate_proposals_with_stats(scene, target_id)
                return proposals, {"source_mode": "scene", **dict(stats)}
            raise ValueError(f"Unsupported proposal protocol mode: {protocol.mode}")

        cache_dir = getattr(self, "shared_observation_cache_dir", None)
        if cache_dir is None:
            proposals, stats = compute()
            cache_hit = False
        else:
            key = scene_observation_cache_key(
                scene,
                target_id=target_id,
                namespace=str(getattr(self, "shared_observation_cache_namespace", "")),
            )
            (proposals, stats), cache_hit = load_or_compute_locked_pickle(
                Path(cache_dir) / "proposals" / f"{key}.pkl",
                compute,
            )
        self.last_proposal_stats = {
            **dict(stats),
            "shared_observation_cache_hit": bool(cache_hit),
        }
        return list(proposals)

    def proposal_trace(self, scene: StableScene, target_id: str, proposals: list[GraspProposal]) -> JsonDict:
        return summarize_proposal_trace(proposals, getattr(self, "last_proposal_stats", {}))

    def refresh(self, scene: StableScene, target_id: str) -> StableScene:
        cache_dir = getattr(self, "shared_observation_cache_dir", None)
        if cache_dir is None:
            return scene_with_refreshed_visibility(self.runner, scene, target_id)

        key = scene_observation_cache_key(
            scene,
            target_id=target_id,
            namespace=str(getattr(self, "shared_observation_cache_namespace", "")),
        )

        def compute() -> list[JsonDict]:
            clear_provider_caches(self.runner)
            return list(self.runner.rank_targets_by_visibility(scene))

        ranking, cache_hit = load_or_compute_locked_pickle(
            Path(cache_dir) / "visibility" / f"{key}.pkl",
            compute,
        )
        metadata = dict(scene.metadata)
        target_selection = dict(metadata.get("target_selection") or {})
        target_selection["visibility_ranking"] = list(ranking)
        target_selection["fresh_visibility_target_id"] = target_id
        target_selection["shared_observation_cache_hit"] = bool(cache_hit)
        metadata["target_selection"] = target_selection
        return replace(scene, metadata=metadata)

    def validate(self, scene: StableScene, target_id: str, proposal: GraspProposal) -> ValidationResult:
        return self.runner.validator.validate(scene=scene, target_id=target_id, proposal=proposal)

    def resettle(self, scene: StableScene, removed_object_id: str, step_index: int) -> StableScene:
        if not scene.objects:
            return scene
        settled = self.runner.scene_generator._settle_states(
            list(scene.objects),
            settle_steps=max(0, int(getattr(self, "resettle_steps", 1200))),
        )
        objects = self.runner.scene_generator._apply_settled_poses(list(scene.objects), settled.body_poses)
        out_of_bin_object_ids: list[str] = []
        if hasattr(self.runner.scene_generator, "_out_of_bin_object_ids"):
            out_of_bin_object_ids = list(self.runner.scene_generator._out_of_bin_object_ids(objects))
        accepted = bool(getattr(settled, "stable", False)) and not out_of_bin_object_ids
        metadata = dict(scene.metadata)
        metadata["fresh_resettle"] = {
            "removed_object_id": removed_object_id,
            "step_index": int(step_index),
            **dict(getattr(settled, "metadata", {}) or {}),
            "stable": bool(getattr(settled, "stable", False)),
            "accepted": bool(accepted),
            "out_of_bin_object_ids": list(out_of_bin_object_ids),
        }
        if bool(getattr(self, "reject_unstable_resettle", False)) and not accepted:
            return replace(scene, metadata=metadata)
        return replace(scene, objects=tuple(objects), metadata=metadata)

    def plan(
        self,
        scene: StableScene,
        target_id: str,
        proposals: list[GraspProposal],
        remaining_budget: int,
    ) -> JsonDict:
        scratch_root = self.output_dir / "scratch" / self.planner_type
        ref = write_transient_graph_sample(
            runner=self.runner,
            scene=scene,
            target_id=target_id,
            proposals=proposals,
            output_root=scratch_root,
        )
        dataset = HeteroGraphDataset(
            scratch_root,
            refs=[ref],
            sample_ids=[ref.sample_id],
            feature_config=self.feature_config,
            class_map=self.class_map,
        )
        predicted_maps, oracle_maps = self.dependency_predictor.predict(dataset)
        graph_sample = dataset[0]
        labels_payload = read_json(ref.labels_path)
        planner_sample = planner_sample_from_graph(
            graph_sample,
            labels_payload,
            predicted_maps.get(graph_sample.sample_id, {}),
            oracle_maps.get(graph_sample.sample_id, {}),
        )
        params = replace(self.planner_params, max_steps=max(0, int(remaining_budget)))
        planner = DependencyGuidedPlanner(params, rng_seed=self.planner_seed)
        return planner.plan(planner_sample, planner_type=self.planner_type, closed_loop=True)


def load_initial_scene(ref: SampleRef) -> StableScene:
    scene = load_scene_json(ref.scene_path)
    if ref.target_id not in scene.target_ids:
        scene = replace(scene, target_ids=tuple(sorted(set(scene.target_ids) | {ref.target_id})))
    return scene


def select_refs(
    refs: list[SampleRef],
    *,
    split_file: Path | None,
    split: str,
    sample_ids: list[str],
    max_episodes: int | None,
) -> list[SampleRef]:
    selected = refs
    if split_file is not None and split_file.exists():
        splits = json.loads(split_file.read_text(encoding="utf-8"))
        allowed = set(str(item) for item in splits.get(split, []))
        selected = [ref for ref in selected if ref.sample_id in allowed]
    if sample_ids:
        allowed = set(sample_ids)
        selected = [ref for ref in selected if ref.sample_id in allowed]
    if max_episodes is not None:
        selected = selected[: max(0, int(max_episodes))]
    return selected


def summarize_episodes(episodes: list[JsonDict]) -> dict[str, JsonDict]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for episode in episodes:
        grouped[str(episode["planner_type"])].append(episode)

    summary: dict[str, JsonDict] = {}
    for planner_type, rows in sorted(grouped.items()):
        n = len(rows)
        successes = sum(1 for row in rows if row.get("success"))
        removals = [float(row.get("num_removals", 0)) for row in rows]
        fallback_removals = [float(row.get("num_fallback_removals", 0)) for row in rows]
        planner_removals = [float(row.get("num_planner_removals", 0)) for row in rows]
        fallback_rows = [row for row in rows if int(row.get("num_fallback_removals", 0)) > 0]
        recovered_fallback_rows = [
            row
            for row in fallback_rows
            if bool(
                row.get("shared_fallback", {})
                .get("proposal_recovery", {})
                .get("recovered", False)
            )
        ]
        reasons = Counter(str(row.get("terminal_reason", "")) for row in rows)
        summary[planner_type] = {
            "num_episodes": n,
            "success_rate": successes / max(n, 1),
            "mean_removals": sum(removals) / max(n, 1),
            "mean_total_removals": sum(removals) / max(n, 1),
            "mean_fallback_removals": sum(fallback_removals) / max(n, 1),
            "mean_planner_removals": sum(planner_removals) / max(n, 1),
            "fallback_trigger_rate": len(fallback_rows) / max(n, 1),
            "proposal_recovery_rate_given_fallback": (
                len(recovered_fallback_rows) / max(len(fallback_rows), 1)
            ),
            "terminal_reason_counts": dict(sorted(reasons.items())),
        }
    return summary


def write_summary_csv(path: Path, summary: dict[str, JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "planner_type",
            "num_episodes",
            "success_rate",
            "mean_removals",
            "mean_total_removals",
            "mean_fallback_removals",
            "mean_planner_removals",
            "fallback_trigger_rate",
            "proposal_recovery_rate_given_fallback",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for planner_type, row in sorted(summary.items()):
            payload = {"planner_type": planner_type}
            payload.update({key: row[key] for key in fieldnames if key != "planner_type"})
            writer.writerow(payload)


def run_cli(args: argparse.Namespace) -> tuple[list[JsonDict], dict[str, JsonDict]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    planner_cfg = load_yaml(Path(args.planner_config))
    graph_cfg = load_config(Path(args.graph_config))
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    backend = args.prediction_backend
    if backend == "auto":
        backend = choose_backend(graph_cfg, checkpoint, "auto")
    device = "cpu"
    if backend == "torch":
        device = resolve_torch_device(str(args.device))

    runner = DatasetPipelineRunner.from_configs(
        config_dir="configs",
        output_root=output_dir / "pipeline_cache",
        proposal_config_path=args.proposal_config,
        scene_config_path=args.scene_config,
    )

    refs = discover_sample_refs(Path(args.dataset_root))
    selected_refs = select_refs(
        refs,
        split_file=Path(args.split_file) if args.split_file else None,
        split=args.split,
        sample_ids=[item for item in args.sample_ids.split(",") if item],
        max_episodes=args.max_episodes,
    )
    if not selected_refs:
        raise RuntimeError("No episodes selected. Check dataset root, split file, sample ids, and max episodes.")

    prediction_cfg = planner_cfg.get("prediction", {})
    class_map = checkpoint_class_map(checkpoint, backend, None)
    feature_config = dict(graph_cfg.get("features", {}))
    feature_config.update(planner_cfg.get("features", {}))

    raw_planner_cfg = dict(planner_cfg.get("planner", {}))
    raw_planner_cfg["max_steps"] = int(args.max_steps)
    raw_planner_cfg["run_closed_loop"] = True
    raw_planner_cfg["run_open_loop"] = False
    if args.calibration_thresholds:
        raw_planner_cfg["use_calibrated_thresholds"] = True
        raw_planner_cfg["calibration_thresholds_path"] = args.calibration_thresholds
    calibration_payload, _calibration_path = resolve_calibration_thresholds(raw_planner_cfg, checkpoint)
    raw_planner_cfg, _applied = apply_calibration_to_planner_cfg(raw_planner_cfg, calibration_payload)
    planner_params = PlannerParams.from_dict(raw_planner_cfg)
    planner_seed = int(raw_planner_cfg.get("seed", 7))
    dependency_predictor = CachedDependencyMapPredictor(
        checkpoint=checkpoint,
        backend=backend,
        device=device,
        batch_size=1,
        heuristic_cfg=graph_cfg.get("geometry_heuristic", {}),
    )
    proposal_protocol_config = FreshProposalProtocolConfig(
        mode=str(args.proposal_protocol),
        augmented_parallel_top_k=int(args.isolated_augmented_parallel_top_k),
        augmented_suction_top_k=int(args.isolated_augmented_suction_top_k),
        max_augmented_parallel=int(args.max_isolated_augmented_parallel),
        max_augmented_suction=int(args.max_isolated_augmented_suction),
        augmented_max_approach_angle_from_down_deg=(
            None
            if args.isolated_augmented_max_approach_angle_from_down_deg is None
            else float(args.isolated_augmented_max_approach_angle_from_down_deg)
        ),
        nms_position_thresh=float(args.isolated_nms_position_thresh),
        nms_direction_cos_thresh=float(args.isolated_nms_direction_cos_thresh),
        nms_inplane_cos_thresh=float(args.isolated_nms_inplane_cos_thresh),
    )
    shared_observation_cache_dir = (
        Path(args.shared_observation_cache_dir).resolve()
        if str(args.shared_observation_cache_dir).strip()
        else None
    )
    proposal_config_bytes = Path(args.proposal_config).resolve().read_bytes()
    shared_observation_cache_namespace = hashlib.sha256(
        b"fresh_scene_observation_v1\0"
        + proposal_config_bytes
        + b"\0"
        + repr(proposal_protocol_config).encode("utf-8")
    ).hexdigest()

    episodes_path = output_dir / "episodes.json"
    episodes: list[JsonDict] = []
    if bool(args.resume) and episodes_path.exists():
        loaded = json.loads(episodes_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"Resume file must contain a JSON list: {episodes_path}")
        episodes = [dict(row) for row in loaded if isinstance(row, dict)]
    completed_episode_keys = {
        (str(row.get("planner_type", "")), str(row.get("sample_id", "")))
        for row in episodes
    }
    planner_types = [item.strip() for item in args.planner_types.split(",") if item.strip()]
    for planner_type in planner_types:
        runtime = RealFreshMujocoRuntime(
            runner=runner,
            output_dir=output_dir,
            graph_cfg=graph_cfg,
            checkpoint=checkpoint,
            prediction_backend=backend,
            device=device,
            feature_config=feature_config,
            class_map=class_map,
            planner_params=planner_params,
            planner_seed=planner_seed,
            planner_type=planner_type,
            proposal_protocol_config=proposal_protocol_config,
            dependency_predictor=dependency_predictor,
            shared_observation_cache_dir=shared_observation_cache_dir,
            shared_observation_cache_namespace=shared_observation_cache_namespace,
        )
        runtime.resettle_steps = int(args.resettle_steps)
        episode_runner = FreshClosedLoopEpisodeRunner(
            propose=runtime.proposals,
            plan=runtime.plan,
            validate_target_grasp=runtime.validate,
            resettle_after_removal=runtime.resettle,
            refresh_observation_metadata=runtime.refresh,
            proposal_trace=runtime.proposal_trace,
        )
        config = FreshLoopConfig(
            planner_type=planner_type,
            max_steps=int(args.max_steps),
            resettle_after_removal=not bool(args.no_resettle),
            resettle_steps=int(args.resettle_steps),
            refresh_visibility=not bool(args.no_refresh_visibility),
            lock_planned_removal_set=bool(args.lock_planned_removal_set),
            force_target_after_locked_set=not bool(args.no_force_target_after_locked_set),
            max_locked_removal_phases=int(args.max_locked_removal_phases),
            validation_guided_recovery=bool(args.validation_guided_recovery),
            recover_after_any_target_failure=bool(args.recover_after_any_target_failure),
            retry_bin_wall_target_grasps=bool(args.retry_bin_wall_target_grasps),
            max_target_grasp_retries=int(args.max_target_grasp_retries),
            bin_wall_safe_target_selection=bool(args.bin_wall_safe_target_selection),
            max_bin_wall_safe_candidates=int(args.max_bin_wall_safe_candidates),
            target_feasibility_selection=bool(args.target_feasibility_selection),
            max_target_feasibility_candidates=int(args.max_target_feasibility_candidates),
            target_feasibility_prefer_recoverable_failure=not bool(
                args.no_target_feasibility_prefer_recoverable_failure
            ),
            target_feasibility_recoverable_max_planner_score=args.target_feasibility_recoverable_max_planner_score,
            pre_removal_target_feasibility_guard=bool(args.pre_removal_target_feasibility_guard),
            enable_shared_visibility_fallback=bool(args.enable_shared_visibility_fallback),
            max_shared_fallback_removals=args.max_shared_fallback_removals,
            shared_fallback_min_visible_pixels=int(args.shared_fallback_min_visible_pixels),
            terminate_on_invalid_resettle=bool(args.terminate_on_invalid_resettle),
        )
        for ref in selected_refs:
            episode_key = (planner_type, ref.sample_id)
            if episode_key in completed_episode_keys:
                print(
                    f"[fresh-loop] resume skip {planner_type} {ref.sample_id}",
                    flush=True,
                )
                continue
            scene = load_initial_scene(ref)
            episode = episode_runner.run_episode(scene, ref.target_id, config)
            episode["sample_id"] = ref.sample_id
            episode["initial_scene_path"] = str(ref.scene_path)
            episode["proposal_protocol"] = proposal_protocol_config.mode
            episodes.append(episode)
            completed_episode_keys.add(episode_key)
            partial_summary = summarize_episodes(episodes)
            write_json(episodes_path, episodes)
            write_json(output_dir / "summary.json", partial_summary)
            write_summary_csv(output_dir / "summary.csv", partial_summary)
            print(
                f"[fresh-loop] {planner_type} {ref.sample_id} "
                f"success={episode['success']} total_removals={episode['num_removals']} "
                f"fallback_removals={episode['num_fallback_removals']} "
                f"planner_removals={episode['num_planner_removals']} "
                f"reason={episode['terminal_reason']}",
                flush=True,
            )

    summary = summarize_episodes(episodes)
    write_json(episodes_path, episodes)
    write_json(output_dir / "summary.json", summary)
    write_summary_csv(output_dir / "summary.csv", summary)
    return episodes, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="outputs/all_targets_visualized_500scenes_strict_roi")
    parser.add_argument("--planner-config", default="configs/planner.yaml")
    parser.add_argument(
        "--graph-config",
        default="configs/hetero_gnn_500_progress_edge_context_relation_edge_context_no_interactions_seed7.yaml",
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--prediction-backend", default="geometry_heuristic")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--proposal-config", default="configs/proposal_sources.yaml")
    parser.add_argument(
        "--shared-observation-cache-dir",
        default="",
        help=(
            "Optional cross-process cache for deterministic fresh visibility rankings and "
            "proposal outputs. Cache keys include the exact scene poses, target, proposal "
            "configuration, and proposal protocol."
        ),
    )
    parser.add_argument(
        "--proposal-protocol",
        choices=("scene", "stage1_aligned_isolated"),
        default="scene",
        help=(
            "scene regenerates proposals on the current clutter scene only; "
            "stage1_aligned_isolated merges current-scene proposals with "
            "Stage-1-style isolated-target AnyGrasp/SuctionNet proposals."
        ),
    )
    parser.add_argument("--isolated-augmented-parallel-top-k", type=int, default=128)
    parser.add_argument("--isolated-augmented-suction-top-k", type=int, default=128)
    parser.add_argument("--max-isolated-augmented-parallel", type=int, default=64)
    parser.add_argument("--max-isolated-augmented-suction", type=int, default=64)
    parser.add_argument(
        "--isolated-augmented-max-approach-angle-from-down-deg",
        type=float,
        default=75.0,
    )
    parser.add_argument("--isolated-nms-position-thresh", type=float, default=0.010)
    parser.add_argument("--isolated-nms-direction-cos-thresh", type=float, default=0.95)
    parser.add_argument("--isolated-nms-inplane-cos-thresh", type=float, default=0.95)
    parser.add_argument("--scene-config", default="configs/scene_generation_icra2026_mesh_collision_stage4.yaml")
    parser.add_argument("--split-file", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from output-dir/episodes.json, skipping completed "
            "(planner_type, sample_id) pairs and checkpointing after every episode."
        ),
    )
    parser.add_argument("--planner-types", default="budgeted_dependency")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--resettle-steps", type=int, default=1200)
    parser.add_argument(
        "--terminate-on-invalid-resettle",
        action="store_true",
        help=(
            "After a removal, terminate the episode as a failure if the resettle audit is "
            "missing, unstable/unaccepted, or contains any out-of-bin object."
        ),
    )
    parser.add_argument("--no-resettle", action="store_true")
    parser.add_argument("--no-refresh-visibility", action="store_true")
    parser.add_argument(
        "--enable-shared-visibility-fallback",
        action="store_true",
        help=(
            "Before invoking any planner, remove the largest currently visible non-target "
            "object by physical proxy-AABB volume when the target is fully hidden or the "
            "current clutter observation yields no proposals, then reobserve."
        ),
    )
    parser.add_argument(
        "--max-shared-fallback-removals",
        type=int,
        default=None,
        help=(
            "Maximum pre-planner fallback removals per target episode. "
            "Defaults to --max-steps; these actions consume the same total removal budget."
        ),
    )
    parser.add_argument(
        "--shared-fallback-min-visible-pixels",
        type=int,
        default=1,
        help="Visibility threshold for the shared fallback (1 means only a zero-pixel target is hidden).",
    )
    parser.add_argument(
        "--lock-planned-removal-set",
        action="store_true",
        help=(
            "Execute the planner's selected removal sequence as a locked blocker set, "
            "then force a fresh target-grasp attempt before expanding to another set."
        ),
    )
    parser.add_argument(
        "--no-force-target-after-locked-set",
        action="store_true",
        help="When using --lock-planned-removal-set, do not force a target attempt after the locked set is cleared.",
    )
    parser.add_argument(
        "--max-locked-removal-phases",
        type=int,
        default=1,
        help=(
            "Maximum number of locked removal phases when --lock-planned-removal-set is enabled. "
            "Use 2 to allow one recovery phase after a failed forced target attempt."
        ),
    )
    parser.add_argument(
        "--validation-guided-recovery",
        action="store_true",
        help=(
            "After a failed forced target attempt, prepend validator-observed object blockers "
            "to the next locked recovery phase."
        ),
    )
    parser.add_argument(
        "--recover-after-any-target-failure",
        action="store_true",
        help=(
            "When validation-guided recovery is enabled, allow recovery after any failed target attempt "
            "that reports removable object blockers, not only forced target attempts."
        ),
    )
    parser.add_argument(
        "--retry-bin-wall-target-grasps",
        action="store_true",
        help="Retry another target proposal when a target grasp fails only against the bin wall.",
    )
    parser.add_argument(
        "--max-target-grasp-retries",
        type=int,
        default=3,
        help="Maximum target-grasp attempts in one fresh step when --retry-bin-wall-target-grasps is enabled.",
    )
    parser.add_argument(
        "--bin-wall-safe-target-selection",
        action="store_true",
        help=(
            "Before accepting a bin-wall-only target grasp failure, scan a small candidate prefix "
            "and switch only to a feasible target grasp."
        ),
    )
    parser.add_argument(
        "--max-bin-wall-safe-candidates",
        type=int,
        default=3,
        help="Maximum target candidates evaluated by --bin-wall-safe-target-selection.",
    )
    parser.add_argument(
        "--target-feasibility-selection",
        action="store_true",
        help=(
            "Before executing a target grasp, validate an ordered candidate prefix and "
            "switch to a feasible candidate when available."
        ),
    )
    parser.add_argument(
        "--max-target-feasibility-candidates",
        type=int,
        default=8,
        help="Maximum target candidates evaluated by --target-feasibility-selection.",
    )
    parser.add_argument(
        "--no-target-feasibility-prefer-recoverable-failure",
        action="store_true",
        help=(
            "When --target-feasibility-selection finds no feasible candidate, keep the "
            "original failure instead of choosing a candidate with removable object blockers."
        ),
    )
    parser.add_argument(
        "--target-feasibility-recoverable-max-planner-score",
        type=float,
        default=None,
        help=(
            "When target-feasibility fallback finds a recoverable failed candidate, "
            "accept it only if its planner candidate score is <= this threshold. "
            "Lower planner score is better. Omit to preserve ungated behavior."
        ),
    )
    parser.add_argument(
        "--pre-removal-target-feasibility-guard",
        action="store_true",
        help=(
            "Before executing a planned object removal, validate a small target-grasp "
            "candidate prefix and execute the first feasible target grasp instead."
        ),
    )
    parser.add_argument("--calibration-thresholds", default="")
    parser.add_argument("--output-dir", default="fresh_mujoco_closed_loop/outputs/smoke")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _episodes, summary = run_cli(args)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
