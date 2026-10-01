"""Ordered multi-target retrieval protocol for Fresh MuJoCo scenes."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable

from fresh_mujoco_closed_loop.run_closed_loop import (
    FreshClosedLoopEpisodeRunner,
    FreshLoopConfig,
    PlanFn,
    ProposalTraceFn,
    ProposeFn,
    RefreshFn,
    ResettleFn,
    ValidateFn,
    summarize_proposal_trace,
)
from grasp_dependency_dataset.common.types import Pose, SceneObjectState, StableScene


JsonDict = dict[str, Any]
RelocateFn = Callable[[StableScene, str, str, int], StableScene]


@dataclass(frozen=True)
class OrderedSequenceConfig:
    fresh_loop_config: FreshLoopConfig
    sequence_seed: int = 20260512
    resettle_after_target_retrieval: bool = True


def shuffled_target_order(scene: StableScene, seed: int, scene_index: int) -> list[str]:
    """Return a deterministic random order over every object in the scene."""

    object_ids = [obj.object_id for obj in scene.objects]
    rng = random.Random(int(seed) + int(scene_index) * 1_000_003)
    rng.shuffle(object_ids)
    return object_ids


def _replace_object(scene: StableScene, updated_object: SceneObjectState, metadata: JsonDict) -> StableScene:
    return replace(
        scene,
        objects=tuple(
            updated_object if obj.object_id == updated_object.object_id else obj
            for obj in scene.objects
        ),
        metadata=metadata,
    )


def relocate_object_to_bin_edge(
    scene: StableScene,
    object_id: str,
    active_target_id: str,
    relocation_index: int,
) -> StableScene:
    """Move a non-current object to an in-bin edge pose without removing it."""

    scene_object = scene.get_object(object_id)
    active_target = scene.get_object(active_target_id)
    half_extents = scene_object.scaled_proxy_half_extents
    margin = 0.025
    usable_x = max(0.0, scene.bin_size[0] / 2.0 - scene.wall_thickness - half_extents[0] - margin)
    usable_y = max(0.0, scene.bin_size[1] / 2.0 - scene.wall_thickness - half_extents[1] - margin)
    z = half_extents[2] + 0.004
    candidates = [
        (usable_x, usable_y, z),
        (usable_x, -usable_y, z),
        (-usable_x, usable_y, z),
        (-usable_x, -usable_y, z),
        (usable_x, 0.0, z),
        (-usable_x, 0.0, z),
        (0.0, usable_y, z),
        (0.0, -usable_y, z),
    ]
    target_x, target_y, _target_z = active_target.pose.position
    ranked = sorted(
        candidates,
        key=lambda pos: (pos[0] - target_x) ** 2 + (pos[1] - target_y) ** 2,
        reverse=True,
    )
    chosen = ranked[int(relocation_index) % len(ranked)]
    relocated = replace(
        scene_object,
        pose=Pose(tuple(float(v) for v in chosen), scene_object.pose.quaternion_wxyz),
    )
    metadata = dict(scene.metadata)
    events = list(metadata.get("ordered_sequence_relocations") or [])
    events.append(
        {
            "object_id": object_id,
            "active_target_id": active_target_id,
            "relocation_index": int(relocation_index),
            "from_position": [float(v) for v in scene_object.pose.position],
            "to_position": [float(v) for v in chosen],
        }
    )
    metadata["ordered_sequence_relocations"] = events
    return _replace_object(scene, relocated, metadata)


def summarize_ordered_sequences(rows: list[JsonDict]) -> dict[str, JsonDict]:
    """Aggregate ordered-sequence results by planner type."""

    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["planner_type"])].append(row)

    summary: dict[str, JsonDict] = {}
    for planner_type, planner_rows in sorted(grouped.items()):
        n = len(planner_rows)
        completed = [int(row.get("completed_count", 0)) for row in planner_rows]
        totals = [int(row.get("total_targets", 0)) for row in planner_rows]
        relocations = [int(row.get("num_relocations", 0)) for row in planner_rows]
        fallback_relocations = [
            int(row.get("num_fallback_relocations", 0)) for row in planner_rows
        ]
        planner_relocations = [
            int(row.get("num_planner_relocations", 0)) for row in planner_rows
        ]
        completion_fractions = [
            completed_count / max(total_count, 1)
            for completed_count, total_count in zip(completed, totals, strict=True)
        ]
        terminal_reasons = Counter(str(row.get("terminal_reason", "")) for row in planner_rows)
        total_completed = sum(completed)
        summary[planner_type] = {
            "num_scenes": n,
            "sequence_success_rate": sum(1 for row in planner_rows if row.get("success")) / max(n, 1),
            "mean_completed_targets": sum(completed) / max(n, 1),
            "mean_total_targets": sum(totals) / max(n, 1),
            "mean_completion_fraction": sum(completion_fractions) / max(n, 1),
            "mean_relocations": sum(relocations) / max(n, 1),
            "mean_fallback_relocations": sum(fallback_relocations) / max(n, 1),
            "mean_planner_relocations": sum(planner_relocations) / max(n, 1),
            "relocations_per_retrieved_target": sum(relocations) / max(total_completed, 1),
            "fallback_relocations_per_retrieved_target": (
                sum(fallback_relocations) / max(total_completed, 1)
            ),
            "planner_relocations_per_retrieved_target": (
                sum(planner_relocations) / max(total_completed, 1)
            ),
            "terminal_reason_counts": dict(sorted(terminal_reasons.items())),
        }
    return summary


class OrderedSequenceRunner:
    """Run one ordered retrieval sequence over a mutable Fresh MuJoCo scene."""

    def __init__(
        self,
        *,
        propose: ProposeFn,
        plan: PlanFn,
        validate_target_grasp: ValidateFn,
        relocate_non_current_object: RelocateFn = relocate_object_to_bin_edge,
        resettle_after_intervention: ResettleFn | None = None,
        refresh_observation_metadata: RefreshFn | None = None,
        proposal_trace: ProposalTraceFn | None = None,
    ) -> None:
        self._propose = propose
        self._plan = plan
        self._validate = validate_target_grasp
        self._relocate = relocate_non_current_object
        self._resettle = resettle_after_intervention or (lambda scene, _object_id, _step: scene)
        self._refresh = refresh_observation_metadata or (lambda scene, _target_id: scene)
        self._proposal_trace = proposal_trace or (
            lambda _scene, _target_id, proposals: summarize_proposal_trace(proposals, {})
        )

    def run_sequence(
        self,
        scene: StableScene,
        target_order: list[str],
        config: OrderedSequenceConfig,
    ) -> JsonDict:
        current_scene = replace(scene, target_ids=tuple(target_order))
        retrieved_targets: list[str] = []
        relocation_sequence: list[str] = []
        phase_results: list[JsonDict] = []
        relocation_index = 0
        terminal_reason = "sequence_complete"

        for target_index, target_id in enumerate(target_order):
            current_ids = [obj.object_id for obj in current_scene.objects]
            if target_id not in current_ids:
                terminal_reason = "target_missing_before_phase"
                break

            phase_metadata = dict(current_scene.metadata)
            phase_metadata["ordered_sequence_base_scene_id"] = scene.scene_id
            phase_metadata["ordered_sequence_target_index"] = int(target_index)
            phase_metadata["ordered_sequence_target_id"] = target_id
            phase_scene = replace(
                current_scene,
                scene_id=f"{scene.scene_id}__target_{int(target_index):02d}",
                metadata=phase_metadata,
            )
            latest_scene: dict[str, StableScene] = {"value": phase_scene}

            def apply_relocation(
                step_scene: StableScene,
                active_target_id: str,
                object_id: str,
                step_index: int,
            ) -> StableScene:
                nonlocal relocation_index
                relocated = self._relocate(step_scene, object_id, active_target_id, relocation_index)
                relocation_index += 1
                relocation_sequence.append(object_id)
                if config.fresh_loop_config.resettle_after_removal:
                    relocated = self._resettle(relocated, object_id, step_index)
                latest_scene["value"] = relocated
                return relocated

            episode_runner = FreshClosedLoopEpisodeRunner(
                propose=self._propose,
                plan=self._plan,
                validate_target_grasp=self._validate,
                resettle_after_removal=self._resettle,
                refresh_observation_metadata=self._refresh,
                proposal_trace=self._proposal_trace,
                apply_non_target_intervention=apply_relocation,
            )
            phase_config = replace(
                config.fresh_loop_config,
                planner_type=config.fresh_loop_config.planner_type,
            )
            episode = episode_runner.run_episode(phase_scene, target_id, phase_config)
            phase_scene = latest_scene["value"]
            if episode.get("success"):
                retrieved_targets.append(target_id)
                next_scene = phase_scene.without_objects({target_id})
                if config.resettle_after_target_retrieval:
                    next_scene = self._resettle(next_scene, target_id, target_index)
                current_scene = next_scene
                terminal_reason = "sequence_complete"
            else:
                terminal_reason = str(episode.get("terminal_reason") or "target_phase_failed")

            phase_record = dict(episode)
            phase_record["target_order_index"] = int(target_index)
            phase_record["remaining_object_ids_after_phase"] = [
                obj.object_id for obj in current_scene.objects
            ]
            phase_results.append(phase_record)
            if not episode.get("success"):
                break

        completed_count = len(retrieved_targets)
        total_targets = len(target_order)
        fallback_relocation_sequence = [
            str(object_id)
            for phase in phase_results
            for object_id in phase.get("fallback_removal_sequence", [])
        ]
        planner_relocation_sequence = [
            str(object_id)
            for phase in phase_results
            for object_id in phase.get("planner_removal_sequence", [])
        ]
        return {
            "scene_id": scene.scene_id,
            "planner_type": config.fresh_loop_config.planner_type,
            "target_order": list(target_order),
            "success": completed_count == total_targets,
            "completed_count": completed_count,
            "total_targets": total_targets,
            "completion_fraction": completed_count / max(total_targets, 1),
            "num_relocations": len(relocation_sequence),
            "relocation_sequence": list(relocation_sequence),
            "num_fallback_relocations": len(fallback_relocation_sequence),
            "fallback_relocation_sequence": fallback_relocation_sequence,
            "num_planner_relocations": len(planner_relocation_sequence),
            "planner_relocation_sequence": planner_relocation_sequence,
            "retrieved_target_sequence": list(retrieved_targets),
            "terminal_reason": terminal_reason,
            "target_phase_results": phase_results,
            "remaining_object_ids": [obj.object_id for obj in current_scene.objects],
        }
