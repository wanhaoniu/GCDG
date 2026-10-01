"""Non-terminating ordered process protocol for qualitative visualization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable

from fresh_mujoco_closed_loop.ordered_sequence import RelocateFn, relocate_object_to_bin_edge
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
from grasp_dependency_dataset.common.types import StableScene


JsonDict = dict[str, Any]
RecordMainStateFn = Callable[[StableScene, int, str, bool], JsonDict]


@dataclass(frozen=True)
class OrderedProcessVisualizationConfig:
    fresh_loop_config: FreshLoopConfig
    resettle_after_target_retrieval: bool = True
    record_main_state: RecordMainStateFn | None = None


def summarize_ordered_processes(rows: list[JsonDict]) -> dict[str, JsonDict]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["planner_type"])].append(row)

    summary: dict[str, JsonDict] = {}
    for planner_type, planner_rows in sorted(grouped.items()):
        n = len(planner_rows)
        completed = [int(row.get("completed_count", 0)) for row in planner_rows]
        failed = [int(row.get("failed_count", 0)) for row in planner_rows]
        totals = [int(row.get("total_targets", 0)) for row in planner_rows]
        relocations = [int(row.get("num_relocations", 0)) for row in planner_rows]
        total_completed = sum(completed)
        completion_fractions = [
            completed_count / max(total_count, 1)
            for completed_count, total_count in zip(completed, totals, strict=True)
        ]
        summary[planner_type] = {
            "num_scenes": n,
            "mean_completed_targets": sum(completed) / max(n, 1),
            "mean_failed_targets": sum(failed) / max(n, 1),
            "mean_total_targets": sum(totals) / max(n, 1),
            "mean_completion_fraction": sum(completion_fractions) / max(n, 1),
            "mean_relocations": sum(relocations) / max(n, 1),
            "relocations_per_retrieved_target": sum(relocations) / max(total_completed, 1),
        }
    return summary


class OrderedProcessVisualizationRunner:
    """Run all ordered targets for qualitative process visualization."""

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

    def run_process(
        self,
        scene: StableScene,
        target_order: list[str],
        config: OrderedProcessVisualizationConfig,
    ) -> JsonDict:
        current_scene = replace(scene, target_ids=tuple(target_order))
        retrieved_targets: list[str] = []
        failed_targets: list[str] = []
        relocation_sequence: list[str] = []
        target_results: list[JsonDict] = []
        relocation_index = 0

        for target_index, target_id in enumerate(target_order):
            current_ids = [obj.object_id for obj in current_scene.objects]
            if target_id not in current_ids:
                state_record = self._record_main_state(
                    current_scene,
                    target_index,
                    target_id,
                    success=False,
                    config=config,
                )
                target_results.append(
                    {
                        "target_order_index": int(target_index),
                        "target_id": target_id,
                        "success": False,
                        "terminal_reason": "target_missing_before_phase",
                        "num_relocations": 0,
                        "relocation_sequence": [],
                        "failed_target_ids_so_far": list(failed_targets),
                        "remaining_object_ids_after_step": [obj.object_id for obj in current_scene.objects],
                        **state_record,
                    }
                )
                continue

            phase_metadata = dict(current_scene.metadata)
            phase_metadata["ordered_process_base_scene_id"] = scene.scene_id
            phase_metadata["ordered_process_target_index"] = int(target_index)
            phase_metadata["ordered_process_target_id"] = target_id
            phase_scene = replace(
                current_scene,
                scene_id=f"{scene.scene_id}__process_target_{int(target_index):02d}",
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

            before_relocation_count = len(relocation_sequence)
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
            else:
                failed_targets.append(target_id)
                current_scene = phase_scene

            state_record = self._record_main_state(
                current_scene,
                target_index,
                target_id,
                success=bool(episode.get("success")),
                config=config,
            )
            phase_relocations = relocation_sequence[before_relocation_count:]
            phase_record = dict(episode)
            phase_record["target_order_index"] = int(target_index)
            phase_record["remaining_object_ids_after_step"] = [
                obj.object_id for obj in current_scene.objects
            ]
            phase_record["failed_target_ids_so_far"] = list(failed_targets)
            phase_record["num_relocations"] = len(phase_relocations)
            phase_record["relocation_sequence"] = list(phase_relocations)
            phase_record.update(state_record)
            target_results.append(phase_record)

        total_targets = len(target_order)
        return {
            "scene_id": scene.scene_id,
            "planner_type": config.fresh_loop_config.planner_type,
            "target_order": list(target_order),
            "success": len(retrieved_targets) == total_targets,
            "completed_count": len(retrieved_targets),
            "failed_count": len(failed_targets),
            "total_targets": total_targets,
            "completion_fraction": len(retrieved_targets) / max(total_targets, 1),
            "num_relocations": len(relocation_sequence),
            "relocation_sequence": list(relocation_sequence),
            "retrieved_target_sequence": list(retrieved_targets),
            "failed_target_ids": list(failed_targets),
            "target_results": target_results,
            "main_state_count": len(target_results),
            "remaining_object_ids": [obj.object_id for obj in current_scene.objects],
        }

    def _record_main_state(
        self,
        scene: StableScene,
        target_index: int,
        target_id: str,
        *,
        success: bool,
        config: OrderedProcessVisualizationConfig,
    ) -> JsonDict:
        metadata = dict(scene.metadata)
        base_scene_id = str(
            metadata.get("ordered_process_base_scene_id")
            or metadata.get("ordered_sequence_base_scene_id")
            or scene.scene_id.split("__process_target_", 1)[0]
        )
        process_scene_id = f"{base_scene_id}__process_after_target_{int(target_index):02d}"
        metadata["ordered_process_main_state"] = {
            "base_scene_id": base_scene_id,
            "target_index": int(target_index),
            "target_id": target_id,
            "target_success": bool(success),
        }
        state_scene = replace(scene, scene_id=process_scene_id, metadata=metadata)
        if config.record_main_state is not None:
            record = config.record_main_state(state_scene, int(target_index), target_id, bool(success))
            return dict(record)
        return {
            "scene_id_after_target": state_scene.scene_id,
            "scene_path_after_target": "",
        }
