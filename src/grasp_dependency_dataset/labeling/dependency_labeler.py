"""Single-object removal dependency labeling."""

from __future__ import annotations

from dataclasses import dataclass
import math

from grasp_dependency_dataset.common.types import DependencyLabel, FailureStage, GraspProposal, StableScene, ValidationResult
from grasp_dependency_dataset.validation.validator import StagedGraspValidator

_BIN_WALL_BLOCKER_ID = "__bin_wall__"


@dataclass
class DependencyLabeler:
    """Generate grasp-conditioned dependency labels by single-object removal."""

    validator: StagedGraspValidator

    def label(
        self,
        scene: StableScene,
        target_id: str,
        proposals: list[GraspProposal],
        original_results: dict[str, ValidationResult],
    ) -> list[DependencyLabel]:
        """Generate exhaustive single-removal labels for every non-target object and proposal."""

        labels: list[DependencyLabel] = []
        scene_cache: dict[frozenset[str], StableScene] = {frozenset(): scene}
        validation_cache: dict[tuple[frozenset[str], str], ValidationResult] = {}
        non_target_ids = scene.non_target_ids(target_id)
        non_target_objects = {object_id: scene.get_object(object_id) for object_id in non_target_ids}
        for proposal in proposals:
            original = original_results[proposal.grasp_id]
            original_is_unremovable = _single_object_removal_cannot_help(original)
            for object_id in non_target_ids:
                obstacle = non_target_objects[object_id]
                if original.feasible or original_is_unremovable:
                    modified = original
                else:
                    removed = frozenset({object_id})
                    cache_key = (removed, proposal.grasp_id)
                    modified = validation_cache.get(cache_key)
                    if modified is None:
                        modified_scene = scene_cache.get(removed)
                        if modified_scene is None:
                            modified_scene = scene.without_objects(set(removed))
                            scene_cache[removed] = modified_scene
                        modified = self.validator.validate(
                            scene=modified_scene,
                            target_id=target_id,
                            proposal=proposal,
                        )
                        validation_cache[cache_key] = modified
                labels.append(
                    DependencyLabel(
                        object_id=object_id,
                        grasp_id=proposal.grasp_id,
                        dep_any=(not original.feasible and modified.feasible),
                        dep_collision_approach=(
                            original.failure_stage == FailureStage.APPROACH_COLLISION
                            and modified.stage_rank > original.stage_rank
                        ),
                        dep_collision_lift=(
                            original.failure_stage in {FailureStage.LIFT_COLLISION, FailureStage.UNSTABLE_LIFT}
                            and modified.stage_rank > original.stage_rank
                        ),
                        metadata={
                            "original_failure_stage": original.failure_stage.value,
                            "modified_failure_stage": modified.failure_stage.value,
                            "original_feasible": original.feasible,
                            "modified_feasible": modified.feasible,
                            "restored_if_removed": bool(modified.feasible),
                            "stage_progress_delta": int(modified.stage_rank - original.stage_rank),
                            "original_failure_blockers": original.stage_blockers.get(
                                original.failure_stage.value,
                                [],
                            ),
                            "modified_failure_blockers": modified.stage_blockers.get(
                                modified.failure_stage.value,
                                [],
                            ),
                            "relative_object_grasp": _relative_object_grasp_payload(
                                object_position=obstacle.pose.position,
                                grasp_position=proposal.pose.position,
                                approach_vector=proposal.approach_vector,
                                lift_vector=proposal.lift_vector,
                            ),
                        },
                    )
                )
        return labels


def _relative_object_grasp_payload(
    *,
    object_position: tuple[float, float, float],
    grasp_position: tuple[float, float, float],
    approach_vector: tuple[float, float, float],
    lift_vector: tuple[float, float, float],
) -> dict[str, object]:
    """Describe the relative pose between one clutter object and one target grasp."""

    relative = tuple(
        float(grasp_position[index]) - float(object_position[index])
        for index in range(3)
    )
    distance = math.sqrt(sum(component * component for component in relative))
    approach_dir = _normalize(approach_vector)
    lift_dir = _normalize(lift_vector)
    if distance < 1e-12:
        object_to_grasp_dir = (0.0, 0.0, 0.0)
    else:
        object_to_grasp_dir = tuple(component / distance for component in relative)

    return {
        "object_position": [float(value) for value in object_position],
        "grasp_position": [float(value) for value in grasp_position],
        "vector_object_to_grasp": [float(value) for value in relative],
        "distance": float(distance),
        "direction_object_to_grasp": [float(value) for value in object_to_grasp_dir],
        "approach_alignment": float(
            sum(object_to_grasp_dir[index] * approach_dir[index] for index in range(3))
        ),
        "lift_alignment": float(
            sum(object_to_grasp_dir[index] * lift_dir[index] for index in range(3))
        ),
    }


def _normalize(values: tuple[float, float, float]) -> tuple[float, float, float]:
    """Normalize a 3D vector without introducing a numpy dependency."""

    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    if norm < 1e-12:
        return (0.0, 0.0, 0.0)
    return tuple(float(value) / norm for value in values)


def _single_object_removal_cannot_help(result: ValidationResult) -> bool:
    """Return whether removing clutter objects cannot improve this proposal result."""

    blockers = tuple(result.stage_blockers.get(result.failure_stage.value, []))
    if _BIN_WALL_BLOCKER_ID in blockers:
        return True
    if result.failure_stage == FailureStage.UNREACHABLE and not blockers:
        return True
    if result.failure_stage == FailureStage.CLOSING_OR_SEAL_FAILURE and not blockers:
        return True
    return False
