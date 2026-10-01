"""Grasp-level visualization status and colors.

These labels describe the current feasibility status of one grasp proposal in
the full scene. They are intentionally separate from object--grasp dependency
labels such as progress-any or sufficient-removal.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping, Sequence

from grasp_dependency_dataset.common.types import FailureStage, ValidationResult


class GraspStatus(str, Enum):
    """Display categories for colors drawn directly on grasp proposals."""

    FEASIBLE = "feasible"
    APPROACH_BLOCKED = "approach-blocked"
    LIFT_BLOCKED = "lift-blocked"
    APPROACH_AND_LIFT_BLOCKED = "approach+lift"
    OTHER_INVALID = "other-invalid"


GRASP_STATUS_COLORS: dict[GraspStatus, tuple[int, int, int]] = {
    GraspStatus.FEASIBLE: (52, 211, 153),  # green
    GraspStatus.APPROACH_BLOCKED: (167, 139, 250),  # purple
    GraspStatus.LIFT_BLOCKED: (96, 165, 250),  # blue
    GraspStatus.APPROACH_AND_LIFT_BLOCKED: (248, 113, 113),  # red
    GraspStatus.OTHER_INVALID: (156, 163, 175),  # gray
}


GRASP_STATUS_LABELS: dict[GraspStatus, str] = {
    GraspStatus.FEASIBLE: "feasible",
    GraspStatus.APPROACH_BLOCKED: "approach-blocked",
    GraspStatus.LIFT_BLOCKED: "lift-blocked",
    GraspStatus.APPROACH_AND_LIFT_BLOCKED: "approach+lift",
    GraspStatus.OTHER_INVALID: "other-invalid",
}


def _has_blockers(
    stage_blockers: Mapping[str, Sequence[str]],
    stage: FailureStage,
) -> bool:
    return bool(stage_blockers.get(stage.value, ()))


def grasp_status_from_validation_result(
    result: ValidationResult,
    *,
    stage_blockers: Mapping[str, Sequence[str]] | None = None,
    promote_terminal_approach_to_both: bool = False,
) -> GraspStatus:
    """Classify one grasp proposal for direct overlay coloring.

    `ValidationResult.failure_stage` is terminal and may be short-circuited.
    When a caller can provide independently computed stage blockers, this
    function can show the combined approach+lift status; otherwise it falls back
    to the stored terminal failure stage.
    """

    blockers = stage_blockers if stage_blockers is not None else result.stage_blockers
    if result.feasible or result.failure_stage == FailureStage.NONE:
        return GraspStatus.FEASIBLE

    approach_blocked = _has_blockers(blockers, FailureStage.APPROACH_COLLISION)
    lift_blocked = _has_blockers(blockers, FailureStage.LIFT_COLLISION)

    if approach_blocked and lift_blocked:
        return GraspStatus.APPROACH_AND_LIFT_BLOCKED
    if promote_terminal_approach_to_both and (
        approach_blocked or result.failure_stage == FailureStage.APPROACH_COLLISION
    ):
        return GraspStatus.APPROACH_AND_LIFT_BLOCKED
    if approach_blocked or result.failure_stage == FailureStage.APPROACH_COLLISION:
        return GraspStatus.APPROACH_BLOCKED
    if lift_blocked or result.failure_stage in {
        FailureStage.LIFT_COLLISION,
        FailureStage.UNSTABLE_LIFT,
    }:
        return GraspStatus.LIFT_BLOCKED
    return GraspStatus.OTHER_INVALID


def color_for_grasp_status(status: GraspStatus | str) -> tuple[int, int, int]:
    """Return the RGB color for a grasp visualization status."""

    status_value = status if isinstance(status, GraspStatus) else GraspStatus(str(status))
    return GRASP_STATUS_COLORS[status_value]


def label_for_grasp_status(status: GraspStatus | str) -> str:
    """Return the short legend label for a grasp visualization status."""

    status_value = status if isinstance(status, GraspStatus) else GraspStatus(str(status))
    return GRASP_STATUS_LABELS[status_value]
