"""Minimal blocker set solver."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from grasp_dependency_dataset.common.types import (
    FailureStage,
    GraspProposal,
    PlanningLabel,
    PlanningTerminalReason,
    PlanningStatus,
    PlanningSummary,
    StableScene,
    ValidationResult,
)
from grasp_dependency_dataset.validation.validator import StagedGraspValidator

_BIN_WALL_BLOCKER_ID = "__bin_wall__"


@dataclass
class MinimalBlockerSolver:
    """Find bounded minimal blocker sets for fixed grasp proposals."""

    validator: StagedGraspValidator
    max_search_depth: int

    def solve(
        self,
        scene: StableScene,
        target_id: str,
        proposals: list[GraspProposal],
        original_results: dict[str, ValidationResult] | None = None,
    ) -> list[PlanningLabel]:
        """Return one planning label for every grasp proposal."""

        ordered_proposals = sorted(proposals, key=lambda proposal: proposal.proposal_score, reverse=True)
        if not ordered_proposals:
            return []

        original_results = original_results or {
            proposal.grasp_id: self.validator.validate(scene=scene, target_id=target_id, proposal=proposal)
            for proposal in ordered_proposals
        }
        scene_cache: dict[frozenset[str], StableScene] = {frozenset(): scene}
        validation_cache: dict[tuple[frozenset[str], str], ValidationResult] = {
            (frozenset(), proposal.grasp_id): original_results[proposal.grasp_id]
            for proposal in ordered_proposals
        }

        return [
            self._solve_for_proposal(
                scene=scene,
                target_id=target_id,
                proposal=proposal,
                original_result=original_results[proposal.grasp_id],
                scene_cache=scene_cache,
                validation_cache=validation_cache,
            )
            for proposal in ordered_proposals
        ]

    def summarize(self, labels: list[PlanningLabel]) -> PlanningSummary:
        """Collapse per-grasp planning labels into one target-level summary."""

        if not labels:
            return PlanningSummary(
                best_grasp_id=None,
                minimal_blocker_set=tuple(),
                oracle_removal_sequence=tuple(),
                status=PlanningStatus.NO_PROPOSALS,
                status_counts={PlanningStatus.NO_PROPOSALS.value: 1},
                terminal_reason=PlanningTerminalReason.NO_BLOCKER_ATTRIBUTION,
                terminal_reason_counts={PlanningTerminalReason.NO_BLOCKER_ATTRIBUTION.value: 1},
            )

        status_counts = Counter(label.status.value for label in labels)
        reason_counts = Counter(label.terminal_reason.value for label in labels)
        best_label = min(labels, key=self._summary_sort_key)
        return PlanningSummary(
            best_grasp_id=best_label.grasp_id,
            minimal_blocker_set=best_label.minimal_blocker_set,
            oracle_removal_sequence=best_label.oracle_removal_sequence,
            status=best_label.status,
            status_counts=dict(sorted(status_counts.items())),
            terminal_reason=best_label.terminal_reason,
            terminal_reason_counts=dict(sorted(reason_counts.items())),
        )

    def _solve_for_proposal(
        self,
        scene: StableScene,
        target_id: str,
        proposal: GraspProposal,
        original_result: ValidationResult,
        scene_cache: dict[frozenset[str], StableScene],
        validation_cache: dict[tuple[frozenset[str], str], ValidationResult],
    ) -> PlanningLabel:
        """Search bounded blocker subsets for one fixed proposal."""

        non_target_ids = scene.non_target_ids(target_id)
        if original_result.feasible:
            return PlanningLabel(
                grasp_id=proposal.grasp_id,
                minimal_blocker_set=tuple(),
                oracle_removal_sequence=tuple(),
                status=PlanningStatus.ALREADY_FEASIBLE,
                terminal_reason=PlanningTerminalReason.FEASIBLE,
            )
        if _object_removal_cannot_help(original_result):
            return PlanningLabel(
                grasp_id=proposal.grasp_id,
                minimal_blocker_set=tuple(),
                oracle_removal_sequence=tuple(),
                status=PlanningStatus.UNSOLVED_WITHIN_DEPTH,
                terminal_reason=self._terminal_reason(original_result),
                terminal_failure_stage=original_result.failure_stage,
                terminal_failure_blockers=tuple(
                    original_result.stage_blockers.get(original_result.failure_stage.value, [])
                ),
                terminal_failure_notes=original_result.notes,
            )

        max_depth = min(self.max_search_depth, len(non_target_ids))
        best_result = original_result
        best_key = self._terminal_result_sort_key(original_result)
        ordered_non_target_ids = tuple(non_target_ids)

        def _search_exact_depth(
            *,
            start_index: int,
            removed: frozenset[str],
            current_result: ValidationResult,
            remaining_to_add: int,
        ) -> frozenset[str] | None:
            nonlocal best_result, best_key

            if removed:
                result_key = self._terminal_result_sort_key(current_result)
                if result_key > best_key:
                    best_key = result_key
                    best_result = current_result

            if current_result.feasible:
                return removed
            if remaining_to_add <= 0 or _object_removal_cannot_help(current_result):
                return None

            for idx in range(start_index, len(ordered_non_target_ids)):
                object_id = ordered_non_target_ids[idx]
                candidate_removed = frozenset(set(removed) | {object_id})
                candidate_scene = scene_cache.get(candidate_removed)
                if candidate_scene is None:
                    candidate_scene = scene.without_objects(set(candidate_removed))
                    scene_cache[candidate_removed] = candidate_scene
                result = self._validate_cached(
                    scene=candidate_scene,
                    target_id=target_id,
                    proposal=proposal,
                    original_result=original_result,
                    removed=candidate_removed,
                    validation_cache=validation_cache,
                )
                solution = _search_exact_depth(
                    start_index=idx + 1,
                    removed=candidate_removed,
                    current_result=result,
                    remaining_to_add=remaining_to_add - 1,
                )
                if solution is not None:
                    return solution
            return None

        for subset_size in range(1, max_depth + 1):
            solved_removed = _search_exact_depth(
                start_index=0,
                removed=frozenset(),
                current_result=original_result,
                remaining_to_add=subset_size,
            )
            if solved_removed is not None:
                ordered_solution = tuple(
                    object_id for object_id in ordered_non_target_ids if object_id in solved_removed
                )
                oracle_sequence = self._derive_oracle_sequence(
                    scene=scene,
                    target_id=target_id,
                    proposal=proposal,
                    original_result=original_result,
                    blocker_set=set(ordered_solution),
                    scene_cache=scene_cache,
                    validation_cache=validation_cache,
                )
                return PlanningLabel(
                    grasp_id=proposal.grasp_id,
                    minimal_blocker_set=ordered_solution,
                    oracle_removal_sequence=tuple(oracle_sequence),
                    status=PlanningStatus.SOLVED_WITHIN_DEPTH,
                    terminal_reason=PlanningTerminalReason.FEASIBLE,
                )

        return PlanningLabel(
            grasp_id=proposal.grasp_id,
            minimal_blocker_set=tuple(),
            oracle_removal_sequence=tuple(),
            status=PlanningStatus.UNSOLVED_WITHIN_DEPTH,
            terminal_reason=self._terminal_reason(best_result),
            terminal_failure_stage=best_result.failure_stage,
            terminal_failure_blockers=tuple(
                best_result.stage_blockers.get(best_result.failure_stage.value, [])
            ),
            terminal_failure_notes=best_result.notes,
        )

    def _derive_oracle_sequence(
        self,
        scene: StableScene,
        target_id: str,
        proposal: GraspProposal,
        original_result: ValidationResult,
        blocker_set: set[str],
        scene_cache: dict[frozenset[str], StableScene],
        validation_cache: dict[tuple[frozenset[str], str], ValidationResult],
    ) -> list[str]:
        """Order a blocker set greedily by stage-progress improvement for one grasp."""

        if not blocker_set:
            return []

        sequence: list[str] = []
        removed: set[str] = set()
        remaining = set(blocker_set)

        while remaining:
            best_object = None
            best_key = (-1, "")
            for object_id in sorted(remaining):
                candidate_removed = removed | {object_id}
                candidate_removed_fs = frozenset(candidate_removed)
                candidate_scene = scene_cache.get(candidate_removed_fs)
                if candidate_scene is None:
                    candidate_scene = scene.without_objects(candidate_removed)
                    scene_cache[candidate_removed_fs] = candidate_scene
                result = self._validate_cached(
                    scene=candidate_scene,
                    target_id=target_id,
                    proposal=proposal,
                    original_result=original_result,
                    removed=candidate_removed_fs,
                    validation_cache=validation_cache,
                )
                key = (result.stage_rank, f"{object_id}")
                if key > best_key:
                    best_key = key
                    best_object = object_id

            assert best_object is not None
            sequence.append(best_object)
            removed.add(best_object)
            remaining.remove(best_object)

        return sequence

    def _validate_cached(
        self,
        scene: StableScene,
        target_id: str,
        proposal: GraspProposal,
        original_result: ValidationResult,
        removed: frozenset[str],
        validation_cache: dict[tuple[frozenset[str], str], ValidationResult],
    ) -> ValidationResult:
        """Reuse exact revalidation results across subset searches."""

        cache_key = (removed, proposal.grasp_id)
        cached = validation_cache.get(cache_key)
        if cached is not None:
            return cached

        if not removed:
            validation_cache[cache_key] = original_result
            return original_result

        result = self.validator.validate(scene=scene, target_id=target_id, proposal=proposal)
        validation_cache[cache_key] = result
        return result

    @staticmethod
    def _summary_sort_key(label: PlanningLabel) -> tuple[int, int, str]:
        """Rank per-grasp planning labels from easiest to hardest."""

        status_priority = {
            PlanningStatus.ALREADY_FEASIBLE: 0,
            PlanningStatus.SOLVED_WITHIN_DEPTH: 1,
            PlanningStatus.UNSOLVED_WITHIN_DEPTH: 2,
            PlanningStatus.NO_PROPOSALS: 3,
        }
        return (
            status_priority[label.status],
            label.minimal_blocker_set_size,
            label.grasp_id,
        )

    @staticmethod
    def _terminal_reason(result: ValidationResult) -> PlanningTerminalReason:
        """Classify the remaining failure cause after bounded search."""

        if result.feasible or result.failure_stage == FailureStage.NONE:
            return PlanningTerminalReason.FEASIBLE

        blockers = result.stage_blockers.get(result.failure_stage.value, [])
        has_bin_wall = _BIN_WALL_BLOCKER_ID in blockers
        has_object_blockers = any(blocker != _BIN_WALL_BLOCKER_ID for blocker in blockers)
        if has_bin_wall and not has_object_blockers:
            return PlanningTerminalReason.BIN_WALL_ONLY
        if has_bin_wall and has_object_blockers:
            return PlanningTerminalReason.BIN_WALL_AND_CLUTTER
        if has_object_blockers:
            return PlanningTerminalReason.CLUTTER_ONLY_OR_DEPTH_LIMIT
        return PlanningTerminalReason.NO_BLOCKER_ATTRIBUTION

    @classmethod
    def _terminal_result_sort_key(cls, result: ValidationResult) -> tuple[int, int, int, int]:
        """Prefer the least blocked non-feasible result reached within the depth bound."""

        blockers = result.stage_blockers.get(result.failure_stage.value, [])
        object_blocker_count = sum(1 for blocker in blockers if blocker != _BIN_WALL_BLOCKER_ID)
        has_bin_wall = int(_BIN_WALL_BLOCKER_ID in blockers)
        return (
            result.stage_rank,
            -object_blocker_count,
            -has_bin_wall,
            -len(blockers),
        )


def _object_removal_cannot_help(result: ValidationResult) -> bool:
    """Return whether removing scene objects cannot improve this proposal result."""

    blockers = tuple(result.stage_blockers.get(result.failure_stage.value, []))
    if _BIN_WALL_BLOCKER_ID in blockers:
        return True
    if result.failure_stage == FailureStage.UNREACHABLE and not blockers:
        return True
    if result.failure_stage == FailureStage.CLOSING_OR_SEAL_FAILURE and not blockers:
        return True
    return False
