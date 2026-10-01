"""Stage-1-aligned proposal generation for fresh closed-loop evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Iterator

from grasp_dependency_dataset.common.types import (
    GraspProposal,
    GraspType,
    StableScene,
)
from grasp_dependency_dataset.pipeline.runner import DatasetPipelineRunner


_AUGMENTED_SOURCE_BY_TYPE = {
    GraspType.PARALLEL_JAW: "isolated_target_anygrasp",
    GraspType.SUCTION: "isolated_target_suctionnet",
}


@dataclass(frozen=True)
class FreshProposalProtocolConfig:
    """Proposal protocol settings used by the fresh closed-loop runtime."""

    mode: str = "scene"
    include_base_scene_proposals: bool = True
    augmented_parallel_top_k: int = 128
    augmented_suction_top_k: int = 128
    max_augmented_parallel: int = 64
    max_augmented_suction: int = 64
    augmented_max_approach_angle_from_down_deg: float | None = 75.0
    nms_position_thresh: float = 0.010
    nms_direction_cos_thresh: float = 0.95
    nms_inplane_cos_thresh: float = 0.95


def build_isolated_target_scene(scene: StableScene, target_id: str) -> StableScene:
    """Return a target-only scene matching Stage-1 isolated augmentation."""

    target = scene.get_object(target_id)
    return StableScene(
        scene_id=f"{scene.scene_id}__isolated__{target_id}",
        objects=(target,),
        target_ids=(target_id,),
        bin_size=scene.bin_size,
        wall_thickness=scene.wall_thickness,
        metadata={
            **dict(scene.metadata),
            "proposal_augmentation_source_scene_id": scene.scene_id,
            "proposal_augmentation_mode": "oracle_isolated",
        },
    )


def _visibility_lookup(scene: StableScene) -> dict[str, dict[str, Any]]:
    ranking = scene.metadata.get("target_selection", {}).get("visibility_ranking", [])
    return {
        str(item.get("object_id")): dict(item)
        for item in ranking
        if isinstance(item, dict) and item.get("object_id") is not None
    }


def _visibility_status(info: dict[str, Any]) -> str:
    visible_pixels = int(info.get("visible_pixels", 0) or 0)
    visible_ratio = float(info.get("visible_ratio", 0.0) or 0.0)
    if visible_pixels <= 0:
        return "fully_hidden"
    if visible_ratio >= 0.30:
        return "visible"
    return "partially_visible"


def annotate_isolated_proposals(
    proposals: list[GraspProposal],
    *,
    target_id: str,
    visibility_status: str,
    visible_info: dict[str, Any],
) -> list[GraspProposal]:
    """Rename and tag isolated-target proposals the same way Stage 1 does."""

    annotated: list[GraspProposal] = []
    by_type_counter: Counter[GraspType] = Counter()
    for proposal in sorted(proposals, key=lambda item: float(item.proposal_score), reverse=True):
        source = _AUGMENTED_SOURCE_BY_TYPE[proposal.grasp_type]
        type_index = by_type_counter[proposal.grasp_type]
        by_type_counter[proposal.grasp_type] += 1
        type_token = "parallel" if proposal.grasp_type == GraspType.PARALLEL_JAW else "suction"
        metadata = dict(proposal.metadata or {})
        metadata.update(
            {
                "is_augmented_proposal": True,
                "completion_mode": "oracle_isolated",
                "augmentation_source_mode": "oracle_isolated",
                "target_visibility_status": visibility_status,
                "base_proposal_source": proposal.source,
                "augmented_from_grasp_id": proposal.grasp_id,
                "target_visible_pixels": int(visible_info.get("visible_pixels", 0) or 0),
                "target_visible_ratio": float(visible_info.get("visible_ratio", 0.0) or 0.0),
            }
        )
        annotated.append(
            replace(
                proposal,
                grasp_id=f"{target_id}_{type_token}_isolated_aug_{type_index:03d}",
                source=source,
                metadata=metadata,
            )
        )
    return annotated


def _is_augmented(proposal: GraspProposal) -> bool:
    metadata = proposal.metadata or {}
    if bool(metadata.get("is_augmented_proposal", False)):
        return True
    return str(proposal.source).startswith("isolated_target_")


def merge_stage1_augmented_proposals(
    existing: list[GraspProposal],
    augmented: list[GraspProposal],
    config: FreshProposalProtocolConfig,
) -> tuple[list[GraspProposal], list[GraspProposal], dict[str, Any]]:
    """Merge isolated proposals into base proposals using Stage-1 NMS rules."""

    base = [proposal for proposal in existing if not _is_augmented(proposal)]
    previous_augmented = [proposal for proposal in existing if _is_augmented(proposal)]
    kept = list(base)
    added: list[GraspProposal] = []
    skipped_by_type: Counter[str] = Counter()
    added_by_type: Counter[GraspType] = Counter()
    max_by_type = {
        GraspType.PARALLEL_JAW: max(0, int(config.max_augmented_parallel)),
        GraspType.SUCTION: max(0, int(config.max_augmented_suction)),
    }

    for proposal in sorted(augmented, key=lambda item: float(item.proposal_score), reverse=True):
        if added_by_type[proposal.grasp_type] >= max_by_type[proposal.grasp_type]:
            skipped_by_type[f"{proposal.grasp_type.value}_cap"] += 1
            continue
        duplicate = any(
            existing_proposal.grasp_type == proposal.grasp_type
            and DatasetPipelineRunner._are_similar_proposals(
                proposal,
                existing_proposal,
                grasp_type=proposal.grasp_type,
                distance_threshold=float(config.nms_position_thresh),
                approach_cos_threshold=float(config.nms_direction_cos_thresh),
                inplane_cos_threshold=float(config.nms_inplane_cos_thresh),
            )
            for existing_proposal in kept
        )
        if duplicate:
            skipped_by_type[f"{proposal.grasp_type.value}_nms"] += 1
            continue
        kept.append(proposal)
        added.append(proposal)
        added_by_type[proposal.grasp_type] += 1

    stats = {
        "base_existing_count": len(base),
        "previous_augmented_removed_count": len(previous_augmented),
        "augmented_generated_count": len(augmented),
        "augmented_added_count": len(added),
        "augmented_added_parallel_count": added_by_type[GraspType.PARALLEL_JAW],
        "augmented_added_suction_count": added_by_type[GraspType.SUCTION],
        "augmented_skipped_counts": dict(sorted(skipped_by_type.items())),
        "final_count": len(kept),
    }
    return kept, added, stats


def _clear_provider_state(provider: Any) -> None:
    for cache_name in ("_renderer_pool", "_observation_cache", "_generation_observation_cache"):
        cache = getattr(provider, cache_name, None)
        if hasattr(cache, "clear"):
            cache.clear()


def _apply_provider_config(provider: Any, config: Any) -> None:
    provider.config = config
    if hasattr(provider, "_renderer"):
        from grasp_dependency_dataset.observation.renderer import (
            MujocoSceneObservationRenderer,
            RenderCameraSpec,
        )

        provider._renderer = MujocoSceneObservationRenderer(
            RenderCameraSpec.from_proposal_config(config)
        )
    _clear_provider_state(provider)


@contextmanager
def isolated_generation_context(
    runner: DatasetPipelineRunner,
    config: FreshProposalProtocolConfig,
) -> Iterator[None]:
    """Temporarily configure providers for Stage-1 isolated-target generation."""

    max_keep = max(int(config.max_augmented_parallel), int(config.max_augmented_suction), 1)
    providers = (runner.parallel_provider, runner.suction_provider)
    original_configs = [provider.config for provider in providers]
    try:
        for provider in providers:
            isolated_config = replace(
                provider.config,
                parallel_top_k=int(config.augmented_parallel_top_k),
                suction_top_k=int(config.augmented_suction_top_k),
                max_keep_per_type=max_keep,
                parallel_scene_generation_enabled=False,
                parallel_target_only_generation_enabled=True,
                parallel_multiview_enabled=True,
                parallel_target_support_fill_enabled=True,
            )
            _apply_provider_config(provider, isolated_config)
        yield
    finally:
        for provider, original_config in zip(providers, original_configs):
            _apply_provider_config(provider, original_config)


def generate_stage1_aligned_proposals(
    *,
    runner: DatasetPipelineRunner,
    scene: StableScene,
    target_id: str,
    config: FreshProposalProtocolConfig,
) -> tuple[list[GraspProposal], dict[str, Any]]:
    """Generate base plus isolated-target proposals using the Stage-1 protocol."""

    base: list[GraspProposal] = []
    base_stats: dict[str, Any] = {}
    if config.include_base_scene_proposals:
        base, base_stats = runner.generate_proposals_with_stats(scene, target_id)

    isolated = build_isolated_target_scene(scene, target_id)
    raw_parallel: list[GraspProposal] = []
    raw_suction: list[GraspProposal] = []
    with isolated_generation_context(runner, config):
        if int(config.augmented_parallel_top_k) > 0:
            raw_parallel = runner.parallel_provider.generate(
                scene=isolated,
                target_id=target_id,
                top_k=int(config.augmented_parallel_top_k),
            )
        if int(config.augmented_suction_top_k) > 0:
            raw_suction = runner.suction_provider.generate(
                scene=isolated,
                target_id=target_id,
                top_k=int(config.augmented_suction_top_k),
            )

    max_angle = (
        float(config.augmented_max_approach_angle_from_down_deg)
        if config.augmented_max_approach_angle_from_down_deg is not None
        else float(runner.parallel_provider.config.max_approach_angle_from_down_deg)
    )
    parallel = DatasetPipelineRunner._filter_proposals_by_approach_angle(
        raw_parallel,
        max_angle_from_down_deg=max_angle,
    )
    suction = DatasetPipelineRunner._filter_proposals_by_approach_angle(
        raw_suction,
        max_angle_from_down_deg=max_angle,
    )

    visible_info = _visibility_lookup(scene).get(target_id, {})
    visibility_status = _visibility_status(visible_info)
    annotated = annotate_isolated_proposals(
        parallel + suction,
        target_id=target_id,
        visibility_status=visibility_status,
        visible_info=visible_info,
    )
    proposals, _added, merge_stats = merge_stage1_augmented_proposals(base, annotated, config)
    stats = {
        "source_mode": "stage1_aligned_isolated",
        "isolated_scene_id": isolated.scene_id,
        "target_visibility_status": visibility_status,
        "visible_pixels": int(visible_info.get("visible_pixels", 0) or 0),
        "visible_ratio": float(visible_info.get("visible_ratio", 0.0) or 0.0),
        "base_generation_stats": dict(base_stats),
        "parallel_raw_count": len(raw_parallel),
        "suction_raw_count": len(raw_suction),
        "parallel_after_approach_angle_filter_count": len(parallel),
        "suction_after_approach_angle_filter_count": len(suction),
        "total_after_approach_angle_filter_count": len(annotated),
        "augmented_max_approach_angle_from_down_deg": float(max_angle),
        **merge_stats,
    }
    return proposals, stats
