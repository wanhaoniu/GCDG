#!/usr/bin/env python3
"""Freeze proposals and labels for a dense all-object benchmark.

This adapts the submitted Stage-3 stored-label protocol to the 100 dense
MuJoCo scenes: proposals, static grasp-feasibility labels, dependency labels,
and minimal blocker labels are generated once from the initial scene and then
consumed without MuJoCo rollouts by the Stage-3 planner.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from fresh_mujoco_closed_loop.proposal_protocol import (
    FreshProposalProtocolConfig,
    generate_stage1_aligned_proposals,
)
from fresh_mujoco_closed_loop.run_closed_loop import load_scene_json, write_json
from grasp_dependency_dataset.hetero_gnn.graph_dataset import discover_sample_refs
from grasp_dependency_dataset.pipeline.runner import DatasetPipelineRunner


JsonDict = dict[str, Any]


def with_visibility_ranking(
    runner: DatasetPipelineRunner,
    scene: Any,
    cache: dict[str, list[JsonDict]],
) -> tuple[Any, str]:
    """Attach an initial-scene visibility ranking when legacy scenes omit it."""

    metadata_ranking = scene.metadata.get("target_selection", {}).get(
        "visibility_ranking", []
    )
    ranking = [dict(item) for item in metadata_ranking if isinstance(item, dict)]
    source = "scene_metadata"
    cache_key = str(scene.scene_id)
    if not ranking:
        if cache_key not in cache:
            cache[cache_key] = [
                dict(item) for item in runner.rank_targets_by_visibility(scene)
            ]
        ranking = cache[cache_key]
        source = "recomputed_from_initial_scene"

    ranked_ids = {
        str(item.get("object_id"))
        for item in ranking
        if item.get("object_id") is not None
    }
    expected_ids = {str(obj.object_id) for obj in scene.objects}
    missing_ids = sorted(expected_ids - ranked_ids)
    if missing_ids:
        raise RuntimeError(
            f"Visibility ranking for {scene.scene_id} is incomplete; "
            f"missing {missing_ids[:8]}"
        )

    metadata = dict(scene.metadata)
    target_selection = dict(metadata.get("target_selection") or {})
    target_selection["visibility_ranking"] = ranking
    target_selection["visibility_ranking_source"] = source
    metadata["target_selection"] = target_selection
    return replace(scene, metadata=metadata), source


def target_visibility(scene: Any, target_id: str) -> tuple[str, int, float]:
    ranking = scene.metadata.get("target_selection", {}).get("visibility_ranking", [])
    info = next(
        (
            item
            for item in ranking
            if isinstance(item, dict) and str(item.get("object_id")) == target_id
        ),
        None,
    )
    if info is None:
        raise RuntimeError(
            f"Target {target_id} is absent from the visibility ranking for {scene.scene_id}"
        )
    visible_pixels = int(info.get("visible_pixels", 0) or 0)
    visible_ratio = float(info.get("visible_ratio", 0.0) or 0.0)
    if visible_pixels <= 0:
        return "fully_hidden", visible_pixels, visible_ratio
    if visible_ratio >= 0.30:
        return "visible", visible_pixels, visible_ratio
    return "partially_visible", visible_pixels, visible_ratio


def is_complete(target_dir: Path) -> bool:
    proposals_path = target_dir / "proposals.json"
    labels_path = target_dir / "labels.json"
    manifest_path = target_dir / "manifest.json"
    if not proposals_path.exists() or not labels_path.exists() or not manifest_path.exists():
        return False
    try:
        proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    # A target can legitimately have no proposals.  In that case the pipeline
    # still writes a complete, auditable record with planning_summary.status
    # == "no_proposals".  Treat it as complete on resume; otherwise dense
    # scenes with many fully occluded targets are regenerated indefinitely.
    if bool(proposals.get("raw_proposals")) and bool(
        labels.get("grasp_feasibility") or labels.get("planning_labels")
    ):
        return True
    summary = labels.get("planning_summary") or {}
    return str(summary.get("status", "")) == "no_proposals"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--proposal-config", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--shard-unit",
        choices=("target", "scene"),
        default="target",
        help=(
            "Assign individual targets (legacy behavior) or whole scenes to shards. "
            "Scene sharding avoids recomputing visibility for the same scene."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the globally ordered target list before sharding (for preflight runs).",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard-index must be in [0, num-shards)")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max-samples must be positive")

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    refs = discover_sample_refs(source_root)
    limited_refs = refs if args.max_samples is None else refs[: int(args.max_samples)]
    if args.shard_unit == "scene":
        scene_order = {
            scene_id: index
            for index, scene_id in enumerate(
                dict.fromkeys(str(ref.scene_id) for ref in limited_refs)
            )
        }
        selected_refs = [
            ref
            for ref in limited_refs
            if scene_order[str(ref.scene_id)] % int(args.num_shards)
            == int(args.shard_index)
        ]
    else:
        selected_refs = [
            ref
            for idx, ref in enumerate(limited_refs)
            if idx % int(args.num_shards) == int(args.shard_index)
        ]
    runner = DatasetPipelineRunner.from_configs(
        config_dir="configs",
        output_root=output_root / "pipeline_cache",
        proposal_config_path=args.proposal_config,
        scene_config_path=args.scene_config,
    )

    proposal_protocol = FreshProposalProtocolConfig(
        mode="stage1_aligned_isolated",
        augmented_parallel_top_k=128,
        augmented_suction_top_k=128,
        max_augmented_parallel=24,
        max_augmented_suction=24,
        augmented_max_approach_angle_from_down_deg=75.0,
        nms_position_thresh=0.010,
        nms_direction_cos_thresh=0.95,
        nms_inplane_cos_thresh=0.95,
    )
    protocol = {
        "protocol_name": "dense100_stage3_stored_label_stage1_aligned_isolated_v2",
        "base_protocol": "Stage-3 stored-label minimal-intervention planning benchmark",
        "dataset_root": str(source_root),
        "num_available_target_trials": len(refs),
        "num_target_trials": len(limited_refs),
        "stored_from_initial_scene": True,
        "mujoco_rollout_during_planning": False,
        "proposal_protocol": "current clutter scene plus Stage-1 isolated-target augmentation",
        "proposal_protocol_config": vars(proposal_protocol),
        "visibility_protocol": (
            "reuse scene metadata when present; otherwise recompute the complete "
            "object ranking once per initial scene with rank_targets_by_visibility"
        ),
        "skip_isolated_augmentation_for_fully_hidden_targets": True,
        "proposal_config": str(Path(args.proposal_config)),
        "scene_config": str(Path(args.scene_config)),
        "shards": int(args.num_shards),
        "shard_unit": str(args.shard_unit),
    }
    write_json(output_root / "stored_label_protocol.json", protocol)
    progress_path = output_root / f"generation_progress_shard_{args.shard_index:02d}.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    visibility_cache: dict[str, list[JsonDict]] = {}

    for local_index, ref in enumerate(selected_refs, start=1):
        target_dir = output_root / "scenes" / ref.scene_id / "targets" / ref.target_id
        if args.resume and is_complete(target_dir):
            print(f"[stored-label] skip {ref.sample_id}", flush=True)
            continue

        scene = load_scene_json(ref.scene_path)
        scene, visibility_source = with_visibility_ranking(
            runner, scene, visibility_cache
        )
        scene_dir = output_root / "scenes" / ref.scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_path = scene_dir / "scene.json"
        if not scene_path.exists():
            # Persist the exact initial-scene metadata used for evaluation,
            # including a recomputed visibility ranking for legacy scenes.
            write_json(scene_path, scene.to_dict())

        visibility_status, visible_pixels, visible_ratio = target_visibility(scene, ref.target_id)
        if visibility_status == "fully_hidden":
            proposals, base_stats = runner.generate_proposals_with_stats(scene, ref.target_id)
            proposal_stats = {
                "source_mode": "stage1_aligned_isolated",
                "target_visibility_status": visibility_status,
                "visible_pixels": visible_pixels,
                "visible_ratio": visible_ratio,
                "isolated_augmentation_skipped": True,
                "isolated_augmentation_skip_reason": "fully_hidden",
                "base_generation_stats": dict(base_stats),
                "base_existing_count": len(proposals),
                "previous_augmented_removed_count": 0,
                "augmented_generated_count": 0,
                "augmented_added_count": 0,
                "augmented_added_parallel_count": 0,
                "augmented_added_suction_count": 0,
                "augmented_skipped_counts": {},
                "final_count": len(proposals),
            }
        else:
            proposals, proposal_stats = generate_stage1_aligned_proposals(
                runner=runner,
                scene=scene,
                target_id=ref.target_id,
                config=proposal_protocol,
            )
            proposal_stats["isolated_augmentation_skipped"] = False
        proposal_stats["visibility_source"] = visibility_source
        results = runner.validator.validate_all(scene, ref.target_id, proposals)
        dependencies = runner.dependency_labeler.label(scene, ref.target_id, proposals, results)
        planning_labels = runner.blocker_solver.solve(
            scene,
            ref.target_id,
            proposals,
            original_results=results,
        )
        planning_summary = runner.blocker_solver.summarize(planning_labels)

        write_json(
            target_dir / "proposals.json",
            runner._proposal_export_payload(ref.target_id, proposals),
        )
        write_json(
            target_dir / "labels.json",
            runner._label_export_payload(
                proposals,
                results,
                dependencies,
                planning_labels,
                planning_summary,
            ),
        )
        manifest = runner._build_manifest(
            scene=scene,
            target_id=ref.target_id,
            proposals=proposals,
            results=results,
            dependencies=dependencies,
            planning_summary=planning_summary,
            scene_path=str(scene_path),
            proposal_path=str(target_dir / "proposals.json"),
            label_path=str(target_dir / "labels.json"),
        )
        manifest["proposal_generation_protocol"] = proposal_stats
        write_json(target_dir / "manifest.json", manifest)
        record = {
            "sample_id": ref.sample_id,
            "num_proposals": len(proposals),
            "num_base_proposals": int(proposal_stats["base_existing_count"]),
            "num_isolated_proposals_generated": int(proposal_stats["augmented_generated_count"]),
            "num_isolated_proposals_added": int(proposal_stats["augmented_added_count"]),
            "num_feasible": sum(int(result.feasible) for result in results.values()),
            "num_dependencies": len(dependencies),
            "num_planning_labels": len(planning_labels),
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"[stored-label] {local_index}/{len(selected_refs)} {ref.sample_id} "
            f"proposals={len(proposals)} isolated_added={record['num_isolated_proposals_added']} "
            f"feasible={record['num_feasible']} "
            f"planning_labels={len(planning_labels)}",
            flush=True,
        )

    print(
        f"[stored-label] shard {args.shard_index} complete: "
        f"{len(selected_refs)} assigned samples",
        flush=True,
    )


if __name__ == "__main__":
    main()
