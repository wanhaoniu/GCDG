"""Run Fresh MuJoCo ordered multi-target retrieval episodes."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fresh_mujoco_closed_loop.ordered_sequence import (  # noqa: E402
    OrderedSequenceConfig,
    OrderedSequenceRunner,
    shuffled_target_order,
    summarize_ordered_sequences,
)
from fresh_mujoco_closed_loop.proposal_protocol import FreshProposalProtocolConfig  # noqa: E402
from fresh_mujoco_closed_loop.run_closed_loop import (  # noqa: E402
    CachedDependencyMapPredictor,
    FreshLoopConfig,
    RealFreshMujocoRuntime,
    apply_calibration_to_planner_cfg,
    checkpoint_class_map,
    load_config,
    load_yaml,
    resolve_calibration_thresholds,
    write_json,
)
from grasp_dependency_dataset.common.types import StableScene  # noqa: E402
from grasp_dependency_dataset.hetero_gnn.eval_hetero_gnn import choose_backend  # noqa: E402
from grasp_dependency_dataset.hetero_gnn.train_hetero_gnn import resolve_torch_device  # noqa: E402
from grasp_dependency_dataset.pipeline.runner import DatasetPipelineRunner  # noqa: E402
from planners.planner_utils import PlannerParams  # noqa: E402


def write_ordered_summary_csv(path: Path, summary: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "planner_type",
        "num_scenes",
        "sequence_success_rate",
        "mean_completion_fraction",
        "mean_completed_targets",
        "mean_total_targets",
        "mean_relocations",
        "mean_fallback_relocations",
        "mean_planner_relocations",
        "relocations_per_retrieved_target",
        "fallback_relocations_per_retrieved_target",
        "planner_relocations_per_retrieved_target",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for planner_type, row in sorted(summary.items()):
            payload = {"planner_type": planner_type}
            payload.update({key: row[key] for key in fieldnames if key != "planner_type"})
            writer.writerow(payload)


def generate_stable_ordered_scenes(
    runner: DatasetPipelineRunner,
    *,
    scene_index_start: int,
    num_scenes: int,
    sequence_seed: int,
    max_scene_index_scan: int | None = None,
) -> list[tuple[int, StableScene, list[str]]]:
    scenes: list[tuple[int, StableScene, list[str]]] = []
    scan_limit = int(max_scene_index_scan) if max_scene_index_scan is not None else max(int(num_scenes) * 20, 50)
    candidate_offset = 0
    while len(scenes) < int(num_scenes) and candidate_offset < scan_limit:
        scene_index = int(scene_index_start) + candidate_offset
        candidate_offset += 1
        scene: StableScene | None = None
        for attempt in range(1, int(runner.max_scene_generation_attempts) + 1):
            candidate = runner.scene_generator.generate_scene(scene_index)
            if not bool(candidate.metadata.get("stable", False)):
                continue
            metadata = dict(candidate.metadata)
            metadata["scene_generation_attempt"] = attempt
            all_object_ids = tuple(obj.object_id for obj in candidate.objects)
            scene = replace(candidate, target_ids=all_object_ids, metadata=metadata)
            break
        if scene is None:
            continue
        target_order = shuffled_target_order(scene, seed=sequence_seed, scene_index=len(scenes))
        scenes.append((scene_index, scene, target_order))
    if len(scenes) < int(num_scenes):
        raise RuntimeError(
            f"Could only generate {len(scenes)} stable ordered scenes out of {num_scenes} "
            f"after scanning {scan_limit} scene indices from {scene_index_start}."
        )
    return scenes


def build_loop_config(args: argparse.Namespace, planner_type: str) -> FreshLoopConfig:
    return FreshLoopConfig(
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
    )


def run_cli(args: argparse.Namespace) -> tuple[list[dict], dict[str, dict]]:
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
    scenes = generate_stable_ordered_scenes(
        runner,
        scene_index_start=int(args.scene_index_start),
        num_scenes=int(args.num_scenes),
        sequence_seed=int(args.sequence_seed),
        max_scene_index_scan=args.max_scene_index_scan,
    )
    target_orders = [
        {
            "scene_index": scene_index,
            "scene_id": scene.scene_id,
            "target_order": target_order,
            "num_targets": len(target_order),
        }
        for scene_index, scene, target_order in scenes
    ]
    write_json(output_dir / "target_orders.json", target_orders)

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
    class_map = checkpoint_class_map(checkpoint, backend, None)
    feature_config = dict(graph_cfg.get("features", {}))
    feature_config.update(planner_cfg.get("features", {}))
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

    sequence_rows: list[dict] = []
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
        )
        runtime.resettle_steps = int(args.resettle_steps)
        runtime.reject_unstable_resettle = bool(getattr(args, "reject_unstable_resettle", False))
        ordered_runner = OrderedSequenceRunner(
            propose=runtime.proposals,
            plan=runtime.plan,
            validate_target_grasp=runtime.validate,
            resettle_after_intervention=runtime.resettle,
            refresh_observation_metadata=runtime.refresh,
            proposal_trace=runtime.proposal_trace,
        )
        ordered_config = OrderedSequenceConfig(
            fresh_loop_config=build_loop_config(args, planner_type),
            sequence_seed=int(args.sequence_seed),
            resettle_after_target_retrieval=not bool(args.no_resettle_after_target_retrieval),
        )
        for scene_index, scene, target_order in scenes:
            row = ordered_runner.run_sequence(scene, target_order, ordered_config)
            row["scene_index"] = int(scene_index)
            row["proposal_protocol"] = proposal_protocol_config.mode
            sequence_rows.append(row)
            print(
                f"[ordered-sequence] {planner_type} {scene.scene_id} "
                f"completed={row['completed_count']}/{row['total_targets']} "
                f"relocations={row['num_relocations']} reason={row['terminal_reason']}",
                flush=True,
            )

    summary = summarize_ordered_sequences(sequence_rows)
    write_json(output_dir / "ordered_sequences.json", sequence_rows)
    write_json(output_dir / "summary.json", summary)
    write_ordered_summary_csv(output_dir / "summary.csv", summary)
    return sequence_rows, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-scenes", type=int, default=1)
    parser.add_argument("--scene-index-start", type=int, default=0)
    parser.add_argument("--max-scene-index-scan", type=int, default=None)
    parser.add_argument("--sequence-seed", type=int, default=20260512)
    parser.add_argument("--planner-types", default="budgeted_dependency")
    parser.add_argument("--prediction-backend", choices=("auto", "torch", "geometry_heuristic"), default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--planner-config", default="fresh_mujoco_closed_loop/planner_depth_fixed_success_biased.yaml")
    parser.add_argument("--graph-config", default="configs/hetero_gnn_500_progress_edge_context_relation_edge_context_no_interactions_seed7.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--calibration-thresholds", default="")
    parser.add_argument("--proposal-config", default="configs/proposal_sources_icra2026.yaml")
    parser.add_argument(
        "--proposal-protocol",
        choices=("scene", "stage1_aligned_isolated"),
        default="stage1_aligned_isolated",
    )
    parser.add_argument("--scene-config", default="configs/scene_generation_icra2026_mesh_collision_stage4.yaml")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--resettle-steps", type=int, default=1200)
    parser.add_argument("--no-resettle", action="store_true")
    parser.add_argument("--reject-unstable-resettle", action="store_true")
    parser.add_argument("--no-refresh-visibility", action="store_true")
    parser.add_argument(
        "--enable-shared-visibility-fallback",
        action="store_true",
        help=(
            "Apply the same pre-planner largest-visible-object fallback in every target phase "
            "when the target is hidden or current-scene proposals are empty."
        ),
    )
    parser.add_argument("--max-shared-fallback-removals", type=int, default=None)
    parser.add_argument("--shared-fallback-min-visible-pixels", type=int, default=1)
    parser.add_argument("--no-resettle-after-target-retrieval", action="store_true")
    parser.add_argument("--lock-planned-removal-set", action="store_true")
    parser.add_argument("--no-force-target-after-locked-set", action="store_true")
    parser.add_argument("--max-locked-removal-phases", type=int, default=1)
    parser.add_argument("--validation-guided-recovery", action="store_true")
    parser.add_argument("--recover-after-any-target-failure", action="store_true")
    parser.add_argument("--retry-bin-wall-target-grasps", action="store_true")
    parser.add_argument("--max-target-grasp-retries", type=int, default=3)
    parser.add_argument("--bin-wall-safe-target-selection", action="store_true")
    parser.add_argument("--max-bin-wall-safe-candidates", type=int, default=3)
    parser.add_argument("--target-feasibility-selection", action="store_true")
    parser.add_argument("--max-target-feasibility-candidates", type=int, default=8)
    parser.add_argument("--no-target-feasibility-prefer-recoverable-failure", action="store_true")
    parser.add_argument("--target-feasibility-recoverable-max-planner-score", type=float, default=None)
    parser.add_argument("--pre-removal-target-feasibility-guard", action="store_true")
    parser.add_argument("--isolated-augmented-parallel-top-k", type=int, default=16)
    parser.add_argument("--isolated-augmented-suction-top-k", type=int, default=16)
    parser.add_argument("--max-isolated-augmented-parallel", type=int, default=8)
    parser.add_argument("--max-isolated-augmented-suction", type=int, default=8)
    parser.add_argument("--isolated-augmented-max-approach-angle-from-down-deg", type=float, default=None)
    parser.add_argument("--isolated-nms-position-thresh", type=float, default=0.025)
    parser.add_argument("--isolated-nms-direction-cos-thresh", type=float, default=0.965)
    parser.add_argument("--isolated-nms-inplane-cos-thresh", type=float, default=0.94)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _rows, summary = run_cli(args)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
