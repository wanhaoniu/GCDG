"""Run non-terminating ordered retrieval processes for qualitative visualization."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fresh_mujoco_closed_loop.ordered_process_visualization import (  # noqa: E402
    OrderedProcessVisualizationConfig,
    OrderedProcessVisualizationRunner,
    summarize_ordered_processes,
)
from fresh_mujoco_closed_loop.proposal_protocol import FreshProposalProtocolConfig  # noqa: E402
from fresh_mujoco_closed_loop.run_closed_loop import (  # noqa: E402
    CachedDependencyMapPredictor,
    RealFreshMujocoRuntime,
    apply_calibration_to_planner_cfg,
    checkpoint_class_map,
    load_config,
    load_yaml,
    resolve_calibration_thresholds,
    write_json,
)
from fresh_mujoco_closed_loop.run_ordered_sequence import (  # noqa: E402
    build_arg_parser as build_ordered_sequence_arg_parser,
    build_loop_config,
    generate_stable_ordered_scenes,
)
from grasp_dependency_dataset.common.types import StableScene  # noqa: E402
from grasp_dependency_dataset.hetero_gnn.eval_hetero_gnn import choose_backend  # noqa: E402
from grasp_dependency_dataset.hetero_gnn.train_hetero_gnn import resolve_torch_device  # noqa: E402
from grasp_dependency_dataset.pipeline.runner import DatasetPipelineRunner  # noqa: E402
from planners.planner_utils import PlannerParams  # noqa: E402


def write_process_summary_csv(path: Path, summary: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "planner_type",
        "num_scenes",
        "mean_completion_fraction",
        "mean_completed_targets",
        "mean_failed_targets",
        "mean_total_targets",
        "mean_relocations",
        "relocations_per_retrieved_target",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for planner_type, row in sorted(summary.items()):
            payload = {"planner_type": planner_type}
            payload.update({key: row[key] for key in fieldnames if key != "planner_type"})
            writer.writerow(payload)


def _write_main_state(
    *,
    output_dir: Path,
    planner_type: str,
    scene: StableScene,
    target_index: int,
    target_id: str,
    target_success: bool,
) -> dict:
    scene_dir = output_dir / "scratch" / planner_type / "scenes" / scene.scene_id
    scene_path = scene_dir / "scene.json"
    scene_dir.mkdir(parents=True, exist_ok=True)
    write_json(scene_path, scene.to_dict())
    return {
        "scene_id_after_target": scene.scene_id,
        "scene_path_after_target": str(scene_path),
        "target_order_index": int(target_index),
        "target_success": bool(target_success),
        "target_id": target_id,
    }


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
    for scene_index, scene, target_order in scenes:
        if len(scene.objects) != 10 or len(set(target_order)) != 10:
            raise ValueError(
                f"Expected scene {scene.scene_id} at index {scene_index} to contain "
                f"10 distinct objects, got objects={len(scene.objects)} order={len(set(target_order))}."
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

    process_rows: list[dict] = []
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
        process_runner = OrderedProcessVisualizationRunner(
            propose=runtime.proposals,
            plan=runtime.plan,
            validate_target_grasp=runtime.validate,
            resettle_after_intervention=runtime.resettle,
            refresh_observation_metadata=runtime.refresh,
            proposal_trace=runtime.proposal_trace,
        )

        def record_main_state(
            scene: StableScene,
            target_index: int,
            target_id: str,
            target_success: bool,
            *,
            _planner_type: str = planner_type,
        ) -> dict:
            return _write_main_state(
                output_dir=output_dir,
                planner_type=_planner_type,
                scene=scene,
                target_index=target_index,
                target_id=target_id,
                target_success=target_success,
            )

        process_config = OrderedProcessVisualizationConfig(
            fresh_loop_config=build_loop_config(args, planner_type),
            resettle_after_target_retrieval=not bool(args.no_resettle_after_target_retrieval),
            record_main_state=record_main_state,
        )
        for scene_index, scene, target_order in scenes:
            row = process_runner.run_process(scene, target_order, process_config)
            row["scene_index"] = int(scene_index)
            row["proposal_protocol"] = proposal_protocol_config.mode
            process_rows.append(row)
            print(
                f"[ordered-process] {planner_type} {scene.scene_id} "
                f"completed={row['completed_count']}/{row['total_targets']} "
                f"failed={row['failed_count']} relocations={row['num_relocations']}",
                flush=True,
            )

    summary = summarize_ordered_processes(process_rows)
    write_json(output_dir / "ordered_processes.json", process_rows)
    write_json(output_dir / "summary.json", summary)
    write_process_summary_csv(output_dir / "summary.csv", summary)
    return process_rows, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = build_ordered_sequence_arg_parser()
    parser.description = __doc__
    parser.set_defaults(pre_removal_target_feasibility_guard=True)
    parser.add_argument(
        "--no-pre-removal-target-feasibility-guard",
        dest="pre_removal_target_feasibility_guard",
        action="store_false",
        help=(
            "Disable the ordered-process target-first guard. By default, "
            "qualitative runs validate target grasps before relocating blockers."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _rows, summary = run_cli(args)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
