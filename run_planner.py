"""Run dependency-guided minimal intervention planners on existing samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from eval_planner import annotate_result_with_oracle, summarize_results, summary_text
from grasp_dependency_dataset.hetero_gnn.eval_hetero_gnn import choose_backend, torch_model_from_payload
from grasp_dependency_dataset.hetero_gnn.graph_dataset import (
    HeteroGraphDataset,
    TargetGraphSample,
    collate_samples,
    discover_sample_refs,
    extract_grasps,
    read_json,
)
from grasp_dependency_dataset.hetero_gnn.graph_features import (
    GRASP_FEATURE_NAMES,
    OBJECT_FEATURE_NAMES,
    OG_EDGE_FEATURE_NAMES,
)
from grasp_dependency_dataset.hetero_gnn.hetero_gnn import (
    TORCH_AVAILABLE,
    NumpyEdgeLogisticBaseline,
    extract_edge_logits,
    geometry_heuristic_scores_from_batch,
    sigmoid_np,
)
from grasp_dependency_dataset.hetero_gnn.train_hetero_gnn import (
    adapt_batch_feature_dims,
    batch_to_torch,
    iter_sample_batches,
    load_config,
    resolve_torch_device,
)
from planners.dependency_planner import DependencyGuidedPlanner, PlannerSample
from planners.planner_utils import PlannerParams


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def auto_checkpoint() -> Path | None:
    candidates = [
        Path("outputs/hetero_gnn/checkpoints/best.pt"),
        Path("outputs/hetero_gnn_icra_cuda_cached_50ep/checkpoints/best.pt"),
        Path("outputs/hetero_gnn_icra_cuda_cached_10ep/checkpoints/best.pt"),
        Path("outputs/hetero_gnn_icra_cuda_short/checkpoints/best.pt"),
        Path("outputs/hetero_gnn_smoke/checkpoints/best_numpy.npz"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_calibration_thresholds(
    planner_cfg: dict[str, Any],
    checkpoint: Path | None,
) -> tuple[dict[str, Any], Path | None]:
    """Load validation-selected dependency thresholds for planner decisions."""
    if not bool(planner_cfg.get("use_calibrated_thresholds", False)):
        return {}, None

    raw_path = planner_cfg.get("calibration_thresholds_path", "auto")
    candidates: list[Path] = []
    if raw_path and raw_path != "auto":
        candidates.append(Path(raw_path))
    elif checkpoint is not None:
        if len(checkpoint.parents) >= 2:
            candidates.append(checkpoint.parents[1] / "calibration_thresholds.json")
        candidates.append(checkpoint.parent / "calibration_thresholds.json")

    for path in candidates:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            print(f"[planner] using calibrated thresholds from {path}")
            return payload, path

    if candidates:
        tried = ", ".join(str(path) for path in candidates)
        print(f"[planner][warning] calibrated thresholds requested but not found. Tried: {tried}")
    else:
        print("[planner][warning] calibrated thresholds requested but no checkpoint/path was provided.")
    return {}, None


def apply_calibration_to_planner_cfg(
    planner_cfg: dict[str, Any],
    thresholds: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    calibrated = dict(planner_cfg)
    mapping = {
        "dep_any": "tau_any",
        "dep_progress_any": "tau_any",
        "dep_sufficient": "tau_sufficient",
        "dep_approach": "tau_app",
        "dep_app": "tau_app",
        "dep_lift": "tau_lift",
    }
    applied: dict[str, float] = {}
    for metric_name, cfg_name in mapping.items():
        entry = thresholds.get(metric_name)
        if isinstance(entry, dict) and "threshold" in entry:
            value = float(entry["threshold"])
            calibrated[cfg_name] = value
            applied[cfg_name] = value
    return calibrated, applied


def checkpoint_class_map(checkpoint: Path | None, backend: str, class_map_path: Path | None) -> dict[str, int] | None:
    if class_map_path and class_map_path.exists():
        return {str(k): int(v) for k, v in json.loads(class_map_path.read_text(encoding="utf-8")).items()}
    if checkpoint is None:
        return None
    if backend == "torch" and checkpoint.suffix == ".pt":
        if not TORCH_AVAILABLE:
            return None
        import torch

        payload = torch.load(checkpoint, map_location="cpu")
        class_map = payload.get("class_map")
        if isinstance(class_map, dict):
            return {str(k): int(v) for k, v in class_map.items()}
    sibling = checkpoint.parents[1] / "class_map.json" if len(checkpoint.parents) >= 2 else None
    if sibling and sibling.exists():
        return {str(k): int(v) for k, v in json.loads(sibling.read_text(encoding="utf-8")).items()}
    return None


def split_sample_ids(refs: list[Any], planner_cfg: dict[str, Any], graph_cfg: dict[str, Any], checkpoint: Path | None) -> list[str]:
    dataset_cfg = planner_cfg.get("dataset", {})
    split_name = str(dataset_cfg.get("split", "test"))
    split_file_raw = dataset_cfg.get("split_file", "auto")
    split_path: Path | None = None
    if split_file_raw and split_file_raw != "auto":
        split_path = Path(split_file_raw)
    elif checkpoint is not None and len(checkpoint.parents) >= 2 and (checkpoint.parents[1] / "splits.json").exists():
        split_path = checkpoint.parents[1] / "splits.json"
    else:
        graph_split = Path(graph_cfg.get("output", {}).get("dir", "outputs/hetero_gnn")) / "splits.json"
        if graph_split.exists():
            split_path = graph_split

    ref_ids = {ref.sample_id for ref in refs}
    if split_path and split_path.exists():
        splits = json.loads(split_path.read_text(encoding="utf-8"))
        ids = [sample_id for sample_id in splits.get(split_name, []) if sample_id in ref_ids]
    else:
        ids = [ref.sample_id for ref in refs]

    sample_ids = dataset_cfg.get("sample_ids") or []
    if sample_ids:
        keep = set(str(x) for x in sample_ids)
        ids = [sample_id for sample_id in ids if sample_id in keep]

    max_samples = dataset_cfg.get("max_samples")
    if max_samples is not None:
        ids = ids[: int(max_samples)]
    return ids


def load_torch_model(checkpoint: Path, device: str) -> tuple[Any, dict[str, Any]]:
    if not TORCH_AVAILABLE:
        raise ImportError("Torch checkpoint requested, but torch is not available.")
    import torch

    payload = torch.load(checkpoint, map_location=device)
    dims = payload["model_dims"]
    model = torch_model_from_payload(payload, device)
    state = payload["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[planner] checkpoint loaded with missing={len(missing)} unexpected={len(unexpected)} keys.")
    model.eval()
    return model, dims


def predict_dependency_maps(
    dataset: HeteroGraphDataset,
    *,
    checkpoint: Path | None,
    backend: str,
    device: str,
    batch_size: int,
    heuristic_cfg: dict[str, Any] | None = None,
) -> tuple[
    dict[str, dict[str, dict[str, dict[str, float]]]],
    dict[str, dict[str, dict[str, dict[str, float]]]],
]:
    predicted: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    oracle: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    torch_dims: dict[str, Any] | None = None

    heuristic_cfg = heuristic_cfg or {}

    def edge_payload(values: np.ndarray, idx: int) -> dict[str, float]:
        # Stage-2 label order is [progress_any, sufficient, approach, lift].
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

    if backend == "torch":
        if checkpoint is None:
            raise FileNotFoundError("Torch backend requires a checkpoint.")
        model, torch_dims = load_torch_model(checkpoint, device)
    elif backend == "numpy_baseline":
        if checkpoint is None:
            raise FileNotFoundError("NumPy backend requires a checkpoint.")
        model = NumpyEdgeLogisticBaseline.load(checkpoint)
    elif backend == "geometry_heuristic":
        model = None
    elif backend == "random_score":
        model = None
        rng = np.random.default_rng(int(heuristic_cfg.get("seed", 7)))
    else:
        model = None

    if backend == "torch":
        import torch

    for samples in iter_sample_batches(dataset, batch_size=batch_size, shuffle=False, seed=0):
        batch_np = collate_samples(samples)
        if backend == "torch":
            if torch_dims is not None:
                batch_np_model = adapt_batch_feature_dims(batch_np, torch_dims)
            else:
                batch_np_model = batch_np
            batch = batch_to_torch(batch_np_model, device)
            with torch.no_grad():
                outputs = model(batch)  # type: ignore[misc]
                probs = torch.sigmoid(extract_edge_logits(outputs)).detach().cpu().numpy()
        elif backend == "numpy_baseline":
            probs = sigmoid_np(model.predict_logits_from_batch(batch_np))  # type: ignore[union-attr]
        elif backend == "geometry_heuristic":
            probs = geometry_heuristic_scores_from_batch(
                batch_np,
                mode=str(heuristic_cfg.get("mode", "geometry")),
                distance_scale=float(heuristic_cfg.get("distance_scale", 0.020)),
                overlap_weight=float(heuristic_cfg.get("overlap_weight", 1.40)),
                swept_iou_weight=float(heuristic_cfg.get("swept_iou_weight", 1.20)),
                bias=float(heuristic_cfg.get("bias", -1.00)),
            )
        elif backend == "random_score":
            probs = rng.random(batch_np["edge_label_og"].shape, dtype=np.float32)
        elif backend == "oracle":
            probs = batch_np["edge_label_og"].astype(np.float32)
        else:
            raise ValueError(f"Unsupported prediction backend: {backend}")

        labels = batch_np["edge_label_og"].astype(np.float32)
        if labels.shape[0] > 0:
            for idx, sample_id in enumerate(batch_np["edge_sample_ids"]):
                object_id = batch_np["edge_object_ids"][idx]
                grasp_id = batch_np["edge_grasp_ids"][idx]
                predicted.setdefault(sample_id, {}).setdefault(grasp_id, {})[object_id] = edge_payload(probs, idx)
                oracle.setdefault(sample_id, {}).setdefault(grasp_id, {})[object_id] = edge_payload(labels, idx)
    return predicted, oracle


def planner_sample_from_graph(
    graph_sample: TargetGraphSample,
    labels_payload: dict[str, Any],
    predicted_dependencies: dict[str, dict[str, dict[str, float]]],
    oracle_dependencies: dict[str, dict[str, dict[str, float]]],
) -> PlannerSample:
    score_idx = GRASP_FEATURE_NAMES.index("score")
    area_idx = OBJECT_FEATURE_NAMES.index("bbox_area_norm")
    dist_idx = OBJECT_FEATURE_NAMES.index("rel_target_dist")
    target_iou_idx = OBJECT_FEATURE_NAMES.index("rel_target_bbox_iou")
    depth_order_idx = OBJECT_FEATURE_NAMES.index("depth_order")
    grasps = {
        grasp_id: {
            "score": float(graph_sample.x_grasp[idx, score_idx]) if idx < graph_sample.x_grasp.shape[0] else 0.0,
            "type": graph_sample.metadata.get("grasp_types", [""] * len(graph_sample.grasp_ids))[idx],
        }
        for idx, grasp_id in enumerate(graph_sample.grasp_ids)
    }
    object_costs = {
        object_id: float(graph_sample.x_obj[idx, dist_idx]) if idx < graph_sample.x_obj.shape[0] else 0.0
        for idx, object_id in enumerate(graph_sample.object_ids)
    }
    object_graspability = {
        object_id: max(0.1, min(1.0, 10.0 * float(graph_sample.x_obj[idx, area_idx]) + 0.5))
        for idx, object_id in enumerate(graph_sample.object_ids)
    }
    object_visible_area = {
        object_id: float(graph_sample.x_obj[idx, area_idx]) if idx < graph_sample.x_obj.shape[0] else 0.0
        for idx, object_id in enumerate(graph_sample.object_ids)
    }
    object_target_iou = {
        object_id: float(graph_sample.x_obj[idx, target_iou_idx]) if idx < graph_sample.x_obj.shape[0] else 0.0
        for idx, object_id in enumerate(graph_sample.object_ids)
    }
    object_depth_order = {
        object_id: float(graph_sample.x_obj[idx, depth_order_idx]) if idx < graph_sample.x_obj.shape[0] else 0.0
        for idx, object_id in enumerate(graph_sample.object_ids)
    }
    geometry_names = [
        "center_to_grasp_dist",
        "approach_clearance",
        "lift_clearance",
        "bbox_approach_swept_iou",
        "bbox_lift_swept_iou",
        "relative_depth",
        "object_radius_est",
    ]
    geometry_indices = {
        name: OG_EDGE_FEATURE_NAMES.index(name)
        for name in geometry_names
        if name in OG_EDGE_FEATURE_NAMES
    }
    edge_object_ids = [str(value) for value in graph_sample.metadata.get("edge_object_ids", [])]
    edge_grasp_ids = [str(value) for value in graph_sample.metadata.get("edge_grasp_ids", [])]
    object_grasp_geometry: dict[str, dict[str, dict[str, float]]] = {}
    for idx, (object_id, grasp_id) in enumerate(zip(edge_object_ids, edge_grasp_ids)):
        if idx >= graph_sample.edge_attr_og.shape[0]:
            break
        object_grasp_geometry.setdefault(grasp_id, {})[object_id] = {
            name: float(graph_sample.edge_attr_og[idx, feature_idx])
            for name, feature_idx in geometry_indices.items()
        }
    oracle_min_by_grasp = {
        str(item.get("grasp_id")): item
        for item in labels_payload.get("planning_labels", [])
        if item.get("grasp_id")
    }
    grasp_feasibility = {
        str(item.get("grasp_id")): item
        for item in labels_payload.get("grasp_feasibility", [])
        if item.get("grasp_id")
    }
    return PlannerSample(
        scene_id=graph_sample.scene_id,
        target_id=graph_sample.target_id,
        object_ids=list(graph_sample.object_ids),
        grasps=grasps,
        predicted_dependencies=predicted_dependencies,
        oracle_dependencies=oracle_dependencies,
        object_costs=object_costs,
        object_graspability=object_graspability,
        object_visible_area=object_visible_area,
        object_target_iou=object_target_iou,
        object_depth_order=object_depth_order,
        object_grasp_geometry=object_grasp_geometry,
        oracle_min_by_grasp=oracle_min_by_grasp,
        grasp_feasibility=grasp_feasibility,
        planning_summary=labels_payload.get("planning_summary", {}),
    )


def run(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graph_cfg = load_config(Path(cfg.get("graph_config", "configs/hetero_gnn.yaml")))
    dataset_root = Path(cfg.get("dataset", {}).get("root") or graph_cfg["dataset"]["root"])
    prediction_cfg = cfg.get("prediction", {})
    checkpoint = Path(prediction_cfg["checkpoint"]) if prediction_cfg.get("checkpoint") else auto_checkpoint()
    backend = choose_backend(graph_cfg, checkpoint, prediction_cfg.get("backend", "auto"))
    if prediction_cfg.get("backend") == "oracle":
        backend = "oracle"
    device = "cpu"
    if backend == "torch":
        device = resolve_torch_device(str(prediction_cfg.get("device", graph_cfg.get("training", {}).get("device", "auto"))))

    refs = discover_sample_refs(dataset_root)
    sample_ids = split_sample_ids(refs, cfg, graph_cfg, checkpoint)
    class_map_path = Path(prediction_cfg["class_map_path"]) if prediction_cfg.get("class_map_path") else None
    class_map = checkpoint_class_map(checkpoint, backend, class_map_path)
    feature_cfg = dict(graph_cfg.get("features", {}))
    feature_cfg.update(cfg.get("features", {}))
    dataset = HeteroGraphDataset(
        dataset_root,
        refs=refs,
        sample_ids=sample_ids,
        feature_config=feature_cfg,
        class_map=class_map,
    )
    batch_size = int(prediction_cfg.get("batch_size", graph_cfg.get("training", {}).get("batch_size", 4)))
    print(
        f"[planner] samples={len(dataset)} backend={backend} checkpoint={checkpoint} "
        f"device={device} dataset_root={dataset_root}"
    )
    predicted_maps, oracle_maps = predict_dependency_maps(
        dataset,
        checkpoint=checkpoint,
        backend=backend,
        device=device,
        batch_size=batch_size,
        heuristic_cfg=graph_cfg.get("geometry_heuristic", {}),
    )

    planner_cfg = dict(cfg.get("planner", {}))
    calibration_payload, calibration_path = resolve_calibration_thresholds(planner_cfg, checkpoint)
    planner_cfg, applied_thresholds = apply_calibration_to_planner_cfg(planner_cfg, calibration_payload)
    params = PlannerParams.from_dict(planner_cfg)
    threshold_info = {
        "tau_any": params.tau_any,
        "tau_sufficient": params.tau_sufficient,
        "tau_app": params.tau_app,
        "tau_lift": params.tau_lift,
        "use_calibrated_thresholds": bool(applied_thresholds),
        "calibration_thresholds_path": str(calibration_path) if calibration_path else None,
        "applied_calibrated_thresholds": applied_thresholds,
    }
    print(
        "[planner] thresholds: "
        f"tau_any={params.tau_any:.3f} tau_sufficient={params.tau_sufficient:.3f} "
        f"tau_app={params.tau_app:.3f} tau_lift={params.tau_lift:.3f} "
        f"calibrated={bool(applied_thresholds)}"
    )
    planner_types = list(cfg.get("baselines", {}).get("enabled", ["predicted_dependency"]))
    if backend == "oracle" and "predicted_dependency" in planner_types:
        planner_types = ["oracle_dependency" if item == "predicted_dependency" else item for item in planner_types]
    modes = []
    if bool(planner_cfg.get("run_open_loop", True)):
        modes.append(False)
    if bool(planner_cfg.get("run_closed_loop", True)):
        modes.append(True)
    planner = DependencyGuidedPlanner(
        params,
        rng_seed=int(planner_cfg.get("seed", 7)),
        verbose=bool(planner_cfg.get("verbose", False)),
    )

    results: list[dict[str, Any]] = []
    for idx in range(len(dataset)):
        graph_sample = dataset[idx]
        ref = dataset.ref_by_id[graph_sample.sample_id]
        labels_payload = read_json(ref.labels_path)
        planner_sample = planner_sample_from_graph(
            graph_sample,
            labels_payload,
            predicted_maps.get(graph_sample.sample_id, {}),
            oracle_maps.get(graph_sample.sample_id, {}),
        )
        if bool(planner_cfg.get("verbose", False)):
            print(
                f"[planner] sample {graph_sample.scene_id}/{graph_sample.target_id}: "
                f"observed={graph_sample.metadata.get('num_observed_objects', len(graph_sample.object_ids))}"
            )
        for planner_type in planner_types:
            for closed_loop in modes:
                result = planner.plan(planner_sample, planner_type=planner_type, closed_loop=closed_loop)
                result = annotate_result_with_oracle(result, planner_sample)
                result["planner_thresholds"] = threshold_info
                results.append(result)

    metrics = summarize_results(results)
    metrics["planner_thresholds"] = threshold_info
    metrics["planner_params"] = {
        "tau_any": params.tau_any,
        "tau_sufficient": params.tau_sufficient,
        "tau_app": params.tau_app,
        "tau_lift": params.tau_lift,
        "alpha": params.alpha,
        "beta": params.beta,
        "gamma": params.gamma,
        "eta": params.eta,
        "max_steps": params.max_steps,
        "budget_residual_weight": params.budget_residual_weight,
        "budget_residual_ratio_weight": params.budget_residual_ratio_weight,
        "budget_removal_weight": params.budget_removal_weight,
        "budget_removed_reward": params.budget_removed_reward,
        "budget_coverage_reward": params.budget_coverage_reward,
        "mbs_residual_threshold": params.mbs_residual_threshold,
        "topk_min_strength": params.topk_min_strength,
        "candidate_grasp_min_score_ratio": params.candidate_grasp_min_score_ratio,
        "candidate_grasp_top_k": params.candidate_grasp_top_k,
        "use_typed_dependency": params.use_typed_dependency,
        "use_sufficient_dependency": params.use_sufficient_dependency,
        "use_grasp_score": params.use_grasp_score,
        "mechanical_search_target_score_thresh": params.mechanical_search_target_score_thresh,
        "xray_target_score_thresh": params.xray_target_score_thresh,
        "xray_overlap_weight": params.xray_overlap_weight,
        "xray_near_weight": params.xray_near_weight,
        "xray_front_weight": params.xray_front_weight,
        "xray_area_weight": params.xray_area_weight,
    }
    return results, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/planner.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--backend",
        choices=["auto", "torch", "numpy_baseline", "geometry_heuristic", "random_score", "oracle"],
        default=None,
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.checkpoint is not None:
        cfg.setdefault("prediction", {})["checkpoint"] = str(args.checkpoint)
    if args.backend is not None:
        cfg.setdefault("prediction", {})["backend"] = args.backend
    if args.max_samples is not None:
        cfg.setdefault("dataset", {})["max_samples"] = args.max_samples
    if args.output_dir is not None:
        cfg.setdefault("output", {})["dir"] = str(args.output_dir)

    results, metrics = run(cfg)
    output_dir = Path(cfg.get("output", {}).get("dir", "outputs"))
    results_path = output_dir / "planner_results.json"
    metrics_path = output_dir / "planner_metrics.json"
    summary_path = output_dir / "planner_summary.txt"
    write_json(results_path, results)
    write_json(metrics_path, metrics)
    summary = summary_text(metrics)
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"[planner] wrote {results_path}, {metrics_path}, {summary_path}")


if __name__ == "__main__":
    main()
