"""Evaluate simple stage-2 dependency baselines on a fixed split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from grasp_dependency_dataset.hetero_gnn.eval_hetero_gnn import (
    collect_geometry_predictions,
    collect_random_predictions,
)
from grasp_dependency_dataset.hetero_gnn.graph_dataset import FEATURE_SCHEMA
from grasp_dependency_dataset.hetero_gnn.metrics import (
    LABEL_NAMES,
    binary_metrics,
    blocker_set_metrics,
    edge_level_metrics,
)
from grasp_dependency_dataset.hetero_gnn.train_hetero_gnn import (
    calibration_thresholds_from_metrics,
    load_config,
    make_datasets,
    write_json,
)


def compact_blocker_sets(metrics: dict[str, Any]) -> dict[str, Any]:
    out = dict(metrics)
    out.pop("per_grasp", None)
    return out


def metrics_from_prediction(pred: dict[str, Any], *, threshold: float) -> dict[str, Any]:
    blocker_sets = blocker_set_metrics(
        pred["edge_sample_ids"],
        pred["edge_object_ids"],
        pred["edge_grasp_ids"],
        pred["y_true"],
        pred["y_score"],
        pred["mask"],
        threshold=threshold,
    )
    return {
        "edge": edge_level_metrics(pred["y_true"], pred["y_score"], pred["mask"], threshold=threshold),
        "blocker_sets": compact_blocker_sets(blocker_sets),
    }


def metrics_with_validation_thresholds(
    pred: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    y_true = np.asarray(pred["y_true"], dtype=np.float32)
    y_score = np.asarray(pred["y_score"], dtype=np.float32)
    keep = np.asarray(pred["mask"], dtype=bool)
    yt = y_true[keep]
    ys = y_score[keep]
    num_labels = min(len(LABEL_NAMES), yt.shape[1], ys.shape[1])

    thresholds: dict[str, float] = {}
    label_metrics: dict[str, Any] = {}
    for idx, label in enumerate(LABEL_NAMES[:num_labels]):
        threshold = float(calibration.get(label, {}).get("threshold", 0.5))
        thresholds[label] = threshold
        values = binary_metrics(yt[:, idx], ys[:, idx], threshold=threshold)
        values["threshold"] = threshold
        label_metrics[label] = values

    dep_any_threshold = thresholds.get("dep_progress_any", 0.5)
    blocker_sets = blocker_set_metrics(
        pred["edge_sample_ids"],
        pred["edge_object_ids"],
        pred["edge_grasp_ids"],
        pred["y_true"],
        pred["y_score"],
        pred["mask"],
        threshold=dep_any_threshold,
    )
    return {
        "threshold_source": "validation_best_f1",
        "thresholds": thresholds,
        "edge": {"labels": label_metrics},
        "blocker_sets": compact_blocker_sets(blocker_sets),
    }


def collect_predictions(cfg: dict[str, Any], dataset: Any) -> dict[str, Any]:
    backend = str(cfg.get("backend", "geometry_heuristic"))
    batch_size = int(cfg.get("training", {}).get("batch_size", 32))
    if backend == "random_score":
        return collect_random_predictions(
            dataset,
            batch_size=batch_size,
            seed=int(cfg.get("training", {}).get("seed", 0)),
        )
    if backend == "geometry_heuristic":
        return collect_geometry_predictions(
            dataset,
            batch_size=batch_size,
            heuristic_cfg=cfg.get("geometry_heuristic", {}),
        )
    raise ValueError(f"Unsupported simple baseline backend: {backend}")


def run_one(config_path: Path) -> dict[str, float]:
    cfg = load_config(config_path)
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, test_ds, class_map, refs, splits = make_datasets(cfg)
    write_json(output_dir / "class_map.json", class_map)
    write_json(output_dir / "feature_schema.json", FEATURE_SCHEMA)
    write_json(output_dir / "config_used.json", cfg)

    threshold = float(cfg.get("evaluation", {}).get("threshold", 0.5))
    val_pred = collect_predictions(cfg, val_ds)
    test_pred = collect_predictions(cfg, test_ds)
    val_metrics = metrics_from_prediction(val_pred, threshold=threshold)
    test_metrics = metrics_from_prediction(test_pred, threshold=threshold)
    calibration = calibration_thresholds_from_metrics(val_metrics)
    test_val_thresholds = metrics_with_validation_thresholds(test_pred, calibration)

    write_json(output_dir / "val_metrics.json", val_metrics)
    write_json(output_dir / "test_metrics.json", test_metrics)
    write_json(output_dir / "calibration_thresholds.json", calibration)
    write_json(output_dir / "test_metrics_val_thresholds.json", test_val_thresholds)

    dep_any_val = val_metrics["edge"]["labels"]["dep_progress_any"]
    dep_any_test = test_metrics["edge"]["labels"]["dep_progress_any"]
    dep_any_test_thr = test_val_thresholds["edge"]["labels"]["dep_progress_any"]
    blocker = test_val_thresholds["blocker_sets"]
    return {
        "num_refs": float(len(refs)),
        "num_train": float(len(train_ds)),
        "num_val": float(len(val_ds)),
        "num_test": float(len(test_ds)),
        "val_ap": float(dep_any_val["ap"]),
        "test_ap": float(dep_any_test["ap"]),
        "test_auc": float(dep_any_test["auc"]),
        "test_f1_at_val_threshold": float(dep_any_test_thr["f1"]),
        "test_precision_at_val_threshold": float(dep_any_test_thr["precision"]),
        "test_recall_at_val_threshold": float(dep_any_test_thr["recall"]),
        "blocker_iou_at_val_threshold": float(blocker["mean_set_iou"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    args = parser.parse_args()
    for config_path in args.configs:
        row = run_one(config_path)
        print(
            f"{config_path}: test_ap={row['test_ap']:.4f} "
            f"test_auc={row['test_auc']:.4f} "
            f"f1@val-thr={row['test_f1_at_val_threshold']:.4f} "
            f"blocker_iou@val-thr={row['blocker_iou_at_val_threshold']:.4f}"
        )


if __name__ == "__main__":
    main()
