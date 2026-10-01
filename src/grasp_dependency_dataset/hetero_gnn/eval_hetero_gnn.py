"""Evaluation entry point for hetero GNN dependency checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from grasp_dependency_dataset.hetero_gnn.graph_dataset import HeteroGraphDataset, collate_samples
from grasp_dependency_dataset.hetero_gnn.hetero_gnn import (
    TORCH_AVAILABLE,
    EdgeMLPDependencyBaseline,
    G2N2StyleDependencyBaseline,
    HeteroDependencyGNN,
    NumpyEdgeLogisticBaseline,
    ObjectOnlyDependencyBaseline,
    extract_edge_logits,
    geometry_heuristic_scores_from_batch,
    sigmoid_np,
)
from grasp_dependency_dataset.hetero_gnn.metrics import LABEL_NAMES, blocker_set_metrics, edge_level_metrics
from grasp_dependency_dataset.hetero_gnn.minimal_blocker_metrics import minimal_blocker_set_metrics
from grasp_dependency_dataset.hetero_gnn.train_hetero_gnn import (
    adapt_batch_feature_dims,
    batch_to_torch,
    iter_sample_batches,
    load_config,
    make_datasets,
    resolve_torch_device,
    write_json,
)


def default_checkpoint(output_dir: Path, backend: str) -> Path:
    if backend in {"torch", "edge_mlp", "g2n2_style", "object_only"}:
        return output_dir / "checkpoints" / "best.pt"
    return output_dir / "checkpoints" / "best_numpy.npz"


def choose_backend(cfg: dict[str, Any], checkpoint: Path | None, requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    if checkpoint is not None:
        if checkpoint.suffix == ".pt":
            return "torch"
        if checkpoint.suffix == ".npz":
            return "numpy_baseline"
    output_dir = Path(cfg["output"]["dir"])
    if (output_dir / "checkpoints" / "best.pt").exists():
        return "torch"
    if (output_dir / "checkpoints" / "best_numpy.npz").exists():
        return "numpy_baseline"
    return "torch" if TORCH_AVAILABLE else "numpy_baseline"


def torch_model_from_payload(payload: dict[str, Any], device: str) -> Any:
    dims = payload["model_dims"]
    model_cfg = payload.get("model_config", {})
    model_type = str(payload.get("model_type") or ("edge_mlp" if payload.get("backend") == "edge_mlp" else "hetero_gnn"))
    num_edge_labels = int(dims.get("num_edge_labels", 3))
    if model_type == "edge_mlp":
        return EdgeMLPDependencyBaseline(
            object_in_dim=int(dims["object_in_dim"]),
            grasp_in_dim=int(dims["grasp_in_dim"]),
            og_edge_dim=int(dims["og_edge_dim"]),
            num_edge_labels=num_edge_labels,
            hidden_dim=int(model_cfg.get("edge_mlp_hidden_dim", model_cfg.get("hidden_dim", 128))),
            decoder_layers=int(model_cfg.get("edge_mlp_layers", model_cfg.get("decoder_layers", 3))),
            dropout=float(model_cfg.get("dropout", 0.10)),
        ).to(device)
    if model_type == "g2n2_style":
        return G2N2StyleDependencyBaseline(
            object_in_dim=int(dims["object_in_dim"]),
            grasp_in_dim=int(dims["grasp_in_dim"]),
            og_edge_dim=int(dims["og_edge_dim"]),
            num_edge_labels=num_edge_labels,
            hidden_dim=int(model_cfg.get("g2n2_hidden_dim", model_cfg.get("hidden_dim", 128))),
            message_layers=int(model_cfg.get("g2n2_message_layers", 3)),
            decoder_layers=int(model_cfg.get("g2n2_decoder_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.10)),
            use_og_edge_features=bool(model_cfg.get("g2n2_use_og_edge_features", True)),
        ).to(device)
    if model_type == "object_only":
        return ObjectOnlyDependencyBaseline(
            object_in_dim=int(dims["object_in_dim"]),
            num_edge_labels=num_edge_labels,
            hidden_dim=int(model_cfg.get("object_only_hidden_dim", model_cfg.get("hidden_dim", 128))),
            decoder_layers=int(model_cfg.get("object_only_layers", model_cfg.get("decoder_layers", 3))),
            dropout=float(model_cfg.get("dropout", 0.10)),
        ).to(device)
    return HeteroDependencyGNN(
        object_in_dim=int(dims["object_in_dim"]),
        grasp_in_dim=int(dims["grasp_in_dim"]),
        oo_edge_dim=int(dims["oo_edge_dim"]),
        gg_edge_dim=int(dims["gg_edge_dim"]),
        og_edge_dim=int(dims["og_edge_dim"]),
        num_object_classes=int(dims["num_object_classes"]),
        num_edge_labels=num_edge_labels,
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        class_emb_dim=int(model_cfg.get("class_emb_dim", 16)),
        grasp_type_emb_dim=int(model_cfg.get("grasp_type_emb_dim", 8)),
        message_layers=int(model_cfg.get("message_layers", 2)),
        cross_layers=int(model_cfg.get("cross_layers", 1)),
        decoder_layers=int(model_cfg.get("decoder_layers", 3)),
        dropout=float(model_cfg.get("dropout", 0.10)),
        conv_variant=str(model_cfg.get("conv_variant", "legacy")),
        decoder_interactions=bool(model_cfg.get("decoder_interactions", False)),
        decoder_use_edge_attr=bool(model_cfg.get("decoder_use_edge_attr", True)),
        decoder_raw_features=bool(model_cfg.get("decoder_raw_features", False)),
        separate_label_heads=bool(model_cfg.get("separate_label_heads", False)),
        decoder_head_layers=int(model_cfg.get("decoder_head_layers", 2)),
        edge_context_layers=int(model_cfg.get("edge_context_layers", 0)),
        edge_context_source=str(model_cfg.get("edge_context_source", "raw")),
        edge_context_use_edge_attr=bool(model_cfg.get("edge_context_use_edge_attr", True)),
        edge_context_use_object=bool(model_cfg.get("edge_context_use_object", True)),
        edge_context_interactions=bool(model_cfg.get("edge_context_interactions", True)),
        edge_context_attention=bool(model_cfg.get("edge_context_attention", False)),
        edge_context_attention_query=bool(model_cfg.get("edge_context_attention_query", False)),
        edge_context_logit_residual=bool(model_cfg.get("edge_context_logit_residual", False)),
        edge_context_logit_init=float(model_cfg.get("edge_context_logit_init", 0.10)),
        grasp_conditioned_object_layers=int(model_cfg.get("grasp_conditioned_object_layers", 0)),
        grasp_conditioned_object_use_edge_attr=bool(model_cfg.get("grasp_conditioned_object_use_edge_attr", True)),
        grasp_conditioned_object_require_edges=bool(model_cfg.get("grasp_conditioned_object_require_edges", False)),
        grasp_conditioned_object_interactions=bool(model_cfg.get("grasp_conditioned_object_interactions", True)),
        grasp_conditioned_object_decoder_context=bool(model_cfg.get("grasp_conditioned_object_decoder_context", True)),
        grasp_conditioned_object_logit_residual=bool(model_cfg.get("grasp_conditioned_object_logit_residual", False)),
        grasp_conditioned_object_logit_init=float(model_cfg.get("grasp_conditioned_object_logit_init", 0.10)),
        blocker_relation_layers=int(model_cfg.get("blocker_relation_layers", 0)),
        blocker_relation_decoder_context=bool(model_cfg.get("blocker_relation_decoder_context", True)),
        blocker_relation_interactions=bool(model_cfg.get("blocker_relation_interactions", True)),
        blocker_relation_logit_residual=bool(model_cfg.get("blocker_relation_logit_residual", False)),
        blocker_relation_logit_init=float(model_cfg.get("blocker_relation_logit_init", 0.05)),
    ).to(device)


def collect_numpy_predictions(
    model: NumpyEdgeLogisticBaseline,
    dataset: HeteroGraphDataset,
    *,
    batch_size: int,
) -> dict[str, Any]:
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    edge_sample_ids: list[str] = []
    edge_object_ids: list[str] = []
    edge_grasp_ids: list[str] = []
    planning_labels_by_sample: dict[str, list[dict[str, Any]]] = {}
    for samples in iter_sample_batches(dataset, batch_size=batch_size, shuffle=False, seed=0):
        batch = collate_samples(samples)
        for sample in samples:
            planning_labels_by_sample[sample.sample_id] = list(sample.metadata.get("planning_labels", []))
        if batch["edge_label_og"].shape[0] == 0:
            continue
        logits = model.predict_logits_from_batch(batch)
        labels.append(batch["edge_label_og"])
        scores.append(sigmoid_np(logits))
        masks.append(batch["edge_label_mask_og"])
        edge_sample_ids.extend(batch["edge_sample_ids"])
        edge_object_ids.extend(batch["edge_object_ids"])
        edge_grasp_ids.extend(batch["edge_grasp_ids"])
    return merge_prediction_chunks(
        labels,
        scores,
        masks,
        edge_sample_ids,
        edge_object_ids,
        edge_grasp_ids,
        planning_labels_by_sample=planning_labels_by_sample,
    )


def collect_geometry_predictions(
    dataset: HeteroGraphDataset,
    *,
    batch_size: int,
    heuristic_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    edge_sample_ids: list[str] = []
    edge_object_ids: list[str] = []
    edge_grasp_ids: list[str] = []
    planning_labels_by_sample: dict[str, list[dict[str, Any]]] = {}
    heuristic_cfg = heuristic_cfg or {}
    for samples in iter_sample_batches(dataset, batch_size=batch_size, shuffle=False, seed=0):
        batch = collate_samples(samples)
        for sample in samples:
            planning_labels_by_sample[sample.sample_id] = list(sample.metadata.get("planning_labels", []))
        if batch["edge_label_og"].shape[0] == 0:
            continue
        labels.append(batch["edge_label_og"])
        scores.append(
            geometry_heuristic_scores_from_batch(
                batch,
                mode=str(heuristic_cfg.get("mode", "geometry")),
                distance_scale=float(heuristic_cfg.get("distance_scale", 0.020)),
                overlap_weight=float(heuristic_cfg.get("overlap_weight", 1.40)),
                swept_iou_weight=float(heuristic_cfg.get("swept_iou_weight", 1.20)),
                bias=float(heuristic_cfg.get("bias", -1.00)),
            )
        )
        masks.append(batch["edge_label_mask_og"])
        edge_sample_ids.extend(batch["edge_sample_ids"])
        edge_object_ids.extend(batch["edge_object_ids"])
        edge_grasp_ids.extend(batch["edge_grasp_ids"])
    return merge_prediction_chunks(
        labels,
        scores,
        masks,
        edge_sample_ids,
        edge_object_ids,
        edge_grasp_ids,
        planning_labels_by_sample=planning_labels_by_sample,
    )


def collect_random_predictions(
    dataset: HeteroGraphDataset,
    *,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """No-signal non-learning baseline for sanity-checking ranking metrics."""

    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    edge_sample_ids: list[str] = []
    edge_object_ids: list[str] = []
    edge_grasp_ids: list[str] = []
    planning_labels_by_sample: dict[str, list[dict[str, Any]]] = {}
    rng = np.random.default_rng(seed)
    for samples in iter_sample_batches(dataset, batch_size=batch_size, shuffle=False, seed=0):
        batch = collate_samples(samples)
        for sample in samples:
            planning_labels_by_sample[sample.sample_id] = list(sample.metadata.get("planning_labels", []))
        if batch["edge_label_og"].shape[0] == 0:
            continue
        labels.append(batch["edge_label_og"])
        scores.append(rng.random(batch["edge_label_og"].shape, dtype=np.float32))
        masks.append(batch["edge_label_mask_og"])
        edge_sample_ids.extend(batch["edge_sample_ids"])
        edge_object_ids.extend(batch["edge_object_ids"])
        edge_grasp_ids.extend(batch["edge_grasp_ids"])
    return merge_prediction_chunks(
        labels,
        scores,
        masks,
        edge_sample_ids,
        edge_object_ids,
        edge_grasp_ids,
        planning_labels_by_sample=planning_labels_by_sample,
    )


def collect_torch_predictions(
    checkpoint: Path,
    dataset: HeteroGraphDataset,
    *,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise ImportError("Cannot evaluate a torch checkpoint because PyTorch is not installed.")
    import torch

    payload = torch.load(checkpoint, map_location=device)
    dims = payload["model_dims"]
    model = torch_model_from_payload(payload, device)
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
    if missing or unexpected:
        print(f"[eval] checkpoint loaded with missing={len(missing)} unexpected={len(unexpected)} keys.")
    model.eval()

    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    edge_sample_ids: list[str] = []
    edge_object_ids: list[str] = []
    edge_grasp_ids: list[str] = []
    planning_labels_by_sample: dict[str, list[dict[str, Any]]] = {}
    with torch.no_grad():
        for samples in iter_sample_batches(dataset, batch_size=batch_size, shuffle=False, seed=0):
            batch_np = collate_samples(samples)
            for sample in samples:
                planning_labels_by_sample[sample.sample_id] = list(sample.metadata.get("planning_labels", []))
            if batch_np["edge_label_og"].shape[0] == 0:
                continue
            batch_np = adapt_batch_feature_dims(batch_np, dims)
            batch = batch_to_torch(batch_np, device)
            logits = extract_edge_logits(model(batch))
            labels.append(batch_np["edge_label_og"])
            scores.append(torch.sigmoid(logits).cpu().numpy())
            masks.append(batch_np["edge_label_mask_og"])
            edge_sample_ids.extend(batch_np["edge_sample_ids"])
            edge_object_ids.extend(batch_np["edge_object_ids"])
            edge_grasp_ids.extend(batch_np["edge_grasp_ids"])
    return merge_prediction_chunks(
        labels,
        scores,
        masks,
        edge_sample_ids,
        edge_object_ids,
        edge_grasp_ids,
        planning_labels_by_sample=planning_labels_by_sample,
    )


def merge_prediction_chunks(
    labels: list[np.ndarray],
    scores: list[np.ndarray],
    masks: list[np.ndarray],
    edge_sample_ids: list[str],
    edge_object_ids: list[str],
    edge_grasp_ids: list[str],
    *,
    planning_labels_by_sample: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if not labels:
        return {
            "y_true": np.zeros((0, len(LABEL_NAMES)), dtype=np.float32),
            "y_score": np.zeros((0, len(LABEL_NAMES)), dtype=np.float32),
            "mask": np.zeros((0,), dtype=np.float32),
            "edge_sample_ids": [],
            "edge_object_ids": [],
            "edge_grasp_ids": [],
            "planning_labels_by_sample": planning_labels_by_sample or {},
        }
    return {
        "y_true": np.concatenate(labels, axis=0),
        "y_score": np.concatenate(scores, axis=0),
        "mask": np.concatenate(masks, axis=0),
        "edge_sample_ids": edge_sample_ids,
        "edge_object_ids": edge_object_ids,
        "edge_grasp_ids": edge_grasp_ids,
        "planning_labels_by_sample": planning_labels_by_sample or {},
    }


def eval_metrics_from_prediction(
    pred: dict[str, Any],
    *,
    threshold: float,
    backend: str,
    checkpoint: str,
    split: str,
) -> dict[str, Any]:
    """Build all evaluation metrics from a collected prediction payload."""

    metrics = {
        "backend": backend,
        "checkpoint": checkpoint,
        "split": split,
        "edge": edge_level_metrics(pred["y_true"], pred["y_score"], pred["mask"], threshold=threshold),
        "blocker_sets": blocker_set_metrics(
            pred["edge_sample_ids"],
            pred["edge_object_ids"],
            pred["edge_grasp_ids"],
            pred["y_true"],
            pred["y_score"],
            pred["mask"],
            threshold=threshold,
        ),
    }
    planning_labels_by_sample = pred.get("planning_labels_by_sample") or {}
    if planning_labels_by_sample:
        metrics["minimal_blocker_sets"] = minimal_blocker_set_metrics(
            edge_sample_ids=pred["edge_sample_ids"],
            edge_object_ids=pred["edge_object_ids"],
            edge_grasp_ids=pred["edge_grasp_ids"],
            y_score=pred["y_score"],
            planning_labels_by_sample=planning_labels_by_sample,
            mask=pred["mask"],
            threshold=threshold,
            score_label_index=0,
            recall_ks=(1, 2, 3, 5),
            y_true=pred["y_true"],
        )
    return metrics


def prediction_json(pred: dict[str, Any], threshold: float) -> dict[str, Any]:
    y_true = pred["y_true"]
    y_score = pred["y_score"]
    mask = pred["mask"].astype(bool)
    active_label_names = LABEL_NAMES[: min(len(LABEL_NAMES), y_true.shape[1], y_score.shape[1])]
    edges = []
    for idx, valid in enumerate(mask.tolist()):
        if not valid:
            continue
        edges.append(
            {
                "sample_id": pred["edge_sample_ids"][idx],
                "object_id": pred["edge_object_ids"][idx],
                "grasp_id": pred["edge_grasp_ids"][idx],
                "probabilities": {name: float(y_score[idx, label_i]) for label_i, name in enumerate(active_label_names)},
                "labels": {name: int(y_true[idx, label_i] >= 0.5) for label_i, name in enumerate(active_label_names)},
                "predicted_dep_progress_any": bool(y_score[idx, 0] >= threshold),
                "predicted_dep_sufficient": bool(y_score[idx, 1] >= threshold) if y_score.shape[1] > 1 else False,
            }
        )
    blocker_sets = blocker_set_metrics(
        pred["edge_sample_ids"],
        pred["edge_object_ids"],
        pred["edge_grasp_ids"],
        y_true,
        y_score,
        pred["mask"],
        threshold=threshold,
    )
    return {
        "threshold": threshold,
        "edges": edges,
        "planner_blocker_sets": blocker_sets["per_grasp"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hetero_gnn.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--backend",
        choices=[
            "auto",
            "torch",
            "edge_mlp",
            "g2n2_style",
            "object_only",
            "numpy_baseline",
            "geometry_heuristic",
            "random_score",
        ],
        default="auto",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output"]["dir"])
    backend = choose_backend(cfg, args.checkpoint, args.backend)
    checkpoint = args.checkpoint or default_checkpoint(output_dir, backend)
    if backend not in {"geometry_heuristic", "random_score"} and not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    train_ds, val_ds, test_ds, _class_map, _refs, _splits = make_datasets(cfg)
    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
    dataset = datasets[args.split]
    batch_size = int(cfg.get("training", {}).get("batch_size", 4))
    threshold = float(cfg.get("evaluation", {}).get("threshold", 0.5))
    if backend in {"torch", "edge_mlp", "g2n2_style", "object_only"}:
        device = resolve_torch_device(str(cfg.get("training", {}).get("device", "auto")))
        pred = collect_torch_predictions(checkpoint, dataset, batch_size=batch_size, device=device)
    elif backend == "geometry_heuristic":
        pred = collect_geometry_predictions(
            dataset,
            batch_size=batch_size,
            heuristic_cfg=cfg.get("geometry_heuristic", {}),
        )
    elif backend == "random_score":
        pred = collect_random_predictions(
            dataset,
            batch_size=batch_size,
            seed=int(cfg.get("training", {}).get("seed", 0)),
        )
    else:
        model = NumpyEdgeLogisticBaseline.load(checkpoint)
        pred = collect_numpy_predictions(model, dataset, batch_size=batch_size)

    metrics = eval_metrics_from_prediction(
        pred,
        threshold=threshold,
        backend=backend,
        checkpoint="" if backend in {"geometry_heuristic", "random_score"} else str(checkpoint),
        split=args.split,
    )
    out_prefix = args.output or (output_dir / f"eval_{args.split}")
    write_json(Path(str(out_prefix) + "_metrics.json"), metrics)
    write_json(Path(str(out_prefix) + "_predictions.json"), prediction_json(pred, threshold))
    print(json.dumps(metrics["edge"], indent=2))
    print(json.dumps(metrics["blocker_sets"], indent=2)[:2000])


if __name__ == "__main__":
    main()
