"""Training entry point for target-centric heterogeneous dependency graphs."""

from __future__ import annotations

import argparse
import json
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from grasp_dependency_dataset.hetero_gnn.graph_dataset import (
    FEATURE_SCHEMA,
    HeteroGraphDataset,
    TargetGraphSample,
    collate_samples,
    dependency_label_lookup,
    discover_sample_refs,
    extract_graph_support_objects,
    extract_grasps,
    load_or_create_splits,
    read_json,
)
from grasp_dependency_dataset.hetero_gnn.graph_features import (
    object_has_visibility_info,
    visibility_status_from_object,
)
from grasp_dependency_dataset.hetero_gnn.hetero_gnn import (
    TORCH_AVAILABLE,
    EdgeMLPDependencyBaseline,
    G2N2StyleDependencyBaseline,
    HeteroDependencyGNN,
    NumpyEdgeLogisticBaseline,
    ObjectOnlyDependencyBaseline,
    edge_design_matrix,
    extract_edge_logits,
    geometry_heuristic_scores_from_batch,
    multilabel_dependency_loss,
    require_torch,
    sigmoid_np,
)
from grasp_dependency_dataset.hetero_gnn.minimal_blocker_metrics import minimal_blocker_set_metrics
from grasp_dependency_dataset.hetero_gnn.metrics import LABEL_NAMES, blocker_set_metrics, edge_level_metrics


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def iter_sample_batches(
    dataset: HeteroGraphDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> list[list[TargetGraphSample]]:
    indices = list(range(len(dataset)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    batches: list[list[TargetGraphSample]] = []
    for start in range(0, len(indices), batch_size):
        samples = [dataset[idx] for idx in indices[start : start + batch_size]]
        batches.append(samples)
    return batches


def iter_indexed_sample_batches(
    dataset: HeteroGraphDataset,
    indices: list[int],
    *,
    batch_size: int,
) -> list[list[TargetGraphSample]]:
    batches: list[list[TargetGraphSample]] = []
    for start in range(0, len(indices), batch_size):
        samples = [dataset[idx] for idx in indices[start : start + batch_size]]
        batches.append(samples)
    return batches


def dependency_balanced_epoch_indices(
    *,
    dataset_size: int,
    positive_indices: list[int],
    negative_indices: list[int],
    positive_sample_prob: float,
    epoch_size: int,
    seed: int,
) -> list[int]:
    """Sample scene-target indices with replacement using dependency labels.

    This is a training-only sampler. It uses labels to choose which samples are
    seen more often, but labels are never inserted into model inputs.
    """

    if dataset_size <= 0:
        return []
    rng = random.Random(seed)
    if not positive_indices or not negative_indices:
        indices = list(range(dataset_size))
        rng.shuffle(indices)
        return indices
    p = min(max(float(positive_sample_prob), 0.0), 1.0)
    n = max(int(epoch_size), 1)
    sampled: list[int] = []
    for _ in range(n):
        pool = positive_indices if rng.random() < p else negative_indices
        sampled.append(rng.choice(pool))
    rng.shuffle(sampled)
    return sampled


def collect_labels(dataset: HeteroGraphDataset) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    masks = []
    for sample in iter_sample_batches(dataset, batch_size=8, shuffle=False, seed=0):
        batch = collate_samples(sample)
        if batch["edge_label_og"].size:
            labels.append(batch["edge_label_og"])
            masks.append(batch["edge_label_mask_og"])
    if not labels:
        return np.zeros((0, len(LABEL_NAMES)), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(labels, axis=0), np.concatenate(masks, axis=0)


def collect_label_stats(
    dataset: HeteroGraphDataset,
    *,
    sample_positive_label: str = "dep_progress_any",
) -> dict[str, Any]:
    labels = []
    masks = []
    positive_indices: list[int] = []
    negative_indices: list[int] = []
    label_to_index = {name: idx for idx, name in enumerate(LABEL_NAMES)}
    sample_label_index = int(label_to_index.get(sample_positive_label, 0))
    for idx, ref in enumerate(dataset.refs):
        try:
            sample_labels, sample_masks = fast_edge_labels_for_ref(dataset, ref)
        except Exception:
            sample = dataset[idx]
            sample_labels = sample.edge_label_og
            sample_masks = sample.edge_label_mask_og
        if sample_labels.size:
            labels.append(sample_labels)
            masks.append(sample_masks)
            keep = sample_masks.astype(bool)
            has_positive = bool(keep.sum() > 0 and np.any(sample_labels[keep, sample_label_index] >= 0.5))
        else:
            has_positive = False
        if has_positive:
            positive_indices.append(idx)
        else:
            negative_indices.append(idx)
    if labels:
        label_arr = np.concatenate(labels, axis=0)
        mask_arr = np.concatenate(masks, axis=0)
    else:
        label_arr = np.zeros((0, len(LABEL_NAMES)), dtype=np.float32)
        mask_arr = np.zeros((0,), dtype=np.float32)
    return {
        "labels": label_arr,
        "masks": mask_arr,
        "positive_indices": positive_indices,
        "negative_indices": negative_indices,
        "sample_positive_label": sample_positive_label,
    }


def fast_edge_labels_for_ref(dataset: HeteroGraphDataset, ref: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read object-grasp labels without constructing full visual/geometric features."""

    scene = read_json(ref.scene_path)
    proposals = read_json(ref.proposals_path)
    labels = read_json(ref.labels_path)
    manifest = read_json(ref.manifest_path) if ref.manifest_path else None
    _target_node, object_nodes = extract_graph_support_objects(scene, manifest, ref.target_id)

    hidden_thresh = float(dataset.feature_config.get("hidden_thresh", 0.01))
    visible_thresh = float(dataset.feature_config.get("visible_thresh", 0.30))
    observed_object_ids: list[str] = []
    for obj in object_nodes:
        object_id = str(obj.get("object_id") or obj.get("id") or "")
        if not object_id or object_id == ref.target_id:
            continue
        has_visibility = object_has_visibility_info(obj)
        status = (
            visibility_status_from_object(
                obj,
                hidden_thresh=hidden_thresh,
                visible_thresh=visible_thresh,
                default="visible",
            )
            if has_visibility
            else "visible"
        )
        if status != "fully_hidden":
            observed_object_ids.append(object_id)

    grasp_ids = [str(grasp.get("grasp_id") or "") for grasp in extract_grasps(proposals)]
    grasp_ids = [grasp_id for grasp_id in grasp_ids if grasp_id]
    if not observed_object_ids or not grasp_ids:
        return np.zeros((0, len(LABEL_NAMES)), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    lookup = dependency_label_lookup(labels)
    edge_labels: list[np.ndarray] = []
    edge_masks: list[float] = []
    for object_id in observed_object_ids:
        for grasp_id in grasp_ids:
            label, has_label = lookup.get((object_id, grasp_id), (np.zeros(len(LABEL_NAMES), dtype=np.float32), False))
            edge_labels.append(label)
            edge_masks.append(1.0 if has_label else 0.0)
    return np.vstack(edge_labels).astype(np.float32), np.asarray(edge_masks, dtype=np.float32)


def pos_weight_from_labels(labels: np.ndarray, masks: np.ndarray, *, max_value: float = 50.0) -> np.ndarray:
    if labels.size == 0:
        return np.ones(len(LABEL_NAMES), dtype=np.float32)
    keep = masks.astype(bool)
    if keep.sum() == 0:
        return np.ones(labels.shape[1] if labels.ndim == 2 else len(LABEL_NAMES), dtype=np.float32)
    y = labels[keep]
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    weights = neg / np.maximum(pos, 1.0)
    return np.clip(weights, 1.0, max_value).astype(np.float32)


def pos_weight_from_config(loss_cfg: dict[str, Any]) -> np.ndarray | None:
    explicit = loss_cfg.get("pos_weight_values")
    if explicit is None:
        return None
    if isinstance(explicit, dict):
        values = [float(explicit.get(name, 1.0)) for name in LABEL_NAMES]
    else:
        values = [float(value) for value in explicit]
    if len(values) != len(LABEL_NAMES):
        raise ValueError(f"pos_weight_values must contain {len(LABEL_NAMES)} values, got {len(values)}")
    return np.asarray(values, dtype=np.float32)


def label_weights_from_config(loss_cfg: dict[str, Any]) -> np.ndarray:
    explicit = loss_cfg.get("label_weights")
    if isinstance(explicit, dict):
        return np.asarray([float(explicit.get(name, 1.0)) for name in LABEL_NAMES], dtype=np.float32)
    defaults = {
        "dep_progress_any": float(loss_cfg.get("lambda_progress_any", loss_cfg.get("lambda_any", 1.0))),
        "dep_sufficient": float(loss_cfg.get("lambda_sufficient", 1.0)),
        "dep_approach": float(loss_cfg.get("lambda_app", loss_cfg.get("lambda_approach", 1.0))),
        "dep_lift": float(loss_cfg.get("lambda_lift", 1.0)),
    }
    return np.asarray([defaults.get(name, 1.0) for name in LABEL_NAMES], dtype=np.float32)


def edge_sampling_mask(
    batch: dict[str, Any],
    sampling_cfg: dict[str, Any],
    *,
    seed: int,
) -> np.ndarray:
    """Build a training-only object-grasp edge mask.

    The model still receives the full graph. This mask only decides which
    object-to-grasp edges contribute to the BCE loss, so labels are used only
    for training sampling and never become model inputs.
    """

    labels = np.asarray(batch["edge_label_og"], dtype=np.float32)
    base_mask = np.asarray(batch["edge_label_mask_og"], dtype=np.float32)
    if labels.size == 0 or not bool(sampling_cfg.get("edge_sampling_enabled", False)):
        return base_mask

    label_to_index = {name: idx for idx, name in enumerate(LABEL_NAMES)}
    label_name = str(sampling_cfg.get("edge_positive_label", sampling_cfg.get("sample_positive_label", "dep_progress_any")))
    label_idx = int(label_to_index.get(label_name, 0))
    valid = base_mask.astype(bool)
    positives = np.flatnonzero(valid & (labels[:, label_idx] >= 0.5))
    negatives = np.flatnonzero(valid & (labels[:, label_idx] < 0.5))
    if positives.size == 0 and negatives.size == 0:
        return base_mask

    rng = np.random.default_rng(seed)
    keep = np.zeros_like(base_mask, dtype=bool)
    keep[positives] = True

    if positives.size > 0:
        ratio = float(sampling_cfg.get("edge_negative_ratio", 10.0))
        min_neg = int(sampling_cfg.get("edge_min_negative_edges", 0))
        target_neg = max(min_neg, int(math.ceil(float(positives.size) * max(ratio, 0.0))))
    else:
        target_neg = int(sampling_cfg.get("edge_negative_only_keep_edges", 0))
        if target_neg <= 0:
            target_neg = int(sampling_cfg.get("edge_min_negative_edges", 64))
    target_neg = min(int(target_neg), int(negatives.size))
    if target_neg <= 0:
        return keep.astype(np.float32)

    hard_fraction = min(max(float(sampling_cfg.get("edge_hard_negative_fraction", 0.0)), 0.0), 1.0)
    hard_count = min(target_neg, int(round(target_neg * hard_fraction)))
    selected: list[int] = []
    if hard_count > 0:
        scores = geometry_heuristic_scores_from_batch(
            batch,
            mode=str(sampling_cfg.get("edge_hard_mode", "geometry")),
            distance_scale=float(sampling_cfg.get("edge_hard_distance_scale", 0.020)),
            overlap_weight=float(sampling_cfg.get("edge_hard_overlap_weight", 1.40)),
            swept_iou_weight=float(sampling_cfg.get("edge_hard_swept_iou_weight", 1.20)),
            bias=float(sampling_cfg.get("edge_hard_bias", -1.00)),
        )[:, label_idx]
        hard_order = negatives[np.argsort(scores[negatives])[::-1]]
        selected.extend([int(idx) for idx in hard_order[:hard_count]])

    remaining_count = target_neg - len(selected)
    if remaining_count > 0:
        remaining_pool = np.asarray([idx for idx in negatives.tolist() if idx not in set(selected)], dtype=np.int64)
        if remaining_pool.size > 0:
            chosen = rng.choice(remaining_pool, size=min(remaining_count, remaining_pool.size), replace=False)
            selected.extend([int(idx) for idx in chosen.tolist()])

    keep[np.asarray(selected, dtype=np.int64)] = True
    return keep.astype(np.float32)


def sampling_config_for_epoch(sampling_cfg: dict[str, Any], epoch: int) -> dict[str, Any]:
    """Resolve optional curriculum sampling settings for one epoch."""

    out = deepcopy(sampling_cfg)
    phases = sampling_cfg.get("curriculum") or sampling_cfg.get("phases") or []
    if not isinstance(phases, list):
        return out
    for phase_idx, phase in enumerate(phases):
        if not isinstance(phase, dict):
            continue
        start = int(phase.get("start_epoch", phase.get("start", 1)))
        end = int(phase.get("end_epoch", phase.get("end", 10**9)))
        if start <= epoch <= end:
            out.update({k: v for k, v in phase.items() if k not in {"start", "start_epoch", "end", "end_epoch"}})
            out["active_phase"] = str(phase.get("name", f"phase_{phase_idx}"))
            return out
    return out


def sampling_ever_enabled(sampling_cfg: dict[str, Any]) -> bool:
    if bool(sampling_cfg.get("enabled", False)):
        return True
    phases = sampling_cfg.get("curriculum") or sampling_cfg.get("phases") or []
    if isinstance(phases, list):
        return any(isinstance(phase, dict) and bool(phase.get("enabled", False)) for phase in phases)
    return False


def compute_pos_weight(dataset: HeteroGraphDataset, *, max_value: float = 50.0) -> np.ndarray:
    labels, masks = collect_labels(dataset)
    return pos_weight_from_labels(labels, masks, max_value=max_value)


def metric_value(metrics: dict[str, Any], path: str) -> float:
    cur: Any = metrics
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return -math.inf
    try:
        return float(cur)
    except (TypeError, ValueError):
        return -math.inf


def compact_metrics_for_log(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep epoch logs readable by omitting large per-grasp lists."""

    out = dict(metrics)
    if isinstance(out.get("blocker_sets"), dict):
        blocker_sets = dict(out["blocker_sets"])
        blocker_sets.pop("per_grasp", None)
        out["blocker_sets"] = blocker_sets
    if isinstance(out.get("minimal_blocker_sets"), dict):
        minimal_blocker_sets = dict(out["minimal_blocker_sets"])
        minimal_blocker_sets.pop("per_grasp", None)
        out["minimal_blocker_sets"] = minimal_blocker_sets
    return out


def calibration_thresholds_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    labels = metrics.get("edge", {}).get("labels", {})
    return {
        label: {
            "threshold": float(values.get("best_threshold", 0.5)),
            "best_f1": float(values.get("best_f1", 0.0)),
            "precision": float(values.get("best_precision", 0.0)),
            "recall": float(values.get("best_recall", 0.0)),
        }
        for label, values in labels.items()
    }


def resolve_torch_device(requested: str) -> str:
    require_torch()
    import torch

    if requested.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def make_datasets(cfg: dict[str, Any]) -> tuple[HeteroGraphDataset, HeteroGraphDataset, HeteroGraphDataset, dict[str, int], list[Any], dict[str, list[str]]]:
    dataset_root = Path(cfg["dataset"]["root"])
    output_dir = Path(cfg["output"]["dir"])
    refs = filter_sample_refs(discover_sample_refs(dataset_root), cfg)
    if not refs:
        raise RuntimeError(f"No scene-target samples found under {dataset_root}")
    feature_cfg = cfg.get("features", {})
    full_dataset = HeteroGraphDataset(dataset_root, refs=refs, feature_config=feature_cfg)
    splits = load_or_create_splits(refs, cfg.get("split", {}), output_dir)
    train_ds = HeteroGraphDataset(
        dataset_root,
        refs=refs,
        sample_ids=splits.get("train", []),
        feature_config=feature_cfg,
        class_map=full_dataset.class_map,
    )
    val_ds = HeteroGraphDataset(
        dataset_root,
        refs=refs,
        sample_ids=splits.get("val", []),
        feature_config=feature_cfg,
        class_map=full_dataset.class_map,
    )
    test_ds = HeteroGraphDataset(
        dataset_root,
        refs=refs,
        sample_ids=splits.get("test", []),
        feature_config=feature_cfg,
        class_map=full_dataset.class_map,
    )
    return train_ds, val_ds, test_ds, full_dataset.class_map, refs, splits


def _scene_id_bound(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return f"scene_{value:04d}"
    text = str(value)
    if text.isdigit():
        return f"scene_{int(text):04d}"
    return text


def filter_sample_refs(refs: list[Any], cfg: dict[str, Any]) -> list[Any]:
    """Apply optional scene-level filters before split construction.

    This keeps small controlled experiments, such as the first 200 generated
    scenes, from accidentally discovering later scene-target samples.
    """

    dataset_cfg = cfg.get("dataset", {})
    split_cfg = cfg.get("split", {})
    filter_cfg = dataset_cfg.get("sample_filter") or split_cfg.get("sample_filter") or {}
    if not isinstance(filter_cfg, dict):
        filter_cfg = {}

    first_n = dataset_cfg.get("first_n_scenes", split_cfg.get("first_n_scenes", filter_cfg.get("first_n_scenes")))
    scene_start = _scene_id_bound(
        dataset_cfg.get("scene_id_start", split_cfg.get("scene_id_start", filter_cfg.get("scene_id_start")))
    )
    scene_end = _scene_id_bound(
        dataset_cfg.get("scene_id_end", split_cfg.get("scene_id_end", filter_cfg.get("scene_id_end")))
    )
    explicit_scene_ids = (
        dataset_cfg.get("scene_ids")
        or split_cfg.get("scene_ids")
        or filter_cfg.get("scene_ids")
        or []
    )

    filtered = list(refs)
    if explicit_scene_ids:
        allowed = {_scene_id_bound(item) for item in explicit_scene_ids}
        filtered = [ref for ref in filtered if ref.scene_id in allowed]
    if scene_start is not None:
        filtered = [ref for ref in filtered if ref.scene_id >= scene_start]
    if scene_end is not None:
        filtered = [ref for ref in filtered if ref.scene_id <= scene_end]
    if first_n is not None:
        n = int(first_n)
        if n < 0:
            raise ValueError("first_n_scenes must be non-negative")
        first_scenes = sorted({ref.scene_id for ref in filtered})[:n]
        allowed = set(first_scenes)
        filtered = [ref for ref in filtered if ref.scene_id in allowed]
    return filtered


def first_nonempty_sample(*datasets: HeteroGraphDataset) -> TargetGraphSample:
    for dataset in datasets:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if sample.num_og_edges > 0:
                return sample
    raise RuntimeError("All samples are empty; no object-to-grasp edges are available.")


def batch_to_torch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    require_torch()
    import torch

    tensor_keys_float = [
        "x_obj",
        "x_grasp",
        "edge_attr_oo",
        "edge_attr_gg",
        "edge_attr_og",
        "edge_label_og",
        "edge_label_mask_og",
    ]
    tensor_keys_long = [
        "object_class_ids",
        "grasp_type_ids",
        "object_batch",
        "grasp_batch",
        "edge_index_oo",
        "edge_index_gg",
        "edge_index_og",
        "edge_sample_index",
    ]
    out = dict(batch)
    for key in tensor_keys_float:
        out[key] = torch.as_tensor(batch[key], dtype=torch.float32, device=device)
    for key in tensor_keys_long:
        out[key] = torch.as_tensor(batch[key], dtype=torch.long, device=device)
    return out


def pairwise_dependency_ranking_loss(
    logits: Any,
    targets: Any,
    mask: Any,
    grasp_index: Any,
    *,
    label_names: list[str],
    margin: float = 1.0,
    max_pairs_per_grasp: int = 256,
) -> Any:
    """Rank true blocker edges above non-blocker edges within each grasp.

    This is a training-only objective. It uses labels to compare edges sharing
    the same grasp proposal, but it does not alter graph inputs or evaluation.
    """

    require_torch()
    import torch
    import torch.nn.functional as F

    if logits.numel() == 0:
        return logits.sum()
    label_to_index = {name: idx for idx, name in enumerate(LABEL_NAMES)}
    label_indices = [int(label_to_index[name]) for name in label_names if name in label_to_index]
    if not label_indices:
        label_indices = [0]

    valid = mask.float() > 0.5
    if int(valid.sum().detach().cpu().item()) == 0:
        return logits.sum() * 0.0

    losses: list[Any] = []
    unique_grasps = torch.unique(grasp_index[valid].long())
    for label_idx in label_indices:
        y = targets[:, label_idx] >= 0.5
        for grasp_id in unique_grasps:
            group = valid & (grasp_index.long() == grasp_id)
            pos = group & y
            neg = group & (~y)
            if int(pos.sum().detach().cpu().item()) == 0 or int(neg.sum().detach().cpu().item()) == 0:
                continue
            pos_scores = logits[pos, label_idx].view(-1, 1)
            neg_scores = logits[neg, label_idx].view(1, -1)
            pair_losses = F.softplus(float(margin) - pos_scores + neg_scores).reshape(-1)
            if max_pairs_per_grasp > 0 and pair_losses.numel() > max_pairs_per_grasp:
                _, hard_idx = torch.topk(pair_losses.detach(), k=max_pairs_per_grasp, largest=True)
                pair_losses = pair_losses[hard_idx]
            losses.append(pair_losses.mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def adapt_batch_feature_dims(batch: dict[str, Any], dims: dict[str, Any]) -> dict[str, Any]:
    """Pad/crop feature arrays so older checkpoints still run after schema additions."""

    out = dict(batch)
    key_to_dim = {
        "x_obj": "object_in_dim",
        "x_grasp": "grasp_in_dim",
        "edge_attr_oo": "oo_edge_dim",
        "edge_attr_gg": "gg_edge_dim",
        "edge_attr_og": "og_edge_dim",
    }
    for key, dim_key in key_to_dim.items():
        if key not in out or dim_key not in dims:
            continue
        arr = np.asarray(out[key], dtype=np.float32)
        if arr.ndim != 2:
            continue
        target_dim = int(dims[dim_key])
        current_dim = int(arr.shape[1])
        if current_dim == target_dim:
            continue
        if current_dim > target_dim:
            out[key] = arr[:, :target_dim].astype(np.float32)
        else:
            pad = np.zeros((arr.shape[0], target_dim - current_dim), dtype=np.float32)
            out[key] = np.concatenate([arr, pad], axis=1).astype(np.float32)
    return out


def evaluate_torch(
    model: Any,
    dataset: HeteroGraphDataset,
    *,
    batch_size: int,
    device: str,
    threshold: float,
    include_blocker_sets: bool = True,
    include_minimal_blocker_sets: bool = True,
) -> dict[str, Any]:
    require_torch()
    import torch

    model.eval()
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    edge_sample_ids: list[str] = []
    edge_object_ids: list[str] = []
    edge_grasp_ids: list[str] = []
    planning_labels_by_sample: dict[str, list[dict[str, Any]]] = {}
    with torch.no_grad():
        for samples in iter_sample_batches(dataset, batch_size=batch_size, shuffle=False, seed=0):
            batch_np = collate_samples(samples)
            if batch_np["edge_label_og"].shape[0] == 0:
                continue
            for sample in samples:
                planning_labels_by_sample[sample.sample_id] = list(sample.metadata.get("planning_labels", []))
            batch = batch_to_torch(batch_np, device)
            outputs = model(batch)
            edge_logits = extract_edge_logits(outputs)
            scores = torch.sigmoid(edge_logits).detach().cpu().numpy()
            all_scores.append(scores)
            all_labels.append(batch_np["edge_label_og"])
            all_masks.append(batch_np["edge_label_mask_og"])
            edge_sample_ids.extend(batch_np["edge_sample_ids"])
            edge_object_ids.extend(batch_np["edge_object_ids"])
            edge_grasp_ids.extend(batch_np["edge_grasp_ids"])
    return summarize_predictions(
        all_labels,
        all_scores,
        all_masks,
        edge_sample_ids,
        edge_object_ids,
        edge_grasp_ids,
        threshold,
        planning_labels_by_sample=planning_labels_by_sample,
        include_blocker_sets=include_blocker_sets,
        include_minimal_blocker_sets=include_minimal_blocker_sets,
    )


def evaluate_geometry_heuristic(
    dataset: HeteroGraphDataset,
    *,
    batch_size: int,
    threshold: float,
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
        if batch["edge_label_og"].shape[0] == 0:
            continue
        for sample in samples:
            planning_labels_by_sample[sample.sample_id] = list(sample.metadata.get("planning_labels", []))
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
        labels.append(batch["edge_label_og"])
        masks.append(batch["edge_label_mask_og"])
        edge_sample_ids.extend(batch["edge_sample_ids"])
        edge_object_ids.extend(batch["edge_object_ids"])
        edge_grasp_ids.extend(batch["edge_grasp_ids"])
    return summarize_predictions(
        labels,
        scores,
        masks,
        edge_sample_ids,
        edge_object_ids,
        edge_grasp_ids,
        threshold,
        planning_labels_by_sample=planning_labels_by_sample,
    )


def summarize_predictions(
    labels: list[np.ndarray],
    scores: list[np.ndarray],
    masks: list[np.ndarray],
    edge_sample_ids: list[str],
    edge_object_ids: list[str],
    edge_grasp_ids: list[str],
    threshold: float,
    *,
    planning_labels_by_sample: dict[str, list[dict[str, Any]]] | None = None,
    include_blocker_sets: bool = True,
    include_minimal_blocker_sets: bool = True,
) -> dict[str, Any]:
    if not labels:
        out: dict[str, Any] = {"edge": {"labels": {}, "overall": {}}}
        if include_blocker_sets:
            out["blocker_sets"] = {"count": 0}
        if include_minimal_blocker_sets and planning_labels_by_sample is not None:
            out["minimal_blocker_sets"] = {"all_solved": {"count": 0}, "nontrivial": {"count": 0}, "zero_blocker": {"count": 0}}
        return out
    y_true = np.concatenate(labels, axis=0)
    y_score = np.concatenate(scores, axis=0)
    mask = np.concatenate(masks, axis=0)
    out = {"edge": edge_level_metrics(y_true, y_score, mask, threshold=threshold)}
    if include_blocker_sets:
        out["blocker_sets"] = blocker_set_metrics(
            edge_sample_ids,
            edge_object_ids,
            edge_grasp_ids,
            y_true,
            y_score,
            mask,
            threshold=threshold,
        )
    if include_minimal_blocker_sets and planning_labels_by_sample is not None:
        out["minimal_blocker_sets"] = minimal_blocker_set_metrics(
            edge_sample_ids=edge_sample_ids,
            edge_object_ids=edge_object_ids,
            edge_grasp_ids=edge_grasp_ids,
            y_score=y_score,
            planning_labels_by_sample=planning_labels_by_sample,
            mask=mask,
            threshold=threshold,
            score_label_index=0,
            recall_ks=(1, 2, 3, 5),
            y_true=y_true,
        )
    return out


def train_torch_backend(
    cfg: dict[str, Any],
    train_ds: HeteroGraphDataset,
    val_ds: HeteroGraphDataset,
    test_ds: HeteroGraphDataset,
    class_map: dict[str, int],
    *,
    backend_name: str = "torch",
) -> None:
    require_torch()
    import torch

    output_dir = Path(cfg["output"]["dir"])
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample = first_nonempty_sample(train_ds, val_ds, test_ds)
    print(
        "[hetero_gnn] visibility probe "
        f"{sample.sample_id}: observed={sample.metadata.get('num_observed_objects', sample.num_objects)}"
    )
    model_cfg = cfg.get("model", {})
    num_edge_labels = int(sample.edge_label_og.shape[1] if sample.edge_label_og.ndim == 2 else len(LABEL_NAMES))
    if backend_name == "edge_mlp":
        model = EdgeMLPDependencyBaseline(
            object_in_dim=sample.x_obj.shape[1],
            grasp_in_dim=sample.x_grasp.shape[1],
            og_edge_dim=sample.edge_attr_og.shape[1],
            num_edge_labels=num_edge_labels,
            hidden_dim=int(model_cfg.get("edge_mlp_hidden_dim", model_cfg.get("hidden_dim", 128))),
            decoder_layers=int(model_cfg.get("edge_mlp_layers", model_cfg.get("decoder_layers", 3))),
            dropout=float(model_cfg.get("dropout", 0.10)),
        )
    elif backend_name == "object_only":
        model = ObjectOnlyDependencyBaseline(
            object_in_dim=sample.x_obj.shape[1],
            num_edge_labels=num_edge_labels,
            hidden_dim=int(model_cfg.get("object_only_hidden_dim", model_cfg.get("hidden_dim", 128))),
            decoder_layers=int(model_cfg.get("object_only_layers", model_cfg.get("decoder_layers", 3))),
            dropout=float(model_cfg.get("dropout", 0.10)),
        )
    elif backend_name == "g2n2_style":
        model = G2N2StyleDependencyBaseline(
            object_in_dim=sample.x_obj.shape[1],
            grasp_in_dim=sample.x_grasp.shape[1],
            og_edge_dim=sample.edge_attr_og.shape[1],
            num_edge_labels=num_edge_labels,
            hidden_dim=int(model_cfg.get("g2n2_hidden_dim", model_cfg.get("hidden_dim", 128))),
            message_layers=int(model_cfg.get("g2n2_message_layers", 3)),
            decoder_layers=int(model_cfg.get("g2n2_decoder_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.10)),
            use_og_edge_features=bool(model_cfg.get("g2n2_use_og_edge_features", True)),
        )
    else:
        model = HeteroDependencyGNN(
            object_in_dim=sample.x_obj.shape[1],
            grasp_in_dim=sample.x_grasp.shape[1],
            oo_edge_dim=sample.edge_attr_oo.shape[1],
            gg_edge_dim=sample.edge_attr_gg.shape[1],
            og_edge_dim=sample.edge_attr_og.shape[1],
            num_object_classes=max(1, len(class_map)),
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
        )
    train_cfg = cfg.get("training", {})
    device = resolve_torch_device(str(train_cfg.get("device", "auto")))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(train_cfg.get("epochs", 30))
    batch_size = int(train_cfg.get("batch_size", 4))
    seed = int(train_cfg.get("seed", 7))
    threshold = float(cfg.get("evaluation", {}).get("threshold", 0.5))
    loss_cfg = cfg.get("loss", {})
    label_weights_np = label_weights_from_config(loss_cfg)
    label_weights = torch.as_tensor(label_weights_np, dtype=torch.float32, device=device)
    sampling_cfg = cfg.get("sampling", {})
    sampling_enabled = sampling_ever_enabled(sampling_cfg)
    sample_positive_label = str(sampling_cfg.get("sample_positive_label", "dep_progress_any"))
    train_label_stats: dict[str, Any] | None = None
    positive_indices: list[int] = []
    negative_indices: list[int] = []
    explicit_pos_weight_np = pos_weight_from_config(loss_cfg)
    use_pos_weight = bool(loss_cfg.get("use_pos_weight", True))
    if sampling_enabled or (use_pos_weight and explicit_pos_weight_np is None):
        train_label_stats = collect_label_stats(train_ds, sample_positive_label=sample_positive_label)
    if sampling_enabled:
        assert train_label_stats is not None
        positive_indices = list(train_label_stats["positive_indices"])
        negative_indices = list(train_label_stats["negative_indices"])
        print(
            "[hetero_gnn] dependency-aware sampling enabled: "
            f"positive_label={sample_positive_label} "
            f"positive_samples={len(positive_indices)} negative_only_samples={len(negative_indices)} "
            f"positive_sample_prob={float(sampling_cfg.get('positive_sample_prob', 0.7)):.3f} "
            f"curriculum_phases={len(sampling_cfg.get('curriculum') or sampling_cfg.get('phases') or [])}"
        )
    if use_pos_weight:
        if explicit_pos_weight_np is not None:
            pos_weight_np = explicit_pos_weight_np
        else:
            assert train_label_stats is not None
            pos_weight_np = pos_weight_from_labels(
                train_label_stats["labels"],
                train_label_stats["masks"],
                max_value=float(loss_cfg.get("pos_weight_max", 50.0)),
            )
    else:
        pos_weight_np = np.ones(len(LABEL_NAMES), dtype=np.float32)
    pos_weight = torch.as_tensor(pos_weight_np, dtype=torch.float32, device=device)
    ranking_weight = float(loss_cfg.get("ranking_weight", loss_cfg.get("lambda_rank", 0.0)))
    ranking_label_names_raw = loss_cfg.get("ranking_labels", ["dep_progress_any"])
    if isinstance(ranking_label_names_raw, str):
        ranking_label_names = [ranking_label_names_raw]
    else:
        ranking_label_names = [str(item) for item in ranking_label_names_raw]
    ranking_margin = float(loss_cfg.get("ranking_margin", 1.0))
    ranking_max_pairs = int(loss_cfg.get("ranking_max_pairs_per_grasp", 256))
    ranking_use_sampled_edges = bool(loss_cfg.get("ranking_use_sampled_edges", True))
    if ranking_weight > 0:
        print(
            "[hetero_gnn] pairwise ranking loss enabled: "
            f"weight={ranking_weight:.4f} labels={ranking_label_names} "
            f"margin={ranking_margin:.3f} sampled_edges={ranking_use_sampled_edges}"
        )
    best_score = -math.inf
    best_epoch = -1
    selection_metric = str(train_cfg.get("selection_metric", "edge.labels.dep_progress_any.f1"))
    checkpoint_selection = str(train_cfg.get("checkpoint_selection", "validation")).lower()
    use_validation_selection = len(val_ds) > 0 and checkpoint_selection not in {
        "final",
        "latest",
        "none",
        "no_val",
        "no-validation",
    }
    if not use_validation_selection:
        print("[hetero_gnn] validation selection disabled; final epoch checkpoint will be used.")
    log_path = output_dir / "training_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        ranking_losses = []
        epoch_sampling_cfg = sampling_config_for_epoch(sampling_cfg, epoch)
        epoch_sampling_enabled = bool(epoch_sampling_cfg.get("enabled", False))
        if epoch_sampling_enabled:
            epoch_size = int(epoch_sampling_cfg.get("epoch_size") or len(train_ds))
            epoch_indices = dependency_balanced_epoch_indices(
                dataset_size=len(train_ds),
                positive_indices=positive_indices,
                negative_indices=negative_indices,
                positive_sample_prob=float(epoch_sampling_cfg.get("positive_sample_prob", 0.7)),
                epoch_size=epoch_size,
                seed=seed + epoch,
            )
            epoch_batches = iter_indexed_sample_batches(train_ds, epoch_indices, batch_size=batch_size)
        else:
            epoch_batches = iter_sample_batches(train_ds, batch_size=batch_size, shuffle=True, seed=seed + epoch)
        for samples in epoch_batches:
            batch_np = collate_samples(samples)
            if batch_np["edge_label_og"].shape[0] == 0:
                continue
            edge_loss_mask_np = edge_sampling_mask(
                batch_np,
                epoch_sampling_cfg,
                seed=seed * 1_000_003 + epoch * 10_007 + len(losses),
            )
            batch = batch_to_torch(batch_np, device)
            edge_loss_mask = torch.as_tensor(edge_loss_mask_np, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            edge_logits = extract_edge_logits(outputs)
            loss = multilabel_dependency_loss(
                edge_logits,
                batch["edge_label_og"],
                edge_loss_mask,
                label_weights=label_weights,
                pos_weight=pos_weight,
                focal_gamma=float(loss_cfg.get("focal_gamma", 0.0)),
                consistency_weight=float(loss_cfg.get("consistency_weight", 0.0)),
            )
            epoch_ranking_weight = float(epoch_sampling_cfg.get("ranking_weight", ranking_weight))
            if epoch_ranking_weight > 0:
                ranking_mask = edge_loss_mask if ranking_use_sampled_edges else batch["edge_label_mask_og"]
                rank_loss = pairwise_dependency_ranking_loss(
                    edge_logits,
                    batch["edge_label_og"],
                    ranking_mask,
                    batch["edge_index_og"][1],
                    label_names=ranking_label_names,
                    margin=ranking_margin,
                    max_pairs_per_grasp=ranking_max_pairs,
                )
                loss = loss + epoch_ranking_weight * rank_loss
                ranking_losses.append(float(rank_loss.detach().cpu().item()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 5.0)))
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        if use_validation_selection:
            val_metrics = evaluate_torch(
                model,
                val_ds,
                batch_size=batch_size,
                device=device,
                threshold=threshold,
                include_blocker_sets=False,
                include_minimal_blocker_sets=False,
            )
            score = metric_value(val_metrics, selection_metric)
        else:
            val_metrics = {"disabled": True, "num_samples": len(val_ds)}
            score = float(epoch)
        latest_payload = {
            "backend": backend_name,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model_dims": {
                "object_in_dim": sample.x_obj.shape[1],
                "grasp_in_dim": sample.x_grasp.shape[1],
                "oo_edge_dim": sample.edge_attr_oo.shape[1],
                "gg_edge_dim": sample.edge_attr_gg.shape[1],
                "og_edge_dim": sample.edge_attr_og.shape[1],
                "num_object_classes": max(1, len(class_map)),
                "num_edge_labels": num_edge_labels,
            },
            "model_type": backend_name if backend_name in {"edge_mlp", "g2n2_style", "object_only"} else "hetero_gnn",
            "model_config": model_cfg,
            "class_map": class_map,
            "feature_schema": FEATURE_SCHEMA,
            "config": cfg,
        }
        torch.save(latest_payload, ckpt_dir / "latest.pt")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(latest_payload, ckpt_dir / "best.pt")
        log_item = {
            "epoch": epoch,
            "backend": backend_name,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "ranking_loss": float(np.mean(ranking_losses)) if ranking_losses else 0.0,
            "val": compact_metrics_for_log(val_metrics),
            "selection_metric": selection_metric,
            "checkpoint_selection": "validation" if use_validation_selection else "final",
            "sampling": {
                "enabled": epoch_sampling_enabled,
                "active_phase": str(epoch_sampling_cfg.get("active_phase", "")),
                "positive_sample_prob": float(epoch_sampling_cfg.get("positive_sample_prob", 0.0))
                if epoch_sampling_enabled
                else 0.0,
                "positive_samples": len(positive_indices),
                "negative_only_samples": len(negative_indices),
                "edge_sampling_enabled": bool(epoch_sampling_cfg.get("edge_sampling_enabled", False)),
                "edge_negative_ratio": float(epoch_sampling_cfg.get("edge_negative_ratio", 0.0))
                if bool(epoch_sampling_cfg.get("edge_sampling_enabled", False))
                else 0.0,
                "edge_hard_negative_fraction": float(epoch_sampling_cfg.get("edge_hard_negative_fraction", 0.0))
                if bool(epoch_sampling_cfg.get("edge_sampling_enabled", False))
                else 0.0,
                "ranking_weight": float(epoch_sampling_cfg.get("ranking_weight", ranking_weight)),
            },
            "selection_score": score,
            "best_epoch": best_epoch,
        }
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(log_item) + "\n")
        print(
            f"[epoch {epoch:03d}] loss={log_item['train_loss']:.4f} "
            f"rank={log_item['ranking_loss']:.4f} "
            f"selection_score={score:.4f} best_epoch={best_epoch}"
        )

    if test_ds:
        best_payload = torch.load(ckpt_dir / "best.pt", map_location=device)
        model.load_state_dict(best_payload["model_state"])
        if len(val_ds) > 0:
            val_metrics = evaluate_torch(model, val_ds, batch_size=batch_size, device=device, threshold=threshold)
        else:
            val_metrics = {"disabled": True, "num_samples": 0}
        test_metrics = evaluate_torch(model, test_ds, batch_size=batch_size, device=device, threshold=threshold)
        write_json(output_dir / "val_metrics.json", val_metrics)
        write_json(output_dir / "test_metrics.json", test_metrics)
        if use_validation_selection and len(val_ds) > 0:
            write_json(output_dir / "calibration_thresholds.json", calibration_thresholds_from_metrics(val_metrics))


def evaluate_numpy_model(
    model: NumpyEdgeLogisticBaseline,
    dataset: HeteroGraphDataset,
    *,
    batch_size: int,
    threshold: float,
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
        if batch["edge_label_og"].shape[0] == 0:
            continue
        for sample in samples:
            planning_labels_by_sample[sample.sample_id] = list(sample.metadata.get("planning_labels", []))
        logits = model.predict_logits_from_batch(batch)
        labels.append(batch["edge_label_og"])
        scores.append(sigmoid_np(logits))
        masks.append(batch["edge_label_mask_og"])
        edge_sample_ids.extend(batch["edge_sample_ids"])
        edge_object_ids.extend(batch["edge_object_ids"])
        edge_grasp_ids.extend(batch["edge_grasp_ids"])
    return summarize_predictions(
        labels,
        scores,
        masks,
        edge_sample_ids,
        edge_object_ids,
        edge_grasp_ids,
        threshold,
        planning_labels_by_sample=planning_labels_by_sample,
    )


def train_numpy_backend(cfg: dict[str, Any], train_ds: HeteroGraphDataset, val_ds: HeteroGraphDataset, test_ds: HeteroGraphDataset) -> None:
    output_dir = Path(cfg["output"]["dir"])
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample = first_nonempty_sample(train_ds, val_ds, test_ds)
    print(
        "[hetero_gnn] visibility probe "
        f"{sample.sample_id}: observed={sample.metadata.get('num_observed_objects', sample.num_objects)}"
    )
    input_dim = sample.x_obj.shape[1] + sample.x_grasp.shape[1] + sample.edge_attr_og.shape[1]
    train_cfg = cfg.get("training", {})
    loss_cfg = cfg.get("loss", {})
    batch_size = int(train_cfg.get("batch_size", 4))
    epochs = int(train_cfg.get("epochs", 30))
    seed = int(train_cfg.get("seed", 7))
    threshold = float(cfg.get("evaluation", {}).get("threshold", 0.5))
    lr = float(train_cfg.get("numpy_lr", train_cfg.get("lr", 1e-3)))
    l2 = float(train_cfg.get("weight_decay", 1e-4))
    selection_metric = str(train_cfg.get("selection_metric", "edge.labels.dep_progress_any.f1"))
    label_weights = label_weights_from_config(loss_cfg)
    sampling_cfg = cfg.get("sampling", {})
    sampling_enabled = sampling_ever_enabled(sampling_cfg)
    sample_positive_label = str(sampling_cfg.get("sample_positive_label", "dep_progress_any"))
    train_label_stats: dict[str, Any] | None = None
    positive_indices: list[int] = []
    negative_indices: list[int] = []
    explicit_pos_weight = pos_weight_from_config(loss_cfg)
    use_pos_weight = bool(loss_cfg.get("use_pos_weight", True))
    if sampling_enabled or (use_pos_weight and explicit_pos_weight is None):
        train_label_stats = collect_label_stats(train_ds, sample_positive_label=sample_positive_label)
    if sampling_enabled:
        assert train_label_stats is not None
        positive_indices = list(train_label_stats["positive_indices"])
        negative_indices = list(train_label_stats["negative_indices"])
        print(
            "[numpy_baseline] dependency-aware sampling enabled: "
            f"positive_label={sample_positive_label} "
            f"positive_samples={len(positive_indices)} negative_only_samples={len(negative_indices)} "
            f"positive_sample_prob={float(sampling_cfg.get('positive_sample_prob', 0.7)):.3f}"
        )
    if use_pos_weight:
        if explicit_pos_weight is not None:
            pos_weight = explicit_pos_weight
        else:
            assert train_label_stats is not None
            pos_weight = pos_weight_from_labels(
                train_label_stats["labels"],
                train_label_stats["masks"],
                max_value=float(loss_cfg.get("pos_weight_max", 50.0)),
            )
    else:
        pos_weight = np.ones(len(LABEL_NAMES), dtype=np.float32)
    model = NumpyEdgeLogisticBaseline(input_dim=input_dim, output_dim=sample.edge_label_og.shape[1], seed=seed)

    matrices = []
    for samples in iter_sample_batches(train_ds, batch_size=batch_size, shuffle=False, seed=seed):
        matrices.append(edge_design_matrix(collate_samples(samples)))
    model.fit_standardizer(matrices)

    best_score = -math.inf
    best_epoch = -1
    checkpoint_selection = str(train_cfg.get("checkpoint_selection", "validation")).lower()
    use_validation_selection = len(val_ds) > 0 and checkpoint_selection not in {
        "final",
        "latest",
        "none",
        "no_val",
        "no-validation",
    }
    if not use_validation_selection:
        print("[numpy_baseline] validation selection disabled; final epoch checkpoint will be used.")
    log_path = output_dir / "training_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    for epoch in range(1, epochs + 1):
        losses = []
        epoch_sampling_cfg = sampling_config_for_epoch(sampling_cfg, epoch)
        epoch_sampling_enabled = bool(epoch_sampling_cfg.get("enabled", False))
        if epoch_sampling_enabled:
            epoch_size = int(epoch_sampling_cfg.get("epoch_size") or len(train_ds))
            epoch_indices = dependency_balanced_epoch_indices(
                dataset_size=len(train_ds),
                positive_indices=positive_indices,
                negative_indices=negative_indices,
                positive_sample_prob=float(epoch_sampling_cfg.get("positive_sample_prob", 0.7)),
                epoch_size=epoch_size,
                seed=seed + epoch,
            )
            epoch_batches = iter_indexed_sample_batches(train_ds, epoch_indices, batch_size=batch_size)
        else:
            epoch_batches = iter_sample_batches(train_ds, batch_size=batch_size, shuffle=True, seed=seed + epoch)
        for samples in epoch_batches:
            batch = collate_samples(samples)
            x = edge_design_matrix(batch)
            if x.shape[0] == 0:
                continue
            edge_loss_mask = edge_sampling_mask(
                batch,
                epoch_sampling_cfg,
                seed=seed * 1_000_003 + epoch * 10_007 + len(losses),
            )
            loss = model.partial_fit(
                x,
                batch["edge_label_og"],
                edge_loss_mask,
                lr=lr,
                l2=l2,
                pos_weight=pos_weight,
                label_weights=label_weights,
            )
            losses.append(loss)
        if use_validation_selection:
            val_metrics = evaluate_numpy_model(model, val_ds, batch_size=batch_size, threshold=threshold)
            score = metric_value(val_metrics, selection_metric)
        else:
            val_metrics = {"disabled": True, "num_samples": len(val_ds)}
            score = float(epoch)
        model.save(ckpt_dir / "latest_numpy.npz")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            model.save(ckpt_dir / "best_numpy.npz")
        log_item = {
            "epoch": epoch,
            "backend": "numpy_baseline",
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val": compact_metrics_for_log(val_metrics),
            "selection_metric": selection_metric,
            "checkpoint_selection": "validation" if use_validation_selection else "final",
            "sampling": {
                "enabled": epoch_sampling_enabled,
                "positive_sample_prob": float(epoch_sampling_cfg.get("positive_sample_prob", 0.0))
                if epoch_sampling_enabled
                else 0.0,
                "positive_samples": len(positive_indices),
                "negative_only_samples": len(negative_indices),
                "edge_sampling_enabled": bool(epoch_sampling_cfg.get("edge_sampling_enabled", False)),
                "edge_negative_ratio": float(epoch_sampling_cfg.get("edge_negative_ratio", 0.0))
                if bool(epoch_sampling_cfg.get("edge_sampling_enabled", False))
                else 0.0,
            },
            "selection_score": score,
            "best_epoch": best_epoch,
        }
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(log_item) + "\n")
        print(
            f"[epoch {epoch:03d}] numpy_loss={log_item['train_loss']:.4f} "
            f"val_score={score:.4f} best_epoch={best_epoch}"
        )

    best_model = NumpyEdgeLogisticBaseline.load(ckpt_dir / "best_numpy.npz")
    if len(val_ds) > 0:
        val_metrics = evaluate_numpy_model(best_model, val_ds, batch_size=batch_size, threshold=threshold)
    else:
        val_metrics = {"disabled": True, "num_samples": 0}
    test_metrics = evaluate_numpy_model(best_model, test_ds, batch_size=batch_size, threshold=threshold)
    write_json(output_dir / "val_metrics.json", val_metrics)
    write_json(output_dir / "test_metrics.json", test_metrics)
    if use_validation_selection and len(val_ds) > 0:
        write_json(output_dir / "calibration_thresholds.json", calibration_thresholds_from_metrics(val_metrics))


def run_geometry_heuristic_backend(
    cfg: dict[str, Any],
    train_ds: HeteroGraphDataset,
    val_ds: HeteroGraphDataset,
    test_ds: HeteroGraphDataset,
) -> None:
    output_dir = Path(cfg["output"]["dir"])
    batch_size = int(cfg.get("training", {}).get("batch_size", 4))
    threshold = float(cfg.get("evaluation", {}).get("threshold", 0.5))
    heuristic_cfg = cfg.get("geometry_heuristic", {})
    val_metrics = evaluate_geometry_heuristic(
        val_ds,
        batch_size=batch_size,
        threshold=threshold,
        heuristic_cfg=heuristic_cfg,
    )
    test_metrics = evaluate_geometry_heuristic(
        test_ds,
        batch_size=batch_size,
        threshold=threshold,
        heuristic_cfg=heuristic_cfg,
    )
    write_json(output_dir / "val_metrics.json", val_metrics)
    write_json(output_dir / "test_metrics.json", test_metrics)
    print("[geometry_heuristic] validation edge metrics:")
    print(json.dumps(val_metrics.get("edge", {}), indent=2)[:2000])
    print("[geometry_heuristic] test edge metrics:")
    print(json.dumps(test_metrics.get("edge", {}), indent=2)[:2000])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hetero_gnn.yaml"))
    parser.add_argument(
        "--backend",
        choices=["auto", "torch", "edge_mlp", "g2n2_style", "object_only", "numpy_baseline", "geometry_heuristic"],
        default=None,
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset_root is not None:
        cfg.setdefault("dataset", {})["root"] = str(args.dataset_root)
    if args.output_dir is not None:
        cfg.setdefault("output", {})["dir"] = str(args.output_dir)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    backend = args.backend or str(cfg.get("backend", "auto"))
    if backend == "auto":
        backend = "torch" if TORCH_AVAILABLE else "numpy_baseline"

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, test_ds, class_map, refs, splits = make_datasets(cfg)
    write_json(output_dir / "class_map.json", class_map)
    write_json(output_dir / "feature_schema.json", FEATURE_SCHEMA)
    write_json(output_dir / "config_used.json", cfg)
    print(
        f"Discovered {len(refs)} samples | train={len(train_ds)} val={len(val_ds)} "
        f"test={len(test_ds)} | backend={backend}"
    )
    print(f"Split file contains {sum(len(v) for v in splits.values())} sample ids.")
    if backend in {"torch", "edge_mlp", "g2n2_style", "object_only"}:
        train_torch_backend(cfg, train_ds, val_ds, test_ds, class_map, backend_name=backend)
    elif backend == "numpy_baseline":
        if not TORCH_AVAILABLE:
            print("PyTorch is not installed; running NumPy edge baseline for a runnable local smoke test.")
        train_numpy_backend(cfg, train_ds, val_ds, test_ds)
    elif backend == "geometry_heuristic":
        run_geometry_heuristic_backend(cfg, train_ds, val_ds, test_ds)
    else:
        raise ValueError(f"Unknown backend: {backend}")


if __name__ == "__main__":
    main()
