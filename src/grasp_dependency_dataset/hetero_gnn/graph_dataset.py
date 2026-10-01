"""Dataset adapter for target-centric heterogeneous dependency graphs."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from grasp_dependency_dataset.hetero_gnn.graph_features import (
    GG_EDGE_FEATURE_NAMES,
    GRASP_FEATURE_NAMES,
    OBJECT_FEATURE_NAMES,
    OG_EDGE_FEATURE_NAMES,
    OO_EDGE_FEATURE_NAMES,
    GraspFeatureRecord,
    ObjectFeatureRecord,
    grasp_feature_record,
    load_depth_image,
    make_directed_knn_edges,
    object_feature_record,
    object_has_visibility_info,
    object_grasp_edge_feature,
    object_object_edge_feature,
    grasp_grasp_edge_feature,
    resolve_dataset_path,
    visible_ratio_from_object,
    visibility_status_from_object,
)


@dataclass(frozen=True)
class SampleRef:
    sample_id: str
    scene_id: str
    target_id: str
    scene_dir: Path
    target_dir: Path
    scene_path: Path
    proposals_path: Path
    labels_path: Path
    manifest_path: Path | None


@dataclass
class TargetGraphSample:
    sample_id: str
    scene_id: str
    target_id: str
    object_ids: list[str]
    grasp_ids: list[str]
    object_class_ids: np.ndarray
    grasp_type_ids: np.ndarray
    x_obj: np.ndarray
    x_grasp: np.ndarray
    edge_index_oo: np.ndarray
    edge_attr_oo: np.ndarray
    edge_index_gg: np.ndarray
    edge_attr_gg: np.ndarray
    edge_index_og: np.ndarray
    edge_attr_og: np.ndarray
    edge_label_og: np.ndarray
    edge_label_mask_og: np.ndarray
    metadata: dict[str, Any]

    @property
    def num_objects(self) -> int:
        return int(self.x_obj.shape[0])

    @property
    def num_grasps(self) -> int:
        return int(self.x_grasp.shape[0])

    @property
    def num_og_edges(self) -> int:
        return int(self.edge_index_og.shape[1])


EDGE_LABEL_NAMES = ["dep_progress_any", "dep_sufficient", "dep_approach", "dep_lift"]


FEATURE_SCHEMA = {
    "object": OBJECT_FEATURE_NAMES,
    "grasp": GRASP_FEATURE_NAMES,
    "object_object_edge": OO_EDGE_FEATURE_NAMES,
    "grasp_grasp_edge": GG_EDGE_FEATURE_NAMES,
    "object_grasp_edge": OG_EDGE_FEATURE_NAMES,
    "labels": EDGE_LABEL_NAMES,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_sample_refs(dataset_root: Path | str) -> list[SampleRef]:
    root = Path(dataset_root)
    scenes_root = root / "scenes" if (root / "scenes").exists() else root
    refs: list[SampleRef] = []
    for labels_path in sorted(scenes_root.glob("scene_*/targets/*/labels.json")):
        target_dir = labels_path.parent
        scene_dir = target_dir.parents[1]
        scene_id = scene_dir.name
        target_id = target_dir.name
        proposals_path = target_dir / "proposals.json"
        scene_path = scene_dir / "scene.json"
        manifest_path = target_dir / "manifest.json"
        if not proposals_path.exists() or not scene_path.exists():
            continue
        refs.append(
            SampleRef(
                sample_id=f"{scene_id}__{target_id}",
                scene_id=scene_id,
                target_id=target_id,
                scene_dir=scene_dir,
                target_dir=target_dir,
                scene_path=scene_path,
                proposals_path=proposals_path,
                labels_path=labels_path,
                manifest_path=manifest_path if manifest_path.exists() else None,
            )
        )
    return refs


def extract_graph_support_objects(scene: dict[str, Any], manifest: dict[str, Any] | None, target_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    support = (manifest or {}).get("graph_support") or {}
    target_node = support.get("target_node")
    object_nodes = support.get("object_nodes")
    if isinstance(target_node, dict) and isinstance(object_nodes, list):
        return target_node, [obj for obj in object_nodes if str(obj.get("object_id") or obj.get("id")) != target_id]

    scene_objects = scene.get("objects") or []
    target = None
    clutter: list[dict[str, Any]] = []
    for obj in scene_objects:
        object_id = str(obj.get("object_id") or obj.get("id") or "")
        if object_id == target_id:
            target = obj
        else:
            clutter.append(obj)
    if target is None:
        target = {"object_id": target_id, "asset_name": target_id}
    return target, clutter


def extract_grasps(proposals: dict[str, Any]) -> list[dict[str, Any]]:
    raw = proposals.get("raw_proposals")
    if isinstance(raw, list) and raw:
        grasps = [dict(item) for item in raw if isinstance(item, dict)]
    else:
        grasps = []
        for item in proposals.get("parallel_grasps") or []:
            if isinstance(item, dict):
                payload = dict(item)
                payload.setdefault("grasp_type", "parallel_jaw")
                grasps.append(payload)
        for item in proposals.get("suction_grasps") or []:
            if isinstance(item, dict):
                payload = dict(item)
                payload.setdefault("grasp_type", "suction")
                grasps.append(payload)

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for grasp in grasps:
        grasp_id = str(grasp.get("grasp_id") or "")
        if not grasp_id or grasp_id in seen:
            continue
        seen.add(grasp_id)
        unique.append(grasp)
    return unique


def dependency_label_lookup(labels: dict[str, Any]) -> dict[tuple[str, str], tuple[np.ndarray, bool]]:
    lookup: dict[tuple[str, str], tuple[np.ndarray, bool]] = {}
    for dep in labels.get("dependencies") or []:
        if not isinstance(dep, dict):
            continue
        object_id = str(dep.get("object_id") or "")
        grasp_id = str(dep.get("grasp_id") or "")
        if not object_id or not grasp_id:
            continue
        app = bool(dep.get("dep_collision_approach", dep.get("dep_approach", False)))
        lift = bool(dep.get("dep_collision_lift", dep.get("dep_lift", False)))
        progress_any = bool(app or lift)
        sufficient = bool(dep.get("dep_any", False))
        lookup[(object_id, grasp_id)] = (
            np.asarray([float(progress_any), float(sufficient), float(app), float(lift)], dtype=np.float32),
            True,
        )
    return lookup


def _object_ids_from_stage_payload(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, dict):
        out: set[str] = set()
        for nested in value.values():
            out.update(_object_ids_from_stage_payload(nested))
        return out
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if item is not None}
    return {str(value)}


def scene_level_split(
    refs: Iterable[SampleRef],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    refs = list(refs)
    scenes = sorted({ref.scene_id for ref in refs})
    rng = random.Random(seed)
    rng.shuffle(scenes)
    n = len(scenes)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        scene_splits = {"train": set(scenes), "val": set(), "test": set()}
    else:
        n_train = max(1, int(round(n * train_ratio)))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_val = max(1, n_val)
        if n_train + n_val >= n:
            n_train = max(1, n - 2 if n >= 3 else n - 1)
            n_val = 1 if n >= 3 else 0
        train_scenes = set(scenes[:n_train])
        val_scenes = set(scenes[n_train : n_train + n_val])
        test_scenes = set(scenes[n_train + n_val :])
        if not test_scenes and n >= 2:
            moved = sorted(val_scenes or train_scenes)[-1]
            val_scenes.discard(moved)
            train_scenes.discard(moved)
            test_scenes.add(moved)
        scene_splits = {"train": train_scenes, "val": val_scenes, "test": test_scenes}

    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for split, split_scenes in scene_splits.items():
        out[split] = [ref.sample_id for ref in refs if ref.scene_id in split_scenes]
    return out


def load_or_create_splits(refs: list[SampleRef], split_cfg: dict[str, Any], output_dir: Path) -> dict[str, list[str]]:
    split_file = split_cfg.get("split_file")
    if split_file:
        path = Path(split_file)
    else:
        path = output_dir / "splits.json"
    if path.exists() and not bool(split_cfg.get("overwrite", False)):
        payload = read_json(path)
        splits = {
            "train": list(payload.get("train", [])),
            "val": list(payload.get("val", [])),
            "test": list(payload.get("test", [])),
        }
        current_ids = {ref.sample_id for ref in refs}
        split_ids = set().union(*(set(values) for values in splits.values()))
        if split_ids == current_ids or bool(split_cfg.get("reuse_existing_if_mismatch", False)):
            return splits
    splits = scene_level_split(
        refs,
        train_ratio=float(split_cfg.get("train_ratio", 0.70)),
        val_ratio=float(split_cfg.get("val_ratio", 0.15)),
        test_ratio=float(split_cfg.get("test_ratio", 0.15)),
        seed=int(split_cfg.get("seed", 7)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return splits


class HeteroGraphDataset:
    """Build one heterogeneous graph per scene-target sample."""

    def __init__(
        self,
        dataset_root: Path | str,
        *,
        sample_ids: Iterable[str] | None = None,
        feature_config: dict[str, Any] | None = None,
        class_map: dict[str, int] | None = None,
        refs: list[SampleRef] | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.feature_config = feature_config or {}
        all_refs = refs if refs is not None else discover_sample_refs(self.dataset_root)
        allowed = set(sample_ids) if sample_ids is not None else None
        self.refs = [ref for ref in all_refs if allowed is None or ref.sample_id in allowed]
        self.ref_by_id = {ref.sample_id: ref for ref in self.refs}
        self.class_map = dict(class_map or self._build_class_map(all_refs))
        self.cache_samples = bool(self.feature_config.get("cache_samples", True))
        self.cache_depth_images = bool(self.feature_config.get("cache_depth_images", True))
        self._sample_cache: dict[str, TargetGraphSample] = {}
        self._depth_cache: dict[str, np.ndarray | None] = {}
        self._warned_missing_visibility = False

    def _build_class_map(self, refs: list[SampleRef]) -> dict[str, int]:
        asset_names: set[str] = set()
        for ref in refs:
            try:
                scene = read_json(ref.scene_path)
                manifest = read_json(ref.manifest_path) if ref.manifest_path else None
                target, objects = extract_graph_support_objects(scene, manifest, ref.target_id)
                for obj in [target, *objects]:
                    asset_names.add(str(obj.get("asset_name") or obj.get("mesh_id") or obj.get("object_id") or "unknown"))
            except Exception:
                continue
        return {name: index for index, name in enumerate(sorted(asset_names))}

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> TargetGraphSample:
        ref = self.refs[index]
        if self.cache_samples and ref.sample_id in self._sample_cache:
            return self._sample_cache[ref.sample_id]
        sample = self.build_sample(ref)
        if self.cache_samples:
            self._sample_cache[ref.sample_id] = sample
        return sample

    def sample_by_id(self, sample_id: str) -> TargetGraphSample:
        if self.cache_samples and sample_id in self._sample_cache:
            return self._sample_cache[sample_id]
        sample = self.build_sample(self.ref_by_id[sample_id])
        if self.cache_samples:
            self._sample_cache[sample_id] = sample
        return sample

    @property
    def num_classes(self) -> int:
        return max(1, len(self.class_map))

    def build_sample(self, ref: SampleRef) -> TargetGraphSample:
        scene = read_json(ref.scene_path)
        proposals = read_json(ref.proposals_path)
        labels = read_json(ref.labels_path)
        manifest = read_json(ref.manifest_path) if ref.manifest_path else None
        target_node, object_nodes = extract_graph_support_objects(scene, manifest, ref.target_id)

        depth_path = None
        shared_obs = ((manifest or {}).get("scene") or {}).get("shared_observations") or {}
        if isinstance(shared_obs, dict):
            depth_path = resolve_dataset_path(self.dataset_root, shared_obs.get("depth_path"))
        if depth_path is None or not depth_path.exists():
            depth_path = ref.scene_dir / "observations" / "shared" / "depth.png"
        if self.cache_depth_images:
            depth_key = str(depth_path)
            if depth_key not in self._depth_cache:
                self._depth_cache[depth_key] = load_depth_image(depth_path)
            depth_image = self._depth_cache[depth_key]
        else:
            depth_image = load_depth_image(depth_path)

        def class_id_for(obj: dict[str, Any]) -> int:
            key = str(obj.get("asset_name") or obj.get("mesh_id") or obj.get("object_id") or "unknown")
            return int(self.class_map.get(key, 0))

        hidden_thresh = float(self.feature_config.get("hidden_thresh", 0.01))
        visible_thresh = float(self.feature_config.get("visible_thresh", 0.30))
        visibility_available = any(object_has_visibility_info(obj) for obj in object_nodes)
        if object_nodes and not visibility_available and not self._warned_missing_visibility:
            print(
                "[hetero_gnn] warning: no visibility fields found in graph_support objects; "
                "defaulting all non-target objects to visible."
            )
            self._warned_missing_visibility = True

        object_visibility: dict[str, dict[str, Any]] = {}
        observed_nodes: list[dict[str, Any]] = []
        fully_hidden_object_ids: list[str] = []
        for obj in object_nodes:
            object_id = str(obj.get("object_id") or obj.get("id") or "")
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
            visible_ratio = visible_ratio_from_object(obj, default=1.0)
            used_as_graph_node = status != "fully_hidden"
            object_visibility[object_id] = {
                "visibility_status": status,
                "visible_ratio": float(visible_ratio),
                "used_as_graph_node": bool(used_as_graph_node),
            }
            if used_as_graph_node:
                observed_nodes.append(obj)
            else:
                fully_hidden_object_ids.append(object_id)

        target_record = object_feature_record(
            target_node,
            None,
            class_id=class_id_for(target_node),
            num_classes=self.num_classes,
            depth_image=depth_image,
            det_conf_default=float(self.feature_config.get("det_conf_default", 1.0)),
            hidden_thresh=hidden_thresh,
            visible_thresh=visible_thresh,
            visibility_status_override=visibility_status_from_object(
                target_node,
                hidden_thresh=hidden_thresh,
                visible_thresh=visible_thresh,
                default="visible",
            )
            if object_has_visibility_info(target_node)
            else "visible",
            used_as_graph_node=False,
        )
        object_records: list[ObjectFeatureRecord] = [
            object_feature_record(
                obj,
                target_record,
                class_id=class_id_for(obj),
                num_classes=self.num_classes,
                depth_image=depth_image,
                det_conf_default=float(self.feature_config.get("det_conf_default", 1.0)),
                hidden_thresh=hidden_thresh,
                visible_thresh=visible_thresh,
                visibility_status_override=object_visibility[str(obj.get("object_id") or obj.get("id") or "")][
                    "visibility_status"
                ],
                used_as_graph_node=True,
            )
            for obj in observed_nodes
        ]
        grasps = extract_grasps(proposals)
        grasp_records: list[GraspFeatureRecord] = [grasp_feature_record(grasp) for grasp in grasps]

        x_obj = (
            np.vstack([rec.feature for rec in object_records]).astype(np.float32)
            if object_records
            else np.zeros((0, len(OBJECT_FEATURE_NAMES)), dtype=np.float32)
        )
        x_grasp = (
            np.vstack([rec.feature for rec in grasp_records]).astype(np.float32)
            if grasp_records
            else np.zeros((0, len(GRASP_FEATURE_NAMES)), dtype=np.float32)
        )
        object_class_ids = np.asarray([rec.class_id for rec in object_records], dtype=np.int64)
        grasp_type_ids = np.asarray([rec.type_id for rec in grasp_records], dtype=np.int64)

        oo_k = int(self.feature_config.get("object_knn_k", 4))
        oo_max_distance = self.feature_config.get("object_edge_max_distance")
        edge_index_oo, edge_attr_oo = make_directed_knn_edges(
            object_records,
            object_object_edge_feature,
            k=oo_k,
            max_distance=float(oo_max_distance) if oo_max_distance is not None else None,
        )
        if edge_attr_oo.shape[1:] == (0,):
            edge_attr_oo = np.zeros((0, len(OO_EDGE_FEATURE_NAMES)), dtype=np.float32)

        gg_k = int(self.feature_config.get("grasp_knn_k", 6))
        gg_max_distance = self.feature_config.get("grasp_edge_max_distance")
        edge_index_gg, edge_attr_gg = make_directed_knn_edges(
            grasp_records,
            grasp_grasp_edge_feature,
            k=gg_k,
            max_distance=float(gg_max_distance) if gg_max_distance is not None else None,
        )
        if edge_attr_gg.shape[1:] == (0,):
            edge_attr_gg = np.zeros((0, len(GG_EDGE_FEATURE_NAMES)), dtype=np.float32)

        label_lookup = dependency_label_lookup(labels)
        og_edges: list[tuple[int, int]] = []
        og_attrs: list[np.ndarray] = []
        og_labels: list[np.ndarray] = []
        og_masks: list[float] = []
        edge_object_ids: list[str] = []
        edge_grasp_ids: list[str] = []
        lift_distance = float(self.feature_config.get("lift_distance", 0.10))
        bin_size = scene.get("bin_size") or self.feature_config.get("bin_size")
        for obj_i, obj in enumerate(object_records):
            for grasp_j, grasp in enumerate(grasp_records):
                og_edges.append((obj_i, grasp_j))
                og_attrs.append(
                    object_grasp_edge_feature(
                        obj,
                        grasp,
                        bin_size=bin_size,
                        lift_distance=lift_distance,
                    )
                )
                label, has_label = label_lookup.get(
                    (obj.object_id, grasp.grasp_id),
                    (np.zeros(len(EDGE_LABEL_NAMES), dtype=np.float32), False),
                )
                og_labels.append(label)
                og_masks.append(1.0 if has_label else 0.0)
                edge_object_ids.append(obj.object_id)
                edge_grasp_ids.append(grasp.grasp_id)

        if og_edges:
            edge_index_og = np.asarray(og_edges, dtype=np.int64).T
            edge_attr_og = np.vstack(og_attrs).astype(np.float32)
            edge_label_og = np.vstack(og_labels).astype(np.float32)
            edge_label_mask_og = np.asarray(og_masks, dtype=np.float32)
        else:
            edge_index_og = np.zeros((2, 0), dtype=np.int64)
            edge_attr_og = np.zeros((0, len(OG_EDGE_FEATURE_NAMES)), dtype=np.float32)
            edge_label_og = np.zeros((0, len(EDGE_LABEL_NAMES)), dtype=np.float32)
            edge_label_mask_og = np.zeros((0,), dtype=np.float32)

        metadata = {
            "depth_path": str(depth_path) if depth_path is not None else "",
            "used_depth_image": bool(depth_image is not None),
            "object_asset_names": [rec.asset_name for rec in object_records],
            "object_visibility": object_visibility,
            "observed_object_ids": [rec.object_id for rec in object_records],
            "fully_hidden_object_ids": fully_hidden_object_ids,
            "num_fully_hidden_objects": len(fully_hidden_object_ids),
            "num_observed_objects": len(object_records),
            "visibility_thresholds": {
                "hidden_thresh": hidden_thresh,
                "visible_thresh": visible_thresh,
            },
            "visibility_available": bool(visibility_available),
            "grasp_types": [rec.grasp_type for rec in grasp_records],
            "edge_object_ids": edge_object_ids,
            "edge_grasp_ids": edge_grasp_ids,
            "planning_labels": labels.get("planning_labels", []),
            "planning_summary": labels.get("planning_summary", {}),
            "feature_policy": {
                "object_pose_used": False,
                "mesh_or_contact_used": False,
                "label_metadata_used_as_input": False,
            },
        }
        return TargetGraphSample(
            sample_id=ref.sample_id,
            scene_id=ref.scene_id,
            target_id=ref.target_id,
            object_ids=[rec.object_id for rec in object_records],
            grasp_ids=[rec.grasp_id for rec in grasp_records],
            object_class_ids=object_class_ids,
            grasp_type_ids=grasp_type_ids,
            x_obj=x_obj,
            x_grasp=x_grasp,
            edge_index_oo=edge_index_oo,
            edge_attr_oo=edge_attr_oo,
            edge_index_gg=edge_index_gg,
            edge_attr_gg=edge_attr_gg,
            edge_index_og=edge_index_og,
            edge_attr_og=edge_attr_og,
            edge_label_og=edge_label_og,
            edge_label_mask_og=edge_label_mask_og,
            metadata=metadata,
        )


def _concat_or_empty(chunks: list[np.ndarray], shape_tail: tuple[int, ...], dtype: Any) -> np.ndarray:
    if not chunks:
        return np.zeros((0, *shape_tail), dtype=dtype)
    return np.concatenate(chunks, axis=0).astype(dtype)


def collate_samples(samples: list[TargetGraphSample]) -> dict[str, Any]:
    x_obj: list[np.ndarray] = []
    x_grasp: list[np.ndarray] = []
    object_class_ids: list[np.ndarray] = []
    grasp_type_ids: list[np.ndarray] = []
    object_batch: list[np.ndarray] = []
    grasp_batch: list[np.ndarray] = []
    edge_index_oo: list[np.ndarray] = []
    edge_attr_oo: list[np.ndarray] = []
    edge_index_gg: list[np.ndarray] = []
    edge_attr_gg: list[np.ndarray] = []
    edge_index_og: list[np.ndarray] = []
    edge_attr_og: list[np.ndarray] = []
    edge_label_og: list[np.ndarray] = []
    edge_label_mask_og: list[np.ndarray] = []
    edge_sample_index: list[np.ndarray] = []
    edge_object_ids: list[str] = []
    edge_grasp_ids: list[str] = []
    edge_sample_ids: list[str] = []
    grasp_sample_ids: list[str] = []
    grasp_ids_flat: list[str] = []

    obj_offset = 0
    grasp_offset = 0
    for sample_i, sample in enumerate(samples):
        x_obj.append(sample.x_obj)
        x_grasp.append(sample.x_grasp)
        object_class_ids.append(sample.object_class_ids)
        grasp_type_ids.append(sample.grasp_type_ids)
        object_batch.append(np.full(sample.num_objects, sample_i, dtype=np.int64))
        grasp_batch.append(np.full(sample.num_grasps, sample_i, dtype=np.int64))
        grasp_sample_ids.extend([sample.sample_id] * sample.num_grasps)
        grasp_ids_flat.extend(sample.grasp_ids)

        if sample.edge_index_oo.size:
            idx = sample.edge_index_oo.copy()
            idx += obj_offset
            edge_index_oo.append(idx)
            edge_attr_oo.append(sample.edge_attr_oo)
        if sample.edge_index_gg.size:
            idx = sample.edge_index_gg.copy()
            idx += grasp_offset
            edge_index_gg.append(idx)
            edge_attr_gg.append(sample.edge_attr_gg)
        if sample.edge_index_og.size:
            idx = sample.edge_index_og.copy()
            idx[0] += obj_offset
            idx[1] += grasp_offset
            edge_index_og.append(idx)
            edge_attr_og.append(sample.edge_attr_og)
            edge_label_og.append(sample.edge_label_og)
            edge_label_mask_og.append(sample.edge_label_mask_og)
            edge_sample_index.append(np.full(sample.num_og_edges, sample_i, dtype=np.int64))
            edge_object_ids.extend(sample.metadata.get("edge_object_ids", []))
            edge_grasp_ids.extend(sample.metadata.get("edge_grasp_ids", []))
            edge_sample_ids.extend([sample.sample_id] * sample.num_og_edges)

        obj_offset += sample.num_objects
        grasp_offset += sample.num_grasps

    return {
        "sample_ids": [sample.sample_id for sample in samples],
        "scene_ids": [sample.scene_id for sample in samples],
        "target_ids": [sample.target_id for sample in samples],
        "object_ids": [sample.object_ids for sample in samples],
        "grasp_ids": [sample.grasp_ids for sample in samples],
        "x_obj": _concat_or_empty(x_obj, (len(OBJECT_FEATURE_NAMES),), np.float32),
        "x_grasp": _concat_or_empty(x_grasp, (len(GRASP_FEATURE_NAMES),), np.float32),
        "object_class_ids": _concat_or_empty(object_class_ids, (), np.int64),
        "grasp_type_ids": _concat_or_empty(grasp_type_ids, (), np.int64),
        "object_batch": _concat_or_empty(object_batch, (), np.int64),
        "grasp_batch": _concat_or_empty(grasp_batch, (), np.int64),
        "edge_index_oo": np.concatenate(edge_index_oo, axis=1).astype(np.int64)
        if edge_index_oo
        else np.zeros((2, 0), dtype=np.int64),
        "edge_attr_oo": _concat_or_empty(edge_attr_oo, (len(OO_EDGE_FEATURE_NAMES),), np.float32),
        "edge_index_gg": np.concatenate(edge_index_gg, axis=1).astype(np.int64)
        if edge_index_gg
        else np.zeros((2, 0), dtype=np.int64),
        "edge_attr_gg": _concat_or_empty(edge_attr_gg, (len(GG_EDGE_FEATURE_NAMES),), np.float32),
        "edge_index_og": np.concatenate(edge_index_og, axis=1).astype(np.int64)
        if edge_index_og
        else np.zeros((2, 0), dtype=np.int64),
        "edge_attr_og": _concat_or_empty(edge_attr_og, (len(OG_EDGE_FEATURE_NAMES),), np.float32),
        "edge_label_og": _concat_or_empty(edge_label_og, (len(EDGE_LABEL_NAMES),), np.float32),
        "edge_label_mask_og": _concat_or_empty(edge_label_mask_og, (), np.float32),
        "edge_sample_index": _concat_or_empty(edge_sample_index, (), np.int64),
        "edge_object_ids": edge_object_ids,
        "edge_grasp_ids": edge_grasp_ids,
        "edge_sample_ids": edge_sample_ids,
        "grasp_sample_ids": grasp_sample_ids,
        "grasp_ids_flat": grasp_ids_flat,
    }
