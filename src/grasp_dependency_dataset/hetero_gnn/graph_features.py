"""Observable feature construction for target-centric dependency graphs.

This module deliberately avoids simulator-only signals such as contact lists,
mesh embeddings, penetration depth, oracle blocker sets, and ground-truth 6D
object pose. Object geometry is derived from detection-style 2D boxes plus a
shared depth image when available. Grasp pose fields are treated as proposal
outputs from an AnyGrasp/SuctionNet-style system and are allowed model inputs.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EPS = 1e-8
VISIBILITY_STATUSES = ("visible", "partially_visible", "fully_hidden")

OBJECT_FEATURE_NAMES = [
    "bbox_cx_norm",
    "bbox_cy_norm",
    "bbox_w_norm",
    "bbox_h_norm",
    "bbox_area_norm",
    "bbox_aspect",
    "bbox_x0_norm",
    "bbox_y0_norm",
    "bbox_x1_norm",
    "bbox_y1_norm",
    "det_conf",
    "visible_ratio",
    "visibility_visible",
    "visibility_partially_visible",
    "visibility_fully_hidden",
    "visible_pixels_norm",
    "touches_border",
    "depth_mean",
    "depth_median",
    "depth_min",
    "depth_std",
    "est_center_x",
    "est_center_y",
    "est_center_z",
    "est_size_x",
    "est_size_y",
    "est_size_z",
    "rel_target_x",
    "rel_target_y",
    "rel_target_z",
    "rel_target_dist",
    "rel_target_bbox_iou",
    "depth_order",
    "class_id_norm",
    "class_hash_sin",
    "class_hash_cos",
]

GRASP_FEATURE_NAMES = [
    "type_parallel_jaw",
    "type_suction",
    "score",
    "pos_x",
    "pos_y",
    "pos_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "approach_x",
    "approach_y",
    "approach_z",
    "lift_x",
    "lift_y",
    "lift_z",
    "jaw_width",
    "suction_radius",
    "closing_x",
    "closing_y",
    "closing_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "contact_rel_x",
    "contact_rel_y",
    "contact_rel_z",
    "pregrasp_offset",
]

OO_EDGE_FEATURE_NAMES = [
    "rel_center_x",
    "rel_center_y",
    "rel_center_z",
    "dist_3d",
    "dist_2d",
    "bbox_iou",
    "delta_u",
    "delta_v",
    "delta_w",
    "delta_h",
    "depth_diff",
    "area_ratio",
    "center_2d_dist",
]

GG_EDGE_FEATURE_NAMES = [
    "rel_pos_x",
    "rel_pos_y",
    "rel_pos_z",
    "position_dist",
    "approach_cos",
    "lift_cos",
    "same_type",
    "score_diff",
    "jaw_width_diff",
    "suction_radius_diff",
]

OG_EDGE_FEATURE_NAMES = [
    "rel_center_to_grasp_x",
    "rel_center_to_grasp_y",
    "rel_center_to_grasp_z",
    "center_to_grasp_dist",
    "dist_to_approach_segment",
    "dist_to_lift_segment",
    "approach_clearance",
    "lift_clearance",
    "approach_overlap_soft",
    "lift_overlap_soft",
    "relative_depth",
    "bbox_to_grasp_u",
    "bbox_to_grasp_v",
    "bbox_to_grasp_uv_dist",
    "bbox_approach_swept_iou",
    "bbox_lift_swept_iou",
    "grasp_type_parallel_jaw",
    "grasp_type_suction",
    "grasp_score",
    "object_radius_est",
    "approach_alignment",
    "lift_alignment",
]


@dataclass(frozen=True)
class ObjectFeatureRecord:
    object_id: str
    asset_name: str
    class_id: int
    visibility_status: str
    visible_ratio: float
    used_as_graph_node: bool
    feature: np.ndarray
    bbox_xyxy: tuple[float, float, float, float]
    bbox_norm: tuple[float, float, float, float, float, float]
    image_size: tuple[int, int]
    depth_stats: tuple[float, float, float, float]
    est_center: np.ndarray
    est_size: np.ndarray
    radius_est: float


@dataclass(frozen=True)
class GraspFeatureRecord:
    grasp_id: str
    grasp_type: str
    type_id: int
    feature: np.ndarray
    position: np.ndarray
    approach_dir: np.ndarray
    lift_dir: np.ndarray
    score: float
    jaw_width: float
    suction_radius: float
    pregrasp_offset: float


def stable_hash_unit(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big", signed=False)
    return float(integer / float(2**64 - 1))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def as_vector(value: Any, length: int, default: float = 0.0) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("position") or value.get("quaternion_wxyz") or []
    if not isinstance(value, (list, tuple, np.ndarray)):
        value = []
    out = [as_float(v, default) for v in list(value)[:length]]
    if len(out) < length:
        out.extend([default] * (length - len(out)))
    return np.asarray(out, dtype=np.float32)


def object_has_visibility_info(obj: dict[str, Any]) -> bool:
    obs = observation_from_object(obj)
    return any(
        key in obs or key in obj
        for key in ("visible_ratio", "visibility_ratio", "visibility_status", "visible_pixels")
    )


def visible_ratio_from_object(obj: dict[str, Any], default: float = 1.0) -> float:
    obs = observation_from_object(obj)
    raw = obs.get("visible_ratio", obs.get("visibility_ratio", obj.get("visible_ratio", obj.get("visibility_ratio"))))
    if raw is not None:
        return as_float(raw, default)
    visible_pixels = obs.get("visible_pixels", obj.get("visible_pixels"))
    if visible_pixels is not None:
        width = as_float(obs.get("image_width", obj.get("image_width")), 0.0)
        height = as_float(obs.get("image_height", obj.get("image_height")), 0.0)
        if width > 0 and height > 0:
            return as_float(visible_pixels, 0.0) / max(width * height, EPS)
    return default


def visibility_status_from_ratio(
    visible_ratio: float,
    *,
    hidden_thresh: float = 0.01,
    visible_thresh: float = 0.30,
) -> str:
    ratio = max(0.0, min(1.0, float(visible_ratio)))
    if ratio < hidden_thresh:
        return "fully_hidden"
    if ratio < visible_thresh:
        return "partially_visible"
    return "visible"


def visibility_status_from_object(
    obj: dict[str, Any],
    *,
    hidden_thresh: float = 0.01,
    visible_thresh: float = 0.30,
    default: str = "visible",
) -> str:
    obs = observation_from_object(obj)
    raw_status = obs.get("visibility_status", obj.get("visibility_status"))
    if raw_status:
        status = str(raw_status).lower()
        if status in VISIBILITY_STATUSES:
            return status
    if not object_has_visibility_info(obj):
        return default
    visible_pixels = obs.get("visible_pixels", obj.get("visible_pixels"))
    if visible_pixels is not None:
        pixels = as_float(visible_pixels, 0.0)
        width = as_float(obs.get("image_width", obj.get("image_width")), 0.0)
        height = as_float(obs.get("image_height", obj.get("image_height")), 0.0)
        ratio = visible_ratio_from_object(obj, default=0.0)
        image_ratio = pixels / max(width * height, EPS) if width > 0 and height > 0 else None
        if pixels <= 0:
            return "fully_hidden"
        # Current generated manifests store visible_ratio as visible image
        # occupancy, not object-surface visibility. In that schema a small
        # positive object should stay observed instead of being called hidden.
        if image_ratio is not None and abs(ratio - image_ratio) < 1e-6:
            return "visible" if ratio >= visible_thresh else "partially_visible"
    return visibility_status_from_ratio(
        visible_ratio_from_object(obj, default=1.0),
        hidden_thresh=hidden_thresh,
        visible_thresh=visible_thresh,
    )


def visibility_one_hot(status: str) -> tuple[float, float, float]:
    status = status if status in VISIBILITY_STATUSES else "visible"
    return tuple(1.0 if status == item else 0.0 for item in VISIBILITY_STATUSES)


def normalize_vector(value: Any, length: int, fallback: Iterable[float]) -> np.ndarray:
    vec = as_vector(value, length)
    norm = float(np.linalg.norm(vec))
    if norm < EPS:
        vec = np.asarray(list(fallback), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
    if norm < EPS:
        return np.zeros(length, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def bbox_iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= EPS:
        return 0.0
    return float(inter / union)


def bbox_from_points(points: np.ndarray, image_size: tuple[int, int], pad_norm: float = 0.025) -> tuple[float, float, float, float]:
    width, height = image_size
    if points.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    u = np.clip(points[:, 0], 0.0, 1.0)
    v = np.clip(points[:, 1], 0.0, 1.0)
    x0 = float(max(0.0, np.min(u) - pad_norm) * width)
    y0 = float(max(0.0, np.min(v) - pad_norm) * height)
    x1 = float(min(1.0, np.max(u) + pad_norm) * width)
    y1 = float(min(1.0, np.max(v) + pad_norm) * height)
    return (x0, y0, x1, y1)


def segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    seg = end - start
    denom = float(np.dot(seg, seg))
    if denom <= EPS:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, seg) / denom)
    t = min(1.0, max(0.0, t))
    closest = start + t * seg
    return float(np.linalg.norm(point - closest))


def _read_png_chunks(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG file")
    pos = 8
    info: dict[str, Any] = {}
    compressed = bytearray()
    while pos < len(raw):
        length = struct.unpack(">I", raw[pos : pos + 4])[0]
        chunk_type = raw[pos + 4 : pos + 8]
        data = raw[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", data
            )
            info.update(
                {
                    "width": width,
                    "height": height,
                    "bit_depth": bit_depth,
                    "color_type": color_type,
                    "compression": compression,
                    "filter_method": filter_method,
                    "interlace": interlace,
                }
            )
        elif chunk_type == b"IDAT":
            compressed.extend(data)
        elif chunk_type == b"IEND":
            break
    return info, bytes(compressed)


def read_png_grayscale(path: Path) -> np.ndarray:
    """Read common non-interlaced PNG files into a float32 grayscale array.

    The project environment used for generation does not necessarily include
    Pillow. This tiny decoder supports the 8/16-bit grayscale/RGB/RGBA PNGs
    emitted by the dataset scripts and keeps graph construction dependency-light.
    """

    info, compressed = _read_png_chunks(path)
    width = int(info["width"])
    height = int(info["height"])
    bit_depth = int(info["bit_depth"])
    color_type = int(info["color_type"])
    interlace = int(info["interlace"])
    if interlace != 0:
        raise ValueError(f"interlaced PNG is not supported: {path}")
    if bit_depth not in (8, 16):
        raise ValueError(f"unsupported PNG bit depth {bit_depth}: {path}")
    channels_by_type = {0: 1, 2: 3, 4: 2, 6: 4}
    if color_type not in channels_by_type:
        raise ValueError(f"unsupported PNG color type {color_type}: {path}")

    channels = channels_by_type[color_type]
    bytes_per_sample = bit_depth // 8
    bpp = channels * bytes_per_sample
    stride = width * bpp
    data = zlib.decompress(compressed)
    rows = []
    prev = bytearray(stride)
    offset = 0
    for _ in range(height):
        filt = data[offset]
        offset += 1
        cur = bytearray(data[offset : offset + stride])
        offset += stride
        for i in range(stride):
            left = cur[i - bpp] if i >= bpp else 0
            up = prev[i]
            up_left = prev[i - bpp] if i >= bpp else 0
            if filt == 0:
                value = cur[i]
            elif filt == 1:
                value = (cur[i] + left) & 0xFF
            elif filt == 2:
                value = (cur[i] + up) & 0xFF
            elif filt == 3:
                value = (cur[i] + ((left + up) // 2)) & 0xFF
            elif filt == 4:
                p = left + up - up_left
                pa = abs(p - left)
                pb = abs(p - up)
                pc = abs(p - up_left)
                predictor = left if pa <= pb and pa <= pc else up if pb <= pc else up_left
                value = (cur[i] + predictor) & 0xFF
            else:
                raise ValueError(f"unsupported PNG filter {filt}: {path}")
            cur[i] = value
        rows.append(bytes(cur))
        prev = cur

    arr = np.frombuffer(b"".join(rows), dtype=np.uint8)
    if bit_depth == 16:
        arr = arr.reshape(height, width, channels, 2)
        values = (arr[..., 0].astype(np.uint16) << 8) | arr[..., 1].astype(np.uint16)
        values = values.astype(np.float32) / 65535.0
    else:
        values = arr.reshape(height, width, channels).astype(np.float32) / 255.0

    if channels == 1:
        gray = values[..., 0]
    elif color_type == 4:
        gray = values[..., 0]
    else:
        gray = values[..., :3].mean(axis=-1)
    return gray.astype(np.float32)


def load_depth_image(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    try:
        return read_png_grayscale(path)
    except Exception:
        return None


def resolve_dataset_path(dataset_root: Path, maybe_path: str | None) -> Path | None:
    if not maybe_path:
        return None
    path = Path(maybe_path)
    if path.is_absolute():
        return path
    candidate = dataset_root / path
    if candidate.exists():
        return candidate
    # Manifests often store paths relative to the repository root.
    repo_candidate = Path.cwd() / path
    if repo_candidate.exists():
        return repo_candidate
    return candidate


def observation_from_object(obj: dict[str, Any]) -> dict[str, Any]:
    obs = obj.get("observation") or {}
    if not isinstance(obs, dict):
        return {}
    return obs


def bbox_record(obj: dict[str, Any], default_size: tuple[int, int] = (640, 480)) -> tuple[
    tuple[float, float, float, float],
    tuple[int, int],
    tuple[float, float, float, float, float, float],
]:
    obs = observation_from_object(obj)
    width = int(as_float(obs.get("image_width"), default_size[0]) or default_size[0])
    height = int(as_float(obs.get("image_height"), default_size[1]) or default_size[1])
    if width <= 0:
        width = default_size[0]
    if height <= 0:
        height = default_size[1]
    bbox = obs.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = [as_float(v) for v in list(bbox)[:4]]
    x0 = min(max(x0, 0.0), float(width))
    x1 = min(max(x1, 0.0), float(width))
    y0 = min(max(y0, 0.0), float(height))
    y1 = min(max(y1, 0.0), float(height))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    bw = max(0.0, x1 - x0)
    bh = max(0.0, y1 - y0)
    cx = x0 + 0.5 * bw
    cy = y0 + 0.5 * bh
    area = bw * bh / max(float(width * height), EPS)
    aspect = bw / max(bh, EPS)
    bbox_norm = (
        cx / width,
        cy / height,
        bw / width,
        bh / height,
        area,
        aspect,
    )
    return (x0, y0, x1, y1), (width, height), bbox_norm


def depth_stats_for_bbox(depth: np.ndarray | None, bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if depth is None or depth.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    h, w = depth.shape[:2]
    x0, y0, x1, y1 = bbox
    ix0 = int(max(0, min(w, math.floor(x0))))
    iy0 = int(max(0, min(h, math.floor(y0))))
    ix1 = int(max(0, min(w, math.ceil(x1))))
    iy1 = int(max(0, min(h, math.ceil(y1))))
    if ix1 <= ix0 or iy1 <= iy0:
        return (0.0, 0.0, 0.0, 0.0)
    crop = depth[iy0:iy1, ix0:ix1].astype(np.float32)
    crop = crop[np.isfinite(crop)]
    if crop.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(np.mean(crop)),
        float(np.median(crop)),
        float(np.min(crop)),
        float(np.std(crop)),
    )


def object_feature_record(
    obj: dict[str, Any],
    target_record: ObjectFeatureRecord | None,
    *,
    class_id: int,
    num_classes: int,
    depth_image: np.ndarray | None,
    det_conf_default: float = 1.0,
    hidden_thresh: float = 0.01,
    visible_thresh: float = 0.30,
    visibility_status_override: str | None = None,
    used_as_graph_node: bool = True,
) -> ObjectFeatureRecord:
    object_id = str(obj.get("object_id") or obj.get("id") or "")
    asset_name = str(obj.get("asset_name") or obj.get("mesh_id") or object_id)
    bbox, image_size, bbox_norm = bbox_record(obj)
    cx, cy, bw, bh, area, aspect = bbox_norm
    depth_mean, depth_median, depth_min, depth_std = depth_stats_for_bbox(depth_image, bbox)

    est_center = np.asarray([cx - 0.5, cy - 0.5, depth_mean], dtype=np.float32)
    est_size = np.asarray([bw, bh, max(depth_std, depth_median - depth_min, 0.0)], dtype=np.float32)
    radius_est = float(max(0.5 * math.sqrt(max(bw * bw + bh * bh, 0.0)), est_size[2], 1e-3))

    if target_record is None:
        rel = np.zeros(3, dtype=np.float32)
        rel_dist = 0.0
        target_iou = 0.0
        depth_order = 0.0
    else:
        rel = est_center - target_record.est_center
        rel_dist = float(np.linalg.norm(rel))
        target_iou = bbox_iou_xyxy(bbox, target_record.bbox_xyxy)
        depth_order = float(depth_mean - target_record.depth_stats[0])

    obs = observation_from_object(obj)
    visible_ratio = visible_ratio_from_object(obj, default=1.0)
    visibility_status = visibility_status_override or visibility_status_from_object(
        obj,
        hidden_thresh=hidden_thresh,
        visible_thresh=visible_thresh,
        default="visible",
    )
    visibility_visible, visibility_partially_visible, visibility_fully_hidden = visibility_one_hot(visibility_status)
    visible_pixels = as_float(obs.get("visible_pixels"), 0.0)
    width, height = image_size
    visible_pixels_norm = visible_pixels / max(float(width * height), EPS)
    det_conf = as_float(obs.get("det_confidence", obj.get("det_confidence")), det_conf_default)
    touches_border = 1.0 if bool(obs.get("touches_border", False)) else 0.0
    x0, y0, x1, y1 = bbox
    x0n, y0n, x1n, y1n = x0 / width, y0 / height, x1 / width, y1 / height
    class_norm = class_id / max(float(num_classes - 1), 1.0)
    class_hash = stable_hash_unit(asset_name)
    class_angle = 2.0 * math.pi * class_hash

    feature = np.asarray(
        [
            cx,
            cy,
            bw,
            bh,
            area,
            aspect,
            x0n,
            y0n,
            x1n,
            y1n,
            det_conf,
            visible_ratio,
            visibility_visible,
            visibility_partially_visible,
            visibility_fully_hidden,
            visible_pixels_norm,
            touches_border,
            depth_mean,
            depth_median,
            depth_min,
            depth_std,
            float(est_center[0]),
            float(est_center[1]),
            float(est_center[2]),
            float(est_size[0]),
            float(est_size[1]),
            float(est_size[2]),
            float(rel[0]),
            float(rel[1]),
            float(rel[2]),
            rel_dist,
            target_iou,
            depth_order,
            class_norm,
            math.sin(class_angle),
            math.cos(class_angle),
        ],
        dtype=np.float32,
    )
    return ObjectFeatureRecord(
        object_id=object_id,
        asset_name=asset_name,
        class_id=class_id,
        visibility_status=visibility_status,
        visible_ratio=visible_ratio,
        used_as_graph_node=used_as_graph_node,
        feature=feature,
        bbox_xyxy=bbox,
        bbox_norm=bbox_norm,
        image_size=image_size,
        depth_stats=(depth_mean, depth_median, depth_min, depth_std),
        est_center=est_center,
        est_size=est_size,
        radius_est=radius_est,
    )


def grasp_type_name(grasp: dict[str, Any], fallback: str = "") -> str:
    value = str(grasp.get("grasp_type") or fallback or "").lower()
    if "suction" in value:
        return "suction"
    if "parallel" in value or "jaw" in value or "gripper" in value:
        return "parallel_jaw"
    source = str(grasp.get("source") or grasp.get("proposal_source") or "").lower()
    if "suction" in source:
        return "suction"
    return "parallel_jaw"


def grasp_feature_record(grasp: dict[str, Any], *, fallback_type: str = "") -> GraspFeatureRecord:
    grasp_id = str(grasp.get("grasp_id") or "")
    gtype = grasp_type_name(grasp, fallback_type)
    type_id = 1 if gtype == "suction" else 0
    type_parallel = 1.0 if type_id == 0 else 0.0
    type_suction = 1.0 if type_id == 1 else 0.0

    pose = grasp.get("pose") or grasp.get("grasp_pose") or {}
    position = as_vector(grasp.get("position") or pose.get("position"), 3)
    quat = as_vector(grasp.get("orientation") or pose.get("quaternion_wxyz"), 4)
    if float(np.linalg.norm(quat)) < EPS:
        quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        quat = (quat / np.linalg.norm(quat)).astype(np.float32)
    approach = normalize_vector(grasp.get("approach_dir") or grasp.get("approach_vector"), 3, (0.0, 0.0, -1.0))
    lift = normalize_vector(grasp.get("lift_dir") or grasp.get("lift_vector"), 3, (0.0, 0.0, 1.0))
    closing = normalize_vector(grasp.get("closing_dir") or grasp.get("closing_vector"), 3, (1.0, 0.0, 0.0))
    normal = normalize_vector(grasp.get("normal") or grasp.get("suction_normal"), 3, (0.0, 0.0, 1.0))
    contact = as_vector(grasp.get("contact_center"), 3)
    contact_rel = contact - position if float(np.linalg.norm(contact)) > EPS else np.zeros(3, dtype=np.float32)
    score = as_float(grasp.get("proposal_score", grasp.get("score_init", grasp.get("score"))), 0.0)
    jaw_width = as_float(grasp.get("jaw_width"), 0.0)
    suction_radius = as_float(grasp.get("suction_radius"), 0.0)
    pregrasp_offset = as_float(grasp.get("pregrasp_offset"), 0.1)

    feature = np.asarray(
        [
            type_parallel,
            type_suction,
            score,
            *position.tolist(),
            *quat.tolist(),
            *approach.tolist(),
            *lift.tolist(),
            jaw_width,
            suction_radius,
            *closing.tolist(),
            *normal.tolist(),
            *contact_rel.tolist(),
            pregrasp_offset,
        ],
        dtype=np.float32,
    )
    return GraspFeatureRecord(
        grasp_id=grasp_id,
        grasp_type=gtype,
        type_id=type_id,
        feature=feature,
        position=position.astype(np.float32),
        approach_dir=approach,
        lift_dir=lift,
        score=score,
        jaw_width=jaw_width,
        suction_radius=suction_radius,
        pregrasp_offset=pregrasp_offset,
    )


def object_object_edge_feature(src: ObjectFeatureRecord, dst: ObjectFeatureRecord) -> np.ndarray:
    rel = dst.est_center - src.est_center
    dist_3d = float(np.linalg.norm(rel))
    dist_2d = float(np.linalg.norm(rel[:2]))
    iou = bbox_iou_xyxy(src.bbox_xyxy, dst.bbox_xyxy)
    scx, scy, sw, sh, sarea, _ = src.bbox_norm
    dcx, dcy, dw, dh, darea, _ = dst.bbox_norm
    area_ratio = (darea + EPS) / (sarea + EPS)
    return np.asarray(
        [
            float(rel[0]),
            float(rel[1]),
            float(rel[2]),
            dist_3d,
            dist_2d,
            iou,
            dcx - scx,
            dcy - scy,
            dw - sw,
            dh - sh,
            dst.depth_stats[0] - src.depth_stats[0],
            area_ratio,
            math.sqrt((dcx - scx) ** 2 + (dcy - scy) ** 2),
        ],
        dtype=np.float32,
    )


def grasp_grasp_edge_feature(src: GraspFeatureRecord, dst: GraspFeatureRecord) -> np.ndarray:
    rel = dst.position - src.position
    return np.asarray(
        [
            float(rel[0]),
            float(rel[1]),
            float(rel[2]),
            float(np.linalg.norm(rel)),
            float(np.dot(src.approach_dir, dst.approach_dir)),
            float(np.dot(src.lift_dir, dst.lift_dir)),
            1.0 if src.type_id == dst.type_id else 0.0,
            dst.score - src.score,
            dst.jaw_width - src.jaw_width,
            dst.suction_radius - src.suction_radius,
        ],
        dtype=np.float32,
    )


def workspace_uv(position: np.ndarray, bin_size: Iterable[float] | None, image_size: tuple[int, int]) -> np.ndarray:
    values = list(bin_size or [])
    sx = as_float(values[0], 0.22) if len(values) >= 1 else 0.22
    sy = as_float(values[1], 0.16) if len(values) >= 2 else 0.16
    u = 0.5 + float(position[0]) / max(sx, EPS)
    v = 0.5 - float(position[1]) / max(sy, EPS)
    return np.asarray([min(1.0, max(0.0, u)), min(1.0, max(0.0, v))], dtype=np.float32)


def object_grasp_edge_feature(
    obj: ObjectFeatureRecord,
    grasp: GraspFeatureRecord,
    *,
    bin_size: Iterable[float] | None,
    lift_distance: float = 0.10,
) -> np.ndarray:
    rel = grasp.position - obj.est_center
    center_dist = float(np.linalg.norm(rel))
    approach_start = grasp.position - grasp.approach_dir * grasp.pregrasp_offset
    approach_end = grasp.position
    lift_start = grasp.position
    lift_end = grasp.position + grasp.lift_dir * lift_distance
    dist_approach = segment_distance(obj.est_center, approach_start, approach_end)
    dist_lift = segment_distance(obj.est_center, lift_start, lift_end)
    radius = obj.radius_est
    approach_clearance = dist_approach - radius
    lift_clearance = dist_lift - radius
    approach_overlap = math.exp(-max(dist_approach, 0.0) / max(radius, 1e-3))
    lift_overlap = math.exp(-max(dist_lift, 0.0) / max(radius, 1e-3))
    rel_depth = float(obj.est_center[2] - grasp.position[2])

    cx, cy, bw, bh, _, _ = obj.bbox_norm
    g_uv = workspace_uv(grasp.position, bin_size, obj.image_size)
    du = float(g_uv[0] - cx)
    dv = float(g_uv[1] - cy)
    uv_dist = math.sqrt(du * du + dv * dv)

    approach_uv0 = workspace_uv(approach_start, bin_size, obj.image_size)
    approach_uv1 = workspace_uv(approach_end, bin_size, obj.image_size)
    lift_uv0 = workspace_uv(lift_start, bin_size, obj.image_size)
    lift_uv1 = workspace_uv(lift_end, bin_size, obj.image_size)
    approach_swept = bbox_from_points(np.vstack([approach_uv0, approach_uv1]), obj.image_size, pad_norm=max(bw, bh, 0.02) * 0.5)
    lift_swept = bbox_from_points(np.vstack([lift_uv0, lift_uv1]), obj.image_size, pad_norm=max(bw, bh, 0.02) * 0.5)
    approach_iou = bbox_iou_xyxy(obj.bbox_xyxy, approach_swept)
    lift_iou = bbox_iou_xyxy(obj.bbox_xyxy, lift_swept)

    rel_norm = np.linalg.norm(rel)
    rel_dir = rel / rel_norm if rel_norm > EPS else np.zeros(3, dtype=np.float32)
    approach_alignment = float(np.dot(rel_dir, grasp.approach_dir))
    lift_alignment = float(np.dot(rel_dir, grasp.lift_dir))
    return np.asarray(
        [
            float(rel[0]),
            float(rel[1]),
            float(rel[2]),
            center_dist,
            dist_approach,
            dist_lift,
            approach_clearance,
            lift_clearance,
            approach_overlap,
            lift_overlap,
            rel_depth,
            du,
            dv,
            uv_dist,
            approach_iou,
            lift_iou,
            1.0 if grasp.type_id == 0 else 0.0,
            1.0 if grasp.type_id == 1 else 0.0,
            grasp.score,
            radius,
            approach_alignment,
            lift_alignment,
        ],
        dtype=np.float32,
    )


def make_directed_knn_edges(
    records: list[Any],
    feature_fn,
    *,
    k: int,
    max_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(records) <= 1:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    centers = []
    for rec in records:
        centers.append(getattr(rec, "est_center", getattr(rec, "position", np.zeros(3, dtype=np.float32))))
    centers_arr = np.asarray(centers, dtype=np.float32)
    edges: list[tuple[int, int]] = []
    attrs: list[np.ndarray] = []
    for i in range(len(records)):
        dists = np.linalg.norm(centers_arr - centers_arr[i], axis=1)
        order = np.argsort(dists)
        kept = 0
        for j in order:
            if int(j) == i:
                continue
            distance = float(dists[j])
            if max_distance is not None and distance > max_distance:
                continue
            edges.append((i, int(j)))
            attrs.append(feature_fn(records[i], records[int(j)]))
            kept += 1
            if kept >= k:
                break
    if not edges:
        dim = len(feature_fn(records[0], records[1]))
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, dim), dtype=np.float32)
    return np.asarray(edges, dtype=np.int64).T, np.vstack(attrs).astype(np.float32)
