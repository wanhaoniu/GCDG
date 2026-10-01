from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..base.types import DetectionResult, SensorFrame
from .preprocessing import (
    PreprocessedInput,
    bbox_from_mask,
    compute_workspace_lims,
    create_point_cloud_from_depth,
)


SUPPORT_PLANE_APPENDED_KEY = "support_plane_appended"
SUPPORT_PLANE_HEIGHT_WORLD_KEY = "support_plane_height_world"


def estimate_support_plane_height_world(
    *,
    sensor_frame: SensorFrame,
    detection_result: DetectionResult,
    preprocessed_input: PreprocessedInput,
    support_plane_config: Dict[str, Any],
    anchor_world: Optional[Sequence[float]] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    metadata = dict(detection_result.metadata or {})
    cached_height = metadata.get(SUPPORT_PLANE_HEIGHT_WORLD_KEY)
    if _is_vector3(metadata.get("support_plane_anchor_world")) and anchor_world is None:
        anchor_world = metadata.get("support_plane_anchor_world")
    if cached_height is not None:
        try:
            return float(cached_height), {
                "source": "metadata",
                SUPPORT_PLANE_HEIGHT_WORLD_KEY: float(cached_height),
            }
        except Exception:
            pass

    if sensor_frame.camera_to_world is None:
        return None, {"source": "unavailable", "reason": "missing_camera_to_world"}

    point_cloud_camera = np.asarray(preprocessed_input.point_cloud_camera, dtype=np.float32)
    if point_cloud_camera.size == 0:
        return None, {"source": "unavailable", "reason": "empty_point_cloud"}

    depth_m = np.asarray(preprocessed_input.depth_m, dtype=np.float32)
    valid_mask = np.isfinite(depth_m) & (depth_m > 1e-6)
    if int(np.count_nonzero(valid_mask)) <= 0:
        return None, {"source": "unavailable", "reason": "no_valid_depth"}

    valid_points_camera = np.asarray(point_cloud_camera[valid_mask], dtype=np.float32).reshape(-1, 3)
    valid_points_world = _transform_points_camera_to_world(valid_points_camera, sensor_frame.camera_to_world)
    if valid_points_world.shape[0] == 0:
        return None, {"source": "unavailable", "reason": "no_world_points"}

    resolved_anchor_world = _resolve_anchor_world(
        sensor_frame=sensor_frame,
        detection_metadata=metadata,
        preprocessed_input=preprocessed_input,
        fallback_anchor_world=anchor_world,
    )

    candidate_points_world = _select_plane_estimation_points(
        valid_points_world,
        resolved_anchor_world,
        support_plane_config=support_plane_config,
    )
    if candidate_points_world.shape[0] == 0:
        candidate_points_world = valid_points_world

    plane_height_world = _estimate_plane_height_from_world_points(candidate_points_world, support_plane_config)
    if plane_height_world is None:
        return None, {
            "source": "unavailable",
            "reason": "plane_height_estimation_failed",
            "candidate_point_count": int(candidate_points_world.shape[0]),
        }

    result_metadata: Dict[str, Any] = {
        "source": "estimated",
        SUPPORT_PLANE_HEIGHT_WORLD_KEY: float(plane_height_world),
        "candidate_point_count": int(candidate_points_world.shape[0]),
    }
    if resolved_anchor_world is not None:
        result_metadata["support_plane_anchor_world"] = [float(value) for value in resolved_anchor_world.tolist()]
    return float(plane_height_world), result_metadata


def build_support_plane_cloud_world(
    *,
    roi_points_world: np.ndarray,
    detection_metadata: Dict[str, Any],
    support_plane_height_world: float,
    support_plane_config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if not bool(support_plane_config.get("enabled", True)):
        return _empty_points(), _empty_colors(), {"enabled": False, "reason": "disabled"}

    point_array_world = np.asarray(roi_points_world, dtype=np.float32).reshape(-1, 3)
    margin_m = max(0.0, float(support_plane_config.get("xy_margin_m", 0.05)))
    fallback_half_extent_m = max(0.02, float(support_plane_config.get("fallback_half_extent_m", 0.12)))
    grid_step_m = max(0.002, float(support_plane_config.get("grid_step_m", 0.008)))
    max_points = max(256, int(support_plane_config.get("max_points", 12000)))

    reference_radius_m = float(detection_metadata.get("grasp_reference_radius_m", 0.0) or 0.0)
    anchor_world = _first_vector3(
        detection_metadata.get("grasp_reference_world_position"),
        detection_metadata.get("fused_world_position"),
    )
    anchor_xy = None
    if _is_vector3(anchor_world):
        anchor_xy = np.asarray(anchor_world, dtype=np.float32).reshape(3)[:2]

    if point_array_world.shape[0] > 0:
        x_min = float(np.min(point_array_world[:, 0])) - margin_m
        x_max = float(np.max(point_array_world[:, 0])) + margin_m
        y_min = float(np.min(point_array_world[:, 1])) - margin_m
        y_max = float(np.max(point_array_world[:, 1])) + margin_m
    elif anchor_xy is not None:
        extent = max(reference_radius_m, fallback_half_extent_m)
        x_min = float(anchor_xy[0] - extent - margin_m)
        x_max = float(anchor_xy[0] + extent + margin_m)
        y_min = float(anchor_xy[1] - extent - margin_m)
        y_max = float(anchor_xy[1] + extent + margin_m)
    else:
        return _empty_points(), _empty_colors(), {"enabled": False, "reason": "missing_extent"}

    if anchor_xy is not None and reference_radius_m > 0.0:
        extent = max(reference_radius_m, fallback_half_extent_m)
        x_min = min(x_min, float(anchor_xy[0] - extent - margin_m))
        x_max = max(x_max, float(anchor_xy[0] + extent + margin_m))
        y_min = min(y_min, float(anchor_xy[1] - extent - margin_m))
        y_max = max(y_max, float(anchor_xy[1] + extent + margin_m))

    x_span = max(0.0, x_max - x_min)
    y_span = max(0.0, y_max - y_min)
    if x_span <= 1e-6 or y_span <= 1e-6:
        return _empty_points(), _empty_colors(), {"enabled": False, "reason": "degenerate_extent"}

    estimated_point_count = int(np.ceil(x_span / grid_step_m) + 1) * int(np.ceil(y_span / grid_step_m) + 1)
    if estimated_point_count > max_points:
        target_area_per_point = (x_span * y_span) / float(max_points)
        grid_step_m = max(grid_step_m, float(np.sqrt(max(target_area_per_point, 1e-8))))

    xs = np.arange(x_min, x_max + 0.5 * grid_step_m, grid_step_m, dtype=np.float32)
    ys = np.arange(y_min, y_max + 0.5 * grid_step_m, grid_step_m, dtype=np.float32)
    if xs.size == 0 or ys.size == 0:
        return _empty_points(), _empty_colors(), {"enabled": False, "reason": "empty_grid"}

    grid_x, grid_y = np.meshgrid(xs, ys)
    plane_points_world = np.stack(
        [
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            np.full(grid_x.size, float(support_plane_height_world), dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    plane_color = _resolve_plane_color_rgb(support_plane_config)
    plane_colors = np.repeat(plane_color.reshape(1, 3), plane_points_world.shape[0], axis=0).astype(np.float32)
    return plane_points_world, plane_colors, {
        "enabled": True,
        "point_count": int(plane_points_world.shape[0]),
        "grid_step_m": float(grid_step_m),
        "xy_margin_m": float(margin_m),
        SUPPORT_PLANE_HEIGHT_WORLD_KEY: float(support_plane_height_world),
        "extent_xyxy_world": [float(x_min), float(y_min), float(x_max), float(y_max)],
    }


def augment_preprocessed_with_support_plane(
    *,
    sensor_frame: SensorFrame,
    detection_result: DetectionResult,
    preprocessed_input: PreprocessedInput,
    support_plane_config: Dict[str, Any],
    adapter_name: str = "",
    workspace_padding_m: float = 0.0,
) -> Tuple[PreprocessedInput, Dict[str, Any]]:
    metadata = dict(detection_result.metadata or {})
    normalized_adapter_name = str(adapter_name or "").strip().lower()
    if not bool(support_plane_config.get("enabled", True)):
        return preprocessed_input, {"enabled": False, "reason": "disabled"}
    if bool(metadata.get(SUPPORT_PLANE_APPENDED_KEY, False)):
        return preprocessed_input, {"enabled": True, "skipped": "already_appended"}
    if sensor_frame.camera_to_world is None:
        return preprocessed_input, {"enabled": False, "reason": "missing_camera_to_world"}

    support_plane_height_world, plane_estimation_metadata = estimate_support_plane_height_world(
        sensor_frame=sensor_frame,
        detection_result=detection_result,
        preprocessed_input=preprocessed_input,
        support_plane_config=support_plane_config,
    )
    if support_plane_height_world is None:
        return preprocessed_input, {
            "enabled": False,
            "reason": "plane_height_unavailable",
            "plane_estimation": plane_estimation_metadata,
        }

    roi_points_world = _transform_points_camera_to_world(preprocessed_input.roi_points_camera, sensor_frame.camera_to_world)
    plane_points_world, plane_colors, plane_metadata = build_support_plane_cloud_world(
        roi_points_world=roi_points_world,
        detection_metadata=metadata,
        support_plane_height_world=float(support_plane_height_world),
        support_plane_config=support_plane_config,
    )
    if plane_points_world.shape[0] == 0:
        return preprocessed_input, {
            "enabled": False,
            "reason": "plane_cloud_empty",
            "plane_estimation": plane_estimation_metadata,
            "plane_cloud": plane_metadata,
        }

    plane_points_camera = _transform_points_world_to_camera(plane_points_world, sensor_frame.camera_to_world)
    valid_plane_mask = np.isfinite(plane_points_camera).all(axis=1) & (plane_points_camera[:, 2] > 1e-6)
    plane_points_camera = plane_points_camera[valid_plane_mask]
    plane_colors = plane_colors[valid_plane_mask]
    if plane_points_camera.shape[0] == 0:
        return preprocessed_input, {
            "enabled": False,
            "reason": "plane_points_not_visible",
            "plane_estimation": plane_estimation_metadata,
            "plane_cloud": plane_metadata,
        }

    depth_aug = np.asarray(preprocessed_input.depth_m, dtype=np.float32).copy()
    roi_mask_aug = np.asarray(preprocessed_input.roi_mask, dtype=bool).copy()
    synthetic_pixel_mask = np.zeros_like(roi_mask_aug, dtype=bool)

    if (
        normalized_adapter_name == "suctionnet"
        and bool(support_plane_config.get("project_to_depth_for_suctionnet", True))
    ):
        synthetic_pixel_mask = _project_plane_points_to_depth(
            depth_m=depth_aug,
            roi_mask=roi_mask_aug,
            plane_points_camera=plane_points_camera,
            sensor_frame=sensor_frame,
        )

    point_cloud_aug = create_point_cloud_from_depth(depth_aug, sensor_frame)
    color_image_aug = sensor_frame.color[..., :3].astype(np.float32) / 255.0
    if np.any(synthetic_pixel_mask):
        plane_color = _resolve_plane_color_rgb(support_plane_config)
        color_image_aug[synthetic_pixel_mask] = plane_color.reshape(1, 3)

    roi_points_from_depth = point_cloud_aug[roi_mask_aug].astype(np.float32)
    roi_colors_from_depth = color_image_aug[roi_mask_aug].astype(np.float32)
    roi_points_aug = np.concatenate([roi_points_from_depth, plane_points_camera.astype(np.float32)], axis=0)
    roi_colors_aug = np.concatenate([roi_colors_from_depth, plane_colors.astype(np.float32)], axis=0)
    roi_bbox_aug = bbox_from_mask(roi_mask_aug, sensor_frame.image_shape, fallback_bbox=preprocessed_input.roi_bbox_xyxy)
    workspace_lims_aug = compute_workspace_lims(roi_points_aug, padding_m=float(workspace_padding_m))

    augmented = PreprocessedInput(
        depth_m=depth_aug.astype(np.float32),
        point_cloud_camera=point_cloud_aug.astype(np.float32),
        roi_mask=roi_mask_aug,
        roi_bbox_xyxy=roi_bbox_aug,
        roi_points_camera=roi_points_aug.astype(np.float32),
        roi_colors=roi_colors_aug.astype(np.float32),
        workspace_lims_camera=workspace_lims_aug,
    )
    return augmented, {
        "enabled": True,
        "adapter": normalized_adapter_name,
        SUPPORT_PLANE_HEIGHT_WORLD_KEY: float(support_plane_height_world),
        "synthetic_plane_point_count": int(plane_points_camera.shape[0]),
        "synthetic_plane_projected_pixel_count": int(np.count_nonzero(synthetic_pixel_mask)),
        "plane_estimation": plane_estimation_metadata,
        "plane_cloud": plane_metadata,
    }


def _resolve_anchor_world(
    *,
    sensor_frame: SensorFrame,
    detection_metadata: Dict[str, Any],
    preprocessed_input: PreprocessedInput,
    fallback_anchor_world: Optional[Sequence[float]],
) -> Optional[np.ndarray]:
    candidate = _first_vector3(
        fallback_anchor_world,
        detection_metadata.get("grasp_reference_world_position"),
        detection_metadata.get("fused_world_position"),
    )
    if _is_vector3(candidate):
        return np.asarray(candidate, dtype=np.float32).reshape(3)

    roi_points_camera = np.asarray(preprocessed_input.roi_points_camera, dtype=np.float32).reshape(-1, 3)
    if roi_points_camera.shape[0] == 0 or sensor_frame.camera_to_world is None:
        return None
    roi_points_world = _transform_points_camera_to_world(roi_points_camera, sensor_frame.camera_to_world)
    if roi_points_world.shape[0] == 0:
        return None
    return np.median(roi_points_world, axis=0).astype(np.float32)


def _select_plane_estimation_points(
    points_world: np.ndarray,
    anchor_world: Optional[np.ndarray],
    *,
    support_plane_config: Dict[str, Any],
) -> np.ndarray:
    point_array = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    if point_array.shape[0] == 0 or anchor_world is None:
        return point_array

    xy_radius_m = max(0.05, float(support_plane_config.get("estimation_xy_radius_m", 0.30)))
    max_below_anchor_m = max(0.02, float(support_plane_config.get("estimation_max_below_anchor_m", 0.35)))
    max_above_anchor_m = max(0.0, float(support_plane_config.get("estimation_max_above_anchor_m", 0.05)))
    min_point_count = max(24, int(support_plane_config.get("min_points", 60)))

    xy_distance = np.linalg.norm(point_array[:, :2] - anchor_world[:2].reshape(1, 2), axis=1)
    z_relative = point_array[:, 2] - float(anchor_world[2])
    local_mask = (
        (xy_distance <= float(xy_radius_m))
        & (z_relative >= -float(max_below_anchor_m))
        & (z_relative <= float(max_above_anchor_m))
    )
    if int(np.count_nonzero(local_mask)) >= min_point_count:
        return point_array[local_mask]
    return point_array


def _estimate_plane_height_from_world_points(
    points_world: np.ndarray,
    support_plane_config: Dict[str, Any],
) -> Optional[float]:
    point_array = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    if point_array.shape[0] == 0:
        return None

    z_values = np.asarray(point_array[:, 2], dtype=np.float32)
    z_values = z_values[np.isfinite(z_values)]
    if z_values.size == 0:
        return None

    if z_values.size <= 8:
        return float(np.median(z_values))

    bin_size_m = max(0.001, float(support_plane_config.get("estimation_bin_size_m", 0.004)))
    inlier_tolerance_m = max(bin_size_m, float(support_plane_config.get("estimation_inlier_tolerance_m", 0.008)))
    lower_bound = float(np.percentile(z_values, 5.0))
    upper_bound = float(np.percentile(z_values, 60.0))
    if upper_bound <= lower_bound + 1e-6:
        return float(np.median(z_values))

    bin_count = max(12, int(np.ceil((upper_bound - lower_bound) / bin_size_m)))
    hist, edges = np.histogram(z_values, bins=bin_count, range=(lower_bound, upper_bound))
    if hist.size == 0 or int(np.max(hist)) <= 0:
        return float(np.median(z_values))

    mode_index = int(np.argmax(hist))
    mode_center = 0.5 * (edges[mode_index] + edges[mode_index + 1])
    inlier_mask = np.abs(z_values - float(mode_center)) <= float(inlier_tolerance_m)
    if int(np.count_nonzero(inlier_mask)) <= 0:
        return float(mode_center)
    return float(np.median(z_values[inlier_mask]))


def _project_plane_points_to_depth(
    *,
    depth_m: np.ndarray,
    roi_mask: np.ndarray,
    plane_points_camera: np.ndarray,
    sensor_frame: SensorFrame,
) -> np.ndarray:
    intr = sensor_frame.intrinsics
    height, width = depth_m.shape[:2]
    projected_mask = np.zeros((height, width), dtype=bool)
    if plane_points_camera.shape[0] == 0:
        return projected_mask

    z_values = np.asarray(plane_points_camera[:, 2], dtype=np.float32)
    valid_mask = np.isfinite(plane_points_camera).all(axis=1) & (z_values > 1e-6)
    if int(np.count_nonzero(valid_mask)) <= 0:
        return projected_mask

    plane_points_camera = plane_points_camera[valid_mask]
    z_values = z_values[valid_mask]
    cols = np.rint((plane_points_camera[:, 0] * float(intr.fx) / z_values) + float(intr.cx)).astype(np.int32)
    rows = np.rint((plane_points_camera[:, 1] * float(intr.fy) / z_values) + float(intr.cy)).astype(np.int32)
    inside_mask = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    if int(np.count_nonzero(inside_mask)) <= 0:
        return projected_mask

    rows = rows[inside_mask]
    cols = cols[inside_mask]
    z_values = z_values[inside_mask]

    pixel_to_depth: Dict[Tuple[int, int], float] = {}
    for row, col, depth_value in zip(rows.tolist(), cols.tolist(), z_values.tolist()):
        key = (int(row), int(col))
        previous = pixel_to_depth.get(key)
        if previous is None or float(depth_value) < float(previous):
            pixel_to_depth[key] = float(depth_value)

    for (row, col), depth_value in pixel_to_depth.items():
        current_depth = float(depth_m[row, col])
        if not np.isfinite(current_depth) or current_depth <= 1e-6 or current_depth > float(depth_value):
            depth_m[row, col] = float(depth_value)
            roi_mask[row, col] = True
            projected_mask[row, col] = True
    return projected_mask


def _transform_points_camera_to_world(points_camera: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    point_array = np.asarray(points_camera, dtype=np.float32).reshape(-1, 3)
    if point_array.shape[0] == 0:
        return point_array.reshape(0, 3)
    homogeneous = np.concatenate(
        [point_array.astype(np.float64), np.ones((point_array.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    transform = np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4)
    return homogeneous.dot(transform.T)[:, :3].astype(np.float32)


def _transform_points_world_to_camera(points_world: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    point_array = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    if point_array.shape[0] == 0:
        return point_array.reshape(0, 3)
    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4))
    homogeneous = np.concatenate(
        [point_array.astype(np.float64), np.ones((point_array.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    return homogeneous.dot(world_to_camera.T)[:, :3].astype(np.float32)


def _resolve_plane_color_rgb(support_plane_config: Dict[str, Any]) -> np.ndarray:
    raw = support_plane_config.get("color_rgb", [150, 150, 150])
    color = np.asarray(raw, dtype=np.float32).reshape(-1)[:3]
    if color.size < 3:
        color = np.array([150.0, 150.0, 150.0], dtype=np.float32)
    if float(np.max(color)) > 1.0:
        color = color / 255.0
    return np.clip(color, 0.0, 1.0).astype(np.float32)


def _is_vector3(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3


def _first_vector3(*candidates: Any) -> Optional[Sequence[float]]:
    for candidate in candidates:
        if _is_vector3(candidate):
            return candidate
    return None


def _empty_points() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)


def _empty_colors() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)
