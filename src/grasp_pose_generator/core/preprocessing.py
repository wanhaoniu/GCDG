from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..base.types import DetectionResult, SensorFrame


@dataclass
class PreprocessedInput:
    """预处理后的统一中间结果。"""

    depth_m: np.ndarray
    point_cloud_camera: np.ndarray
    roi_mask: np.ndarray
    roi_bbox_xyxy: Tuple[int, int, int, int]
    roi_points_camera: np.ndarray
    roi_colors: np.ndarray
    workspace_lims_camera: Tuple[float, float, float, float, float, float]


def preprocess_sensor_frame(
    sensor_frame: SensorFrame,
    detection_result: DetectionResult,
    roi_padding_pixels: int = 0,
    workspace_padding_m: float = 0.0,
    min_depth_m: float = 0.001,
    max_depth_m: Optional[float] = None,
) -> PreprocessedInput:
    """执行 ROI 裁剪、点云重建与 AnyGrasp/SuctionNet 共用输入准备。"""

    sensor_frame.validate()
    detection_result.validate(sensor_frame.image_shape)

    depth_m = sensor_frame.depth_in_meters()
    points_camera = create_point_cloud_from_depth(depth_m, sensor_frame)

    valid_depth = depth_m > float(min_depth_m)
    if max_depth_m is not None:
        valid_depth &= depth_m < float(max_depth_m)

    metadata = dict(detection_result.metadata or {})
    # `disable_bbox_roi` 只保留给显式的“全场景预计算”请求使用。
    # 常规在线抓取阶段统一使用最终投票得到的 YOLO bbox 作为 ROI。
    disable_bbox_roi = bool(metadata.get('disable_bbox_roi', False))
    if disable_bbox_roi:
        height, width = sensor_frame.image_shape
        roi_bbox = (0, 0, width, height)
        roi_mask = valid_depth.copy()
        roi_mask = _apply_world_anchor_roi_filter(
            roi_mask=roi_mask,
            points_camera=points_camera,
            sensor_frame=sensor_frame,
            detection_metadata=metadata,
        )
        roi_bbox = bbox_from_mask(roi_mask, sensor_frame.image_shape, fallback_bbox=roi_bbox)
    else:
        roi_bbox = resolve_bbox(detection_result, sensor_frame.image_shape, padding=roi_padding_pixels)
        roi_mask = build_roi_mask(detection_result, sensor_frame.image_shape, roi_bbox)
        roi_mask &= valid_depth

    roi_points = points_camera[roi_mask]
    roi_colors = sensor_frame.color[..., :3][roi_mask]
    workspace_lims = compute_workspace_lims(roi_points, padding_m=workspace_padding_m)

    return PreprocessedInput(
        depth_m=depth_m,
        point_cloud_camera=points_camera,
        roi_mask=roi_mask,
        roi_bbox_xyxy=roi_bbox,
        roi_points_camera=roi_points.astype(np.float32),
        roi_colors=roi_colors.astype(np.float32) / 255.0 if roi_colors.size else roi_colors.astype(np.float32),
        workspace_lims_camera=workspace_lims,
    )


def _apply_world_anchor_roi_filter(
    roi_mask: np.ndarray,
    points_camera: np.ndarray,
    sensor_frame: SensorFrame,
    detection_metadata: dict,
) -> np.ndarray:
    """在关闭 bbox 依赖时，按世界坐标锚点裁剪 ROI。

    逻辑说明：
    - 若提供了 `grasp_reference_world_position`，则在 world 坐标系下做半径过滤。
    - 若没有可靠锚点，则回退为“整幅图有效深度”，避免再次被 bbox 卡住。
    """

    anchor_world = detection_metadata.get('grasp_reference_world_position')
    if not isinstance(anchor_world, (list, tuple, np.ndarray)) or len(anchor_world) != 3:
        anchor_world = detection_metadata.get('fused_world_position')
    radius_m = float(detection_metadata.get('grasp_reference_radius_m', 0.0) or 0.0)
    if not isinstance(anchor_world, (list, tuple, np.ndarray)) or len(anchor_world) != 3 or radius_m <= 0.0:
        return roi_mask
    if sensor_frame.camera_to_world is None:
        return roi_mask

    active_points_camera = np.asarray(points_camera[roi_mask], dtype=np.float64).reshape(-1, 3)
    if active_points_camera.size == 0:
        return roi_mask

    transform = np.asarray(sensor_frame.camera_to_world, dtype=np.float64).reshape(4, 4)
    homogeneous = np.concatenate(
        [active_points_camera, np.ones((active_points_camera.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    active_points_world = homogeneous.dot(transform.T)[:, :3]
    anchor = np.asarray(anchor_world, dtype=np.float64).reshape(1, 3)
    keep_mask = np.linalg.norm(active_points_world - anchor, axis=1) <= float(radius_m)

    # 如果锚点过滤后的点一个都没有，说明锚点可能偏了，保留原有效深度 ROI 兜底。
    if not np.any(keep_mask):
        return roi_mask

    refined_mask = np.zeros_like(roi_mask, dtype=bool)
    active_rows, active_cols = np.nonzero(roi_mask)
    refined_mask[active_rows[keep_mask], active_cols[keep_mask]] = True
    return refined_mask


def create_point_cloud_from_depth(depth_m: np.ndarray, sensor_frame: SensorFrame) -> np.ndarray:
    """将对齐深度图反投影到 camera 坐标系点云。"""

    intrinsics = sensor_frame.intrinsics
    height, width = depth_m.shape

    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    z = depth_m.astype(np.float32)
    x = (xx - float(intrinsics.cx)) * z / float(intrinsics.fx)
    y = (yy - float(intrinsics.cy)) * z / float(intrinsics.fy)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def resolve_bbox(
    detection_result: DetectionResult,
    image_shape: Tuple[int, int],
    padding: int = 0,
) -> Tuple[int, int, int, int]:
    """将输入 bbox/mask 统一转换成裁剪框。"""

    height, width = image_shape

    if detection_result.bbox_xyxy is not None:
        x_min, y_min, x_max, y_max = [int(v) for v in detection_result.bbox_xyxy]
    elif detection_result.mask is not None and np.any(detection_result.mask):
        ys, xs = np.nonzero(detection_result.mask)
        x_min, x_max = int(xs.min()), int(xs.max()) + 1
        y_min, y_max = int(ys.min()), int(ys.max()) + 1
    else:
        x_min, y_min, x_max, y_max = 0, 0, width, height

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)

    if x_min >= x_max or y_min >= y_max:
        raise ValueError("经过 padding 后的 ROI bbox 非法。")
    return x_min, y_min, x_max, y_max


def build_roi_mask(
    detection_result: DetectionResult,
    image_shape: Tuple[int, int],
    roi_bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    """优先使用实例 mask；若没有 mask，则退化为 bbox mask。"""

    height, width = image_shape
    roi_mask = np.zeros((height, width), dtype=bool)
    x_min, y_min, x_max, y_max = roi_bbox
    roi_mask[y_min:y_max, x_min:x_max] = True

    if detection_result.mask is None:
        return roi_mask
    return roi_mask & detection_result.mask.astype(bool)


def bbox_from_mask(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
    fallback_bbox: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    """根据布尔 mask 反推出 bbox，便于归档和调试。"""

    if mask is None or not np.any(mask):
        return fallback_bbox

    ys, xs = np.nonzero(mask)
    height, width = image_shape
    x_min = max(0, int(xs.min()))
    x_max = min(width, int(xs.max()) + 1)
    y_min = max(0, int(ys.min()))
    y_max = min(height, int(ys.max()) + 1)
    if x_min >= x_max or y_min >= y_max:
        return fallback_bbox
    return x_min, y_min, x_max, y_max


def compute_workspace_lims(roi_points: np.ndarray, padding_m: float = 0.0) -> Tuple[float, float, float, float, float, float]:
    """根据 ROI 点云自动生成 AnyGrasp 所需 3D 工作空间。"""

    if roi_points.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    mins = roi_points.min(axis=0) - float(padding_m)
    maxs = roi_points.max(axis=0) + float(padding_m)
    return (
        float(mins[0]),
        float(maxs[0]),
        float(mins[1]),
        float(maxs[1]),
        float(mins[2]),
        float(maxs[2]),
    )
