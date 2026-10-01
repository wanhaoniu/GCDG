from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class CameraIntrinsics:
    """相机内参。

    约定:
    - 所有深度反投影均在 camera 坐标系下完成。
    - depth_scale 表示 depth 原始数值乘上该值后得到米。
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 1.0
    distortion_model: str = ""
    distortion_coeffs: Sequence[float] = field(default_factory=tuple)

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("相机分辨率必须为正整数。")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("相机焦距 fx/fy 必须大于 0。")
        if self.depth_scale <= 0:
            raise ValueError("depth_scale 必须大于 0。")


@dataclass
class SensorFrame:
    """统一传感器帧输入。

    约定:
    - color/depth 必须是同一时刻、同一坐标对齐后的 RGB-D 数据。
    - depth 可以是原始深度图(u16)或已转换到米的浮点图，统一由
      CameraIntrinsics.depth_scale 负责解释。
    - camera_to_world 若存在，表示 T_world_camera，即 camera 坐标到
      world/robot_base 坐标的 4x4 齐次变换矩阵。
    """

    color: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics
    camera_to_world: Optional[np.ndarray] = None
    camera_frame_id: str = "camera"
    world_frame_id: str = "world"
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.intrinsics.validate()

        if self.color is None or self.depth is None:
            raise ValueError("SensorFrame 的 color/depth 不能为空。")
        if self.color.ndim != 3 or self.color.shape[2] not in (3, 4):
            raise ValueError("color 必须为 HxWx3 或 HxWx4 图像。")
        if self.depth.ndim != 2:
            raise ValueError("depth 必须为 HxW 单通道图像。")
        if self.color.shape[0] != self.depth.shape[0] or self.color.shape[1] != self.depth.shape[1]:
            raise ValueError("color 与 depth 的分辨率必须一致。")

        if self.camera_to_world is not None:
            transform = np.asarray(self.camera_to_world, dtype=np.float64)
            if transform.shape != (4, 4):
                raise ValueError("camera_to_world 必须是 4x4 齐次变换矩阵。")

    @property
    def image_shape(self) -> Tuple[int, int]:
        return int(self.depth.shape[0]), int(self.depth.shape[1])

    def depth_in_meters(self) -> np.ndarray:
        """统一输出米制深度图。"""

        depth = np.asarray(self.depth)
        if np.issubdtype(depth.dtype, np.integer):
            return depth.astype(np.float32) * float(self.intrinsics.depth_scale)

        depth_m = depth.astype(np.float32)
        if float(np.nanmax(depth_m)) > 20.0:
            depth_m *= float(self.intrinsics.depth_scale)
        return depth_m


@dataclass
class DetectionResult:
    """统一检测结果输入。

    bbox_xyxy 采用半开区间:
    - [x_min, y_min, x_max, y_max]
    - 实际切片时对应 image[y_min:y_max, x_min:x_max]
    """

    bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
    mask: Optional[np.ndarray] = None
    label: str = ""
    detector_name: str = ""
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_region(self) -> bool:
        return self.bbox_xyxy is not None or self.mask is not None

    def validate(self, image_shape: Tuple[int, int]) -> None:
        height, width = image_shape
        if self.bbox_xyxy is not None:
            x_min, y_min, x_max, y_max = [int(v) for v in self.bbox_xyxy]
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("bbox_xyxy 非法，必须满足 x_min < x_max 且 y_min < y_max。")
            if x_max <= 0 or y_max <= 0 or x_min >= width or y_min >= height:
                raise ValueError("bbox_xyxy 完全超出图像范围。")
        if self.mask is not None and self.mask.shape != image_shape:
            raise ValueError("DetectionResult.mask 的尺寸必须与输入图像一致。")


@dataclass
class GraspCandidate:
    """统一抓取候选位姿。

    约定:
    - position 单位为米。
    - quaternion 使用 xyzw 顺序。
    - pose_frame 显式标注该位姿处于 camera 还是 world 坐标系。
    """

    position: Tuple[float, float, float]
    quaternion: Tuple[float, float, float, float]
    score: float
    grasp_type: str
    end_effector_id: str
    source_algorithm: str
    pose_frame: str = "camera"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": [float(v) for v in self.position],
            "quaternion": [float(v) for v in self.quaternion],
            "score": float(self.score),
            "grasp_type": self.grasp_type,
            "end_effector_id": self.end_effector_id,
            "source_algorithm": self.source_algorithm,
            "pose_frame": self.pose_frame,
            "metadata": _to_serializable(self.metadata),
        }


@dataclass
class GraspGenerationResult:
    """统一抓取生成结果。"""

    success: bool
    candidates: List[GraspCandidate]
    best_candidate: Optional[GraspCandidate]
    status_code: str
    message: str = ""
    source_algorithm: str = ""
    retryable: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "best_candidate": None if self.best_candidate is None else self.best_candidate.to_dict(),
            "status_code": self.status_code,
            "message": self.message,
            "source_algorithm": self.source_algorithm,
            "retryable": bool(self.retryable),
            "metadata": _to_serializable(self.metadata),
        }


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
