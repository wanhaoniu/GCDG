from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from ..base.types import CameraIntrinsics, DetectionResult, SensorFrame


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_camera_intrinsics(intrinsics_path: str, camera_name: str) -> CameraIntrinsics:
    """兼容 data_capturer 输出的相机内参格式。"""

    payload = load_json(intrinsics_path)

    if "intrinsics" in payload and "cameras" in payload["intrinsics"]:
        camera_payload = payload["intrinsics"]["cameras"][camera_name]
    else:
        camera_payload = payload

    return CameraIntrinsics(
        width=int(camera_payload["width"]),
        height=int(camera_payload["height"]),
        fx=float(camera_payload["fx"]),
        fy=float(camera_payload["fy"]),
        cx=float(camera_payload["cx"]),
        cy=float(camera_payload["cy"]),
        depth_scale=float(camera_payload.get("depth_scale", 1.0)),
        distortion_model=str(camera_payload.get("distortion_model", "")),
        distortion_coeffs=tuple(camera_payload.get("distortion_coeffs", [])),
    )


def load_camera_to_world(extrinsics_path: Optional[str]) -> Optional[np.ndarray]:
    """读取 camera_to_world 变换。

    当前优先兼容 data_capturer/handeye_calib 输出的 result.json:
    - transform_name == T_base_camera
    - transform_matrix 即视为 T_world_camera
    - transform_name == T_camera_base 时，会自动求逆后转成 T_world_camera
    """

    if not extrinsics_path:
        return None

    payload = load_json(extrinsics_path)
    if "transform_matrix" in payload:
        transform = np.asarray(payload["transform_matrix"], dtype=np.float64)
        transform_name = str(payload.get("transform_name", "")).strip()
        if transform_name == "T_camera_base":
            return np.linalg.inv(transform)
        return transform
    if "camera_to_world" in payload:
        return np.asarray(payload["camera_to_world"], dtype=np.float64)
    raise KeyError(f"无法从外参文件中找到 camera_to_world: {extrinsics_path}")


def load_sensor_frame_from_files(
    color_path: str,
    depth_path: str,
    intrinsics_path: str,
    camera_name: str,
    extrinsics_path: Optional[str] = None,
    camera_frame_id: str = "camera",
    world_frame_id: str = "world",
    metadata: Optional[Dict[str, Any]] = None,
) -> SensorFrame:
    """从本地 RGB-D 与相机参数文件构造 SensorFrame。"""

    color = np.array(Image.open(color_path))
    depth = np.array(Image.open(depth_path))
    intrinsics = load_camera_intrinsics(intrinsics_path, camera_name)
    camera_to_world = load_camera_to_world(extrinsics_path)

    return SensorFrame(
        color=color,
        depth=depth,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        camera_frame_id=camera_frame_id,
        world_frame_id=world_frame_id,
        metadata=metadata or {},
    )


def load_detection_from_json(path: str) -> DetectionResult:
    """从 JSON 文件加载检测结果。"""

    payload = load_json(path)
    bbox = payload.get("bbox_xyxy") or payload.get("bbox")
    if bbox is not None:
        bbox = tuple(int(v) for v in bbox)

    return DetectionResult(
        bbox_xyxy=bbox,
        label=str(payload.get("label", "")),
        detector_name=str(payload.get("detector_name", "")),
        score=None if payload.get("score") is None else float(payload["score"]),
        metadata={key: value for key, value in payload.items() if key not in {"bbox_xyxy", "bbox", "label", "detector_name", "score"}},
    )
