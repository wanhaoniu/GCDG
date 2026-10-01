from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def build_transform(rotation_matrix: np.ndarray, translation: Iterable[float]) -> np.ndarray:
    """根据旋转矩阵与平移构造 4x4 齐次变换。"""

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64)
    transform[:3, 3] = np.asarray(list(translation), dtype=np.float64)
    return transform


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """将旋转矩阵转换为 xyzw 四元数。"""

    rotation = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    quaternion /= norm
    return tuple(float(value) for value in quaternion.tolist())


def quaternion_to_rotation_matrix(quaternion: Iterable[float]) -> np.ndarray:
    """将 xyzw 四元数转换为旋转矩阵。"""

    x, y, z, w = [float(value) for value in quaternion]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def transform_pose(
    position: Iterable[float],
    quaternion: Iterable[float],
    transform: np.ndarray,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """将 pose 从一个坐标系转换到另一个坐标系。

    约定 transform 为 T_target_source。
    """

    transform = np.asarray(transform, dtype=np.float64)
    source_rotation = quaternion_to_rotation_matrix(quaternion)
    source_transform = build_transform(source_rotation, position)
    target_transform = transform @ source_transform

    target_position = tuple(float(v) for v in target_transform[:3, 3])
    target_quaternion = rotation_matrix_to_quaternion(target_transform[:3, :3])
    return target_position, target_quaternion


def normalize_vector(vector: Iterable[float], eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(list(vector), dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm < eps:
        raise ValueError("向量范数过小，无法归一化。")
    return vector / norm


def normal_to_quaternion(
    surface_normal: Iterable[float],
    approach_align: str = "-normal",
) -> Tuple[float, float, float, float]:
    """根据吸附表面法向量构造末端姿态。

    统一约定:
    - 末端工具坐标系的 +Z 轴表示接近方向。
    - 当 surface_normal 指向物体外侧时，通常接近方向应取 -normal。
    """

    normal = normalize_vector(surface_normal)
    if approach_align == "-normal":
        z_axis = -normal
    elif approach_align == "+normal":
        z_axis = normal
    else:
        raise ValueError(f"不支持的吸附法向对齐方式: {approach_align}")

    up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(z_axis, up_hint))) > 0.95:
        up_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    x_axis = np.cross(up_hint, z_axis)
    x_axis = normalize_vector(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = normalize_vector(y_axis)

    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    return rotation_matrix_to_quaternion(rotation_matrix)
