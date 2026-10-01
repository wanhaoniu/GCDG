from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from ..base.interfaces import GraspAlgorithmAdapter
from ..base.status_codes import StatusCode
from ..base.types import DetectionResult, GraspCandidate, GraspGenerationResult, SensorFrame
from ..core.compat import ensure_sys_path, patch_numpy_legacy_aliases
from ..core.preprocessing import PreprocessedInput
from ..core.transforms import normal_to_quaternion


@dataclass
class _CameraInfo:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    scale: float


class SuctionNetAdapter(GraspAlgorithmAdapter):
    """SuctionNet 吸附抓取适配器。

    当前默认接入仓库内可直接运行的 normal_std 基线，后续可在同一接口下继续扩展
    到 SuctionNet 的学习型推理分支。
    """

    adapter_name = "suctionnet"
    source_algorithm = "SuctionNet"
    grasp_type = "suction"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def is_available(self) -> Tuple[bool, str]:
        root = Path(self.config.get("paths", {}).get("suctionnet_root", ""))
        if not root.is_dir():
            return False, f"SuctionNet 根目录不存在: {root}"
        if not (root / "normal_std" / "policy.py").is_file():
            return False, f"SuctionNet normal_std policy 不存在: {root / 'normal_std' / 'policy.py'}"
        return True, "ok"

    def generate(
        self,
        sensor_frame: SensorFrame,
        detection_result: DetectionResult,
        end_effector_id: str,
        preprocessed_input: PreprocessedInput,
        algorithm_config: Dict[str, Any],
    ) -> GraspGenerationResult:
        if preprocessed_input.roi_points_camera.size == 0:
            return GraspGenerationResult(
                success=False,
                candidates=[],
                best_candidate=None,
                status_code=StatusCode.NO_VALID_ROI_POINTS,
                message="ROI 点云为空，SuctionNet 无法执行。",
                source_algorithm=self.source_algorithm,
                retryable=False,
            )

        backend = str(algorithm_config.get("backend", "normal_std"))
        if backend != "normal_std":
            return GraspGenerationResult(
                success=False,
                candidates=[],
                best_candidate=None,
                status_code=StatusCode.NOT_IMPLEMENTED,
                message=f"当前仅实现 SuctionNet normal_std backend，收到: {backend}",
                source_algorithm=self.source_algorithm,
                retryable=False,
            )

        available, reason = self.is_available()
        if not available:
            return GraspGenerationResult(
                success=False,
                candidates=[],
                best_candidate=None,
                status_code=StatusCode.ADAPTER_UNAVAILABLE,
                message=reason,
                source_algorithm=self.source_algorithm,
                retryable=False,
            )

        try:
            estimate_suction = self._load_normal_std_entry()
            intrinsics = sensor_frame.intrinsics
            camera_info = _CameraInfo(
                width=int(intrinsics.width),
                height=int(intrinsics.height),
                fx=float(intrinsics.fx),
                fy=float(intrinsics.fy),
                cx=float(intrinsics.cx),
                cy=float(intrinsics.cy),
                scale=1.0,
            )
            roi_mask = np.asarray(preprocessed_input.roi_mask, dtype=bool)
            if not np.any(roi_mask):
                return GraspGenerationResult(
                    success=False,
                    candidates=[],
                    best_candidate=None,
                    status_code=StatusCode.NO_VALID_ROI_POINTS,
                    message="ROI mask 为空，SuctionNet 无法执行。",
                    source_algorithm=self.source_algorithm,
                    retryable=False,
                )
            # 严格只把 ROI 范围内的深度送进 SuctionNet，避免利用 ROI 外的全局深度上下文。
            masked_depth_m = np.where(
                roi_mask,
                np.asarray(preprocessed_input.depth_m, dtype=np.float32),
                0.0,
            ).astype(np.float32, copy=False)

            heatmap, normals, point_cloud = estimate_suction(
                masked_depth_m,
                roi_mask,
                camera_info,
            )

            smoothing_kernel_size = int(algorithm_config.get("smoothing_kernel_size", 15))
            if smoothing_kernel_size > 1:
                heatmap = self._apply_uniform_filter(
                    heatmap.astype(np.float32),
                    size=smoothing_kernel_size,
                )

            scores, rows, cols = self._grid_sample(
                heatmap,
                down_rate=int(algorithm_config.get("down_rate", 10)),
                top_k=int(algorithm_config.get("native_top_k", 64)),
            )

            candidates: List[GraspCandidate] = []
            for score, row, col in zip(scores, rows, cols):
                if not preprocessed_input.roi_mask[row, col]:
                    continue

                normal = normals[row, col]
                point = point_cloud[row, col]
                if float(point[2]) <= 0.0:
                    continue
                if float(np.linalg.norm(normal)) <= 1e-6:
                    continue

                quaternion = normal_to_quaternion(
                    surface_normal=normal.tolist(),
                    approach_align=str(algorithm_config.get("approach_align", "-normal")),
                )
                metadata = {
                    "pixel_row": int(row),
                    "pixel_col": int(col),
                    "surface_normal_camera": [float(value) for value in normal.tolist()],
                    "backend": backend,
                    "approach_axis_local": [0.0, 0.0, 1.0],
                    "pose_convention": "tcp_plus_z_is_approach",
                }
                candidates.append(
                    GraspCandidate(
                        position=tuple(float(value) for value in point.tolist()),
                        quaternion=quaternion,
                        score=float(score),
                        grasp_type=self.grasp_type,
                        end_effector_id=end_effector_id,
                        source_algorithm=self.source_algorithm,
                        pose_frame=sensor_frame.camera_frame_id,
                        metadata=metadata,
                    )
                )

            return GraspGenerationResult(
                success=bool(candidates),
                candidates=candidates,
                best_candidate=candidates[0] if candidates else None,
                status_code=StatusCode.OK if candidates else StatusCode.NO_CANDIDATE,
                message="SuctionNet 吸附候选生成完成。" if candidates else "SuctionNet 未生成候选。",
                source_algorithm=self.source_algorithm,
                retryable=not bool(candidates),
                metadata={
                    "roi_bbox_xyxy": list(preprocessed_input.roi_bbox_xyxy),
                    "backend": backend,
                    "heatmap_max": float(np.max(heatmap)) if heatmap.size else 0.0,
                    "roi_mask_pixel_count": int(np.count_nonzero(roi_mask)),
                    "roi_depth_masked_input": True,
                    "roi_origin_normalization": {
                        "applied": False,
                        "reason": "depth_backend_not_translated",
                    },
                },
            )
        except Exception as exc:  # pragma: no cover - 依赖底层原始库，保留防御性保护
            return GraspGenerationResult(
                success=False,
                candidates=[],
                best_candidate=None,
                status_code=StatusCode.NATIVE_RUNTIME_ERROR,
                message=f"SuctionNet 运行失败: {exc}",
                source_algorithm=self.source_algorithm,
                retryable=True,
                metadata={"exception": repr(exc)},
            )

    def _load_normal_std_entry(self):
        patch_numpy_legacy_aliases()
        root = Path(self.config["paths"]["suctionnet_root"]) / "normal_std"
        ensure_sys_path(root)
        from policy import estimate_suction

        return estimate_suction

    @staticmethod
    def _apply_uniform_filter(array: np.ndarray, size: int) -> np.ndarray:
        try:
            from scipy.ndimage import uniform_filter
        except Exception:
            return array
        return uniform_filter(array, size=size, mode="nearest")

    @staticmethod
    def _grid_sample(pred_score_map: np.ndarray, down_rate: int = 20, top_k: int = 512):
        """按网格均匀采样热力图峰值，避免候选过于集中。"""

        height, width = pred_score_map.shape[:2]
        num_row = max(1, height // down_rate)
        num_col = max(1, width // down_rate)

        indices: List[Tuple[int, int]] = []
        for row_id in range(num_row):
            for col_id in range(num_col):
                row_start = row_id * down_rate
                row_end = min(height, (row_id + 1) * down_rate)
                col_start = col_id * down_rate
                col_end = min(width, (col_id + 1) * down_rate)
                score_patch = pred_score_map[row_start:row_end, col_start:col_end]
                if score_patch.size == 0:
                    continue

                flat_index = int(np.argmax(score_patch))
                local_row = flat_index // score_patch.shape[1]
                local_col = flat_index % score_patch.shape[1]
                indices.append((row_start + local_row, col_start + local_col))

        if not indices:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        rows = np.asarray([item[0] for item in indices], dtype=np.int32)
        cols = np.asarray([item[1] for item in indices], dtype=np.int32)
        scores = pred_score_map[rows, cols]
        order = np.argsort(scores)[::-1][:top_k]
        return scores[order], rows[order], cols[order]
