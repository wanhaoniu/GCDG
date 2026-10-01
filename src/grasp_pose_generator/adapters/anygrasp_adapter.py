from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base.interfaces import GraspAlgorithmAdapter
from ..base.status_codes import StatusCode
from ..base.types import DetectionResult, GraspCandidate, GraspGenerationResult, SensorFrame
from ..core.compat import (
    ensure_sys_path,
    patch_numpy_legacy_aliases,
    patch_torch_load_cpu_checkpoint_compat,
    suppress_output,
    temporary_cwd,
)
from ..core.preprocessing import PreprocessedInput, compute_workspace_lims
from ..core.transforms import rotation_matrix_to_quaternion

_ANYGRASP_GRIPPER_TO_TCP_ROTATION = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
], dtype=np.float64)


class AnyGraspAdapter(GraspAlgorithmAdapter):
    """AnyGrasp 夹爪抓取适配器。

    设计目标:
    - 不修改 AnyGrasp 原仓库代码。
    - 仅在 wrapper 中完成 numpy 兼容补丁、路径注入与统一结果转换。
    """

    adapter_name = "anygrasp"
    source_algorithm = "AnyGrasp"
    grasp_type = "parallel_gripper"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._model: Optional[Any] = None
        self._model_signature: Optional[Tuple[str, float, float, bool, str]] = None

    def is_available(self) -> Tuple[bool, str]:
        paths = self.config.get("paths", {})
        root = Path(paths.get("anygrasp_root", ""))
        checkpoint = Path(paths.get("anygrasp_checkpoint", ""))

        if not root.is_dir():
            return False, f"AnyGrasp 根目录不存在: {root}"
        if not checkpoint.is_file():
            return False, f"AnyGrasp checkpoint 不存在: {checkpoint}"
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
                message="ROI 点云为空，AnyGrasp 无法执行。",
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
            model = self._get_model(algorithm_config)
            paths = self.config.get("paths", {})
            root = Path(paths["anygrasp_root"])
            prepared_points, prepared_colors, prepared_workspace_lims, input_preparation_metadata = self._prepare_anygrasp_input(
                preprocessed_input=preprocessed_input,
                algorithm_config=algorithm_config,
            )
            centered_points, centered_workspace_lims, origin_offset, normalization_metadata = self._normalize_roi_to_origin(
                roi_points_camera=prepared_points,
                workspace_lims_camera=prepared_workspace_lims,
                algorithm_config=algorithm_config,
            )

            with temporary_cwd(root):
                grasp_group, _ = model.get_grasp(
                    centered_points,
                    prepared_colors,
                    lims=list(centered_workspace_lims),
                    apply_object_mask=bool(algorithm_config.get("apply_object_mask", True)),
                    dense_grasp=bool(algorithm_config.get("dense_grasp", False)),
                    collision_detection=bool(algorithm_config.get("collision_detection", True)),
                )

            if grasp_group is None or len(grasp_group) == 0:
                return GraspGenerationResult(
                    success=False,
                    candidates=[],
                    best_candidate=None,
                    status_code=StatusCode.NO_CANDIDATE,
                    message=(
                        "AnyGrasp 未返回任何有效抓取。"
                        f"{self._format_input_preparation_summary(input_preparation_metadata)}"
                    ),
                    source_algorithm=self.source_algorithm,
                    retryable=True,
                    metadata={
                        "roi_bbox_xyxy": list(preprocessed_input.roi_bbox_xyxy),
                        "workspace_lims_camera": list(preprocessed_input.workspace_lims_camera),
                        "workspace_lims_camera_prepared": list(prepared_workspace_lims),
                        "workspace_lims_camera_normalized": list(centered_workspace_lims),
                        "roi_origin_normalization": dict(normalization_metadata),
                        "input_preparation": dict(input_preparation_metadata),
                    },
                )

            grasp_group = grasp_group.nms().sort_by_score()
            native_top_k = int(algorithm_config.get("native_top_k", 64))

            candidates: List[GraspCandidate] = []
            for index in range(min(len(grasp_group), native_top_k)):
                grasp = grasp_group[index]
                system_rotation = np.asarray(grasp.rotation_matrix, dtype=np.float64).reshape(3, 3).dot(
                    _ANYGRASP_GRIPPER_TO_TCP_ROTATION
                )
                quaternion = rotation_matrix_to_quaternion(system_rotation)
                position = np.asarray(grasp.translation, dtype=np.float64).reshape(3) + origin_offset.astype(np.float64)
                metadata = {
                    "width_m": float(grasp.width),
                    "height_m": float(grasp.height),
                    "depth_m": float(grasp.depth),
                    "approach_axis_local": [0.0, 0.0, 1.0],
                    "pose_convention": "tcp_plus_z_is_approach",
                    "raw_anygrasp_approach_axis_local": [1.0, 0.0, 0.0],
                    "roi_origin_normalization": dict(normalization_metadata),
                }
                if hasattr(grasp, "object_id"):
                    metadata["object_id"] = int(grasp.object_id)

                candidates.append(
                    GraspCandidate(
                        position=tuple(float(value) for value in position.tolist()),
                        quaternion=quaternion,
                        score=float(grasp.score),
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
                message=(
                    "AnyGrasp 抓取候选生成完成。"
                    if candidates
                    else "AnyGrasp 未生成候选。"
                ) + self._format_input_preparation_summary(input_preparation_metadata),
                source_algorithm=self.source_algorithm,
                retryable=not bool(candidates),
                metadata={
                    "roi_bbox_xyxy": list(preprocessed_input.roi_bbox_xyxy),
                    "workspace_lims_camera": list(preprocessed_input.workspace_lims_camera),
                    "workspace_lims_camera_prepared": list(prepared_workspace_lims),
                    "workspace_lims_camera_normalized": list(centered_workspace_lims),
                    "candidate_count_raw": int(len(grasp_group)),
                    "roi_origin_normalization": dict(normalization_metadata),
                    "input_preparation": dict(input_preparation_metadata),
                },
            )
        except Exception as exc:  # pragma: no cover - 依赖底层原生库，保留防御性保护
            return GraspGenerationResult(
                success=False,
                candidates=[],
                best_candidate=None,
                status_code=StatusCode.NATIVE_RUNTIME_ERROR,
                message=f"AnyGrasp 运行失败: {exc}",
                source_algorithm=self.source_algorithm,
                retryable=True,
                metadata={
                    "exception": repr(exc),
                    "input_preparation": dict(locals().get("input_preparation_metadata", {})),
                },
            )

    @staticmethod
    def _normalize_roi_to_origin(
        *,
        roi_points_camera: np.ndarray,
        workspace_lims_camera: Tuple[float, float, float, float, float, float],
        algorithm_config: Dict[str, Any],
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float, float, float], np.ndarray, Dict[str, Any]]:
        roi_points = np.asarray(roi_points_camera, dtype=np.float32).reshape(-1, 3)
        workspace_lims = tuple(float(value) for value in workspace_lims_camera)
        configured_center_method = str(algorithm_config.get("origin_center_method", "median"))
        configured_z_shift_mode = str(algorithm_config.get("origin_z_shift_mode", "min"))
        disabled_metadata = {
            "applied": False,
            "reason": "disabled",
            "offset_xyz": [0.0, 0.0, 0.0],
            "center_method": configured_center_method,
            "z_shift_mode": configured_z_shift_mode,
        }
        if not bool(algorithm_config.get("normalize_roi_to_origin", True)):
            return roi_points, workspace_lims, np.zeros(3, dtype=np.float32), disabled_metadata
        if roi_points.shape[0] == 0:
            return roi_points, workspace_lims, np.zeros(3, dtype=np.float32), {
                **disabled_metadata,
                "reason": "empty_roi",
            }

        center_method = str(algorithm_config.get("origin_center_method", "median")).strip().lower()
        if center_method == "mean":
            xy_offset = np.mean(roi_points[:, :2], axis=0).astype(np.float32)
        else:
            center_method = "median"
            xy_offset = np.median(roi_points[:, :2], axis=0).astype(np.float32)

        z_shift_mode = str(algorithm_config.get("origin_z_shift_mode", "min")).strip().lower()
        if z_shift_mode == "workspace_min":
            z_offset = np.float32(workspace_lims[4])
        else:
            z_shift_mode = "min"
            z_offset = np.min(roi_points[:, 2]).astype(np.float32)

        origin_offset = np.array([xy_offset[0], xy_offset[1], z_offset], dtype=np.float32)

        centered_points = (roi_points - origin_offset.reshape(1, 3)).astype(np.float32)
        centered_workspace_lims = (
            float(workspace_lims[0] - origin_offset[0]),
            float(workspace_lims[1] - origin_offset[0]),
            float(workspace_lims[2] - origin_offset[1]),
            float(workspace_lims[3] - origin_offset[1]),
            float(workspace_lims[4] - origin_offset[2]),
            float(workspace_lims[5] - origin_offset[2]),
        )
        normalization_metadata = {
            "applied": True,
            "center_method": center_method,
            "z_shift_mode": z_shift_mode,
            "xy_offset": [float(value) for value in xy_offset.tolist()],
            "z_offset": float(z_offset),
            "offset_xyz": [float(value) for value in origin_offset.tolist()],
        }
        return centered_points, centered_workspace_lims, origin_offset, normalization_metadata

    @staticmethod
    def _prepare_anygrasp_input(
        *,
        preprocessed_input: PreprocessedInput,
        algorithm_config: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float, float, float], Dict[str, Any]]:
        points = np.asarray(preprocessed_input.roi_points_camera, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(preprocessed_input.roi_colors, dtype=np.float32).reshape(-1, 3)
        if points.shape[0] == 0:
            return (
                points,
                colors,
                tuple(float(value) for value in preprocessed_input.workspace_lims_camera),
                {
                    "applied": False,
                    "reason": "empty_roi",
                    "raw_point_count": 0,
                    "point_count_after_voxel": 0,
                    "point_count_final": 0,
                    "voxel_size_m": float(max(0.0, float(algorithm_config.get("input_voxel_size_m", 0.0) or 0.0))),
                    "max_input_points": int(max(0, int(algorithm_config.get("max_input_points", 0) or 0))),
                    "steps": [],
                },
            )

        if colors.shape[0] != points.shape[0]:
            aligned_count = min(points.shape[0], colors.shape[0])
            points = points[:aligned_count]
            colors = colors[:aligned_count]

        voxel_size_m = max(0.0, float(algorithm_config.get("input_voxel_size_m", 0.0) or 0.0))
        max_input_points = max(0, int(algorithm_config.get("max_input_points", 0) or 0))
        prepared_points = points
        prepared_colors = colors
        steps: List[str] = []

        if voxel_size_m > 0.0 and prepared_points.shape[0] > 1:
            prepared_points, prepared_colors = AnyGraspAdapter._voxel_downsample_points(
                prepared_points,
                prepared_colors,
                voxel_size_m,
            )
            if prepared_points.shape[0] < points.shape[0]:
                steps.append("voxel_downsample")

        point_count_after_voxel = int(prepared_points.shape[0])

        if max_input_points > 0 and prepared_points.shape[0] > max_input_points:
            prepared_points, prepared_colors = AnyGraspAdapter._cap_point_count(
                prepared_points,
                prepared_colors,
                max_input_points,
            )
            steps.append("max_point_cap")

        workspace_padding_m = AnyGraspAdapter._estimate_workspace_padding_m(
            roi_points_camera=points,
            workspace_lims_camera=preprocessed_input.workspace_lims_camera,
        )
        prepared_workspace_lims = compute_workspace_lims(
            prepared_points,
            padding_m=workspace_padding_m,
        )
        metadata = {
            "applied": bool(steps),
            "raw_point_count": int(points.shape[0]),
            "point_count_after_voxel": point_count_after_voxel,
            "point_count_final": int(prepared_points.shape[0]),
            "voxel_size_m": float(voxel_size_m),
            "max_input_points": int(max_input_points),
            "workspace_padding_m": float(workspace_padding_m),
            "steps": list(steps),
        }
        return prepared_points, prepared_colors, prepared_workspace_lims, metadata

    @staticmethod
    def _voxel_downsample_points(
        points: np.ndarray,
        colors: np.ndarray,
        voxel_size_m: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        color_array = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
        if voxel_size_m <= 0.0 or point_array.shape[0] <= 1:
            return point_array, color_array

        voxel_indices = np.floor(point_array / float(voxel_size_m)).astype(np.int64)
        unique_voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
        downsampled_points = np.zeros((unique_voxels.shape[0], 3), dtype=np.float64)
        downsampled_colors = np.zeros((unique_voxels.shape[0], 3), dtype=np.float64)
        counts = np.bincount(inverse)
        for axis in range(3):
            downsampled_points[:, axis] = np.bincount(
                inverse,
                weights=point_array[:, axis],
                minlength=unique_voxels.shape[0],
            )
            downsampled_colors[:, axis] = np.bincount(
                inverse,
                weights=color_array[:, axis],
                minlength=unique_voxels.shape[0],
            )
        downsampled_points /= counts.reshape(-1, 1)
        downsampled_colors /= counts.reshape(-1, 1)
        return downsampled_points.astype(np.float32), downsampled_colors.astype(np.float32)

    @staticmethod
    def _cap_point_count(points: np.ndarray, colors: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
        point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        color_array = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
        if max_points <= 0 or point_array.shape[0] <= int(max_points):
            return point_array, color_array

        step = max(1, int(np.ceil(point_array.shape[0] / float(max_points))))
        indices = np.arange(0, point_array.shape[0], step, dtype=np.int64)[: int(max_points)]
        return point_array[indices].astype(np.float32), color_array[indices].astype(np.float32)

    @staticmethod
    def _estimate_workspace_padding_m(
        *,
        roi_points_camera: np.ndarray,
        workspace_lims_camera: Tuple[float, float, float, float, float, float],
    ) -> float:
        points = np.asarray(roi_points_camera, dtype=np.float32).reshape(-1, 3)
        if points.shape[0] == 0:
            return 0.0

        workspace_lims = np.asarray(workspace_lims_camera, dtype=np.float32).reshape(6)
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        lower_padding = mins - workspace_lims[[0, 2, 4]]
        upper_padding = workspace_lims[[1, 3, 5]] - maxs
        padding_values = np.concatenate([lower_padding, upper_padding], axis=0)
        padding_values = padding_values[np.isfinite(padding_values)]
        if padding_values.size == 0:
            return 0.0
        return float(max(0.0, np.median(np.clip(padding_values, a_min=0.0, a_max=None))))

    @staticmethod
    def _format_input_preparation_summary(metadata: Dict[str, Any]) -> str:
        if not metadata:
            return ""

        raw_point_count = int(metadata.get("raw_point_count", 0) or 0)
        point_count_after_voxel = int(metadata.get("point_count_after_voxel", raw_point_count) or raw_point_count)
        point_count_final = int(metadata.get("point_count_final", point_count_after_voxel) or point_count_after_voxel)
        voxel_size_m = float(metadata.get("voxel_size_m", 0.0) or 0.0)
        max_input_points = int(metadata.get("max_input_points", 0) or 0)
        return (
            " 输入点云处理："
            f"原始={raw_point_count}，"
            f"体素后={point_count_after_voxel}，"
            f"最终送入AnyGrasp={point_count_final}，"
            f"体素大小={voxel_size_m:.4f}m，"
            f"点数上限={max_input_points}。"
        )

    def _get_model(self, algorithm_config: Dict[str, Any]) -> Any:
        runtime_config = self._build_runtime_config(algorithm_config)
        model_signature = (
            str(runtime_config.checkpoint_path),
            float(runtime_config.max_gripper_width),
            float(runtime_config.gripper_height),
            bool(runtime_config.top_down_grasp),
            str(runtime_config.device),
        )

        if self._model is not None and self._model_signature == model_signature:
            return self._model

        patch_numpy_legacy_aliases()
        root = Path(self.config["paths"]["anygrasp_root"])
        ensure_sys_path(root)

        with temporary_cwd(root):
            with suppress_output():
                from gsnet import AnyGrasp

                with patch_torch_load_cpu_checkpoint_compat():
                    model = AnyGrasp(runtime_config)
                    model.load_net()

        self._model = model
        self._model_signature = model_signature
        return model

    def _build_runtime_config(self, algorithm_config: Dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            checkpoint_path=self.config["paths"]["anygrasp_checkpoint"],
            max_gripper_width=float(algorithm_config.get("max_gripper_width", 0.1)),
            gripper_height=float(algorithm_config.get("gripper_height", 0.03)),
            top_down_grasp=bool(algorithm_config.get("top_down_grasp", False)),
            debug=bool(algorithm_config.get("debug", False)),
            device=self._resolve_runtime_device(),
        )

    @staticmethod
    def _resolve_runtime_device() -> str:
        try:
            import torch
        except Exception:
            return "cpu"

        try:
            if bool(torch.cuda.is_available()):
                return "cuda:0"
        except Exception:
            pass
        return "cpu"
