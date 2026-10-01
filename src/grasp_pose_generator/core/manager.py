from __future__ import annotations

from typing import Any, Dict, Optional, Set

from ..adapters.anygrasp_adapter import AnyGraspAdapter
from ..adapters.suctionnet_adapter import SuctionNetAdapter
from ..base.interfaces import GraspAlgorithmAdapter, RobotExecutionInterface
from ..base.status_codes import StatusCode
from ..base.types import DetectionResult, GraspGenerationResult, SensorFrame
from .config_loader import load_config
from .end_effector_policy import ConfigurableEndEffectorPolicy, EndEffectorSelection
from .postprocessing import finalize_generation_result
from .preprocessing import PreprocessedInput, preprocess_sensor_frame
from .support_plane import augment_preprocessed_with_support_plane


ARCHIVE_RAW_RESULT_KEY = "archive_raw_generation_result"
ARCHIVE_RAW_COUNT_KEY = "archive_raw_candidate_count"


class GraspGeneratorManager:
    """统一抓取位姿生成调度器。

    外部统一调用入口:
        generate(sensor_frame, detection_result, end_effector_id=None)

    内部职责拆分:
    1. 根据显式传入的 end_effector_id 或策略模块自动选择末端执行器
    2. 执行 ROI 点云裁剪 / 坐标前处理
    3. 调用底层算法 wrapper
    4. 统一完成排序、过滤、top-k 与 camera/world 坐标转换
    """

    def __init__(self, config: Dict[str, Any], robot_executor: Optional[RobotExecutionInterface] = None) -> None:
        self.config = config
        self.robot_executor = robot_executor
        self.adapters: Dict[str, GraspAlgorithmAdapter] = {}
        self.end_effector_policy = ConfigurableEndEffectorPolicy(config)
        self._register_builtin_adapters()

    @classmethod
    def from_yaml(cls, config_path: str) -> "GraspGeneratorManager":
        return cls(load_config(config_path))

    def register_adapter(self, adapter_name: str, adapter: GraspAlgorithmAdapter) -> None:
        """预留新算法扩展接口: 新 adapter 在这里注册即可接入 manager。"""

        self.adapters[adapter_name] = adapter

    def available_end_effectors(self):
        return sorted(self.config.get("algorithm_mapping", {}).keys())

    def select_end_effector(
        self,
        sensor_frame: SensorFrame,
        detection_result: DetectionResult,
    ) -> EndEffectorSelection:
        return self.end_effector_policy.select(
            sensor_frame=sensor_frame,
            detection_result=detection_result,
            available_end_effectors=self.available_end_effectors(),
        )

    def generate(
        self,
        sensor_frame: SensorFrame,
        detection_result: DetectionResult,
        end_effector_id: Optional[str] = None,
    ) -> GraspGenerationResult:
        selection = self._resolve_end_effector_selection(sensor_frame, detection_result, end_effector_id)
        preprocessing_config = self.config.get('preprocessing', {})

        try:
            preprocessed_input = preprocess_sensor_frame(
                sensor_frame=sensor_frame,
                detection_result=detection_result,
                roi_padding_pixels=int(preprocessing_config.get('roi_padding_pixels', 0)),
                workspace_padding_m=float(preprocessing_config.get('workspace_padding_m', 0.0)),
                min_depth_m=float(preprocessing_config.get('min_depth_m', 0.001)),
                max_depth_m=preprocessing_config.get('max_depth_m'),
            )
        except Exception as exc:
            return self._attach_selection_metadata(
                GraspGenerationResult(
                    success=False,
                    candidates=[],
                    best_candidate=None,
                    status_code=StatusCode.INVALID_INPUT,
                    message=f'输入预处理失败: {exc}',
                    source_algorithm='',
                    retryable=False,
                    metadata={'exception': repr(exc)},
                ),
                selection,
            )

        return self._generate_from_preprocessed_with_selection(
            sensor_frame=sensor_frame,
            detection_result=detection_result,
            selection=selection,
            preprocessed_input=preprocessed_input,
        )

    def generate_from_preprocessed(
        self,
        sensor_frame: SensorFrame,
        detection_result: DetectionResult,
        preprocessed_input: PreprocessedInput,
        end_effector_id: Optional[str] = None,
    ) -> GraspGenerationResult:
        selection = self._resolve_end_effector_selection(sensor_frame, detection_result, end_effector_id)
        return self._generate_from_preprocessed_with_selection(
            sensor_frame=sensor_frame,
            detection_result=detection_result,
            selection=selection,
            preprocessed_input=preprocessed_input,
        )

    def _generate_from_preprocessed_with_selection(
        self,
        sensor_frame: SensorFrame,
        detection_result: DetectionResult,
        selection: EndEffectorSelection,
        preprocessed_input: PreprocessedInput,
    ) -> GraspGenerationResult:
        mapping = self.config.get('algorithm_mapping', {}).get(selection.end_effector_id)
        if mapping is None:
            return self._attach_selection_metadata(
                GraspGenerationResult(
                    success=False,
                    candidates=[],
                    best_candidate=None,
                    status_code=StatusCode.CONFIG_ERROR,
                    message=f'未在配置中找到 end_effector_id: {selection.end_effector_id}',
                    source_algorithm='',
                    retryable=False,
                ),
                selection,
            )

        adapter_name = str(mapping.get('adapter', '')).strip()
        adapter = self.adapters.get(adapter_name)
        if adapter is None:
            return self._attach_selection_metadata(
                GraspGenerationResult(
                    success=False,
                    candidates=[],
                    best_candidate=None,
                    status_code=StatusCode.ADAPTER_NOT_FOUND,
                    message=f'未注册的 adapter: {adapter_name}',
                    source_algorithm=adapter_name,
                    retryable=False,
                ),
                selection,
            )

        algorithm_config = self.config.get('algorithms', {}).get(adapter_name, {})
        visible_points = preprocessed_input.roi_points_camera.size > 0
        if not visible_points:
            return self._attach_selection_metadata(
                GraspGenerationResult(
                    success=False,
                    candidates=[],
                    best_candidate=None,
                    status_code=StatusCode.NO_VALID_ROI_POINTS,
                    message='ROI 内没有有效深度点，无法生成抓取。',
                    source_algorithm=adapter.source_algorithm,
                    retryable=False,
                    metadata={'roi_bbox_xyxy': list(preprocessed_input.roi_bbox_xyxy)},
                ),
                selection,
            )

        support_plane_config = self.config.get('support_plane', {})
        support_plane_adapters = self._resolve_support_plane_adapters()
        support_plane_metadata = {}
        augmented_preprocessed_input = preprocessed_input
        if adapter_name.strip().lower() in support_plane_adapters:
            try:
                augmented_preprocessed_input, support_plane_metadata = augment_preprocessed_with_support_plane(
                    sensor_frame=sensor_frame,
                    detection_result=detection_result,
                    preprocessed_input=preprocessed_input,
                    support_plane_config=support_plane_config,
                    adapter_name=adapter_name,
                    workspace_padding_m=float(self.config.get('preprocessing', {}).get('workspace_padding_m', 0.0)),
                )
            except Exception as exc:
                support_plane_metadata = {
                    "enabled": False,
                    "reason": "augmentation_failed",
                    "adapter": adapter_name,
                    "exception": repr(exc),
                }
        else:
            support_plane_metadata = {
                "enabled": False,
                "reason": "skipped_for_adapter",
                "adapter": adapter_name,
                "apply_to_adapters": sorted(support_plane_adapters),
            }

        raw_result = adapter.generate(
            sensor_frame=sensor_frame,
            detection_result=detection_result,
            end_effector_id=selection.end_effector_id,
            preprocessed_input=augmented_preprocessed_input,
            algorithm_config=algorithm_config,
        )
        if support_plane_metadata:
            metadata = dict(raw_result.metadata or {})
            metadata["support_plane"] = support_plane_metadata
            raw_result.metadata = metadata

        terminal_errors = {
            StatusCode.CONFIG_ERROR,
            StatusCode.INVALID_INPUT,
            StatusCode.ADAPTER_NOT_FOUND,
            StatusCode.ADAPTER_UNAVAILABLE,
            StatusCode.NATIVE_RUNTIME_ERROR,
            StatusCode.NOT_IMPLEMENTED,
        }
        if raw_result.status_code in terminal_errors and not raw_result.candidates:
            raw_result = self._attach_selection_metadata(raw_result, selection)
            return self._attach_archive_raw_result(raw_result, raw_result.to_dict())

        return self.finalize_result(
            sensor_frame=sensor_frame,
            selection=selection,
            raw_result=raw_result,
            visible_points=visible_points,
            visible_point_count=int(preprocessed_input.roi_points_camera.shape[0]),
        )

    def finalize_result(
        self,
        sensor_frame: SensorFrame,
        selection: EndEffectorSelection,
        raw_result: GraspGenerationResult,
        *,
        visible_points: bool,
        visible_point_count: int = 0,
    ) -> GraspGenerationResult:
        mapping = self.config.get('algorithm_mapping', {}).get(selection.end_effector_id)
        if mapping is None:
            return self._attach_selection_metadata(
                GraspGenerationResult(
                    success=False,
                    candidates=[],
                    best_candidate=None,
                    status_code=StatusCode.CONFIG_ERROR,
                    message=f'未在配置中找到 end_effector_id: {selection.end_effector_id}',
                    source_algorithm='',
                    retryable=False,
                ),
                selection,
            )

        runtime_config = self.config.get('runtime', {})
        target_frame = str(mapping.get('target_frame', runtime_config.get('target_frame', sensor_frame.camera_frame_id)))
        score_threshold = float(mapping.get('score_threshold', runtime_config.get('score_threshold', 0.0)))
        top_k = int(mapping.get('top_k', runtime_config.get('top_k', 5)))
        top_down_filter_enabled = bool(mapping.get('top_down_filter_enabled', runtime_config.get('top_down_filter_enabled', False)))
        top_down_max_angle_deg = float(mapping.get('top_down_max_angle_deg', runtime_config.get('top_down_max_angle_deg', 180.0)))

        raw_result = self._attach_selection_metadata(raw_result, selection)
        raw_archive_result = raw_result.to_dict()
        finalized = finalize_generation_result(
            result=raw_result,
            sensor_frame=sensor_frame,
            target_frame=target_frame,
            score_threshold=score_threshold,
            top_k=top_k,
            visible_points=visible_points,
            visible_point_count=visible_point_count,
            top_down_filter_enabled=top_down_filter_enabled,
            top_down_max_angle_deg=top_down_max_angle_deg,
        )
        finalized = self._attach_selection_metadata(finalized, selection)
        return self._attach_archive_raw_result(finalized, raw_archive_result)

    def build_robot_execution_request(self, result: GraspGenerationResult) -> Dict[str, Any]:
        """预留: 机器人执行接口的请求构造，不直接执行真机动作。"""

        if result.best_candidate is None:
            raise ValueError("当前结果没有 best_candidate，无法构造执行请求。")

        return {
            "status_code": result.status_code,
            "best_candidate": result.best_candidate.to_dict(),
            "source_algorithm": result.source_algorithm,
        }

    def should_retry(self, result: GraspGenerationResult) -> bool:
        """预留: 重抓策略入口。当前先提供最小可用的 retry 判定。"""

        return bool(result.retryable or result.status_code == StatusCode.VISIBLE_BUT_UNGRASPABLE)

    def _resolve_end_effector_selection(
        self,
        sensor_frame: SensorFrame,
        detection_result: DetectionResult,
        end_effector_id: Optional[str],
    ) -> EndEffectorSelection:
        explicit = str(end_effector_id or "").strip()
        if explicit:
            return EndEffectorSelection(
                end_effector_id=explicit,
                rule_name="manual_override",
                reason="Using the explicit end effector passed to GraspGeneratorManager.generate().",
                metadata={"requested_end_effector_id": explicit},
            )
        return self.select_end_effector(sensor_frame, detection_result)

    def _attach_selection_metadata(
        self,
        result: GraspGenerationResult,
        selection: EndEffectorSelection,
    ) -> GraspGenerationResult:
        selection_metadata = {
            "selected_end_effector_id": selection.end_effector_id,
            "end_effector_policy_rule": selection.rule_name,
            "end_effector_policy_reason": selection.reason,
            "end_effector_policy_metadata": dict(selection.metadata),
        }
        metadata = dict(result.metadata or {})
        metadata.update(selection_metadata)
        result.metadata = metadata

        for candidate in result.candidates:
            candidate.metadata = dict(candidate.metadata or {})
            candidate.metadata.update(selection_metadata)
        if result.best_candidate is not None:
            result.best_candidate.metadata = dict(result.best_candidate.metadata or {})
            result.best_candidate.metadata.update(selection_metadata)

        return result

    def _attach_archive_raw_result(
        self,
        result: GraspGenerationResult,
        raw_result_payload: Dict[str, Any],
    ) -> GraspGenerationResult:
        metadata = dict(result.metadata or {})
        metadata[ARCHIVE_RAW_RESULT_KEY] = dict(raw_result_payload or {})
        metadata[ARCHIVE_RAW_COUNT_KEY] = int(len(raw_result_payload.get("candidates", [])))
        result.metadata = metadata
        return result

    def _resolve_support_plane_adapters(self) -> Set[str]:
        raw_adapters = self.config.get('support_plane', {}).get('apply_to_adapters', ['anygrasp'])
        if isinstance(raw_adapters, str):
            adapter_names = [raw_adapters]
        elif isinstance(raw_adapters, (list, tuple, set)):
            adapter_names = list(raw_adapters)
        else:
            adapter_names = ['anygrasp']

        normalized = {
            str(adapter_name).strip().lower()
            for adapter_name in adapter_names
            if str(adapter_name).strip()
        }
        if normalized:
            return normalized
        return {'anygrasp'}

    def _register_builtin_adapters(self) -> None:
        self.register_adapter("anygrasp", AnyGraspAdapter(self.config))
        self.register_adapter("suctionnet", SuctionNetAdapter(self.config))
