from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from .types import DetectionResult, GraspGenerationResult, SensorFrame


class GraspAlgorithmAdapter(ABC):
    """抓取算法适配器抽象基类。

    新算法接入时，只需要实现该接口并在 manager 中注册即可。
    """

    adapter_name: str = "base"
    source_algorithm: str = "BaseAdapter"
    grasp_type: str = "unknown"

    @abstractmethod
    def is_available(self) -> Tuple[bool, str]:
        """检查底层算法运行条件是否满足。"""

    @abstractmethod
    def generate(
        self,
        sensor_frame: SensorFrame,
        detection_result: DetectionResult,
        end_effector_id: str,
        preprocessed_input: Any,
        algorithm_config: Dict[str, Any],
    ) -> GraspGenerationResult:
        """执行底层算法，并返回统一 grasp 结果。"""


class RobotExecutionInterface(ABC):
    """预留: 机器人执行接口。"""

    @abstractmethod
    def execute(self, result: GraspGenerationResult, end_effector_id: str) -> Any:
        """将 best grasp 下发给机器人执行。"""


class RetryPolicyInterface(ABC):
    """预留: 重抓 / retry 接口。"""

    @abstractmethod
    def should_retry(
        self,
        result: GraspGenerationResult,
        sensor_frame: SensorFrame,
        detection_result: DetectionResult,
        end_effector_id: str,
    ) -> Tuple[bool, Optional[str]]:
        """根据当前结果判断是否应该重抓。"""
