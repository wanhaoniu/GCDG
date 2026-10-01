"""基础数据结构与接口定义。"""

from .interfaces import GraspAlgorithmAdapter, RetryPolicyInterface, RobotExecutionInterface
from .status_codes import StatusCode
from .types import (
    CameraIntrinsics,
    DetectionResult,
    GraspCandidate,
    GraspGenerationResult,
    SensorFrame,
)

__all__ = [
    "CameraIntrinsics",
    "DetectionResult",
    "GraspAlgorithmAdapter",
    "GraspCandidate",
    "GraspGenerationResult",
    "RetryPolicyInterface",
    "RobotExecutionInterface",
    "SensorFrame",
    "StatusCode",
]
