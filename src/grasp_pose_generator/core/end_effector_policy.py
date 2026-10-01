from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Sequence

from ..base.types import DetectionResult, SensorFrame


@dataclass
class EndEffectorSelection:
    end_effector_id: str
    rule_name: str
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "end_effector_id": self.end_effector_id,
            "rule_name": self.rule_name,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


class ConfigurableEndEffectorPolicy:
    """配置驱动的末端执行器选择策略。

    目标：
    1. 外部任务请求不再指定 end effector。
    2. 规则集中在独立模块，便于后续直接改策略。
    3. 规则本身尽量放在 YAML 中，减少改代码的频率。
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def select(
        self,
        sensor_frame: SensorFrame,
        detection_result: DetectionResult,
        available_end_effectors: Sequence[str],
    ) -> EndEffectorSelection:
        available = [str(item).strip() for item in available_end_effectors if str(item).strip()]
        policy_config = self.config.get("end_effector_policy", {}) or {}
        rules = policy_config.get("rules", []) or []
        default_end_effector_id = str(policy_config.get("default_end_effector_id", "")).strip()

        if not available:
            return EndEffectorSelection(
                end_effector_id="",
                rule_name="policy_error",
                reason="No end effectors are configured in algorithm_mapping.",
                metadata={"camera_name": sensor_frame.metadata.get("camera_name", "")},
            )

        fallback_end_effector_id = default_end_effector_id if default_end_effector_id in available else available[0]

        for index, rule in enumerate(rules):
            if not isinstance(rule, dict):
                continue
            match_config = rule.get("match", {}) or {}
            if not self._rule_matches(match_config, detection_result):
                continue

            rule_name = str(rule.get("name", f"rule_{index}"))
            requested_end_effector_id = str(rule.get("end_effector_id", "")).strip()
            if requested_end_effector_id in available:
                return EndEffectorSelection(
                    end_effector_id=requested_end_effector_id,
                    rule_name=rule_name,
                    reason=f"Matched end-effector policy rule {rule_name}.",
                    metadata={
                        "camera_name": sensor_frame.metadata.get("camera_name", ""),
                        "detection_label": detection_result.label,
                        "requested_end_effector_id": requested_end_effector_id,
                    },
                )

            return EndEffectorSelection(
                end_effector_id=fallback_end_effector_id,
                rule_name=rule_name,
                reason=(
                    f"Matched end-effector policy rule {rule_name}, but requested end effector "
                    f"{requested_end_effector_id!r} is not configured. Falling back to {fallback_end_effector_id!r}."
                ),
                metadata={
                    "camera_name": sensor_frame.metadata.get("camera_name", ""),
                    "detection_label": detection_result.label,
                    "requested_end_effector_id": requested_end_effector_id,
                    "fallback_end_effector_id": fallback_end_effector_id,
                },
            )

        return EndEffectorSelection(
            end_effector_id=fallback_end_effector_id,
            rule_name="default",
            reason=(
                f"No explicit end-effector policy rule matched target {detection_result.label!r}; "
                f"using default {fallback_end_effector_id!r}."
            ),
            metadata={
                "camera_name": sensor_frame.metadata.get("camera_name", ""),
                "detection_label": detection_result.label,
            },
        )

    def _rule_matches(self, match_config: Dict[str, Any], detection_result: DetectionResult) -> bool:
        metadata = detection_result.metadata or {}
        label = str(detection_result.label).strip()
        object_id = str(metadata.get("object_id", "")).strip()
        detector_name = str(detection_result.detector_name).strip()
        matched_by = str(metadata.get("matched_by", "")).strip()
        class_name = str(metadata.get("class_name", label)).strip()
        class_id = metadata.get("class_id")

        if not self._match_exact(label, match_config.get("labels_any")):
            return False
        if not self._match_prefix(label, match_config.get("label_prefixes_any")):
            return False
        if not self._match_exact(class_name, match_config.get("class_names_any")):
            return False
        if not self._match_exact(object_id, match_config.get("object_ids_any")):
            return False
        if not self._match_prefix(object_id, match_config.get("object_id_prefixes_any")):
            return False
        if not self._match_exact(detector_name, match_config.get("detector_names_any")):
            return False
        if not self._match_exact(matched_by, match_config.get("matched_by_any")):
            return False

        class_ids_any = match_config.get("class_ids_any")
        if class_ids_any not in (None, []):
            normalized = {self._scalar_token(item) for item in self._as_sequence(class_ids_any)}
            if self._scalar_token(class_id) not in normalized:
                return False

        metadata_equals = match_config.get("metadata_equals", {}) or {}
        if not isinstance(metadata_equals, dict):
            return False
        for key, expected in metadata_equals.items():
            if metadata.get(key) != expected:
                return False

        metadata_in = match_config.get("metadata_in", {}) or {}
        if not isinstance(metadata_in, dict):
            return False
        for key, expected_values in metadata_in.items():
            normalized = {self._scalar_token(item) for item in self._as_sequence(expected_values)}
            if self._scalar_token(metadata.get(key)) not in normalized:
                return False

        return True

    def _match_exact(self, value: str, allowed_values: Any) -> bool:
        if allowed_values in (None, []):
            return True
        normalized = {str(item).strip() for item in self._as_sequence(allowed_values) if str(item).strip()}
        return value in normalized

    def _match_prefix(self, value: str, prefixes: Any) -> bool:
        if prefixes in (None, []):
            return True
        normalized = [str(item).strip() for item in self._as_sequence(prefixes) if str(item).strip()]
        return any(value.startswith(prefix) for prefix in normalized)

    def _as_sequence(self, value: Any) -> Iterable[Any]:
        if isinstance(value, (list, tuple, set)):
            return value
        return [value]

    def _scalar_token(self, value: Any) -> str:
        return str(value).strip()
