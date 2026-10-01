from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from ..base.status_codes import StatusCode
from ..base.types import GraspCandidate, GraspGenerationResult, SensorFrame
from .transforms import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion, transform_pose


def finalize_generation_result(
    result: GraspGenerationResult,
    sensor_frame: SensorFrame,
    target_frame: str,
    score_threshold: float,
    top_k: int,
    visible_points: bool,
    visible_point_count: int = 0,
    top_down_filter_enabled: bool = False,
    top_down_max_angle_deg: float = 180.0,
) -> GraspGenerationResult:
    """统一执行坐标变换、过滤、排序与 best candidate 选择。"""

    raw_candidates = list(result.candidates)
    candidates = [candidate for candidate in raw_candidates if float(candidate.score) >= float(score_threshold)]
    candidates.sort(key=lambda item: float(item.score), reverse=True)
    metadata = dict(result.metadata or {})
    message = str(result.message or "").strip()

    needs_world_transform_for_output = bool(target_frame == sensor_frame.world_frame_id) and any(
        _candidate_needs_world_transform(candidate, sensor_frame) for candidate in candidates
    )
    needs_world_transform_for_top_down = bool(top_down_filter_enabled) and any(
        _candidate_needs_world_transform(candidate, sensor_frame) for candidate in candidates
    )
    if (needs_world_transform_for_output or needs_world_transform_for_top_down) and sensor_frame.camera_to_world is None:
        missing_reasons = []
        if needs_world_transform_for_output:
            missing_reasons.append("输出 world 坐标")
        if needs_world_transform_for_top_down:
            missing_reasons.append("执行 top-down 角度筛选")
        reason_text = "和".join(missing_reasons) if missing_reasons else "执行 world 坐标相关后处理"
        return GraspGenerationResult(
            success=False,
            candidates=[],
            best_candidate=None,
            status_code=StatusCode.MISSING_WORLD_TRANSFORM,
            message=f"配置要求{reason_text}，但 SensorFrame 未提供 camera_to_world。",
            source_algorithm=result.source_algorithm,
            retryable=False,
            metadata=metadata,
        )

    if top_down_filter_enabled:
        evaluations = _evaluate_candidates_for_top_down_world_angle(
            candidates=candidates,
            sensor_frame=sensor_frame,
            max_angle_deg=top_down_max_angle_deg,
            annotate_metadata=True,
        )
        candidates = [item["candidate"] for item in evaluations if bool(item["passes"])]
        if not candidates:
            adjusted_candidate, adjustment_metadata = _build_top_down_adjusted_fallback_candidate(
                evaluations=evaluations,
                sensor_frame=sensor_frame,
                max_angle_deg=top_down_max_angle_deg,
            )
            if adjusted_candidate is not None:
                candidates = [adjusted_candidate]
                metadata.update(adjustment_metadata)
                original_angle_deg = float(adjustment_metadata.get("postprocess_top_down_auto_adjust_original_angle_deg", float("nan")))
                adjusted_angle_deg = float(adjustment_metadata.get("postprocess_top_down_auto_adjust_final_angle_deg", float("nan")))
                detail = (
                    "所有候选均未直接满足 top-down 角度约束，"
                    f"已将最接近约束的候选从 {original_angle_deg:.2f}deg 调整到 {adjusted_angle_deg:.2f}deg 后继续执行。"
                )
                message = f"{message} {detail}".strip() if message else detail

    if top_k > 0:
        candidates = candidates[:top_k]

    if target_frame == sensor_frame.world_frame_id:
        candidates = [_transform_candidate_to_world(candidate, sensor_frame) for candidate in candidates]

    if candidates:
        return GraspGenerationResult(
            success=True,
            candidates=candidates,
            best_candidate=candidates[0],
            status_code=StatusCode.OK,
            message=message,
            source_algorithm=result.source_algorithm,
            retryable=False,
            metadata=metadata,
        )

    if visible_points:
        status_code = StatusCode.VISIBLE_BUT_UNGRASPABLE
        message, detail_metadata = _build_visible_but_ungraspable_detail(
            result=result,
            raw_candidates=raw_candidates,
            sensor_frame=sensor_frame,
            score_threshold=score_threshold,
            top_k=top_k,
            visible_point_count=visible_point_count,
            top_down_filter_enabled=top_down_filter_enabled,
            top_down_max_angle_deg=top_down_max_angle_deg,
        )
    else:
        status_code = result.status_code or StatusCode.NO_CANDIDATE
        message = result.message or "ROI 内无有效点云，无法生成抓取。"
        detail_metadata = {}

    metadata.update(detail_metadata)

    return GraspGenerationResult(
        success=False,
        candidates=[],
        best_candidate=None,
        status_code=status_code,
        message=message,
        source_algorithm=result.source_algorithm,
        retryable=visible_points,
        metadata=metadata,
    )


def _evaluate_candidates_for_top_down_world_angle(
    *,
    candidates: list[GraspCandidate],
    sensor_frame: SensorFrame,
    max_angle_deg: float,
    annotate_metadata: bool = True,
) -> list[dict]:
    evaluations: list[dict] = []
    normalized_max_angle_deg = max(0.0, float(max_angle_deg))
    for candidate in candidates:
        try:
            angle_deg, approach_axis_world = _compute_top_down_world_angle(candidate, sensor_frame)
            annotated_candidate = candidate
            if annotate_metadata:
                candidate_metadata = dict(candidate.metadata or {})
                candidate_metadata["top_down_world_angle_deg"] = float(angle_deg)
                candidate_metadata["top_down_world_approach_axis"] = [float(value) for value in approach_axis_world.tolist()]
                candidate_metadata["top_down_max_angle_deg"] = float(normalized_max_angle_deg)
                annotated_candidate = _clone_candidate_with_metadata(candidate, candidate_metadata)
            evaluations.append(
                {
                    "candidate": annotated_candidate,
                    "original_candidate": candidate,
                    "angle_deg": float(angle_deg),
                    "approach_axis_world": np.asarray(approach_axis_world, dtype=np.float64).reshape(3),
                    "passes": bool(float(angle_deg) <= normalized_max_angle_deg),
                    "error": "",
                }
            )
        except Exception as exc:
            evaluations.append(
                {
                    "candidate": candidate,
                    "original_candidate": candidate,
                    "angle_deg": float("inf"),
                    "approach_axis_world": None,
                    "passes": False,
                    "error": str(exc),
                }
            )
    return evaluations


def _build_visible_but_ungraspable_detail(
    *,
    result: GraspGenerationResult,
    raw_candidates: list[GraspCandidate],
    sensor_frame: SensorFrame,
    score_threshold: float,
    top_k: int,
    visible_point_count: int,
    top_down_filter_enabled: bool,
    top_down_max_angle_deg: float,
) -> Tuple[str, dict]:
    metadata = dict(result.metadata or {})
    selected_end_effector_id = str(metadata.get("selected_end_effector_id", "")).strip()
    source_algorithm = str(result.source_algorithm or "").strip()
    raw_status_code = str(result.status_code or StatusCode.NO_CANDIDATE)
    raw_message = str(result.message or "").strip()
    raw_candidate_count = int(len(raw_candidates))
    raw_best_score = max((float(candidate.score) for candidate in raw_candidates), default=float("nan"))
    kept_after_threshold = [candidate for candidate in raw_candidates if float(candidate.score) >= float(score_threshold)]
    kept_after_threshold_count = int(len(kept_after_threshold))
    best_score_after_threshold = max((float(candidate.score) for candidate in kept_after_threshold), default=float("nan"))

    kept_after_top_down = kept_after_threshold
    rejected_by_top_down = []
    smallest_top_down_angle_deg = float("nan")
    if top_down_filter_enabled and kept_after_threshold:
        kept_after_top_down, rejected_by_top_down = _filter_candidates_by_top_down_world_angle(
            candidates=kept_after_threshold,
            sensor_frame=sensor_frame,
            max_angle_deg=top_down_max_angle_deg,
            annotate_metadata=False,
        )
        if rejected_by_top_down:
            smallest_top_down_angle_deg = min(float(item["angle_deg"]) for item in rejected_by_top_down)
        for candidate in kept_after_top_down:
            candidate_angle_deg = float(candidate.metadata.get("top_down_world_angle_deg", float("nan")))
            if candidate_angle_deg == candidate_angle_deg:
                if smallest_top_down_angle_deg != smallest_top_down_angle_deg:
                    smallest_top_down_angle_deg = candidate_angle_deg
                else:
                    smallest_top_down_angle_deg = min(smallest_top_down_angle_deg, candidate_angle_deg)
    kept_after_top_down_count = int(len(kept_after_top_down))

    detail_metadata = {
        "postprocess_visible_point_count": int(max(0, visible_point_count)),
        "postprocess_score_threshold": float(score_threshold),
        "postprocess_top_k": int(top_k),
        "postprocess_raw_candidate_count": raw_candidate_count,
        "postprocess_kept_after_threshold_count": kept_after_threshold_count,
        "postprocess_raw_status_code": raw_status_code,
        "postprocess_raw_message": raw_message,
        "postprocess_top_down_filter_enabled": bool(top_down_filter_enabled),
    }
    if top_down_filter_enabled:
        detail_metadata["postprocess_top_down_max_angle_deg"] = float(top_down_max_angle_deg)
        detail_metadata["postprocess_kept_after_top_down_count"] = kept_after_top_down_count
        detail_metadata["postprocess_rejected_by_top_down_count"] = int(len(rejected_by_top_down))
        if smallest_top_down_angle_deg == smallest_top_down_angle_deg:
            detail_metadata["postprocess_smallest_top_down_angle_deg"] = float(smallest_top_down_angle_deg)
    if raw_candidate_count > 0 and raw_best_score == raw_best_score:
        detail_metadata["postprocess_raw_best_score"] = float(raw_best_score)
    if kept_after_threshold_count > 0 and best_score_after_threshold == best_score_after_threshold:
        detail_metadata["postprocess_best_score_after_threshold"] = float(best_score_after_threshold)

    parts = ["目标可见，但当前没有可执行抓取。"]
    if selected_end_effector_id:
        parts.append(f"末端执行器={selected_end_effector_id}.")
    if source_algorithm:
        parts.append(f"算法={source_algorithm}.")
    if visible_point_count > 0:
        parts.append(f"ROI有效点数={int(visible_point_count)}.")

    if raw_candidate_count <= 0:
        parts.append(f"底层候选数=0(raw_status={raw_status_code}).")
        if raw_message:
            parts.append(f"底层返回={raw_message}")
    else:
        parts.append(
            f"底层候选数={raw_candidate_count}, score_threshold={float(score_threshold):.3f}, "
            f"阈值后保留={kept_after_threshold_count}, top_k={int(top_k)}."
        )
        if top_down_filter_enabled:
            parts.append(
                f"top_down约束=与世界系-Z轴夹角<={float(top_down_max_angle_deg):.1f}deg, "
                f"角度筛后保留={kept_after_top_down_count}."
            )
            if smallest_top_down_angle_deg == smallest_top_down_angle_deg:
                parts.append(f"最小夹角={float(smallest_top_down_angle_deg):.2f}deg.")
        if raw_best_score == raw_best_score:
            parts.append(f"原始最高分={float(raw_best_score):.3f}.")
        if kept_after_threshold_count <= 0:
            parts.append("原因=所有候选分数都低于阈值。")
        elif top_down_filter_enabled and kept_after_top_down_count <= 0:
            parts.append(f"原因=所有候选都不满足 top-down 角度约束({float(top_down_max_angle_deg):.1f}deg)。")
        elif raw_message and raw_message != "目标可见，但当前阈值与后处理筛选后没有可执行抓取。":
            parts.append(f"底层返回={raw_message}")

    return " ".join(parts).strip(), detail_metadata


def _candidate_needs_world_transform(candidate: GraspCandidate, sensor_frame: SensorFrame) -> bool:
    return str(candidate.pose_frame or "").strip() != str(sensor_frame.world_frame_id or "").strip()


def _filter_candidates_by_top_down_world_angle(
    *,
    candidates: list[GraspCandidate],
    sensor_frame: SensorFrame,
    max_angle_deg: float,
    annotate_metadata: bool = True,
) -> Tuple[list[GraspCandidate], list[dict]]:
    kept: list[GraspCandidate] = []
    rejected: list[dict] = []
    for item in _evaluate_candidates_for_top_down_world_angle(
        candidates=candidates,
        sensor_frame=sensor_frame,
        max_angle_deg=max_angle_deg,
        annotate_metadata=annotate_metadata,
    ):
        candidate_with_angle = item["candidate"]
        angle_deg = float(item["angle_deg"])
        if bool(item["passes"]):
            kept.append(candidate_with_angle)
        else:
            rejected.append(
                {
                    "angle_deg": float(angle_deg),
                    "score": float(candidate_with_angle.score),
                    "grasp_type": str(candidate_with_angle.grasp_type),
                    "error": str(item.get("error", "")),
                }
            )
    return kept, rejected


def _compute_top_down_world_angle(candidate: GraspCandidate, sensor_frame: SensorFrame) -> Tuple[float, np.ndarray]:
    local_approach_axis = _resolve_local_approach_axis(candidate)
    rotation = quaternion_to_rotation_matrix(candidate.quaternion)
    approach_axis = rotation.dot(local_approach_axis.reshape(3, 1)).reshape(3)
    if _candidate_needs_world_transform(candidate, sensor_frame):
        world_rotation = np.asarray(sensor_frame.camera_to_world, dtype=np.float64).reshape(4, 4)[:3, :3]
        approach_axis = world_rotation.dot(approach_axis.reshape(3, 1)).reshape(3)

    norm = float(np.linalg.norm(approach_axis))
    if norm <= 1e-8:
        raise ValueError("抓取接近方向范数过小，无法计算 top-down 夹角。")

    normalized_approach_axis = approach_axis / norm
    reference_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    cosine = float(np.clip(np.dot(normalized_approach_axis, reference_axis), -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cosine))
    return float(angle_deg), normalized_approach_axis.astype(np.float64)


def _resolve_local_approach_axis(candidate: GraspCandidate) -> np.ndarray:
    metadata = dict(candidate.metadata or {})
    configured_axis = metadata.get("approach_axis_local")
    if isinstance(configured_axis, (list, tuple)) and len(configured_axis) == 3:
        axis = np.asarray(configured_axis, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(axis))
        if norm > 1e-8:
            return axis / norm

    normalized_grasp_type = str(candidate.grasp_type or "").strip().lower()
    if "parallel" in normalized_grasp_type or "gripper" in normalized_grasp_type:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if "suction" in normalized_grasp_type or "magnet" in normalized_grasp_type:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _build_top_down_adjusted_fallback_candidate(
    *,
    evaluations: list[dict],
    sensor_frame: SensorFrame,
    max_angle_deg: float,
) -> Tuple[Optional[GraspCandidate], dict]:
    valid_evaluations = [
        item for item in evaluations
        if item.get("approach_axis_world") is not None and math.isfinite(float(item.get("angle_deg", float("inf"))))
    ]
    if not valid_evaluations:
        return None, {}

    normalized_max_angle_deg = max(0.0, float(max_angle_deg))
    selected = min(
        valid_evaluations,
        key=lambda item: (
            max(0.0, float(item["angle_deg"]) - normalized_max_angle_deg),
            -float(item["candidate"].score),
        ),
    )
    adjusted_candidate = _adjust_candidate_to_top_down_limit(
        candidate=selected["candidate"],
        sensor_frame=sensor_frame,
        max_angle_deg=normalized_max_angle_deg,
        original_angle_deg=float(selected["angle_deg"]),
        original_approach_axis_world=np.asarray(selected["approach_axis_world"], dtype=np.float64).reshape(3),
    )
    adjusted_metadata = dict(adjusted_candidate.metadata or {})
    final_angle_deg = float(adjusted_metadata.get("top_down_world_angle_deg", float("nan")))
    correction_deg = float(adjusted_metadata.get("top_down_world_auto_adjust_correction_deg", float("nan")))
    return adjusted_candidate, {
        "postprocess_top_down_auto_adjust_applied": True,
        "postprocess_top_down_auto_adjust_selection_rule": "min_required_correction_then_highest_score",
        "postprocess_top_down_auto_adjust_original_angle_deg": float(selected["angle_deg"]),
        "postprocess_top_down_auto_adjust_final_angle_deg": final_angle_deg,
        "postprocess_top_down_auto_adjust_correction_deg": correction_deg,
        "postprocess_top_down_auto_adjust_candidate_score": float(adjusted_candidate.score),
        "postprocess_top_down_auto_adjust_grasp_type": str(adjusted_candidate.grasp_type),
    }


def _adjust_candidate_to_top_down_limit(
    *,
    candidate: GraspCandidate,
    sensor_frame: SensorFrame,
    max_angle_deg: float,
    original_angle_deg: float,
    original_approach_axis_world: np.ndarray,
) -> GraspCandidate:
    current_rotation_world = _candidate_rotation_matrix_in_world(candidate, sensor_frame)
    target_approach_axis_world = _project_axis_to_top_down_cone(
        np.asarray(original_approach_axis_world, dtype=np.float64).reshape(3),
        max_angle_deg=max_angle_deg,
        current_rotation_world=current_rotation_world,
    )
    delta_rotation_world = _minimal_rotation_matrix_between_vectors(
        source_vector=np.asarray(original_approach_axis_world, dtype=np.float64).reshape(3),
        target_vector=target_approach_axis_world,
        current_rotation_world=current_rotation_world,
    )
    adjusted_rotation_world = delta_rotation_world.dot(current_rotation_world)
    adjusted_quaternion = _candidate_quaternion_from_world_rotation(candidate, sensor_frame, adjusted_rotation_world)
    correction_cosine = float(
        np.clip(
            np.dot(
                _normalize_vector(np.asarray(original_approach_axis_world, dtype=np.float64).reshape(3)),
                _normalize_vector(target_approach_axis_world),
            ),
            -1.0,
            1.0,
        )
    )
    correction_deg = float(math.degrees(math.acos(correction_cosine)))

    metadata = dict(candidate.metadata or {})
    metadata.update(
        {
            "top_down_auto_adjusted": True,
            "top_down_auto_adjust_reason": "fallback_no_candidate_within_angle_limit",
            "top_down_world_angle_deg_before_adjustment": float(original_angle_deg),
            "top_down_world_approach_axis_before_adjustment": [
                float(value) for value in np.asarray(original_approach_axis_world, dtype=np.float64).reshape(3).tolist()
            ],
            "top_down_world_auto_adjust_correction_deg": correction_deg,
            "top_down_max_angle_deg": float(max_angle_deg),
        }
    )
    adjusted_candidate = _clone_candidate_with_pose_and_metadata(
        candidate,
        quaternion=adjusted_quaternion,
        metadata=metadata,
    )
    adjusted_angle_deg, adjusted_approach_axis_world = _compute_top_down_world_angle(adjusted_candidate, sensor_frame)
    adjusted_metadata = dict(adjusted_candidate.metadata or {})
    adjusted_metadata["top_down_world_angle_deg"] = float(adjusted_angle_deg)
    adjusted_metadata["top_down_world_approach_axis"] = [
        float(value) for value in np.asarray(adjusted_approach_axis_world, dtype=np.float64).reshape(3).tolist()
    ]
    adjusted_metadata["top_down_max_angle_deg"] = float(max_angle_deg)
    return _clone_candidate_with_pose_and_metadata(
        adjusted_candidate,
        metadata=adjusted_metadata,
    )


def _candidate_rotation_matrix_in_world(candidate: GraspCandidate, sensor_frame: SensorFrame) -> np.ndarray:
    rotation = quaternion_to_rotation_matrix(candidate.quaternion)
    if _candidate_needs_world_transform(candidate, sensor_frame):
        world_rotation = np.asarray(sensor_frame.camera_to_world, dtype=np.float64).reshape(4, 4)[:3, :3]
        rotation = world_rotation.dot(rotation)
    return np.asarray(rotation, dtype=np.float64).reshape(3, 3)


def _candidate_quaternion_from_world_rotation(
    candidate: GraspCandidate,
    sensor_frame: SensorFrame,
    world_rotation: np.ndarray,
) -> Tuple[float, float, float, float]:
    rotation = np.asarray(world_rotation, dtype=np.float64).reshape(3, 3)
    if _candidate_needs_world_transform(candidate, sensor_frame):
        camera_to_world = np.asarray(sensor_frame.camera_to_world, dtype=np.float64).reshape(4, 4)
        rotation = camera_to_world[:3, :3].T.dot(rotation)
    return rotation_matrix_to_quaternion(rotation)


def _project_axis_to_top_down_cone(
    approach_axis_world: np.ndarray,
    *,
    max_angle_deg: float,
    current_rotation_world: np.ndarray,
) -> np.ndarray:
    reference_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    normalized_axis = _normalize_vector(approach_axis_world)
    normalized_max_angle_deg = max(0.0, float(max_angle_deg))
    cosine = float(np.clip(np.dot(normalized_axis, reference_axis), -1.0, 1.0))
    current_angle_deg = float(math.degrees(math.acos(cosine)))
    if current_angle_deg <= normalized_max_angle_deg + 1e-6:
        return normalized_axis

    tangential = normalized_axis - cosine * reference_axis
    tangential_norm = float(np.linalg.norm(tangential))
    if tangential_norm <= 1e-8:
        fallback_axis = np.asarray(current_rotation_world, dtype=np.float64).reshape(3, 3)[:, 0]
        tangential = fallback_axis - float(np.dot(fallback_axis, reference_axis)) * reference_axis
        tangential_norm = float(np.linalg.norm(tangential))
        if tangential_norm <= 1e-8:
            fallback_axis = np.asarray(current_rotation_world, dtype=np.float64).reshape(3, 3)[:, 1]
            tangential = fallback_axis - float(np.dot(fallback_axis, reference_axis)) * reference_axis
            tangential_norm = float(np.linalg.norm(tangential))
        if tangential_norm <= 1e-8:
            tangential = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            tangential_norm = 1.0
    tangential = tangential / tangential_norm

    target_angle_rad = math.radians(normalized_max_angle_deg)
    projected = math.cos(target_angle_rad) * reference_axis + math.sin(target_angle_rad) * tangential
    return _normalize_vector(projected)


def _minimal_rotation_matrix_between_vectors(
    *,
    source_vector: np.ndarray,
    target_vector: np.ndarray,
    current_rotation_world: np.ndarray,
) -> np.ndarray:
    source = _normalize_vector(source_vector)
    target = _normalize_vector(target_vector)
    cross = np.cross(source, target)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cross_norm <= 1e-8:
        if dot >= 0.0:
            return np.eye(3, dtype=np.float64)
        fallback_axis = np.asarray(current_rotation_world, dtype=np.float64).reshape(3, 3)[:, 0]
        rotation_axis = np.cross(source, fallback_axis)
        if float(np.linalg.norm(rotation_axis)) <= 1e-8:
            fallback_axis = np.asarray(current_rotation_world, dtype=np.float64).reshape(3, 3)[:, 1]
            rotation_axis = np.cross(source, fallback_axis)
        if float(np.linalg.norm(rotation_axis)) <= 1e-8:
            rotation_axis = np.cross(source, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if float(np.linalg.norm(rotation_axis)) <= 1e-8:
            rotation_axis = np.cross(source, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        return _rotation_matrix_from_axis_angle(_normalize_vector(rotation_axis), math.pi)

    rotation_axis = cross / cross_norm
    rotation_angle = math.atan2(cross_norm, dot)
    return _rotation_matrix_from_axis_angle(rotation_axis, rotation_angle)


def _rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    x, y, z = _normalize_vector(axis)
    cosine = float(math.cos(angle_rad))
    sine = float(math.sin(angle_rad))
    one_minus_cosine = 1.0 - cosine
    return np.array(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ],
        dtype=np.float64,
    )


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-8:
        raise ValueError("向量范数过小，无法归一化。")
    return array / norm


def _clone_candidate_with_metadata(candidate: GraspCandidate, metadata: dict) -> GraspCandidate:
    return GraspCandidate(
        position=tuple(float(value) for value in candidate.position),
        quaternion=tuple(float(value) for value in candidate.quaternion),
        score=float(candidate.score),
        grasp_type=candidate.grasp_type,
        end_effector_id=candidate.end_effector_id,
        source_algorithm=candidate.source_algorithm,
        pose_frame=candidate.pose_frame,
        metadata=metadata,
    )


def _clone_candidate_with_pose_and_metadata(
    candidate: GraspCandidate,
    *,
    position: Optional[Tuple[float, float, float]] = None,
    quaternion: Optional[Tuple[float, float, float, float]] = None,
    metadata: Optional[dict] = None,
) -> GraspCandidate:
    return GraspCandidate(
        position=tuple(float(value) for value in (candidate.position if position is None else position)),
        quaternion=tuple(float(value) for value in (candidate.quaternion if quaternion is None else quaternion)),
        score=float(candidate.score),
        grasp_type=candidate.grasp_type,
        end_effector_id=candidate.end_effector_id,
        source_algorithm=candidate.source_algorithm,
        pose_frame=candidate.pose_frame,
        metadata=dict(candidate.metadata or {}) if metadata is None else metadata,
    )


def _transform_candidate_to_world(candidate: GraspCandidate, sensor_frame: SensorFrame) -> GraspCandidate:
    """将候选位姿从 camera 坐标转换到 world 坐标。"""

    if candidate.pose_frame == sensor_frame.world_frame_id:
        return candidate

    position_world, quaternion_world = transform_pose(
        position=candidate.position,
        quaternion=candidate.quaternion,
        transform=sensor_frame.camera_to_world,
    )

    metadata = dict(candidate.metadata)
    metadata["original_pose_frame"] = candidate.pose_frame
    metadata["camera_position"] = list(candidate.position)
    metadata["camera_quaternion"] = list(candidate.quaternion)

    return GraspCandidate(
        position=position_world,
        quaternion=quaternion_world,
        score=candidate.score,
        grasp_type=candidate.grasp_type,
        end_effector_id=candidate.end_effector_id,
        source_algorithm=candidate.source_algorithm,
        pose_frame=sensor_frame.world_frame_id,
        metadata=metadata,
    )
