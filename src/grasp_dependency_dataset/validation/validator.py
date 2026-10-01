"""Stage-wise feasibility validator for grasp proposals."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - fallback path for environments without open3d.
    o3d = None

from grasp_dependency_dataset.common.config import GraspValidationConfig
from grasp_dependency_dataset.common.types import (
    FailureStage,
    GraspProposal,
    GraspType,
    SceneObjectState,
    StableScene,
    ValidationResult,
)

from .geometry import (
    as_array,
    distance,
    normalize,
    segment_intersects_sphere,
    within_sphere,
)

_MIN_SEGMENT_SAMPLE_STEP_M = 0.005
_SEGMENT_SAMPLE_STEP_RATIO = 0.5
_BIN_WALL_BLOCKER_ID = "__bin_wall__"
_APPROACH_TERMINAL_CLEARANCE_M = 0.003
_TOP_DOWN_APPROACH_COS_THRESHOLD = 0.80
_SUPPORT_Z_TOLERANCE_M = 0.0025
_TARGET_SHIELD_DISTANCE_M = 0.004
_TARGET_SHIELD_MARGIN_M = 0.001
_LIFT_INITIAL_CLEARANCE_M = 0.003
_LIFT_BOTTOM_FALSE_POSITIVE_BAND_M = 0.004
_TARGET_SWEEP_SURFACE_EPSILON_M = 0.0005
_MAX_TARGET_SWEEP_POINTS = 192


@dataclass(frozen=True)
class _CollisionQueryData:
    """Cached mesh-query payload for repeated signed-distance lookups."""

    vertices: np.ndarray
    raycasting_scene: Any


@dataclass
class StagedGraspValidator:
    """Validate grasp proposals with ordered, stage-aware checks."""

    config: GraspValidationConfig

    def validate_all(
        self,
        scene: StableScene,
        target_id: str,
        proposals: list[GraspProposal],
    ) -> dict[str, ValidationResult]:
        """Validate a batch of proposals for one target."""

        return {
            proposal.grasp_id: self.validate(scene=scene, target_id=target_id, proposal=proposal)
            for proposal in proposals
        }

    def validate(
        self,
        scene: StableScene,
        target_id: str,
        proposal: GraspProposal,
    ) -> ValidationResult:
        """Validate one proposal with stage-aware failure attribution."""

        target = scene.get_object(target_id)
        obstacles = tuple(self._obstacles(scene, target_id))
        contact = as_array(proposal.pose.position)
        approach = normalize(proposal.approach_vector)
        lift = normalize(proposal.lift_vector)
        tool_radius = self._tool_radius_for_proposal(proposal)
        pregrasp = contact - approach * proposal.pregrasp_offset
        approach_end = _trimmed_approach_end(
            contact=contact,
            approach=approach,
            pregrasp=pregrasp,
        )
        lift_end = contact + lift * self.config.lift_height

        stage_blockers = {
            FailureStage.UNREACHABLE.value: [],
            FailureStage.APPROACH_COLLISION.value: [],
            FailureStage.CLOSING_OR_SEAL_FAILURE.value: [],
            FailureStage.LIFT_COLLISION.value: [],
        }

        if not within_sphere(pregrasp, self.config.workspace_center, self.config.reach_radius):
            return ValidationResult(
                grasp_id=proposal.grasp_id,
                feasible=False,
                failure_stage=FailureStage.UNREACHABLE,
                stage_blockers=stage_blockers,
                notes="Pre-grasp pose lies outside the configured workspace bounds.",
            )

        if not within_sphere(contact, self.config.workspace_center, self.config.reach_radius):
            return ValidationResult(
                grasp_id=proposal.grasp_id,
                feasible=False,
                failure_stage=FailureStage.UNREACHABLE,
                stage_blockers=stage_blockers,
                notes="Contact pose lies outside the configured workspace bounds.",
            )

        if contact[2] < self.config.min_height:
            return ValidationResult(
                grasp_id=proposal.grasp_id,
                feasible=False,
                failure_stage=FailureStage.UNREACHABLE,
                stage_blockers=stage_blockers,
                notes="Contact point falls below the configured floor-height tolerance.",
            )

        if contact[2] > self.config.max_height:
            return ValidationResult(
                grasp_id=proposal.grasp_id,
                feasible=False,
                failure_stage=FailureStage.UNREACHABLE,
                stage_blockers=stage_blockers,
                notes="Contact point exceeds the configured workspace ceiling.",
            )

        if self.config.consider_robot_body:
            kinematic_blockers = self._segment_blockers(
                scene=scene,
                target_id=target_id,
                start=self.config.robot_base_position,
                end=tuple(pregrasp.tolist()),
                radius=self.config.arm_corridor_radius,
                obstacles=obstacles,
            )
            stage_blockers[FailureStage.UNREACHABLE.value] = kinematic_blockers
            if kinematic_blockers:
                return ValidationResult(
                    grasp_id=proposal.grasp_id,
                    feasible=False,
                    failure_stage=FailureStage.UNREACHABLE,
                    stage_blockers=stage_blockers,
                    notes="Robot arm corridor to pre-grasp is blocked by clutter.",
                )

        approach_blockers = self._segment_blockers(
            scene=scene,
            target_id=target_id,
            start=tuple(pregrasp.tolist()),
            end=tuple(approach_end.tolist()),
            radius=tool_radius,
            motion_name="approach",
            target=target,
            approach=tuple(approach.tolist()),
            obstacle_shrink_margin=self.config.approach_object_shrink_margin,
            obstacles=obstacles,
        )
        if self.config.consider_container_collision:
            approach_blockers = sorted(
                set(approach_blockers)
                | set(
                    self._container_segment_blockers(
                        scene=scene,
                        start=tuple(pregrasp.tolist()),
                        end=tuple(approach_end.tolist()),
                        wall_shrink_margin=self.config.approach_object_shrink_margin,
                    )
                )
            )
        stage_blockers[FailureStage.APPROACH_COLLISION.value] = approach_blockers
        if approach_blockers:
            return ValidationResult(
                grasp_id=proposal.grasp_id,
                feasible=False,
                failure_stage=FailureStage.APPROACH_COLLISION,
                stage_blockers=stage_blockers,
                notes=self._collision_note(
                    blockers=approach_blockers,
                    motion_name="Approach path",
                ),
            )

        if self.config.enable_closing_check:
            closing_blockers, closing_note = self._closing_or_seal_blockers(
                scene=scene,
                target=target,
                target_id=target_id,
                proposal=proposal,
                contact=tuple(contact.tolist()),
                obstacles=obstacles,
            )
            stage_blockers[FailureStage.CLOSING_OR_SEAL_FAILURE.value] = closing_blockers
            if closing_blockers or closing_note:
                return ValidationResult(
                    grasp_id=proposal.grasp_id,
                    feasible=False,
                    failure_stage=FailureStage.CLOSING_OR_SEAL_FAILURE,
                    stage_blockers=stage_blockers,
                    notes=closing_note or "Closing or seal stage is blocked.",
                )

        lift_blockers = self._segment_blockers(
            scene=scene,
            target_id=target_id,
            start=tuple(contact.tolist()),
            end=tuple(lift_end.tolist()),
            radius=tool_radius,
            motion_name="lift",
            target=target,
            lift=tuple(lift.tolist()),
            obstacle_shrink_margin=self.config.lift_object_shrink_margin,
            obstacles=obstacles,
        )
        lift_blockers = sorted(
            set(lift_blockers)
            | set(
                self._swept_target_lift_blockers(
                    scene=scene,
                    target_id=target_id,
                    target=target,
                    contact=tuple(contact.tolist()),
                    lift_end=tuple(lift_end.tolist()),
                    lift=tuple(lift.tolist()),
                    obstacle_shrink_margin=self.config.lift_object_shrink_margin,
                    obstacles=obstacles,
                )
            )
        )
        if self.config.consider_container_collision:
            lift_blockers = sorted(
                set(lift_blockers)
                | set(
                    self._container_segment_blockers(
                        scene=scene,
                        start=tuple(contact.tolist()),
                        end=tuple(lift_end.tolist()),
                        wall_shrink_margin=self.config.lift_object_shrink_margin,
                    )
                )
            )
        stage_blockers[FailureStage.LIFT_COLLISION.value] = lift_blockers
        if lift_blockers:
            return ValidationResult(
                grasp_id=proposal.grasp_id,
                feasible=False,
                failure_stage=FailureStage.LIFT_COLLISION,
                stage_blockers=stage_blockers,
                notes=self._collision_note(
                    blockers=lift_blockers,
                    motion_name="Lift path",
                ),
            )

        return ValidationResult(
            grasp_id=proposal.grasp_id,
            feasible=True,
            failure_stage=FailureStage.NONE,
            stage_blockers=stage_blockers,
            notes="All staged checks passed.",
        )

    def _tool_radius_for_proposal(self, proposal: GraspProposal) -> float:
        """Return the modality-specific swept-volume radius used for motion checks."""

        if proposal.grasp_type == GraspType.SUCTION:
            return self.config.suction_tool_radius
        return self.config.parallel_jaw_tool_radius

    def _swept_target_lift_blockers(
        self,
        scene: StableScene,
        target_id: str,
        target: SceneObjectState,
        contact,
        lift_end,
        lift: tuple[float, float, float],
        obstacle_shrink_margin: float = 0.0,
        obstacles: tuple[SceneObjectState, ...] | None = None,
    ) -> list[str]:
        """Check whether translating the whole target along lift causes collisions."""

        lift_arr = normalize(lift)
        contact_arr = as_array(contact)
        lift_end_arr = as_array(lift_end)
        lift_distance = float(np.linalg.norm(lift_end_arr - contact_arr))
        if lift_distance < 1e-9:
            return []

        trimmed_clearance = min(_LIFT_INITIAL_CLEARANCE_M, lift_distance * 0.45)
        translation_start = as_array(target.pose.position) + lift_arr * trimmed_clearance
        translation_end = as_array(target.pose.position) + (lift_end_arr - contact_arr)
        if float(np.linalg.norm(translation_end - translation_start)) < 1e-9:
            return []

        target_points_body = _target_sample_points_body(target)
        target_radius = max(float(target.spec.bounding_radius), 1e-6)
        center_samples = _sample_segment_points(
            start=tuple(translation_start.tolist()),
            end=tuple(translation_end.tolist()),
            radius=max(target_radius * 0.25, _MIN_SEGMENT_SAMPLE_STEP_M),
        )
        target_points_world = _transform_body_points_to_world_samples(
            points_body=target_points_body,
            center_samples=center_samples,
            body_quaternion_wxyz=target.pose.quaternion_wxyz,
        )
        target_point_lift_projections = _body_point_projections_along_world_axis(
            points_body=target_points_body,
            body_quaternion_wxyz=target.pose.quaternion_wxyz,
            axis_world=lift_arr,
        )

        blockers: list[str] = []
        center_start = tuple(translation_start.tolist())
        center_end = tuple(translation_end.tolist())
        shrink_margin = max(float(obstacle_shrink_margin), 0.0)
        collision_threshold = shrink_margin - _TARGET_SWEEP_SURFACE_EPSILON_M
        flat_target_points_world = target_points_world.reshape(-1, 3)
        for obstacle in (obstacles or tuple(self._obstacles(scene, target_id))):
            inflated_radius = max(obstacle.spec.bounding_radius + target_radius - shrink_margin, 0.0)
            if not segment_intersects_sphere(
                center_start,
                center_end,
                obstacle.pose.position,
                inflated_radius,
            ):
                continue
            signed_distance = _signed_distance_to_object(
                points_world=flat_target_points_world,
                scene_object=obstacle,
            ).reshape(target_points_world.shape[:2])
            colliding_mask = signed_distance >= collision_threshold
            if not np.any(colliding_mask):
                continue
            if self._is_lift_bottom_contact_false_positive(
                target=target,
                obstacle=obstacle,
                lift=lift,
                target_point_lift_projections=target_point_lift_projections,
                colliding_mask=colliding_mask,
            ):
                continue
            if np.any(colliding_mask):
                blockers.append(obstacle.object_id)

        if self.config.consider_container_collision and self._swept_target_hits_bin_wall(
            scene=scene,
            target_points_world=target_points_world,
            wall_shrink_margin=shrink_margin,
        ):
            blockers.append(_BIN_WALL_BLOCKER_ID)

        return sorted(set(blockers))

    def _swept_target_hits_bin_wall(
        self,
        scene: StableScene,
        target_points_world: np.ndarray,
        wall_shrink_margin: float = 0.0,
    ) -> bool:
        """Return whether lifted target samples exceed the relaxed bin footprint."""

        clearance = float(self.config.container_wall_clearance) - max(
            float(wall_shrink_margin),
            0.0,
        )
        half_x = float(scene.bin_size[0]) * 0.5 - clearance
        half_y = float(scene.bin_size[1]) * 0.5 - clearance
        return bool(
            np.any(np.abs(target_points_world[:, :, 0]) > half_x)
            or np.any(np.abs(target_points_world[:, :, 1]) > half_y)
        )

    def _segment_blockers(
        self,
        scene: StableScene,
        target_id: str,
        start,
        end,
        radius: float,
        motion_name: str = "generic",
        target: SceneObjectState | None = None,
        approach: tuple[float, float, float] | None = None,
        lift: tuple[float, float, float] | None = None,
        obstacle_shrink_margin: float = 0.0,
        obstacles: tuple[SceneObjectState, ...] | None = None,
    ) -> list[str]:
        blockers = []
        segment_samples_world = _sample_segment_points(start=start, end=end, radius=radius)
        for obstacle in (obstacles or tuple(self._obstacles(scene, target_id))):
            if self._segment_collides_with_obstacle(
                start=start,
                end=end,
                radius=radius,
                obstacle=obstacle,
                target=target,
                approach=approach,
                lift=lift,
                motion_name=motion_name,
                obstacle_shrink_margin=obstacle_shrink_margin,
                segment_samples_world=segment_samples_world,
            ):
                blockers.append(obstacle.object_id)
        return sorted(blockers)

    def _closing_or_seal_blockers(
        self,
        scene: StableScene,
        target: SceneObjectState,
        target_id: str,
        proposal: GraspProposal,
        contact,
        obstacles: tuple[SceneObjectState, ...] | None = None,
    ) -> tuple[list[str], str]:
        blockers = []
        target_distance = distance(contact, target.pose.position)

        if proposal.grasp_type == GraspType.PARALLEL_JAW:
            jaw_width = proposal.jaw_width or (2.0 * target.spec.bounding_radius)
            if jaw_width > self.config.parallel_jaw.max_width:
                return [], "Target is wider than the configured parallel jaw opening."
            interaction_radius = jaw_width / 2.0 + self.config.parallel_jaw.closing_clearance
            if target_distance > target.spec.bounding_radius + self.config.parallel_jaw.finger_depth:
                return [], "Parallel-jaw contact is too far from the target support volume."
        else:
            interaction_radius = (
                proposal.suction_radius
                if proposal.suction_radius is not None
                else self.config.suction.cup_radius
            ) + self.config.suction.seal_tolerance
            if contact[2] < target.pose.position[2]:
                return [], "Suction proposal does not approach the target from above."
            if target_distance > target.spec.bounding_radius + self.config.suction.seal_tolerance:
                return [], "Suction contact point falls outside the target support region."

        for obstacle in (obstacles or tuple(self._obstacles(scene, target_id))):
            if self._point_collides_with_obstacle(
                point=contact,
                radius=interaction_radius,
                obstacle=obstacle,
            ):
                blockers.append(obstacle.object_id)

        return sorted(blockers), ""

    @staticmethod
    def _obstacles(scene: StableScene, target_id: str) -> list[SceneObjectState]:
        return [obj for obj in scene.objects if obj.object_id != target_id]

    def _container_segment_blockers(
        self,
        scene: StableScene,
        start,
        end,
        wall_shrink_margin: float = 0.0,
    ) -> list[str]:
        """Return a synthetic blocker id when a swept centerline exceeds the relaxed bin footprint."""

        clearance = float(self.config.container_wall_clearance) - max(
            float(wall_shrink_margin),
            0.0,
        )
        samples = _sample_segment_points(start=start, end=end, radius=abs(clearance))
        half_x = float(scene.bin_size[0]) * 0.5 - clearance
        half_y = float(scene.bin_size[1]) * 0.5 - clearance
        for sample in samples:
            if abs(float(sample[0])) > half_x or abs(float(sample[1])) > half_y:
                return [_BIN_WALL_BLOCKER_ID]
        return []

    @staticmethod
    def _collision_note(blockers: list[str], motion_name: str) -> str:
        """Describe whether a collision came from clutter, the bin wall, or both."""

        has_bin_wall = _BIN_WALL_BLOCKER_ID in blockers
        has_object_blockers = any(blocker_id != _BIN_WALL_BLOCKER_ID for blocker_id in blockers)
        if has_bin_wall and has_object_blockers:
            return f"{motion_name} collides with the bin wall and non-target clutter."
        if has_bin_wall:
            return f"{motion_name} collides with the bin wall."
        return f"{motion_name} collides with non-target clutter."

    def _segment_collides_with_obstacle(
        self,
        start,
        end,
        radius: float,
        obstacle: SceneObjectState,
        target: SceneObjectState | None = None,
        approach: tuple[float, float, float] | None = None,
        lift: tuple[float, float, float] | None = None,
        motion_name: str = "generic",
        obstacle_shrink_margin: float = 0.0,
        segment_samples_world: np.ndarray | None = None,
    ) -> bool:
        shrink_margin = max(float(obstacle_shrink_margin), 0.0)
        effective_radius = max(float(radius) - shrink_margin, 0.0)
        inflated_radius = obstacle.spec.bounding_radius + effective_radius
        if not segment_intersects_sphere(start, end, obstacle.pose.position, inflated_radius):
            return False

        segment_samples_world = (
            segment_samples_world
            if segment_samples_world is not None
            else _sample_segment_points(start=start, end=end, radius=radius)
        )
        obstacle_signed_distance = _signed_distance_to_object(
            points_world=segment_samples_world,
            scene_object=obstacle,
        )
        colliding_mask = obstacle_signed_distance >= -effective_radius
        if not np.any(colliding_mask):
            return False

        if (
            motion_name == "lift"
            and target is not None
            and lift is not None
            and self._is_lift_below_target_false_positive(
                target=target,
                obstacle=obstacle,
                lift=lift,
            )
        ):
            return False

        if (
            motion_name == "approach"
            and target is not None
            and approach is not None
            and self._is_support_occlusion_false_positive(
                target=target,
                obstacle=obstacle,
                approach=approach,
            )
        ):
            colliding_samples = segment_samples_world[colliding_mask]
            colliding_obstacle_signed_distance = obstacle_signed_distance[colliding_mask]
            target_signed_distance = _signed_distance_to_object(
                points_world=colliding_samples,
                scene_object=target,
            )
            visible_collision_mask = ~_target_shielding_mask(
                target_signed_distance=target_signed_distance,
                obstacle_signed_distance=colliding_obstacle_signed_distance,
                radius=radius,
            )
            return bool(np.any(visible_collision_mask))

        return True

    def _point_collides_with_obstacle(
        self,
        point,
        radius: float,
        obstacle: SceneObjectState,
    ) -> bool:
        if distance(point, obstacle.pose.position) > obstacle.spec.bounding_radius + radius:
            return False

        signed_distance = _signed_distance_to_object(
            points_world=np.asarray([point], dtype=np.float64),
            scene_object=obstacle,
        )
        return bool(float(signed_distance[0]) >= -float(radius))

    def _is_support_occlusion_false_positive(
        self,
        target: SceneObjectState,
        obstacle: SceneObjectState,
        approach: tuple[float, float, float],
    ) -> bool:
        """Detect obstacles that sit below the target and should not block top-down approach."""

        approach_arr = normalize(approach)
        if float(-approach_arr[2]) < _TOP_DOWN_APPROACH_COS_THRESHOLD:
            return False

        target_bottom = float(target.pose.position[2]) - _object_support_extent_along_world_axis(
            scene_object=target,
            axis_world=(0.0, 0.0, 1.0),
        )
        obstacle_top = float(obstacle.pose.position[2]) + _object_support_extent_along_world_axis(
            scene_object=obstacle,
            axis_world=(0.0, 0.0, 1.0),
        )
        return obstacle_top <= target_bottom + _SUPPORT_Z_TOLERANCE_M

    def _is_lift_below_target_false_positive(
        self,
        target: SceneObjectState,
        obstacle: SceneObjectState,
        lift: tuple[float, float, float],
    ) -> bool:
        """Detect lower obstacles that should not block an upward target lift."""

        return _obstacle_upper_support_below_target_midplane(
            target=target,
            obstacle=obstacle,
            axis_world=lift,
        )

    def _is_lift_bottom_contact_false_positive(
        self,
        target: SceneObjectState,
        obstacle: SceneObjectState,
        lift: tuple[float, float, float],
        target_point_lift_projections: np.ndarray,
        colliding_mask: np.ndarray,
    ) -> bool:
        """Ignore initial bottom-face penetration against an obstacle that sits below the target."""

        if not self._is_lift_below_target_false_positive(
            target=target,
            obstacle=obstacle,
            lift=lift,
        ):
            return False

        colliding_rows, colliding_cols = np.where(colliding_mask)
        if colliding_rows.size == 0:
            return False
        if not np.all(colliding_rows == 0):
            return False

        lower_half_threshold = max(
            _LIFT_BOTTOM_FALSE_POSITIVE_BAND_M,
            _TARGET_SWEEP_SURFACE_EPSILON_M,
        )
        return bool(np.all(target_point_lift_projections[colliding_cols] <= lower_half_threshold))


def _sample_segment_points(start, end, radius: float) -> np.ndarray:
    """Sample a segment densely enough to approximate a swept-sphere tool path."""

    start_arr = as_array(start)
    end_arr = as_array(end)
    length = float(np.linalg.norm(end_arr - start_arr))
    if length < 1e-12:
        return start_arr.reshape(1, 3)

    max_step = max(_MIN_SEGMENT_SAMPLE_STEP_M, float(radius) * _SEGMENT_SAMPLE_STEP_RATIO)
    count = max(2, int(np.ceil(length / max_step)) + 1)
    weights = np.linspace(0.0, 1.0, num=count, dtype=np.float64)
    return start_arr[None, :] + (end_arr - start_arr)[None, :] * weights[:, None]


def _trimmed_approach_end(contact: np.ndarray, approach: np.ndarray, pregrasp: np.ndarray) -> np.ndarray:
    """Stop the approach corridor a few millimeters before the final contact point."""

    motion_length = float(np.linalg.norm(contact - pregrasp))
    if motion_length < 1e-9:
        return contact.copy()
    clearance = min(_APPROACH_TERMINAL_CLEARANCE_M, motion_length * 0.45)
    return contact - normalize(approach) * clearance


def _world_points_to_object_local(points_world: np.ndarray, pose) -> np.ndarray:
    """Transform world-frame points into an object's local frame."""

    rotation = _rotation_matrix_from_wxyz(pose.quaternion_wxyz)
    translation = as_array(pose.position)
    return (np.asarray(points_world, dtype=np.float64) - translation[None, :]).dot(rotation)


@lru_cache(maxsize=1024)
def _rotation_matrix_from_wxyz(quaternion_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    """Convert a `wxyz` quaternion into a rotation matrix."""

    w, x, y, z = quaternion_wxyz
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _signed_distance_to_object(points_world: np.ndarray, scene_object: SceneObjectState) -> np.ndarray:
    """Return signed distance with positive-inside, negative-outside convention."""

    points_body = _world_points_to_object_local(
        points_world=np.asarray(points_world, dtype=np.float64),
        pose=scene_object.pose,
    )
    if points_body.size == 0:
        return np.zeros((0,), dtype=np.float64)
    query_data = _load_collision_query(
        mesh_path=scene_object.spec.mesh_path,
        mesh_scale=scene_object.spec.mesh_scale,
        mesh_offset=scene_object.spec.mesh_offset,
    )
    if query_data is not None and o3d is not None:
        try:
            surface_distance = query_data.raycasting_scene.compute_distance(
                o3d.core.Tensor(
                    np.asarray(points_body, dtype=np.float32),
                    dtype=o3d.core.Dtype.Float32,
                )
            ).numpy()
            # Some source meshes have inconsistent winding, so we take the
            # mesh-accurate surface distance magnitude from Open3D and keep the
            # sign convention from the existing primitive proxy.
            surface_distance = np.asarray(surface_distance, dtype=np.float64)
            proxy_signed_distance = _signed_distance_to_primitive(
                points_body=points_body,
                scene_object=scene_object,
            )
            return np.where(proxy_signed_distance >= 0.0, surface_distance, -surface_distance)
        except Exception:
            pass
    return _signed_distance_to_primitive(points_body=points_body, scene_object=scene_object)


def _signed_distance_to_primitive(points_body: np.ndarray, scene_object: SceneObjectState) -> np.ndarray:
    """Analytic signed distance fallback for proxy primitives."""

    primitive_type = scene_object.spec.primitive_type
    size = scene_object.spec.size
    points = _body_points_to_proxy_local(
        points_body=np.asarray(points_body, dtype=np.float64),
        scene_object=scene_object,
    )
    if primitive_type == "sphere":
        radius = float(size[0])
        return radius - np.linalg.norm(points, axis=1)
    if primitive_type == "box":
        half_extents = np.asarray(size[:3], dtype=np.float64)
        q = np.abs(points) - half_extents[None, :]
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return -(outside + inside)
    if primitive_type == "cylinder":
        radius = float(size[0])
        half_height = float(size[1])
        radial = np.linalg.norm(points[:, :2], axis=1) - radius
        axial = np.abs(points[:, 2]) - half_height
        d = np.stack([radial, axial], axis=1)
        outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
        inside = np.minimum(np.max(d, axis=1), 0.0)
        return -(outside + inside)
    if primitive_type == "capsule":
        radius = float(size[0])
        half_height = float(size[1])
        closest = points.copy()
        closest[:, 2] = np.clip(closest[:, 2], -half_height, half_height)
        return radius - np.linalg.norm(points - closest, axis=1)

    fallback_radius = float(scene_object.spec.bounding_radius)
    return fallback_radius - np.linalg.norm(points, axis=1)


def _object_support_extent_along_world_axis(
    scene_object: SceneObjectState,
    axis_world: tuple[float, float, float],
) -> float:
    """Approximate one-sided support extent for proxy geometry along a world axis."""

    axis = normalize(axis_world)
    rotation = _rotation_matrix_from_wxyz(scene_object.pose.quaternion_wxyz)
    axis_body = np.asarray(axis, dtype=np.float64).dot(rotation)
    axis_local = _body_vectors_to_proxy_local(
        vectors_body=axis_body.reshape(1, 3),
        scene_object=scene_object,
    )[0]
    primitive_type = scene_object.spec.primitive_type
    size = scene_object.spec.size

    if primitive_type == "sphere":
        return float(size[0])
    if primitive_type == "box":
        half_extents = np.asarray(size[:3], dtype=np.float64)
        return float(np.sum(np.abs(axis_local) * half_extents))
    if primitive_type == "cylinder":
        radius = float(size[0])
        half_height = float(size[1])
        return float(radius * np.linalg.norm(axis_local[:2]) + half_height * abs(axis_local[2]))
    if primitive_type == "capsule":
        radius = float(size[0])
        half_height = float(size[1])
        return float(radius + half_height * abs(axis_local[2]))

    return float(scene_object.spec.bounding_radius)


def _obstacle_upper_support_below_target_midplane(
    target: SceneObjectState,
    obstacle: SceneObjectState,
    axis_world: tuple[float, float, float],
) -> bool:
    """Return whether the obstacle stays below the target midpoint along the motion axis."""

    axis = normalize(axis_world)
    target_center_projection = float(np.dot(as_array(target.pose.position), axis))
    obstacle_center_projection = float(np.dot(as_array(obstacle.pose.position), axis))
    obstacle_upper_projection = obstacle_center_projection + _object_support_extent_along_world_axis(
        scene_object=obstacle,
        axis_world=axis_world,
    )
    if obstacle_center_projection >= target_center_projection - _SUPPORT_Z_TOLERANCE_M:
        return False
    return obstacle_upper_projection <= target_center_projection + _SUPPORT_Z_TOLERANCE_M


def _body_point_projections_along_world_axis(
    points_body: np.ndarray,
    body_quaternion_wxyz: tuple[float, float, float, float],
    axis_world: tuple[float, float, float],
) -> np.ndarray:
    """Project body-frame points onto a world-space axis after applying body orientation."""

    rotation_world_from_body = _rotation_matrix_from_wxyz(body_quaternion_wxyz).T
    lifted_axis = normalize(axis_world)
    world_points = np.asarray(points_body, dtype=np.float64).dot(rotation_world_from_body)
    return world_points.dot(lifted_axis)


def _body_points_to_proxy_local(points_body: np.ndarray, scene_object: SceneObjectState) -> np.ndarray:
    """Transform body-frame sample points into the primitive proxy's local frame."""

    return _body_vectors_to_proxy_local(
        vectors_body=np.asarray(points_body, dtype=np.float64),
        scene_object=scene_object,
    )


def _body_vectors_to_proxy_local(vectors_body: np.ndarray, scene_object: SceneObjectState) -> np.ndarray:
    """Rotate body-frame vectors into proxy-local coordinates when needed."""

    proxy_quaternion = scene_object.spec.proxy_quaternion_wxyz
    vectors = np.asarray(vectors_body, dtype=np.float64)
    if proxy_quaternion is None:
        return vectors
    proxy_rotation = _rotation_matrix_from_wxyz(proxy_quaternion)
    return vectors.dot(proxy_rotation)


def _target_shielding_mask(
    target_signed_distance: np.ndarray,
    obstacle_signed_distance: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Return samples where the target shields a below-target obstacle from approach checks."""

    shielding_band = max(_TARGET_SHIELD_DISTANCE_M, float(radius))
    near_target = target_signed_distance >= -shielding_band
    target_is_closer = target_signed_distance >= (obstacle_signed_distance + _TARGET_SHIELD_MARGIN_M)
    return near_target & target_is_closer


@lru_cache(maxsize=256)
def _load_collision_query(
    mesh_path: str | None,
    mesh_scale: tuple[float, float, float] | None,
    mesh_offset: tuple[float, float, float] | None,
):
    """Build one cached Open3D raycasting scene for repeated signed-distance calls."""

    if o3d is None or not mesh_path:
        return None

    try:
        legacy_mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=False)
    except TypeError:
        legacy_mesh = o3d.io.read_triangle_mesh(mesh_path)
    except Exception:
        return None

    vertices = np.asarray(legacy_mesh.vertices, dtype=np.float64)
    triangles = np.asarray(legacy_mesh.triangles, dtype=np.int32)
    if len(vertices) <= 0 or len(triangles) <= 0:
        return None

    vertices = vertices.copy()
    if mesh_scale is not None:
        vertices *= np.asarray(mesh_scale, dtype=np.float64)[None, :]
    if mesh_offset is not None:
        vertices += np.asarray(mesh_offset, dtype=np.float64)[None, :]

    try:
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vertices),
                o3d.utility.Vector3iVector(triangles),
            )
        )
        raycasting_scene = o3d.t.geometry.RaycastingScene()
        raycasting_scene.add_triangles(tensor_mesh)
    except Exception:
        return None

    return _CollisionQueryData(
        vertices=vertices,
        raycasting_scene=raycasting_scene,
    )


@lru_cache(maxsize=256)
def _target_sample_points_body(scene_object: SceneObjectState) -> np.ndarray:
    """Return deterministic body-frame sample points that approximate the target surface."""

    query_data = _load_collision_query(
        mesh_path=scene_object.spec.mesh_path,
        mesh_scale=scene_object.spec.mesh_scale,
        mesh_offset=scene_object.spec.mesh_offset,
    )
    if query_data is not None and len(query_data.vertices) > 0:
        vertices = np.asarray(query_data.vertices, dtype=np.float64)
        if len(vertices) > _MAX_TARGET_SWEEP_POINTS:
            indices = np.linspace(
                0,
                len(vertices) - 1,
                num=_MAX_TARGET_SWEEP_POINTS,
                dtype=int,
            )
            vertices = vertices[indices]
        if not np.any(np.all(np.isclose(vertices, 0.0), axis=1)):
            vertices = np.vstack([vertices, np.zeros((1, 3), dtype=np.float64)])
        return vertices

    primitive_points_proxy = _primitive_sample_points_proxy_local(scene_object)
    proxy_quaternion = scene_object.spec.proxy_quaternion_wxyz
    if proxy_quaternion is None:
        return primitive_points_proxy
    proxy_rotation = _rotation_matrix_from_wxyz(proxy_quaternion)
    return primitive_points_proxy.dot(proxy_rotation.T)


def _primitive_sample_points_proxy_local(scene_object: SceneObjectState) -> np.ndarray:
    """Return proxy-local sample points for the primitive fallback geometry."""

    primitive_type = scene_object.spec.primitive_type
    size = scene_object.spec.size

    if primitive_type == "sphere":
        radius = float(size[0])
        return np.asarray(
            [
                (0.0, 0.0, 0.0),
                (radius, 0.0, 0.0),
                (-radius, 0.0, 0.0),
                (0.0, radius, 0.0),
                (0.0, -radius, 0.0),
                (0.0, 0.0, radius),
                (0.0, 0.0, -radius),
            ],
            dtype=np.float64,
        )

    if primitive_type == "box":
        hx, hy, hz = (float(v) for v in size[:3])
        corners = np.asarray(
            [
                (sx * hx, sy * hy, sz * hz)
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        face_centers = np.asarray(
            [
                (hx, 0.0, 0.0),
                (-hx, 0.0, 0.0),
                (0.0, hy, 0.0),
                (0.0, -hy, 0.0),
                (0.0, 0.0, hz),
                (0.0, 0.0, -hz),
                (0.0, 0.0, 0.0),
            ],
            dtype=np.float64,
        )
        return np.vstack([corners, face_centers])

    if primitive_type in {"cylinder", "capsule"}:
        radius = float(size[0])
        half_height = float(size[1])
        angles = np.linspace(0.0, 2.0 * np.pi, num=12, endpoint=False, dtype=np.float64)
        ring_top = np.stack(
            [
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.full_like(angles, half_height),
            ],
            axis=1,
        )
        ring_bottom = np.stack(
            [
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.full_like(angles, -half_height),
            ],
            axis=1,
        )
        mid_ring = np.stack(
            [
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.zeros_like(angles),
            ],
            axis=1,
        )
        axial = np.asarray(
            [
                (0.0, 0.0, half_height),
                (0.0, 0.0, -half_height),
                (0.0, 0.0, 0.0),
            ],
            dtype=np.float64,
        )
        return np.vstack([ring_top, ring_bottom, mid_ring, axial])

    radius = float(scene_object.spec.bounding_radius)
    return np.asarray(
        [
            (0.0, 0.0, 0.0),
            (radius, 0.0, 0.0),
            (-radius, 0.0, 0.0),
            (0.0, radius, 0.0),
            (0.0, -radius, 0.0),
            (0.0, 0.0, radius),
            (0.0, 0.0, -radius),
        ],
        dtype=np.float64,
    )


def _transform_body_points_to_world_samples(
    points_body: np.ndarray,
    center_samples: np.ndarray,
    body_quaternion_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    """Transform body-frame points to world coordinates for each sampled center."""

    rotation_world_from_body = _rotation_matrix_from_wxyz(body_quaternion_wxyz).T
    rotated = np.asarray(points_body, dtype=np.float64).dot(rotation_world_from_body)
    return rotated[None, :, :] + np.asarray(center_samples, dtype=np.float64)[:, None, :]
