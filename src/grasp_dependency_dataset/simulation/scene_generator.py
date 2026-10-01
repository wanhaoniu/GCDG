"""Bin clutter scene generation built on top of MuJoCo settling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grasp_dependency_dataset.common.config import SceneGenerationConfig
from grasp_dependency_dataset.common.types import Pose, SceneObjectState, StableScene

from .assets import AssetCatalog
from .backend import SimulationBackend
from .scene_builder import build_bin_scene_xml

_MAX_WAVE_DROP_ATTEMPTS = 8
_MIN_WAVE_SIZE = 2
_MAX_WAVE_SIZE = 3
_MIN_INTERMEDIATE_SETTLE_STEPS = 500
_MAX_INTERMEDIATE_SETTLE_STEPS = 1200
_PILE_TOP_MARGIN_M = 0.020
_PILE_EXTRA_DROP_SPAN_M = 0.020
_PILE_TILT_DEG = 18.0
_WAVE_PLANAR_FRACTION_SCALE = 2.6
_MIN_WAVE_PLANAR_FRACTION = 0.28
_MAX_WAVE_PLANAR_FRACTION = 0.66
_WAVE_CENTER_BIAS_POWER_SCALE = 0.45
_WAVE_OVERLAP_CLEARANCE_SCALE = 0.78


@dataclass
class BinClutterSceneGenerator:
    """Generate stable bin clutter scenes with random object subsets."""

    config: SceneGenerationConfig
    catalog: AssetCatalog
    backend: SimulationBackend

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.config.seed)

    def generate_scene(self, scene_index: int) -> StableScene:
        """Generate and settle one scene, then sample target ids."""

        num_objects = int(
            self._rng.integers(self.config.num_objects_min, self.config.num_objects_max + 1)
        )
        specs = self._sample_specs(num_objects)
        scene_id = self._format_scene_id(scene_index)
        final_states, settled, pile_metadata = self._generate_wave_pile(
            scene_id=scene_id,
            specs=specs,
        )
        out_of_bin_object_ids = self._out_of_bin_object_ids(final_states)
        stable = bool(settled.stable) and not out_of_bin_object_ids

        target_count = int(
            self._rng.integers(self.config.num_targets_min, self.config.num_targets_max + 1)
        )
        target_count = min(target_count, len(final_states))
        target_ids = tuple(
            sorted(
                self._rng.choice(
                    [state.object_id for state in final_states],
                    size=target_count,
                    replace=False,
                ).tolist()
            )
        )
        metadata = {
            "asset_catalog": self.catalog.name,
            "physics_stable": bool(settled.stable),
            "stable": stable,
            "in_bin": not out_of_bin_object_ids,
            "out_of_bin_object_ids": out_of_bin_object_ids,
            **pile_metadata,
            **settled.metadata,
        }
        return StableScene(
            scene_id=scene_id,
            objects=tuple(final_states),
            target_ids=target_ids,
            bin_size=self.config.bin.size,
            wall_thickness=self.config.bin.wall_thickness,
            metadata=metadata,
        )

    def _format_scene_id(self, scene_index: int) -> str:
        """Build a shorter scene id for exported artifacts."""

        prefix = self._compact_scene_prefix(self.config.scene_id_prefix)
        return f"{prefix}_{scene_index:04d}"

    @staticmethod
    def _compact_scene_prefix(scene_id_prefix: str) -> str:
        """Shorten verbose CEPB prefixes while keeping non-CEPB names stable."""

        if not scene_id_prefix.startswith("cepb_"):
            return scene_id_prefix
        return "scene"

    def _generate_wave_pile(
        self,
        scene_id: str,
        specs,
    ) -> tuple[list[SceneObjectState], object, dict[str, object]]:
        """Build one scene by dropping small waves of objects so clutter can pile and spread."""

        settled_states: list[SceneObjectState] = []
        per_wave_attempts: list[int] = []
        wave_sizes = self._sample_wave_sizes(len(specs))
        last_settled = None
        intermediate_steps = self._intermediate_settle_steps(len(specs))
        spec_start_index = 0

        for wave_index, wave_size in enumerate(wave_sizes):
            wave_specs = specs[spec_start_index : spec_start_index + wave_size]
            accepted_states: list[SceneObjectState] | None = None
            accepted_settled = None
            last_candidate_states = settled_states
            last_candidate_settled = last_settled
            used_attempts = 0
            for drop_attempt in range(1, _MAX_WAVE_DROP_ATTEMPTS + 1):
                used_attempts = drop_attempt
                candidate_wave_states = self._sample_wave_drop_states(
                    scene_id=scene_id,
                    start_index=spec_start_index,
                    wave_specs=wave_specs,
                    settled_states=settled_states,
                )
                candidate_scene = [*settled_states, *candidate_wave_states]
                settled_candidate = self._settle_states(
                    candidate_scene,
                    settle_steps=intermediate_steps,
                )
                candidate_final_states = self._apply_settled_poses(
                    candidate_scene,
                    settled_candidate.body_poses,
                )
                last_candidate_states = candidate_final_states
                last_candidate_settled = settled_candidate
                if bool(settled_candidate.stable) and not self._out_of_bin_object_ids(candidate_final_states):
                    accepted_states = candidate_final_states
                    accepted_settled = settled_candidate
                    break

            per_wave_attempts.append(used_attempts)
            if accepted_states is None or accepted_settled is None:
                failure_metadata = {
                    "generation_mode": "wave_drop_pile",
                    "wave_drop_attempts": per_wave_attempts,
                    "wave_sizes": wave_sizes,
                    "failed_wave_index": wave_index,
                    "failed_object_index": spec_start_index,
                }
                body_poses = {state.object_id: state.pose for state in last_candidate_states}
                stable_flag = False
                if last_candidate_settled is not None:
                    body_poses = last_candidate_settled.body_poses
                    stable_flag = bool(last_candidate_settled.stable)
                fallback_settled = type(last_candidate_settled)(
                    body_poses=body_poses,
                    stable=stable_flag,
                    metadata={"progressive_drop_failed": True},
                ) if last_candidate_settled is not None else type("FallbackSettled", (), {})()
                if last_candidate_settled is None:
                    class _FallbackSettled:
                        def __init__(self, poses):
                            self.body_poses = poses
                            self.stable = False
                            self.metadata = {"progressive_drop_failed": True}

                    fallback_settled = _FallbackSettled(body_poses)
                return last_candidate_states, fallback_settled, failure_metadata

            settled_states = accepted_states
            last_settled = accepted_settled
            spec_start_index += wave_size

        final_settled = self._settle_states(
            settled_states,
            settle_steps=self.config.simulation.settle_steps,
        )
        final_states = self._apply_settled_poses(settled_states, final_settled.body_poses)
        metadata = {
            "generation_mode": "wave_drop_pile",
            "wave_drop_attempts": per_wave_attempts,
            "wave_sizes": wave_sizes,
        }
        return final_states, final_settled, metadata

    def _sample_wave_sizes(self, total_objects: int) -> list[int]:
        """Split the requested object count into small drop waves."""

        if total_objects <= 0:
            return []
        configured_max = self.config.simulation.drop_wave_size_max
        if configured_max is not None and int(configured_max) <= 1:
            return [1] * total_objects
        if total_objects <= 10:
            return [1] * total_objects

        max_wave_size = int(configured_max) if configured_max is not None else _MAX_WAVE_SIZE
        max_wave_size = max(1, min(max_wave_size, _MAX_WAVE_SIZE))
        remaining = total_objects
        wave_sizes: list[int] = []
        while remaining > 0:
            if remaining <= max_wave_size:
                wave_size = remaining
            else:
                wave_size = int(
                    self._rng.integers(
                        _MIN_WAVE_SIZE,
                        min(max_wave_size, remaining) + 1,
                    )
                )
            wave_sizes.append(wave_size)
            remaining -= wave_size
        return wave_sizes

    def _sample_specs(self, num_objects: int):
        excluded_names = set(self.config.excluded_asset_names)
        candidates = [
            spec for spec in self.catalog.sampleable_specs()
            if spec.name not in excluded_names
        ]
        if not candidates:
            raise ValueError(
                "No sampleable assets remain after applying excluded_asset_names."
            )
        replace = len(candidates) < num_objects
        sampled_indices = self._rng.choice(len(candidates), size=num_objects, replace=replace)
        return [candidates[int(index)] for index in sampled_indices]

    def _sample_initial_states(self, scene_id: str, specs):
        size_x, size_y, _ = self.config.bin.size
        wall = self.config.bin.wall_thickness
        spawn_min, spawn_max = self.config.simulation.spawn_height_range
        states = []
        placed_xy: list[tuple[float, float, float]] = []
        for object_index, spec in enumerate(specs):
            body_name = f"obj_{object_index:02d}"
            placement_radius = self._planar_placement_radius(spec)
            pos_x, pos_y = self._sample_planar_position(
                size_x=size_x,
                size_y=size_y,
                wall=wall,
                placement_radius=placement_radius,
                placed_xy=placed_xy,
            )
            quat = self._random_quaternion()
            vertical_half_extent = self._world_vertical_half_extent(
                spec=spec,
                body_quaternion_wxyz=quat,
            )
            pos_z = (
                float(self._rng.uniform(spawn_min, spawn_max))
                + 2.0 * vertical_half_extent
                + 0.015 * object_index
            )
            states.append(
                SceneObjectState(
                    object_id=body_name,
                    asset_name=spec.name,
                    scale=1.0,
                    spec=spec,
                    pose=Pose((pos_x, pos_y, pos_z), quat),
                )
            )
            placed_xy.append((pos_x, pos_y, placement_radius))
        return states

    def _sample_wave_drop_states(
        self,
        scene_id: str,
        start_index: int,
        wave_specs,
        settled_states: list[SceneObjectState],
    ) -> list[SceneObjectState]:
        """Sample one wave of objects above the current pile with loose planar spacing."""

        wave_states: list[SceneObjectState] = []
        wave_positions: list[tuple[float, float, float]] = []
        for local_index, spec in enumerate(wave_specs):
            body_name = f"obj_{start_index + local_index:02d}"
            quat = self._random_quaternion(max_tilt_deg=_PILE_TILT_DEG)
            pos_x, pos_y = self._sample_wave_drop_position(
                spec=spec,
                wave_positions=wave_positions,
            )
            pos_z = self._sample_progressive_drop_height(
                spec=spec,
                settled_states=settled_states,
                body_quaternion_wxyz=quat,
                wave_offset_index=local_index,
            )
            wave_states.append(
                SceneObjectState(
                    object_id=body_name,
                    asset_name=spec.name,
                    scale=1.0,
                    spec=spec,
                    pose=Pose((pos_x, pos_y, pos_z), quat),
                )
            )
            wave_positions.append((pos_x, pos_y, self._planar_placement_radius(spec)))
        return wave_states

    def _sample_wave_drop_position(
        self,
        spec,
        wave_positions: list[tuple[float, float, float]],
        max_attempts: int = 48,
    ) -> tuple[float, float]:
        """Sample a wave-drop position that is centered but not unrealistically collapsed."""

        size_x, size_y, _ = self.config.bin.size
        wall = self.config.bin.wall_thickness
        placement_radius = self._planar_placement_radius(spec)
        clearance = wall * 2.0 + 0.003
        x_limit = max(0.0, size_x / 2.0 - clearance - placement_radius)
        y_limit = max(0.0, size_y / 2.0 - clearance - placement_radius)
        spawn_fraction = min(
            max(
                self.config.simulation.spawn_planar_fraction * _WAVE_PLANAR_FRACTION_SCALE,
                _MIN_WAVE_PLANAR_FRACTION,
            ),
            _MAX_WAVE_PLANAR_FRACTION,
        )
        x_limit *= spawn_fraction
        y_limit *= spawn_fraction
        best_candidate = (0.0, 0.0)
        best_clearance = float("-inf")
        for _ in range(max_attempts):
            pos_x = self._sample_wave_center_biased_coordinate(x_limit)
            pos_y = self._sample_wave_center_biased_coordinate(y_limit)
            min_clearance = float("inf")
            for placed_x, placed_y, placed_radius in wave_positions:
                center_distance = float(np.hypot(pos_x - placed_x, pos_y - placed_y))
                clearance_value = center_distance - (
                    _WAVE_OVERLAP_CLEARANCE_SCALE * (placement_radius + placed_radius)
                )
                min_clearance = min(min_clearance, clearance_value)
            if not wave_positions:
                return (pos_x, pos_y)
            if min_clearance >= 0.0:
                return (pos_x, pos_y)
            if min_clearance > best_clearance:
                best_candidate = (pos_x, pos_y)
                best_clearance = min_clearance
        return best_candidate

    def _sample_wave_center_biased_coordinate(self, limit: float) -> float:
        """Use a softer center bias than the old single-point pile generator."""

        if limit <= 0.0:
            return 0.0
        base_power = max(float(self.config.simulation.spawn_center_bias_power), 1.0)
        bias_power = max(1.0, base_power * _WAVE_CENTER_BIAS_POWER_SCALE)
        raw = float(self._rng.uniform(-1.0, 1.0))
        magnitude = abs(raw) ** bias_power
        return float(np.sign(raw) * limit * magnitude)

    def _sample_progressive_drop_height(
        self,
        spec,
        settled_states: list[SceneObjectState],
        body_quaternion_wxyz: tuple[float, float, float, float],
        wave_offset_index: int = 0,
    ) -> float:
        """Drop a new object above the current pile top instead of near the floor."""

        spawn_min, spawn_max = self.config.simulation.spawn_height_range
        vertical_half_extent = self._world_vertical_half_extent(
            spec=spec,
            body_quaternion_wxyz=body_quaternion_wxyz,
        )
        pile_top = 0.0
        if settled_states:
            pile_top = max(
                float(state.pose.position[2])
                + self._world_vertical_half_extent(
                    spec=state.spec,
                    body_quaternion_wxyz=state.pose.quaternion_wxyz,
                )
                for state in settled_states
            )
        base_height = max(float(spawn_min), pile_top + _PILE_TOP_MARGIN_M)
        extra_span = max(float(spawn_max) - float(spawn_min), _PILE_EXTRA_DROP_SPAN_M)
        wave_offset = 0.008 * float(wave_offset_index)
        return float(
            self._rng.uniform(base_height + wave_offset, base_height + wave_offset + extra_span)
            + vertical_half_extent
        )

    def _intermediate_settle_steps(self, num_objects: int) -> int:
        """Use shorter settle windows while the pile is still being built."""

        if num_objects <= 0:
            return _MIN_INTERMEDIATE_SETTLE_STEPS
        estimated = int(self.config.simulation.settle_steps / max(3, num_objects))
        return max(_MIN_INTERMEDIATE_SETTLE_STEPS, min(_MAX_INTERMEDIATE_SETTLE_STEPS, estimated))

    def _settle_states(
        self,
        states: list[SceneObjectState],
        settle_steps: int,
    ):
        """Build XML and settle one intermediate or final pile configuration."""

        scene_xml = build_bin_scene_xml(
            scene_objects=[
                (state.object_id, state.spec, state.pose)
                for state in states
            ],
            bin_config=self.config.bin,
            simulation_config=self.config.simulation,
        )
        return self.backend.settle_scene(
            scene_xml=scene_xml,
            body_names=[state.object_id for state in states],
            settle_steps=settle_steps,
            stability_window=self.config.simulation.stability_window,
            velocity_threshold=self.config.simulation.stability_velocity_threshold,
        )

    @staticmethod
    def _apply_settled_poses(
        states: list[SceneObjectState],
        body_poses: dict[str, Pose],
    ) -> list[SceneObjectState]:
        """Overwrite each object's pose with the latest settled result."""

        return [
            SceneObjectState(
                object_id=state.object_id,
                asset_name=state.asset_name,
                scale=state.scale,
                spec=state.spec,
                pose=body_poses[state.object_id],
            )
            for state in states
        ]

    def _sample_planar_position(
        self,
        size_x: float,
        size_y: float,
        wall: float,
        placement_radius: float,
        placed_xy: list[tuple[float, float, float]],
        max_attempts: int = 80,
    ) -> tuple[float, float]:
        """Sample an `(x, y)` spawn point that respects bin margins and loose packing."""

        clearance = wall * 2.0 + 0.004
        x_limit = size_x / 2.0 - clearance - placement_radius
        y_limit = size_y / 2.0 - clearance - placement_radius
        x_limit = max(0.0, x_limit)
        y_limit = max(0.0, y_limit)
        spawn_fraction = min(max(self.config.simulation.spawn_planar_fraction, 0.05), 1.0)
        x_limit *= spawn_fraction
        y_limit *= spawn_fraction

        best_candidate = (0.0, 0.0)
        best_clearance = float("-inf")
        for _ in range(max_attempts):
            pos_x = self._sample_center_biased_coordinate(x_limit)
            pos_y = self._sample_center_biased_coordinate(y_limit)
            min_clearance = float("inf")
            for placed_x, placed_y, placed_radius in placed_xy:
                center_distance = float(np.hypot(pos_x - placed_x, pos_y - placed_y))
                clearance_value = center_distance - (placement_radius + placed_radius)
                min_clearance = min(min_clearance, clearance_value)
            if not placed_xy:
                return (pos_x, pos_y)
            if min_clearance > 0.0:
                return (pos_x, pos_y)
            if min_clearance > best_clearance:
                best_candidate = (pos_x, pos_y)
                best_clearance = min_clearance
        return best_candidate

    def _sample_center_biased_coordinate(self, limit: float) -> float:
        """Sample one planar coordinate with an optional bias toward the bin center."""

        if limit <= 0.0:
            return 0.0
        bias_power = max(float(self.config.simulation.spawn_center_bias_power), 1.0)
        raw = float(self._rng.uniform(-1.0, 1.0))
        magnitude = abs(raw) ** bias_power
        return float(np.sign(raw) * limit * magnitude)

    @staticmethod
    def _proxy_half_extents(spec) -> tuple[float, float, float]:
        """Return axis-aligned proxy half extents for placement heuristics."""

        return spec.proxy_half_extents

    def _planar_placement_radius(self, spec) -> float:
        """Return a conservative planar placement radius for pre-settle spawning."""

        half_x, half_y, _ = self._proxy_half_extents(spec)
        return max(float(np.hypot(half_x, half_y)), 0.008)

    def _vertical_half_extent(self, spec) -> float:
        """Return the proxy half extent along the local vertical axis."""

        _, _, half_z = self._proxy_half_extents(spec)
        return max(half_z, 0.005)

    def _world_vertical_half_extent(
        self,
        spec,
        body_quaternion_wxyz: tuple[float, float, float, float],
    ) -> float:
        """Return a conservative vertical half extent in world coordinates."""

        half_extents = np.asarray(self._proxy_half_extents(spec), dtype=np.float64)
        rotation = self._rotation_matrix_from_wxyz(body_quaternion_wxyz)
        world_z_in_body = np.asarray((0.0, 0.0, 1.0), dtype=np.float64).dot(rotation)
        return max(float(np.sum(np.abs(world_z_in_body) * half_extents)), 0.005)

    def _out_of_bin_object_ids(
        self,
        final_states: list[SceneObjectState],
    ) -> list[str]:
        """Return object ids whose settled center of mass has spilled out of the bin.

        We intentionally use the settled body position here instead of a full
        oriented footprint. In dense bin clutter, many valid piled objects lean
        on the side walls or protrude above them. What we want to reject are
        true spill-out failures where the object's body frame has crossed the
        interior bin boundary or sunk below the floor.
        """

        half_x = float(self.config.bin.size[0]) * 0.5
        half_y = float(self.config.bin.size[1]) * 0.5
        tolerance = 1e-4
        floor_penetration_tolerance = 2e-3
        out_of_bin: list[str] = []
        for state in final_states:
            pos_x, pos_y, pos_z = (float(value) for value in state.pose.position)
            if (
                abs(pos_x) > half_x + tolerance
                or abs(pos_y) > half_y + tolerance
                or pos_z < -floor_penetration_tolerance
            ):
                out_of_bin.append(state.object_id)
        return sorted(out_of_bin)

    def _random_quaternion(self, max_tilt_deg: float = 22.0) -> tuple[float, float, float, float]:
        max_tilt_rad = float(np.deg2rad(max_tilt_deg))
        yaw = float(self._rng.uniform(0.0, 2.0 * np.pi))
        pitch = float(self._rng.uniform(-max_tilt_rad, max_tilt_rad))
        roll = float(self._rng.uniform(-max_tilt_rad, max_tilt_rad))
        return self._quaternion_multiply(
            self._axis_angle_to_quaternion((0.0, 0.0, 1.0), yaw),
            self._quaternion_multiply(
                self._axis_angle_to_quaternion((0.0, 1.0, 0.0), pitch),
                self._axis_angle_to_quaternion((1.0, 0.0, 0.0), roll),
            ),
        )

    @staticmethod
    def _axis_angle_to_quaternion(
        axis: tuple[float, float, float],
        angle: float,
    ) -> tuple[float, float, float, float]:
        """Convert an axis-angle pair to a normalized quaternion."""

        axis_array = np.asarray(axis, dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis_array))
        if axis_norm < 1e-8:
            return (1.0, 0.0, 0.0, 0.0)
        axis_array /= axis_norm
        sin_half = float(np.sin(angle / 2.0))
        return (
            float(np.cos(angle / 2.0)),
            float(axis_array[0] * sin_half),
            float(axis_array[1] * sin_half),
            float(axis_array[2] * sin_half),
        )

    @staticmethod
    def _quaternion_multiply(
        lhs: tuple[float, float, float, float],
        rhs: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Multiply two `wxyz` quaternions."""

        lw, lx, ly, lz = lhs
        rw, rx, ry, rz = rhs
        quat = np.asarray(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=np.float64,
        )
        quat /= np.linalg.norm(quat)
        return tuple(float(value) for value in quat.tolist())

    @staticmethod
    def _rotation_matrix_from_wxyz(
        quaternion_wxyz: tuple[float, float, float, float],
    ) -> np.ndarray:
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
