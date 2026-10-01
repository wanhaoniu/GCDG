"""Core dataset and pipeline data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any


class GraspType(str, Enum):
    """Supported grasp families."""

    PARALLEL_JAW = "parallel_jaw"
    SUCTION = "suction"


class FailureStage(str, Enum):
    """Stage-aware feasibility outcomes."""

    NONE = "none"
    UNREACHABLE = "unreachable"
    APPROACH_COLLISION = "approach_collision"
    CLOSING_OR_SEAL_FAILURE = "closing_or_seal_failure"
    LIFT_COLLISION = "lift_collision"
    UNSTABLE_LIFT = "unstable_lift"
    UNKNOWN = "unknown"


class PlanningStatus(str, Enum):
    """Status of the bounded minimal-blocker search."""

    NO_PROPOSALS = "no_proposals"
    ALREADY_FEASIBLE = "already_feasible"
    SOLVED_WITHIN_DEPTH = "solved_within_depth"
    UNSOLVED_WITHIN_DEPTH = "unsolved_within_depth"


class PlanningTerminalReason(str, Enum):
    """Reason that remains after bounded blocker search for one fixed grasp."""

    FEASIBLE = "feasible"
    BIN_WALL_ONLY = "bin_wall_only"
    BIN_WALL_AND_CLUTTER = "bin_wall_and_clutter"
    CLUTTER_ONLY_OR_DEPTH_LIMIT = "clutter_only_or_depth_limit"
    NO_BLOCKER_ATTRIBUTION = "no_blocker_attribution"


STAGE_ORDER: dict[FailureStage, int] = {
    FailureStage.UNREACHABLE: 0,
    FailureStage.APPROACH_COLLISION: 1,
    FailureStage.CLOSING_OR_SEAL_FAILURE: 2,
    FailureStage.LIFT_COLLISION: 3,
    FailureStage.UNSTABLE_LIFT: 3,
    FailureStage.NONE: 4,
    FailureStage.UNKNOWN: -1,
}


@dataclass(frozen=True)
class Pose:
    """Rigid pose represented by position and quaternion."""

    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, list[float]]:
        """Convert the pose into JSON-friendly lists."""

        return {
            "position": [float(v) for v in self.position],
            "quaternion_wxyz": [float(v) for v in self.quaternion_wxyz],
        }


@dataclass(frozen=True)
class ObjectSpec:
    """Description of a scene object asset.

    `primitive_type` and `size` define the proxy geometry used for lightweight
    physics and geometric checks. When `mesh_path` is present, the renderer may
    still visualize the object with its mesh instead of the proxy primitive.
    `mesh_offset` lets us shift the visual mesh so its bounding-box center
    aligns with the body frame used by the proxy geometry. When
    `proxy_quaternion_wxyz` is present, the lightweight primitive proxy is
    rotated relative to the body frame before physics and geometric checks.
    """

    name: str
    primitive_type: str
    size: tuple[float, ...]
    mass: float
    rgba: tuple[float, float, float, float]
    mesh_path: str | None = None
    collision_mesh_paths: tuple[str, ...] | None = None
    texture_path: str | None = None
    mesh_scale: tuple[float, float, float] | None = None
    mesh_offset: tuple[float, float, float] | None = None
    proxy_quaternion_wxyz: tuple[float, float, float, float] | None = None
    notes: str | None = None

    @property
    def bounding_radius(self) -> float:
        """Return a conservative radius used by lightweight geometry checks."""

        if self.primitive_type == "sphere":
            return float(self.size[0])
        if self.primitive_type in {"cylinder", "capsule"}:
            return float((self.size[0] ** 2 + self.size[1] ** 2) ** 0.5)
        return float(sum(axis * axis for axis in self.size) ** 0.5)

    @property
    def proxy_half_extents(self) -> tuple[float, float, float]:
        """Return body-frame AABB half extents for the primitive proxy.

        This is the conservative size that spawning and containment heuristics
        should use. When the primitive proxy is rotated relative to the body
        frame, we expand to the proxy's body-frame axis-aligned bounding box.
        """

        if self.primitive_type == "sphere":
            radius = float(self.size[0])
            return (radius, radius, radius)
        if self.primitive_type in {"cylinder", "capsule"}:
            base_half_extents = (float(self.size[0]), float(self.size[0]), float(self.size[1]))
        else:
            size = tuple(float(value) for value in self.size)
            if len(size) == 3:
                base_half_extents = size
            elif len(size) == 2:
                base_half_extents = (size[0], size[0], size[1])
            else:
                base_half_extents = (size[0], size[0], size[0])

        if self.proxy_quaternion_wxyz is None:
            return base_half_extents

        rotation = _rotation_matrix_from_wxyz(self.proxy_quaternion_wxyz)
        return tuple(
            sum(abs(rotation[row][col]) * base_half_extents[col] for col in range(3))
            for row in range(3)
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-friendly dictionary."""

        return asdict(self)


def _rotation_matrix_from_wxyz(
    quaternion_wxyz: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Convert a `wxyz` quaternion into a rotation matrix."""

    w, x, y, z = (float(value) for value in quaternion_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    w /= norm
    x /= norm
    y /= norm
    z /= norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


@dataclass(frozen=True)
class SceneObjectState:
    """Scene object with a concrete pose."""

    object_id: str
    asset_name: str
    scale: float
    spec: ObjectSpec
    pose: Pose

    @property
    def scaled_proxy_half_extents(self) -> tuple[float, float, float]:
        """Return proxy half extents after the instance scale is applied."""

        return tuple(float(value) * float(self.scale) for value in self.spec.proxy_half_extents)

    @property
    def scaled_bounding_radius(self) -> float:
        """Return the conservative proxy radius after applying scale."""

        return float(self.spec.bounding_radius) * float(self.scale)

    def bbox_3d_dict(self) -> dict[str, Any]:
        """Return a lightweight oriented 3D bounding box payload."""

        return {
            "center": [float(value) for value in self.pose.position],
            "quaternion_wxyz": [float(value) for value in self.pose.quaternion_wxyz],
            "half_extents": [float(value) for value in self.scaled_proxy_half_extents],
            "primitive_type": str(self.spec.primitive_type),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-friendly dictionary."""

        return {
            "object_id": self.object_id,
            "instance_id": self.object_id,
            "asset_name": self.asset_name,
            "mesh_id": self.asset_name,
            "scale": self.scale,
            "spec": self.spec.to_dict(),
            "pose": self.pose.to_dict(),
            "position": [float(value) for value in self.pose.position],
            "orientation": [float(value) for value in self.pose.quaternion_wxyz],
            "bbox_3d": self.bbox_3d_dict(),
            "geometry": {
                "primitive_type": str(self.spec.primitive_type),
                "proxy_half_extents": [float(value) for value in self.scaled_proxy_half_extents],
                "bounding_radius": float(self.scaled_bounding_radius),
            },
            "physical_attr": {
                "mass": float(self.spec.mass),
            },
        }


@dataclass(frozen=True)
class StableScene:
    """Stable clutter scene after MuJoCo settling."""

    scene_id: str
    objects: tuple[SceneObjectState, ...]
    target_ids: tuple[str, ...]
    bin_size: tuple[float, float, float]
    wall_thickness: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_object(self, object_id: str) -> SceneObjectState:
        """Fetch an object by id or raise a clear error."""

        for scene_object in self.objects:
            if scene_object.object_id == object_id:
                return scene_object
        raise KeyError(f"Object '{object_id}' not found in scene '{self.scene_id}'.")

    def non_target_ids(self, target_id: str) -> list[str]:
        """Return every object id except the requested target."""

        return [obj.object_id for obj in self.objects if obj.object_id != target_id]

    def without_objects(self, object_ids: set[str]) -> "StableScene":
        """Return a shallow-copied scene with a subset of objects removed."""

        kept = tuple(obj for obj in self.objects if obj.object_id not in object_ids)
        kept_targets = tuple(t for t in self.target_ids if t not in object_ids)
        metadata = dict(self.metadata)
        metadata["removed_object_ids"] = sorted(object_ids)
        return StableScene(
            scene_id=self.scene_id,
            objects=kept,
            target_ids=kept_targets,
            bin_size=self.bin_size,
            wall_thickness=self.wall_thickness,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the scene into a JSON-friendly dictionary."""

        return {
            "scene_id": self.scene_id,
            "target_ids": list(self.target_ids),
            "object_ids": [obj.object_id for obj in self.objects],
            "object_mesh_names": [obj.asset_name for obj in self.objects],
            "object_poses": [
                list(obj.pose.position) + list(obj.pose.quaternion_wxyz)
                for obj in self.objects
            ],
            "object_scales": [obj.scale for obj in self.objects],
            "bin_size": list(self.bin_size),
            "wall_thickness": self.wall_thickness,
            "objects": [obj.to_dict() for obj in self.objects],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class GraspProposal:
    """Candidate grasp proposal from AnyGrasp or SuctionNet style providers."""

    grasp_id: str
    target_id: str
    grasp_type: GraspType
    pose: Pose
    source: str
    proposal_score: float
    approach_vector: tuple[float, float, float]
    lift_vector: tuple[float, float, float] = (0.0, 0.0, 1.0)
    pregrasp_offset: float = 0.10
    jaw_width: float | None = None
    suction_radius: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-friendly dictionary."""

        rotation = _rotation_matrix_from_wxyz(self.pose.quaternion_wxyz)
        closing_dir = [float(rotation[row][0]) for row in range(3)]
        payload = {
            "grasp_id": self.grasp_id,
            "target_id": self.target_id,
            "grasp_type": self.grasp_type.value,
            "pose": self.pose.to_dict(),
            "grasp_pose": self.pose.to_dict(),
            "position": [float(value) for value in self.pose.position],
            "orientation": [float(value) for value in self.pose.quaternion_wxyz],
            "source": self.source,
            "proposal_source": self.source,
            "proposal_score": self.proposal_score,
            "score_init": self.proposal_score,
            "approach_vector": list(self.approach_vector),
            "approach_dir": list(self.approach_vector),
            "lift_vector": list(self.lift_vector),
            "lift_dir": list(self.lift_vector),
            "pregrasp_offset": self.pregrasp_offset,
            "jaw_width": self.jaw_width,
            "suction_radius": self.suction_radius,
            "metadata": self.metadata,
        }
        if self.grasp_type == GraspType.PARALLEL_JAW:
            payload["closing_dir"] = closing_dir
        else:
            payload["suction_normal"] = [float(-value) for value in self.approach_vector]
            payload["contact_center"] = [float(value) for value in self.pose.position]
        return payload


@dataclass(frozen=True)
class ValidationResult:
    """Feasibility result for a single grasp proposal."""

    grasp_id: str
    feasible: bool
    failure_stage: FailureStage
    stage_blockers: dict[str, list[str]]
    notes: str = ""

    @property
    def stage_rank(self) -> int:
        """Map the failure stage to an ordinal progress level."""

        return STAGE_ORDER[self.failure_stage]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-friendly dictionary."""

        return {
            "grasp_id": self.grasp_id,
            "feasible": self.feasible,
            "failure_stage": self.failure_stage.value,
            "stage_blockers": self.stage_blockers,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class DependencyLabel:
    """Single-object removal label for one object and one grasp proposal."""

    object_id: str
    grasp_id: str
    dep_any: bool
    dep_collision_approach: bool
    dep_collision_lift: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-friendly dictionary."""

        return {
            "object_id": self.object_id,
            "grasp_id": self.grasp_id,
            "dep_any": self.dep_any,
            "dep_collision_approach": self.dep_collision_approach,
            "dep_collision_lift": self.dep_collision_lift,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class PlanningLabel:
    """Per-grasp bounded minimal-blocker search result."""

    grasp_id: str
    minimal_blocker_set: tuple[str, ...]
    oracle_removal_sequence: tuple[str, ...]
    status: PlanningStatus
    terminal_reason: PlanningTerminalReason = PlanningTerminalReason.FEASIBLE
    terminal_failure_stage: FailureStage | None = None
    terminal_failure_blockers: tuple[str, ...] = tuple()
    terminal_failure_notes: str = ""

    @property
    def minimal_blocker_set_size(self) -> int:
        """Return the blocker set cardinality."""

        return len(self.minimal_blocker_set)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-friendly dictionary."""

        return {
            "grasp_id": self.grasp_id,
            "minimal_blocker_set": list(self.minimal_blocker_set),
            "minimal_blocker_set_size": self.minimal_blocker_set_size,
            "oracle_removal_sequence": list(self.oracle_removal_sequence),
            "status": self.status.value,
            "terminal_reason": self.terminal_reason.value,
            "terminal_failure_stage": (
                None if self.terminal_failure_stage is None else self.terminal_failure_stage.value
            ),
            "terminal_failure_blockers": list(self.terminal_failure_blockers),
            "terminal_failure_notes": self.terminal_failure_notes,
        }


@dataclass(frozen=True)
class PlanningSummary:
    """Target-level summary derived from per-grasp planning labels."""

    best_grasp_id: str | None
    minimal_blocker_set: tuple[str, ...]
    oracle_removal_sequence: tuple[str, ...]
    status: PlanningStatus
    status_counts: dict[str, int] = field(default_factory=dict)
    terminal_reason: PlanningTerminalReason = PlanningTerminalReason.FEASIBLE
    terminal_reason_counts: dict[str, int] = field(default_factory=dict)

    @property
    def minimal_blocker_set_size(self) -> int:
        """Return the blocker set cardinality for the best grasp."""

        return len(self.minimal_blocker_set)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-friendly dictionary."""

        return {
            "best_grasp_id": self.best_grasp_id,
            "minimal_blocker_set": list(self.minimal_blocker_set),
            "minimal_blocker_set_size": self.minimal_blocker_set_size,
            "oracle_removal_sequence": list(self.oracle_removal_sequence),
            "status": self.status.value,
            "status_counts": dict(self.status_counts),
            "terminal_reason": self.terminal_reason.value,
            "terminal_reason_counts": dict(self.terminal_reason_counts),
        }
