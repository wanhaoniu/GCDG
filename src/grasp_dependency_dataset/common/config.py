"""YAML-backed configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import load_yaml


@dataclass(frozen=True)
class BinConfig:
    """Bin geometry used by the clutter scene generator."""

    size: tuple[float, float, float]
    wall_thickness: float


@dataclass(frozen=True)
class SimulationConfig:
    """MuJoCo settling parameters."""

    timestep: float
    integrator: str
    settle_steps: int
    stability_window: int
    stability_velocity_threshold: float
    spawn_height_range: tuple[float, float]
    spawn_planar_fraction: float
    spawn_center_bias_power: float
    use_volume_scaled_mass: bool
    nominal_density_kgm3: float
    min_object_mass_kg: float
    max_object_mass_kg: float
    contact_friction: tuple[float, float, float]
    contact_solimp: tuple[float, float, float]
    contact_solref: tuple[float, float]
    drop_wave_size_max: int | None = None


@dataclass(frozen=True)
class SceneGenerationConfig:
    """Scene generation settings."""

    seed: int
    num_scenes: int
    scene_id_prefix: str
    object_source: str
    catalog_manifest: str
    excluded_asset_names: tuple[str, ...]
    num_objects_min: int
    num_objects_max: int
    num_targets_min: int
    num_targets_max: int
    bin: BinConfig
    simulation: SimulationConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SceneGenerationConfig":
        """Load the scene generation config from YAML."""

        payload = load_yaml(path)
        return cls(
            seed=int(payload["seed"]),
            num_scenes=int(payload["num_scenes"]),
            scene_id_prefix=str(payload["scene_id_prefix"]),
            object_source=str(payload["object_source"]),
            catalog_manifest=str(payload["catalog_manifest"]),
            excluded_asset_names=tuple(
                str(value) for value in payload.get("excluded_asset_names", [])
            ),
            num_objects_min=int(payload["num_objects_min"]),
            num_objects_max=int(payload["num_objects_max"]),
            num_targets_min=int(payload["num_targets_min"]),
            num_targets_max=int(payload["num_targets_max"]),
            bin=BinConfig(
                size=tuple(float(v) for v in payload["bin"]["size"]),
                wall_thickness=float(payload["bin"]["wall_thickness"]),
            ),
            simulation=SimulationConfig(
                timestep=float(payload["simulation"]["timestep"]),
                integrator=str(payload["simulation"].get("integrator", "Euler")),
                settle_steps=int(payload["simulation"]["settle_steps"]),
                stability_window=int(payload["simulation"]["stability_window"]),
                stability_velocity_threshold=float(
                    payload["simulation"]["stability_velocity_threshold"]
                ),
                spawn_height_range=tuple(
                    float(v) for v in payload["simulation"]["spawn_height_range"]
                ),
                spawn_planar_fraction=float(
                    payload["simulation"].get("spawn_planar_fraction", 1.0)
                ),
                spawn_center_bias_power=float(
                    payload["simulation"].get("spawn_center_bias_power", 1.0)
                ),
                use_volume_scaled_mass=bool(
                    payload["simulation"].get("use_volume_scaled_mass", False)
                ),
                nominal_density_kgm3=float(
                    payload["simulation"].get("nominal_density_kgm3", 350.0)
                ),
                min_object_mass_kg=float(
                    payload["simulation"].get("min_object_mass_kg", 0.03)
                ),
                max_object_mass_kg=float(
                    payload["simulation"].get("max_object_mass_kg", 0.45)
                ),
                contact_friction=tuple(
                    float(v)
                    for v in payload["simulation"].get(
                        "contact_friction",
                        [0.8, 0.05, 0.02],
                    )
                ),
                contact_solimp=tuple(
                    float(v)
                    for v in payload["simulation"].get(
                        "contact_solimp",
                        [0.95, 0.995, 0.0001],
                    )
                ),
                contact_solref=tuple(
                    float(v)
                    for v in payload["simulation"].get(
                        "contact_solref",
                        [0.005, 1.0],
                    )
                ),
                drop_wave_size_max=(
                    None
                    if payload["simulation"].get("drop_wave_size_max") is None
                    else int(payload["simulation"]["drop_wave_size_max"])
                ),
            ),
        )


@dataclass(frozen=True)
class ParallelJawValidationConfig:
    """Parallel jaw collision and geometry limits."""

    max_width: float
    finger_depth: float
    closing_clearance: float


@dataclass(frozen=True)
class SuctionValidationConfig:
    """Suction cup geometry limits."""

    cup_radius: float
    seal_tolerance: float


@dataclass(frozen=True)
class GraspValidationConfig:
    """Stage-wise feasibility validation settings."""

    consider_robot_body: bool
    consider_container_collision: bool
    enable_closing_check: bool
    robot_base_position: tuple[float, float, float]
    workspace_center: tuple[float, float, float]
    reach_radius: float
    min_height: float
    max_height: float
    pregrasp_offset: float
    lift_height: float
    arm_corridor_radius: float
    parallel_jaw_tool_radius: float
    suction_tool_radius: float
    container_wall_clearance: float
    approach_object_shrink_margin: float
    lift_object_shrink_margin: float
    parallel_jaw: ParallelJawValidationConfig
    suction: SuctionValidationConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GraspValidationConfig":
        """Load the grasp validation config from YAML."""

        payload = load_yaml(path)
        return cls(
            consider_robot_body=bool(payload.get("consider_robot_body", False)),
            consider_container_collision=bool(payload.get("consider_container_collision", True)),
            enable_closing_check=bool(payload.get("enable_closing_check", False)),
            robot_base_position=tuple(float(v) for v in payload["robot_base_position"]),
            workspace_center=tuple(float(v) for v in payload["workspace_center"]),
            reach_radius=float(payload["reach_radius"]),
            min_height=float(payload["min_height"]),
            max_height=float(payload["max_height"]),
            pregrasp_offset=float(payload["pregrasp_offset"]),
            lift_height=float(payload["lift_height"]),
            arm_corridor_radius=float(payload["arm_corridor_radius"]),
            parallel_jaw_tool_radius=float(
                payload.get("parallel_jaw_tool_radius", payload.get("tool_radius", 0.02))
            ),
            suction_tool_radius=float(
                payload.get("suction_tool_radius", payload.get("tool_radius", 0.02))
            ),
            container_wall_clearance=float(payload.get("container_wall_clearance", 0.006)),
            approach_object_shrink_margin=float(payload.get("approach_object_shrink_margin", 0.0)),
            lift_object_shrink_margin=float(payload.get("lift_object_shrink_margin", 0.0)),
            parallel_jaw=ParallelJawValidationConfig(
                max_width=float(payload["parallel_jaw"]["max_width"]),
                finger_depth=float(payload["parallel_jaw"]["finger_depth"]),
                closing_clearance=float(payload["parallel_jaw"]["closing_clearance"]),
            ),
            suction=SuctionValidationConfig(
                cup_radius=float(payload["suction"]["cup_radius"]),
                seal_tolerance=float(payload["suction"]["seal_tolerance"]),
            ),
        )


@dataclass(frozen=True)
class ProposalConfig:
    """Proposal generation backend configuration."""

    parallel_backend: str
    suction_backend: str
    parallel_top_k: int
    suction_top_k: int
    max_keep_per_type: int
    enable_similarity_filter: bool = True
    max_approach_angle_from_down_deg: float = 180.0
    post_generation_roi_filter_enabled: bool = True
    post_generation_roi_bbox_margin_px: int = 2
    post_generation_roi_mask_dilation_px: int = 2
    post_generation_roi_world_margin_m: float = 0.01
    similarity_position_threshold_m: float = 0.008
    similarity_approach_angle_deg: float = 10.0
    parallel_jaw_similarity_inplane_angle_deg: float = 15.0
    icra2026_root: str = "."
    icra2026_grasp_config: str = "configs/provider.yaml"
    render_width: int = 640
    render_height: int = 480
    render_fovy_deg: float = 45.0
    render_lookat: tuple[float, float, float] = (0.0, 0.0, 0.06)
    render_distance: float = 0.62
    render_azimuth_deg: float = -90.0
    render_elevation_deg: float = 55.0
    render_wall_height: float | None = None
    parallel_scene_generation_enabled: bool = True
    parallel_scene_raw_top_k_multiplier: float = 4.0
    parallel_scene_raw_top_k_min: int = 0
    parallel_target_only_generation_enabled: bool = False
    parallel_multiview_enabled: bool = False
    parallel_multiview_azimuth_deg: tuple[float, ...] = ()
    parallel_multiview_elevation_deg: tuple[float, ...] = ()
    parallel_multiview_distance: float | None = None
    parallel_multiview_voxel_size_m: float = 0.003
    parallel_multiview_max_points: int = 35000
    parallel_target_support_fill_enabled: bool = False
    parallel_target_support_fill_xy_step_m: float = 0.004
    parallel_target_support_fill_z_step_m: float = 0.004
    parallel_target_support_fill_max_points: int = 20000

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProposalConfig":
        """Load proposal backend settings from YAML."""

        payload = load_yaml(path)
        return cls(
            parallel_backend=str(payload["parallel_backend"]),
            suction_backend=str(payload["suction_backend"]),
            parallel_top_k=int(payload["parallel_top_k"]),
            suction_top_k=int(payload["suction_top_k"]),
            max_keep_per_type=int(payload["max_keep_per_type"]),
            enable_similarity_filter=bool(payload.get("enable_similarity_filter", True)),
            max_approach_angle_from_down_deg=float(
                payload.get("max_approach_angle_from_down_deg", 180.0)
            ),
            post_generation_roi_filter_enabled=bool(
                payload.get("post_generation_roi_filter_enabled", True)
            ),
            post_generation_roi_bbox_margin_px=int(
                payload.get("post_generation_roi_bbox_margin_px", 2)
            ),
            post_generation_roi_mask_dilation_px=int(
                payload.get("post_generation_roi_mask_dilation_px", 2)
            ),
            post_generation_roi_world_margin_m=float(
                payload.get("post_generation_roi_world_margin_m", 0.01)
            ),
            similarity_position_threshold_m=float(
                payload.get("similarity_position_threshold_m", 0.008)
            ),
            similarity_approach_angle_deg=float(
                payload.get("similarity_approach_angle_deg", 10.0)
            ),
            parallel_jaw_similarity_inplane_angle_deg=float(
                payload.get("parallel_jaw_similarity_inplane_angle_deg", 15.0)
            ),
            icra2026_root=str(payload.get("icra2026_root", ".")),
            icra2026_grasp_config=str(
                payload.get(
                    "icra2026_grasp_config",
                    "configs/provider.yaml",
                )
            ),
            render_width=int(payload.get("render_width", 640)),
            render_height=int(payload.get("render_height", 480)),
            render_fovy_deg=float(payload.get("render_fovy_deg", 45.0)),
            render_lookat=tuple(float(v) for v in payload.get("render_lookat", [0.0, 0.0, 0.06])),
            render_distance=float(payload.get("render_distance", 0.62)),
            render_azimuth_deg=float(payload.get("render_azimuth_deg", -90.0)),
            render_elevation_deg=float(payload.get("render_elevation_deg", 55.0)),
            render_wall_height=(
                None
                if payload.get("render_wall_height") is None
                else float(payload.get("render_wall_height"))
            ),
            parallel_scene_generation_enabled=bool(
                payload.get("parallel_scene_generation_enabled", True)
            ),
            parallel_scene_raw_top_k_multiplier=float(
                payload.get("parallel_scene_raw_top_k_multiplier", 4.0)
            ),
            parallel_scene_raw_top_k_min=int(payload.get("parallel_scene_raw_top_k_min", 0)),
            parallel_target_only_generation_enabled=bool(
                payload.get("parallel_target_only_generation_enabled", False)
            ),
            parallel_multiview_enabled=bool(payload.get("parallel_multiview_enabled", False)),
            parallel_multiview_azimuth_deg=tuple(
                float(value) for value in payload.get("parallel_multiview_azimuth_deg", [])
            ),
            parallel_multiview_elevation_deg=tuple(
                float(value) for value in payload.get("parallel_multiview_elevation_deg", [])
            ),
            parallel_multiview_distance=(
                None
                if payload.get("parallel_multiview_distance") is None
                else float(payload.get("parallel_multiview_distance"))
            ),
            parallel_multiview_voxel_size_m=float(
                payload.get("parallel_multiview_voxel_size_m", 0.003)
            ),
            parallel_multiview_max_points=int(
                payload.get("parallel_multiview_max_points", 35000)
            ),
            parallel_target_support_fill_enabled=bool(
                payload.get("parallel_target_support_fill_enabled", False)
            ),
            parallel_target_support_fill_xy_step_m=float(
                payload.get("parallel_target_support_fill_xy_step_m", 0.004)
            ),
            parallel_target_support_fill_z_step_m=float(
                payload.get("parallel_target_support_fill_z_step_m", 0.004)
            ),
            parallel_target_support_fill_max_points=int(
                payload.get("parallel_target_support_fill_max_points", 20000)
            ),
        )


@dataclass(frozen=True)
class ExportConfig:
    """Output writing and search limits."""

    output_root: str
    write_scene_snapshots: bool
    write_grasp_proposals: bool
    write_labels: bool
    write_manifests: bool
    max_blocker_search_depth: int
    enumerate_all_targets: bool
    include_targets_without_proposals: bool

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExportConfig":
        """Load export settings from YAML."""

        payload = load_yaml(path)
        return cls(
            output_root=str(payload["output_root"]),
            write_scene_snapshots=bool(payload["write_scene_snapshots"]),
            write_grasp_proposals=bool(payload["write_grasp_proposals"]),
            write_labels=bool(payload["write_labels"]),
            write_manifests=bool(payload["write_manifests"]),
            max_blocker_search_depth=int(payload["max_blocker_search_depth"]),
            enumerate_all_targets=bool(payload.get("enumerate_all_targets", True)),
            include_targets_without_proposals=bool(
                payload.get("include_targets_without_proposals", True)
            ),
        )


def load_config_bundle(config_dir: str | Path = "configs") -> dict[str, Any]:
    """Load the four main YAML config files from one directory."""

    config_root = Path(config_dir)
    return {
        "scene_generation": SceneGenerationConfig.from_yaml(
            config_root / "scene_generation.yaml"
        ),
        "grasp_validation": GraspValidationConfig.from_yaml(
            config_root / "grasp_validation.yaml"
        ),
        "proposal_sources": ProposalConfig.from_yaml(
            config_root / "proposal_sources.yaml"
        ),
        "dataset_export": ExportConfig.from_yaml(
            config_root / "dataset_export.yaml"
        ),
    }
