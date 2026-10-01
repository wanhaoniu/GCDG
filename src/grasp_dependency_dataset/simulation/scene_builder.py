"""MuJoCo XML construction for bin clutter scenes."""

from __future__ import annotations

import math

from grasp_dependency_dataset.common.config import BinConfig, SimulationConfig
from grasp_dependency_dataset.common.types import ObjectSpec, Pose


def _vec(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def _geom_size(spec: ObjectSpec) -> str:
    if spec.primitive_type == "sphere":
        return _vec((spec.size[0],))
    if spec.primitive_type in {"cylinder", "capsule"}:
        return _vec((spec.size[0], spec.size[1]))
    return _vec(spec.size)


def _geom_quat(spec: ObjectSpec) -> str:
    """Return an optional geom-local quaternion for oriented primitive proxies."""

    if spec.primitive_type == "mesh" or spec.proxy_quaternion_wxyz is None:
        return ""
    return f'quat="{_vec(spec.proxy_quaternion_wxyz)}" '


def _geom_attributes(spec: ObjectSpec) -> str:
    if spec.primitive_type == "mesh":
        if not spec.mesh_path:
            raise ValueError(f"Mesh object '{spec.name}' is missing mesh_path.")
        local_pos_attribute = (
            f'pos="{_vec(spec.mesh_offset)}" '
            if spec.mesh_offset is not None
            else ""
        )
        return (
            f'type="mesh" '
            f'mesh="{_mesh_asset_name(spec)}" '
            f'{local_pos_attribute}'
            f'mass="{spec.mass:.6f}" '
            f'rgba="{_vec(spec.rgba)}" '
            'friction="0.8 0.05 0.02" '
            'solimp="0.95 0.995 0.0001" '
            'solref="0.005 1"'
        )
    return (
        f'type="{spec.primitive_type}" '
        f'size="{_geom_size(spec)}" '
        f'{_geom_quat(spec)}'
        f'mass="{spec.mass:.6f}" '
        f'rgba="{_vec(spec.rgba)}" '
        'friction="0.8 0.05 0.02" '
        'solimp="0.95 0.995 0.0001" '
        'solref="0.005 1"'
    )


def _mesh_asset_name(spec: ObjectSpec) -> str:
    return f"mesh_{spec.name.lower().replace(' ', '_').replace('-', '_')}"


def _collision_mesh_asset_name(spec: ObjectSpec, piece_index: int) -> str:
    return f"{_mesh_asset_name(spec)}_collision_{piece_index:02d}"


def _mesh_asset_tag(spec: ObjectSpec) -> str:
    if spec.primitive_type != "mesh" or not spec.mesh_path:
        return ""
    mesh_scale = spec.mesh_scale or (1.0, 1.0, 1.0)
    return (
        f'<mesh name="{_mesh_asset_name(spec)}" file="{spec.mesh_path}" '
        f'scale="{_vec(mesh_scale)}"/>'
    )


def _collision_mesh_asset_tags(spec: ObjectSpec) -> list[str]:
    if not spec.collision_mesh_paths:
        return []
    mesh_scale = spec.mesh_scale or (1.0, 1.0, 1.0)
    return [
        f'<mesh name="{_collision_mesh_asset_name(spec, piece_index)}" file="{collision_mesh_path}" '
        f'scale="{_vec(mesh_scale)}"/>'
        for piece_index, collision_mesh_path in enumerate(spec.collision_mesh_paths)
    ]


def _effective_mass(spec: ObjectSpec, simulation_config: SimulationConfig) -> float:
    """Return the mass used during settling.

    The imported local proxy catalog currently carries near-uniform masses, which
    is fine for placeholders but causes unrealistic drop dynamics in clutter.
    When enabled, we derive a more physical mass from the proxy primitive volume
    and clamp it into a conservative range so large items feel heavier without
    making the simulation numerically brittle.
    """

    if not simulation_config.use_volume_scaled_mass:
        return float(spec.mass)

    volume_m3 = _proxy_volume_m3(spec)
    estimated_mass = volume_m3 * float(simulation_config.nominal_density_kgm3)
    estimated_mass = max(float(simulation_config.min_object_mass_kg), estimated_mass)
    estimated_mass = min(float(simulation_config.max_object_mass_kg), estimated_mass)
    return estimated_mass


def _proxy_volume_m3(spec: ObjectSpec) -> float:
    """Approximate one proxy primitive's volume in cubic meters."""

    if spec.primitive_type == "sphere":
        radius = float(spec.size[0])
        return (4.0 / 3.0) * math.pi * radius ** 3
    if spec.primitive_type == "cylinder":
        radius = float(spec.size[0])
        half_height = float(spec.size[1])
        return math.pi * radius ** 2 * (2.0 * half_height)
    if spec.primitive_type == "capsule":
        radius = float(spec.size[0])
        half_height = float(spec.size[1])
        cylinder_height = max(0.0, 2.0 * half_height - 2.0 * radius)
        return math.pi * radius ** 2 * cylinder_height + (4.0 / 3.0) * math.pi * radius ** 3
    if len(spec.size) >= 3:
        half_x, half_y, half_z = (float(value) for value in spec.size[:3])
        return (2.0 * half_x) * (2.0 * half_y) * (2.0 * half_z)
    if len(spec.size) == 2:
        half_r, half_z = (float(value) for value in spec.size)
        return math.pi * half_r ** 2 * (2.0 * half_z)
    radius = float(spec.size[0])
    return (4.0 / 3.0) * math.pi * radius ** 3


def _contact_attributes(simulation_config: SimulationConfig) -> str:
    """Serialize shared contact parameters for clutter settling geoms."""

    return (
        f'friction="{_vec(simulation_config.contact_friction)}" '
        f'solimp="{_vec(simulation_config.contact_solimp)}" '
        f'solref="{_vec(simulation_config.contact_solref)}"'
    )


def build_bin_scene_xml(
    scene_objects: list[tuple[str, ObjectSpec, Pose]],
    bin_config: BinConfig,
    simulation_config: SimulationConfig,
) -> str:
    """Build a MuJoCo XML scene with a single bin and multiple free objects."""

    size_x, size_y, size_z = bin_config.size
    wall = bin_config.wall_thickness
    floor_half_z = wall / 2.0
    wall_half_z = size_z / 2.0
    wall_half_y = (size_y + 2.0 * wall) / 2.0
    wall_half_x = (size_x + 2.0 * wall) / 2.0

    asset_tags = []
    seen_assets: set[str] = set()
    object_bodies = []
    for body_name, spec, pose in scene_objects:
        mesh_asset_tag = _mesh_asset_tag(spec)
        if mesh_asset_tag and spec.name not in seen_assets:
            asset_tags.append(mesh_asset_tag)
            seen_assets.add(spec.name)
        if spec.collision_mesh_paths and f"{spec.name}__collision" not in seen_assets:
            asset_tags.extend(_collision_mesh_asset_tags(spec))
            seen_assets.add(f"{spec.name}__collision")
        contact_attributes = _contact_attributes(simulation_config)
        mass_kg = _effective_mass(spec, simulation_config)
        if spec.collision_mesh_paths:
            local_pos_attribute = (
                f'pos="{_vec(spec.mesh_offset)}" '
                if spec.mesh_offset is not None
                else ""
            )
            piece_mass = mass_kg / float(len(spec.collision_mesh_paths))
            geom_lines = [
                f'<geom name="{body_name}_collision_{piece_index:02d}" '
                f'type="mesh" mesh="{_collision_mesh_asset_name(spec, piece_index)}" '
                f'{local_pos_attribute}'
                f'mass="{piece_mass:.6f}" '
                f'rgba="{_vec(spec.rgba)}" '
                f'{contact_attributes}/>'
                for piece_index, _ in enumerate(spec.collision_mesh_paths)
            ]
            geom_block = "\n        ".join(geom_lines)
        else:
            local_pos_attribute = (
                f'pos="{_vec(spec.mesh_offset)}" '
                if spec.primitive_type == "mesh" and spec.mesh_offset is not None
                else ""
            )
            if spec.primitive_type == "mesh":
                geom_attributes = (
                    f'type="mesh" '
                    f'mesh="{_mesh_asset_name(spec)}" '
                    f'{local_pos_attribute}'
                    f'mass="{mass_kg:.6f}" '
                    f'rgba="{_vec(spec.rgba)}" '
                    f'{contact_attributes}'
                )
            else:
                geom_attributes = (
                    f'type="{spec.primitive_type}" '
                    f'size="{_geom_size(spec)}" '
                    f'{_geom_quat(spec)}'
                    f'mass="{mass_kg:.6f}" '
                    f'rgba="{_vec(spec.rgba)}" '
                    f'{contact_attributes}'
                )
            geom_block = f'<geom name="{body_name}_geom" {geom_attributes}/>'
        object_bodies.append(
            f"""
      <body name="{body_name}" pos="{_vec(pose.position)}" quat="{_vec(pose.quaternion_wxyz)}">
        <freejoint name="{body_name}_joint"/>
        {geom_block}
      </body>"""
        )

    return f"""
<mujoco model="grasp_dependency_bin_clutter">
  <compiler angle="radian"/>
  <option timestep="{simulation_config.timestep:.6f}" gravity="0 0 -9.81" integrator="{simulation_config.integrator}"/>
  <size njmax="4000" nconmax="800"/>
  <asset>
    {"".join(asset_tags)}
  </asset>
  <visual>
    <headlight ambient="0.6 0.6 0.6" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
  </visual>
  <worldbody>
    <light name="light" pos="0 0 1.2" dir="0 0 -1"/>
    <geom name="floor" type="box" pos="0 0 {-floor_half_z:.6f}" size="{size_x:.6f} {size_y:.6f} {floor_half_z:.6f}" rgba="0.8 0.8 0.8 1" {_contact_attributes(simulation_config)}/>
    <geom name="wall_pos_x" type="box" pos="{(size_x + wall) / 2.0:.6f} 0 {wall_half_z:.6f}" size="{wall / 2.0:.6f} {wall_half_y:.6f} {wall_half_z:.6f}" rgba="0.7 0.7 0.7 1" {_contact_attributes(simulation_config)}/>
    <geom name="wall_neg_x" type="box" pos="{-(size_x + wall) / 2.0:.6f} 0 {wall_half_z:.6f}" size="{wall / 2.0:.6f} {wall_half_y:.6f} {wall_half_z:.6f}" rgba="0.7 0.7 0.7 1" {_contact_attributes(simulation_config)}/>
    <geom name="wall_pos_y" type="box" pos="0 {(size_y + wall) / 2.0:.6f} {wall_half_z:.6f}" size="{wall_half_x:.6f} {wall / 2.0:.6f} {wall_half_z:.6f}" rgba="0.7 0.7 0.7 1" {_contact_attributes(simulation_config)}/>
    <geom name="wall_neg_y" type="box" pos="0 {-(size_y + wall) / 2.0:.6f} {wall_half_z:.6f}" size="{wall_half_x:.6f} {wall / 2.0:.6f} {wall_half_z:.6f}" rgba="0.7 0.7 0.7 1" {_contact_attributes(simulation_config)}/>
    {"".join(object_bodies)}
  </worldbody>
</mujoco>
""".strip()
