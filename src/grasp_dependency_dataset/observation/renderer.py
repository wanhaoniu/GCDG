"""Render RGB-D and target masks from stable MuJoCo scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from grasp_dependency_dataset.common.config import ProposalConfig
from grasp_dependency_dataset.common.types import StableScene

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    mujoco = None
    _MUJOCO_IMPORT_ERROR = exc
else:
    _MUJOCO_IMPORT_ERROR = None


@dataclass(frozen=True)
class RenderedTargetObservation:
    """RGB-D rendering and target-region metadata for one `(scene, target)` pair."""

    color: np.ndarray
    depth_m: np.ndarray
    target_mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    camera_to_world: np.ndarray
    camera_frame_id: str
    world_frame_id: str
    intrinsics: dict[str, float | int]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RenderCameraSpec:
    """Virtual camera parameters used for synthetic observation rendering."""

    width: int
    height: int
    fovy_deg: float
    lookat: tuple[float, float, float]
    distance: float
    azimuth_deg: float
    elevation_deg: float
    wall_height: float | None = None
    camera_name: str = "dataset_camera"
    bin_floor_rgba: tuple[float, float, float, float] = (0.85, 0.85, 0.85, 1.0)
    bin_wall_rgba: tuple[float, float, float, float] = (0.72, 0.72, 0.72, 1.0)

    @classmethod
    def from_proposal_config(cls, config: ProposalConfig) -> "RenderCameraSpec":
        """Build a camera spec from the proposal config."""

        return cls(
            width=config.render_width,
            height=config.render_height,
            fovy_deg=config.render_fovy_deg,
            lookat=config.render_lookat,
            distance=config.render_distance,
            azimuth_deg=config.render_azimuth_deg,
            elevation_deg=config.render_elevation_deg,
            wall_height=config.render_wall_height,
        )


class MujocoSceneObservationRenderer:
    """Off-screen MuJoCo renderer that exposes RGB-D and per-target masks."""

    def __init__(self, camera_spec: RenderCameraSpec) -> None:
        if mujoco is None:
            raise RuntimeError("MuJoCo is not installed in the active environment.") from _MUJOCO_IMPORT_ERROR
        self.camera_spec = camera_spec

    def render_target(self, scene: StableScene, target_id: str) -> RenderedTargetObservation:
        """Render a static scene and extract the observation for one target object."""

        camera_pose = _camera_pose_from_spec(self.camera_spec)
        scene_xml = _build_static_scene_xml(scene, self.camera_spec, camera_pose)
        return self._render_target_from_scene_xml(
            scene_xml=scene_xml,
            target_id=target_id,
            camera_pose=camera_pose,
            extra_metadata={},
        )

    def render_target_isolated(
        self,
        scene: StableScene,
        target_id: str,
    ) -> RenderedTargetObservation:
        """Render only the target object at its settled pose with a local support plane."""

        camera_pose = _camera_pose_from_spec(self.camera_spec)
        target_object = scene.get_object(target_id)
        scene_xml, metadata = _build_isolated_target_scene_xml(
            scene=scene,
            target_object=target_object,
            camera_spec=self.camera_spec,
            camera_pose=camera_pose,
        )
        return self._render_target_from_scene_xml(
            scene_xml=scene_xml,
            target_id=target_id,
            camera_pose=camera_pose,
            extra_metadata=metadata,
        )

    def _render_target_from_scene_xml(
        self,
        *,
        scene_xml: str,
        target_id: str,
        camera_pose: dict[str, np.ndarray],
        extra_metadata: dict[str, Any],
    ) -> RenderedTargetObservation:
        """Shared MuJoCo rendering path for one target-specific scene XML."""

        model = mujoco.MjModel.from_xml_string(scene_xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        renderer = mujoco.Renderer(model, self.camera_spec.height, self.camera_spec.width)
        try:
            renderer.update_scene(data, camera=self.camera_spec.camera_name)
            color = renderer.render().copy()

            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=self.camera_spec.camera_name)
            depth_m = renderer.render().copy()

            renderer.disable_depth_rendering()
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera=self.camera_spec.camera_name)
            segmentation = renderer.render().copy()

            target_geom_name = f"{target_id}_geom"
            target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, target_geom_name)
            target_mask = _extract_target_mask(segmentation, renderer.scene, target_geom_id)
        finally:
            renderer.close()

        bbox_xyxy = _bbox_from_mask(target_mask, color.shape[:2])
        intrinsics = _intrinsics_from_fovy(
            width=self.camera_spec.width,
            height=self.camera_spec.height,
            fovy_deg=self.camera_spec.fovy_deg,
        )
        return RenderedTargetObservation(
            color=color,
            depth_m=depth_m,
            target_mask=target_mask,
            bbox_xyxy=bbox_xyxy,
            camera_to_world=camera_pose["camera_to_world"],
            camera_frame_id=self.camera_spec.camera_name,
            world_frame_id="robot_base",
            intrinsics=intrinsics,
            metadata={
                "target_geom_id": int(target_geom_id),
                "camera_eye": [float(v) for v in camera_pose["eye"].tolist()],
                "camera_forward_world": [float(v) for v in camera_pose["forward"].tolist()],
                "target_mask_pixel_count": int(np.count_nonzero(target_mask)),
                **dict(extra_metadata),
            },
        )


def _camera_pose_from_spec(camera_spec: RenderCameraSpec) -> dict[str, np.ndarray]:
    lookat = np.asarray(camera_spec.lookat, dtype=np.float64).reshape(3)
    azimuth_rad = np.deg2rad(float(camera_spec.azimuth_deg))
    elevation_rad = np.deg2rad(float(camera_spec.elevation_deg))

    eye = lookat + np.array(
        [
            camera_spec.distance * np.cos(elevation_rad) * np.cos(azimuth_rad),
            camera_spec.distance * np.cos(elevation_rad) * np.sin(azimuth_rad),
            camera_spec.distance * np.sin(elevation_rad),
        ],
        dtype=np.float64,
    )

    forward = lookat - eye
    forward /= np.linalg.norm(forward)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

    x_right = np.cross(forward, world_up)
    x_right /= np.linalg.norm(x_right)
    y_up = np.cross(x_right, forward)
    y_up /= np.linalg.norm(y_up)
    y_down = -y_up

    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, 0] = x_right
    camera_to_world[:3, 1] = y_down
    camera_to_world[:3, 2] = forward
    camera_to_world[:3, 3] = eye

    return {
        "eye": eye,
        "forward": forward,
        "x_right": x_right,
        "y_up": y_up,
        "camera_to_world": camera_to_world,
    }


def _build_static_scene_xml(
    scene: StableScene,
    camera_spec: RenderCameraSpec,
    camera_pose: dict[str, np.ndarray],
) -> str:
    size_x, size_y, size_z = scene.bin_size
    wall = scene.wall_thickness
    wall_height = float(camera_spec.wall_height) if camera_spec.wall_height is not None else float(size_z)
    floor_half_z = wall / 2.0
    wall_half_z = wall_height / 2.0
    wall_half_y = (size_y + 2.0 * wall) / 2.0
    wall_half_x = (size_x + 2.0 * wall) / 2.0
    camera_pos = _vec(camera_pose["eye"])
    camera_xyaxes = _vec(np.concatenate([camera_pose["x_right"], camera_pose["y_up"]]))
    floor_rgba = _vec(camera_spec.bin_floor_rgba)
    wall_rgba = _vec(camera_spec.bin_wall_rgba)

    asset_tags = []
    seen_assets: set[str] = set()
    object_bodies = []
    for obj in scene.objects:
        asset_bundle = _mesh_asset_bundle(obj.spec)
        if asset_bundle and obj.spec.name not in seen_assets:
            asset_tags.append(asset_bundle)
            seen_assets.add(obj.spec.name)
        object_bodies.append(
            f"""
      <body name="{obj.object_id}" pos="{_vec(obj.pose.position)}" quat="{_vec(obj.pose.quaternion_wxyz)}">
        <geom name="{obj.object_id}_geom" {_geom_attributes(obj.spec)}/>
      </body>"""
        )

    return f"""
<mujoco model="grasp_dependency_observation">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset>
    {"".join(asset_tags)}
  </asset>
  <visual>
    <map znear="0.01" zfar="5"/>
    <global offwidth="{camera_spec.width}" offheight="{camera_spec.height}" fovy="{camera_spec.fovy_deg:.6f}"/>
    <headlight ambient="0.6 0.6 0.6" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
  </visual>
  <worldbody>
    <light name="light" pos="0 0 1.2" dir="0 0 -1"/>
    <camera name="{camera_spec.camera_name}" pos="{camera_pos}" xyaxes="{camera_xyaxes}" fovy="{camera_spec.fovy_deg:.6f}"/>
    <geom name="floor" type="box" pos="0 0 {-floor_half_z:.6f}" size="{size_x:.6f} {size_y:.6f} {floor_half_z:.6f}" rgba="{floor_rgba}"/>
    <geom name="wall_pos_x" type="box" pos="{(size_x + wall) / 2.0:.6f} 0 {wall_half_z:.6f}" size="{wall / 2.0:.6f} {wall_half_y:.6f} {wall_half_z:.6f}" rgba="{wall_rgba}"/>
    <geom name="wall_neg_x" type="box" pos="{-(size_x + wall) / 2.0:.6f} 0 {wall_half_z:.6f}" size="{wall / 2.0:.6f} {wall_half_y:.6f} {wall_half_z:.6f}" rgba="{wall_rgba}"/>
    <geom name="wall_pos_y" type="box" pos="0 {(size_y + wall) / 2.0:.6f} {wall_half_z:.6f}" size="{wall_half_x:.6f} {wall / 2.0:.6f} {wall_half_z:.6f}" rgba="{wall_rgba}"/>
    <geom name="wall_neg_y" type="box" pos="0 {-(size_y + wall) / 2.0:.6f} {wall_half_z:.6f}" size="{wall_half_x:.6f} {wall / 2.0:.6f} {wall_half_z:.6f}" rgba="{wall_rgba}"/>
    {"".join(object_bodies)}
  </worldbody>
</mujoco>
""".strip()


def _build_isolated_target_scene_xml(
    scene: StableScene,
    target_object,
    camera_spec: RenderCameraSpec,
    camera_pose: dict[str, np.ndarray],
) -> tuple[str, dict[str, Any]]:
    """Build a target-only XML that keeps the settled pose and adds a local support plane."""

    camera_pos = _vec(camera_pose["eye"])
    camera_xyaxes = _vec(np.concatenate([camera_pose["x_right"], camera_pose["y_up"]]))
    asset_bundle = _mesh_asset_bundle(target_object.spec)
    support_plane_height_world = _support_plane_height_world(target_object)
    support_half_extents_xy = _support_plane_half_extents_xy(target_object)
    support_thickness = 0.002
    support_plane_center_z = float(support_plane_height_world - support_thickness / 2.0)

    target_body = f"""
      <body name="{target_object.object_id}" pos="{_vec(target_object.pose.position)}" quat="{_vec(target_object.pose.quaternion_wxyz)}">
        <geom name="{target_object.object_id}_geom" {_geom_attributes(target_object.spec)}/>
      </body>"""

    scene_xml = f"""
<mujoco model="grasp_dependency_observation_target_only">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset>
    {asset_bundle}
  </asset>
  <visual>
    <map znear="0.01" zfar="5"/>
    <global offwidth="{camera_spec.width}" offheight="{camera_spec.height}" fovy="{camera_spec.fovy_deg:.6f}"/>
    <headlight ambient="0.6 0.6 0.6" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
  </visual>
  <worldbody>
    <light name="light" pos="0 0 1.2" dir="0 0 -1"/>
    <camera name="{camera_spec.camera_name}" pos="{camera_pos}" xyaxes="{camera_xyaxes}" fovy="{camera_spec.fovy_deg:.6f}"/>
    <geom
      name="support_plane"
      type="box"
      pos="{target_object.pose.position[0]:.6f} {target_object.pose.position[1]:.6f} {support_plane_center_z:.6f}"
      size="{support_half_extents_xy[0]:.6f} {support_half_extents_xy[1]:.6f} {support_thickness / 2.0:.6f}"
      rgba="0.82 0.82 0.82 1"
    />
    {target_body}
  </worldbody>
</mujoco>
""".strip()

    metadata = {
        "generation_scene_mode": "target_only_with_support_plane",
        "support_plane_height_world": float(support_plane_height_world),
        "support_plane_half_extents_xy": [
            float(support_half_extents_xy[0]),
            float(support_half_extents_xy[1]),
        ],
    }
    return scene_xml, metadata


def _geom_attributes(spec) -> str:
    if spec.mesh_path:
        material_attribute = (
            f'material="{_material_asset_name(spec)}" '
            if spec.texture_path
            else f'rgba="{_vec(spec.rgba)}" '
        )
        local_pos_attribute = (
            f'pos="{_vec(spec.mesh_offset)}" '
            if getattr(spec, "mesh_offset", None) is not None
            else ""
        )
        return (
            f'type="mesh" '
            f'mesh="{_mesh_asset_name(spec)}" '
            f'{local_pos_attribute}'
            f'{material_attribute}'
            'friction="0.8 0.05 0.02" '
            'solimp="0.95 0.995 0.0001" '
            'solref="0.005 1"'
        )
    if spec.primitive_type == "sphere":
        size = _vec((spec.size[0],))
    elif spec.primitive_type in {"cylinder", "capsule"}:
        size = _vec((spec.size[0], spec.size[1]))
    else:
        size = _vec(spec.size)
    return (
        f'type="{spec.primitive_type}" '
        f'size="{size}" '
        f'{_primitive_quat_attribute(spec)}'
        f'rgba="{_vec(spec.rgba)}" '
        'friction="0.8 0.05 0.02" '
        'solimp="0.95 0.995 0.0001" '
        'solref="0.005 1"'
    )


def _rotation_matrix_from_wxyz(quaternion_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    norm = np.linalg.norm([w, x, y, z])
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w /= norm
    x /= norm
    y /= norm
    z /= norm
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


def _object_world_proxy_half_extents(scene_object) -> np.ndarray:
    rotation = _rotation_matrix_from_wxyz(scene_object.pose.quaternion_wxyz)
    half_extents = np.asarray(scene_object.scaled_proxy_half_extents, dtype=np.float64).reshape(3)
    return np.abs(rotation).dot(half_extents.reshape(3, 1)).reshape(3)


def _support_plane_height_world(scene_object) -> float:
    world_half_extents = _object_world_proxy_half_extents(scene_object)
    return float(scene_object.pose.position[2] - world_half_extents[2])


def _support_plane_half_extents_xy(scene_object) -> tuple[float, float]:
    world_half_extents = _object_world_proxy_half_extents(scene_object)
    margin_xy = 0.08
    return (
        float(max(world_half_extents[0] + margin_xy, 0.12)),
        float(max(world_half_extents[1] + margin_xy, 0.12)),
    )


def _primitive_quat_attribute(spec) -> str:
    if getattr(spec, "mesh_path", None) or getattr(spec, "proxy_quaternion_wxyz", None) is None:
        return ""
    return f'quat="{_vec(spec.proxy_quaternion_wxyz)}" '


def _mesh_asset_name(spec) -> str:
    return f"mesh_{spec.name.lower().replace(' ', '_').replace('-', '_')}"


def _texture_asset_name(spec) -> str:
    return f"tex_{spec.name.lower().replace(' ', '_').replace('-', '_')}"


def _material_asset_name(spec) -> str:
    return f"mat_{spec.name.lower().replace(' ', '_').replace('-', '_')}"


def _mesh_asset_bundle(spec) -> str:
    if not spec.mesh_path:
        return ""
    mesh_scale = spec.mesh_scale or (1.0, 1.0, 1.0)
    asset_tags = [
        f'<mesh name="{_mesh_asset_name(spec)}" file="{spec.mesh_path}" '
        f'scale="{_vec(mesh_scale)}"/>'
    ]
    if spec.texture_path:
        asset_tags.append(
            f'<texture name="{_texture_asset_name(spec)}" type="2d" file="{spec.texture_path}"/>'
        )
        asset_tags.append(
            f'<material name="{_material_asset_name(spec)}" texture="{_texture_asset_name(spec)}" '
            'rgba="1 1 1 1" specular="0.15" shininess="0.1"/>'
        )
    return "".join(asset_tags)


def _intrinsics_from_fovy(width: int, height: int, fovy_deg: float) -> dict[str, float | int]:
    fovy_rad = np.deg2rad(float(fovy_deg))
    fy = 0.5 * float(height) / np.tan(fovy_rad / 2.0)
    fx = fy
    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0
    return {
        "width": int(width),
        "height": int(height),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "depth_scale": 1.0,
    }


def _extract_target_mask(
    segmentation: np.ndarray,
    scene: "mujoco.MjvScene",
    target_geom_id: int,
) -> np.ndarray:
    """Extract a target mask from MuJoCo segmentation output.

    MuJoCo's segmentation renderer encodes `(segid, objtype)` pairs in the last
    dimension. We therefore first resolve which visible `segid` entries in the
    rendered scene correspond to the requested model geom id.
    """

    geom_type_id = int(mujoco.mjtObj.mjOBJ_GEOM)
    target_segids = []
    for scene_geom_index in range(int(scene.ngeom)):
        scene_geom = scene.geoms[scene_geom_index]
        if int(scene_geom.objtype) != geom_type_id:
            continue
        if int(scene_geom.objid) != int(target_geom_id):
            continue
        segid = int(scene_geom.segid)
        if segid >= 0:
            target_segids.append(segid)

    if target_segids:
        return np.isin(segmentation[..., 0], np.asarray(target_segids, dtype=np.int32)) & (
            segmentation[..., 1] == geom_type_id
        )

    # Conservative fallback for unexpected MuJoCo versions.
    id_first = (segmentation[..., 0] == int(target_geom_id)) & (segmentation[..., 1] == geom_type_id)
    id_second = (segmentation[..., 1] == int(target_geom_id)) & (segmentation[..., 0] == geom_type_id)
    if np.any(id_first):
        return id_first
    if np.any(id_second):
        return id_second
    return np.zeros(segmentation.shape[:2], dtype=bool)


def _bbox_from_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    if not np.any(mask):
        height, width = image_shape
        return (0, 0, width, height)
    ys, xs = np.nonzero(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _vec(values) -> str:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return " ".join(f"{float(value):.6f}" for value in array.tolist())
