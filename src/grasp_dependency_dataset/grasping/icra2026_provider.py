"""Integration with the prepared ICRA2026 AnyGrasp and SuctionNet wrappers."""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from grasp_dependency_dataset.common.config import ProposalConfig
from grasp_dependency_dataset.common.types import GraspProposal, GraspType, Pose, StableScene
from grasp_dependency_dataset.observation.renderer import (
    MujocoSceneObservationRenderer,
    RenderCameraSpec,
    RenderedTargetObservation,
)


@dataclass(frozen=True)
class _VisibleTargetPointCloud:
    """Visible target-only point cloud extracted from one rendered view."""

    observation: RenderedTargetObservation
    point_cloud_camera: np.ndarray
    roi_mask: np.ndarray
    roi_points_camera: np.ndarray
    roi_points_world: np.ndarray
    roi_colors: np.ndarray


@dataclass
class ICRA2026ProposalProviderBase:
    """Shared bridge code for reusing the local ICRA2026 grasp wrappers."""

    config: ProposalConfig
    end_effector_id: str
    grasp_type: GraspType
    source_name: str
    _manager: Any = field(init=False, repr=False, default=None)
    _types_module: Any = field(init=False, repr=False, default=None)
    _renderer: MujocoSceneObservationRenderer = field(init=False, repr=False)
    _renderer_pool: dict[str, MujocoSceneObservationRenderer] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )
    _observation_cache: dict[tuple[str, str, str], RenderedTargetObservation] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )
    _generation_observation_cache: dict[tuple[str, str, str], RenderedTargetObservation] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        self._prepare_runtime_environment()
        self._renderer = MujocoSceneObservationRenderer(
            RenderCameraSpec.from_proposal_config(self.config)
        )
        self._load_external_manager()

    def generate(self, scene: StableScene, target_id: str, top_k: int) -> list[GraspProposal]:
        """Render a scene observation, generate raw grasps, then crop to the target."""

        reference_observation = self._get_observation(scene, target_id)
        if int(np.count_nonzero(reference_observation.target_mask)) <= 0:
            return []

        generation_observation = (
            self._get_generation_observation(scene, target_id)
            if self._target_only_generation_enabled
            else reference_observation
        )

        target_object = scene.get_object(target_id)
        if self._scene_generation_enabled:
            sensor_frame, detection_result = self._build_scene_generation_external_inputs(
                generation_observation,
                target_label=target_object.asset_name,
                target_world_position=target_object.pose.position,
            )
        else:
            sensor_frame, detection_result = self._build_external_inputs(
                generation_observation,
                target_label=target_object.asset_name,
                target_world_position=target_object.pose.position,
                target_bounding_radius=target_object.scaled_bounding_radius,
            )
        raw_top_k = self._resolve_external_raw_top_k(top_k)
        self._override_external_top_k(raw_top_k)
        raw_candidates: list[Any] = []
        if self._multiview_generation_enabled:
            fused_inputs = self._build_multiview_generation_inputs(
                scene,
                target_id=target_id,
                primary_observation=generation_observation,
                target_label=target_object.asset_name,
                target_world_position=target_object.pose.position,
                target_bounding_radius=target_object.scaled_bounding_radius,
            )
            if fused_inputs is not None:
                multiview_result = self._generate_from_preprocessed_with_target_frame(
                    sensor_frame=fused_inputs.sensor_frame,
                    detection_result=fused_inputs.detection_result,
                    preprocessed_input=fused_inputs.preprocessed_input,
                    target_frame=fused_inputs.sensor_frame.world_frame_id,
                    support_plane_enabled=not self._disable_support_plane_for_generation,
                )
                raw_candidates.extend(list(getattr(multiview_result, "candidates", None) or []))
        singleview_result = self._manager.generate(
            sensor_frame=sensor_frame,
            detection_result=detection_result,
            end_effector_id=self.end_effector_id,
        )
        raw_candidates.extend(list(getattr(singleview_result, "candidates", None) or []))
        if not raw_candidates:
            return []
        raw_candidates.sort(
            key=lambda candidate: float(getattr(candidate, "score", 0.0)),
            reverse=True,
        )
        filtered_candidates = self._filter_external_candidates_by_target_roi(
            raw_candidates,
            observation=reference_observation,
            target_world_position=target_object.pose.position,
            target_bounding_radius=target_object.scaled_bounding_radius,
        )
        if not filtered_candidates:
            return []
        if top_k > 0:
            filtered_candidates = filtered_candidates[: int(top_k)]
        return [
            self._convert_candidate(
                candidate,
                target_id=target_id,
                index=index,
                camera_to_world=reference_observation.camera_to_world,
                world_frame_id=reference_observation.world_frame_id,
            )
            for index, candidate in enumerate(filtered_candidates)
        ]

    def _prepare_runtime_environment(self) -> None:
        icra_root = Path(self.config.icra2026_root).expanduser().resolve()
        env_prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix)).resolve()
        compat_bin = icra_root / "scripts" / "setup" / "nettools_compat"

        path_items = [
            str(compat_bin),
            str(env_prefix / "bin"),
            "/usr/local/cuda-12.8/bin",
            os.environ.get("PATH", ""),
        ]
        os.environ["PATH"] = ":".join(item for item in path_items if item)

        ld_items = [str(env_prefix / "lib"), os.environ.get("LD_LIBRARY_PATH", "")]
        os.environ["LD_LIBRARY_PATH"] = ":".join(item for item in ld_items if item)
        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
        os.environ.setdefault("OMP_NUM_THREADS", "8")

        icra_root_text = str(icra_root)
        if icra_root_text not in sys.path:
            sys.path.insert(0, icra_root_text)

    def _load_external_manager(self) -> None:
        from grasp_pose_generator.base.types import CameraIntrinsics, DetectionResult, SensorFrame
        from grasp_pose_generator.core.manager import GraspGeneratorManager

        manager = GraspGeneratorManager.from_yaml(self.config.icra2026_grasp_config)
        self._patch_anygrasp_runtime_loader(manager)
        manager.config = copy.deepcopy(manager.config)
        self._configure_manager_for_proposal_generation(manager.config)

        self._manager = manager
        self._types_module = {
            "CameraIntrinsics": CameraIntrinsics,
            "DetectionResult": DetectionResult,
            "SensorFrame": SensorFrame,
        }

    @staticmethod
    def _patch_anygrasp_runtime_loader(manager: Any) -> None:
        adapter = getattr(manager, "adapters", {}).get("anygrasp")
        if adapter is None:
            return
        adapter_cls = type(adapter)
        if bool(getattr(adapter_cls, "_icra2026_prefers_root_gsnet", False)):
            return
        original_resolver = getattr(adapter_cls, "_resolve_runtime_gsnet_path", None)
        if original_resolver is None:
            return

        def _resolve_runtime_gsnet_path(root: Path) -> Path:
            root_path = Path(root)
            root_binary = root_path / "gsnet.so"
            if root_binary.is_file():
                return root_binary
            return original_resolver(root_path)

        adapter_cls._resolve_runtime_gsnet_path = staticmethod(_resolve_runtime_gsnet_path)
        adapter_cls._icra2026_prefers_root_gsnet = True

    @property
    def _camera_world_frame_id(self) -> str:
        return "dataset_camera"

    @property
    def _scene_generation_enabled(self) -> bool:
        return bool(
            self.grasp_type == GraspType.PARALLEL_JAW
            and getattr(self.config, "parallel_scene_generation_enabled", True)
        )

    @property
    def _multiview_generation_enabled(self) -> bool:
        return bool(
            self.grasp_type == GraspType.PARALLEL_JAW
            and not self._scene_generation_enabled
            and getattr(self.config, "parallel_multiview_enabled", False)
        )

    @property
    def _target_only_generation_enabled(self) -> bool:
        return bool(
            self.grasp_type == GraspType.PARALLEL_JAW
            and not self._scene_generation_enabled
            and getattr(self.config, "parallel_target_only_generation_enabled", False)
        )

    def _get_observation(
        self,
        scene: StableScene,
        target_id: str,
        camera_spec: RenderCameraSpec | None = None,
    ) -> RenderedTargetObservation:
        spec = camera_spec or self._renderer.camera_spec
        cache_key = (scene.scene_id, target_id, spec.camera_name)
        cached = self._observation_cache.get(cache_key)
        if cached is not None:
            return cached
        rendered = self._get_renderer(spec).render_target(scene, target_id)
        self._observation_cache[cache_key] = rendered
        return rendered

    def _get_generation_observation(
        self,
        scene: StableScene,
        target_id: str,
        camera_spec: RenderCameraSpec | None = None,
    ) -> RenderedTargetObservation:
        spec = camera_spec or self._renderer.camera_spec
        cache_key = (scene.scene_id, target_id, spec.camera_name)
        cached = self._generation_observation_cache.get(cache_key)
        if cached is not None:
            return cached
        rendered = self._get_renderer(spec).render_target_isolated(scene, target_id)
        self._generation_observation_cache[cache_key] = rendered
        return rendered

    def _get_renderer(self, camera_spec: RenderCameraSpec) -> MujocoSceneObservationRenderer:
        primary_spec = self._renderer.camera_spec
        if camera_spec.camera_name == primary_spec.camera_name:
            return self._renderer

        cached = self._renderer_pool.get(camera_spec.camera_name)
        if cached is not None:
            return cached

        renderer = MujocoSceneObservationRenderer(camera_spec)
        self._renderer_pool[camera_spec.camera_name] = renderer
        return renderer

    def _configure_manager_for_proposal_generation(self, manager_config: dict[str, Any]) -> None:
        """Tune the external wrapper for offline proposal generation.

        AnyGrasp still benefits from support-plane augmentation when it receives
        a scene-level point cloud, while SuctionNet remains on its raw ROI depth.
        """

        support_plane_config = manager_config.setdefault("support_plane", {})
        runtime_config = manager_config.setdefault("runtime", {})
        algorithm_mapping = manager_config.setdefault("algorithm_mapping", {})

        # Proposal generation should preserve diversity rather than inherit the
        # more aggressive online robot-execution filters from the runtime system.
        support_plane_config["enabled"] = True
        runtime_config["top_down_filter_enabled"] = False
        runtime_config["score_threshold"] = 0.0
        runtime_config["target_frame"] = self._camera_world_frame_id

        for end_effector_id in ("gripper_main", "suction_main"):
            mapping = algorithm_mapping.get(end_effector_id)
            if mapping is None:
                continue
            mapping["target_frame"] = self._camera_world_frame_id
            mapping["score_threshold"] = 0.0

    @staticmethod
    def _build_detection_metadata(
        observation: RenderedTargetObservation,
        target_world_position: tuple[float, float, float] | None = None,
        target_bounding_radius: float | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "synthetic_scene": True,
            **observation.metadata,
        }
        if target_world_position is not None:
            reference_world = [float(value) for value in target_world_position]
            metadata["grasp_reference_world_position"] = reference_world
            metadata["support_plane_anchor_world"] = reference_world
            metadata.setdefault("fused_world_position", reference_world)
        if target_bounding_radius is not None and float(target_bounding_radius) > 0.0:
            metadata["grasp_reference_radius_m"] = float(target_bounding_radius)
        return metadata

    def _build_external_inputs(
        self,
        observation: RenderedTargetObservation,
        target_label: str,
        target_world_position: tuple[float, float, float] | None = None,
        target_bounding_radius: float | None = None,
    ):
        camera_intrinsics_cls = self._types_module["CameraIntrinsics"]
        sensor_frame_cls = self._types_module["SensorFrame"]
        detection_result_cls = self._types_module["DetectionResult"]

        intrinsics = camera_intrinsics_cls(**observation.intrinsics)
        detection_metadata = self._build_detection_metadata(
            observation,
            target_world_position=target_world_position,
            target_bounding_radius=target_bounding_radius,
        )
        sensor_frame = sensor_frame_cls(
            color=observation.color,
            depth=observation.depth_m,
            intrinsics=intrinsics,
            camera_to_world=observation.camera_to_world,
            camera_frame_id=observation.camera_frame_id,
            world_frame_id=observation.world_frame_id,
            metadata=dict(detection_metadata),
        )
        detection_result = detection_result_cls(
            bbox_xyxy=observation.bbox_xyxy,
            mask=observation.target_mask,
            label=target_label,
            detector_name="mujoco_segmentation",
            score=1.0,
            metadata=detection_metadata,
        )
        return sensor_frame, detection_result

    def _build_scene_generation_external_inputs(
        self,
        observation: RenderedTargetObservation,
        target_label: str,
        target_world_position: tuple[float, float, float] | None = None,
    ):
        """Build full-scene inputs for AnyGrasp, leaving target crop to post-filtering."""

        camera_intrinsics_cls = self._types_module["CameraIntrinsics"]
        sensor_frame_cls = self._types_module["SensorFrame"]
        detection_result_cls = self._types_module["DetectionResult"]

        intrinsics = camera_intrinsics_cls(**observation.intrinsics)
        height, width = np.asarray(observation.depth_m).shape
        detection_metadata = self._build_detection_metadata(observation)
        detection_metadata.update(
            {
                "disable_bbox_roi": True,
                "generation_roi_mode": "full_scene_valid_depth",
                "post_generation_target_crop": True,
            }
        )
        if target_world_position is not None:
            reference_world = [float(value) for value in target_world_position]
            detection_metadata["support_plane_anchor_world"] = reference_world
            detection_metadata["target_crop_reference_world_position"] = reference_world

        sensor_frame = sensor_frame_cls(
            color=observation.color,
            depth=observation.depth_m,
            intrinsics=intrinsics,
            camera_to_world=observation.camera_to_world,
            camera_frame_id=observation.camera_frame_id,
            world_frame_id=observation.world_frame_id,
            metadata=dict(detection_metadata),
        )
        detection_result = detection_result_cls(
            bbox_xyxy=(0, 0, int(width), int(height)),
            mask=None,
            label=target_label,
            detector_name="mujoco_scene_context",
            score=1.0,
            metadata=detection_metadata,
        )
        return sensor_frame, detection_result

    def _resolve_external_raw_top_k(self, top_k: int) -> int:
        requested_top_k = max(0, int(top_k))
        if not self._scene_generation_enabled:
            return requested_top_k

        multiplier = max(
            1.0,
            float(getattr(self.config, "parallel_scene_raw_top_k_multiplier", 4.0)),
        )
        minimum = max(0, int(getattr(self.config, "parallel_scene_raw_top_k_min", 0)))
        return max(requested_top_k, int(np.ceil(requested_top_k * multiplier)), minimum)

    def _override_external_top_k(self, top_k: int) -> None:
        mapping = self._manager.config["algorithm_mapping"][self.end_effector_id]
        mapping["top_k"] = int(top_k)
        mapping["score_threshold"] = 0.0
        mapping["top_down_filter_enabled"] = False
        adapter_name = str(mapping.get("adapter", "")).strip()
        algorithm_config = self._manager.config.setdefault("algorithms", {}).get(adapter_name)
        if adapter_name == "anygrasp" and isinstance(algorithm_config, dict):
            algorithm_config["native_top_k"] = max(
                int(algorithm_config.get("native_top_k", 0) or 0),
                int(top_k),
            )

    def _generate_from_preprocessed_with_target_frame(
        self,
        *,
        sensor_frame: Any,
        detection_result: Any,
        preprocessed_input: Any,
        target_frame: str,
        support_plane_enabled: bool | None = None,
    ) -> Any:
        runtime_config = self._manager.config.setdefault("runtime", {})
        mapping = self._manager.config["algorithm_mapping"][self.end_effector_id]
        support_plane_config = self._manager.config.setdefault("support_plane", {})
        original_runtime_target_frame = runtime_config.get("target_frame")
        original_mapping_target_frame = mapping.get("target_frame")
        original_support_plane_enabled = support_plane_config.get("enabled")
        runtime_config["target_frame"] = target_frame
        mapping["target_frame"] = target_frame
        if support_plane_enabled is not None:
            support_plane_config["enabled"] = bool(support_plane_enabled)
        try:
            return self._manager.generate_from_preprocessed(
                sensor_frame=sensor_frame,
                detection_result=detection_result,
                preprocessed_input=preprocessed_input,
                end_effector_id=self.end_effector_id,
            )
        finally:
            runtime_config["target_frame"] = original_runtime_target_frame
            mapping["target_frame"] = original_mapping_target_frame
            support_plane_config["enabled"] = original_support_plane_enabled

    def _build_multiview_generation_inputs(
        self,
        scene: StableScene,
        *,
        target_id: str,
        primary_observation: RenderedTargetObservation,
        target_label: str,
        target_world_position: tuple[float, float, float],
        target_bounding_radius: float,
    ) -> SimpleNamespace | None:
        view_clouds: list[_VisibleTargetPointCloud] = []
        plane_height_values: list[float] = []
        plane_height_weights: list[float] = []

        for camera_spec in self._make_multiview_camera_specs():
            observation = (
                self._get_generation_observation(scene, target_id, camera_spec=camera_spec)
                if self._target_only_generation_enabled
                else self._get_observation(scene, target_id, camera_spec=camera_spec)
            )
            cloud = self._extract_visible_target_point_cloud(observation)
            if cloud is None:
                continue
            view_clouds.append(cloud)
            plane_height_world = self._estimate_support_plane_height_world_from_observation(
                observation,
                target_label=target_label,
                target_world_position=target_world_position,
                target_bounding_radius=target_bounding_radius,
            )
            if plane_height_world is not None and np.isfinite(float(plane_height_world)):
                plane_height_values.append(float(plane_height_world))
                plane_height_weights.append(float(max(cloud.roi_points_camera.shape[0], 1)))

        if not view_clouds:
            return None

        fused_points_world = np.concatenate(
            [cloud.roi_points_world for cloud in view_clouds],
            axis=0,
        ).astype(np.float32)
        fused_colors = np.concatenate(
            [cloud.roi_colors for cloud in view_clouds],
            axis=0,
        ).astype(np.float32)
        fused_points_world, fused_colors = self._voxel_downsample_points_with_colors(
            fused_points_world,
            fused_colors,
            voxel_size_m=float(getattr(self.config, "parallel_multiview_voxel_size_m", 0.0)),
            max_points=int(getattr(self.config, "parallel_multiview_max_points", 0)),
        )

        detection_metadata = self._build_detection_metadata(
            primary_observation,
            target_world_position=target_world_position,
            target_bounding_radius=target_bounding_radius,
        )
        detection_metadata.update(
            {
                "multiview_generation_enabled": True,
                "multiview_camera_names": [cloud.observation.camera_frame_id for cloud in view_clouds],
                "multiview_view_count": int(len(view_clouds)),
                "multiview_point_count_raw": int(
                    sum(cloud.roi_points_world.shape[0] for cloud in view_clouds)
                ),
                "multiview_point_count_fused": int(fused_points_world.shape[0]),
            }
        )
        if plane_height_values:
            detection_metadata["support_plane_height_world"] = float(
                np.average(
                    np.asarray(plane_height_values, dtype=np.float64),
                    weights=np.asarray(plane_height_weights, dtype=np.float64),
                )
            )
        support_fill_metadata: dict[str, Any] = {"enabled": False, "reason": "disabled"}
        if self._target_only_generation_enabled:
            (
                fused_points_world,
                fused_colors,
                support_fill_metadata,
            ) = self._augment_target_only_world_cloud_with_support_fill(
                fused_points_world,
                fused_colors,
                support_plane_height_world=detection_metadata.get("support_plane_height_world"),
            )
        detection_metadata["target_only_support_fill"] = dict(support_fill_metadata)

        return self._build_fused_generation_inputs(
            target_label=target_label,
            target_world_position=target_world_position,
            target_bounding_radius=target_bounding_radius,
            target_world_points=fused_points_world,
            target_colors=fused_colors,
            primary_observation=primary_observation,
            detection_metadata=detection_metadata,
        )

    def _make_multiview_camera_specs(self) -> list[RenderCameraSpec]:
        primary = self._renderer.camera_spec
        azimuths = tuple(getattr(self.config, "parallel_multiview_azimuth_deg", ()) or ())
        if not azimuths:
            return [primary]

        elevations = tuple(getattr(self.config, "parallel_multiview_elevation_deg", ()) or ())
        distance = getattr(self.config, "parallel_multiview_distance", None)
        resolved_distance = float(primary.distance if distance is None else distance)
        seen_keys = {
            (
                round(float(primary.azimuth_deg), 6),
                round(float(primary.elevation_deg), 6),
                round(float(primary.distance), 6),
            )
        }
        specs: list[RenderCameraSpec] = [primary]

        for index, azimuth_deg in enumerate(azimuths):
            if len(elevations) == len(azimuths):
                elevation_deg = float(elevations[index])
            elif elevations:
                elevation_deg = float(elevations[0])
            else:
                elevation_deg = float(primary.elevation_deg)

            key = (
                round(float(azimuth_deg), 6),
                round(float(elevation_deg), 6),
                round(float(resolved_distance), 6),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            specs.append(
                RenderCameraSpec(
                    width=primary.width,
                    height=primary.height,
                    fovy_deg=primary.fovy_deg,
                    lookat=primary.lookat,
                    distance=resolved_distance,
                    azimuth_deg=float(azimuth_deg),
                    elevation_deg=float(elevation_deg),
                    wall_height=primary.wall_height,
                    camera_name=f"{primary.camera_name}_mv{len(specs):02d}",
                )
            )
        return specs

    def _extract_visible_target_point_cloud(
        self,
        observation: RenderedTargetObservation,
    ) -> _VisibleTargetPointCloud | None:
        point_cloud_camera = self._create_point_cloud_from_observation(observation)
        valid_depth = np.isfinite(observation.depth_m) & (np.asarray(observation.depth_m) > 1e-6)
        roi_mask = np.asarray(observation.target_mask, dtype=bool) & valid_depth
        if int(np.count_nonzero(roi_mask)) <= 0:
            return None

        roi_points_camera = np.asarray(point_cloud_camera[roi_mask], dtype=np.float32).reshape(-1, 3)
        roi_colors = np.asarray(observation.color[..., :3][roi_mask], dtype=np.float32).reshape(-1, 3)
        if roi_colors.size > 0:
            roi_colors = roi_colors / 255.0
        roi_points_world = self._transform_points_camera_to_world(
            roi_points_camera,
            observation.camera_to_world,
        )
        return _VisibleTargetPointCloud(
            observation=observation,
            point_cloud_camera=np.asarray(point_cloud_camera, dtype=np.float32),
            roi_mask=np.asarray(roi_mask, dtype=bool),
            roi_points_camera=roi_points_camera,
            roi_points_world=roi_points_world,
            roi_colors=roi_colors.astype(np.float32),
        )

    def _estimate_support_plane_height_world_from_observation(
        self,
        observation: RenderedTargetObservation,
        *,
        target_label: str,
        target_world_position: tuple[float, float, float],
        target_bounding_radius: float,
    ) -> float | None:
        from grasp_pose_generator.core.support_plane import estimate_support_plane_height_world

        sensor_frame, detection_result = self._build_external_inputs(
            observation,
            target_label=target_label,
            target_world_position=target_world_position,
            target_bounding_radius=target_bounding_radius,
        )
        point_cloud_camera = self._create_point_cloud_from_observation(observation)
        valid_depth = np.isfinite(observation.depth_m) & (np.asarray(observation.depth_m) > 1e-6)
        roi_mask = np.asarray(observation.target_mask, dtype=bool) & valid_depth
        roi_points_camera = np.asarray(point_cloud_camera[roi_mask], dtype=np.float32).reshape(-1, 3)
        roi_colors = np.asarray(observation.color[..., :3][roi_mask], dtype=np.float32).reshape(-1, 3)
        if roi_colors.size > 0:
            roi_colors = roi_colors / 255.0
        preprocessed_input = SimpleNamespace(
            depth_m=np.asarray(observation.depth_m, dtype=np.float32),
            point_cloud_camera=np.asarray(point_cloud_camera, dtype=np.float32),
            roi_mask=np.asarray(roi_mask, dtype=bool),
            roi_bbox_xyxy=tuple(int(value) for value in observation.bbox_xyxy),
            roi_points_camera=roi_points_camera.astype(np.float32),
            roi_colors=roi_colors.astype(np.float32),
            workspace_lims_camera=self._compute_workspace_lims(roi_points_camera),
        )
        height_world, _ = estimate_support_plane_height_world(
            sensor_frame=sensor_frame,
            detection_result=detection_result,
            preprocessed_input=preprocessed_input,
            support_plane_config=self._manager.config.get("support_plane", {}),
        )
        if height_world is None:
            return None
        return float(height_world)

    def _build_fused_generation_inputs(
        self,
        *,
        target_label: str,
        target_world_position: tuple[float, float, float],
        target_bounding_radius: float,
        target_world_points: np.ndarray,
        target_colors: np.ndarray,
        primary_observation: RenderedTargetObservation,
        detection_metadata: dict[str, Any],
    ) -> SimpleNamespace | None:
        point_array = np.asarray(target_world_points, dtype=np.float32).reshape(-1, 3)
        color_array = np.asarray(target_colors, dtype=np.float32).reshape(-1, 3)
        if point_array.shape[0] == 0:
            return None

        point_array_camera = self._transform_points_world_to_camera(
            point_array,
            primary_observation.camera_to_world,
        )
        if point_array_camera.shape[0] == 0:
            return None

        camera_intrinsics_cls = self._types_module["CameraIntrinsics"]
        sensor_frame_cls = self._types_module["SensorFrame"]
        detection_result_cls = self._types_module["DetectionResult"]

        intrinsics = camera_intrinsics_cls(
            width=int(primary_observation.intrinsics["width"]),
            height=int(primary_observation.intrinsics["height"]),
            fx=float(primary_observation.intrinsics["fx"]),
            fy=float(primary_observation.intrinsics["fy"]),
            cx=float(primary_observation.intrinsics["cx"]),
            cy=float(primary_observation.intrinsics["cy"]),
            depth_scale=float(primary_observation.intrinsics.get("depth_scale", 1.0)),
        )
        sensor_frame = sensor_frame_cls(
            color=np.zeros((1, 1, 3), dtype=np.uint8),
            depth=np.zeros((1, 1), dtype=np.float32),
            intrinsics=intrinsics,
            camera_to_world=np.asarray(primary_observation.camera_to_world, dtype=np.float64),
            camera_frame_id=str(primary_observation.camera_frame_id),
            world_frame_id=str(primary_observation.world_frame_id),
            metadata={
                **dict(detection_metadata),
                "multiview_generation_point_frame": str(primary_observation.camera_frame_id),
            },
        )
        detection_result = detection_result_cls(
            bbox_xyxy=(0, 0, 1, 1),
            mask=np.ones((1, 1), dtype=bool),
            label=target_label,
            detector_name="mujoco_multiview_fusion",
            score=1.0,
            metadata=dict(detection_metadata),
        )
        preprocessed_input = SimpleNamespace(
            depth_m=np.zeros((1, 1), dtype=np.float32),
            point_cloud_camera=point_array_camera.astype(np.float32),
            roi_mask=np.ones((1, 1), dtype=bool),
            roi_bbox_xyxy=(0, 0, 1, 1),
            roi_points_camera=point_array_camera.astype(np.float32),
            roi_colors=color_array.astype(np.float32),
            workspace_lims_camera=self._compute_workspace_lims(point_array_camera),
        )
        return SimpleNamespace(
            sensor_frame=sensor_frame,
            detection_result=detection_result,
            preprocessed_input=preprocessed_input,
        )

    def _augment_target_only_world_cloud_with_support_fill(
        self,
        points_world: np.ndarray,
        colors: np.ndarray,
        *,
        support_plane_height_world: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Fill the gap between the target and its local support plane.

        Target-only multiview fusion otherwise presents many objects as floating
        above the support plane, which encourages side grasps. We rasterize the
        fused target footprint in XY and append a compact vertical support
        volume from the support-plane height up to the lowest observed target
        point in each occupied cell.
        """

        if not bool(getattr(self.config, "parallel_target_support_fill_enabled", False)):
            return (
                np.asarray(points_world, dtype=np.float32).reshape(-1, 3),
                np.asarray(colors, dtype=np.float32).reshape(-1, 3),
                {"enabled": False, "reason": "disabled"},
            )

        point_array = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
        color_array = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
        if point_array.shape[0] == 0:
            return point_array, color_array, {"enabled": False, "reason": "empty_target_points"}

        try:
            plane_height = float(support_plane_height_world)
        except Exception:
            return point_array, color_array, {"enabled": False, "reason": "missing_plane_height"}
        if not np.isfinite(plane_height):
            return point_array, color_array, {"enabled": False, "reason": "invalid_plane_height"}

        xy_step_m = max(
            0.002,
            float(getattr(self.config, "parallel_target_support_fill_xy_step_m", 0.004)),
        )
        z_step_m = max(
            0.002,
            float(getattr(self.config, "parallel_target_support_fill_z_step_m", 0.004)),
        )
        max_points = max(
            512,
            int(getattr(self.config, "parallel_target_support_fill_max_points", 20000)),
        )

        xy_indices = np.floor(point_array[:, :2] / float(xy_step_m)).astype(np.int64)
        unique_xy, inverse = np.unique(xy_indices, axis=0, return_inverse=True)
        if unique_xy.shape[0] == 0:
            return point_array, color_array, {"enabled": False, "reason": "empty_xy_footprint"}

        counts = np.bincount(inverse, minlength=unique_xy.shape[0]).astype(np.float64)
        xy_centers = np.zeros((unique_xy.shape[0], 2), dtype=np.float64)
        for axis in range(2):
            xy_centers[:, axis] = np.bincount(
                inverse,
                weights=point_array[:, axis].astype(np.float64),
                minlength=unique_xy.shape[0],
            ) / np.maximum(counts, 1.0)

        min_z_by_cell = np.full(unique_xy.shape[0], np.inf, dtype=np.float64)
        np.minimum.at(min_z_by_cell, inverse, point_array[:, 2].astype(np.float64))

        occupied_cells = {tuple(int(value) for value in cell.tolist()) for cell in unique_xy}
        plane_points = np.column_stack(
            [
                xy_centers[:, 0].astype(np.float32),
                xy_centers[:, 1].astype(np.float32),
                np.full(unique_xy.shape[0], plane_height, dtype=np.float32),
            ]
        ).astype(np.float32)

        support_surfaces: list[np.ndarray] = [plane_points]
        boundary_cell_count = 0
        for cell_index, cell in enumerate(unique_xy):
            cell_key = tuple(int(value) for value in cell.tolist())
            is_boundary = False
            for offset in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor_key = (cell_key[0] + offset[0], cell_key[1] + offset[1])
                if neighbor_key not in occupied_cells:
                    is_boundary = True
                    break
            if not is_boundary:
                continue
            boundary_cell_count += 1

            top_z = float(min_z_by_cell[cell_index])
            if not np.isfinite(top_z) or top_z <= plane_height + 1e-4:
                continue
            z_values = np.arange(
                plane_height,
                top_z + 0.5 * z_step_m,
                z_step_m,
                dtype=np.float32,
            )
            if z_values.size == 0 or float(z_values[-1]) < top_z - 1e-4:
                z_values = np.concatenate(
                    [z_values, np.asarray([top_z], dtype=np.float32)],
                    axis=0,
                )
            x_value = np.full(z_values.shape[0], float(xy_centers[cell_index, 0]), dtype=np.float32)
            y_value = np.full(z_values.shape[0], float(xy_centers[cell_index, 1]), dtype=np.float32)
            support_surfaces.append(
                np.stack([x_value, y_value, z_values.astype(np.float32)], axis=1).astype(np.float32)
            )

        if not support_surfaces:
            return point_array, color_array, {"enabled": False, "reason": "no_support_surfaces"}

        support_points = np.concatenate(support_surfaces, axis=0).astype(np.float32)
        support_colors = np.repeat(
            np.asarray([[0.60, 0.60, 0.60]], dtype=np.float32),
            support_points.shape[0],
            axis=0,
        )
        support_points, support_colors = self._voxel_downsample_points_with_colors(
            support_points,
            support_colors,
            voxel_size_m=min(xy_step_m, z_step_m),
            max_points=max_points,
        )

        augmented_points = np.concatenate([point_array, support_points], axis=0).astype(np.float32)
        augmented_colors = np.concatenate([color_array, support_colors], axis=0).astype(np.float32)
        metadata = {
            "enabled": True,
            "plane_height_world": float(plane_height),
            "xy_step_m": float(xy_step_m),
            "z_step_m": float(z_step_m),
            "occupied_xy_cell_count": int(unique_xy.shape[0]),
            "boundary_xy_cell_count": int(boundary_cell_count),
            "support_point_count": int(support_points.shape[0]),
        }
        return augmented_points, augmented_colors, metadata

    @property
    def _disable_support_plane_for_generation(self) -> bool:
        """Avoid synthetic-plane domination for uncluttered multiview target clouds.

        In the target-only multiview setting, the fused object cloud already
        provides enough geometry for top-down parallel-jaw generation. Appending
        another synthetic support plane tends to dominate AnyGrasp and produce
        plane grasps instead of object grasps.
        """

        return bool(self._target_only_generation_enabled and self._multiview_generation_enabled)

    @staticmethod
    def _create_point_cloud_from_observation(
        observation: RenderedTargetObservation,
    ) -> np.ndarray:
        intrinsics = observation.intrinsics
        depth_m = np.asarray(observation.depth_m, dtype=np.float32)
        height, width = depth_m.shape
        xx, yy = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        z = depth_m
        x = (xx - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
        y = (yy - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
        return np.stack([x, y, z], axis=-1).astype(np.float32)

    @staticmethod
    def _transform_points_camera_to_world(
        points_camera: np.ndarray,
        camera_to_world: np.ndarray,
    ) -> np.ndarray:
        point_array = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
        if point_array.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        homogeneous = np.concatenate(
            [point_array, np.ones((point_array.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        return homogeneous.dot(np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4).T)[:, :3].astype(np.float32)

    @staticmethod
    def _transform_points_world_to_camera(
        points_world: np.ndarray,
        camera_to_world: np.ndarray,
    ) -> np.ndarray:
        point_array = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        if point_array.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        world_to_camera = np.linalg.inv(
            np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4)
        )
        homogeneous = np.concatenate(
            [point_array, np.ones((point_array.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        return homogeneous.dot(world_to_camera.T)[:, :3].astype(np.float32)

    @staticmethod
    def _compute_workspace_lims(points: np.ndarray) -> tuple[float, float, float, float, float, float]:
        point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if point_array.shape[0] == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        mins = point_array.min(axis=0)
        maxs = point_array.max(axis=0)
        return (
            float(mins[0]),
            float(maxs[0]),
            float(mins[1]),
            float(maxs[1]),
            float(mins[2]),
            float(maxs[2]),
        )

    @staticmethod
    def _voxel_downsample_points_with_colors(
        points: np.ndarray,
        colors: np.ndarray,
        *,
        voxel_size_m: float,
        max_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        color_array = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
        if point_array.shape[0] == 0:
            return point_array.reshape(0, 3), color_array.reshape(0, 3)

        if voxel_size_m > 0.0 and point_array.shape[0] > 1:
            voxel_indices = np.floor(point_array / float(voxel_size_m)).astype(np.int64)
            unique_voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
            fused_points = np.zeros((unique_voxels.shape[0], 3), dtype=np.float64)
            fused_colors = np.zeros((unique_voxels.shape[0], 3), dtype=np.float64)
            counts = np.bincount(inverse)
            for axis in range(3):
                fused_points[:, axis] = np.bincount(
                    inverse,
                    weights=point_array[:, axis],
                    minlength=unique_voxels.shape[0],
                )
                fused_colors[:, axis] = np.bincount(
                    inverse,
                    weights=color_array[:, axis],
                    minlength=unique_voxels.shape[0],
                )
            fused_points /= counts.reshape(-1, 1)
            fused_colors /= counts.reshape(-1, 1)
            point_array = fused_points.astype(np.float32)
            color_array = fused_colors.astype(np.float32)

        if max_points > 0 and point_array.shape[0] > int(max_points):
            step = max(1, int(np.ceil(point_array.shape[0] / float(max_points))))
            indices = np.arange(0, point_array.shape[0], step, dtype=np.int64)[: int(max_points)]
            point_array = point_array[indices]
            color_array = color_array[indices]

        return point_array.astype(np.float32), color_array.astype(np.float32)

    def _filter_external_candidates_by_target_roi(
        self,
        candidates: list[Any],
        *,
        observation: RenderedTargetObservation,
        target_world_position: tuple[float, float, float] | None = None,
        target_bounding_radius: float | None = None,
    ) -> list[Any]:
        """Keep only grasps whose center falls inside the target crop.

        Scene-level AnyGrasp is intentionally allowed to see all visible depth.
        This post-filter is the crop stage: candidates must re-project onto the
        target instance mask and, when available, remain near the target's world
        position.
        """

        if not bool(getattr(self.config, "post_generation_roi_filter_enabled", True)):
            return list(candidates)

        target_mask = np.asarray(observation.target_mask, dtype=bool)
        if int(np.count_nonzero(target_mask)) <= 0:
            return list(candidates)

        bbox_margin_px = max(
            0,
            int(getattr(self.config, "post_generation_roi_bbox_margin_px", 0)),
        )
        mask_dilation_px = max(
            0,
            int(getattr(self.config, "post_generation_roi_mask_dilation_px", 0)),
        )
        x0, y0, x1, y1 = (int(value) for value in observation.bbox_xyxy)
        mask_rows, mask_cols = np.nonzero(target_mask)
        mask_radius_sq = float(mask_dilation_px * mask_dilation_px)
        use_world_crop = target_world_position is not None and target_bounding_radius is not None
        target_world = (
            np.asarray(target_world_position, dtype=np.float64).reshape(3)
            if use_world_crop
            else None
        )
        world_crop_radius = (
            float(target_bounding_radius)
            + max(0.0, float(getattr(self.config, "post_generation_roi_world_margin_m", 0.04)))
            if use_world_crop
            else 0.0
        )

        kept: list[Any] = []
        for candidate in candidates:
            if use_world_crop and target_world is not None:
                candidate_position_world = self._candidate_position_in_world_frame(
                    candidate,
                    observation=observation,
                )
                if candidate_position_world is None:
                    continue
                if float(np.linalg.norm(candidate_position_world - target_world)) > world_crop_radius:
                    continue

            candidate_position_camera = self._candidate_position_in_camera_frame(
                candidate,
                observation=observation,
            )
            if candidate_position_camera is None:
                continue
            projected_uv = self._project_camera_point_to_image(
                candidate_position_camera,
                intrinsics=observation.intrinsics,
            )
            if projected_uv is None:
                continue
            u, v = projected_uv
            if not (
                float(x0 - bbox_margin_px) <= u < float(x1 + bbox_margin_px)
                and float(y0 - bbox_margin_px) <= v < float(y1 + bbox_margin_px)
            ):
                continue

            row = int(round(v))
            col = int(round(u))
            if 0 <= row < target_mask.shape[0] and 0 <= col < target_mask.shape[1] and bool(target_mask[row, col]):
                kept.append(candidate)
                continue

            if mask_dilation_px <= 0:
                continue

            distance_sq = (mask_rows.astype(np.float64) - float(v)) ** 2 + (
                mask_cols.astype(np.float64) - float(u)
            ) ** 2
            if distance_sq.size > 0 and float(np.min(distance_sq)) <= mask_radius_sq:
                kept.append(candidate)
        return kept

    @staticmethod
    def _candidate_position_in_world_frame(
        candidate: Any,
        *,
        observation: RenderedTargetObservation,
    ) -> np.ndarray | None:
        position = np.asarray(getattr(candidate, "position", ()), dtype=np.float64).reshape(-1)
        if position.shape != (3,) or not np.isfinite(position).all():
            return None

        pose_frame = str(getattr(candidate, "pose_frame", observation.camera_frame_id) or "").strip()
        if pose_frame == observation.world_frame_id:
            return position.astype(np.float64)
        camera_to_world = np.asarray(observation.camera_to_world, dtype=np.float64).reshape(4, 4)
        homogeneous = np.concatenate([position.astype(np.float64), np.ones(1, dtype=np.float64)])
        return camera_to_world.dot(homogeneous)[:3]

    @staticmethod
    def _candidate_position_in_camera_frame(
        candidate: Any,
        *,
        observation: RenderedTargetObservation,
    ) -> np.ndarray | None:
        position = np.asarray(getattr(candidate, "position", ()), dtype=np.float64).reshape(-1)
        if position.shape != (3,) or not np.isfinite(position).all():
            return None

        pose_frame = str(getattr(candidate, "pose_frame", observation.camera_frame_id) or "").strip()
        if not pose_frame or pose_frame == observation.camera_frame_id:
            return position.astype(np.float64)
        if pose_frame == observation.world_frame_id:
            world_to_camera = np.linalg.inv(
                np.asarray(observation.camera_to_world, dtype=np.float64).reshape(4, 4)
            )
            homogeneous = np.concatenate([position.astype(np.float64), np.ones(1, dtype=np.float64)])
            return world_to_camera.dot(homogeneous)[:3]
        return position.astype(np.float64)

    @staticmethod
    def _project_camera_point_to_image(
        position_camera: np.ndarray,
        *,
        intrinsics: dict[str, float | int],
    ) -> tuple[float, float] | None:
        point = np.asarray(position_camera, dtype=np.float64).reshape(-1)
        if point.shape != (3,) or not np.isfinite(point).all() or float(point[2]) <= 1e-6:
            return None

        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        u = fx * float(point[0]) / float(point[2]) + cx
        v = fy * float(point[1]) / float(point[2]) + cy
        if not np.isfinite(u) or not np.isfinite(v):
            return None
        return float(u), float(v)

    def _convert_candidate(
        self,
        candidate: Any,
        target_id: str,
        index: int,
        camera_to_world: np.ndarray,
        world_frame_id: str,
    ) -> GraspProposal:
        pose_frame = str(getattr(candidate, "pose_frame", "") or "").strip()
        quaternion_xyzw = tuple(float(value) for value in candidate.quaternion)
        rotation = _rotation_matrix_from_xyzw(quaternion_xyzw)

        if pose_frame == str(world_frame_id or "").strip():
            position_world = np.asarray(candidate.position, dtype=np.float64).reshape(3)
            rotation_world = rotation
        else:
            position_camera = np.asarray(candidate.position, dtype=np.float64).reshape(3)
            rotation_world = camera_to_world[:3, :3].dot(rotation)
            position_world = camera_to_world[:3, :3].dot(position_camera) + camera_to_world[:3, 3]
        approach_world = rotation_world[:, 2]
        quaternion_world_wxyz = _wxyz_from_rotation_matrix(rotation_world)

        jaw_width = None
        if hasattr(candidate, "metadata"):
            jaw_width = candidate.metadata.get("width_m")
        suction_radius = 0.018 if self.grasp_type == GraspType.SUCTION else None

        return GraspProposal(
            grasp_id=f"{target_id}_{self.grasp_type.value}_{index:02d}",
            target_id=target_id,
            grasp_type=self.grasp_type,
            pose=Pose(
                position=tuple(float(value) for value in position_world.tolist()),
                quaternion_wxyz=quaternion_world_wxyz,
            ),
            source=self.source_name,
            proposal_score=float(candidate.score),
            approach_vector=tuple(float(value) for value in approach_world.tolist()),
            jaw_width=None if jaw_width is None else float(jaw_width),
            suction_radius=suction_radius,
            metadata={
                "external_source_algorithm": getattr(candidate, "source_algorithm", self.source_name),
                "external_pose_frame": getattr(candidate, "pose_frame", "camera"),
                "external_metadata": dict(getattr(candidate, "metadata", {}) or {}),
            },
        )


@dataclass
class ICRA2026AnyGraspProvider(ICRA2026ProposalProviderBase):
    """Parallel-jaw proposal provider backed by the local ICRA2026 wrapper."""

    def __init__(self, config: ProposalConfig) -> None:
        super().__init__(
            config=config,
            end_effector_id="gripper_main",
            grasp_type=GraspType.PARALLEL_JAW,
            source_name="AnyGrasp-ICRA2026",
        )


@dataclass
class ICRA2026SuctionNetProvider(ICRA2026ProposalProviderBase):
    """Suction proposal provider backed by the local ICRA2026 wrapper."""

    def __init__(self, config: ProposalConfig) -> None:
        super().__init__(
            config=config,
            end_effector_id="suction_main",
            grasp_type=GraspType.SUCTION,
            source_name="SuctionNet-ICRA2026",
        )


def _rotation_matrix_from_xyzw(quaternion_xyzw: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = quaternion_xyzw
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


def _wxyz_from_rotation_matrix(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return tuple(float(value) for value in quat.tolist())
