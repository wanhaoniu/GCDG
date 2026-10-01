"""End-to-end pipeline runner and export helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from grasp_dependency_dataset.common.config import ExportConfig, load_config_bundle
from grasp_dependency_dataset.common.export_layout import sample_export_paths, scene_export_paths
from grasp_dependency_dataset.common.io import ensure_dir, write_json
from grasp_dependency_dataset.common.types import (
    GraspProposal,
    GraspType,
    PlanningSummary,
    StableScene,
    ValidationResult,
)
from grasp_dependency_dataset.grasping.anygrasp_interface import build_parallel_provider
from grasp_dependency_dataset.grasping.suctionnet_interface import build_suction_provider
from grasp_dependency_dataset.labeling.dependency_labeler import DependencyLabeler
from grasp_dependency_dataset.observation.renderer import (
    MujocoSceneObservationRenderer,
    RenderCameraSpec,
    RenderedTargetObservation,
)
from grasp_dependency_dataset.planning.blocker_solver import MinimalBlockerSolver
from grasp_dependency_dataset.simulation.assets import load_asset_catalog
from grasp_dependency_dataset.simulation.mujoco_backend import MujocoBackend
from grasp_dependency_dataset.simulation.scene_generator import BinClutterSceneGenerator
from grasp_dependency_dataset.validation.validator import StagedGraspValidator

_OBJECT_CONTEXT_ADJACENCY_MARGIN_M = 0.02
_OBJECT_CONTACT_GAP_TOLERANCE_M = 0.005
_OBJECT_SUPPORT_VERTICAL_TOLERANCE_M = 0.008
_OBJECT_SUPPORT_MIN_XY_OVERLAP_RATIO = 0.05


@dataclass
class DatasetPipelineRunner:
    """Orchestrate scene generation, labeling, planning, and export."""

    scene_generator: BinClutterSceneGenerator
    parallel_provider: Any
    suction_provider: Any
    validator: StagedGraspValidator
    dependency_labeler: DependencyLabeler
    blocker_solver: MinimalBlockerSolver
    export_config: ExportConfig
    target_observation_renderer: MujocoSceneObservationRenderer | None = None
    min_target_visible_pixels: int = 1024
    target_edge_margin_px: int = 12
    require_target_proposals: bool = True
    enumerate_all_targets: bool = True
    include_targets_without_proposals: bool = True
    max_scene_generation_attempts: int = 12

    @classmethod
    def from_configs(
        cls,
        config_dir: str | Path = "configs",
        output_root: str | Path | None = None,
        proposal_config_path: str | Path | None = None,
        scene_config_path: str | Path | None = None,
    ) -> "DatasetPipelineRunner":
        """Construct the runner from YAML config files."""

        configs = load_config_bundle(config_dir)
        if scene_config_path is not None:
            from grasp_dependency_dataset.common.config import SceneGenerationConfig

            configs["scene_generation"] = SceneGenerationConfig.from_yaml(scene_config_path)
        if proposal_config_path is not None:
            from grasp_dependency_dataset.common.config import ProposalConfig

            configs["proposal_sources"] = ProposalConfig.from_yaml(proposal_config_path)
        export_config = configs["dataset_export"]
        if output_root is not None:
            export_config = replace(export_config, output_root=str(output_root))

        catalog = load_asset_catalog(
            object_source=configs["scene_generation"].object_source,
            catalog_manifest=configs["scene_generation"].catalog_manifest,
        )
        validator = StagedGraspValidator(configs["grasp_validation"])
        observation_renderer = MujocoSceneObservationRenderer(
            RenderCameraSpec.from_proposal_config(configs["proposal_sources"])
        )
        return cls(
            scene_generator=BinClutterSceneGenerator(
                config=configs["scene_generation"],
                catalog=catalog,
                backend=MujocoBackend(),
            ),
            parallel_provider=build_parallel_provider(configs["proposal_sources"]),
            suction_provider=build_suction_provider(configs["proposal_sources"]),
            validator=validator,
            dependency_labeler=DependencyLabeler(validator),
            blocker_solver=MinimalBlockerSolver(
                validator=validator,
                max_search_depth=export_config.max_blocker_search_depth,
            ),
            export_config=export_config,
            target_observation_renderer=observation_renderer,
            enumerate_all_targets=export_config.enumerate_all_targets,
            include_targets_without_proposals=export_config.include_targets_without_proposals,
        )

    def run(
        self,
        num_scenes: int | None = None,
        max_targets_per_scene: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run the full pipeline for a small number of scenes."""

        output_root = ensure_dir(self.export_config.output_root)
        summaries: list[dict[str, Any]] = []
        scene_count = num_scenes or self.scene_generator.config.num_scenes

        for scene_index in range(scene_count):
            prepared = self.prepare_scene(
                scene_index=scene_index,
                max_targets_per_scene=max_targets_per_scene,
            )
            if prepared is None:
                continue
            scene, proposal_cache = prepared

            if self.export_config.write_scene_snapshots:
                self._write_scene(scene, output_root)

            for target_id in scene.target_ids:
                proposals = proposal_cache.get(target_id)
                if proposals is None:
                    proposals = self.generate_proposals(scene, target_id)
                results = self.validator.validate_all(scene, target_id, proposals)
                dependencies = self.dependency_labeler.label(scene, target_id, proposals, results)
                planning_labels = self.blocker_solver.solve(
                    scene,
                    target_id,
                    proposals,
                    original_results=results,
                )
                planning_summary = self.blocker_solver.summarize(planning_labels)

                sample_paths = sample_export_paths(output_root, scene.scene_id, target_id)
                if self.export_config.write_grasp_proposals:
                    write_json(
                        sample_paths.proposal_path,
                        self._proposal_export_payload(target_id, proposals),
                    )
                if self.export_config.write_labels:
                    write_json(
                        sample_paths.label_path,
                        self._label_export_payload(
                            proposals,
                            results,
                            dependencies,
                            planning_labels,
                            planning_summary,
                        ),
                    )

                manifest_path = sample_paths.manifest_path
                if self.export_config.write_manifests:
                    write_json(
                        manifest_path,
                        self._build_manifest(
                            scene=scene,
                            target_id=target_id,
                            proposals=proposals,
                            results=results,
                            dependencies=dependencies,
                            planning_summary=planning_summary,
                            scene_path=str(sample_paths.scene.scene_path),
                            proposal_path=str(sample_paths.proposal_path),
                            label_path=str(sample_paths.label_path),
                        ),
                    )

                feasible_count = sum(1 for result in results.values() if result.feasible)
                summaries.append(
                    {
                        "scene_id": scene.scene_id,
                        "target_id": target_id,
                        "num_proposals": len(proposals),
                        "num_feasible": feasible_count,
                        "planning_status": planning_summary.status.value,
                        "minimal_blocker_set_size": planning_summary.minimal_blocker_set_size,
                        "manifest_path": str(manifest_path),
                    }
                )

        return summaries

    def rank_targets_by_visibility(self, scene: StableScene) -> list[dict[str, Any]]:
        """Rank all scene objects by visible target pixels in the proposal camera."""

        ranking: list[dict[str, Any]] = []
        for obj in scene.objects:
            observation = self._get_target_observation(scene, obj.object_id)
            width = int(observation.intrinsics["width"])
            height = int(observation.intrinsics["height"])
            visible_pixels = int(observation.metadata["target_mask_pixel_count"])
            x0, y0, x1, y1 = (int(value) for value in observation.bbox_xyxy)
            bbox_area = max(0, x1 - x0) * max(0, y1 - y0)
            image_area = max(width * height, 1)
            ranking.append(
                {
                    "object_id": obj.object_id,
                    "asset_name": obj.asset_name,
                    "visible_pixels": visible_pixels,
                    "visible_ratio": float(visible_pixels / image_area),
                    "bbox_xyxy": (x0, y0, x1, y1),
                    "bbox_area_ratio": float(bbox_area / image_area),
                    "image_width": width,
                    "image_height": height,
                    "touches_border": self._touches_image_border(
                        observation.bbox_xyxy,
                        width,
                        height,
                    ),
                }
            )
        return sorted(
            ranking,
            key=lambda item: (int(item["visible_pixels"]), not bool(item["touches_border"])),
            reverse=True,
        )

    def select_targets(
        self,
        scene: StableScene,
        max_targets_per_scene: int | None = None,
    ) -> tuple[StableScene, dict[str, list[GraspProposal]]]:
        """Select targets and precompute proposal caches.

        In formal dataset mode we enumerate every object as a potential target so
        each `(scene, target)` pair becomes one sample. Demo callers can still
        cap the number of targets per scene via `max_targets_per_scene`.
        """

        desired_target_count = (
            len(scene.objects) if self.enumerate_all_targets else len(scene.target_ids)
        )
        if max_targets_per_scene is not None:
            desired_target_count = min(desired_target_count, max_targets_per_scene)
        desired_target_count = max(1, min(desired_target_count, len(scene.objects)))

        visibility_ranking = self.rank_targets_by_visibility(scene)
        candidate_order = self._ordered_target_candidates(visibility_ranking)

        selected_targets: list[str] = []
        proposal_cache: dict[str, list[GraspProposal]] = {}
        selection_mode = "visibility_filtered"
        if self.enumerate_all_targets and max_targets_per_scene is None:
            selection_mode = "all_scene_objects"
            for candidate in candidate_order:
                target_id = str(candidate["object_id"])
                proposals = self.generate_proposals(scene, target_id)
                proposal_cache[target_id] = proposals
                if self.include_targets_without_proposals or proposals:
                    selected_targets.append(target_id)
        else:
            for candidate in candidate_order:
                target_id = str(candidate["object_id"])
                proposals = self.generate_proposals(scene, target_id)
                if self.require_target_proposals and not proposals:
                    continue
                proposal_cache[target_id] = proposals
                selected_targets.append(target_id)
                if len(selected_targets) >= desired_target_count:
                    break

            if not selected_targets and visibility_ranking:
                fallback_target_id = str(visibility_ranking[0]["object_id"])
                proposal_cache[fallback_target_id] = self.generate_proposals(scene, fallback_target_id)
                selected_targets.append(fallback_target_id)

        metadata = dict(scene.metadata)
        metadata["target_selection"] = {
            "selection_mode": selection_mode,
            "desired_target_count": desired_target_count,
            "min_visible_pixels": self.min_target_visible_pixels,
            "edge_margin_px": self.target_edge_margin_px,
            "require_target_proposals": self.require_target_proposals,
            "include_targets_without_proposals": self.include_targets_without_proposals,
            "visibility_ranking": visibility_ranking,
            "proposal_counts_by_target": {
                target_id: len(proposal_cache.get(target_id, []))
                for target_id in [str(item["object_id"]) for item in candidate_order]
            },
            "selected_target_ids": selected_targets,
        }
        selected_scene = StableScene(
            scene_id=scene.scene_id,
            objects=scene.objects,
            target_ids=tuple(selected_targets),
            bin_size=scene.bin_size,
            wall_thickness=scene.wall_thickness,
            metadata=metadata,
        )
        return selected_scene, proposal_cache

    def prepare_scene(
        self,
        scene_index: int,
        max_targets_per_scene: int | None = None,
    ) -> tuple[StableScene, dict[str, list[GraspProposal]]] | None:
        """Retry scene generation until one stable, usable sample is obtained."""

        for attempt in range(1, self.max_scene_generation_attempts + 1):
            scene = self.scene_generator.generate_scene(scene_index)
            if not bool(scene.metadata.get("stable", False)):
                continue

            selected_scene, proposal_cache = self.select_targets(
                scene,
                max_targets_per_scene=max_targets_per_scene,
            )
            best_proposal_count = max(
                (len(proposals) for proposals in proposal_cache.values()),
                default=0,
            )
            if not selected_scene.target_ids:
                continue
            if self.require_target_proposals and best_proposal_count <= 0:
                continue

            metadata = dict(selected_scene.metadata)
            metadata["scene_generation_attempt"] = attempt
            return StableScene(
                scene_id=selected_scene.scene_id,
                objects=selected_scene.objects,
                target_ids=selected_scene.target_ids,
                bin_size=selected_scene.bin_size,
                wall_thickness=selected_scene.wall_thickness,
                metadata=metadata,
            ), proposal_cache
        return None

    def generate_proposals(self, scene: StableScene, target_id: str) -> list[GraspProposal]:
        """Generate and concatenate proposal sets for both grasp modalities."""

        proposals, _ = self.generate_proposals_with_stats(scene, target_id)
        return proposals

    def generate_proposals_with_stats(
        self,
        scene: StableScene,
        target_id: str,
    ) -> tuple[list[GraspProposal], dict[str, int | bool | float]]:
        """Generate proposals and report how many survive similarity filtering."""

        parallel = self.parallel_provider.generate(
            scene=scene,
            target_id=target_id,
            top_k=self.parallel_provider.config.parallel_top_k,
        )
        suction = self.suction_provider.generate(
            scene=scene,
            target_id=target_id,
            top_k=self.suction_provider.config.suction_top_k,
        )
        keep_per_type = int(self.parallel_provider.config.max_keep_per_type)
        parallel = sorted(parallel, key=lambda proposal: float(proposal.proposal_score), reverse=True)
        suction = sorted(suction, key=lambda proposal: float(proposal.proposal_score), reverse=True)
        stats: dict[str, int | bool | float] = {
            "parallel_raw_count": len(parallel),
            "suction_raw_count": len(suction),
            "total_raw_count": len(parallel) + len(suction),
            "parallel_top_k": int(self.parallel_provider.config.parallel_top_k),
            "suction_top_k": int(self.suction_provider.config.suction_top_k),
        }
        proposal_config = self.parallel_provider.config
        max_approach_angle_from_down_deg = float(
            getattr(proposal_config, "max_approach_angle_from_down_deg", 180.0)
        )
        parallel = self._filter_proposals_by_approach_angle(
            parallel,
            max_angle_from_down_deg=max_approach_angle_from_down_deg,
        )
        suction = self._filter_proposals_by_approach_angle(
            suction,
            max_angle_from_down_deg=max_approach_angle_from_down_deg,
        )
        stats["max_approach_angle_from_down_deg"] = max_approach_angle_from_down_deg
        stats["parallel_after_approach_angle_filter_count"] = len(parallel)
        stats["suction_after_approach_angle_filter_count"] = len(suction)
        stats["total_after_approach_angle_filter_count"] = len(parallel) + len(suction)
        if bool(getattr(proposal_config, "enable_similarity_filter", True)):
            parallel = self._filter_similar_proposals(parallel, GraspType.PARALLEL_JAW)
            suction = self._filter_similar_proposals(suction, GraspType.SUCTION)
        stats["enable_similarity_filter"] = bool(getattr(proposal_config, "enable_similarity_filter", True))
        stats["parallel_after_similarity_filter_count"] = len(parallel)
        stats["suction_after_similarity_filter_count"] = len(suction)
        stats["total_after_similarity_filter_count"] = len(parallel) + len(suction)
        if keep_per_type > 0:
            parallel = parallel[:keep_per_type]
            suction = suction[:keep_per_type]
        stats["max_keep_per_type"] = keep_per_type
        stats["parallel_final_kept_count"] = len(parallel)
        stats["suction_final_kept_count"] = len(suction)
        stats["total_final_kept_count"] = len(parallel) + len(suction)
        return parallel + suction, stats

    def _filter_similar_proposals(
        self,
        proposals: list[GraspProposal],
        grasp_type: GraspType,
    ) -> list[GraspProposal]:
        """Greedily keep diverse, high-score proposals under a small pose-similarity rule."""

        if not proposals:
            return []

        config = self.parallel_provider.config
        distance_threshold = float(config.similarity_position_threshold_m)
        approach_cos_threshold = float(
            np.cos(np.deg2rad(float(config.similarity_approach_angle_deg)))
        )
        inplane_cos_threshold = float(
            np.cos(np.deg2rad(float(config.parallel_jaw_similarity_inplane_angle_deg)))
        )

        kept: list[GraspProposal] = []
        for proposal in proposals:
            if all(
                not self._are_similar_proposals(
                    proposal,
                    existing,
                    grasp_type=grasp_type,
                    distance_threshold=distance_threshold,
                    approach_cos_threshold=approach_cos_threshold,
                    inplane_cos_threshold=inplane_cos_threshold,
                )
                for existing in kept
            ):
                kept.append(proposal)
        return kept

    @staticmethod
    def _filter_proposals_by_approach_angle(
        proposals: list[GraspProposal],
        *,
        max_angle_from_down_deg: float,
    ) -> list[GraspProposal]:
        """Keep only proposals whose approach direction remains close to the downward bin axis."""

        if not proposals:
            return []

        clamped_angle_deg = min(max(float(max_angle_from_down_deg), 0.0), 180.0)
        if clamped_angle_deg >= 180.0:
            return list(proposals)

        down_axis = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        cos_threshold = float(np.cos(np.deg2rad(clamped_angle_deg)))
        kept: list[GraspProposal] = []
        for proposal in proposals:
            approach = DatasetPipelineRunner._normalized(
                np.asarray(proposal.approach_vector, dtype=np.float64)
            )
            if float(np.dot(approach, down_axis)) >= cos_threshold:
                kept.append(proposal)
        return kept

    @staticmethod
    def _are_similar_proposals(
        left: GraspProposal,
        right: GraspProposal,
        *,
        grasp_type: GraspType,
        distance_threshold: float,
        approach_cos_threshold: float,
        inplane_cos_threshold: float,
    ) -> bool:
        """Return whether two proposals should be treated as the same grasp."""

        left_position = np.asarray(left.pose.position, dtype=np.float64)
        right_position = np.asarray(right.pose.position, dtype=np.float64)
        if float(np.linalg.norm(left_position - right_position)) > distance_threshold:
            return False

        left_approach = DatasetPipelineRunner._normalized(np.asarray(left.approach_vector, dtype=np.float64))
        right_approach = DatasetPipelineRunner._normalized(np.asarray(right.approach_vector, dtype=np.float64))
        if float(np.dot(left_approach, right_approach)) < approach_cos_threshold:
            return False

        if grasp_type != GraspType.PARALLEL_JAW:
            return True

        left_open = DatasetPipelineRunner._parallel_jaw_opening_axis(left)
        right_open = DatasetPipelineRunner._parallel_jaw_opening_axis(right)
        return float(abs(np.dot(left_open, right_open))) >= inplane_cos_threshold

    @staticmethod
    def _parallel_jaw_opening_axis(proposal: GraspProposal) -> np.ndarray:
        """Extract the parallel-jaw opening axis from the proposal orientation."""

        rotation = DatasetPipelineRunner._rotation_matrix_from_wxyz(proposal.pose.quaternion_wxyz)
        opening = rotation[:, 0]
        return DatasetPipelineRunner._normalized(opening)

    @staticmethod
    def _rotation_matrix_from_wxyz(quaternion_wxyz: tuple[float, float, float, float]) -> np.ndarray:
        """Convert a quaternion into a rotation matrix."""

        w, x, y, z = (float(value) for value in quaternion_wxyz)
        norm = np.linalg.norm([w, x, y, z])
        if norm < 1e-12:
            return np.eye(3, dtype=np.float64)
        w, x, y, z = (value / norm for value in (w, x, y, z))
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

    @staticmethod
    def _normalized(vector: np.ndarray) -> np.ndarray:
        """Normalize a vector safely."""

        norm = float(np.linalg.norm(vector))
        if norm < 1e-12:
            return vector
        return vector / norm

    def _ordered_target_candidates(self, visibility_ranking: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prefer visible, unclipped targets before falling back to the rest."""

        preferred = [
            item
            for item in visibility_ranking
            if int(item["visible_pixels"]) >= self.min_target_visible_pixels
            and not bool(item["touches_border"])
        ]
        visible = [
            item
            for item in visibility_ranking
            if int(item["visible_pixels"]) >= self.min_target_visible_pixels
            and bool(item["touches_border"])
        ]
        weakly_visible = [
            item
            for item in visibility_ranking
            if 0 < int(item["visible_pixels"]) < self.min_target_visible_pixels
        ]
        remaining = [item for item in visibility_ranking if int(item["visible_pixels"]) <= 0]

        ordered: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in preferred + visible + weakly_visible + remaining:
            object_id = str(item["object_id"])
            if object_id in seen_ids:
                continue
            ordered.append(item)
            seen_ids.add(object_id)
        return ordered

    def _get_target_observation(self, scene: StableScene, target_id: str) -> RenderedTargetObservation:
        """Fetch a rendered target observation, reusing provider caches when available."""

        if hasattr(self.parallel_provider, "_get_observation"):
            return self.parallel_provider._get_observation(scene, target_id)
        if self.target_observation_renderer is None:
            raise RuntimeError("Target observation renderer is not available.")
        return self.target_observation_renderer.render_target(scene, target_id)

    def _touches_image_border(
        self,
        bbox_xyxy: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> bool:
        """Return whether a target bbox lies too close to the camera border."""

        x0, y0, x1, y1 = bbox_xyxy
        margin_px = self.target_edge_margin_px
        return (
            x0 <= margin_px
            or y0 <= margin_px
            or x1 >= width - margin_px
            or y1 >= height - margin_px
        )

    @staticmethod
    def _write_scene(scene: StableScene, output_root: Path) -> None:
        paths = scene_export_paths(output_root, scene.scene_id)
        write_json(paths.scene_path, scene.to_dict())

    @staticmethod
    def _visibility_lookup(scene: StableScene) -> dict[str, dict[str, Any]]:
        """Index per-object visibility stats recorded during target selection."""

        ranking = scene.metadata.get("target_selection", {}).get("visibility_ranking", [])
        return {
            str(item["object_id"]): dict(item)
            for item in ranking
        }

    def _object_node_payload(
        self,
        scene_object,
        *,
        target=None,
        visibility_info: dict[str, Any] | None = None,
        is_target: bool,
    ) -> dict[str, Any]:
        """Build a target-centric object payload for downstream graph construction."""

        payload = scene_object.to_dict()
        payload["id"] = str(scene_object.object_id)
        payload["is_target"] = bool(is_target)
        payload["category_id"] = None
        payload["mesh_path"] = scene_object.spec.mesh_path
        payload["collision_mesh_paths"] = list(scene_object.spec.collision_mesh_paths or ())
        payload["shape_feat"] = None
        payload["point_cloud_feat"] = None
        payload["mask_id"] = None

        if visibility_info is not None:
            payload["observation"] = {
                "visible_pixels": int(visibility_info.get("visible_pixels", 0)),
                "visible_ratio": float(visibility_info.get("visible_ratio", 0.0)),
                "bbox_xyxy": list(visibility_info.get("bbox_xyxy", (0, 0, 0, 0))),
                "bbox_area_ratio": float(visibility_info.get("bbox_area_ratio", 0.0)),
                "touches_border": bool(visibility_info.get("touches_border", False)),
                "image_width": int(visibility_info.get("image_width", 0)),
                "image_height": int(visibility_info.get("image_height", 0)),
            }
            payload["visible_ratio"] = float(visibility_info.get("visible_ratio", 0.0))

        if target is not None and scene_object.object_id != target.object_id:
            relative = np.asarray(scene_object.pose.position, dtype=np.float64) - np.asarray(
                target.pose.position,
                dtype=np.float64,
            )
            payload["relative_to_target"] = {
                "vector": [float(value) for value in relative.tolist()],
                "distance": float(np.linalg.norm(relative)),
                "xy_distance": float(np.linalg.norm(relative[:2])),
                "z_offset": float(relative[2]),
            }
            payload["distance_to_target"] = float(np.linalg.norm(relative))
            payload["bbox_overlap_target"] = float(self._xy_overlap_ratio(scene_object, target))
            payload["contact_with_target"] = bool(self._objects_in_contact(scene_object, target))
            payload["support_relation_to_target"] = str(
                self._support_relation(scene_object, target)
            )

        return payload

    def _object_context_relations(
        self,
        scene: StableScene,
        target_id: str,
    ) -> list[dict[str, Any]]:
        """Approximate object-object context relations among non-target clutter objects."""

        objects = [obj for obj in scene.objects if obj.object_id != target_id]
        relations: list[dict[str, Any]] = []
        for index, left in enumerate(objects):
            for right in objects[index + 1 :]:
                left_position = np.asarray(left.pose.position, dtype=np.float64)
                right_position = np.asarray(right.pose.position, dtype=np.float64)
                relative = right_position - left_position
                distance = float(np.linalg.norm(relative))
                surface_gap = float(
                    distance - (left.scaled_bounding_radius + right.scaled_bounding_radius)
                )
                overlap_score = float(self._xy_overlap_ratio(left, right))
                relations.append(
                    {
                        "src_id": str(left.object_id),
                        "dst_id": str(right.object_id),
                        "edge_type": "object_object_context",
                        "distance": distance,
                        "relative_position": [float(value) for value in relative.tolist()],
                        "adjacent": bool(surface_gap <= _OBJECT_CONTEXT_ADJACENCY_MARGIN_M),
                        "in_contact": bool(self._objects_in_contact(left, right)),
                        "support_relation": str(self._support_relation(left, right)),
                        "overlap_score": overlap_score,
                        "relation_weight": float(1.0 / (1.0 + max(distance, 0.0))),
                        "surface_gap_estimate": surface_gap,
                    }
                )
        return relations

    @staticmethod
    def _axis_extent(scene_object, axis_world: tuple[float, float, float]) -> float:
        """Project the scaled proxy half extents onto a world-space axis."""

        axis = np.asarray(axis_world, dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-12:
            return 0.0
        axis = axis / axis_norm
        rotation = DatasetPipelineRunner._rotation_matrix_from_wxyz(scene_object.pose.quaternion_wxyz)
        axis_body = axis.dot(rotation)
        half_extents = np.asarray(scene_object.scaled_proxy_half_extents, dtype=np.float64)
        return float(np.sum(np.abs(axis_body) * half_extents))

    def _xy_overlap_ratio(self, left, right) -> float:
        """Approximate XY footprint overlap ratio between two proxy objects."""

        left_x = self._axis_extent(left, (1.0, 0.0, 0.0))
        left_y = self._axis_extent(left, (0.0, 1.0, 0.0))
        right_x = self._axis_extent(right, (1.0, 0.0, 0.0))
        right_y = self._axis_extent(right, (0.0, 1.0, 0.0))

        left_x_min = float(left.pose.position[0]) - left_x
        left_x_max = float(left.pose.position[0]) + left_x
        right_x_min = float(right.pose.position[0]) - right_x
        right_x_max = float(right.pose.position[0]) + right_x
        left_y_min = float(left.pose.position[1]) - left_y
        left_y_max = float(left.pose.position[1]) + left_y
        right_y_min = float(right.pose.position[1]) - right_y
        right_y_max = float(right.pose.position[1]) + right_y

        overlap_x = max(0.0, min(left_x_max, right_x_max) - max(left_x_min, right_x_min))
        overlap_y = max(0.0, min(left_y_max, right_y_max) - max(left_y_min, right_y_min))
        overlap_area = overlap_x * overlap_y
        left_area = max(2.0 * left_x, 0.0) * max(2.0 * left_y, 0.0)
        right_area = max(2.0 * right_x, 0.0) * max(2.0 * right_y, 0.0)
        denominator = min(left_area, right_area)
        if denominator <= 1e-12:
            return 0.0
        return float(overlap_area / denominator)

    def _objects_in_contact(self, left, right) -> bool:
        """Approximate whether two objects are touching based on proxy radii."""

        distance = float(
            np.linalg.norm(
                np.asarray(left.pose.position, dtype=np.float64)
                - np.asarray(right.pose.position, dtype=np.float64)
            )
        )
        surface_gap = distance - (left.scaled_bounding_radius + right.scaled_bounding_radius)
        return bool(surface_gap <= _OBJECT_CONTACT_GAP_TOLERANCE_M)

    def _support_relation(self, left, right) -> str:
        """Approximate a vertical support relation using XY overlap and Z extents."""

        overlap_ratio = self._xy_overlap_ratio(left, right)
        if overlap_ratio < _OBJECT_SUPPORT_MIN_XY_OVERLAP_RATIO:
            return "none"

        left_extent_z = self._axis_extent(left, (0.0, 0.0, 1.0))
        right_extent_z = self._axis_extent(right, (0.0, 0.0, 1.0))
        left_top = float(left.pose.position[2]) + left_extent_z
        left_bottom = float(left.pose.position[2]) - left_extent_z
        right_top = float(right.pose.position[2]) + right_extent_z
        right_bottom = float(right.pose.position[2]) - right_extent_z

        if abs(right_bottom - left_top) <= _OBJECT_SUPPORT_VERTICAL_TOLERANCE_M:
            return "src_supports_dst"
        if abs(left_bottom - right_top) <= _OBJECT_SUPPORT_VERTICAL_TOLERANCE_M:
            return "dst_supports_src"
        return "none"

    def _graph_support_payload(
        self,
        scene: StableScene,
        target_id: str,
        dependencies,
        proposal_path: str,
        label_path: str,
    ) -> dict[str, Any]:
        """Collect target-centric raw data that can be converted into a graph later."""

        visibility_lookup = self._visibility_lookup(scene)
        target = scene.get_object(target_id)
        dependency_counts = Counter()
        for label in dependencies:
            dependency_counts["dep_any"] += int(bool(label.dep_any))
            dependency_counts["dep_collision_approach"] += int(bool(label.dep_collision_approach))
            dependency_counts["dep_collision_lift"] += int(bool(label.dep_collision_lift))

        return {
            "target_node": self._object_node_payload(
                target,
                target=target,
                visibility_info=visibility_lookup.get(target_id),
                is_target=True,
            ),
            "object_nodes": [
                self._object_node_payload(
                    scene_object,
                    target=target,
                    visibility_info=visibility_lookup.get(scene_object.object_id),
                    is_target=False,
                )
                for scene_object in scene.objects
                if scene_object.object_id != target_id
            ],
            "object_context_relations": self._object_context_relations(scene, target_id),
            "grasp_label_source": {
                "proposal_path": proposal_path,
                "label_path": label_path,
                "grasp_node_fields": [
                    "grasp_id",
                    "target_id",
                    "grasp_type",
                    "grasp_pose",
                    "position",
                    "orientation",
                    "approach_dir",
                    "lift_dir",
                    "score_init",
                    "proposal_source",
                    "jaw_width",
                    "suction_radius",
                    "closing_dir",
                    "suction_normal",
                    "contact_center",
                ],
                "dependency_edge_fields": [
                    "object_id",
                    "grasp_id",
                    "dep_any",
                    "dep_collision_approach",
                    "dep_collision_lift",
                    "metadata.relative_object_grasp",
                    "metadata.restored_if_removed",
                ],
                "dependency_label_counts": dict(sorted(dependency_counts.items())),
            },
        }

    def _proposal_export_payload(self, target_id: str, proposals: list[GraspProposal]) -> dict[str, Any]:
        parallel = []
        suction = []
        for proposal in proposals:
            proposal_payload = proposal.to_dict()
            metadata = dict(proposal.metadata or {})
            augmentation_fields = {
                key: metadata[key]
                for key in (
                    "is_augmented_proposal",
                    "completion_mode",
                    "target_visibility_status",
                    "base_proposal_source",
                    "augmentation_source_mode",
                )
                if key in metadata
            }
            if proposal.grasp_type == GraspType.PARALLEL_JAW:
                parallel.append(
                    {
                        "grasp_id": proposal.grasp_id,
                        "target_id": proposal.target_id,
                        "grasp_type": proposal.grasp_type.value,
                        "pose": proposal.pose.to_dict(),
                        "grasp_pose": proposal.pose.to_dict(),
                        "position": list(proposal.pose.position),
                        "orientation": list(proposal.pose.quaternion_wxyz),
                        "approach_dir": list(proposal.approach_vector),
                        "lift_dir": list(proposal.lift_vector),
                        "jaw_width": proposal.jaw_width,
                        "closing_dir": proposal_payload.get("closing_dir", []),
                        "source": proposal.source,
                        "proposal_source": proposal.source,
                        "proposal_score": proposal.proposal_score,
                        "score_init": proposal.proposal_score,
                        "metadata": metadata,
                        **augmentation_fields,
                    }
                )
            else:
                suction.append(
                    {
                        "grasp_id": proposal.grasp_id,
                        "target_id": proposal.target_id,
                        "grasp_type": proposal.grasp_type.value,
                        "pose": proposal.pose.to_dict(),
                        "grasp_pose": proposal.pose.to_dict(),
                        "position": list(proposal.pose.position),
                        "orientation": list(proposal.pose.quaternion_wxyz),
                        "approach_dir": list(proposal.approach_vector),
                        "lift_dir": list(proposal.lift_vector),
                        "normal": proposal_payload.get("suction_normal", []),
                        "suction_normal": proposal_payload.get("suction_normal", []),
                        "contact_center": proposal_payload.get("contact_center", []),
                        "suction_radius": proposal.suction_radius,
                        "source": proposal.source,
                        "proposal_source": proposal.source,
                        "proposal_score": proposal.proposal_score,
                        "score_init": proposal.proposal_score,
                        "metadata": metadata,
                        **augmentation_fields,
                    }
                )
        return {
            "target_id": target_id,
            "parallel_grasps": parallel,
            "suction_grasps": suction,
            "raw_proposals": [proposal.to_dict() for proposal in proposals],
        }

    def _label_export_payload(
        self,
        proposals,
        results,
        dependencies,
        planning_labels,
        planning_summary: PlanningSummary,
    ) -> dict[str, Any]:
        """Build the canonical label payload for one target sample."""

        grasp_feasibility = []
        for proposal in proposals:
            result_payload = results[proposal.grasp_id].to_dict()
            result_payload["target_id"] = proposal.target_id
            result_payload["grasp_type"] = proposal.grasp_type.value
            result_payload["proposal_score"] = float(proposal.proposal_score)
            result_payload["position"] = [float(value) for value in proposal.pose.position]
            result_payload["orientation"] = [float(value) for value in proposal.pose.quaternion_wxyz]
            result_payload["approach_dir"] = [float(value) for value in proposal.approach_vector]
            result_payload["lift_dir"] = [float(value) for value in proposal.lift_vector]
            grasp_feasibility.append(result_payload)

        return {
            "grasp_feasibility": grasp_feasibility,
            "dependencies": [label.to_dict() for label in dependencies],
            "planning_labels": [label.to_dict() for label in planning_labels],
            "planning_summary": planning_summary.to_dict(),
        }

    @staticmethod
    def _sample_statistics(
        proposals: list[GraspProposal],
        results: dict[str, ValidationResult],
        planning_summary: PlanningSummary,
    ) -> dict[str, Any]:
        """Summarize one target sample without duplicating the full artifacts."""

        stage_counts = Counter(result.failure_stage.value for result in results.values())
        return {
            "num_proposals": len(proposals),
            "num_feasible": sum(1 for result in results.values() if result.feasible),
            "stage_counts": dict(sorted(stage_counts.items())),
            "planning_status": planning_summary.status.value,
            "planning_terminal_reason": planning_summary.terminal_reason.value,
            "best_planning_grasp_id": planning_summary.best_grasp_id,
            "minimal_blocker_set_size": planning_summary.minimal_blocker_set_size,
            "minimal_blocker_set": list(planning_summary.minimal_blocker_set),
            "oracle_removal_sequence": list(planning_summary.oracle_removal_sequence),
            "planning_status_counts": dict(planning_summary.status_counts),
            "planning_terminal_reason_counts": dict(planning_summary.terminal_reason_counts),
        }

    def _build_manifest(
        self,
        scene: StableScene,
        target_id: str,
        proposals: list[GraspProposal],
        results: dict[str, ValidationResult],
        dependencies,
        planning_summary: PlanningSummary,
        scene_path: str = "",
        proposal_path: str = "",
        label_path: str = "",
        shared_observations: dict[str, str] | None = None,
        target_visualizations: dict[str, str] | None = None,
        target_summary_path: str = "",
    ) -> dict[str, Any]:
        target = scene.get_object(target_id)
        return {
            "dataset_name": "CEPB-Target-Retrieval-Grasp-Dependency",
            "sample_unit": "scene_plus_target",
            "sample_id": f"{scene.scene_id}__{target_id}",
            "scene": {
                "scene_id": scene.scene_id,
                "target_id": target_id,
                "scene_path": scene_path,
                "num_objects": len(scene.objects),
                "shared_observations": shared_observations or {},
            },
            "target": {
                "target_id": target_id,
                "target_asset_name": target.asset_name,
            },
            "artifacts": {
                "proposals_path": proposal_path,
                "labels_path": label_path,
                "summary_path": target_summary_path,
                "visualizations": target_visualizations or {},
            },
            "statistics": self._sample_statistics(proposals, results, planning_summary),
            "graph_support": self._graph_support_payload(
                scene=scene,
                target_id=target_id,
                dependencies=dependencies,
                proposal_path=proposal_path,
                label_path=label_path,
            ),
        }
