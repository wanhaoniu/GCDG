"""Generate, audit, and visualize physically settled dense-layer MuJoCo scenes.

This script deliberately reuses the CoRL project's wave-drop scene generator and
collision meshes.  Scene acceptance is based on the final MuJoCo contact graph,
not on object count or a hand-authored visibility state.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from grasp_dependency_dataset.common.config import SceneGenerationConfig
from grasp_dependency_dataset.common.types import Pose, SceneObjectState, StableScene
from grasp_dependency_dataset.observation.renderer import (
    MujocoSceneObservationRenderer,
    RenderCameraSpec,
)
from grasp_dependency_dataset.simulation.assets import load_asset_catalog
from grasp_dependency_dataset.simulation.mujoco_backend import MujocoBackend
from grasp_dependency_dataset.simulation.scene_builder import build_bin_scene_xml
from grasp_dependency_dataset.simulation.scene_generator import BinClutterSceneGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulation/dense.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/generated_scenes"

AUDIT_SETTLE_STEPS = 600
CONTACT_DISTANCE_TOLERANCE_M = 0.0015
MIN_SUPPORT_CENTER_DELTA_M = 0.004

MIN_UPPER_LAYER_FRACTION = 0.25
MIN_UPPER_LAYER_OBJECTS = 5
MIN_MAX_SUPPORT_DEPTH = 2
MIN_OBJECT_CONTACT_PAIRS = 8
MIN_PILE_HEIGHT_M = 0.095


class DenseLayerSceneGenerator(BinClutterSceneGenerator):
    """Use a longer per-wave settling window for contact-rich tall piles."""

    def _intermediate_settle_steps(self, num_objects: int) -> int:
        del num_objects
        return 1200

VIEW_SPECS = {
    "top_full_wall": RenderCameraSpec(
        width=960,
        height=720,
        fovy_deg=38.0,
        lookat=(0.0, 0.0, 0.055),
        distance=0.48,
        azimuth_deg=-90.0,
        elevation_deg=86.0,
        wall_height=None,
    ),
    "oblique_left_inspection": RenderCameraSpec(
        width=960,
        height=720,
        fovy_deg=39.0,
        lookat=(0.0, 0.0, 0.060),
        distance=0.48,
        azimuth_deg=-48.0,
        elevation_deg=36.0,
        wall_height=0.160,
        bin_wall_rgba=(0.72, 0.72, 0.72, 0.42),
    ),
    "oblique_right_inspection": RenderCameraSpec(
        width=960,
        height=720,
        fovy_deg=39.0,
        lookat=(0.0, 0.0, 0.060),
        distance=0.48,
        azimuth_deg=132.0,
        elevation_deg=34.0,
        wall_height=0.160,
        bin_wall_rgba=(0.72, 0.72, 0.72, 0.42),
    ),
    "side_inspection": RenderCameraSpec(
        width=960,
        height=720,
        fovy_deg=37.0,
        lookat=(0.0, 0.0, 0.065),
        distance=0.50,
        azimuth_deg=2.0,
        elevation_deg=18.0,
        wall_height=0.160,
        bin_wall_rgba=(0.72, 0.72, 0.72, 0.42),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--accepted", type=int, default=6)
    parser.add_argument("--max-attempts", type=int, default=48)
    return parser.parse_args()


def rotation_matrix_wxyz(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    norm = float(np.linalg.norm((w, x, y, z)))
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def world_vertical_half_extent(obj: SceneObjectState) -> float:
    half_extents = np.asarray(obj.spec.proxy_half_extents, dtype=np.float64)
    rotation = rotation_matrix_wxyz(obj.pose.quaternion_wxyz)
    return float(np.sum(np.abs(rotation[2, :]) * half_extents))


def object_name_for_geom(model: mujoco.MjModel, geom_id: int) -> str | None:
    body_id = int(model.geom_bodyid[geom_id])
    if body_id == 0:
        return None
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)


def final_scene_from_data(
    scene: StableScene,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> StableScene:
    objects: list[SceneObjectState] = []
    for obj in scene.objects:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj.object_id)
        objects.append(
            replace(
                obj,
                pose=Pose(
                    position=tuple(float(value) for value in data.xpos[body_id]),
                    quaternion_wxyz=tuple(float(value) for value in data.xquat[body_id]),
                ),
            )
        )
    return replace(scene, objects=tuple(objects))


def support_depths(
    object_ids: set[str],
    floor_contacts: set[str],
    support_edges: set[tuple[str, str]],
) -> dict[str, int | None]:
    depths: dict[str, int | None] = {
        object_id: (0 if object_id in floor_contacts else None)
        for object_id in object_ids
    }
    for _ in range(len(object_ids)):
        changed = False
        for lower, upper in support_edges:
            lower_depth = depths.get(lower)
            if lower_depth is None:
                continue
            candidate = int(lower_depth) + 1
            if depths.get(upper) is None or candidate > int(depths[upper]):
                depths[upper] = candidate
                changed = True
        if not changed:
            break
    return depths


def audit_scene(
    scene: StableScene,
    config: SceneGenerationConfig,
) -> tuple[StableScene, dict[str, Any]]:
    xml = build_bin_scene_xml(
        scene_objects=[(obj.object_id, obj.spec, obj.pose) for obj in scene.objects],
        bin_config=config.bin,
        simulation_config=config.simulation,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(AUDIT_SETTLE_STEPS):
        mujoco.mj_step(model, data)
    scene = final_scene_from_data(scene, model, data)

    centers = {obj.object_id: float(obj.pose.position[2]) for obj in scene.objects}
    floor_contacts: set[str] = set()
    wall_contacts: set[str] = set()
    object_contact_pairs: set[tuple[str, str]] = set()
    support_edges: set[tuple[str, str]] = set()
    contact_records: list[dict[str, Any]] = []

    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if float(contact.dist) > CONTACT_DISTANCE_TOLERANCE_M:
            continue
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ""
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ""
        obj1 = object_name_for_geom(model, geom1)
        obj2 = object_name_for_geom(model, geom2)
        pair_record = {
            "geom1": name1,
            "geom2": name2,
            "object1": obj1,
            "object2": obj2,
            "distance_m": float(contact.dist),
            "position_m": [float(value) for value in contact.pos],
        }
        contact_records.append(pair_record)

        if obj1 is not None and obj2 is None:
            static_name = name2
            dynamic_name = obj1
        elif obj2 is not None and obj1 is None:
            static_name = name1
            dynamic_name = obj2
        else:
            static_name = ""
            dynamic_name = ""

        if dynamic_name and static_name == "floor":
            floor_contacts.add(dynamic_name)
        elif dynamic_name and static_name.startswith("wall_"):
            wall_contacts.add(dynamic_name)

        if obj1 is None or obj2 is None or obj1 == obj2:
            continue
        pair = tuple(sorted((obj1, obj2)))
        object_contact_pairs.add(pair)
        z1, z2 = centers[obj1], centers[obj2]
        if abs(z1 - z2) < MIN_SUPPORT_CENTER_DELTA_M:
            continue
        lower, upper = (obj1, obj2) if z1 < z2 else (obj2, obj1)
        contact_z = float(contact.pos[2])
        if contact_z <= centers[upper] + 0.003:
            support_edges.add((lower, upper))

    object_ids = {obj.object_id for obj in scene.objects}
    depths = support_depths(object_ids, floor_contacts, support_edges)
    upper_layer_ids = sorted(
        object_id for object_id, depth in depths.items()
        if depth is not None and int(depth) >= 1
    )
    unresolved_ids = sorted(object_id for object_id, depth in depths.items() if depth is None)
    layer_counts: dict[str, int] = {}
    for depth in depths.values():
        key = "unresolved" if depth is None else str(int(depth))
        layer_counts[key] = layer_counts.get(key, 0) + 1

    bottoms = []
    tops = []
    for obj in scene.objects:
        half_z = world_vertical_half_extent(obj)
        center_z = float(obj.pose.position[2])
        bottoms.append(center_z - half_z)
        tops.append(center_z + half_z)
    pile_height_m = float(max(tops) - min(0.0, min(bottoms))) if tops else 0.0
    max_support_depth = max(
        (int(depth) for depth in depths.values() if depth is not None),
        default=0,
    )
    upper_fraction = float(len(upper_layer_ids) / max(len(scene.objects), 1))

    recent_max_velocity = 0.0
    for obj in scene.objects:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj.object_id}_joint")
        dof_address = int(model.jnt_dofadr[joint_id])
        linear_velocity = data.qvel[dof_address : dof_address + 3]
        recent_max_velocity = max(recent_max_velocity, float(np.linalg.norm(linear_velocity)))

    half_x = float(config.bin.size[0]) * 0.5
    half_y = float(config.bin.size[1]) * 0.5
    audit_out_of_bin_ids = sorted(
        obj.object_id
        for obj in scene.objects
        if (
            abs(float(obj.pose.position[0])) > half_x + 1e-4
            or abs(float(obj.pose.position[1])) > half_y + 1e-4
            or float(obj.pose.position[2]) < -0.002
        )
    )

    accepted_checks = {
        "complete_object_count": len(scene.objects) >= config.num_objects_min,
        "generator_stable": bool(scene.metadata.get("stable", False)),
        "audit_stable": recent_max_velocity <= config.simulation.stability_velocity_threshold,
        "generator_in_bin": bool(scene.metadata.get("in_bin", False)),
        "audit_in_bin": not audit_out_of_bin_ids,
        "upper_layer_fraction": upper_fraction >= MIN_UPPER_LAYER_FRACTION,
        "upper_layer_objects": len(upper_layer_ids) >= MIN_UPPER_LAYER_OBJECTS,
        "max_support_depth": max_support_depth >= MIN_MAX_SUPPORT_DEPTH,
        "object_contact_pairs": len(object_contact_pairs) >= MIN_OBJECT_CONTACT_PAIRS,
        "pile_height": pile_height_m >= MIN_PILE_HEIGHT_M,
    }
    accepted = all(accepted_checks.values())

    metrics: dict[str, Any] = {
        "accepted": accepted,
        "acceptance_checks": accepted_checks,
        "acceptance_thresholds": {
            "min_upper_layer_fraction": MIN_UPPER_LAYER_FRACTION,
            "min_upper_layer_objects": MIN_UPPER_LAYER_OBJECTS,
            "min_max_support_depth": MIN_MAX_SUPPORT_DEPTH,
            "min_object_contact_pairs": MIN_OBJECT_CONTACT_PAIRS,
            "min_pile_height_m": MIN_PILE_HEIGHT_M,
        },
        "num_objects": len(scene.objects),
        "floor_contact_object_ids": sorted(floor_contacts),
        "wall_contact_object_ids": sorted(wall_contacts),
        "object_contact_pairs": [list(pair) for pair in sorted(object_contact_pairs)],
        "support_edges": [list(edge) for edge in sorted(support_edges)],
        "support_depth_by_object": depths,
        "layer_counts": layer_counts,
        "upper_layer_object_ids": upper_layer_ids,
        "unresolved_object_ids": unresolved_ids,
        "upper_layer_fraction": upper_fraction,
        "max_support_depth": max_support_depth,
        "pile_height_m": pile_height_m,
        "audit_recent_max_linear_velocity_mps": recent_max_velocity,
        "audit_out_of_bin_object_ids": audit_out_of_bin_ids,
        "num_contacts": len(contact_records),
        "contacts": contact_records,
    }
    scene = replace(
        scene,
        metadata={
            **scene.metadata,
            "dense_layer_audit": {
                key: value for key, value in metrics.items() if key != "contacts"
            },
        },
    )
    return scene, metrics


def render_views(scene: StableScene, scene_dir: Path, metrics: dict[str, Any]) -> Path:
    image_paths: dict[str, Path] = {}
    target_id = scene.objects[0].object_id
    for view_name, camera_spec in VIEW_SPECS.items():
        renderer = MujocoSceneObservationRenderer(camera_spec)
        observation = renderer.render_target(scene, target_id)
        output_path = scene_dir / f"{view_name}.png"
        Image.fromarray(observation.color).save(output_path)
        image_paths[view_name] = output_path

    return make_contact_sheet(scene, image_paths, metrics, scene_dir / "contact_sheet.png")


def make_contact_sheet(
    scene: StableScene,
    image_paths: dict[str, Path],
    metrics: dict[str, Any],
    output_path: Path,
) -> Path:
    tile_width, tile_height = 720, 540
    banner_height, footer_height = 76, 136
    canvas = Image.new("RGB", (tile_width * 2, banner_height + tile_height * 2 + footer_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    title = (
        f"{scene.scene_id} | objects={len(scene.objects)} | upper={len(metrics['upper_layer_object_ids'])} "
        f"({metrics['upper_layer_fraction']:.0%}) | max support depth={metrics['max_support_depth']}"
    )
    subtitle = (
        f"object contacts={len(metrics['object_contact_pairs'])} | pile height={metrics['pile_height_m'] * 1000:.1f} mm "
        f"| accepted={metrics['accepted']}"
    )
    draw.text((20, 16), title, fill="black", font=font)
    draw.text((20, 42), subtitle, fill="black", font=font)

    ordered_views = [
        ("top_full_wall", "Top view — full physical wall height"),
        ("oblique_left_inspection", "Oblique left - raised translucent inspection wall"),
        ("oblique_right_inspection", "Oblique right - raised translucent inspection wall"),
        ("side_inspection", "Side view - raised translucent inspection wall"),
    ]
    for index, (view_name, label) in enumerate(ordered_views):
        image = Image.open(image_paths[view_name]).convert("RGB")
        image.thumbnail((tile_width, tile_height - 26), Image.Resampling.LANCZOS)
        x0 = (index % 2) * tile_width
        y0 = banner_height + (index // 2) * tile_height
        x = x0 + (tile_width - image.width) // 2
        y = y0 + 26 + (tile_height - 26 - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text((x0 + 12, y0 + 7), label, fill="black", font=font)

    footer_y = banner_height + tile_height * 2 + 12
    layer_text = ", ".join(
        f"L{key}={value}" if key != "unresolved" else f"unresolved={value}"
        for key, value in sorted(metrics["layer_counts"].items())
    )
    checks_text = ", ".join(
        f"{name}={'PASS' if passed else 'FAIL'}"
        for name, passed in metrics["acceptance_checks"].items()
    )
    footer_lines = [
        f"Contact-derived layer counts: {layer_text}",
        f"Checks: {checks_text}",
        "Physics uses 0.20 m walls; inspection renders use 0.16 m translucent walls.",
    ]
    for line_index, line in enumerate(footer_lines):
        draw.text((20, footer_y + 30 * line_index), line, fill="black", font=font)
    canvas.save(output_path)
    return output_path


def make_gallery(contact_sheets: list[Path], output_path: Path) -> Path:
    previews: list[Image.Image] = []
    for path in contact_sheets:
        image = Image.open(path).convert("RGB")
        image.thumbnail((960, 860), Image.Resampling.LANCZOS)
        previews.append(image.copy())
    columns = 2
    rows = (len(previews) + columns - 1) // columns
    tile_width = max((image.width for image in previews), default=960)
    tile_height = max((image.height for image in previews), default=860)
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), (235, 235, 235))
    for index, image in enumerate(previews):
        x = (index % columns) * tile_width + (tile_width - image.width) // 2
        y = (index // columns) * tile_height + (tile_height - image.height) // 2
        canvas.paste(image, (x, y))
    canvas.save(output_path)
    return output_path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = SceneGenerationConfig.from_yaml(args.config)
    catalog = load_asset_catalog(config.object_source, config.catalog_manifest)
    generator = DenseLayerSceneGenerator(config=config, catalog=catalog, backend=MujocoBackend())

    accepted_summaries: list[dict[str, Any]] = []
    attempt_summaries: list[dict[str, Any]] = []
    contact_sheets: list[Path] = []
    for attempt_index in range(args.max_attempts):
        print(f"[dense-layers] generating attempt {attempt_index + 1}/{args.max_attempts}", flush=True)
        scene = generator.generate_scene(attempt_index)
        scene, metrics = audit_scene(scene, config)
        summary = {
            "attempt_index": attempt_index,
            "scene_id": scene.scene_id,
            "accepted": bool(metrics["accepted"]),
            "num_objects": int(metrics["num_objects"]),
            "upper_layer_objects": len(metrics["upper_layer_object_ids"]),
            "upper_layer_fraction": float(metrics["upper_layer_fraction"]),
            "max_support_depth": int(metrics["max_support_depth"]),
            "object_contact_pairs": len(metrics["object_contact_pairs"]),
            "pile_height_m": float(metrics["pile_height_m"]),
            "checks": metrics["acceptance_checks"],
        }
        attempt_summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if not metrics["accepted"]:
            continue

        accepted_index = len(accepted_summaries)
        scene_dir = args.output / f"accepted_{accepted_index:02d}_{scene.scene_id}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        write_json(scene_dir / "scene.json", scene.to_dict())
        write_json(scene_dir / "metrics.json", metrics)
        contact_sheet = render_views(scene, scene_dir, metrics)
        summary["scene_dir"] = str(scene_dir)
        summary["contact_sheet"] = str(contact_sheet)
        accepted_summaries.append(summary)
        contact_sheets.append(contact_sheet)
        write_json(
            args.output / "summary.json",
            {
                "config_path": str(args.config),
                "project_root": str(PROJECT_ROOT),
                "accepted_requested": args.accepted,
                "accepted_count": len(accepted_summaries),
                "attempt_count": len(attempt_summaries),
                "accepted_scenes": accepted_summaries,
                "attempts": attempt_summaries,
            },
        )
        if len(accepted_summaries) >= args.accepted:
            break

    gallery_path = make_gallery(contact_sheets, args.output / "gallery.png")
    final_payload = {
        "config_path": str(args.config),
        "project_root": str(PROJECT_ROOT),
        "accepted_requested": args.accepted,
        "accepted_count": len(accepted_summaries),
        "attempt_count": len(attempt_summaries),
        "gallery": str(gallery_path),
        "accepted_scenes": accepted_summaries,
        "attempts": attempt_summaries,
    }
    write_json(args.output / "summary.json", final_payload)
    print(json.dumps(final_payload, indent=2, ensure_ascii=False), flush=True)
    if len(accepted_summaries) < args.accepted:
        raise SystemExit(
            f"accepted {len(accepted_summaries)} scenes, fewer than requested {args.accepted}"
        )


if __name__ == "__main__":
    main()
