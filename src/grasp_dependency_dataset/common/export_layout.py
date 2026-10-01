"""Canonical export layout helpers for scene-level and target-level artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneExportPaths:
    """Filesystem paths shared by every target in one scene."""

    output_root: Path
    scene_id: str
    scene_dir: Path
    scene_path: Path
    shared_observation_dir: Path
    targets_dir: Path


@dataclass(frozen=True)
class SampleExportPaths:
    """Filesystem paths for one `(scene, target)` sample."""

    scene: SceneExportPaths
    target_id: str
    target_dir: Path
    observation_dir: Path
    proposal_path: Path
    label_path: Path
    manifest_path: Path


def scene_export_paths(output_root: str | Path, scene_id: str) -> SceneExportPaths:
    """Build canonical paths for one scene."""

    root = Path(output_root)
    scene_dir = root / "scenes" / scene_id
    return SceneExportPaths(
        output_root=root,
        scene_id=scene_id,
        scene_dir=scene_dir,
        scene_path=scene_dir / "scene.json",
        shared_observation_dir=scene_dir / "observations" / "shared",
        targets_dir=scene_dir / "targets",
    )


def sample_export_paths(output_root: str | Path, scene_id: str, target_id: str) -> SampleExportPaths:
    """Build canonical paths for one target sample in a scene."""

    scene_paths = scene_export_paths(output_root, scene_id)
    target_dir = scene_paths.targets_dir / target_id
    return SampleExportPaths(
        scene=scene_paths,
        target_id=target_id,
        target_dir=target_dir,
        observation_dir=target_dir / "observations",
        proposal_path=target_dir / "proposals.json",
        label_path=target_dir / "labels.json",
        manifest_path=target_dir / "manifest.json",
    )


def pilot_export_dir(output_root: str | Path, scene_id: str, name: str = "dependency_pilot") -> Path:
    """Return the canonical directory for pilot summaries and galleries."""

    return Path(output_root) / "pilots" / f"{scene_id}__{name}"
