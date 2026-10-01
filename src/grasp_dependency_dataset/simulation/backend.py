"""Backend protocols for scene stabilization."""

from __future__ import annotations

from dataclasses import dataclass

from grasp_dependency_dataset.common.types import Pose


@dataclass(frozen=True)
class SettledSceneState:
    """Settled object poses plus stabilization metadata."""

    body_poses: dict[str, Pose]
    stable: bool
    metadata: dict[str, float | int | bool]


class SimulationBackend:
    """Minimal interface for a scene stabilization backend."""

    def settle_scene(
        self,
        scene_xml: str,
        body_names: list[str],
        settle_steps: int,
        stability_window: int,
        velocity_threshold: float,
    ) -> SettledSceneState:
        """Simulate the provided XML until the scene stabilizes."""

        raise NotImplementedError
