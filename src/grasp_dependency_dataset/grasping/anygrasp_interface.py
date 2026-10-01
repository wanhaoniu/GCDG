"""Parallel jaw proposal provider skeletons."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, tau

from grasp_dependency_dataset.common.config import ProposalConfig
from grasp_dependency_dataset.common.types import GraspProposal, GraspType, Pose, StableScene

from .icra2026_provider import ICRA2026AnyGraspProvider


@dataclass
class ExternalAnyGraspProvider:
    """Placeholder for the real AnyGrasp inference backend."""

    config: ProposalConfig

    def generate(self, scene: StableScene, target_id: str, top_k: int) -> list[GraspProposal]:
        raise NotImplementedError(
            "Real AnyGrasp integration is not implemented yet. "
            "Use `mock_anygrasp` for the current runnable demo."
        )


@dataclass
class MockAnyGraspProvider:
    """Deterministic parallel-jaw proposal generator used by the demo pipeline."""

    config: ProposalConfig

    def generate(self, scene: StableScene, target_id: str, top_k: int) -> list[GraspProposal]:
        target = scene.get_object(target_id)
        center = target.pose.position
        radius = target.spec.bounding_radius
        jaw_width = min(0.10, max(0.04, 2.4 * radius))

        proposals: list[GraspProposal] = []
        for index in range(max(top_k, 8)):
            angle = tau * (index / max(top_k, 8))
            approach = (-cos(angle), -sin(angle), -0.25)
            contact = (
                center[0] + 0.01 * cos(angle),
                center[1] + 0.01 * sin(angle),
                center[2] + min(0.02, 0.35 * radius),
            )
            score = 1.0 - 0.015 * index
            proposals.append(
                GraspProposal(
                    grasp_id=f"{target_id}_parallel_{index:02d}",
                    target_id=target_id,
                    grasp_type=GraspType.PARALLEL_JAW,
                    pose=Pose(position=contact),
                    source="AnyGrasp-mock",
                    proposal_score=score,
                    approach_vector=approach,
                    jaw_width=jaw_width,
                    metadata={"azimuth_rad": angle},
                )
            )

        return proposals[:top_k]


def build_parallel_provider(config: ProposalConfig):
    """Instantiate the configured parallel-jaw proposal provider."""

    if config.parallel_backend == "mock_anygrasp":
        return MockAnyGraspProvider(config)
    if config.parallel_backend == "icra2026_anygrasp":
        return ICRA2026AnyGraspProvider(config)
    if config.parallel_backend == "external_anygrasp":
        return ExternalAnyGraspProvider(config)
    raise ValueError(f"Unsupported parallel backend: {config.parallel_backend}")
