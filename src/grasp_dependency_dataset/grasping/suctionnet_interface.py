"""Suction proposal provider skeletons."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, tau

from grasp_dependency_dataset.common.config import ProposalConfig
from grasp_dependency_dataset.common.types import GraspProposal, GraspType, Pose, StableScene

from .icra2026_provider import ICRA2026SuctionNetProvider


@dataclass
class ExternalSuctionNetProvider:
    """Placeholder for the real SuctionNet inference backend."""

    config: ProposalConfig

    def generate(self, scene: StableScene, target_id: str, top_k: int) -> list[GraspProposal]:
        raise NotImplementedError(
            "Real SuctionNet integration is not implemented yet. "
            "Use `mock_suctionnet` for the current runnable demo."
        )


@dataclass
class MockSuctionNetProvider:
    """Deterministic suction proposal generator used by the demo pipeline."""

    config: ProposalConfig

    def generate(self, scene: StableScene, target_id: str, top_k: int) -> list[GraspProposal]:
        target = scene.get_object(target_id)
        center = target.pose.position
        radius = target.spec.bounding_radius

        proposals: list[GraspProposal] = []
        for index in range(max(top_k, 6)):
            angle = tau * (index / max(top_k, 6))
            radial_scale = 0.25 if index % 2 == 0 else 0.45
            contact = (
                center[0] + radial_scale * radius * cos(angle),
                center[1] + radial_scale * radius * sin(angle),
                center[2] + 0.75 * radius,
            )
            score = 1.0 - 0.02 * index
            proposals.append(
                GraspProposal(
                    grasp_id=f"{target_id}_suction_{index:02d}",
                    target_id=target_id,
                    grasp_type=GraspType.SUCTION,
                    pose=Pose(position=contact),
                    source="SuctionNet-mock",
                    proposal_score=score,
                    approach_vector=(0.0, 0.0, -1.0),
                    suction_radius=min(0.025, max(0.012, 0.6 * radius)),
                    metadata={"azimuth_rad": angle},
                )
            )

        return proposals[:top_k]


def build_suction_provider(config: ProposalConfig):
    """Instantiate the configured suction proposal provider."""

    if config.suction_backend == "mock_suctionnet":
        return MockSuctionNetProvider(config)
    if config.suction_backend == "icra2026_suctionnet":
        return ICRA2026SuctionNetProvider(config)
    if config.suction_backend == "external_suctionnet":
        return ExternalSuctionNetProvider(config)
    raise ValueError(f"Unsupported suction backend: {config.suction_backend}")
