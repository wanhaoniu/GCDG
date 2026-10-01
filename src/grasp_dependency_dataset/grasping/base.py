"""Proposal provider protocols."""

from __future__ import annotations

from typing import Protocol

from grasp_dependency_dataset.common.types import GraspProposal, StableScene


class ProposalProvider(Protocol):
    """Provider interface shared by AnyGrasp-style and SuctionNet-style backends."""

    def generate(
        self,
        scene: StableScene,
        target_id: str,
        top_k: int,
    ) -> list[GraspProposal]:
        """Generate candidate grasp proposals for the target object."""

        ...
