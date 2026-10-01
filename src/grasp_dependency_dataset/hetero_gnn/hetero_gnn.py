"""Models for object-to-grasp dependency prediction.

The main model is a PyTorch implementation of heterogeneous message passing
that does not require PyTorch Geometric. A small NumPy edge baseline is also
provided so dataset plumbing and metrics can run in minimal environments where
PyTorch is not installed.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from grasp_dependency_dataset.hetero_gnn.graph_features import OG_EDGE_FEATURE_NAMES
from grasp_dependency_dataset.hetero_gnn.metrics import LABEL_NAMES


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:  # pragma: no cover - exercised implicitly in minimal environments.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed. Install torch for the full HeteroDependencyGNN "
            "or use backend: numpy_baseline to validate the dataset pipeline."
        )


if TORCH_AVAILABLE:

    def make_mlp(
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.0,
        final_activation: bool = False,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        if final_activation:
            layers.append(nn.SiLU())
        return nn.Sequential(*layers)


    class MultiTaskEdgeDecoder(nn.Module):
        """Shared edge trunk with small label-specific heads.

        The dependency labels are related but not identical: progress-any and
        sufficient encode planner-facing notions of usefulness, while
        approach/lift encode phase-specific failure modes. A shared trunk keeps
        the common object-grasp representation, and shallow heads give each
        label a little extra capacity without changing the graph message-passing
        interface.
        """

        def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int = 3,
            *,
            trunk_layers: int = 2,
            head_layers: int = 2,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.trunk = make_mlp(
                in_dim,
                hidden_dim,
                hidden_dim,
                num_layers=max(1, trunk_layers),
                dropout=dropout,
                final_activation=True,
            )
            self.heads = nn.ModuleList(
                [
                    make_mlp(
                        hidden_dim,
                        hidden_dim,
                        1,
                        num_layers=max(1, head_layers),
                        dropout=dropout,
                    )
                    for _ in range(out_dim)
                ]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.trunk(x)
            return torch.cat([head(h) for head in self.heads], dim=-1)


    class BipartiteEdgeGatedConv(nn.Module):
        """Edge-conditioned gated aggregation from source to destination nodes."""

        def __init__(
            self,
            hidden_dim: int,
            edge_dim: int,
            *,
            dropout: float = 0.0,
            variant: str = "legacy",
        ) -> None:
            super().__init__()
            self.variant = str(variant)
            if self.variant == "endpoint_attention":
                edge_input_dim = 2 * hidden_dim + edge_dim
                self.message = make_mlp(edge_input_dim, hidden_dim, hidden_dim, dropout=dropout)
                self.gate = make_mlp(edge_input_dim, hidden_dim, hidden_dim, dropout=dropout)
                self.attention = make_mlp(edge_input_dim, hidden_dim, 1, dropout=dropout)
                self.update = make_mlp(3 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
                self.residual_gate = nn.Linear(3 * hidden_dim, hidden_dim)
            elif self.variant == "legacy":
                edge_input_dim = hidden_dim + edge_dim
                self.message = make_mlp(edge_input_dim, hidden_dim, hidden_dim, dropout=dropout)
                self.gate = make_mlp(edge_input_dim, hidden_dim, hidden_dim, dropout=dropout)
                self.update = make_mlp(2 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
                self.attention = None
                self.residual_gate = None
            else:
                raise ValueError(f"Unknown BipartiteEdgeGatedConv variant: {variant}")
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            src_x: torch.Tensor,
            dst_x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
        ) -> torch.Tensor:
            if edge_index.numel() == 0 or src_x.numel() == 0 or dst_x.numel() == 0:
                return dst_x
            src_idx = edge_index[0].long()
            dst_idx = edge_index[1].long()
            if self.variant == "endpoint_attention":
                msg_input = torch.cat([src_x[src_idx], dst_x[dst_idx], edge_attr], dim=-1)
                assert self.attention is not None and self.residual_gate is not None
                edge_weight = torch.sigmoid(self.attention(msg_input))
                msg = self.message(msg_input) * torch.sigmoid(self.gate(msg_input)) * edge_weight
                agg = torch.zeros_like(dst_x)
                agg.index_add_(0, dst_idx, msg)
                deg = torch.zeros(dst_x.shape[0], 1, dtype=dst_x.dtype, device=dst_x.device)
                deg.index_add_(0, dst_idx, edge_weight)
                agg = agg / deg.clamp_min(1.0)
                update_input = torch.cat([dst_x, agg, dst_x * agg], dim=-1)
                delta = self.update(update_input)
                residual_weight = torch.sigmoid(self.residual_gate(update_input))
                return self.norm(dst_x + residual_weight * delta)

            msg_input = torch.cat([src_x[src_idx], edge_attr], dim=-1)
            msg = self.message(msg_input) * torch.sigmoid(self.gate(msg_input))
            agg = torch.zeros_like(dst_x)
            agg.index_add_(0, dst_idx, msg)
            deg = torch.zeros(dst_x.shape[0], 1, dtype=dst_x.dtype, device=dst_x.device)
            deg.index_add_(0, dst_idx, torch.ones(msg.shape[0], 1, dtype=dst_x.dtype, device=dst_x.device))
            agg = agg / deg.clamp_min(1.0)
            delta = self.update(torch.cat([dst_x, agg], dim=-1))
            return self.norm(dst_x + delta)


    class GraspConditionedObjectConv(nn.Module):
        """Object-object message passing inside each grasp-conditioned object graph."""

        def __init__(self, hidden_dim: int, edge_dim: int, *, dropout: float = 0.0) -> None:
            super().__init__()
            input_dim = 2 * hidden_dim + edge_dim
            self.message = make_mlp(input_dim, hidden_dim, hidden_dim, dropout=dropout)
            self.gate = make_mlp(input_dim, hidden_dim, hidden_dim, dropout=dropout)
            self.update = make_mlp(3 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            h: torch.Tensor,
            src_edge_idx: torch.Tensor,
            dst_edge_idx: torch.Tensor,
            edge_attr: torch.Tensor,
        ) -> torch.Tensor:
            if h.numel() == 0 or src_edge_idx.numel() == 0:
                return h
            msg_input = torch.cat([h[src_edge_idx], h[dst_edge_idx], edge_attr], dim=-1)
            msg = self.message(msg_input) * torch.sigmoid(self.gate(msg_input))
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst_edge_idx, msg)
            deg = h.new_zeros((h.shape[0], 1))
            deg.index_add_(0, dst_edge_idx, torch.ones(msg.shape[0], 1, dtype=h.dtype, device=h.device))
            agg = agg / deg.clamp_min(1.0)
            delta = self.update(torch.cat([h, agg, h * agg], dim=-1))
            return self.norm(h + delta)


    class GraspConditionedBlockerRelationConv(nn.Module):
        """Physical relation update over object-grasp edges for the same grasp.

        Unlike generic object-object propagation, the message sees both objects'
        object-grasp geometry for the current grasp. This makes the OO branch
        explicitly grasp-conditioned: a neighbor matters only through how both
        objects relate to the same approach/lift swept volume.
        """

        def __init__(
            self,
            hidden_dim: int,
            oo_edge_dim: int,
            og_edge_dim: int,
            *,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            pair_dim = 2 * hidden_dim + oo_edge_dim + 3 * og_edge_dim
            self.message = make_mlp(pair_dim, hidden_dim, hidden_dim, dropout=dropout)
            self.gate = make_mlp(pair_dim, hidden_dim, 1, dropout=dropout)
            self.update = make_mlp(4 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            h: torch.Tensor,
            src_edge_idx: torch.Tensor,
            dst_edge_idx: torch.Tensor,
            edge_attr_oo: torch.Tensor,
            edge_attr_og: torch.Tensor,
        ) -> torch.Tensor:
            if h.numel() == 0 or src_edge_idx.numel() == 0:
                return h
            src_og = edge_attr_og[src_edge_idx]
            dst_og = edge_attr_og[dst_edge_idx]
            pair_input = torch.cat(
                [
                    h[src_edge_idx],
                    h[dst_edge_idx],
                    edge_attr_oo,
                    src_og,
                    dst_og,
                    torch.abs(src_og - dst_og),
                ],
                dim=-1,
            )
            weights = torch.sigmoid(self.gate(pair_input)).to(dtype=h.dtype).clamp_min(1e-4)
            msg = self.message(pair_input) * weights
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst_edge_idx, msg)
            denom = h.new_zeros((h.shape[0], 1))
            denom.index_add_(0, dst_edge_idx, weights)
            agg = agg / denom.clamp_min(1e-4)
            delta = self.update(torch.cat([h, agg, h * agg, torch.abs(h - agg)], dim=-1))
            return self.norm(h + delta)


    class HeteroDependencyGNN(nn.Module):
        """Target-centric heterogeneous GNN for typed object-to-grasp edges."""

        def __init__(
            self,
            *,
            object_in_dim: int,
            grasp_in_dim: int,
            oo_edge_dim: int,
            gg_edge_dim: int,
            og_edge_dim: int,
            num_object_classes: int,
            num_edge_labels: int = len(LABEL_NAMES),
            hidden_dim: int = 128,
            class_emb_dim: int = 16,
            grasp_type_emb_dim: int = 8,
            message_layers: int = 2,
            cross_layers: int = 1,
            decoder_layers: int = 3,
            dropout: float = 0.10,
            conv_variant: str = "legacy",
            decoder_interactions: bool = False,
            decoder_use_edge_attr: bool = True,
            decoder_raw_features: bool = False,
            separate_label_heads: bool = False,
            decoder_head_layers: int = 2,
            edge_context_layers: int = 0,
            edge_context_source: str = "raw",
            edge_context_use_edge_attr: bool = True,
            edge_context_use_object: bool = True,
            edge_context_interactions: bool = True,
            edge_context_attention: bool = False,
            edge_context_attention_query: bool = False,
            edge_context_logit_residual: bool = False,
            edge_context_logit_init: float = 0.10,
            grasp_conditioned_object_layers: int = 0,
            grasp_conditioned_object_use_edge_attr: bool = True,
            grasp_conditioned_object_require_edges: bool = False,
            grasp_conditioned_object_interactions: bool = True,
            grasp_conditioned_object_decoder_context: bool = True,
            grasp_conditioned_object_logit_residual: bool = False,
            grasp_conditioned_object_logit_init: float = 0.10,
            blocker_relation_layers: int = 0,
            blocker_relation_decoder_context: bool = True,
            blocker_relation_interactions: bool = True,
            blocker_relation_logit_residual: bool = False,
            blocker_relation_logit_init: float = 0.05,
        ) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_edge_labels = int(num_edge_labels)
            self.decoder_interactions = bool(decoder_interactions)
            self.decoder_use_edge_attr = bool(decoder_use_edge_attr)
            self.decoder_raw_features = bool(decoder_raw_features)
            self.separate_label_heads = bool(separate_label_heads)
            self.edge_context_layers_count = max(0, int(edge_context_layers))
            self.edge_context_source = str(edge_context_source or "raw").lower()
            self.edge_context_use_edge_attr = bool(edge_context_use_edge_attr)
            self.edge_context_use_object = bool(edge_context_use_object)
            self.edge_context_interactions = bool(edge_context_interactions)
            self.edge_context_attention = bool(edge_context_attention)
            self.edge_context_attention_query = bool(edge_context_attention_query)
            self.edge_context_logit_residual = bool(edge_context_logit_residual)
            self.gc_object_layers_count = max(0, int(grasp_conditioned_object_layers))
            self.gc_object_use_edge_attr = bool(grasp_conditioned_object_use_edge_attr)
            self.gc_object_require_edges = bool(grasp_conditioned_object_require_edges)
            self.gc_object_interactions = bool(grasp_conditioned_object_interactions)
            self.gc_object_decoder_context = bool(grasp_conditioned_object_decoder_context)
            self.gc_object_logit_residual = bool(grasp_conditioned_object_logit_residual)
            self.blocker_relation_layers_count = max(0, int(blocker_relation_layers))
            self.blocker_relation_decoder_context = bool(blocker_relation_decoder_context)
            self.blocker_relation_interactions = bool(blocker_relation_interactions)
            self.blocker_relation_logit_residual = bool(blocker_relation_logit_residual)
            self.object_class_emb = nn.Embedding(max(1, num_object_classes), class_emb_dim)
            self.grasp_type_emb = nn.Embedding(2, grasp_type_emb_dim)
            self.object_encoder = make_mlp(
                object_in_dim + class_emb_dim,
                hidden_dim,
                hidden_dim,
                num_layers=2,
                dropout=dropout,
                final_activation=True,
            )
            self.grasp_encoder = make_mlp(
                grasp_in_dim + grasp_type_emb_dim,
                hidden_dim,
                hidden_dim,
                num_layers=2,
                dropout=dropout,
                final_activation=True,
            )
            self.oo_layers = nn.ModuleList(
                [
                    BipartiteEdgeGatedConv(hidden_dim, oo_edge_dim, dropout=dropout, variant=conv_variant)
                    for _ in range(message_layers)
                ]
            )
            self.gg_layers = nn.ModuleList(
                [
                    BipartiteEdgeGatedConv(hidden_dim, gg_edge_dim, dropout=dropout, variant=conv_variant)
                    for _ in range(message_layers)
                ]
            )
            self.obj_to_grasp_layers = nn.ModuleList(
                [
                    BipartiteEdgeGatedConv(hidden_dim, og_edge_dim, dropout=dropout, variant=conv_variant)
                    for _ in range(cross_layers)
                ]
            )
            self.grasp_to_obj_layers = nn.ModuleList(
                [
                    BipartiteEdgeGatedConv(hidden_dim, og_edge_dim, dropout=dropout, variant=conv_variant)
                    for _ in range(cross_layers)
                ]
            )
            if self.edge_context_layers_count > 0:
                if self.edge_context_source in {"graph", "graph_state", "state"}:
                    edge_context_in_dim = 4 * hidden_dim if self.edge_context_interactions else 2 * hidden_dim
                elif self.edge_context_source in {"hybrid", "raw_graph", "raw+graph"}:
                    graph_dim = 4 * hidden_dim if self.edge_context_interactions else 2 * hidden_dim
                    edge_context_in_dim = object_in_dim + grasp_in_dim + graph_dim
                elif self.edge_context_source == "raw":
                    edge_context_in_dim = object_in_dim + grasp_in_dim
                else:
                    raise ValueError(f"Unknown edge_context_source: {edge_context_source}")
                if self.edge_context_use_edge_attr:
                    edge_context_in_dim += og_edge_dim
                self.edge_context_encoder = make_mlp(
                    edge_context_in_dim,
                    hidden_dim,
                    hidden_dim,
                    num_layers=2,
                    dropout=dropout,
                    final_activation=True,
                )
                edge_context_update_dim = 3 * hidden_dim if self.edge_context_use_object else 2 * hidden_dim
                self.edge_context_mlps = nn.ModuleList(
                    [
                        make_mlp(
                            edge_context_update_dim,
                            hidden_dim,
                            hidden_dim,
                            num_layers=2,
                            dropout=dropout,
                        )
                        for _ in range(self.edge_context_layers_count)
                    ]
                )
                self.edge_context_norms = nn.ModuleList(
                    [nn.LayerNorm(hidden_dim) for _ in range(self.edge_context_layers_count)]
                )
                if self.edge_context_attention:
                    attention_in_dim = 2 * hidden_dim if self.edge_context_attention_query else hidden_dim
                    self.edge_context_attention_mlps = nn.ModuleList(
                        [
                            make_mlp(attention_in_dim, hidden_dim, 1, num_layers=2, dropout=dropout)
                            for _ in range(self.edge_context_layers_count + 1)
                        ]
                    )
                else:
                    self.edge_context_attention_mlps = nn.ModuleList()
                edge_context_decoder_in_dim = 2 * hidden_dim
                if self.edge_context_use_object:
                    edge_context_decoder_in_dim += hidden_dim
                if self.edge_context_interactions:
                    edge_context_decoder_in_dim += 2 * hidden_dim
                if self.edge_context_logit_residual:
                    self.edge_context_logit_decoder = make_mlp(
                        edge_context_decoder_in_dim,
                        hidden_dim,
                        self.num_edge_labels,
                        num_layers=max(2, decoder_layers - 1),
                        dropout=dropout,
                    )
                    self.edge_context_logit_scale = nn.Parameter(
                        torch.tensor(float(edge_context_logit_init), dtype=torch.float32)
                    )
                else:
                    self.edge_context_logit_decoder = None
                    self.edge_context_logit_scale = None
            else:
                self.edge_context_encoder = None
                self.edge_context_mlps = nn.ModuleList()
                self.edge_context_norms = nn.ModuleList()
                self.edge_context_attention_mlps = nn.ModuleList()
                self.edge_context_logit_decoder = None
                self.edge_context_logit_scale = None
            if self.gc_object_layers_count > 0:
                gc_encoder_in_dim = 2 * hidden_dim
                if self.gc_object_use_edge_attr:
                    gc_encoder_in_dim += og_edge_dim
                self.gc_object_encoder = make_mlp(
                    gc_encoder_in_dim,
                    hidden_dim,
                    hidden_dim,
                    num_layers=2,
                    dropout=dropout,
                    final_activation=True,
                )
                self.gc_object_layers = nn.ModuleList(
                    [
                        GraspConditionedObjectConv(hidden_dim, oo_edge_dim, dropout=dropout)
                        for _ in range(self.gc_object_layers_count)
                    ]
                )
                self.gc_context_mlps = nn.ModuleList(
                    [
                        make_mlp(
                            2 * hidden_dim,
                            hidden_dim,
                            hidden_dim,
                            num_layers=2,
                            dropout=dropout,
                        )
                        for _ in range(self.gc_object_layers_count)
                    ]
                )
                self.gc_context_norms = nn.ModuleList(
                    [nn.LayerNorm(hidden_dim) for _ in range(self.gc_object_layers_count)]
                )
                gc_decoder_in_dim = 4 * hidden_dim if self.gc_object_interactions else 2 * hidden_dim
                self.gc_logit_decoder = make_mlp(
                    gc_decoder_in_dim,
                    hidden_dim,
                    self.num_edge_labels,
                    num_layers=max(2, decoder_layers - 1),
                    dropout=dropout,
                )
                if self.gc_object_logit_residual:
                    self.gc_logit_scale = nn.Parameter(
                        torch.tensor(float(grasp_conditioned_object_logit_init), dtype=torch.float32)
                    )
                else:
                    self.gc_logit_scale = None
            else:
                self.gc_object_encoder = None
                self.gc_object_layers = nn.ModuleList()
                self.gc_context_mlps = nn.ModuleList()
                self.gc_context_norms = nn.ModuleList()
                self.gc_logit_decoder = None
                self.gc_logit_scale = None
            if self.blocker_relation_layers_count > 0:
                self.blocker_relation_encoder = make_mlp(
                    2 * hidden_dim + og_edge_dim,
                    hidden_dim,
                    hidden_dim,
                    num_layers=2,
                    dropout=dropout,
                    final_activation=True,
                )
                self.blocker_relation_layers = nn.ModuleList(
                    [
                        GraspConditionedBlockerRelationConv(
                            hidden_dim,
                            oo_edge_dim,
                            og_edge_dim,
                            dropout=dropout,
                        )
                        for _ in range(self.blocker_relation_layers_count)
                    ]
                )
                self.blocker_relation_context_mlps = nn.ModuleList(
                    [
                        make_mlp(2 * hidden_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
                        for _ in range(self.blocker_relation_layers_count)
                    ]
                )
                self.blocker_relation_context_norms = nn.ModuleList(
                    [nn.LayerNorm(hidden_dim) for _ in range(self.blocker_relation_layers_count)]
                )
                blocker_relation_decoder_in_dim = 4 * hidden_dim if self.blocker_relation_interactions else 2 * hidden_dim
                self.blocker_relation_logit_decoder = make_mlp(
                    blocker_relation_decoder_in_dim,
                    hidden_dim,
                    self.num_edge_labels,
                    num_layers=max(2, decoder_layers - 1),
                    dropout=dropout,
                )
                if self.blocker_relation_logit_residual:
                    self.blocker_relation_logit_scale = nn.Parameter(
                        torch.tensor(float(blocker_relation_logit_init), dtype=torch.float32)
                    )
                else:
                    self.blocker_relation_logit_scale = None
            else:
                self.blocker_relation_encoder = None
                self.blocker_relation_layers = nn.ModuleList()
                self.blocker_relation_context_mlps = nn.ModuleList()
                self.blocker_relation_context_norms = nn.ModuleList()
                self.blocker_relation_logit_decoder = None
                self.blocker_relation_logit_scale = None
            edge_decoder_in_dim = 4 * hidden_dim if self.decoder_interactions else 2 * hidden_dim
            if self.decoder_use_edge_attr:
                edge_decoder_in_dim += og_edge_dim
            if self.decoder_raw_features:
                edge_decoder_in_dim += object_in_dim + grasp_in_dim
            if self.edge_context_layers_count > 0:
                edge_decoder_in_dim += 2 * hidden_dim
                if self.edge_context_use_object:
                    edge_decoder_in_dim += hidden_dim
                if self.edge_context_interactions:
                    edge_decoder_in_dim += 2 * hidden_dim
            if self.gc_object_layers_count > 0 and self.gc_object_decoder_context:
                edge_decoder_in_dim += 2 * hidden_dim
                if self.gc_object_interactions:
                    edge_decoder_in_dim += 2 * hidden_dim
            if self.blocker_relation_layers_count > 0 and self.blocker_relation_decoder_context:
                edge_decoder_in_dim += 2 * hidden_dim
                if self.blocker_relation_interactions:
                    edge_decoder_in_dim += 2 * hidden_dim
            if self.separate_label_heads:
                self.decoder = MultiTaskEdgeDecoder(
                    edge_decoder_in_dim,
                    hidden_dim,
                    self.num_edge_labels,
                    trunk_layers=max(1, decoder_layers - 1),
                    head_layers=decoder_head_layers,
                    dropout=dropout,
                )
            else:
                self.decoder = make_mlp(
                    edge_decoder_in_dim,
                    hidden_dim,
                    self.num_edge_labels,
                    num_layers=decoder_layers,
                    dropout=dropout,
                )

        def _group_mean(self, h: torch.Tensor, group: torch.Tensor, num_groups: int) -> torch.Tensor:
            if num_groups <= 0:
                return h.new_zeros((0, h.shape[1]))
            group = group.long().clamp(0, num_groups - 1)
            context = h.new_zeros((num_groups, h.shape[1]))
            counts = h.new_zeros((num_groups, 1))
            context.index_add_(0, group, h)
            counts.index_add_(0, group, torch.ones(h.shape[0], 1, dtype=h.dtype, device=h.device))
            return context / counts.clamp_min(1.0)

        def _group_weighted_mean(
            self,
            h: torch.Tensor,
            group: torch.Tensor,
            num_groups: int,
            weight_logits: torch.Tensor,
        ) -> torch.Tensor:
            if num_groups <= 0:
                return h.new_zeros((0, h.shape[1]))
            group = group.long().clamp(0, num_groups - 1)
            weights = torch.sigmoid(weight_logits).to(dtype=h.dtype).clamp_min(1e-4)
            context = h.new_zeros((num_groups, h.shape[1]))
            counts = h.new_zeros((num_groups, 1))
            context.index_add_(0, group, h * weights)
            counts.index_add_(0, group, weights)
            return context / counts.clamp_min(1e-4)

        def _edge_context_encoder_input(
            self,
            x_obj_edge: torch.Tensor,
            x_grasp_edge: torch.Tensor,
            obj_edge: torch.Tensor,
            grasp_edge: torch.Tensor,
            edge_attr_og: torch.Tensor,
        ) -> torch.Tensor:
            if self.edge_context_source in {"graph", "graph_state", "state"}:
                if self.edge_context_interactions:
                    parts = [obj_edge, grasp_edge, obj_edge * grasp_edge, torch.abs(obj_edge - grasp_edge)]
                else:
                    parts = [obj_edge, grasp_edge]
            elif self.edge_context_source in {"hybrid", "raw_graph", "raw+graph"}:
                if self.edge_context_interactions:
                    parts = [
                        x_obj_edge,
                        x_grasp_edge,
                        obj_edge,
                        grasp_edge,
                        obj_edge * grasp_edge,
                        torch.abs(obj_edge - grasp_edge),
                    ]
                else:
                    parts = [x_obj_edge, x_grasp_edge, obj_edge, grasp_edge]
            else:
                parts = [x_obj_edge, x_grasp_edge]
            if self.edge_context_use_edge_attr:
                parts.append(edge_attr_og)
            return torch.cat(parts, dim=-1)

        def _edge_context_attention_logits(
            self,
            layer_index: int,
            h_edge: torch.Tensor,
            grasp_edge: torch.Tensor,
        ) -> torch.Tensor:
            attention_mlp = self.edge_context_attention_mlps[layer_index]
            if self.edge_context_attention_query:
                return attention_mlp(torch.cat([h_edge, grasp_edge], dim=-1))
            return attention_mlp(h_edge)

        def _gc_object_encoder_input(
            self,
            obj_edge: torch.Tensor,
            grasp_edge: torch.Tensor,
            edge_attr_og: torch.Tensor,
        ) -> torch.Tensor:
            parts = [obj_edge, grasp_edge]
            if self.gc_object_use_edge_attr:
                parts.append(edge_attr_og)
            return torch.cat(parts, dim=-1)

        def _grasp_conditioned_oo_edges(
            self,
            batch: dict[str, torch.Tensor],
            edge_index_og: torch.Tensor,
            *,
            num_objects: int,
            num_grasps: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            edge_index_oo = batch["edge_index_oo"]
            edge_attr_oo = batch["edge_attr_oo"]
            if edge_index_oo.numel() == 0 or edge_index_og.numel() == 0 or num_objects <= 0 or num_grasps <= 0:
                empty_idx = edge_index_og.new_zeros((0,), dtype=torch.long)
                return empty_idx, empty_idx, edge_attr_oo.new_zeros((0, edge_attr_oo.shape[1]))

            obj_idx = edge_index_og[0].long()
            grasp_idx = edge_index_og[1].long()
            edge_lookup = edge_index_og.new_full((num_objects, num_grasps), -1, dtype=torch.long)
            edge_lookup[obj_idx, grasp_idx] = torch.arange(edge_index_og.shape[1], dtype=torch.long, device=edge_index_og.device)

            object_batch = batch.get("object_batch")
            grasp_batch = batch.get("grasp_batch")
            if object_batch is None or object_batch.numel() == 0 or grasp_batch is None or grasp_batch.numel() == 0:
                empty_idx = edge_index_og.new_zeros((0,), dtype=torch.long)
                return empty_idx, empty_idx, edge_attr_oo.new_zeros((0, edge_attr_oo.shape[1]))

            oo_src = edge_index_oo[0].long()
            oo_dst = edge_index_oo[1].long()
            src_chunks: list[torch.Tensor] = []
            dst_chunks: list[torch.Tensor] = []
            attr_chunks: list[torch.Tensor] = []
            num_samples = int(max(object_batch.max().detach().cpu().item(), grasp_batch.max().detach().cpu().item()) + 1)
            for sample_i in range(num_samples):
                grasp_ids = torch.nonzero(grasp_batch.long() == sample_i, as_tuple=False).flatten()
                if grasp_ids.numel() == 0:
                    continue
                oo_mask = (object_batch[oo_src].long() == sample_i) & (object_batch[oo_dst].long() == sample_i)
                oo_ids = torch.nonzero(oo_mask, as_tuple=False).flatten()
                if oo_ids.numel() == 0:
                    continue
                src_obj = oo_src[oo_ids]
                dst_obj = oo_dst[oo_ids]
                n_grasps = int(grasp_ids.numel())
                src_edges = edge_lookup[src_obj.repeat_interleave(n_grasps), grasp_ids.repeat(src_obj.numel())]
                dst_edges = edge_lookup[dst_obj.repeat_interleave(n_grasps), grasp_ids.repeat(dst_obj.numel())]
                valid = (src_edges >= 0) & (dst_edges >= 0)
                if not bool(valid.any()):
                    continue
                src_chunks.append(src_edges[valid])
                dst_chunks.append(dst_edges[valid])
                attr_chunks.append(edge_attr_oo[oo_ids].repeat_interleave(n_grasps, dim=0)[valid])

            if not src_chunks:
                empty_idx = edge_index_og.new_zeros((0,), dtype=torch.long)
                return empty_idx, empty_idx, edge_attr_oo.new_zeros((0, edge_attr_oo.shape[1]))
            return torch.cat(src_chunks, dim=0), torch.cat(dst_chunks, dim=0), torch.cat(attr_chunks, dim=0)

        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            x_obj = batch["x_obj"]
            x_grasp = batch["x_grasp"]
            object_class_ids = batch["object_class_ids"].clamp_min(0)
            grasp_type_ids = batch["grasp_type_ids"].clamp(0, 1)
            if x_grasp.shape[0] == 0:
                return {
                    "edge_logits": x_obj.new_zeros((batch["edge_index_og"].shape[1], self.num_edge_labels)),
                }

            if x_obj.shape[0] > 0:
                h_obj = self.object_encoder(torch.cat([x_obj, self.object_class_emb(object_class_ids)], dim=-1))
            else:
                h_obj = x_grasp.new_zeros((0, self.hidden_dim))
            h_grasp = self.grasp_encoder(torch.cat([x_grasp, self.grasp_type_emb(grasp_type_ids)], dim=-1))

            for layer in self.oo_layers:
                h_obj = layer(h_obj, h_obj, batch["edge_index_oo"], batch["edge_attr_oo"])
            for layer in self.gg_layers:
                h_grasp = layer(h_grasp, h_grasp, batch["edge_index_gg"], batch["edge_attr_gg"])

            edge_index_og = batch["edge_index_og"]
            edge_attr_og = batch["edge_attr_og"]
            if edge_index_og.numel() > 0 and h_obj.shape[0] > 0:
                edge_index_go = torch.stack([edge_index_og[1], edge_index_og[0]], dim=0)
                for obj_to_grasp, grasp_to_obj in zip(self.obj_to_grasp_layers, self.grasp_to_obj_layers):
                    h_grasp = obj_to_grasp(h_obj, h_grasp, edge_index_og, edge_attr_og)
                    h_obj = grasp_to_obj(h_grasp, h_obj, edge_index_go, edge_attr_og)

                obj_idx = edge_index_og[0].long()
                grasp_idx = edge_index_og[1].long()
                obj_edge = h_obj[obj_idx]
                grasp_edge = h_grasp[grasp_idx]
                edge_context_parts = []
                edge_context_logits = None
                gc_object_parts = []
                gc_logits = None
                blocker_relation_parts = []
                blocker_relation_logits = None
                if self.edge_context_layers_count > 0 and self.edge_context_encoder is not None:
                    h_edge = self.edge_context_encoder(
                        self._edge_context_encoder_input(
                            x_obj[obj_idx],
                            x_grasp[grasp_idx],
                            obj_edge,
                            grasp_edge,
                            edge_attr_og,
                        )
                    )
                    num_grasps = int(max(h_grasp.shape[0], int(grasp_idx.max().detach().cpu().item()) + 1))
                    num_objects = int(max(h_obj.shape[0], int(obj_idx.max().detach().cpu().item()) + 1))
                    if self.edge_context_attention and self.edge_context_attention_mlps:
                        grasp_context = self._group_weighted_mean(
                            h_edge,
                            grasp_idx,
                            num_grasps,
                            self._edge_context_attention_logits(0, h_edge, grasp_edge),
                        )
                    else:
                        grasp_context = self._group_mean(h_edge, grasp_idx, num_grasps)
                    object_context = self._group_mean(h_edge, obj_idx, num_objects)
                    for layer_i, (edge_mlp, edge_norm) in enumerate(zip(self.edge_context_mlps, self.edge_context_norms)):
                        if self.edge_context_use_object:
                            update_input = torch.cat(
                                [h_edge, grasp_context[grasp_idx], object_context[obj_idx]],
                                dim=-1,
                            )
                        else:
                            update_input = torch.cat([h_edge, grasp_context[grasp_idx]], dim=-1)
                        h_edge = edge_norm(h_edge + edge_mlp(update_input))
                        if self.edge_context_attention and self.edge_context_attention_mlps:
                            grasp_context = self._group_weighted_mean(
                                h_edge,
                                grasp_idx,
                                num_grasps,
                                self._edge_context_attention_logits(layer_i + 1, h_edge, grasp_edge),
                            )
                        else:
                            grasp_context = self._group_mean(h_edge, grasp_idx, num_grasps)
                        object_context = self._group_mean(h_edge, obj_idx, num_objects)
                    grasp_edge_context = grasp_context[grasp_idx]
                    edge_context_parts = [h_edge, grasp_edge_context]
                    if self.edge_context_use_object:
                        edge_context_parts.append(object_context[obj_idx])
                    if self.edge_context_interactions:
                        edge_context_parts.extend(
                            [
                                h_edge * grasp_edge_context,
                                torch.abs(h_edge - grasp_edge_context),
                            ]
                        )
                    if self.edge_context_logit_decoder is not None and self.edge_context_logit_scale is not None:
                        edge_context_logits = self.edge_context_logit_decoder(torch.cat(edge_context_parts, dim=-1))
                if self.gc_object_layers_count > 0 and self.gc_object_encoder is not None:
                    h_gc = self.gc_object_encoder(self._gc_object_encoder_input(obj_edge, grasp_edge, edge_attr_og))
                    num_grasps = int(max(h_grasp.shape[0], int(grasp_idx.max().detach().cpu().item()) + 1))
                    src_gc, dst_gc, attr_gc = self._grasp_conditioned_oo_edges(
                        batch,
                        edge_index_og,
                        num_objects=h_obj.shape[0],
                        num_grasps=num_grasps,
                    )
                    context_mask = None
                    if self.gc_object_require_edges:
                        context_mask = h_gc.new_zeros((h_gc.shape[0], 1))
                        if dst_gc.numel() > 0:
                            context_ids = torch.unique(torch.cat([src_gc, dst_gc], dim=0))
                            context_mask[context_ids.long()] = 1.0
                    for gc_layer, context_mlp, context_norm in zip(
                        self.gc_object_layers,
                        self.gc_context_mlps,
                        self.gc_context_norms,
                    ):
                        h_gc = gc_layer(h_gc, src_gc, dst_gc, attr_gc)
                        gc_grasp_context = self._group_mean(h_gc, grasp_idx, num_grasps)
                        h_gc = context_norm(h_gc + context_mlp(torch.cat([h_gc, gc_grasp_context[grasp_idx]], dim=-1)))
                    if context_mask is not None:
                        h_gc = h_gc * context_mask
                    gc_grasp_context = self._group_mean(h_gc, grasp_idx, num_grasps)
                    gc_object_parts = [h_gc, gc_grasp_context[grasp_idx]]
                    if self.gc_object_interactions:
                        gc_object_parts.extend(
                            [
                                h_gc * gc_grasp_context[grasp_idx],
                                torch.abs(h_gc - gc_grasp_context[grasp_idx]),
                            ]
                        )
                    if self.gc_object_logit_residual and self.gc_logit_decoder is not None:
                        gc_logits = self.gc_logit_decoder(torch.cat(gc_object_parts, dim=-1))
                if self.blocker_relation_layers_count > 0 and self.blocker_relation_encoder is not None:
                    h_blocker = self.blocker_relation_encoder(torch.cat([obj_edge, grasp_edge, edge_attr_og], dim=-1))
                    num_grasps = int(max(h_grasp.shape[0], int(grasp_idx.max().detach().cpu().item()) + 1))
                    src_blocker, dst_blocker, attr_blocker = self._grasp_conditioned_oo_edges(
                        batch,
                        edge_index_og,
                        num_objects=h_obj.shape[0],
                        num_grasps=num_grasps,
                    )
                    for blocker_layer, context_mlp, context_norm in zip(
                        self.blocker_relation_layers,
                        self.blocker_relation_context_mlps,
                        self.blocker_relation_context_norms,
                    ):
                        h_blocker = blocker_layer(
                            h_blocker,
                            src_blocker,
                            dst_blocker,
                            attr_blocker,
                            edge_attr_og,
                        )
                        blocker_grasp_context = self._group_mean(h_blocker, grasp_idx, num_grasps)
                        h_blocker = context_norm(
                            h_blocker + context_mlp(torch.cat([h_blocker, blocker_grasp_context[grasp_idx]], dim=-1))
                        )
                    blocker_grasp_context = self._group_mean(h_blocker, grasp_idx, num_grasps)
                    blocker_relation_parts = [h_blocker, blocker_grasp_context[grasp_idx]]
                    if self.blocker_relation_interactions:
                        blocker_relation_parts.extend(
                            [
                                h_blocker * blocker_grasp_context[grasp_idx],
                                torch.abs(h_blocker - blocker_grasp_context[grasp_idx]),
                            ]
                        )
                    if self.blocker_relation_logit_decoder is not None and self.blocker_relation_logit_scale is not None:
                        blocker_relation_logits = self.blocker_relation_logit_decoder(
                            torch.cat(blocker_relation_parts, dim=-1)
                        )
                if self.decoder_interactions:
                    edge_parts = [
                        obj_edge,
                        grasp_edge,
                        obj_edge * grasp_edge,
                        torch.abs(obj_edge - grasp_edge),
                    ]
                    if self.decoder_use_edge_attr:
                        edge_parts.append(edge_attr_og)
                else:
                    edge_parts = [obj_edge, grasp_edge]
                    if self.decoder_use_edge_attr:
                        edge_parts.append(edge_attr_og)
                if self.decoder_raw_features:
                    edge_parts.extend([x_obj[obj_idx], x_grasp[grasp_idx]])
                edge_parts.extend(edge_context_parts)
                if self.gc_object_decoder_context:
                    edge_parts.extend(gc_object_parts)
                if self.blocker_relation_decoder_context:
                    edge_parts.extend(blocker_relation_parts)
                edge_repr = torch.cat(edge_parts, dim=-1)
                edge_logits = self.decoder(edge_repr)
                if edge_context_logits is not None and self.edge_context_logit_scale is not None:
                    edge_logits = edge_logits + self.edge_context_logit_scale.to(
                        dtype=edge_logits.dtype, device=edge_logits.device
                    ) * edge_context_logits
                if gc_logits is not None and self.gc_logit_scale is not None:
                    edge_logits = edge_logits + self.gc_logit_scale.to(dtype=edge_logits.dtype, device=edge_logits.device) * gc_logits
                if blocker_relation_logits is not None and self.blocker_relation_logit_scale is not None:
                    edge_logits = edge_logits + self.blocker_relation_logit_scale.to(
                        dtype=edge_logits.dtype, device=edge_logits.device
                    ) * blocker_relation_logits
            else:
                edge_logits = x_grasp.new_zeros((edge_index_og.shape[1], self.num_edge_labels))

            return {"edge_logits": edge_logits}


    class EdgeMLPDependencyBaseline(nn.Module):
        """Trainable edge-only baseline without object-object or grasp-grasp context."""

        def __init__(
            self,
            *,
            object_in_dim: int,
            grasp_in_dim: int,
            og_edge_dim: int,
            num_edge_labels: int = len(LABEL_NAMES),
            hidden_dim: int = 128,
            decoder_layers: int = 3,
            dropout: float = 0.10,
        ) -> None:
            super().__init__()
            self.num_edge_labels = int(num_edge_labels)
            self.decoder = make_mlp(
                object_in_dim + grasp_in_dim + og_edge_dim,
                hidden_dim,
                self.num_edge_labels,
                num_layers=decoder_layers,
                dropout=dropout,
            )

        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            edge_index = batch["edge_index_og"]
            x_grasp = batch["x_grasp"]
            if edge_index.numel() == 0 or batch["x_obj"].shape[0] == 0 or x_grasp.shape[0] == 0:
                return {"edge_logits": x_grasp.new_zeros((edge_index.shape[1], self.num_edge_labels))}
            obj_idx = edge_index[0].long()
            grasp_idx = edge_index[1].long()
            edge_input = torch.cat(
                [batch["x_obj"][obj_idx], x_grasp[grasp_idx], batch["edge_attr_og"]],
                dim=-1,
            )
            return {"edge_logits": self.decoder(edge_input)}


    class ObjectOnlyDependencyBaseline(nn.Module):
        """No-grasp-conditioning baseline.

        This baseline predicts one dependency vector from the object node alone
        and copies it to every candidate grasp for that object. It intentionally
        ignores grasp features and object-grasp geometry.
        """

        def __init__(
            self,
            *,
            object_in_dim: int,
            num_edge_labels: int = len(LABEL_NAMES),
            hidden_dim: int = 128,
            decoder_layers: int = 3,
            dropout: float = 0.10,
        ) -> None:
            super().__init__()
            self.num_edge_labels = int(num_edge_labels)
            self.decoder = make_mlp(
                object_in_dim,
                hidden_dim,
                self.num_edge_labels,
                num_layers=decoder_layers,
                dropout=dropout,
            )

        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            edge_index = batch["edge_index_og"]
            x_obj = batch["x_obj"]
            if edge_index.numel() == 0 or x_obj.shape[0] == 0:
                return {"edge_logits": x_obj.new_zeros((edge_index.shape[1], self.num_edge_labels))}
            obj_idx = edge_index[0].long()
            return {"edge_logits": self.decoder(x_obj[obj_idx])}


    class G2N2StyleDependencyBaseline(nn.Module):
        """Published-method adaptation of G2N2 for typed dependency prediction.

        Lou et al.'s G2N2 builds one grasp-conditioned object relation graph per
        candidate grasp and predicts grasp success. Our dataset labels are
        object-to-grasp dependencies, so this baseline keeps the grasp-conditioned
        object graph idea but decodes one dependency logit triplet per object.
        It deliberately does not use grasp-grasp context or the explicit
        heterogeneous object-grasp cross message passing of the main model.
        """

        def __init__(
            self,
            *,
            object_in_dim: int,
            grasp_in_dim: int,
            og_edge_dim: int,
            num_edge_labels: int = len(LABEL_NAMES),
            hidden_dim: int = 128,
            message_layers: int = 3,
            decoder_layers: int = 2,
            dropout: float = 0.10,
            use_og_edge_features: bool = True,
        ) -> None:
            super().__init__()
            self.num_edge_labels = int(num_edge_labels)
            self.use_og_edge_features = bool(use_og_edge_features)
            encoder_in_dim = object_in_dim + grasp_in_dim
            if self.use_og_edge_features:
                encoder_in_dim += og_edge_dim
            self.encoder = make_mlp(
                encoder_in_dim,
                hidden_dim,
                hidden_dim,
                num_layers=2,
                dropout=dropout,
                final_activation=True,
            )
            self.message_layers = nn.ModuleList(
                [
                    make_mlp(
                        2 * hidden_dim,
                        hidden_dim,
                        hidden_dim,
                        num_layers=2,
                        dropout=dropout,
                    )
                    for _ in range(message_layers)
                ]
            )
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(message_layers)])
            self.decoder = make_mlp(
                2 * hidden_dim,
                hidden_dim,
                self.num_edge_labels,
                num_layers=decoder_layers,
                dropout=dropout,
            )

        def _group_mean(self, h: torch.Tensor, group: torch.Tensor, num_groups: int) -> torch.Tensor:
            context = h.new_zeros((num_groups, h.shape[1]))
            counts = h.new_zeros((num_groups, 1))
            context.index_add_(0, group, h)
            counts.index_add_(0, group, torch.ones(h.shape[0], 1, dtype=h.dtype, device=h.device))
            return context / counts.clamp_min(1.0)

        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            edge_index = batch["edge_index_og"]
            x_obj = batch["x_obj"]
            x_grasp = batch["x_grasp"]
            if edge_index.numel() == 0 or x_obj.shape[0] == 0 or x_grasp.shape[0] == 0:
                return {"edge_logits": x_grasp.new_zeros((edge_index.shape[1], self.num_edge_labels))}

            obj_idx = edge_index[0].long()
            grasp_idx = edge_index[1].long()
            edge_inputs = [x_obj[obj_idx], x_grasp[grasp_idx]]
            if self.use_og_edge_features:
                edge_inputs.append(batch["edge_attr_og"])
            h = self.encoder(torch.cat(edge_inputs, dim=-1))
            num_grasps = int(max(x_grasp.shape[0], int(grasp_idx.max().detach().cpu().item()) + 1))
            grasp_graph_context = h.new_zeros((num_grasps, h.shape[1]))

            for layer, norm in zip(self.message_layers, self.norms):
                grasp_graph_context = self._group_mean(h, grasp_idx, num_grasps)
                msg = layer(torch.cat([h, grasp_graph_context[grasp_idx]], dim=-1))
                h = norm(h + msg)

            grasp_graph_context = self._group_mean(h, grasp_idx, num_grasps)
            logits = self.decoder(torch.cat([h, grasp_graph_context[grasp_idx]], dim=-1))
            return {"edge_logits": logits}


    def extract_edge_logits(outputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(outputs, dict):
            return outputs["edge_logits"]
        return outputs


    def multilabel_dependency_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        *,
        label_weights: torch.Tensor | None = None,
        pos_weight: torch.Tensor | None = None,
        focal_gamma: float = 0.0,
        consistency_weight: float = 0.0,
    ) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.sum()
        num_labels = int(logits.shape[1])
        mask2 = mask.float().view(-1, 1)
        if label_weights is None:
            label_weights = torch.ones(num_labels, dtype=logits.dtype, device=logits.device)
        else:
            label_weights = label_weights.to(dtype=logits.dtype, device=logits.device)
            if label_weights.numel() != num_labels:
                if label_weights.numel() > num_labels:
                    label_weights = label_weights[:num_labels]
                else:
                    pad = torch.ones(num_labels - label_weights.numel(), dtype=logits.dtype, device=logits.device)
                    label_weights = torch.cat([label_weights, pad], dim=0)
        if pos_weight is not None:
            pos_weight = pos_weight.to(dtype=logits.dtype, device=logits.device)
            if pos_weight.numel() != num_labels:
                if pos_weight.numel() > num_labels:
                    pos_weight = pos_weight[:num_labels]
                else:
                    pad = torch.ones(num_labels - pos_weight.numel(), dtype=logits.dtype, device=logits.device)
                    pos_weight = torch.cat([pos_weight, pad], dim=0)

        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
        if focal_gamma > 0:
            probs = torch.sigmoid(logits)
            pt = probs * targets + (1.0 - probs) * (1.0 - targets)
            bce = bce * (1.0 - pt).clamp_min(1e-6).pow(focal_gamma)
        weighted = bce * label_weights.view(1, num_labels) * mask2
        denom = mask2.sum().clamp_min(1.0) * float(num_labels)
        loss = weighted.sum() / denom
        if consistency_weight > 0:
            probs = torch.sigmoid(logits)
            if num_labels >= 4:
                desired_progress = torch.maximum(probs[:, 2], probs[:, 3]).detach()
            elif num_labels >= 3:
                desired_progress = torch.maximum(probs[:, 1], probs[:, 2]).detach()
            else:
                desired_progress = None
            if desired_progress is not None:
                consistency = F.mse_loss(probs[:, 0], desired_progress, reduction="none") * mask.float()
                loss = loss + consistency_weight * consistency.sum() / mask.float().sum().clamp_min(1.0)
        return loss


else:

    class HeteroDependencyGNN:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class EdgeMLPDependencyBaseline:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class ObjectOnlyDependencyBaseline:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class G2N2StyleDependencyBaseline:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    def multilabel_dependency_loss(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        require_torch()


    def extract_edge_logits(outputs: Any) -> Any:  # type: ignore[no-redef]
        return outputs


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32)


def geometry_heuristic_scores_from_batch(
    batch: dict[str, Any],
    *,
    mode: str = "geometry",
    distance_scale: float = 0.020,
    overlap_weight: float = 1.40,
    swept_iou_weight: float = 1.20,
    bias: float = -1.00,
) -> np.ndarray:
    """Observable geometry baseline for object-to-grasp dependency scores.

    This baseline intentionally has no learning and no graph context. It uses
    only object-grasp edge features that are already derived from bbox/depth and
    grasp proposal geometry.
    """

    edge_attr = np.asarray(batch["edge_attr_og"], dtype=np.float32)
    if edge_attr.shape[0] == 0:
        return np.zeros((0, len(LABEL_NAMES)), dtype=np.float32)

    names = {name: idx for idx, name in enumerate(OG_EDGE_FEATURE_NAMES)}

    def col(name: str, default: float = 0.0) -> np.ndarray:
        idx = names.get(name)
        if idx is None or idx >= edge_attr.shape[1]:
            return np.full(edge_attr.shape[0], default, dtype=np.float32)
        return edge_attr[:, idx].astype(np.float32)

    radius = np.maximum(col("object_radius_est", 0.015), 1e-4)
    center_dist = col("center_to_grasp_dist", 1.0)
    app_dist = col("dist_to_approach_segment", 1.0)
    lift_dist = col("dist_to_lift_segment", 1.0)
    app_overlap = np.maximum(col("approach_overlap_soft"), 0.0)
    lift_overlap = np.maximum(col("lift_overlap_soft"), 0.0)
    app_iou = np.maximum(col("bbox_approach_swept_iou"), 0.0)
    lift_iou = np.maximum(col("bbox_lift_swept_iou"), 0.0)
    rel_depth = col("relative_depth", 0.0)

    scale = max(float(distance_scale), 1e-6)
    mode = str(mode).strip().lower()
    if mode in {"center_distance", "nearest", "nearest_center"}:
        p_near = sigmoid_np(bias - center_dist / scale)
        p_app = p_near
        p_lift = p_near
    elif mode in {"motion_distance", "distance"}:
        p_app = sigmoid_np(bias + (radius - app_dist) / scale)
        p_lift = sigmoid_np(bias + (radius - lift_dist) / scale)
    else:
        app_logit = (
            bias
            + (radius - app_dist) / scale
            + overlap_weight * app_overlap
            + swept_iou_weight * app_iou
            - 0.25 * np.maximum(rel_depth, 0.0)
        )
        lift_logit = (
            bias
            + (radius - lift_dist) / scale
            + overlap_weight * lift_overlap
            + swept_iou_weight * lift_iou
            + 0.15 * np.maximum(rel_depth, 0.0)
        )
        p_app = sigmoid_np(app_logit)
        p_lift = sigmoid_np(lift_logit)
    p_progress_any = np.maximum(p_app, p_lift)
    # A non-learning geometry heuristic cannot tell whether one object is
    # sufficient to restore feasibility, so use progress-any as the calibrated
    # proxy for that auxiliary target.
    p_sufficient = p_progress_any
    return np.stack([p_progress_any, p_sufficient, p_app, p_lift], axis=1).astype(np.float32)


def edge_design_matrix(batch: dict[str, Any]) -> np.ndarray:
    edge_index = batch["edge_index_og"]
    if edge_index.size == 0:
        dim = batch["x_obj"].shape[1] + batch["x_grasp"].shape[1] + batch["edge_attr_og"].shape[1]
        return np.zeros((0, dim), dtype=np.float32)
    obj_idx = edge_index[0]
    grasp_idx = edge_index[1]
    return np.concatenate(
        [
            batch["x_obj"][obj_idx],
            batch["x_grasp"][grasp_idx],
            batch["edge_attr_og"],
        ],
        axis=1,
    ).astype(np.float32)


@dataclass
class NumpyEdgeLogisticBaseline:
    """Minimal trainable edge classifier for environments without PyTorch."""

    input_dim: int
    output_dim: int = len(LABEL_NAMES)
    seed: int = 7
    weights: np.ndarray | None = None
    bias: np.ndarray | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        if self.weights is not None:
            self.output_dim = int(self.weights.shape[1])
        elif self.bias is not None:
            self.output_dim = int(self.bias.shape[0])
        if self.weights is None:
            self.weights = rng.normal(0.0, 0.01, size=(self.input_dim, self.output_dim)).astype(np.float32)
        if self.bias is None:
            self.bias = np.zeros(self.output_dim, dtype=np.float32)

    def fit_standardizer(self, matrices: list[np.ndarray]) -> None:
        x = np.concatenate([m for m in matrices if m.size], axis=0) if any(m.size for m in matrices) else np.zeros((0, self.input_dim), dtype=np.float32)
        if x.size == 0:
            self.mean = np.zeros(self.input_dim, dtype=np.float32)
            self.std = np.ones(self.input_dim, dtype=np.float32)
            return
        self.mean = x.mean(axis=0).astype(np.float32)
        self.std = x.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            self.fit_standardizer([x])
        assert self.mean is not None and self.std is not None
        return ((x - self.mean) / self.std).astype(np.float32)

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        assert self.weights is not None and self.bias is not None
        if x.ndim == 2 and x.shape[1] != self.input_dim:
            if x.shape[1] > self.input_dim:
                x = x[:, : self.input_dim]
            else:
                pad = np.zeros((x.shape[0], self.input_dim - x.shape[1]), dtype=x.dtype)
                x = np.concatenate([x, pad], axis=1)
        x_norm = self.transform(x)
        return x_norm @ self.weights + self.bias

    def predict_logits_from_batch(self, batch: dict[str, Any]) -> np.ndarray:
        return self.predict_logits(edge_design_matrix(batch))

    def partial_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        *,
        lr: float,
        l2: float = 0.0,
        pos_weight: np.ndarray | None = None,
        label_weights: np.ndarray | None = None,
    ) -> float:
        if x.size == 0:
            return 0.0
        assert self.weights is not None and self.bias is not None
        x_norm = self.transform(x)
        logits = x_norm @ self.weights + self.bias
        probs = sigmoid_np(logits)
        mask2 = mask.reshape(-1, 1).astype(np.float32)
        out_dim = int(y.shape[1])
        if self.weights.shape[1] != out_dim:
            raise ValueError(f"Model output_dim={self.weights.shape[1]} does not match labels with {out_dim} columns.")
        pw = np.ones(out_dim, dtype=np.float32) if pos_weight is None else pos_weight.astype(np.float32)
        lw = np.ones(out_dim, dtype=np.float32) if label_weights is None else label_weights.astype(np.float32)
        if pw.shape[0] != out_dim:
            pw = pw[:out_dim] if pw.shape[0] > out_dim else np.pad(pw, (0, out_dim - pw.shape[0]), constant_values=1.0)
        if lw.shape[0] != out_dim:
            lw = lw[:out_dim] if lw.shape[0] > out_dim else np.pad(lw, (0, out_dim - lw.shape[0]), constant_values=1.0)
        weight = (y * pw.reshape(1, out_dim) + (1.0 - y)) * lw.reshape(1, out_dim) * mask2
        grad_logits = (probs - y) * weight
        denom = max(float(mask2.sum() * float(out_dim)), 1.0)
        grad_w = (x_norm.T @ grad_logits) / denom + l2 * self.weights
        grad_b = grad_logits.sum(axis=0) / denom
        self.weights -= lr * grad_w.astype(np.float32)
        self.bias -= lr * grad_b.astype(np.float32)

        bce = -(y * np.log(probs + 1e-8) + (1.0 - y) * np.log(1.0 - probs + 1e-8))
        return float((bce * weight).sum() / denom)

    def save(self, path: Path) -> None:
        assert self.weights is not None and self.bias is not None
        assert self.mean is not None and self.std is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            input_dim=np.asarray([self.input_dim], dtype=np.int64),
            output_dim=np.asarray([self.output_dim], dtype=np.int64),
            weights=self.weights,
            bias=self.bias,
            mean=self.mean,
            std=self.std,
        )

    @classmethod
    def load(cls, path: Path) -> "NumpyEdgeLogisticBaseline":
        payload = np.load(path)
        weights = payload["weights"].astype(np.float32)
        output_dim = int(payload["output_dim"][0]) if "output_dim" in payload else int(weights.shape[1])
        model = cls(
            input_dim=int(payload["input_dim"][0]),
            output_dim=output_dim,
            weights=weights,
            bias=payload["bias"].astype(np.float32),
        )
        model.mean = payload["mean"].astype(np.float32)
        model.std = payload["std"].astype(np.float32)
        return model
