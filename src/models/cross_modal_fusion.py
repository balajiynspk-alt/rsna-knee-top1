#!/usr/bin/env python3
"""
TGCF-IDS: Cross-Modal Feature and Graph Fusion Module.

Fuses high-level flow representations from two complementary inductive modalities:
1. Feature Transformer: Intrinsic tabular flow-level representation (h_feat in R^{N x D_feat})
2. Temporal GraphSAGE: Relational temporal-topological graph representation (h_graph in R^{N x D_graph})

Supported Fusion Strategies:
A. Learned Gating (Adaptive Dynamic Fusion)
B. Linear Concatenation
C. Element-Wise Addition

Returns:
- Fused representation:     h_fused in R^{N x D_fused}
- Feature contribution:     c_feat in R^{N x D_fused}
- Graph contribution:       c_graph in R^{N x D_fused}
- Dynamic gate values:      g in [0, 1]^{N x D_fused} or [0, 1]^{N x 1}
"""

from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionOutput(NamedTuple):
    """Container for CrossModalFusion output components."""
    fused_repr: torch.Tensor          # Fused multi-modal embedding: [N, D_fused]
    feature_contrib: torch.Tensor     # Feature modality contribution: [N, D_fused]
    graph_contrib: torch.Tensor       # Graph modality contribution: [N, D_fused]
    gate_values: torch.Tensor         # Gating weights in [0, 1]: [N, D_fused] or [N, 1]
    strategy: str                     # Fusion strategy used ('gated', 'concat', 'add')


class GatedFusion(nn.Module):
    """
    Adaptive Learned Gated Fusion with Projection, LayerNorm, and Residual connection.

    Mechanism:
        u_feat  = GELU(LayerNorm(Linear(h_feat)))
        u_graph = GELU(LayerNorm(Linear(h_graph)))
        g       = Sigmoid(Linear([u_feat || u_graph]))
        h_gated = g * u_feat + (1 - g) * u_graph
        h_fused = LayerNorm(h_gated + Dropout(Linear(h_gated)))
    """

    def __init__(
        self,
        feat_dim: int = 64,
        graph_dim: int = 64,
        fused_dim: int = 64,
        dropout: float = 0.1,
        gate_type: str = "vector",
        use_residual: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.graph_dim = graph_dim
        self.fused_dim = fused_dim
        self.dropout_rate = dropout
        self.gate_type = gate_type.lower()
        self.use_residual = use_residual

        if self.gate_type not in ("vector", "scalar"):
            raise ValueError(f"gate_type must be 'vector' or 'scalar', got '{gate_type}'")

        # 1. Modality Projection Layers
        self.proj_feat = nn.Linear(feat_dim, fused_dim)
        self.norm_feat = nn.LayerNorm(fused_dim)

        self.proj_graph = nn.Linear(graph_dim, fused_dim)
        self.norm_graph = nn.LayerNorm(fused_dim)

        # 2. Dynamic Gating Network
        gate_out_dim = fused_dim if self.gate_type == "vector" else 1
        self.gate_net = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, gate_out_dim),
            nn.Sigmoid(),
        )

        # 3. Residual & Post-LayerNorm
        self.post_linear = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(fused_dim)

    def forward(
        self,
        h_feat: torch.Tensor,
        h_graph: torch.Tensor,
    ) -> FusionOutput:
        # Project each modality
        u_feat = F.gelu(self.norm_feat(self.proj_feat(h_feat)))    # [N, D_fused]
        u_graph = F.gelu(self.norm_graph(self.proj_graph(h_graph))) # [N, D_fused]

        # Compute dynamic gate g in [0, 1]
        concat_u = torch.cat([u_feat, u_graph], dim=-1)           # [N, 2 * D_fused]
        g = self.gate_net(concat_u)                               # [N, D_fused] or [N, 1]

        # Modality contributions
        feature_contrib = g * u_feat                              # [N, D_fused]
        graph_contrib = (1.0 - g) * u_graph                       # [N, D_fused]

        # Gated combination
        h_comb = feature_contrib + graph_contrib                  # [N, D_fused]

        # Residual connection
        if self.use_residual:
            h_fused = self.final_norm(h_comb + self.post_linear(h_comb))
        else:
            h_fused = self.final_norm(self.post_linear(h_comb))

        return FusionOutput(
            fused_repr=h_fused,
            feature_contrib=feature_contrib,
            graph_contrib=graph_contrib,
            gate_values=g,
            strategy="gated",
        )


class ConcatFusion(nn.Module):
    """
    Linear Concatenation Fusion baseline.

    Mechanism:
        h_concat = [h_feat || h_graph]
        h_fused  = LayerNorm(GELU(Linear(h_concat, D_fused)))
    """

    def __init__(
        self,
        feat_dim: int = 64,
        graph_dim: int = 64,
        fused_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.graph_dim = graph_dim
        self.fused_dim = fused_dim

        self.lin_feat = nn.Linear(feat_dim, fused_dim)
        self.lin_graph = nn.Linear(graph_dim, fused_dim)

        self.fusion_linear = nn.Sequential(
            nn.Linear(feat_dim + graph_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(fused_dim)

    def forward(
        self,
        h_feat: torch.Tensor,
        h_graph: torch.Tensor,
    ) -> FusionOutput:
        concat_h = torch.cat([h_feat, h_graph], dim=-1)
        h_fused = self.final_norm(self.fusion_linear(concat_h))

        # Nominal balanced contributions
        u_feat = self.lin_feat(h_feat)
        u_graph = self.lin_graph(h_graph)
        g_nominal = torch.full((h_feat.size(0), 1), 0.5, device=h_feat.device)

        return FusionOutput(
            fused_repr=h_fused,
            feature_contrib=u_feat,
            graph_contrib=u_graph,
            gate_values=g_nominal,
            strategy="concat",
        )


class AdditionFusion(nn.Module):
    """
    Element-Wise Addition Fusion baseline.

    Mechanism:
        u_feat  = Linear(h_feat, D_fused)
        u_graph = Linear(h_graph, D_fused)
        h_fused = LayerNorm(GELU(u_feat + u_graph))
    """

    def __init__(
        self,
        feat_dim: int = 64,
        graph_dim: int = 64,
        fused_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.graph_dim = graph_dim
        self.fused_dim = fused_dim

        self.proj_feat = nn.Linear(feat_dim, fused_dim)
        self.proj_graph = nn.Linear(graph_dim, fused_dim)
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(fused_dim)

    def forward(
        self,
        h_feat: torch.Tensor,
        h_graph: torch.Tensor,
    ) -> FusionOutput:
        u_feat = self.proj_feat(h_feat)
        u_graph = self.proj_graph(h_graph)

        h_sum = u_feat + u_graph
        h_fused = self.final_norm(self.dropout(F.gelu(h_sum)))
        g_nominal = torch.full((h_feat.size(0), 1), 0.5, device=h_feat.device)

        return FusionOutput(
            fused_repr=h_fused,
            feature_contrib=u_feat,
            graph_contrib=u_graph,
            gate_values=g_nominal,
            strategy="add",
        )


class CrossModalFusion(nn.Module):
    """
    Unified Cross-Modal Fusion Engine.

    Supports configurable fusion strategies:
    - 'gated'  : Dynamic learned feature gating (default)
    - 'concat' : Linear projection of concatenated modalities
    - 'add'    : Linear projection + element-wise sum

    Args:
        feat_dim: Dimension of Feature Transformer flow representation.
        graph_dim: Dimension of Temporal GraphSAGE representation.
        fused_dim: Target dimension of fused representation.
        strategy: Fusion mechanism ('gated', 'concat', 'add').
        gate_type: Gating granularity ('vector' for per-channel, 'scalar' for per-sample).
        dropout: Dropout rate.
        use_residual: Whether to use residual connections (for 'gated').
    """

    def __init__(
        self,
        feat_dim: int = 64,
        graph_dim: int = 64,
        fused_dim: int = 64,
        strategy: str = "gated",
        gate_type: str = "vector",
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.graph_dim = graph_dim
        self.fused_dim = fused_dim
        self.strategy = strategy.lower()
        self.gate_type = gate_type
        self.dropout_rate = dropout
        self.use_residual = use_residual

        if self.strategy == "gated":
            self.fusion_module = GatedFusion(
                feat_dim=feat_dim,
                graph_dim=graph_dim,
                fused_dim=fused_dim,
                dropout=dropout,
                gate_type=gate_type,
                use_residual=use_residual,
            )
        elif self.strategy == "concat":
            self.fusion_module = ConcatFusion(
                feat_dim=feat_dim,
                graph_dim=graph_dim,
                fused_dim=fused_dim,
                dropout=dropout,
            )
        elif self.strategy in ("add", "addition"):
            self.fusion_module = AdditionFusion(
                feat_dim=feat_dim,
                graph_dim=graph_dim,
                fused_dim=fused_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"Unknown fusion strategy '{strategy}'. Choose from 'gated', 'concat', 'add'."
            )

    def forward(
        self,
        h_feat: torch.Tensor,
        h_graph: torch.Tensor,
    ) -> FusionOutput:
        """
        Fuse tabular feature and temporal graph representations.

        Args:
            h_feat: Tensor of shape [N, feat_dim]
            h_graph: Tensor of shape [N, graph_dim]

        Returns:
            FusionOutput containing:
                - fused_repr: [N, fused_dim]
                - feature_contrib: [N, fused_dim]
                - graph_contrib: [N, fused_dim]
                - gate_values: [N, fused_dim] or [N, 1]
                - strategy: str
        """
        if h_feat.dim() != 2:
            raise ValueError(f"Expected 2D h_feat [N, feat_dim], got shape {h_feat.shape}")
        if h_graph.dim() != 2:
            raise ValueError(f"Expected 2D h_graph [N, graph_dim], got shape {h_graph.shape}")
        if h_feat.size(0) != h_graph.size(0):
            raise ValueError(
                f"Sample count mismatch: h_feat has {h_feat.size(0)} rows, "
                f"h_graph has {h_graph.size(0)} rows."
            )
        if h_feat.size(1) != self.feat_dim:
            raise ValueError(f"Expected h_feat dim {self.feat_dim}, got {h_feat.size(1)}")
        if h_graph.size(1) != self.graph_dim:
            raise ValueError(f"Expected h_graph dim {self.graph_dim}, got {h_graph.size(1)}")

        return self.fusion_module(h_feat=h_feat, h_graph=h_graph)

    def extra_repr(self) -> str:
        return (
            f"strategy={self.strategy}, "
            f"feat_dim={self.feat_dim}, "
            f"graph_dim={self.graph_dim}, "
            f"fused_dim={self.fused_dim}, "
            f"gate_type={self.gate_type}"
        )
