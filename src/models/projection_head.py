#!/usr/bin/env python3
"""
TGCF-IDS: Non-Linear Contrastive Projection Head.

Maps latent graph/node representations from the encoder into a lower-dimensional
normalized embedding space where contrastive loss (NT-Xent/InfoNCE) is computed:
    h in R^{N x D_in} -> z in R^{N x D_out} with ||z||_2 = 1

Architecture:
    Linear(in_dim, hidden_dim) -> LayerNorm -> GELU -> Dropout -> Linear(hidden_dim, out_dim) -> L2Norm
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveProjectionHead(nn.Module):
    """
    Non-linear projection MLP with LayerNorm, GELU, and L2 unit-sphere normalization.

    Args:
        in_dim: Input representation dimension from TemporalGraphSAGE encoder.
        hidden_dim: Hidden dimension of projection MLP (defaults to in_dim * 2).
        out_dim: Output contrastive space dimension (defaults to in_dim).
        dropout: Dropout probability.
        use_layer_norm: Whether to apply LayerNorm between linear layers.
    """

    def __init__(
        self,
        in_dim: int = 64,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim or (in_dim * 2)
        self.out_dim = out_dim or in_dim
        self.dropout_rate = dropout
        self.use_layer_norm = use_layer_norm

        self.lin1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim) if use_layer_norm else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        self.lin2 = nn.Linear(self.hidden_dim, self.out_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize projection weights."""
        nn.init.kaiming_uniform_(self.lin1.weight, a=1.0)
        nn.init.zeros_(self.lin1.bias)
        nn.init.kaiming_uniform_(self.lin2.weight, a=1.0)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, h: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Forward pass projecting representations into normalized contrastive space.

        Args:
            h: Node or graph representation tensor [N, in_dim] or [B, in_dim].
            normalize: Whether to apply L2 unit-sphere normalization (required for InfoNCE).

        Returns:
            z: Contrastive embedding tensor [N, out_dim].
        """
        x = self.lin1(h)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        z = self.lin2(x)

        if normalize:
            z = F.normalize(z, p=2, dim=-1)

        return z

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, hidden_dim={self.hidden_dim}, out_dim={self.out_dim}"
