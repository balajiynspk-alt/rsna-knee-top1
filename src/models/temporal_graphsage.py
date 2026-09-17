#!/usr/bin/env python3
"""
TGCF-IDS: Edge-Aware Temporal GraphSAGE Encoder.

Implements an edge-aware inductive graph neural network encoder that aggregates
both temporal neighborhood information and continuous/discrete edge attributes:
    Inputs:
        - Node representations: x in R^{N x D_in}
        - Graph topology:       edge_index in Z^{2 x E}
        - Edge attributes:      edge_attr in R^{E x D_edge}
    Output:
        - Contextualized node/flow graph embeddings: h in R^{N x D_out}

Architecture per layer:
    m_{j -> i} = MLP([h_j || e_{j -> i}])
    h_i^{(l+1)} = W_{root} h_i^{(l)} + Aggr_{j in N(i)}(m_{j -> i})
    h_i^{(l+1)} = LayerNorm(h_i^{(l+1)}) -> GELU -> Dropout -> Residual Connection
"""

from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


class EdgeAwareSAGEConv(MessagePassing):
    """
    Edge-Conditioned GraphSAGE Convolution Layer.

    Aggregates messages combining neighbor node embeddings and rich multi-dimensional
    edge attributes (temporal distance, host proximity, protocol similarity, port relations, feature cosine similarity).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int = 6,
        aggr: str = "mean",
        bias: bool = True,
    ):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim

        # Message transform MLP: projects [h_j || e_{j -> i}] -> out_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_dim, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

        # Root node self-transformation
        self.lin_root = nn.Linear(in_channels, out_channels, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize projection weights."""
        for m in self.edge_mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.kaiming_uniform_(self.lin_root.weight, a=1.0)
        if self.lin_root.bias is not None:
            nn.init.zeros_(self.lin_root.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Node feature matrix of shape [N, in_channels].
            edge_index: Graph connectivity tensor of shape [2, E].
            edge_attr: Edge feature matrix of shape [E, edge_dim].

        Returns:
            out: Updated node embeddings of shape [N, out_channels].
        """
        # Propagate messages across graph edges
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Construct edge-conditioned message from neighbor node x_j and edge_attr.
        """
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.edge_mlp(msg_input)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Combine aggregated neighborhood messages with root node representation.
        """
        return self.lin_root(x) + aggr_out


class TemporalGraphSAGE(nn.Module):
    """
    Edge-Aware Temporal GraphSAGE Network.

    Stacks multiple EdgeAwareSAGEConv layers with Pre-LayerNorm, GELU activations,
    Dropout, and Residual connections.

    Args:
        in_channels: Input node representation dimension (e.g., from FeatureTransformer flow_repr or dense features).
        edge_dim: Number of edge attributes (default 6).
        hidden_dim: Hidden dimension across GraphSAGE message-passing layers.
        out_channels: Output embedding dimension (defaults to hidden_dim).
        num_layers: Number of stacked GraphSAGE layers.
        dropout: Dropout probability.
        aggr: Aggregation operator ('mean', 'max', 'sum', 'add').
        use_residual: Whether to add residual connections between layers.
        use_layer_norm: Whether to apply LayerNorm after message passing.
    """

    def __init__(
        self,
        in_channels: int = 64,
        edge_dim: int = 6,
        hidden_dim: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        aggr: str = "mean",
        use_residual: bool = True,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels or hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.aggr = aggr
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")

        # Input feature projection if dimension mismatch
        if self.in_channels != self.hidden_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(self.in_channels, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
            )
        else:
            self.input_proj = nn.Identity()

        # Stacked Edge-Aware SAGE Convolution Layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            self.convs.append(
                EdgeAwareSAGEConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    edge_dim=self.edge_dim,
                    aggr=self.aggr,
                )
            )
            if self.use_layer_norm:
                self.norms.append(nn.LayerNorm(self.hidden_dim))
            else:
                self.norms.append(nn.Identity())

        self.dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

        # Output projection if out_channels != hidden_dim
        if self.out_channels != self.hidden_dim:
            self.output_proj = nn.Linear(self.hidden_dim, self.out_channels)
        else:
            self.output_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for node embedding generation.

        Args:
            x: Node features [N, in_channels].
            edge_index: Graph edge indices [2, E].
            edge_attr: Edge attribute matrix [E, edge_dim].
            batch: Optional batch assignment vector [N] for batched graph processing.

        Returns:
            h: Node embeddings of shape [N, out_channels].
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D node features [N, in_channels], got shape {x.shape}")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"Expected edge_index of shape [2, E], got shape {edge_index.shape}")
        if edge_attr.dim() != 2 or edge_attr.size(1) != self.edge_dim:
            raise ValueError(
                f"Expected edge_attr of shape [E, {self.edge_dim}], got shape {edge_attr.shape}"
            )

        # 1. Project input features
        h = self.input_proj(x)

        # 2. Sequential Message Passing across layers
        for layer_idx in range(self.num_layers):
            h_in = h
            # Message Passing
            h_conv = self.convs[layer_idx](h, edge_index, edge_attr)
            # LayerNorm
            h_norm = self.norms[layer_idx](h_conv)
            # Activation & Dropout
            h_act = self.dropout(F.gelu(h_norm))

            # Residual connection
            if self.use_residual:
                h = h_in + h_act
            else:
                h = h_act

        # 3. Final Output Projection
        h_out = self.output_proj(h)
        return h_out

    def forward_data(
        self,
        data: Data,
        node_feature_key: str = "x",
    ) -> torch.Tensor:
        """
        Convenience forward wrapper directly taking a PyG Data object.
        """
        if hasattr(data, node_feature_key):
            x = getattr(data, node_feature_key)
        elif hasattr(data, "x_dense"):
            x = data.x_dense
        elif hasattr(data, "x_num"):
            x = data.x_num
        else:
            raise AttributeError(f"Could not find valid node features in Data object.")

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = getattr(data, "batch", None)

        return self.forward(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

    @staticmethod
    def sample_neighborhood(
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        seed_nodes: torch.Tensor,
        num_neighbors: int = 10,
        num_hops: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pure PyTorch inductive neighborhood sampler (no C++ compiled binaries required).

        Args:
            edge_index: [2, E]
            edge_attr: [E, D_edge]
            seed_nodes: 1D tensor of target node indices [B]
            num_neighbors: Maximum neighbors to sample per node per hop
            num_hops: Number of neighborhood expansion hops

        Returns:
            sub_nodes: All sampled node indices [N_sub]
            sub_edge_index: Re-indexed edge index [2, E_sub]
            sub_edge_attr: Sliced edge attributes [E_sub, D_edge]
            seed_mask: Boolean mask over sub_nodes indicating original seed nodes
        """
        device = edge_index.device
        current_nodes = set(seed_nodes.cpu().numpy().tolist())
        src_np = edge_index[0].cpu().numpy()
        dst_np = edge_index[1].cpu().numpy()

        for _ in range(num_hops):
            next_hop_nodes = set()
            for node in current_nodes:
                matching_edges = np.where(src_np == node)[0]
                if len(matching_edges) > 0:
                    if len(matching_edges) > num_neighbors:
                        chosen_edges = np.random.choice(matching_edges, size=num_neighbors, replace=False)
                    else:
                        chosen_edges = matching_edges
                    next_hop_nodes.update(dst_np[chosen_edges].tolist())
            current_nodes.update(next_hop_nodes)

        sub_nodes_list = sorted(list(current_nodes))
        node_to_idx = {n: i for i, n in enumerate(sub_nodes_list)}
        sub_nodes_tensor = torch.tensor(sub_nodes_list, dtype=torch.long, device=device)

        # Filter edges where both src and dst are in sub_nodes
        sub_nodes_set = set(sub_nodes_list)
        mask = np.array([u in sub_nodes_set and v in sub_nodes_set for u, v in zip(src_np, dst_np)])
        selected_edge_indices = np.where(mask)[0]

        new_src = [node_to_idx[src_np[idx]] for idx in selected_edge_indices]
        new_dst = [node_to_idx[dst_np[idx]] for idx in selected_edge_indices]

        sub_edge_index = torch.tensor([new_src, new_dst], dtype=torch.long, device=device)
        sub_edge_attr = edge_attr[selected_edge_indices]

        seed_set = set(seed_nodes.cpu().numpy().tolist())
        seed_mask = torch.tensor([n in seed_set for n in sub_nodes_list], dtype=torch.bool, device=device)

        return sub_nodes_tensor, sub_edge_index, sub_edge_attr, seed_mask

    @staticmethod
    def get_graph_dataloader(
        dataset: List[Data],
        batch_size: int = 4,
        shuffle: bool = True,
    ):
        """
        Standard PyG DataLoader for mini-batching graph snapshots.
        """
        from torch_geometric.loader import DataLoader
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"edge_dim={self.edge_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"out_channels={self.out_channels}, "
            f"num_layers={self.num_layers}, "
            f"aggr={self.aggr}, "
            f"use_residual={self.use_residual}"
        )
