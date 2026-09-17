#!/usr/bin/env python3
"""
TGCF-IDS: Graph Augmentations for Self-Supervised Temporal Graph Contrastive Learning.

Generates semantically valid, temporal-preserving augmented views of network-flow graphs:
1. Feature Masking: Randomly zeros out a fraction of node feature dimensions.
2. Numerical Feature Perturbation: Injects small Gaussian jitter N(0, sigma^2) into continuous metrics.
3. Controlled Edge Dropout: Drops a fraction of graph edges while preserving basic connectivity.
4. Temporal-Preserving Edge Attribute Perturbation: Jitters edge attributes within valid domain bounds.

Crucial Rules:
- Never modifies or uses target labels.
- Preserves relative temporal sequence causality.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


class GraphAugmentor:
    """
    Stochastic Graph Augmentor for generating multi-view contrastive pairs.

    Args:
        feature_mask_ratio: Fraction of node feature entries to mask with zero (default: 0.15).
        feature_noise_std: Standard deviation of Gaussian perturbation on continuous features (default: 0.05).
        edge_drop_ratio: Fraction of graph edges to drop (default: 0.15).
        temporal_jitter_std: Noise std applied to edge attributes (default: 0.02).
        random_seed: Optional random seed for reproducible augmentation testing.
    """

    def __init__(
        self,
        feature_mask_ratio: float = 0.15,
        feature_noise_std: float = 0.05,
        edge_drop_ratio: float = 0.15,
        temporal_jitter_std: float = 0.02,
        random_seed: Optional[int] = None,
    ):
        self.mask_ratio = feature_mask_ratio
        self.noise_std = feature_noise_std
        self.drop_ratio = edge_drop_ratio
        self.jitter_std = temporal_jitter_std
        self.seed = random_seed

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

    def mask_features(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly mask feature entries with zeros."""
        if self.mask_ratio <= 0.0:
            return x
        mask = torch.rand_like(x) > self.mask_ratio
        return x * mask.to(x.dtype)

    def perturb_features(self, x: torch.Tensor) -> torch.Tensor:
        """Add small zero-mean Gaussian noise to continuous features."""
        if self.noise_std <= 0.0:
            return x
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def drop_edges(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Controlled edge dropout: randomly removes a fraction of directed edges.
        Ensures at least 1 edge remains if original graph has edges.
        """
        num_edges = edge_index.size(1)
        if self.drop_ratio <= 0.0 or num_edges <= 1:
            return edge_index, edge_attr

        keep_prob = 1.0 - self.drop_ratio
        mask = torch.rand(num_edges, device=edge_index.device) < keep_prob

        # Guarantee at least 1 edge kept
        if not mask.any():
            mask[0] = True

        new_edge_index = edge_index[:, mask]
        new_edge_attr = edge_attr[mask]
        return new_edge_index, new_edge_attr

    def perturb_edge_attributes(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Add bounded noise to continuous edge attributes while clipping to valid domains:
        - attr[0] (temporal dist): in [0, 1]
        - attr[1, 2] (src/dst rel): in [0, 1]
        - attr[3, 4] (proto/port binary matches): unchanged
        - attr[5] (feature similarity): in [-1, 1]
        """
        if self.jitter_std <= 0.0 or edge_attr.size(0) == 0:
            return edge_attr

        perturbed = edge_attr.clone()

        # Jitter continuous components
        noise = torch.randn_like(perturbed) * self.jitter_std
        # Do not alter discrete/categorical match indicators (cols 3 and 4)
        noise[:, 3] = 0.0
        noise[:, 4] = 0.0

        perturbed = perturbed + noise

        # Clamp to valid theoretical ranges
        perturbed[:, 0] = perturbed[:, 0].clamp(min=0.0, max=1.0)
        perturbed[:, 1] = perturbed[:, 1].clamp(min=0.0, max=1.0)
        perturbed[:, 2] = perturbed[:, 2].clamp(min=0.0, max=1.0)
        perturbed[:, 5] = perturbed[:, 5].clamp(min=-1.0, max=1.0)

        return perturbed

    def augment_graph(self, data: Data) -> Data:
        """
        Create a single augmented graph view G' from an input PyG Data object.
        """
        # Determine node features (x, x_dense, or x_num)
        if hasattr(data, "x_dense") and data.x_dense is not None:
            x_raw = data.x_dense.clone()
        elif hasattr(data, "x") and data.x is not None:
            x_raw = data.x.clone()
        else:
            x_raw = data.x_num.clone()

        # 1. Feature Masking & Numerical Perturbation
        x_aug = self.mask_features(x_raw)
        x_aug = self.perturb_features(x_aug)

        # 2. Controlled Edge Dropout
        edge_index_aug, edge_attr_aug = self.drop_edges(
            data.edge_index.clone(),
            data.edge_attr.clone(),
        )

        # 3. Temporal-Preserving Edge Attribute Perturbation
        edge_attr_aug = self.perturb_edge_attributes(edge_attr_aug)

        # Construct new Data view
        aug_data = Data(
            x=x_aug,
            x_dense=x_aug,
            edge_index=edge_index_aug,
            edge_attr=edge_attr_aug,
            num_nodes=data.num_nodes,
        )

        # Preserve metadata and target keys if present (for evaluation only)
        if hasattr(data, "y_multiclass"):
            aug_data.y_multiclass = data.y_multiclass
        if hasattr(data, "y_binary"):
            aug_data.y_binary = data.y_binary
        if hasattr(data, "window_idx"):
            aug_data.window_idx = data.window_idx

        return aug_data

    def create_augmented_pair(self, data: Data) -> Tuple[Data, Data]:
        """
        Generate two independent stochastic augmented views (G1, G2) of a graph snapshot.
        """
        view1 = self.augment_graph(data)
        view2 = self.augment_graph(data)
        return view1, view2
