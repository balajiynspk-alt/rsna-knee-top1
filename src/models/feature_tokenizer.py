#!/usr/bin/env python3
"""
TGCF-IDS: Feature Tokenizer for Tabular Network-Flow Features.

Transforms heterogeneous tabular network-flow features (numerical continuous + categorical)
into a sequence of learnable token representations:
    [Batch Size, Num Features] -> [Batch Size, Num Features, Embedding Dim]

Architecture:
1. Numerical Features: Individual learnable linear projections (weight + bias) per feature.
2. Categorical Features: Learnable embedding tables with dedicated <UNK> token support (index 0).
3. Feature Identity Embeddings: Learnable positional/identity parameter added to each token.
4. Output: Standardized [B, F, D] representation ready for Transformer attention layers.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """
    Feature Tokenizer for Tabular Network-Flow Data.

    Maps a heterogeneous batch of numerical and categorical network-flow features
    into uniform D-dimensional token representations:
        Input : x_num of shape [B, N_num], x_cat of shape [B, N_cat]
        Output: tokens of shape [B, N_num + N_cat, D]

    Attributes:
        num_numerical (int): Number of numerical continuous features.
        num_categorical (int): Number of categorical features.
        num_features (int): Total number of features (N_num + N_cat).
        embedding_dim (int): Dimensionality D of each feature token.
        use_identity_emb (bool): Whether to add learnable feature-identity embeddings.
    """

    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: Union[List[int], Dict[str, int]],
        embedding_dim: int = 64,
        bias: bool = True,
        use_identity_emb: bool = True,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        numerical_feature_names: Optional[List[str]] = None,
        categorical_feature_names: Optional[List[str]] = None,
    ):
        super().__init__()

        if num_numerical < 0:
            raise ValueError(f"num_numerical must be non-negative, got {num_numerical}")

        # Parse categorical cardinalities
        if isinstance(cat_cardinalities, dict):
            self.cat_cardinalities: List[int] = list(cat_cardinalities.values())
            self.categorical_names: List[str] = list(cat_cardinalities.keys())
        elif isinstance(cat_cardinalities, list):
            self.cat_cardinalities = list(cat_cardinalities)
            self.categorical_names = categorical_feature_names or [
                f"cat_{i}" for i in range(len(self.cat_cardinalities))
            ]
        else:
            raise TypeError("cat_cardinalities must be a list of ints or dict of {col_name: cardinality}")

        for card in self.cat_cardinalities:
            if card <= 0:
                raise ValueError(f"Categorical cardinality must be > 0, got {card}")

        self.num_numerical = num_numerical
        self.num_categorical = len(self.cat_cardinalities)
        self.num_features = self.num_numerical + self.num_categorical
        self.embedding_dim = embedding_dim
        self.use_bias = bias
        self.use_identity_emb = use_identity_emb
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout

        self.numerical_names = numerical_feature_names or [
            f"num_{i}" for i in range(self.num_numerical)
        ]

        if self.num_features == 0:
            raise ValueError("FeatureTokenizer requires at least one feature (numerical or categorical).")

        # 1. Numerical Feature Projections: e_{num, i} = x_{num, i} * w_i + b_i
        if self.num_numerical > 0:
            self.weight_num = nn.Parameter(torch.empty(self.num_numerical, self.embedding_dim))
            if self.use_bias:
                self.bias_num = nn.Parameter(torch.empty(self.num_numerical, self.embedding_dim))
            else:
                self.register_parameter("bias_num", None)
        else:
            self.register_parameter("weight_num", None)
            self.register_parameter("bias_num", None)

        # 2. Categorical Feature Embeddings: e_{cat, j} = Embedding_j(x_{cat, j})
        if self.num_categorical > 0:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(cardinality, self.embedding_dim)
                for cardinality in self.cat_cardinalities
            ])
        else:
            self.cat_embeddings = nn.ModuleList()

        # 3. Learnable Feature-Identity Embeddings: e_{identity, k} in R^D
        if self.use_identity_emb:
            self.identity_emb = nn.Parameter(torch.empty(1, self.num_features, self.embedding_dim))
        else:
            self.register_parameter("identity_emb", None)

        # 4. Optional Token Normalization & Dropout
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.embedding_dim)
        else:
            self.layer_norm = nn.Identity()

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize all projection weights, embeddings, and identity parameters."""
        if self.num_numerical > 0 and self.weight_num is not None:
            nn.init.kaiming_uniform_(self.weight_num, a=math.sqrt(5))
            if self.bias_num is not None:
                fan_in = self.num_numerical
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias_num, -bound, bound)

        for emb in self.cat_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=1.0 / math.sqrt(self.embedding_dim))

        if self.use_identity_emb and self.identity_emb is not None:
            nn.init.normal_(self.identity_emb, mean=0.0, std=1.0 / math.sqrt(self.embedding_dim))

    def forward(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
        x_dict: Optional[Dict[str, torch.Tensor]] = None,
        x_combined: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Tokenize numerical and categorical network-flow features.

        Args:
            x_num: Continuous numerical tensor of shape [B, N_num].
            x_cat: Categorical index tensor of shape [B, N_cat] (int64).
            x_dict: Dictionary containing 'x_num' and/or 'x_cat'.
            x_combined: Combined tensor [B, N_num + N_cat] (splits automatically).

        Returns:
            tokens: Token tensor of shape [B, N_num + N_cat, D].
        """
        # Unpack from dictionary if provided
        if x_dict is not None:
            if x_num is None and "x_num" in x_dict:
                x_num = x_dict["x_num"]
            if x_cat is None and "x_cat" in x_dict:
                x_cat = x_dict["x_cat"]

        # Unpack from combined tensor if provided
        if x_combined is not None:
            if x_num is None and self.num_numerical > 0:
                x_num = x_combined[:, : self.num_numerical].float()
            if x_cat is None and self.num_categorical > 0:
                x_cat = x_combined[:, self.num_numerical :].long()

        tokens_list: List[torch.Tensor] = []
        batch_size: Optional[int] = None

        # 1. Process Numerical Features -> [B, N_num, D]
        if self.num_numerical > 0:
            if x_num is None:
                raise ValueError(
                    f"FeatureTokenizer expected x_num with {self.num_numerical} features, but got None."
                )
            if x_num.dim() == 1:
                x_num = x_num.unsqueeze(0)  # Handle single unbatched instance [N_num] -> [1, N_num]
            if x_num.size(1) != self.num_numerical:
                raise ValueError(
                    f"Expected x_num second dimension {self.num_numerical}, got {x_num.size(1)}"
                )

            batch_size = x_num.size(0)
            x_num_float = x_num.float()

            # Broadcast multiply: [B, N_num, 1] * [N_num, D] -> [B, N_num, D]
            num_tokens = x_num_float.unsqueeze(-1) * self.weight_num.unsqueeze(0)
            if self.bias_num is not None:
                num_tokens = num_tokens + self.bias_num.unsqueeze(0)

            tokens_list.append(num_tokens)

        # 2. Process Categorical Features -> [B, N_cat, D]
        if self.num_categorical > 0:
            if x_cat is None:
                raise ValueError(
                    f"FeatureTokenizer expected x_cat with {self.num_categorical} features, but got None."
                )
            if x_cat.dim() == 1:
                x_cat = x_cat.unsqueeze(0)
            if x_cat.size(1) != self.num_categorical:
                raise ValueError(
                    f"Expected x_cat second dimension {self.num_categorical}, got {x_cat.size(1)}"
                )

            if batch_size is None:
                batch_size = x_cat.size(0)

            x_cat_long = x_cat.long()
            cat_tokens_per_feature: List[torch.Tensor] = []

            for col_idx, emb_layer in enumerate(self.cat_embeddings):
                col_indices = x_cat_long[:, col_idx]
                # Map out-of-range negative values or excess values safely to 0 (<UNK>)
                card = self.cat_cardinalities[col_idx]
                safe_indices = torch.where(
                    (col_indices >= 0) & (col_indices < card),
                    col_indices,
                    torch.zeros_like(col_indices),
                )
                feat_token = emb_layer(safe_indices)  # [B, D]
                cat_tokens_per_feature.append(feat_token.unsqueeze(1))  # [B, 1, D]

            cat_tokens = torch.cat(cat_tokens_per_feature, dim=1)  # [B, N_cat, D]
            tokens_list.append(cat_tokens)

        # 3. Concatenate all feature tokens along feature axis -> [B, F, D]
        if len(tokens_list) == 1:
            tokens = tokens_list[0]
        else:
            tokens = torch.cat(tokens_list, dim=1)

        # 4. Add Learnable Feature Identity Embeddings -> [B, F, D] + [1, F, D]
        if self.use_identity_emb and self.identity_emb is not None:
            tokens = tokens + self.identity_emb

        # 5. Apply Token Normalization & Dropout
        tokens = self.layer_norm(tokens)
        tokens = self.dropout(tokens)

        return tokens

    @classmethod
    def from_metadata_dict(
        cls,
        metadata: Dict[str, Any],
        embedding_dim: int = 64,
        bias: bool = True,
        use_identity_emb: bool = True,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ) -> "FeatureTokenizer":
        """
        Factory constructor to initialize FeatureTokenizer directly from preprocessing metadata dictionary.
        """
        num_numerical = metadata.get("numerical_features", {}).get("count", 0)
        num_names = metadata.get("numerical_features", {}).get("names", [])

        cat_info = metadata.get("categorical_features", {})
        cat_cardinalities = cat_info.get("cardinalities_with_unk", {})
        cat_names = cat_info.get("names", [])

        return cls(
            num_numerical=num_numerical,
            cat_cardinalities=cat_cardinalities,
            embedding_dim=embedding_dim,
            bias=bias,
            use_identity_emb=use_identity_emb,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            numerical_feature_names=num_names,
            categorical_feature_names=cat_names,
        )

    @classmethod
    def from_metadata_file(
        cls,
        metadata_path: Union[str, Path],
        embedding_dim: int = 64,
        bias: bool = True,
        use_identity_emb: bool = True,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ) -> "FeatureTokenizer":
        """
        Factory constructor to initialize FeatureTokenizer from metadata.json path.
        """
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return cls.from_metadata_dict(
            metadata=metadata,
            embedding_dim=embedding_dim,
            bias=bias,
            use_identity_emb=use_identity_emb,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

    def extra_repr(self) -> str:
        return (
            f"num_numerical={self.num_numerical}, "
            f"num_categorical={self.num_categorical}, "
            f"total_features={self.num_features}, "
            f"embedding_dim={self.embedding_dim}, "
            f"use_identity_emb={self.use_identity_emb}"
        )
