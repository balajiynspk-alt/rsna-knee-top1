#!/usr/bin/env python3
"""
TGCF-IDS: Tabular Feature Transformer Encoder.

Processes a batch of feature tokens [B, F, D] using Multi-Head Self-Attention (MHSA)
and Position-wise Feed-Forward Networks with Pre-LayerNorm, GELU activations, and residual connections.

Outputs:
1. Token-level representations: [B, F, D] (contextualized feature representations)
2. Pooled flow representation:  [B, D]   (compact network-flow vector for GNNs / downstream heads)
3. Attention maps (optional):    List of [B, Num_Heads, F, F] for introspection

Note:
Attention distributions reflect internal routing weights across features,
not causal explanations of network attack mechanisms.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.feature_tokenizer import FeatureTokenizer


class FeatureTransformerOutput(NamedTuple):
    """Container for FeatureTransformer forward pass outputs."""
    flow_repr: torch.Tensor             # Pooled flow-level representation: [B, D]
    token_repr: torch.Tensor            # Contextualized token representations: [B, F, D]
    attention_weights: Optional[List[torch.Tensor]] = None  # Per-layer attention matrices: [B, H, F, F]


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder Layer with Pre-LayerNorm and GELU.

    Structure:
        x -> LayerNorm -> Multi-Head Self-Attention -> Dropout -> (+) ->
        x -> LayerNorm -> FeedForward (Linear -> GELU -> Dropout -> Linear -> Dropout) -> (+)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim or (embedding_dim * 4)

        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})."
            )

        # Multi-Head Attention
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        # Feed-Forward Network
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, self.ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ffn_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of encoder block.

        Args:
            x: Input tokens of shape [B, F, D].
            need_weights: Whether to compute and return attention weight matrices.

        Returns:
            out: Updated tokens of shape [B, F, D].
            attn_weights: Attention matrix of shape [B, H, F, F] if need_weights=True, else None.
        """
        # Pre-LN Self-Attention with residual connection
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.mha(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            need_weights=need_weights,
            average_attn_weights=False,  # Keep per-head attention maps [B, H, F, F]
        )
        x = x + self.dropout1(attn_out)

        # Pre-LN FFN with residual connection
        x = x + self.ffn(self.norm2(x))

        return x, attn_weights


class AttentionPooling(nn.Module):
    """
    Learnable Attention-based Weighted Pooling layer over tokens.
    Computes a learned softmax weighting over feature tokens:
        weights = softmax(Linear(tokens)) -> [B, F, 1]
        flow_repr = sum(weights * tokens, dim=1) -> [B, D]
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, F, D]
        scores = self.scorer(tokens)  # [B, F, 1]
        weights = F.softmax(scores, dim=1)  # [B, F, 1]
        pooled = (tokens * weights).sum(dim=1)  # [B, D]
        return pooled


class FeatureTransformer(nn.Module):
    """
    Feature Transformer for Tabular Network-Flow Tokens.

    Takes tokenized network features [B, F, D], applies multiple Transformer encoder layers,
    and pools the contextualized tokens into a global network-flow representation [B, D].

    Args:
        embedding_dim: Dimensionality D of tokens and attention projections.
        num_heads: Number of attention heads (embedding_dim must be divisible by num_heads).
        num_layers: Number of stacked Transformer encoder layers.
        ffn_dim: Hidden dimension of FFN (defaults to 4 * embedding_dim).
        dropout: Dropout rate across attention and FFN.
        attention_dropout: Dropout rate inside attention softmax.
        pooling: Pooling strategy ('mean', 'cls', 'max', 'attention').
        use_final_norm: Whether to apply a final LayerNorm on the output tokens and flow representation.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        pooling: str = "mean",
        use_final_norm: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim or (embedding_dim * 4)
        self.dropout = dropout
        self.pooling = pooling.lower()
        self.use_final_norm = use_final_norm

        if self.pooling not in ("mean", "cls", "max", "attention"):
            raise ValueError(
                f"Unknown pooling method '{pooling}'. Choose from 'mean', 'cls', 'max', 'attention'."
            )

        # Stacked Transformer Encoder Layers
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                embedding_dim=self.embedding_dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                dropout=self.dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(self.num_layers)
        ])

        # Optional [CLS] token
        if self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.empty(1, 1, self.embedding_dim))
            nn.init.normal_(self.cls_token, mean=0.0, std=1.0 / (self.embedding_dim ** 0.5))
        else:
            self.register_parameter("cls_token", None)

        # Optional Attention Pooling Scorer
        if self.pooling == "attention":
            self.attn_pooler = AttentionPooling(self.embedding_dim)
        else:
            self.attn_pooler = None

        # Final LayerNorm
        if self.use_final_norm:
            self.final_norm = nn.LayerNorm(self.embedding_dim)
        else:
            self.final_norm = nn.Identity()

    def forward(
        self,
        tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> FeatureTransformerOutput:
        """
        Transform feature tokens into contextualized representations and pooled flow embedding.

        Args:
            tokens: Input feature tokens of shape [B, F, D].
            return_attention: If True, returns per-layer attention matrices.

        Returns:
            FeatureTransformerOutput with:
                - flow_repr: [B, D]
                - token_repr: [B, F, D] (or [B, F+1, D] if 'cls' pooling)
                - attention_weights: List of [B, H, F, F] or None
        """
        if tokens.dim() != 3:
            raise ValueError(f"Expected tokens of shape [B, F, D], got shape {tokens.shape}")

        batch_size, num_features, dim = tokens.shape
        if dim != self.embedding_dim:
            raise ValueError(
                f"Token dimension ({dim}) does not match Transformer embedding_dim ({self.embedding_dim})."
            )

        x = tokens

        # Prepend [CLS] token if configured
        if self.pooling == "cls" and self.cls_token is not None:
            cls_expanded = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
            x = torch.cat([cls_expanded, x], dim=1)                   # [B, F+1, D]

        attention_maps: Optional[List[torch.Tensor]] = [] if return_attention else None

        # Pass through Transformer encoder layers
        for layer in self.layers:
            x, attn = layer(x, need_weights=return_attention)
            if return_attention and attn is not None:
                attention_maps.append(attn)

        # Apply final LayerNorm
        x = self.final_norm(x)

        # Compute pooled flow-level representation
        if self.pooling == "cls":
            flow_repr = x[:, 0, :]                   # [B, D] from [CLS] position
            token_repr = x[:, 1:, :]                 # [B, F, D]
        elif self.pooling == "mean":
            flow_repr = x.mean(dim=1)                # [B, D]
            token_repr = x                           # [B, F, D]
        elif self.pooling == "max":
            flow_repr = x.max(dim=1).values          # [B, D]
            token_repr = x                           # [B, F, D]
        elif self.pooling == "attention":
            flow_repr = self.attn_pooler(x)          # [B, D]
            token_repr = x                           # [B, F, D]
        else:
            flow_repr = x.mean(dim=1)
            token_repr = x

        return FeatureTransformerOutput(
            flow_repr=flow_repr,
            token_repr=token_repr,
            attention_weights=attention_maps if return_attention else None,
        )


class TabularFeatureTransformer(nn.Module):
    """
    End-to-End Tabular Feature Transformer.

    Combines FeatureTokenizer + FeatureTransformer into a unified module:
        (x_num, x_cat) -> Tokenizer -> Transformer -> (flow_repr, token_repr, attn_weights)
    """

    def __init__(
        self,
        tokenizer: FeatureTokenizer,
        transformer: FeatureTransformer,
    ):
        super().__init__()
        if tokenizer.embedding_dim != transformer.embedding_dim:
            raise ValueError(
                f"Tokenizer embedding_dim ({tokenizer.embedding_dim}) must match "
                f"Transformer embedding_dim ({transformer.embedding_dim})."
            )
        self.tokenizer = tokenizer
        self.transformer = transformer

    def forward(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
        x_dict: Optional[Dict[str, torch.Tensor]] = None,
        x_combined: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> FeatureTransformerOutput:
        """
        End-to-end tokenization and transformation of tabular network features.
        """
        tokens = self.tokenizer(
            x_num=x_num,
            x_cat=x_cat,
            x_dict=x_dict,
            x_combined=x_combined,
        )
        return self.transformer(tokens, return_attention=return_attention)
