#!/usr/bin/env python3
"""
TGCF-IDS: BiLSTM Tabular Network-Flow Baseline Model.

Architecture:
Input (x_dense: [B, 194] or Tokenized Features: [B, 42, D])
  ?
Linear Projection (if dense) / Direct Token Stream
  ?
Bidirectional LSTM (2 Layers, LayerNorm, Dropout)
  ?
Mean & Max Pooling Concatenation
  ?
Linear Classification Head (10 raw logits)
"""

from typing import Dict, Optional, Union, Any
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM Classifier for intrusion detection network flows.
    """

    def __init__(
        self,
        in_features: int = 194,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.2,
        use_tokens: bool = False,
        token_dim: int = 64,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_tokens = use_tokens

        # Linear projection to feed into LSTM as a pseudo-sequence if dense
        # Treating features as sequence of chunks or token dimensions
        self.seq_len = 10
        self.feat_per_step = in_features // self.seq_len
        self.rem = in_features % self.seq_len

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, self.seq_len * hidden_dim),
            nn.LayerNorm(self.seq_len * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.ln_post = nn.LayerNorm(hidden_dim * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # mean + max pooling (hidden_dim * 2 * 2)
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        data: Optional[Union[Data, Batch]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Dense input tensor [B, 194]
            data: Optional PyG Data/Batch containing x_dense or x
        Returns:
            logits: [B, 10]
        """
        if data is not None:
            if hasattr(data, "x_dense"):
                x = data.x_dense
            elif hasattr(data, "x"):
                x = data.x

        if x is None:
            raise ValueError("BiLSTM requires input tensor x or data with x_dense/x.")

        batch_size = x.size(0)

        # Reshape to sequence: [B, seq_len, hidden_dim]
        proj = self.input_proj(x).view(batch_size, self.seq_len, self.hidden_dim)

        lstm_out, _ = self.lstm(proj)  # [B, seq_len, hidden_dim * 2]
        lstm_out = self.ln_post(lstm_out)

        # Global pooling across sequence dimension
        mean_pool = torch.mean(lstm_out, dim=1)  # [B, hidden_dim * 2]
        max_pool, _ = torch.max(lstm_out, dim=1)  # [B, hidden_dim * 2]
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # [B, hidden_dim * 4]

        logits = self.classifier(pooled)  # [B, 10]
        return logits
