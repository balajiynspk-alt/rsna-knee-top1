#!/usr/bin/env python3
"""
TGCF-IDS: Clean PyTorch Multi-Layer Perceptron (MLP) Baseline.
Architecture:
Input -> Linear -> LayerNorm -> GELU -> Dropout -> Linear -> GELU -> Dropout -> Linear -> 10 Logits
"""

from typing import List, Optional, Any, Union
import torch
import torch.nn as nn


class PyTorchMLP(nn.Module):
    """
    Modular PyTorch MLP baseline with LayerNorm and GELU activations.
    """

    def __init__(
        self,
        in_features: int = 194,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        if len(hidden_dims) < 2:
            raise ValueError("hidden_dims must have at least 2 layers, e.g. [256, 128]")

        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_p = dropout

        h1, h2 = hidden_dims[0], hidden_dims[1]

        # Layer 1: Linear -> LayerNorm -> GELU -> Dropout
        self.fc1 = nn.Linear(in_features, h1)
        self.ln1 = nn.LayerNorm(h1)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        # Layer 2: Linear -> GELU -> Dropout
        self.fc2 = nn.Linear(h1, h2)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        # Layer 3: Output Classification Head (10 logits)
        self.fc_out = nn.Linear(h2, num_classes)

        # Weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        data: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor, optional): Input tensor of shape (batch_size, in_features).
            data (Data/Batch, optional): Optional PyG Data object containing x_dense or x.
        Returns:
            torch.Tensor: Unnormalized class logits of shape (batch_size, num_classes).
        """
        if data is not None:
            if hasattr(data, "x_dense"):
                x = data.x_dense
            elif hasattr(data, "x"):
                x = data.x

        if x is None:
            raise ValueError("PyTorchMLP requires input tensor x or data.")

        # Block 1
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.act1(out)
        out = self.drop1(out)

        # Block 2
        out = self.fc2(out)
        out = self.act2(out)
        out = self.drop2(out)

        # Output Head
        logits = self.fc_out(out)
        return logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract pre-logits latent representations (shape: batch_size, h2).
        """
        out = self.drop1(self.act1(self.ln1(self.fc1(x))))
        out = self.act2(self.fc2(out))
        return out
