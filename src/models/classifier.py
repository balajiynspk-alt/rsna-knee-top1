#!/usr/bin/env python3
"""
TGCF-IDS: 10-Class Intrusion Classifier and Loss Functions.

Maps fused multi-modal representations into raw unnormalized classification logits:
    fused_repr in R^{N x D_in} -> Logits in R^{N x 10}

Taxonomy:
    0: Normal
    1: Analysis
    2: Backdoor
    3: DoS
    4: Exploits
    5: Fuzzers
    6: Generic
    7: Reconnaissance
    8: Shellcode
    9: Worms

Loss Functions:
- Class-weighted CrossEntropyLoss
- Multi-Class Focal Loss (with focusing parameter gamma and class weighting alpha)
"""

from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntrusionClassifier(nn.Module):
    """
    10-Class Network Intrusion Detection Classifier Head.

    Architecture:
        Linear(in_features, hidden_dim)
        -> LayerNorm(hidden_dim)
        -> GELU()
        -> Dropout(dropout)
        -> Linear(hidden_dim, 10)
        -> 10 raw unnormalized logits
    """

    CLASS_NAMES = [
        "Normal", "Analysis", "Backdoor", "DoS", "Exploits",
        "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
    ]
    NUM_CLASSES = 10

    def __init__(
        self,
        in_features: int = 128,
        hidden_dim: Optional[int] = None,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim or max(in_features, 128)
        self.num_classes = num_classes
        self.dropout_rate = dropout

        self.net = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize linear projection weights."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute unnormalized classification logits.

        Args:
            x: Fused representation tensor of shape [N, in_features].

        Returns:
            logits: Tensor of shape [N, 10] (no softmax applied).
        """
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"Expected 2D input of shape [N, {self.in_features}], got shape {x.shape}"
            )
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute softmax class probabilities [N, 10]."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Compute argmax class predictions [N]."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, hidden_dim={self.hidden_dim}, num_classes={self.num_classes}"


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss for imbalanced class classification.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (gamma >= 0). Higher gamma suppresses easy examples.
        weight: Optional class weights alpha of shape [num_classes].
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.register_buffer("weight", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw unnormalized predictions [N, num_classes].
            targets: Integer ground-truth class labels [N].

        Returns:
            loss: Scalar or per-sample focal loss.
        """
        log_probs = F.log_softmax(logits, dim=-1)           # [N, C]
        probs = torch.exp(log_probs)                        # [N, C]

        # Gather target probabilities
        targets = targets.view(-1, 1)                       # [N, 1]
        log_pt = log_probs.gather(1, targets).view(-1)      # [N]
        pt = probs.gather(1, targets).view(-1)              # [N]

        # Focal modulating factor: (1 - pt)^gamma
        focal_factor = (1.0 - pt) ** self.gamma             # [N]

        loss = -focal_factor * log_pt                       # [N]

        # Class weight alpha
        if self.weight is not None:
            alpha = self.weight[targets.view(-1)]           # [N]
            loss = loss * alpha

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: '{self.reduction}'")


def get_loss_criterion(
    loss_type: str = "ce",
    class_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
) -> nn.Module:
    """
    Factory function to instantiate classification loss.

    Args:
        loss_type: 'ce' (CrossEntropyLoss) or 'focal' (FocalLoss).
        class_weights: Optional class weighting tensor [num_classes].
        gamma: Focusing parameter for Focal Loss.

    Returns:
        loss_fn: nn.Module
    """
    l_type = loss_type.lower()
    if l_type in ("ce", "crossentropy", "cross_entropy"):
        return nn.CrossEntropyLoss(weight=class_weights)
    elif l_type in ("focal", "focal_loss"):
        return FocalLoss(gamma=gamma, weight=class_weights)
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Choose from 'ce', 'focal'.")
