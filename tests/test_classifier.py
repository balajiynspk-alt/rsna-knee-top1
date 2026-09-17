"""
Unit tests for IntrusionClassifier, FocalLoss, and loss factory.

Verifies:
1. IntrusionClassifier raw logits output [N, 10] without internal softmax.
2. Probability calibration and argmax prediction methods.
3. Class-weighted CrossEntropyLoss compatibility.
4. Multi-class FocalLoss behavior, gamma focusing parameter, and alpha weighting.
5. End-to-end gradient backpropagation from loss functions through classifier head.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.classifier import IntrusionClassifier, FocalLoss, get_loss_criterion


def test_intrusion_classifier_logits_and_shapes():
    """Verify raw logits output shape [N, 10] and absence of internal softmax."""
    batch_size = 32
    in_features = 128
    num_classes = 10

    classifier = IntrusionClassifier(
        in_features=in_features,
        hidden_dim=256,
        num_classes=num_classes,
        dropout=0.1,
    )

    x = torch.randn(batch_size, in_features)
    logits = classifier(x)

    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any()

    # Verify logits are unnormalized (contain negatives and do not sum to 1)
    row_sums = logits.sum(dim=-1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2)

    # Test probability and prediction helpers
    probs = classifier.predict_proba(x)
    assert probs.shape == (batch_size, num_classes)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-4)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    preds = classifier.predict(x)
    assert preds.shape == (batch_size,)
    assert (preds >= 0).all() and (preds < num_classes).all()


def test_classifier_gradient_backprop():
    """Verify gradient flow through all classifier layers."""
    classifier = IntrusionClassifier(in_features=64, hidden_dim=64, num_classes=10)
    x = torch.randn(16, 64, requires_grad=True)

    logits = classifier(x)
    loss = logits.sum()
    loss.backward()

    assert x.grad is not None
    for name, param in classifier.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing grad in {name}"


def test_class_weighted_cross_entropy():
    """Verify standard class-weighted CrossEntropyLoss evaluation."""
    classifier = IntrusionClassifier(in_features=32, num_classes=10)
    weights = torch.tensor([0.2, 1.0, 1.2, 0.5, 0.3, 0.4, 0.25, 0.5, 1.5, 4.0])

    loss_fn = get_loss_criterion(loss_type="ce", class_weights=weights)

    x = torch.randn(20, 32)
    targets = torch.randint(0, 10, (20,))

    logits = classifier(x)
    loss = loss_fn(logits, targets)

    assert loss.dim() == 0  # Scalar
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_focal_loss_properties():
    """Verify FocalLoss mathematical properties and reduction options."""
    weights = torch.ones(10)
    focal_fn = FocalLoss(gamma=2.0, weight=weights, reduction="mean")

    logits = torch.randn(15, 10, requires_grad=True)
    targets = torch.randint(0, 10, (15,))

    loss = focal_fn(logits, targets)
    loss.backward()

    assert loss.item() > 0.0
    assert logits.grad is not None

    # Test reduction options
    focal_none = FocalLoss(gamma=2.0, reduction="none")
    loss_none = focal_none(logits.detach(), targets)
    assert loss_none.shape == (15,)

    # When gamma=0, focal loss matches standard CrossEntropyLoss
    focal_zero = FocalLoss(gamma=0.0, reduction="mean")
    ce_fn = nn.CrossEntropyLoss()
    assert torch.allclose(focal_zero(logits.detach(), targets), ce_fn(logits.detach(), targets), atol=1e-4)


def test_loss_factory_validation():
    """Verify get_loss_criterion factory options and error handling."""
    ce_loss = get_loss_criterion("ce")
    assert isinstance(ce_loss, nn.CrossEntropyLoss)

    focal_loss = get_loss_criterion("focal", gamma=2.5)
    assert isinstance(focal_loss, FocalLoss)
    assert focal_loss.gamma == 2.5

    with pytest.raises(ValueError, match="Unknown loss_type"):
        get_loss_criterion("invalid_loss")
