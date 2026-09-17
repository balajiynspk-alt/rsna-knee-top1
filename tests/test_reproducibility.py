#!/usr/bin/env python3
"""
Unit tests for Strict Multi-Seed Reproducibility & Stability Protocol.
"""

import pytest
import numpy as np
import torch

from src.evaluation.reproducibility import seed_everything, seed_worker, ReproducibilityEvaluator


def test_seed_everything_determinism():
    """Verify that seed_everything produces deterministic tensor generation."""
    seed_everything(42)
    t1 = torch.randn(10, 10)
    a1 = np.random.randn(10)

    seed_everything(42)
    t2 = torch.randn(10, 10)
    a2 = np.random.randn(10)

    assert torch.equal(t1, t2), "PyTorch tensors should be identical for same seed"
    assert np.array_equal(a1, a2), "NumPy arrays should be identical for same seed"


def test_reproducibility_evaluator_metrics():
    """Verify statistics computation (Mean, Std, Min, Max)."""
    evaluator = ReproducibilityEvaluator()
    y_true = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_pred = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    probs = np.eye(10)[y_pred]

    res = evaluator._compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_probs=probs,
        train_time=5.0,
        inf_time=2.5,
    )

    assert res["accuracy"] == 1.0
    assert res["macro_f1"] == 1.0
    assert res["fpr"] == 0.0
    assert res["fnr"] == 0.0
    assert len(res["per_class_f1"]) == 10
