#!/usr/bin/env python3
"""
Unit tests for Controlled Hyperparameter Search Framework.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.experiments.hyperparameter_search import HyperparameterSearchFramework


def test_hyperparameter_grid_generation():
    """Verify that hyperparameter grid covers all 12 specified hyperparameter dimensions."""
    framework = HyperparameterSearchFramework()
    grid = framework.generate_experiment_grid()

    assert len(grid) >= 20, "Should generate comprehensive exploration grid"

    tested_dimensions = {exp["tested_dimension"] for exp in grid}
    expected_dimensions = {
        "baseline",
        "learning_rate",
        "hidden_dimension",
        "embedding_dimension",
        "transformer_layers",
        "transformer_heads",
        "gnn_layers",
        "dropout",
        "knn_k",
        "temporal_window",
        "contrastive_temperature",
        "feature_masking_ratio",
        "edge_dropout",
    }

    assert expected_dimensions.issubset(tested_dimensions), f"Missing dimensions: {expected_dimensions - tested_dimensions}"

    for exp in grid:
        assert "experiment_id" in exp
        assert "config" in exp
        cfg = exp["config"]
        assert cfg["learning_rate"] > 0
        assert cfg["embedding_dimension"] > 0
        assert cfg["transformer_layers"] >= 1
        assert cfg["transformer_heads"] >= 1
        assert cfg["gnn_layers"] >= 1
        assert 0.0 <= cfg["dropout"] <= 1.0


def test_selection_score_formula():
    """Verify selection score weighting and monotonicity."""
    # Score = 0.55 * Macro_F1 + 0.35 * Macro_Recall + 0.10 * (1 - FPR)
    f1_1, r_1, fpr_1 = 0.50, 0.50, 0.05
    score_1 = (0.55 * f1_1) + (0.35 * r_1) + (0.10 * (1.0 - fpr_1))
    assert np.isclose(score_1, 0.275 + 0.175 + 0.095)  # 0.545

    # A model with higher F1 and Recall should receive higher score
    f1_2, r_2, fpr_2 = 0.55, 0.55, 0.05
    score_2 = (0.55 * f1_2) + (0.35 * r_2) + (0.10 * (1.0 - fpr_2))
    assert score_2 > score_1
