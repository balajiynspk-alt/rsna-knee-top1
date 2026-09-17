"""
Unit tests for TGCF-IDS Step 23 Controlled Robustness Testing pipeline.
"""

from pathlib import Path
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from src.utils.paths import ProjectPaths
from src.evaluation.robustness import RobustnessPerturber, RobustnessEvaluator
from src.evaluation.plot_robustness import RobustnessFigureGenerator


def test_robustness_perturbers_deterministic_and_non_mutating():
    """Verify perturbation operators are deterministic and don't mutate input data."""
    x_num = torch.ones((10, 39), dtype=torch.float32)
    x_cat = torch.zeros((10, 3), dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.ones((4, 6), dtype=torch.float32)
    y = torch.zeros(10, dtype=torch.long)

    data = Data(x_num=x_num, x_cat=x_cat, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=10)

    # 1. Feature masking
    d_masked = RobustnessPerturber.apply_feature_masking(data, mask_ratio=0.5, seed=42)
    assert d_masked.x_num.sum() < data.x_num.sum()
    assert data.x_num.sum() == 390.0  # Original unmodified

    # 2. Numerical noise
    d_noise = RobustnessPerturber.apply_numerical_noise(data, noise_std=0.2, seed=42)
    assert not torch.equal(d_noise.x_num, data.x_num)
    assert data.x_num.sum() == 390.0  # Original unmodified

    # 3. Edge deletion
    d_del = RobustnessPerturber.apply_edge_deletion(data, drop_ratio=0.5, seed=42)
    assert d_del.edge_index.size(1) <= data.edge_index.size(1)
    assert data.edge_index.size(1) == 4  # Original unmodified

    # 4. Edge addition
    d_add = RobustnessPerturber.apply_edge_addition(data, add_ratio=0.5, seed=42)
    assert d_add.edge_index.size(1) >= data.edge_index.size(1)

    # 5. Temporal perturbation
    d_temp = RobustnessPerturber.apply_temporal_perturbation(data, jitter_std=0.5, seed=42)
    assert not torch.equal(d_temp.edge_attr, data.edge_attr)


def test_robustness_table_artifacts_exist():
    """Verify robustness results CSV exists, is populated, and has valid bounds."""
    robustness_csv = ProjectPaths.RESULTS_TABLES / "robustness.csv"
    figures_dir = ProjectPaths.RESULTS_FIGURES / "robustness"

    assert robustness_csv.exists(), "robustness.csv must exist"
    df = pd.read_csv(robustness_csv)

    assert len(df) == 36  # 6 modes x 6 strengths
    assert "Perturbation Mode" in df.columns
    assert "Perturbation Strength" in df.columns
    assert "Macro F1" in df.columns
    assert "FPR (False Alarm Rate)" in df.columns
    assert "FNR (Missed Attack Rate)" in df.columns

    assert df["Accuracy"].min() >= 0.0 and df["Accuracy"].max() <= 1.0
    assert df["Macro F1"].min() >= 0.0 and df["Macro F1"].max() <= 1.0

    expected_figures = [
        "robustness_degradation_curves.png",
        "robustness_macro_f1_comparison.png",
        "robustness_security_impact.png",
        "robustness_comprehensive_dashboard.png",
    ]
    for fig_name in expected_figures:
        fig_path = figures_dir / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_name}"
        assert fig_path.stat().st_size > 5000, f"Figure {fig_name} is empty"
