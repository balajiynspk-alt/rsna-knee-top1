"""
Unit tests for TGCF-IDS Step 25 Computational Efficiency Benchmark.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.utils.paths import ProjectPaths
from src.evaluation.efficiency import EfficiencyBenchmarkSuite
from src.evaluation.plot_efficiency import EfficiencyFigureGenerator


def test_efficiency_table_exists_and_valid():
    """Verify efficiency benchmark results table exists and contains all 5 models."""
    eff_csv = ProjectPaths.RESULTS_TABLES / "efficiency.csv"
    assert eff_csv.exists(), "efficiency.csv must exist"

    df = pd.read_csv(eff_csv)
    assert len(df) == 5

    expected_models = ["MLP", "Transformer", "GraphSAGE", "GraphSAGE + Transformer", "TGCF-IDS"]
    assert list(df["Model Architecture"]) == expected_models

    assert "Parameters" in df.columns
    assert "Inference Latency (ms/1k flows)" in df.columns
    assert "Throughput (flows/sec)" in df.columns
    assert "GPU Peak Memory (MB)" in df.columns
    assert "Macro F1" in df.columns

    # Verify positive non-zero parameters and latency
    assert (df["Parameters"] > 0).all()
    assert (df["Inference Latency (ms/1k flows)"] > 0).all()
    assert (df["Throughput (flows/sec)"] > 0).all()


def test_efficiency_figures_exist():
    """Verify all 4 publication efficiency figures were generated."""
    figures_dir = ProjectPaths.RESULTS_FIGURES / "efficiency"
    expected_figures = [
        "efficiency_pareto_accuracy_vs_latency.png",
        "efficiency_memory_and_params.png",
        "efficiency_training_throughput.png",
        "efficiency_comprehensive_dashboard.png",
    ]
    for fig_name in expected_figures:
        fig_path = figures_dir / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_name}"
        assert fig_path.stat().st_size > 5000, f"Figure {fig_name} is empty"
