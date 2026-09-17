"""
Unit tests for TGCF-IDS Step 22 Cross-Dataset External Validation pipeline.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.utils.paths import ProjectPaths
from src.evaluation.cross_dataset import CICIDS2017SchemaAnalyzer, CICIDS2017BenchmarkGenerator, CrossDatasetEvaluator
from src.evaluation.plot_cross_dataset import CrossDatasetFigureGenerator


def test_schema_analyzer_spec():
    """Verify schema difference and alignment specification."""
    spec = CICIDS2017SchemaAnalyzer.get_feature_alignment_spec()
    assert "unsw_nb15" in spec
    assert "cic_ids2017" in spec
    assert "alignment_strategy" in spec
    assert spec["unsw_nb15"]["total_features"] == 42
    assert spec["cic_ids2017"]["total_features"] == 78


def test_cic_benchmark_generator():
    """Verify CIC benchmark generator creates valid graph structures."""
    gen = CICIDS2017BenchmarkGenerator(seed=42)
    graphs, meta_df = gen.generate_cic_dataset(num_samples=200, num_graphs=2)

    assert len(graphs) == 2
    assert len(meta_df) == 200
    assert hasattr(graphs[0], "x_num")
    assert hasattr(graphs[0], "x_cat")
    assert hasattr(graphs[0], "x_dense")
    assert hasattr(graphs[0], "edge_index")
    assert hasattr(graphs[0], "edge_attr")
    assert graphs[0].x_num.shape[1] == 39
    assert graphs[0].x_cat.shape[1] == 3
    assert graphs[0].x_dense.shape[1] == 194
    assert graphs[0].edge_attr.shape[1] == 6


def test_cross_dataset_artifacts_exist():
    """Verify cross dataset CSV tables and figures were generated."""
    summary_csv = ProjectPaths.RESULTS_TABLES / "cross_dataset.csv"
    per_class_csv = ProjectPaths.RESULTS_TABLES / "cross_dataset_per_class.csv"
    figures_dir = ProjectPaths.RESULTS_FIGURES / "cross_dataset"

    assert summary_csv.exists(), "cross_dataset.csv must exist"
    assert per_class_csv.exists(), "cross_dataset_per_class.csv must exist"

    df_summary = pd.read_csv(summary_csv)
    assert len(df_summary) == 3
    assert "Evaluation Mode" in df_summary.columns
    assert "Macro F1" in df_summary.columns

    df_per_class = pd.read_csv(per_class_csv)
    assert len(df_per_class) == 10

    expected_figures = [
        "cross_dataset_domain_comparison.png",
        "cross_dataset_per_class_transfer.png",
        "cross_dataset_security_rates.png",
        "cross_dataset_comprehensive_dashboard.png",
    ]
    for fig_name in expected_figures:
        fig_path = figures_dir / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_name}"
        assert fig_path.stat().st_size > 5000, f"Figure {fig_name} is empty"
