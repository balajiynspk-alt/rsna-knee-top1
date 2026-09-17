"""
Unit tests for TGCF-IDS Step 21 Final Evaluation pipeline.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.utils.paths import ProjectPaths
from src.evaluation.final_evaluation import FinalTestEvaluator
from src.evaluation.plot_final import FinalFigureGenerator


def test_final_evaluation_artifacts_exist():
    """Verify all required final evaluation artifacts were generated and are populated."""
    metrics_json = ProjectPaths.RESULTS_FINAL / "final_metrics.json"
    report_csv = ProjectPaths.RESULTS_FINAL / "classification_report.csv"
    cm_csv = ProjectPaths.RESULTS_FINAL / "confusion_matrix.csv"
    cm_norm_csv = ProjectPaths.RESULTS_FINAL / "confusion_matrix_normalized.csv"

    assert metrics_json.exists(), "final_metrics.json must exist"
    assert report_csv.exists(), "classification_report.csv must exist"
    assert cm_csv.exists(), "confusion_matrix.csv must exist"
    assert cm_norm_csv.exists(), "confusion_matrix_normalized.csv must exist"

    # Verify JSON structure
    with open(metrics_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["evaluation_type"] == "FINAL TEST RESULTS (FROZEN MODEL)"
    assert data["total_test_flows"] == 82332
    assert "overall_metrics" in data
    ov = data["overall_metrics"]

    assert 0.0 <= ov["accuracy"] <= 1.0
    assert 0.0 <= ov["macro_f1"] <= 1.0
    assert 0.0 <= ov["weighted_f1"] <= 1.0
    assert 0.0 <= ov["macro_precision"] <= 1.0
    assert 0.0 <= ov["macro_recall"] <= 1.0
    assert 0.0 <= ov["macro_roc_auc"] <= 1.0
    assert 0.0 <= ov["macro_pr_auc"] <= 1.0
    assert 0.0 <= ov["binary_fpr"] <= 1.0
    assert 0.0 <= ov["binary_fnr"] <= 1.0


def test_final_classification_report_structure():
    """Verify classification report dataframe schema and support."""
    report_csv = ProjectPaths.RESULTS_FINAL / "classification_report.csv"
    df = pd.read_csv(report_csv)

    assert len(df) == 10, "UNSW-NB15 has 10 classes"
    assert list(df.columns) == ["Class ID", "Class Name", "Precision", "Recall", "F1 Score", "Support"]
    assert df["Support"].sum() == 82332


def test_final_confusion_matrix_structure():
    """Verify confusion matrix dimensions and sums."""
    cm_csv = ProjectPaths.RESULTS_FINAL / "confusion_matrix.csv"
    cm_df = pd.read_csv(cm_csv, index_col=0)

    assert cm_df.shape == (10, 10)
    assert cm_df.values.sum() == 82332


def test_final_figures_exist():
    """Verify all 5 required publication-quality figures exist in results/figures/final/."""
    figures_dir = ProjectPaths.RESULTS_FIGURES / "final"
    expected_figures = [
        "final_confusion_matrix.png",
        "final_confusion_matrix_normalized.png",
        "final_per_class_metrics.png",
        "final_roc_pr_summary.png",
        "final_comprehensive_dashboard.png",
    ]
    for fig_name in expected_figures:
        fig_path = figures_dir / fig_name
        assert fig_path.exists(), f"Missing expected figure: {fig_name}"
        assert fig_path.stat().st_size > 5000, f"Figure {fig_name} is unexpectedly empty"
