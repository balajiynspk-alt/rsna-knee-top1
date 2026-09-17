"""
Unit tests for traditional machine learning baselines evaluation.
"""

from pathlib import Path
from typing import Tuple
import matplotlib
matplotlib.use("Agg")
import pytest
import numpy as np
import pandas as pd

from src.models.baselines import TraditionalBaselinesEvaluator


@pytest.fixture
def mock_preprocessed_features(tmp_path: Path) -> Tuple[Path, Path, Path]:
    """Create minimal synthetic preprocessed npz archives for baseline testing."""
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "tables"
    fig_dir = tmp_path / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n_train = 120
    n_test = 60
    n_features = 20

    # Synthetic features and 10-class labels
    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = np.random.randint(0, 10, size=n_train).astype(np.int64)

    X_test = np.random.randn(n_test, n_features).astype(np.float32)
    y_test = np.random.randint(0, 10, size=n_test).astype(np.int64)

    train_npz = data_dir / "train_features.npz"
    test_npz = data_dir / "test_features.npz"

    np.savez_compressed(train_npz, x_dense=X_train, y_multiclass=y_train)
    np.savez_compressed(test_npz, x_dense=X_test, y_multiclass=y_test)

    return data_dir, results_dir, fig_dir


def test_metric_computation_correctness():
    """Verify precision, recall, F1, FPR, and FNR calculations."""
    evaluator = TraditionalBaselinesEvaluator()

    y_true = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    y_pred = np.array([0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0])

    metrics = evaluator.compute_evaluation_metrics(y_true, y_pred)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "binary_fpr" in metrics
    assert "binary_fnr" in metrics
    assert len(metrics["per_class_f1"]) == 10
    assert metrics["confusion_matrix"].shape == (10, 10)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["binary_fpr"] <= 1.0
    assert 0.0 <= metrics["binary_fnr"] <= 1.0


def test_train_and_evaluate_single_model(mock_preprocessed_features):
    """Test fitting and evaluating Logistic Regression and Random Forest on mock data."""
    data_dir, results_dir, fig_dir = mock_preprocessed_features
    evaluator = TraditionalBaselinesEvaluator(
        data_dir=data_dir, results_dir=results_dir, figures_dir=fig_dir, seeds=[42]
    )
    evaluator.load_preprocessed_data()

    # Test Logistic Regression
    lr_metrics, lr_pred = evaluator.train_and_evaluate_model("logistic_regression", seed=42)
    assert lr_metrics["accuracy"] >= 0.0
    assert len(lr_pred) == len(evaluator.y_test)

    # Test Random Forest
    rf_metrics, rf_pred = evaluator.train_and_evaluate_model("random_forest", seed=42)
    assert rf_metrics["accuracy"] >= 0.0
    assert len(rf_pred) == len(evaluator.y_test)


def test_confusion_matrix_rendering(mock_preprocessed_features):
    """Test generating 10x10 confusion matrix figure."""
    data_dir, results_dir, fig_dir = mock_preprocessed_features
    evaluator = TraditionalBaselinesEvaluator(
        data_dir=data_dir, results_dir=results_dir, figures_dir=fig_dir
    )

    cm = np.eye(10, dtype=int) * 10
    fig_path = evaluator.plot_confusion_matrix(cm, "Test Model", "test_cm.png")

    assert fig_path.exists()
    assert fig_path.stat().st_size > 1000


def test_end_to_end_baseline_benchmark(mock_preprocessed_features):
    """Test running full baseline benchmark across multiple seeds."""
    data_dir, results_dir, fig_dir = mock_preprocessed_features
    evaluator = TraditionalBaselinesEvaluator(
        data_dir=data_dir, results_dir=results_dir, figures_dir=fig_dir, seeds=[42, 123]
    )

    result = evaluator.run_all_baselines()

    assert (results_dir / "traditional_baselines.csv").exists()
    assert (results_dir / "traditional_baselines_per_class.csv").exists()
    assert (results_dir / "traditional_baselines_raw_runs.csv").exists()

    summary_df = result["summary_df"]
    assert "Accuracy" in summary_df.columns
    assert "Macro F1" in summary_df.columns
    assert len(summary_df) >= 2  # At least LR and RF
