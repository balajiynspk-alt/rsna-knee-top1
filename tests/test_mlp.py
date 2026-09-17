"""
Unit tests for PyTorch MLP model architecture, training loop, and evaluation.
"""

from pathlib import Path
from typing import Tuple
import matplotlib
matplotlib.use("Agg")
import pytest
import numpy as np
import torch
import torch.nn as nn

from src.models.mlp import PyTorchMLP
from src.training.train_mlp import MLPTrainer


@pytest.fixture
def mock_mlp_tensor_data(tmp_path: Path) -> Tuple[Path, Path, Path]:
    """Create minimal synthetic PyTorch feature tensor bundles."""
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)

    n_train = 150
    n_test = 60
    in_features = 194

    train_dense = torch.randn(n_train, in_features, dtype=torch.float32)
    train_y_multi = torch.randint(0, 10, (n_train,), dtype=torch.long)
    train_y_bin = (train_y_multi > 0).long()

    test_dense = torch.randn(n_test, in_features, dtype=torch.float32)
    test_y_multi = torch.randint(0, 10, (n_test,), dtype=torch.long)
    test_y_bin = (test_y_multi > 0).long()

    train_pt = data_dir / "train_features.pt"
    test_pt = data_dir / "test_features.pt"

    torch.save({
        "x_dense": train_dense,
        "y_multiclass": train_y_multi,
        "y_binary": train_y_bin,
    }, train_pt)

    torch.save({
        "x_dense": test_dense,
        "y_multiclass": test_y_multi,
        "y_binary": test_y_bin,
    }, test_pt)

    return data_dir, results_dir, tmp_path


def test_mlp_model_forward_pass_and_shapes():
    """Verify input-to-logits dimension mapping and embedding extraction."""
    batch_size = 32
    in_features = 194
    num_classes = 10
    hidden_dims = [256, 128]

    model = PyTorchMLP(
        in_features=in_features,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=0.2,
    )

    x = torch.randn(batch_size, in_features)
    logits = model(x)

    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any()

    # Test pre-logits embedding extraction
    emb = model.get_embedding(x)
    assert emb.shape == (batch_size, hidden_dims[1])


def test_mlp_custom_hidden_dims_and_dropout():
    """Verify custom layer dimensions and invalid dimension handling."""
    model = PyTorchMLP(
        in_features=50,
        hidden_dims=[512, 256],
        num_classes=5,
        dropout=0.5,
    )

    x = torch.randn(8, 50)
    logits = model(x)
    assert logits.shape == (8, 5)

    with pytest.raises(ValueError):
        PyTorchMLP(hidden_dims=[128])  # Less than 2 layers should raise error


def test_mlp_trainer_train_and_evaluate(mock_mlp_tensor_data):
    """Test full training step, early stopping, and metric calculation on synthetic tensors."""
    data_dir, results_dir, tmp_path = mock_mlp_tensor_data

    trainer = MLPTrainer(
        in_features=194,
        hidden_dims=[64, 32],
        dropout=0.1,
        learning_rate=1e-3,
        batch_size=32,
        epochs=3,
        early_stopping_patience=2,
        seed=42,
        device="cpu",  # Test on CPU for fast isolated verification
        data_dir=data_dir,
        results_dir=results_dir,
    )

    result = trainer.train_pipeline()

    assert "model" in result
    assert "test_metrics" in result
    assert "class_report_df" in result
    assert "history" in result
    assert result["checkpoint_path"].exists()

    metrics = result["test_metrics"]
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "binary_fpr" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(metrics["per_class_f1"]) == 10
