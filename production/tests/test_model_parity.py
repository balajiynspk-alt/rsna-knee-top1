"""Tests for Model Loading, Forward Pass, and Parity."""

import pytest
import torch
from pathlib import Path

from production.model.model_loader import load_production_model, compute_file_sha256
from production.model.model_adapter import TGCFIDSProductionAdapter
from production.model.schema import PredictionResult


def test_checkpoint_integrity():
    """Verify that checkpoint file exists and matches expected SHA256."""
    ckpt_path = Path("results/checkpoints/final_tgcf_ids_model.pt")
    assert ckpt_path.exists(), "Model checkpoint must exist in results/checkpoints/"
    
    sha256 = compute_file_sha256(str(ckpt_path))
    assert sha256 == "f6a85511ef0d44bed49a481c4a720a9173c57ec9f91ed64192576b019cf8cd54"


def test_model_loading():
    """Verify that model loads into eval mode with frozen parameters."""
    model = load_production_model(device="cpu")
    assert model is not None
    
    # Check that model parameters are frozen (eval mode, no grad required)
    assert not model.training
    for param in model.parameters():
        assert not param.requires_grad


def test_adapter_inference_without_graph(model_adapter, feature_pipeline, sample_flow_dict):
    """Test inference in standalone tabular mode (no graph)."""
    x_num, x_cat, x_dense = feature_pipeline.transform_single(sample_flow_dict)
    
    results = model_adapter.predict_tensor_batch(
        x_num=x_num,
        x_cat=x_cat,
        x_dense=x_dense,
        flow_metadata=[sample_flow_dict]
    )
    
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, PredictionResult)
    assert 0 <= result.class_id < 10
    assert isinstance(result.predicted_class, str)
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.probabilities) == 10
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-3
    assert result.severity in ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert result.inference_latency_ms > 0.0


def test_adapter_inference_with_graph(model_adapter, feature_pipeline, sample_flow_dict):
    """Test inference with graph context tensors."""
    x_num, x_cat, x_dense = feature_pipeline.transform_single(sample_flow_dict)
    
    edge_index = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)
    edge_attr = torch.randn(2, 6, dtype=torch.float32)
    
    results = model_adapter.predict_tensor_batch(
        x_num=x_num,
        x_cat=x_cat,
        x_dense=x_dense,
        edge_index=edge_index,
        edge_attr=edge_attr,
        flow_metadata=[sample_flow_dict]
    )
    
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, PredictionResult)
    assert 0 <= result.class_id < 10
    assert result.gate_feature_weight is not None
    assert 0.0 <= result.gate_feature_weight <= 1.0


def test_batch_inference(model_adapter, feature_pipeline, sample_flow_dict):
    """Test batch inference for multiple flows."""
    batch = [sample_flow_dict.copy() for _ in range(4)]
    x_num, x_cat, x_dense = feature_pipeline.transform_records(batch)
    
    results = model_adapter.predict_tensor_batch(
        x_num=x_num,
        x_cat=x_cat,
        x_dense=x_dense,
        flow_metadata=batch
    )
    
    assert len(results) == 4
    for r in results:
        assert isinstance(r, PredictionResult)
        assert 0 <= r.class_id < 10
