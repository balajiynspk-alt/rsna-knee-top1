"""Tests for Production Feature Pipeline."""

import pytest
import numpy as np
import torch

from production.preprocessing.feature_pipeline import ProductionFeaturePipeline
from production.preprocessing.validation import PreprocessingValidator


def test_feature_pipeline_initialization(feature_pipeline):
    """Test that feature pipeline loads successfully with correct columns."""
    assert feature_pipeline is not None
    assert len(feature_pipeline.NUMERICAL_COLS) == 39
    assert len(feature_pipeline.CATEGORICAL_COLS) == 3


def test_single_flow_transformation(feature_pipeline, sample_flow_dict):
    """Test transforming a single flow record into model-ready tensors."""
    x_num, x_cat, x_dense = feature_pipeline.transform_single(sample_flow_dict)
    
    assert isinstance(x_num, torch.Tensor)
    assert isinstance(x_cat, torch.Tensor)
    assert isinstance(x_dense, torch.Tensor)
    
    # Check shape: single flow -> (1, 39) and (1, 3) and (1, 194)
    assert x_num.shape == (1, 39)
    assert x_cat.shape == (1, 3)
    assert x_dense.shape == (1, 194)
    
    # Check types
    assert x_num.dtype == torch.float32
    assert x_cat.dtype == torch.long
    assert x_dense.dtype == torch.float32
    
    # Check no NaNs or Infs
    assert not torch.isnan(x_num).any()
    assert not torch.isinf(x_num).any()


def test_batch_flow_transformation(feature_pipeline, sample_flow_dict):
    """Test batch transformation of multiple flow records."""
    batch = [sample_flow_dict.copy() for _ in range(5)]
    # Modify some fields in the batch
    batch[1]["sbytes"] = 50000.0
    batch[2]["proto"] = "udp"
    batch[3]["service"] = "dns"
    batch[4]["dur"] = 2.5
    
    x_num, x_cat, x_dense = feature_pipeline.transform_records(batch)
    
    assert x_num.shape == (5, 39)
    assert x_cat.shape == (5, 3)
    assert x_dense.shape == (5, 194)
    assert not torch.isnan(x_num).any()


def test_unseen_categorical_handling(feature_pipeline, sample_flow_dict):
    """Test that unseen/rare categories are handled gracefully without crashing."""
    corrupted_flow = sample_flow_dict.copy()
    corrupted_flow["proto"] = "UNKNOWN_PROTO_999"
    corrupted_flow["state"] = "NON_EXISTENT_STATE"
    corrupted_flow["service"] = "ALIEN_SERVICE"
    
    x_num, x_cat, x_dense = feature_pipeline.transform_single(corrupted_flow)
    assert x_cat.shape == (1, 3)
    assert x_cat.dtype == torch.long


def test_nan_inf_sanitization():
    """Test sanitization function cleans invalid values."""
    dirty_record = {
        "src_ip": "1.1.1.1",
        "dst_ip": "2.2.2.2",
        "dur": float("inf"),
        "sbytes": float("nan"),
        "dload": -999.0,
        "proto": "TCP",
    }
    cleaned = PreprocessingValidator.validate_and_clean_record(dirty_record)
    assert cleaned["dur"] == 0.0
    assert cleaned["sbytes"] == 0.0
    assert cleaned["dload"] == -999.0
    assert cleaned["proto"] == "tcp"

