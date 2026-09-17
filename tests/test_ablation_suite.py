#!/usr/bin/env python3
"""
Unit and Integration Tests for Ablation Models and Benchmark Suite.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from src.models.ablation_models import (
    TransformerOnlyClassifier,
    GraphSAGEOnlyClassifier,
    AblationTGCFIDS,
)
from src.evaluation.ablation_suite import AblationBenchmarkSuite


def test_ablation_models_forward_and_shapes():
    """Verify forward pass and output shapes across all ablation model configurations."""
    N = 16
    E = 32
    num_classes = 10

    x_num = torch.randn(N, 39)
    x_cat = torch.randint(0, 5, (N, 3))
    x_dense = torch.randn(N, 194)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, 6)

    # 1. Transformer Only
    tf_only = TransformerOnlyClassifier(num_numerical=39, token_dim=32, num_classes=num_classes)
    tf_logits = tf_only(x_num=x_num, x_cat=x_cat)
    assert tf_logits.shape == (N, num_classes)

    # 2. GraphSAGE Only
    gnn_only = GraphSAGEOnlyClassifier(in_channels=194, edge_dim=6, hidden_dim=32, num_classes=num_classes)
    gnn_logits = gnn_only(x=x_dense, edge_index=edge_index, edge_attr=edge_attr)
    assert gnn_logits.shape == (N, num_classes)

    # 3. Ablation D: Remove temporal edges
    abl_d = AblationTGCFIDS(
        num_numerical=39, token_dim=32, graph_in_channels=194,
        remove_temporal_edges=True, num_classes=num_classes,
    )
    d_logits = abl_d(x_num=x_num, x_cat=x_cat, edge_index=edge_index, edge_attr=edge_attr, x_dense=x_dense)
    assert d_logits.shape == (N, num_classes)

    # 4. Ablation F: Remove feature tokenizer
    abl_f = AblationTGCFIDS(
        num_numerical=39, token_dim=32, graph_in_channels=194,
        remove_feature_tokenizer=True, num_classes=num_classes,
    )
    f_logits = abl_f(x_num=x_num, x_cat=x_cat, edge_index=edge_index, edge_attr=edge_attr, x_dense=x_dense)
    assert f_logits.shape == (N, num_classes)

    # 5. Ablation H: Remove edge attributes
    abl_h = AblationTGCFIDS(
        num_numerical=39, token_dim=32, graph_in_channels=194,
        remove_edge_attrs=True, num_classes=num_classes,
    )
    h_logits = abl_h(x_num=x_num, x_cat=x_cat, edge_index=edge_index, edge_attr=edge_attr, x_dense=x_dense)
    assert h_logits.shape == (N, num_classes)


def test_ablation_models_gradients():
    """Verify end-to-end backpropagation through ablation models."""
    N = 8
    E = 16
    x_num = torch.randn(N, 39)
    x_cat = torch.randint(0, 5, (N, 3))
    x_dense = torch.randn(N, 194)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, 6)
    labels = torch.randint(0, 10, (N,))

    model = AblationTGCFIDS(
        num_numerical=39,
        token_dim=32,
        graph_in_channels=194,
        fusion_strategy="gated",
        num_classes=10,
    )

    logits = model(x_num=x_num, x_cat=x_cat, edge_index=edge_index, edge_attr=edge_attr, x_dense=x_dense)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"


def test_ablation_metric_computation():
    """Verify minority recall and security metric calculations."""
    suite = AblationBenchmarkSuite()
    y_true = np.array([0, 0, 1, 2, 8, 9, 4, 5, 6, 7])
    y_pred = np.array([0, 0, 1, 2, 0, 9, 4, 5, 6, 7])
    probs = np.eye(10)[y_pred]

    metrics = suite._compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_probs=probs,
        train_time=1.0,
        inf_time=2.0,
    )

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["fpr"] <= 1.0
    assert 0.0 <= metrics["fnr"] <= 1.0
    assert 0.0 <= metrics["minority_recall_rare4"] <= 1.0
    # Class 1, 2, 9 are correct (recall=1), class 8 predicted 0 (recall=0) -> mean = 0.75
    assert np.isclose(metrics["minority_recall_rare4"], 0.75)
