import tempfile
from pathlib import Path
import torch
import numpy as np
import pytest
from torch_geometric.data import Data

from src.models.bilstm import BiLSTMClassifier
from src.models.ablation_models import (
    TransformerOnlyClassifier,
    GraphSAGEOnlyClassifier,
)
from src.evaluation.benchmark_suite import ControlledBenchmarkFramework
from src.evaluation.plot_baselines import BaselineFigureGenerator


def _create_dummy_graphs(num_graphs=4, nodes_per_graph=20):
    graphs = []
    for _ in range(num_graphs):
        x_num = torch.randn(nodes_per_graph, 39)
        x_cat = torch.stack([
            torch.randint(0, 10, (nodes_per_graph,)),
            torch.randint(0, 5, (nodes_per_graph,)),
            torch.randint(0, 3, (nodes_per_graph,)),
        ], dim=-1).float()
        x_dense = torch.randn(nodes_per_graph, 194)

        edge_index = torch.stack([
            torch.randint(0, nodes_per_graph, (nodes_per_graph * 2,)),
            torch.randint(0, nodes_per_graph, (nodes_per_graph * 2,)),
        ], dim=0)
        edge_attr = torch.randn(nodes_per_graph * 2, 6)

        y_multiclass = torch.randint(0, 10, (nodes_per_graph,))
        y_binary = (y_multiclass > 0).long()

        data = Data(
            x_num=x_num,
            x_cat=x_cat,
            x_dense=x_dense,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_multiclass=y_multiclass,
            y_binary=y_binary,
            num_nodes=nodes_per_graph,
        )
        graphs.append(data)
    return graphs


class TestBenchmarkSuite:

    def test_bilstm_forward_and_shapes(self):
        model = BiLSTMClassifier(in_features=194, hidden_dim=32, num_layers=2, num_classes=10)
        x = torch.randn(16, 194)
        logits = model(x)
        assert logits.shape == (16, 10)

    def test_transformer_only_forward_and_shapes(self):
        model = TransformerOnlyClassifier(
            num_numerical=39,
            cat_cardinalities=[10, 5, 3],
            token_dim=16,
            transformer_heads=2,
            transformer_layers=1,
            num_classes=10,
        )
        x_num = torch.randn(8, 39)
        x_cat = torch.randint(0, 3, (8, 3))
        logits = model(x_num=x_num, x_cat=x_cat)
        assert logits.shape == (8, 10)

    def test_graphsage_only_forward_and_shapes(self):
        model = GraphSAGEOnlyClassifier(
            in_channels=194,
            edge_dim=6,
            hidden_dim=16,
            out_channels=16,
            num_layers=1,
            num_classes=10,
        )
        x = torch.randn(10, 194)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_attr = torch.randn(20, 6)
        logits = model(x=x, edge_index=edge_index, edge_attr=edge_attr)
        assert logits.shape == (10, 10)

    def test_benchmark_metrics_computation(self):
        bench = ControlledBenchmarkFramework(device="cpu")
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 0, 4, 0, 1, 2, 3, 4])
        y_proba = np.eye(10)[y_pred]

        metrics = bench.compute_metrics(y_true, y_pred, y_proba)
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "binary_fpr" in metrics
        assert "roc_auc_macro" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
