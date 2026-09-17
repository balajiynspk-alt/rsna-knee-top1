"""
Unit tests for TemporalGraphBuilder.

Verifies:
1. Sparse PyG graph construction from tabular flow batches.
2. Zero label leakage (altering labels does not affect graph topology or edge attributes).
3. 6-dimensional edge attributes computation and valid value ranges.
4. Configurable window size, temporal radius, and KNN parameters.
5. Graph diagnostics calculation (nodes, edges, degrees, components, isolated nodes).
6. Graph visualization rendering.
"""

from pathlib import Path
import pytest
import numpy as np
import torch
from torch_geometric.data import Data

from src.graphs.temporal_graph_builder import TemporalGraphBuilder
from src.utils.paths import ProjectPaths


@pytest.fixture
def synthetic_flow_batch():
    """Create reproducible synthetic tabular flow tensors."""
    np.random.seed(42)
    torch.manual_seed(42)

    N = 100
    x_num = torch.randn(N, 39, dtype=torch.float32)
    x_cat = torch.randint(0, 5, (N, 3), dtype=torch.long)
    x_dense = torch.randn(N, 194, dtype=torch.float32)
    y_multi = torch.randint(0, 10, (N,), dtype=torch.long)
    y_bin = (y_multi > 0).long()
    meta_ids = torch.arange(1000, 1000 + N, dtype=torch.long)
    meta_seq = torch.arange(N, dtype=torch.long)

    return {
        "x_num": x_num,
        "x_cat": x_cat,
        "x_dense": x_dense,
        "y_multiclass": y_multi,
        "y_binary": y_bin,
        "meta_ids": meta_ids,
        "meta_sequence_index": meta_seq,
    }


def test_temporal_graph_builder_structure_and_dimensions(synthetic_flow_batch):
    """Verify PyG Data attributes, tensor shapes, and absence of self-loops."""
    builder = TemporalGraphBuilder(
        window_size=100,
        knn_k=4,
        temporal_radius=2,
    )

    b = synthetic_flow_batch
    graph = builder.build_window_graph(
        x_num=b["x_num"],
        x_cat=b["x_cat"],
        x_dense=b["x_dense"],
        y_multi=b["y_multiclass"],
        y_bin=b["y_binary"],
        meta_ids=b["meta_ids"],
        meta_seq=b["meta_sequence_index"],
        window_idx=0,
    )

    assert isinstance(graph, Data)
    assert graph.num_nodes == 100
    assert graph.edge_index.dim() == 2
    assert graph.edge_index.size(0) == 2
    assert graph.edge_index.size(1) > 0  # Has edges

    # Check edge attribute shape [E, 6]
    E = graph.edge_index.size(1)
    assert graph.edge_attr.shape == (E, 6)
    assert not torch.isnan(graph.edge_attr).any()
    assert not torch.isinf(graph.edge_attr).any()

    # Verify no self loops
    src, dst = graph.edge_index[0], graph.edge_index[1]
    assert not (src == dst).any(), "Graph should not contain self loops"

    # Verify edge indices within range [0, N-1]
    assert (src >= 0).all() and (src < 100).all()
    assert (dst >= 0).all() and (dst < 100).all()


def test_zero_label_leakage_in_graph_construction(synthetic_flow_batch):
    """
    CRITICAL: Verify that modifying or corrupting target labels has ZERO effect on
    the constructed graph topology (edge_index) and edge attributes (edge_attr).
    """
    builder = TemporalGraphBuilder(window_size=100, knn_k=4, temporal_radius=2)
    b = synthetic_flow_batch

    # 1. Build graph with true labels
    g1 = builder.build_window_graph(
        x_num=b["x_num"],
        x_cat=b["x_cat"],
        x_dense=b["x_dense"],
        y_multi=b["y_multiclass"],
        y_bin=b["y_binary"],
    )

    # 2. Build graph with inverted / randomized labels
    scrambled_y_multi = (b["y_multiclass"] + 5) % 10
    scrambled_y_bin = 1 - b["y_binary"]

    g2 = builder.build_window_graph(
        x_num=b["x_num"],
        x_cat=b["x_cat"],
        x_dense=b["x_dense"],
        y_multi=scrambled_y_multi,
        y_bin=scrambled_y_bin,
    )

    # Graph topology and edge features must remain completely identical
    assert torch.equal(g1.edge_index, g2.edge_index), "Edge index differed when labels changed! Label leakage detected."
    assert torch.allclose(g1.edge_attr, g2.edge_attr), "Edge attributes differed when labels changed! Label leakage detected."


def test_graph_diagnostics_computation(synthetic_flow_batch):
    """Verify graph topological diagnostics calculation."""
    builder = TemporalGraphBuilder(window_size=50, knn_k=3, temporal_radius=1)
    graphs = builder.build_dataset_graphs(synthetic_flow_batch)

    assert len(graphs) == 2  # 100 samples / 50 window size = 2 windows
    diag_df = builder.compute_graph_diagnostics(graphs)

    assert len(diag_df) == 2
    assert "avg_degree" in diag_df.columns
    assert "connected_components" in diag_df.columns
    assert "isolated_nodes" in diag_df.columns
    assert (diag_df["num_nodes"] == 50).all()
    assert (diag_df["avg_degree"] > 0).all()


def test_graph_window_visualization(tmp_path: Path, synthetic_flow_batch):
    """Verify rendering of network topology visualization."""
    builder = TemporalGraphBuilder(window_size=40, knn_k=3, temporal_radius=1)
    b = synthetic_flow_batch
    graph = builder.build_window_graph(
        x_num=b["x_num"][:40],
        x_cat=b["x_cat"][:40],
        x_dense=b["x_dense"][:40],
        y_multi=b["y_multiclass"][:40],
        y_bin=b["y_binary"][:40],
    )

    out_fig = tmp_path / "test_graph_vis.png"
    builder.visualize_graph_window(graph, out_fig, max_nodes_to_draw=40)

    assert out_fig.exists()
    assert out_fig.stat().st_size > 1000  # File is valid PNG image
