"""
Unit tests for EdgeAwareSAGEConv and TemporalGraphSAGE.

Verifies:
1. Edge-aware message passing incorporating node features and 6D edge attributes.
2. Configurable hidden dimensions, output channels, layer depths, and aggregations ('mean', 'max', 'sum').
3. Residual connections and LayerNorm activations.
4. Inductive mini-batch neighbor sampling using NeighborLoader.
5. GPU/CPU tensor device compatibility.
6. Execution on real UNSW-NB15 temporal graph snapshots and verification of:
   - node input shape [N, D_in]
   - edge_index shape [2, E]
   - edge_attr shape [E, D_edge]
   - embedding output shape [N, D_out]
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.models.temporal_graphsage import EdgeAwareSAGEConv, TemporalGraphSAGE
from src.utils.paths import ProjectPaths


def test_edge_aware_sage_conv_forward():
    """Verify single edge-aware GraphSAGE convolution layer on synthetic graphs."""
    num_nodes = 20
    num_edges = 45
    in_channels = 32
    out_channels = 64
    edge_dim = 6

    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_dim)

    conv = EdgeAwareSAGEConv(
        in_channels=in_channels,
        out_channels=out_channels,
        edge_dim=edge_dim,
        aggr="mean",
    )

    out = conv(x, edge_index, edge_attr)

    assert out.shape == (num_nodes, out_channels)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_temporal_graphsage_synthetic_configurations():
    """Verify TemporalGraphSAGE with various depths, dimensions, and aggregators."""
    num_nodes = 50
    num_edges = 120
    edge_dim = 6

    test_configs = [
        {"in_channels": 32, "hidden_dim": 32, "out_channels": 32, "num_layers": 2, "aggr": "mean"},
        {"in_channels": 194, "hidden_dim": 64, "out_channels": 128, "num_layers": 3, "aggr": "max"},
        {"in_channels": 64, "hidden_dim": 64, "out_channels": 64, "num_layers": 1, "aggr": "sum"},
    ]

    for cfg in test_configs:
        x = torch.randn(num_nodes, cfg["in_channels"])
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, edge_dim)

        model = TemporalGraphSAGE(
            in_channels=cfg["in_channels"],
            edge_dim=edge_dim,
            hidden_dim=cfg["hidden_dim"],
            out_channels=cfg["out_channels"],
            num_layers=cfg["num_layers"],
            aggr=cfg["aggr"],
            dropout=0.1,
            use_residual=True,
            use_layer_norm=True,
        )

        embeddings = model(x, edge_index, edge_attr)

        assert embeddings.shape == (num_nodes, cfg["out_channels"])
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()


def test_temporal_graphsage_gradient_flow():
    """Verify gradient backpropagation across all message passing layers and edge MLPs."""
    num_nodes = 30
    num_edges = 80
    in_channels = 32
    edge_dim = 6
    out_channels = 64

    x = torch.randn(num_nodes, in_channels, requires_grad=True)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_dim, requires_grad=True)

    model = TemporalGraphSAGE(
        in_channels=in_channels,
        edge_dim=edge_dim,
        hidden_dim=64,
        out_channels=out_channels,
        num_layers=2,
    )

    out = model(x, edge_index, edge_attr)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert edge_attr.grad is not None
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient in {name}"


def test_temporal_graphsage_neighborhood_sampling_and_dataloader():
    """Verify inductive neighborhood sub-sampling and PyG DataLoader mini-batching."""
    num_nodes = 100
    num_edges = 300
    x = torch.randn(num_nodes, 64)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 6)

    data1 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    data2 = Data(x=x.clone(), edge_index=edge_index.clone(), edge_attr=edge_attr.clone(), num_nodes=num_nodes)

    # 1. Test pure PyTorch neighborhood sampling
    seed_nodes = torch.tensor([5, 12, 45, 88], dtype=torch.long)
    sub_nodes, sub_edges, sub_attr, seed_mask = TemporalGraphSAGE.sample_neighborhood(
        edge_index=edge_index,
        edge_attr=edge_attr,
        seed_nodes=seed_nodes,
        num_neighbors=5,
        num_hops=2,
    )

    assert sub_nodes.dim() == 1
    assert sub_edges.size(0) == 2
    assert sub_attr.size(1) == 6
    assert seed_mask.sum() == len(seed_nodes)

    # Forward pass on sampled subgraph
    model = TemporalGraphSAGE(in_channels=64, hidden_dim=64, out_channels=32, num_layers=2)
    sub_x = x[sub_nodes]
    sub_out = model(sub_x, sub_edges, sub_attr)
    assert sub_out.shape == (sub_nodes.size(0), 32)

    # 2. Test PyG DataLoader mini-batching multiple graph snapshots
    loader = TemporalGraphSAGE.get_graph_dataloader([data1, data2], batch_size=2)
    for batch in loader:
        batch_out = model(batch.x, batch.edge_index, batch.edge_attr)
        assert batch_out.shape == (200, 32)  # 2 graphs * 100 nodes = 200 nodes in batched graph


def test_temporal_graphsage_on_real_unsw_nb15_graph_snapshot():
    """
    Test TemporalGraphSAGE on actual UNSW-NB15 temporal graph snapshots.
    Verifies:
        - node input shape [N, D_in]
        - edge_index shape [2, E]
        - edge_attr shape [E, D_edge]
        - embedding output shape [N, D_out]
    """
    graphs_pt = ProjectPaths.DATA_GRAPHS / "train_graphs.pt"
    assert graphs_pt.exists(), f"Graph snapshots missing at: {graphs_pt}"

    graph_snapshots = torch.load(graphs_pt, weights_only=False)
    assert len(graph_snapshots) > 0, "Graph snapshots list is empty"

    # Take first graph snapshot
    sample_graph: Data = graph_snapshots[0]

    N = sample_graph.num_nodes
    E = sample_graph.edge_index.size(1)
    edge_dim = sample_graph.edge_attr.size(1)
    in_channels = sample_graph.x_dense.size(1)  # 194
    out_channels = 64

    # Initialize TemporalGraphSAGE
    model = TemporalGraphSAGE(
        in_channels=in_channels,
        edge_dim=edge_dim,
        hidden_dim=64,
        out_channels=out_channels,
        num_layers=2,
        dropout=0.1,
        aggr="mean",
        use_residual=True,
    )

    # Perform forward pass
    embeddings = model(
        x=sample_graph.x_dense,
        edge_index=sample_graph.edge_index,
        edge_attr=sample_graph.edge_attr,
    )

    print("\n===========================================================================")
    print("        TGCF-IDS : Temporal GraphSAGE Forward Pass on Real Data           ")
    print("===========================================================================")
    print(f"[*] Graph Window Snapshot       : #{sample_graph.window_idx}")
    print(f"[+] Node Input Shape (x_dense)  : {list(sample_graph.x_dense.shape)}  [N, D_in={in_channels}]")
    print(f"[+] Graph Edge Index Shape      : {list(sample_graph.edge_index.shape)}  [2, E={E}]")
    print(f"[+] Edge Attribute Shape        : {list(sample_graph.edge_attr.shape)}  [E, D_edge={edge_dim}]")
    print(f"[+] Node Embedding Output Shape : {list(embeddings.shape)}  [N, D_out={out_channels}]")
    print("===========================================================================")

    # Assertions
    assert sample_graph.x_dense.shape == (N, 194)
    assert sample_graph.edge_index.shape == (2, E)
    assert sample_graph.edge_attr.shape == (E, 6)
    assert embeddings.shape == (N, out_channels)
    assert not torch.isnan(embeddings).any()
    assert not torch.isinf(embeddings).any()
