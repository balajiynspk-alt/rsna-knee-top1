"""
Unit tests for Self-Supervised Graph Contrastive Pretraining.

Verifies:
1. ContrastiveProjectionHead normalized embedding output.
2. GraphAugmentor stochastic multi-view generation (feature masking, edge dropout, noise perturbation, temporal preservation).
3. NTXentLoss (InfoNCE) loss computation properties.
4. End-to-end ContrastiveTrainer training loop and checkpoint generation on isolated synthetic graph data.
"""

from pathlib import Path
import pytest
import numpy as np
import torch
from torch_geometric.data import Data

from src.models.projection_head import ContrastiveProjectionHead
from src.models.graph_augmentations import GraphAugmentor
from src.training.contrastive import NTXentLoss, GraphContrastiveModel, ContrastiveTrainer


@pytest.fixture
def sample_graph():
    """Create reproducible synthetic PyG graph for testing."""
    torch.manual_seed(42)
    num_nodes = 50
    num_edges = 120
    x_dense = torch.randn(num_nodes, 194)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Construct 6D edge attributes within valid domains
    temp_dist = torch.rand(num_edges, 1)
    src_rel = torch.rand(num_edges, 1)
    dst_rel = torch.rand(num_edges, 1)
    proto_sim = torch.randint(0, 2, (num_edges, 1)).float()
    port_rel = torch.randint(0, 2, (num_edges, 1)).float()
    feat_sim = (torch.rand(num_edges, 1) * 2) - 1.0  # [-1, 1]

    edge_attr = torch.cat([temp_dist, src_rel, dst_rel, proto_sim, port_rel, feat_sim], dim=1)
    y_multi = torch.randint(0, 10, (num_nodes,))
    y_bin = (y_multi > 0).long()

    return Data(
        x=x_dense,
        x_dense=x_dense,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_multiclass=y_multi,
        y_binary=y_bin,
        num_nodes=num_nodes,
        window_idx=0,
    )


def test_contrastive_projection_head_dimensions_and_norm():
    """Verify projection head maps representations to L2 unit-sphere vectors."""
    head = ContrastiveProjectionHead(in_dim=64, hidden_dim=128, out_dim=32, dropout=0.1)

    h = torch.randn(20, 64)
    z = head(h, normalize=True)

    assert z.shape == (20, 32)
    # Check L2 unit norm
    norms = torch.norm(z, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_graph_augmentations(sample_graph):
    """Verify stochastic multi-view generation preserves temporal bounds and drops edges."""
    augmentor = GraphAugmentor(
        feature_mask_ratio=0.2,
        feature_noise_std=0.05,
        edge_drop_ratio=0.2,
        temporal_jitter_std=0.02,
        random_seed=42,
    )

    g1, g2 = augmentor.create_augmented_pair(sample_graph)

    # 1. Feature perturbation
    assert not torch.equal(g1.x_dense, sample_graph.x_dense)
    assert not torch.equal(g1.x_dense, g2.x_dense)

    # 2. Edge dropout (dropped ~20% of edges)
    assert g1.edge_index.size(1) < sample_graph.edge_index.size(1)
    assert g2.edge_index.size(1) < sample_graph.edge_index.size(1)

    # 3. Temporal attribute domain bounds preserved
    assert (g1.edge_attr[:, 0] >= 0.0).all() and (g1.edge_attr[:, 0] <= 1.0).all()
    assert (g1.edge_attr[:, 5] >= -1.0).all() and (g1.edge_attr[:, 5] <= 1.0).all()


def test_nt_xent_loss_properties():
    """Verify NTXentLoss properties with identical vs orthogonal embeddings."""
    loss_fn = NTXentLoss(temperature=0.2)

    N = 30
    D = 64

    # Perfect alignment (z1 == z2)
    z_base = torch.randn(N, D)
    loss_identical = loss_fn(z_base, z_base)

    # Randomized alignment
    z_rand = torch.randn(N, D)
    loss_random = loss_fn(z_base, z_rand)

    # Loss on random embeddings should be substantially higher than identical
    assert loss_random.item() > loss_identical.item()
    assert loss_identical.item() >= 0.0


def test_contrastive_trainer_fast_isolated_pipeline(tmp_path: Path, sample_graph):
    """Verify full contrastive pretraining loop and checkpoint creation on isolated synthetic graphs."""
    graphs_dir = tmp_path / "graphs"
    results_dir = tmp_path / "results"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save small synthetic graph list
    graph_list = [sample_graph, sample_graph.clone()]
    torch.save(graph_list, graphs_dir / "train_graphs.pt")
    torch.save(graph_list, graphs_dir / "test_graphs.pt")

    trainer = ContrastiveTrainer(
        in_channels=194,
        edge_dim=6,
        hidden_dim=32,
        proj_dim=32,
        num_layers=1,
        temperature=0.2,
        learning_rate=1e-3,
        batch_size=2,
        epochs=3,
        seed=42,
        device="cpu",
        graphs_dir=graphs_dir,
        results_dir=results_dir,
    )

    res = trainer.train_pipeline()

    assert "model" in res
    assert "best_loss" in res
    assert res["checkpoint_path"].exists()
    assert len(res["history"]["contrastive_loss"]) == 3
