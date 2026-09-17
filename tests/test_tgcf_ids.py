"""
Integration tests for the complete TGCF-IDS architecture.

Verifies:
1. End-to-end forward pass integrating FeatureTokenizer, FeatureTransformer, TemporalGraphSAGE, CrossModalFusion, and IntrusionClassifier.
2. Tensor output shapes:
   - logits: [N, 10]
   - feature_repr: [N, D_feat]
   - graph_repr: [N, D_graph]
   - fused_repr: [N, D_fused]
   - gate_values: [N, D_fused]
3. Model parameter breakdown and summary calculations.
4. Pretrained contrastive GraphSAGE weight loading and parameter freezing.
5. End-to-end forward pass and backward gradient flow on real UNSW-NB15 graph snapshots.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.models.tgcf_ids import TGCFIDS, TGCFIDSOutput
from src.utils.paths import ProjectPaths


@pytest.fixture
def synthetic_graph_snapshot():
    """Create a minimal synthetic graph snapshot."""
    torch.manual_seed(42)
    N = 50
    E = 120

    x_num = torch.randn(N, 39)
    x_cat = torch.tensor([[1, 2, 3]] * N, dtype=torch.long)
    x_dense = torch.randn(N, 194)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, 6)
    y_multi = torch.randint(0, 10, (N,))
    y_bin = (y_multi > 0).long()

    return Data(
        x=x_dense,
        x_num=x_num,
        x_cat=x_cat,
        x_dense=x_dense,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_multiclass=y_multi,
        y_binary=y_bin,
        num_nodes=N,
        window_idx=0,
    )


def test_tgcf_ids_forward_pass_and_shapes(synthetic_graph_snapshot):
    """Verify forward pass returns all required representations with correct tensor dimensions."""
    data = synthetic_graph_snapshot
    N = data.num_nodes

    model = TGCFIDS(
        num_numerical=39,
        cat_cardinalities=[10, 10, 10],
        token_dim=64,
        transformer_heads=4,
        transformer_layers=2,
        graph_in_channels=194,
        graph_edge_dim=6,
        graph_hidden_dim=64,
        graph_out_channels=64,
        graph_layers=2,
        fusion_dim=128,
        fusion_strategy="gated",
        classifier_hidden_dim=128,
        num_classes=10,
    )

    out = model(data=data, return_attention=True)

    assert isinstance(out, TGCFIDSOutput)
    assert out.logits.shape == (N, 10)
    assert out.feature_repr.shape == (N, 64)
    assert out.graph_repr.shape == (N, 64)
    assert out.fused_repr.shape == (N, 128)
    assert out.gate_values.shape == (N, 128)
    assert out.feature_contrib.shape == (N, 128)
    assert out.graph_contrib.shape == (N, 128)
    assert out.attention_weights is not None
    assert len(out.attention_weights) == 2

    assert not torch.isnan(out.logits).any()
    assert not torch.isinf(out.logits).any()


def test_tgcf_ids_parameter_summary():
    """Verify model parameter accounting."""
    model = TGCFIDS(
        num_numerical=39,
        cat_cardinalities=[134, 14, 10],
        token_dim=64,
        graph_in_channels=194,
        fusion_dim=128,
    )

    summary = model.get_model_summary()
    assert "total_parameters" in summary
    assert "trainable_parameters" in summary
    assert summary["total_parameters"] > 50_000
    assert summary["trainable_parameters"] == summary["total_parameters"]
    assert "FeatureTokenizer" in summary["submodule_parameters"]
    assert "FeatureTransformer" in summary["submodule_parameters"]
    assert "TemporalGraphSAGE" in summary["submodule_parameters"]
    assert "CrossModalFusion" in summary["submodule_parameters"]
    assert "IntrusionClassifier" in summary["submodule_parameters"]


def test_tgcf_ids_gradient_backprop(synthetic_graph_snapshot):
    """Verify gradient backpropagation from 10-class CrossEntropyLoss through entire architecture."""
    data = synthetic_graph_snapshot
    model = TGCFIDS(
        num_numerical=39,
        cat_cardinalities=[10, 10, 10],
        token_dim=32,
        graph_in_channels=194,
        fusion_dim=64,
    )

    out = model(data=data)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out.logits, data.y_multiclass)
    loss.backward()

    # Verify gradients reach all sub-networks
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients!"


def test_tgcf_ids_load_pretrained_checkpoint():
    """Verify loading contrastively pretrained weights from results/checkpoints/graph_pretrained.pt."""
    ckpt_path = ProjectPaths.RESULTS_CHECKPOINTS / "graph_pretrained.pt"
    if not ckpt_path.exists():
        pytest.skip(f"Pretrained checkpoint not present at {ckpt_path}")

    model = TGCFIDS(
        num_numerical=39,
        cat_cardinalities=[134, 14, 10],
        token_dim=64,
        graph_in_channels=194,
        graph_hidden_dim=64,
        graph_out_channels=64,
        graph_layers=2,
        fusion_dim=128,
    )

    model.load_pretrained_graph_encoder(ckpt_path, freeze=True)

    # Check that graph encoder parameters are frozen
    for param in model.graph_encoder.parameters():
        assert not param.requires_grad

    # Check that other parameters remain trainable
    for param in model.transformer.parameters():
        assert param.requires_grad
    for param in model.classifier.parameters():
        assert param.requires_grad


def test_tgcf_ids_forward_pass_on_real_unsw_nb15_sample():
    """
    Execute forward pass on real preprocessed UNSW-NB15 temporal graph snapshot.
    Prints model summary, tensor shapes, and verifies outputs.
    """
    meta_path = ProjectPaths.DATA_PROCESSED / "metadata.json"
    graphs_path = ProjectPaths.DATA_GRAPHS / "train_graphs.pt"

    assert meta_path.exists(), f"Metadata missing at {meta_path}"
    assert graphs_path.exists(), f"Graphs missing at {graphs_path}"

    # Construct model using metadata factory
    model = TGCFIDS.from_metadata_file(
        metadata_path=meta_path,
        token_dim=64,
        transformer_heads=4,
        transformer_layers=2,
        graph_hidden_dim=64,
        graph_out_channels=64,
        graph_layers=2,
        fusion_dim=128,
        fusion_strategy="gated",
    )

    # Load first real graph window
    train_graphs = torch.load(graphs_path, weights_only=False)
    sample_window: Data = train_graphs[0]
    N = sample_window.num_nodes  # 1000

    # Print model summary
    model.print_summary()

    # Forward pass
    out = model(data=sample_window, return_attention=True)

    print("\n===========================================================================")
    print("           TGCF-IDS : Full Architecture Real Sample Forward Pass           ")
    print("===========================================================================")
    print(f"[*] Input Graph Snapshot        : Window #{sample_window.window_idx}")
    print(f"[*] Number of Network Flows (N) : {N:,}")
    print(f"[+] 1. Feature Tokens [N, F, D] : [{N}, 42, 64]")
    print(f"[+] 2. Feature Repr (h_feat)    : {list(out.feature_repr.shape)}  [N, D_feat=64]")
    print(f"[+] 3. Graph Repr (h_graph)     : {list(out.graph_repr.shape)}  [N, D_graph=64]")
    print(f"[+] 4. Dynamic Gating (gate_val): {list(out.gate_values.shape)}  [N, D_fused=128]")
    print(f"[+] 5. Fused Repr (h_fused)     : {list(out.fused_repr.shape)}  [N, D_fused=128]")
    print(f"[+] 6. Classifier Logits        : {list(out.logits.shape)}  [N, Classes=10]")
    print("===========================================================================")

    assert out.logits.shape == (N, 10)
    assert out.feature_repr.shape == (N, 64)
    assert out.graph_repr.shape == (N, 64)
    assert out.fused_repr.shape == (N, 128)
    assert not torch.isnan(out.logits).any()
    assert not torch.isinf(out.logits).any()
