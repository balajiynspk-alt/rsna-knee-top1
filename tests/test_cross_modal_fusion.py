"""
Unit tests for CrossModalFusion.

Verifies:
1. Dynamic learned gating fusion (vector and scalar gating modes).
2. Concatenation and Addition fusion baseline strategies.
3. Dimension mapping and range constraints (gate values in [0, 1]).
4. Gradient flow to both tabular feature and graph representations.
5. Asymmetric input dimensions (e.g. feat_dim=64, graph_dim=128, fused_dim=96).
6. Integration with FeatureTransformer and TemporalGraphSAGE representations.
"""

import pytest
import torch
import torch.nn as nn

from src.models.cross_modal_fusion import (
    CrossModalFusion,
    FusionOutput,
    GatedFusion,
    ConcatFusion,
    AdditionFusion,
)
from src.models.feature_tokenizer import FeatureTokenizer
from src.models.feature_transformer import FeatureTransformer, TabularFeatureTransformer
from src.models.temporal_graphsage import TemporalGraphSAGE


def test_gated_fusion_vector_and_scalar_modes():
    """Verify learned gating fusion with vector-wise and scalar-wise gating."""
    N = 32
    feat_dim = 64
    graph_dim = 64
    fused_dim = 64

    h_feat = torch.randn(N, feat_dim)
    h_graph = torch.randn(N, graph_dim)

    # 1. Vector gating mode
    fusion_vec = CrossModalFusion(
        feat_dim=feat_dim,
        graph_dim=graph_dim,
        fused_dim=fused_dim,
        strategy="gated",
        gate_type="vector",
    )
    out_vec = fusion_vec(h_feat, h_graph)

    assert isinstance(out_vec, FusionOutput)
    assert out_vec.fused_repr.shape == (N, fused_dim)
    assert out_vec.feature_contrib.shape == (N, fused_dim)
    assert out_vec.graph_contrib.shape == (N, fused_dim)
    assert out_vec.gate_values.shape == (N, fused_dim)
    assert (out_vec.gate_values >= 0.0).all() and (out_vec.gate_values <= 1.0).all()
    assert out_vec.strategy == "gated"

    # 2. Scalar gating mode
    fusion_scal = CrossModalFusion(
        feat_dim=feat_dim,
        graph_dim=graph_dim,
        fused_dim=fused_dim,
        strategy="gated",
        gate_type="scalar",
    )
    out_scal = fusion_scal(h_feat, h_graph)

    assert out_scal.fused_repr.shape == (N, fused_dim)
    assert out_scal.gate_values.shape == (N, 1)
    assert (out_scal.gate_values >= 0.0).all() and (out_scal.gate_values <= 1.0).all()


def test_concat_and_addition_fusion_strategies():
    """Verify concatenation and element-wise addition fusion strategies."""
    N = 16
    feat_dim = 48
    graph_dim = 32
    fused_dim = 64

    h_feat = torch.randn(N, feat_dim)
    h_graph = torch.randn(N, graph_dim)

    # 1. Concat
    fusion_concat = CrossModalFusion(
        feat_dim=feat_dim,
        graph_dim=graph_dim,
        fused_dim=fused_dim,
        strategy="concat",
    )
    out_c = fusion_concat(h_feat, h_graph)
    assert out_c.fused_repr.shape == (N, fused_dim)
    assert out_c.strategy == "concat"

    # 2. Add
    fusion_add = CrossModalFusion(
        feat_dim=feat_dim,
        graph_dim=graph_dim,
        fused_dim=fused_dim,
        strategy="add",
    )
    out_a = fusion_add(h_feat, h_graph)
    assert out_a.fused_repr.shape == (N, fused_dim)
    assert out_a.strategy == "add"


def test_cross_modal_fusion_gradient_backprop():
    """Verify full gradient propagation back to both modalities through gating layers."""
    N = 20
    feat_dim = 64
    graph_dim = 64
    fused_dim = 32

    h_feat = torch.randn(N, feat_dim, requires_grad=True)
    h_graph = torch.randn(N, graph_dim, requires_grad=True)

    fusion = CrossModalFusion(
        feat_dim=feat_dim,
        graph_dim=graph_dim,
        fused_dim=fused_dim,
        strategy="gated",
    )

    out = fusion(h_feat, h_graph)
    loss = out.fused_repr.sum()
    loss.backward()

    assert h_feat.grad is not None
    assert h_graph.grad is not None
    for name, param in fusion.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient in {name}"


def test_asymmetric_dimensions_and_error_handling():
    """Verify handling of asymmetric input dimensions and exception raising."""
    fusion = CrossModalFusion(feat_dim=50, graph_dim=100, fused_dim=75, strategy="gated")

    h_feat = torch.randn(10, 50)
    h_graph = torch.randn(10, 100)
    out = fusion(h_feat, h_graph)
    assert out.fused_repr.shape == (10, 75)

    # Row count mismatch
    with pytest.raises(ValueError, match="Sample count mismatch"):
        fusion(torch.randn(10, 50), torch.randn(12, 100))

    # Feature dim mismatch
    with pytest.raises(ValueError, match="Expected h_feat dim 50"):
        fusion(torch.randn(10, 40), torch.randn(10, 100))

    # Invalid strategy
    with pytest.raises(ValueError, match="Unknown fusion strategy"):
        CrossModalFusion(strategy="invalid_fusion")


def test_full_cross_modal_pipeline_integration():
    """
    Verify full integration combining:
    1. Tabular Feature Transformer -> h_feat [N, D]
    2. Temporal GraphSAGE -> h_graph [N, D]
    3. CrossModalFusion -> h_fused [N, D_fused]
    """
    N = 64
    D = 64
    D_fused = 128

    # 1. Feature Transformer
    tokenizer = FeatureTokenizer(num_numerical=10, cat_cardinalities=[5, 8], embedding_dim=D)
    transformer = FeatureTransformer(embedding_dim=D, num_heads=4, num_layers=2)
    feat_model = TabularFeatureTransformer(tokenizer, transformer)

    x_num = torch.randn(N, 10)
    x_cat = torch.tensor([[1, 2]] * N, dtype=torch.long)
    feat_out = feat_model(x_num=x_num, x_cat=x_cat)
    h_feat = feat_out.flow_repr  # [N, D]

    # 2. GraphSAGE
    edge_index = torch.randint(0, N, (2, 200))
    edge_attr = torch.randn(200, 6)
    x_dense = torch.randn(N, 194)
    gnn = TemporalGraphSAGE(in_channels=194, edge_dim=6, hidden_dim=D, out_channels=D, num_layers=2)
    h_graph = gnn(x_dense, edge_index, edge_attr)  # [N, D]

    # 3. CrossModalFusion
    fusion = CrossModalFusion(feat_dim=D, graph_dim=D, fused_dim=D_fused, strategy="gated")
    fusion_out = fusion(h_feat=h_feat, h_graph=h_graph)

    print("\n===========================================================================")
    print("      TGCF-IDS : Cross-Modal Fusion Pipeline Dimension Verification        ")
    print("===========================================================================")
    print(f"[*] Input Batch Size            : {N}")
    print(f"[+] Feature Transformer Repr    : {list(h_feat.shape)}  [N, D_feat={D}]")
    print(f"[+] Temporal GraphSAGE Repr     : {list(h_graph.shape)}  [N, D_graph={D}]")
    print(f"[+] Dynamic Gate Values         : {list(fusion_out.gate_values.shape)}  [N, D_fused={D_fused}]")
    print(f"[+] Feature Modality Contrib    : {list(fusion_out.feature_contrib.shape)}  [N, D_fused={D_fused}]")
    print(f"[+] Graph Modality Contrib      : {list(fusion_out.graph_contrib.shape)}  [N, D_fused={D_fused}]")
    print(f"[+] Final Fused Representation  : {list(fusion_out.fused_repr.shape)}  [N, D_fused={D_fused}]")
    print("===========================================================================")

    assert fusion_out.fused_repr.shape == (N, D_fused)
    assert not torch.isnan(fusion_out.fused_repr).any()
    assert not torch.isinf(fusion_out.fused_repr).any()
