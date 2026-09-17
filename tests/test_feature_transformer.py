"""
Unit tests for FeatureTransformer and TabularFeatureTransformer.

Verifies:
1. Multi-head self-attention and Pre-LN Transformer encoder layers.
2. Configurable embedding dimensions, number of heads, layer depths, and FFN dimensions.
3. Multiple pooling mechanisms ('mean', 'cls', 'max', 'attention').
4. Attention weights extraction and dimension verification [B, H, F, F].
5. End-to-end integration with FeatureTokenizer.
6. Forward pass and dimension verification on real preprocessed UNSW-NB15 samples.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.models.feature_tokenizer import FeatureTokenizer
from src.models.feature_transformer import (
    FeatureTransformer,
    FeatureTransformerOutput,
    TabularFeatureTransformer,
)
from src.utils.paths import ProjectPaths


def test_feature_transformer_synthetic_dimensions_and_pooling():
    """Verify flow representation [B, D] and token representation [B, F, D] across pooling methods."""
    batch_size = 16
    num_features = 42
    embedding_dim = 64
    num_heads = 4
    num_layers = 3

    pooling_methods = ["mean", "cls", "max", "attention"]

    tokens = torch.randn(batch_size, num_features, embedding_dim)

    for pool in pooling_methods:
        transformer = FeatureTransformer(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=128,
            dropout=0.1,
            pooling=pool,
        )

        output = transformer(tokens, return_attention=False)

        assert isinstance(output, FeatureTransformerOutput)
        assert output.flow_repr.shape == (batch_size, embedding_dim), (
            f"Pooling {pool}: Expected flow_repr {(batch_size, embedding_dim)}, got {output.flow_repr.shape}"
        )
        assert output.token_repr.shape == (batch_size, num_features, embedding_dim)
        assert output.attention_weights is None
        assert not torch.isnan(output.flow_repr).any()
        assert not torch.isinf(output.flow_repr).any()


def test_feature_transformer_attention_weights_extraction():
    """Verify that multi-head attention weights are returned per layer with shape [B, H, F, F]."""
    batch_size = 8
    num_features = 20
    embedding_dim = 32
    num_heads = 4
    num_layers = 2

    transformer = FeatureTransformer(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        pooling="mean",
    )

    tokens = torch.randn(batch_size, num_features, embedding_dim)
    transformer.eval()
    with torch.no_grad():
        output = transformer(tokens, return_attention=True)

    assert output.attention_weights is not None
    assert len(output.attention_weights) == num_layers

    for layer_idx, attn_map in enumerate(output.attention_weights):
        # Attention map shape: [B, num_heads, num_features, num_features]
        assert attn_map.shape == (batch_size, num_heads, num_features, num_features), (
            f"Layer {layer_idx} attention shape mismatch: {attn_map.shape}"
        )
        # In eval mode without dropout, softmax rows should sum to ~1.0
        row_sums = attn_map.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_feature_transformer_gradient_backprop():
    """Verify full gradient propagation through all Transformer layers and pooling heads."""
    embedding_dim = 32
    transformer = FeatureTransformer(
        embedding_dim=embedding_dim,
        num_heads=4,
        num_layers=2,
        pooling="attention",
    )

    tokens = torch.randn(4, 15, embedding_dim, requires_grad=True)
    output = transformer(tokens)

    loss = output.flow_repr.sum() + output.token_repr.sum()
    loss.backward()

    assert tokens.grad is not None
    for param in transformer.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_feature_transformer_configuration_errors():
    """Verify validation of invalid hyperparameters."""
    # embedding_dim not divisible by num_heads
    with pytest.raises(ValueError, match="must be divisible by num_heads"):
        FeatureTransformer(embedding_dim=65, num_heads=4)

    # Invalid pooling option
    with pytest.raises(ValueError, match="Unknown pooling method"):
        FeatureTransformer(embedding_dim=64, num_heads=4, pooling="invalid_pool")

    # Mismatched token dim in forward
    transformer = FeatureTransformer(embedding_dim=64, num_heads=4)
    with pytest.raises(ValueError, match="does not match Transformer embedding_dim"):
        transformer(torch.randn(4, 10, 32))


def test_end_to_end_tabular_feature_transformer_pipeline():
    """Verify combined Tokenizer + Transformer pipeline."""
    tokenizer = FeatureTokenizer(
        num_numerical=10,
        cat_cardinalities=[5, 8],
        embedding_dim=32,
    )
    transformer = FeatureTransformer(
        embedding_dim=32,
        num_heads=4,
        num_layers=2,
        pooling="mean",
    )
    model = TabularFeatureTransformer(tokenizer=tokenizer, transformer=transformer)

    x_num = torch.randn(8, 10)
    x_cat = torch.tensor([[1, 2]] * 8, dtype=torch.long)

    output = model(x_num=x_num, x_cat=x_cat, return_attention=True)
    assert output.flow_repr.shape == (8, 32)
    assert output.token_repr.shape == (8, 12, 32)
    assert len(output.attention_weights) == 2


def test_feature_transformer_on_actual_unsw_nb15_data():
    """
    Run forward pass on real preprocessed UNSW-NB15 samples and verify all tensor dimensions.
    """
    meta_path = ProjectPaths.DATA_PROCESSED / "metadata.json"
    train_pt = ProjectPaths.DATA_PROCESSED / "train_features.pt"

    assert meta_path.exists(), f"Metadata missing at {meta_path}"
    assert train_pt.exists(), f"Train bundle missing at {train_pt}"

    embedding_dim = 64
    num_heads = 4
    num_layers = 2

    # Construct tokenizer from official metadata
    tokenizer = FeatureTokenizer.from_metadata_file(meta_path, embedding_dim=embedding_dim)
    transformer = FeatureTransformer(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=256,
        dropout=0.1,
        pooling="mean",
    )

    end_to_end_model = TabularFeatureTransformer(tokenizer, transformer)

    # Load real batch
    data_bundle = torch.load(train_pt, weights_only=True)
    batch_size = 256
    x_num = data_bundle["x_num"][:batch_size]  # [256, 39]
    x_cat = data_bundle["x_cat"][:batch_size]  # [256, 3]

    output = end_to_end_model(x_num=x_num, x_cat=x_cat, return_attention=True)

    print("\n===========================================================================")
    print("      TGCF-IDS : Feature Transformer Forward Pass on UNSW-NB15 Data       ")
    print("===========================================================================")
    print(f"[*] Input Batch Size            : {batch_size}")
    print(f"[*] Raw Features (Numerical)    : {list(x_num.shape)}  (39 continuous metrics)")
    print(f"[*] Raw Features (Categorical)  : {list(x_cat.shape)}  (3 ordinal categories)")
    print(f"[*] Tokenized Representation    : [{batch_size}, {tokenizer.num_features}, {embedding_dim}]")
    print(f"[+] Contextualized Token Repr   : {list(output.token_repr.shape)}  [B, F, D]")
    print(f"[+] Pooled Flow Representation  : {list(output.flow_repr.shape)}   [B, D]")
    print(f"[+] Multi-Head Attention Maps   : {len(output.attention_weights)} layers x {list(output.attention_weights[0].shape)} [B, H, F, F]")
    print("===========================================================================")

    assert output.token_repr.shape == (batch_size, 42, embedding_dim)
    assert output.flow_repr.shape == (batch_size, embedding_dim)
    assert len(output.attention_weights) == num_layers
    assert output.attention_weights[0].shape == (batch_size, num_heads, 42, 42)
    assert not torch.isnan(output.flow_repr).any()
    assert not torch.isinf(output.flow_repr).any()
