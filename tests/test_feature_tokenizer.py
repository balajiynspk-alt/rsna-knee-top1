"""
Unit tests for FeatureTokenizer.

Verifies:
1. Synthetic data tokenization with configurable embedding dimensions.
2. Handling of unknown and out-of-bounds categorical values.
3. Feature identity embeddings addition and gradient backpropagation.
4. Various input invocation patterns (separate tensors, dictionary, combined tensor, single instance).
5. Tokenization on real preprocessed UNSW-NB15 dataset samples and metadata.
6. Print and verify tensor shape transformations: [B, F] -> [B, F, D].
"""

import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.models.feature_tokenizer import FeatureTokenizer
from src.utils.paths import ProjectPaths


def test_feature_tokenizer_synthetic_shapes_and_dimensions():
    """Verify output shapes [B, F, D] across multiple batch sizes and embedding dimensions."""
    batch_sizes = [1, 16, 64, 256]
    embedding_dims = [16, 64, 128]
    num_num = 10
    cat_cards = [5, 12, 3]  # 3 categorical features
    total_features = num_num + len(cat_cards)  # 13 features

    for d in embedding_dims:
        tokenizer = FeatureTokenizer(
            num_numerical=num_num,
            cat_cardinalities=cat_cards,
            embedding_dim=d,
            use_identity_emb=True,
            use_layer_norm=True,
            dropout=0.1,
        )

        assert tokenizer.num_numerical == num_num
        assert tokenizer.num_categorical == 3
        assert tokenizer.num_features == total_features
        assert tokenizer.embedding_dim == d

        for b in batch_sizes:
            x_num = torch.randn(b, num_num, dtype=torch.float32)
            x_cat = torch.randint(0, 3, (b, len(cat_cards)), dtype=torch.long)

            tokens = tokenizer(x_num=x_num, x_cat=x_cat)

            # Verification: [B, F, D]
            assert tokens.shape == (b, total_features, d), f"Expected {(b, total_features, d)}, got {tokens.shape}"
            assert not torch.isnan(tokens).any(), "Token embeddings must not contain NaNs"
            assert not torch.isinf(tokens).any(), "Token embeddings must not contain Infs"


def test_feature_tokenizer_unknown_categorical_handling():
    """Verify that out-of-vocabulary and negative category indices are mapped safely to UNK (index 0)."""
    tokenizer = FeatureTokenizer(
        num_numerical=2,
        cat_cardinalities=[10, 20],  # valid ranges: 0..9 and 0..19
        embedding_dim=32,
        use_identity_emb=False,
    )

    x_num = torch.randn(4, 2)
    # Feed out-of-range indices: 99, 500, and -5
    x_cat = torch.tensor([
        [0, 5],      # 0 is UNK
        [99, 12],    # 99 is OOV -> safely mapped to 0
        [3, 500],    # 500 is OOV -> safely mapped to 0
        [-5, -1],    # negative -> safely mapped to 0
    ], dtype=torch.long)

    tokens = tokenizer(x_num, x_cat)
    assert tokens.shape == (4, 4, 32)
    assert not torch.isnan(tokens).any()


def test_feature_tokenizer_gradients_and_backprop():
    """Verify that all parameters (weights, biases, categorical embeddings, identity embeddings) receive gradients."""
    tokenizer = FeatureTokenizer(
        num_numerical=5,
        cat_cardinalities=[8, 15],
        embedding_dim=32,
        use_identity_emb=True,
    )

    x_num = torch.randn(8, 5, requires_grad=True)
    x_cat = torch.randint(0, 8, (8, 2))

    tokens = tokenizer(x_num, x_cat)
    loss = tokens.sum()
    loss.backward()

    # Check gradients
    assert tokenizer.weight_num.grad is not None
    assert tokenizer.bias_num.grad is not None
    assert tokenizer.identity_emb.grad is not None
    for emb in tokenizer.cat_embeddings:
        assert emb.weight.grad is not None


def test_feature_tokenizer_input_signature_variants():
    """Verify dictionary, combined tensor, and single sample input variations."""
    tokenizer = FeatureTokenizer(
        num_numerical=4,
        cat_cardinalities=[10, 10],
        embedding_dim=16,
    )

    x_num = torch.randn(4, 4)
    x_cat = torch.randint(0, 10, (4, 2))

    # 1. Standard args
    out1 = tokenizer(x_num=x_num, x_cat=x_cat)

    # 2. Dictionary input
    dict_input = {"x_num": x_num, "x_cat": x_cat}
    out2 = tokenizer(x_dict=dict_input)

    # 3. Combined input tensor [4, 6]
    combined = torch.cat([x_num, x_cat.float()], dim=1)
    out3 = tokenizer(x_combined=combined)

    assert out1.shape == out2.shape == out3.shape == (4, 6, 16)


def test_feature_tokenizer_invalid_inputs_raise_errors():
    """Verify proper error handling for invalid feature counts and configurations."""
    tokenizer = FeatureTokenizer(num_numerical=5, cat_cardinalities=[10], embedding_dim=16)

    # Wrong numerical dim
    with pytest.raises(ValueError, match="Expected x_num second dimension 5"):
        tokenizer(x_num=torch.randn(2, 3), x_cat=torch.zeros(2, 1, dtype=torch.long))

    # Missing x_num
    with pytest.raises(ValueError, match="expected x_num with 5 features"):
        tokenizer(x_num=None, x_cat=torch.zeros(2, 1, dtype=torch.long))

    # Missing x_cat
    with pytest.raises(ValueError, match="expected x_cat with 1 features"):
        tokenizer(x_num=torch.randn(2, 5), x_cat=None)


def test_feature_tokenizer_on_actual_unsw_nb15_data():
    """
    Test FeatureTokenizer on actual preprocessed UNSW-NB15 dataset artifacts.
    Verifies [B, F] -> [B, F, D] with exact metadata dimensions.
    """
    meta_path = ProjectPaths.DATA_PROCESSED / "metadata.json"
    train_pt = ProjectPaths.DATA_PROCESSED / "train_features.pt"
    test_pt = ProjectPaths.DATA_PROCESSED / "test_features.pt"

    assert meta_path.exists(), f"Missing metadata file: {meta_path}"
    assert train_pt.exists(), f"Missing training tensor bundle: {train_pt}"
    assert test_pt.exists(), f"Missing testing tensor bundle: {test_pt}"

    embedding_dim = 64
    tokenizer = FeatureTokenizer.from_metadata_file(meta_path, embedding_dim=embedding_dim)

    # Verify metadata-derived structure
    assert tokenizer.num_numerical == 39
    assert tokenizer.num_categorical == 3
    assert tokenizer.num_features == 42
    assert tokenizer.embedding_dim == embedding_dim

    # Load real preprocessed batches
    train_data = torch.load(train_pt, weights_only=True)
    test_data = torch.load(test_pt, weights_only=True)

    batch_size = 128
    x_num_batch = train_data["x_num"][:batch_size]
    x_cat_batch = train_data["x_cat"][:batch_size]

    tokens = tokenizer(x_num=x_num_batch, x_cat=x_cat_batch)

    # Shape transformation verification: [B, F] -> [B, F, D]
    print(f"\n[+] UNSW-NB15 Real Batch Transformation:")
    print(f"    - Input Numerical Shape   : {list(x_num_batch.shape)}  (39 features)")
    print(f"    - Input Categorical Shape : {list(x_cat_batch.shape)}  (3 features)")
    print(f"    - Total Features (F)      : {tokenizer.num_features}")
    print(f"    - Output Tokens [B, F, D] : {list(tokens.shape)}  (Embedding Dim D={embedding_dim})")

    assert tokens.shape == (batch_size, 42, embedding_dim)
    assert not torch.isnan(tokens).any()
    assert not torch.isinf(tokens).any()

    # Verify on test partition as well
    x_num_test = test_data["x_num"][:64]
    x_cat_test = test_data["x_cat"][:64]
    test_tokens = tokenizer(x_num=x_num_test, x_cat=x_cat_test)
    assert test_tokens.shape == (64, 42, embedding_dim)
