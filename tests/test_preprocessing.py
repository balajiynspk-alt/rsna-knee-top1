"""
Unit tests for leakage-safe preprocessing, feature transformation, and metadata manifest.
"""

from pathlib import Path
from typing import Tuple
import json
import pytest
import numpy as np
import pandas as pd
import torch
import joblib

from src.preprocessing.preprocessor import LeakageSafePreprocessor


@pytest.fixture
def mock_dataset_splits(tmp_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create controlled train/test splits with known statistics and unseen test categories."""
    np.random.seed(42)
    n_train = 100
    n_test = 50

    numerical_cols = [
        "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
        "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
        "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
        "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src",
        "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
        "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
        "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports"
    ]

    # Distinct distributions for train and test to ensure isolation verification
    train_data = {col: np.random.normal(loc=10.0, scale=2.0, size=n_train) for col in numerical_cols}
    train_data["id"] = list(range(1, n_train + 1))
    train_data["proto"] = np.random.choice(["tcp", "udp", "icmp"], size=n_train)
    train_data["service"] = np.random.choice(["http", "dns", "none"], size=n_train)
    train_data["state"] = np.random.choice(["FIN", "INT", "CON"], size=n_train)
    train_data["attack_cat"] = np.random.choice(
        ["Normal", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms"],
        size=n_train
    )
    train_data["label"] = [0 if a == "Normal" else 1 for a in train_data["attack_cat"]]

    test_data = {col: np.random.normal(loc=50.0, scale=5.0, size=n_test) for col in numerical_cols}
    test_data["id"] = list(range(n_train + 1, n_train + n_test + 1))
    # Note: test set contains unseen categories 'sctp', 'ftp', 'CLO'
    test_data["proto"] = np.random.choice(["tcp", "udp", "sctp"], size=n_test)
    test_data["service"] = np.random.choice(["http", "dns", "ftp"], size=n_test)
    test_data["state"] = np.random.choice(["FIN", "INT", "CLO"], size=n_test)
    test_data["attack_cat"] = np.random.choice(
        ["Normal", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms"],
        size=n_test
    )
    test_data["label"] = [0 if a == "Normal" else 1 for a in test_data["attack_cat"]]

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    return train_df, test_df


def test_fit_strictly_on_training_data(mock_dataset_splits):
    """Verify that scaler parameters are derived only from training split."""
    train_df, test_df = mock_dataset_splits
    preprocessor = LeakageSafePreprocessor(scaler_type="robust")
    preprocessor.fit(train_df)

    assert preprocessor.is_fitted is True

    # Median of training numerical features
    train_medians = np.median(train_df[preprocessor.numerical_cols].to_numpy(), axis=0)
    assert np.allclose(preprocessor.scaler.center_, train_medians, atol=1e-4)

    # Must differ from test medians (which were generated with loc=50 vs loc=10)
    test_medians = np.median(test_df[preprocessor.numerical_cols].to_numpy(), axis=0)
    assert not np.allclose(preprocessor.scaler.center_, test_medians, atol=1.0)


def test_target_and_metadata_isolation(mock_dataset_splits):
    """Verify that label, attack_cat, and id are never used as model input features."""
    train_df, _ = mock_dataset_splits
    preprocessor = LeakageSafePreprocessor()
    preprocessor.fit(train_df)

    assert "label" not in preprocessor.all_feature_names
    assert "attack_cat" not in preprocessor.all_feature_names
    assert "id" not in preprocessor.all_feature_names
    assert "label" not in preprocessor.numerical_cols
    assert "id" not in preprocessor.numerical_cols


def test_class_mapping_correctness(mock_dataset_splits):
    """Verify exact 10-class integer mapping from specification."""
    train_df, _ = mock_dataset_splits
    preprocessor = LeakageSafePreprocessor()

    expected_mapping = {
        "Normal": 0, "Analysis": 1, "Backdoor": 2, "DoS": 3,
        "Exploits": 4, "Fuzzers": 5, "Generic": 6, "Reconnaissance": 7,
        "Shellcode": 8, "Worms": 9,
    }
    assert preprocessor.CLASS_MAPPING == expected_mapping

    y_multi, y_bin = preprocessor.validate_and_map_targets(train_df)
    assert len(y_multi) == len(train_df)
    assert y_multi.min() >= 0
    assert y_multi.max() <= 9
    assert set(np.unique(y_bin)).issubset({0, 1})


def test_unseen_category_handling(mock_dataset_splits):
    """Verify that unseen test categories are safely mapped to <UNK> index (0)."""
    train_df, test_df = mock_dataset_splits
    preprocessor = LeakageSafePreprocessor()
    preprocessor.fit(train_df)

    test_output = preprocessor.transform(test_df)
    x_cat_ord = test_output["numpy"]["x_cat_ord"]

    # 'state' is the 3rd categorical column (index 2)
    state_col_idx = preprocessor.categorical_cols.index("state")
    clo_mask = test_df["state"] == "CLO"
    assert clo_mask.any()

    # All 'CLO' rows must be assigned UNK_INDEX (0)
    assert (x_cat_ord[clo_mask, state_col_idx] == 0).all()


def test_leakage_audit_and_artifact_saving(mock_dataset_splits, tmp_path):
    """Verify leakage audit passes and all artifact files (.pt, .npz, .joblib, .json) are saved correctly."""
    train_df, test_df = mock_dataset_splits
    preprocessor = LeakageSafePreprocessor(scaler_type="robust")
    preprocessor.fit(train_df)

    train_out = preprocessor.transform(train_df)
    test_out = preprocessor.transform(test_df)

    # Leakage Audit
    audit = preprocessor.leakage_audit(train_df, test_df, train_out, test_out)
    assert audit["status"] == "PASSED"
    assert audit["checks"]["target_isolation"]["passed"] is True
    assert audit["checks"]["scaler_isolation"]["passed"] is True
    assert audit["checks"]["vocabulary_isolation"]["passed"] is True
    assert audit["checks"]["numerical_finiteness"]["passed"] is True

    # Save artifacts
    saved_paths = preprocessor.save_artifacts(tmp_path, train_out, test_out)
    assert saved_paths["train_pt"].exists()
    assert saved_paths["test_pt"].exists()
    assert saved_paths["train_npz"].exists()
    assert saved_paths["test_npz"].exists()
    assert saved_paths["preprocessor"].exists()
    assert saved_paths["metadata"].exists()

    # Verify PyTorch loading
    loaded_pt = torch.load(saved_paths["train_pt"], weights_only=True)
    assert "x_num" in loaded_pt
    assert "x_cat" in loaded_pt
    assert "x_dense" in loaded_pt
    assert "y_multiclass" in loaded_pt
    assert "y_binary" in loaded_pt
    assert loaded_pt["x_num"].shape[0] == len(train_df)

    # Verify Metadata JSON loading
    with open(saved_paths["metadata"], "r", encoding="utf-8") as f:
        meta_json = json.load(f)
    assert meta_json["num_classes"] == 10
    assert meta_json["preprocessing_version"] == "1.0.0"
    assert "Normal" in meta_json["class_mapping"]
