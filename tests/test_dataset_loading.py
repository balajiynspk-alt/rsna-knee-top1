"""
Unit tests for dataset acquisition verification, loading, and inspection.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data.inspect_dataset import DatasetInspector
from src.utils.paths import ProjectPaths


@pytest.fixture
def mock_unsw_dataset_dir(tmp_path: Path) -> Path:
    """Create a minimal valid synthetic mock of UNSW-NB15 in a temporary directory."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
        "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt",
        "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
        "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
        "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
        "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
        "is_sm_ips_ports", "attack_cat", "label"
    ]

    # Generate small synthetic dataframe
    n_samples = 50
    np.random.seed(42)

    data = {col: np.random.rand(n_samples) for col in columns}
    data["id"] = list(range(1, n_samples + 1))
    data["proto"] = np.random.choice(["tcp", "udp", "arp", "ospf"], size=n_samples)
    data["service"] = np.random.choice(["http", "ftp", "dns", "-"], size=n_samples)
    data["state"] = np.random.choice(["FIN", "INT", "CON", "REQ"], size=n_samples)
    data["label"] = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    data["attack_cat"] = [
        "Normal" if l == 0 else np.random.choice(["Generic", "Exploits", "DoS", "Fuzzers"])
        for l in data["label"]
    ]

    train_df = pd.DataFrame(data)
    test_df = train_df.copy()

    train_path = raw_dir / "UNSW_NB15_training-set.csv"
    test_path = raw_dir / "UNSW_NB15_testing-set.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return raw_dir


def test_missing_dataset_files_raises_filenotfound(tmp_path: Path):
    """Test that inspector raises FileNotFoundError with explicit guidance when files are absent."""
    empty_dir = tmp_path / "empty_raw"
    empty_dir.mkdir(parents=True, exist_ok=True)

    inspector = DatasetInspector(raw_dir=empty_dir)
    exists, missing = inspector.verify_file_existence()
    assert exists is False
    assert len(missing) == 2

    with pytest.raises(FileNotFoundError) as exc_info:
        inspector.load_datasets()

    assert "Required UNSW-NB15 dataset file(s) missing" in str(exc_info.value)


def test_inspector_load_and_inspect_structure(mock_unsw_dataset_dir: Path):
    """Test loading and structural extraction on valid synthetic dataset."""
    inspector = DatasetInspector(raw_dir=mock_unsw_dataset_dir)
    train_df, test_df = inspector.load_datasets()

    assert train_df.shape[0] == 50
    assert test_df.shape[0] == 50
    assert "label" in train_df.columns
    assert "attack_cat" in train_df.columns

    struct = inspector.inspect_structure(train_df, "Train")
    assert struct["rows"] == 50
    assert struct["columns"] == 45
    assert len(struct["missing_values"]) == 0
    assert len(struct["infinite_values"]) == 0


def test_inspector_targets_and_categories(mock_unsw_dataset_dir: Path):
    """Test target and categorical feature analysis."""
    inspector = DatasetInspector(raw_dir=mock_unsw_dataset_dir)
    train_df, _ = inspector.load_datasets()

    targets = inspector.inspect_targets(train_df, "Train")
    assert "label" in targets
    assert "attack_cat" in targets
    assert targets["label"]["counts"][0] + targets["label"]["counts"][1] == 50

    cats = inspector.inspect_categorical_columns(train_df)
    assert "proto" in cats
    assert "service" in cats
    assert "state" in cats


def test_numerical_statistics_and_plot_generation(mock_unsw_dataset_dir: Path, tmp_path: Path):
    """Test computation of numerical statistics and plot generation without mutation."""
    inspector = DatasetInspector(raw_dir=mock_unsw_dataset_dir)
    train_df, test_df = inspector.load_datasets()

    stats_df = inspector.compute_numerical_statistics(train_df, test_df)
    assert not stats_df.empty
    assert "feature" in stats_df.columns
    assert "train_mean" in stats_df.columns
    assert "train_std" in stats_df.columns

    # Verify targets and metadata are not present as features in numerical stats
    assert "id" not in stats_df["feature"].values
    assert "label" not in stats_df["feature"].values
    assert "attack_cat" not in stats_df["feature"].values

    # Test saving table
    out_table = tmp_path / "stats.csv"
    inspector.save_statistics_table(stats_df, out_table)
    assert out_table.exists()

    # Test plot generation
    fig_dir = tmp_path / "figures"
    c_plot, a_plot = inspector.generate_eda_plots(train_df, test_df, figures_dir=fig_dir)
    assert c_plot.exists()
    assert a_plot.exists()
