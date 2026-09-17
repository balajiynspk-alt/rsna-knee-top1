"""
Unit tests for exploratory data analysis (EDA) metrics, figures, and reporting.
"""

from pathlib import Path
from typing import Tuple
import matplotlib
matplotlib.use("Agg")
import pytest
import numpy as np
import pandas as pd

from src.data.eda_analysis import ExploratoryDataAnalyzer


@pytest.fixture
def mock_eda_splits(tmp_path: Path) -> Tuple[Path, Path, Path, Path]:
    """Create synthetic cleaned CSV datasets and output directories for testing EDA functions."""
    data_dir = tmp_path / "data"
    fig_dir = tmp_path / "figures"
    table_dir = tmp_path / "tables"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n_samples = 200

    numerical_cols = [
        "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
        "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
        "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
        "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src",
        "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
        "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
        "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports"
    ]

    data = {col: np.random.exponential(scale=10.0, size=n_samples) for col in numerical_cols}
    # Add correlated pair
    data["spkts"] = data["sbytes"] * 0.1 + np.random.normal(0, 0.1, n_samples)
    data["id"] = list(range(1, n_samples + 1))
    data["proto"] = np.random.choice(["tcp", "udp", "arp", "icmp"], size=n_samples)
    data["service"] = np.random.choice(["none", "http", "dns", "ftp"], size=n_samples)
    data["state"] = np.random.choice(["FIN", "INT", "CON", "REQ"], size=n_samples)
    data["attack_cat"] = np.random.choice(
        ["Normal", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms"],
        size=n_samples,
        p=[0.40, 0.05, 0.05, 0.10, 0.15, 0.10, 0.08, 0.04, 0.02, 0.01]
    )
    data["label"] = [0 if a == "Normal" else 1 for a in data["attack_cat"]]

    train_df = pd.DataFrame(data)
    test_df = train_df.copy()

    train_path = data_dir / "train_cleaned.csv"
    test_path = data_dir / "test_cleaned.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, test_path, fig_dir, table_dir


def test_class_imbalance_metrics(mock_eda_splits):
    """Test class distribution calculation, sample counts, and imbalance ratios."""
    train_path, test_path, fig_dir, table_dir = mock_eda_splits
    analyzer = ExploratoryDataAnalyzer(train_path=train_path, test_path=test_path, figures_dir=fig_dir, tables_dir=table_dir)
    train_df, _ = analyzer.load_data()

    imbalance_df = analyzer.compute_class_imbalance(train_df)
    assert not imbalance_df.empty
    assert "class_name" in imbalance_df.columns
    assert "imbalance_ratio" in imbalance_df.columns
    assert imbalance_df["sample_count"].sum() == len(train_df)
    assert imbalance_df["percentage"].sum() == pytest.approx(100.0, rel=1e-2)

    # Majority class should have IR = 1.0
    normal_row = imbalance_df[imbalance_df["class_name"] == "Normal"]
    assert normal_row["imbalance_ratio"].values[0] == 1.0


def test_numerical_distributions_and_outliers(mock_eda_splits):
    """Test computation of numerical summary metrics and IQR outliers."""
    train_path, test_path, fig_dir, table_dir = mock_eda_splits
    analyzer = ExploratoryDataAnalyzer(train_path=train_path, test_path=test_path, figures_dir=fig_dir, tables_dir=table_dir)
    train_df, _ = analyzer.load_data()

    dist_df = analyzer.compute_numerical_distribution_summary(train_df)
    assert not dist_df.empty
    assert "skewness" in dist_df.columns
    assert "outlier_pct_iqr" in dist_df.columns
    assert "median" in dist_df.columns
    assert "id" not in dist_df["feature"].values
    assert "label" not in dist_df["feature"].values


def test_feature_variance_and_correlations(mock_eda_splits):
    """Test variance and multicollinearity detection."""
    train_path, test_path, fig_dir, table_dir = mock_eda_splits
    analyzer = ExploratoryDataAnalyzer(train_path=train_path, test_path=test_path, figures_dir=fig_dir, tables_dir=table_dir)
    train_df, _ = analyzer.load_data()

    # Variance
    var_df = analyzer.compute_feature_variance(train_df)
    assert "variance" in var_df.columns
    assert "is_zero_variance" in var_df.columns

    # Correlations
    corr_mat, collinear_df = analyzer.compute_feature_correlations(train_df, threshold=0.80)
    assert corr_mat.shape[0] == len(var_df)
    # The synthetic pair spkts and sbytes should be detected as collinear
    collinear_pairs = set(zip(collinear_df["feature_1"], collinear_df["feature_2"])).union(
        set(zip(collinear_df["feature_2"], collinear_df["feature_1"]))
    )
    assert ("sbytes", "spkts") in collinear_pairs or ("spkts", "sbytes") in collinear_pairs


def test_categorical_crosstabs(mock_eda_splits):
    """Test cross-tabulations for proto, service, and state."""
    train_path, test_path, fig_dir, table_dir = mock_eda_splits
    analyzer = ExploratoryDataAnalyzer(train_path=train_path, test_path=test_path, figures_dir=fig_dir, tables_dir=table_dir)
    train_df, _ = analyzer.load_data()

    crosstabs = analyzer.compute_categorical_crosstabs(train_df)
    assert "proto" in crosstabs
    assert "service" in crosstabs
    assert "state" in crosstabs
    assert "Total" in crosstabs["proto"].columns


def test_end_to_end_eda_execution(mock_eda_splits):
    """Test complete EDA execution and file generation."""
    train_path, test_path, fig_dir, table_dir = mock_eda_splits
    analyzer = ExploratoryDataAnalyzer(train_path=train_path, test_path=test_path, figures_dir=fig_dir, tables_dir=table_dir)

    result = analyzer.run_full_eda()

    # Verify generated tables
    assert (table_dir / "class_imbalance_metrics.csv").exists()
    assert (table_dir / "numerical_distribution_summary.csv").exists()
    assert (table_dir / "feature_variance_analysis.csv").exists()
    assert (table_dir / "top_correlations.csv").exists()
    assert (table_dir / "categorical_attack_crosstabs.csv").exists()

    # Verify generated figures
    assert (fig_dir / "correlation_heatmap.png").exists()
    assert (fig_dir / "numerical_feature_distributions.png").exists()
    assert (fig_dir / "temporal_traffic.png").exists()
    assert (fig_dir / "attack_over_time.png").exists()
    assert (fig_dir / "protocol_attack_distribution.png").exists()
    assert (fig_dir / "service_attack_distribution.png").exists()
    assert (fig_dir / "per_class_feature_analysis.png").exists()
