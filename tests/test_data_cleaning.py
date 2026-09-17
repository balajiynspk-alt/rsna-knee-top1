"""
Unit and integration tests for the reproducible data-cleaning pipeline.
"""

from pathlib import Path
from typing import Tuple
import pytest
import pandas as pd
import numpy as np

from src.data.clean_dataset import DatasetCleaner


@pytest.fixture
def mock_dirty_dataset(tmp_path: Path) -> Tuple[Path, Path]:
    """Create synthetic dirty train/test CSVs to rigorously test edge cases."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Dirty synthetic records
    data = {
        "id ": [1, 2, 3, 4, 5, 6],  # Trailing whitespace in column name
        "dur": [0.1, -0.05, 1.2, np.inf, 0.5, 0.5],  # Negative & infinite values
        "proto": [" TCP ", "udp", "ARP", "ospf", "tcp", "tcp"],  # Whitespace & casing
        "service": ["-", "http", "-", "dns", "ftp", "ftp"],  # Missing service marker
        "state": ["fin", " INT ", "CON", "REQ", "FIN", "FIN"],  # Casing & whitespace
        "spkts": [10, 5, 20, 15, 8, 8],
        "dpkts": [8, 4, 18, 12, 6, 6],
        "sbytes": [1000, 500, 2000, 1500, 800, 800],
        "dbytes": [800, 400, 1800, 1200, 600, 600],
        "rate": [100.0, 50.0, 200.0, 150.0, 80.0, 80.0],
        "is_ftp_login": [0, 2, 4, 0, 1, 1],  # Non-binary flag values
        "is_sm_ips_ports": [0, 0, 0, 1, 0, 0],
        "attack_cat": ["Normal", "Backdoors", " Fuzzers ", "DoS", "Normal", "Exploits"],  # Casing/whitespace/conflict
        "label": [0, 1, 1, 1, 0, 1],  # Row 5 and 6 have identical features but conflicting labels!
    }

    train_df = pd.DataFrame(data)
    test_df = train_df.copy()

    train_path = raw_dir / "UNSW_NB15_training-set.csv"
    test_path = raw_dir / "UNSW_NB15_testing-set.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return raw_dir, processed_dir


def test_clean_column_names_stripping(mock_dirty_dataset):
    """Test column name whitespace removal."""
    raw_dir, processed_dir = mock_dirty_dataset
    cleaner = DatasetCleaner(raw_dir=raw_dir, processed_dir=processed_dir)
    train_df, _ = cleaner.load_raw_data()

    cleaned_df, renamed_count = cleaner.clean_column_names(train_df)
    assert "id" in cleaned_df.columns
    assert "id " not in cleaned_df.columns
    assert renamed_count >= 1


def test_clean_string_and_categorical_fields(mock_dirty_dataset):
    """Test whitespace stripping, service standardization, and category canonicalization."""
    raw_dir, processed_dir = mock_dirty_dataset
    cleaner = DatasetCleaner(raw_dir=raw_dir, processed_dir=processed_dir)
    train_df, _ = cleaner.load_raw_data()
    train_df, _ = cleaner.clean_column_names(train_df)

    cleaned_df, mods = cleaner.clean_string_and_categorical_fields(train_df, "Train")

    # Protocol should be lowercase
    assert cleaned_df["proto"].tolist()[0] == "tcp"
    assert cleaned_df["proto"].tolist()[2] == "arp"

    # State should be uppercase
    assert cleaned_df["state"].tolist()[0] == "FIN"
    assert cleaned_df["state"].tolist()[1] == "INT"

    # Service '-' should become 'none'
    assert cleaned_df["service"].tolist()[0] == "none"
    assert cleaned_df["service"].tolist()[2] == "none"

    # Attack category standardized
    assert "Backdoor" in cleaned_df["attack_cat"].values
    assert "Backdoors" not in cleaned_df["attack_cat"].values
    assert "Fuzzers" in cleaned_df["attack_cat"].values


def test_validate_and_clean_numerical_values(mock_dirty_dataset):
    """Test infinite value replacement, negative clipping, and flag binarization."""
    raw_dir, processed_dir = mock_dirty_dataset
    cleaner = DatasetCleaner(raw_dir=raw_dir, processed_dir=processed_dir)
    train_df, _ = cleaner.load_raw_data()
    train_df, _ = cleaner.clean_column_names(train_df)

    cleaned_df, mods = cleaner.validate_and_clean_numerical_values(train_df, "Train")

    # Negative dur should be clipped to 0
    assert cleaned_df["dur"].min() >= 0
    assert cleaned_df["dur"].iloc[1] == 0.0

    # Infinite dur should be replaced with NaN
    assert not np.isinf(cleaned_df["dur"]).any()

    # is_ftp_login values (2, 4) should become 1
    assert set(cleaned_df["is_ftp_login"].unique()).issubset({0, 1})
    assert cleaned_df["is_ftp_login"].iloc[1] == 1
    assert cleaned_df["is_ftp_login"].iloc[2] == 1


def test_duplicate_and_conflicting_label_audit(mock_dirty_dataset):
    """Test duplicate detection and conflicting label group identification."""
    raw_dir, processed_dir = mock_dirty_dataset
    cleaner = DatasetCleaner(raw_dir=raw_dir, processed_dir=processed_dir)
    train_df, _ = cleaner.load_raw_data()
    train_df, _ = cleaner.clean_column_names(train_df)
    train_df, _ = cleaner.clean_string_and_categorical_fields(train_df, "Train")
    train_df, _ = cleaner.validate_and_clean_numerical_values(train_df, "Train")

    cleaned_df, audit = cleaner.audit_duplicates(train_df, "Train")

    assert audit["conflicting_label_groups"] >= 1


def test_end_to_end_cleaning_pipeline(mock_dirty_dataset, tmp_path):
    """Test complete pipeline execution and file export."""
    raw_dir, processed_dir = mock_dirty_dataset
    cleaner = DatasetCleaner(raw_dir=raw_dir, processed_dir=processed_dir)

    test_report_path = tmp_path / "test_cleaning_report.csv"
    result = cleaner.run_pipeline(report_path=test_report_path)

    assert result["train_out_path"].exists()
    assert result["test_out_path"].exists()
    assert result["report_path"].exists()

    report_df = pd.read_csv(result["report_path"])
    assert len(report_df) == 2  # Train and test rows
    assert "original_rows" in report_df.columns
    assert "cleaned_rows" in report_df.columns
