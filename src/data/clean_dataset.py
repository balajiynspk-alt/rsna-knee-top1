#!/usr/bin/env python3
"""
TGCF-IDS: Reproducible Data Cleaning Pipeline for UNSW-NB15.
Performs deterministic string stripping, categorical standardization, numerical validation,
and duplicate audit without applying premature scaling, encoding, or data leakage.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Cross-platform root path configuration
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.paths import ProjectPaths


class DatasetCleaner:
    """
    Deterministic data cleaning pipeline for network intrusion detection datasets.
    Operates strictly on copies; never mutates raw files.
    """

    TARGET_COLUMNS = ["label", "attack_cat"]
    METADATA_COLUMNS = ["id"]
    BINARY_FLAG_COLUMNS = ["is_ftp_login", "is_sm_ips_ports"]
    CATEGORICAL_COLUMNS = ["proto", "service", "state"]

    # Non-negative numerical features
    NON_NEGATIVE_NUMERICAL_COLS = [
        "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
        "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
        "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
        "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src",
        "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
        "ct_dst_src_ltm", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst"
    ]

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        train_filename: str = "UNSW_NB15_training-set.csv",
        test_filename: str = "UNSW_NB15_testing-set.csv",
        remove_exact_duplicates: bool = False,
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else ProjectPaths.DATA_RAW
        self.processed_dir = Path(processed_dir) if processed_dir else ProjectPaths.DATA_PROCESSED
        self.train_path = self.raw_dir / train_filename
        self.test_path = self.raw_dir / test_filename
        self.remove_exact_duplicates = remove_exact_duplicates

        # Audit logs to populate cleaning report
        self.audit_log: Dict[str, Any] = {}

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw training and testing splits without in-place modification."""
        if not self.train_path.exists():
            raise FileNotFoundError(f"Training file not found: {self.train_path}")
        if not self.test_path.exists():
            raise FileNotFoundError(f"Testing file not found: {self.test_path}")

        print(f"[+] Loading raw training data: {self.train_path.name}")
        train_df = pd.read_csv(self.train_path, low_memory=False)

        print(f"[+] Loading raw testing data : {self.test_path.name}")
        test_df = pd.read_csv(self.test_path, low_memory=False)

        return train_df, test_df

    def clean_column_names(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Strip whitespace and normalize column names."""
        original_cols = list(df.columns)
        cleaned_cols = [c.strip() for c in original_cols]
        df.columns = cleaned_cols
        renamed_count = sum(1 for o, c in zip(original_cols, cleaned_cols) if o != c)
        return df, renamed_count

    def clean_string_and_categorical_fields(
        self, df: pd.DataFrame, split_name: str
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Clean whitespace, standardize missing service designations, and canonicalize categories.
        """
        modifications = {}

        # 1. Clean all object / string dtype columns for leading/trailing whitespace
        str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()

        # 2. Standardize service column: replace '-' or empty with 'none'
        if "service" in df.columns:
            empty_service_mask = df["service"].isin(["-", "", "nan", "None"])
            count_service_mod = int(empty_service_mask.sum())
            df.loc[empty_service_mask, "service"] = "none"
            modifications["service_unspecified_to_none"] = count_service_mod

        # 3. Standardize attack_cat taxonomy (e.g., 'Backdoors' -> 'Backdoor', 'Normal' fill)
        if "attack_cat" in df.columns:
            # Standardize known variations in raw UNSW-NB15
            attack_mapping = {
                "Backdoors": "Backdoor",
                "backdoor": "Backdoor",
                "fuzzers": "Fuzzers",
                "exploits": "Exploits",
                "reconnaissance": "Reconnaissance",
                "shellcode": "Shellcode",
                "worms": "Worms",
                "analysis": "Analysis",
                "dos": "DoS",
                "generic": "Generic",
                "normal": "Normal",
                "nan": "Normal",
                "": "Normal",
            }
            # Fill missing with Normal if binary label is 0
            if "label" in df.columns:
                normal_mask = (df["label"] == 0) & (df["attack_cat"].isin(["nan", "", "Normal"]))
                df.loc[normal_mask, "attack_cat"] = "Normal"

            df["attack_cat"] = df["attack_cat"].replace(attack_mapping)
            modifications["attack_cat_standardized"] = len(df)

        # 4. Standardize protocol and state to lowercase/uppercase consistency
        if "proto" in df.columns:
            df["proto"] = df["proto"].str.lower()
        if "state" in df.columns:
            df["state"] = df["state"].str.upper()

        return df, modifications

    def validate_and_clean_numerical_values(
        self, df: pd.DataFrame, split_name: str
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Validate numerical integrity: check for inf/nan, non-negative bounds, and binary flag consistency.
        """
        modifications = {}

        # 1. Handle Infinite values
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        total_inf_handled = 0
        for col in num_cols:
            inf_mask = np.isinf(df[col])
            inf_cnt = int(inf_mask.sum())
            if inf_cnt > 0:
                # Replace inf with NaN for deterministic downstream imputation
                df.loc[inf_mask, col] = np.nan
                total_inf_handled += inf_cnt

        modifications["infinite_values_replaced_with_nan"] = total_inf_handled

        # 2. Non-negative bounds validation
        total_neg_clipped = 0
        for col in self.NON_NEGATIVE_NUMERICAL_COLS:
            if col in df.columns:
                neg_mask = df[col] < 0
                neg_cnt = int(neg_mask.sum())
                if neg_cnt > 0:
                    df.loc[neg_mask, col] = 0
                    total_neg_clipped += neg_cnt

        modifications["negative_values_clipped_to_zero"] = total_neg_clipped

        # 3. Binary flag standardization (e.g., is_ftp_login: values >= 1 -> 1)
        for col in self.BINARY_FLAG_COLUMNS:
            if col in df.columns:
                non_binary_mask = ~df[col].isin([0, 1])
                non_bin_cnt = int(non_binary_mask.sum())
                if non_bin_cnt > 0:
                    df[col] = (df[col] > 0).astype(int)
                    modifications[f"{col}_binarized"] = non_bin_cnt

        # 4. Target label standardization (label must be strictly 0 or 1)
        if "label" in df.columns:
            df["label"] = df["label"].astype(int)

        return df, modifications

    def audit_duplicates(
        self, df: pd.DataFrame, split_name: str
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Audit duplicate records:
        - Exact full-row duplicates
        - Feature-only duplicates
        - Conflicting label duplicates (identical features, different labels)
        """
        feature_cols = [c for c in df.columns if c not in (self.TARGET_COLUMNS + self.METADATA_COLUMNS)]
        full_cols = [c for c in df.columns if c not in self.METADATA_COLUMNS]

        exact_dups = int(df.duplicated(subset=full_cols, keep="first").sum())
        feature_dups = int(df.duplicated(subset=feature_cols, keep=False).sum())

        # Check conflicting labels
        conflicting_count = 0
        if "label" in df.columns and len(feature_cols) > 0:
            dup_features_df = df[df.duplicated(subset=feature_cols, keep=False)]
            if len(dup_features_df) > 0:
                grouped = dup_features_df.groupby(feature_cols)["label"].nunique()
                conflicting_count = int((grouped > 1).sum())

        audit = {
            "exact_duplicates_count": exact_dups,
            "feature_duplicates_count": feature_dups,
            "conflicting_label_groups": conflicting_count,
            "duplicates_removed": 0,
        }

        if self.remove_exact_duplicates and exact_dups > 0:
            original_len = len(df)
            df = df.drop_duplicates(subset=full_cols, keep="first").reset_index(drop=True)
            removed = original_len - len(df)
            audit["duplicates_removed"] = removed
            print(f"[i] [{split_name}] Removed {removed:,} exact duplicate rows.")

        return df, audit

    def clean_split(self, df: pd.DataFrame, split_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute complete cleaning pipeline for a single dataset split deterministically."""
        # Deep copy to ensure no mutation of original dataframe
        cleaned_df = df.copy(deep=True)
        original_rows = len(cleaned_df)
        original_cols = len(cleaned_df.columns)

        # 1. Clean Column Names
        cleaned_df, renamed_cols = self.clean_column_names(cleaned_df)

        # 2. String & Categorical Cleaning
        cleaned_df, cat_mods = self.clean_string_and_categorical_fields(cleaned_df, split_name)

        # 3. Numerical & Flag Validation
        cleaned_df, num_mods = self.validate_and_clean_numerical_values(cleaned_df, split_name)

        # 4. Duplicate Audit & Optional Deduplication
        cleaned_df, dup_audit = self.audit_duplicates(cleaned_df, split_name)

        final_rows = len(cleaned_df)
        final_cols = len(cleaned_df.columns)

        # Count any remaining nulls
        missing_handled = int(cleaned_df.isnull().sum().sum())

        report_entry = {
            "split": split_name,
            "original_rows": original_rows,
            "cleaned_rows": final_rows,
            "original_columns": original_cols,
            "cleaned_columns": final_cols,
            "renamed_columns": renamed_cols,
            "missing_values_remaining": missing_handled,
            "infinite_values_handled": num_mods.get("infinite_values_replaced_with_nan", 0),
            "negative_values_clipped": num_mods.get("negative_values_clipped_to_zero", 0),
            "exact_duplicates_found": dup_audit["exact_duplicates_count"],
            "duplicates_removed": dup_audit["duplicates_removed"],
            "conflicting_label_feature_groups": dup_audit["conflicting_label_groups"],
            "service_unspecified_normalized": cat_mods.get("service_unspecified_to_none", 0),
            "is_ftp_login_binarized": num_mods.get("is_ftp_login_binarized", 0),
        }

        return cleaned_df, report_entry

    def run_pipeline(
        self,
        output_train_filename: str = "UNSW_NB15_training-set_cleaned.csv",
        output_test_filename: str = "UNSW_NB15_testing-set_cleaned.csv",
        report_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run end-to-end data cleaning, export processed CSVs, and generate structured cleaning report.
        """
        print("=" * 75)
        print("        TGCF-IDS : Reproducible Data Cleaning Pipeline        ")
        print("=" * 75)

        # 1. Load Raw Data
        train_raw, test_raw = self.load_raw_data()

        # 2. Clean Training Split
        print("[*] Processing Training Split...")
        cleaned_train, train_report = self.clean_split(train_raw, "Training Set")

        # 3. Clean Testing Split
        print("[*] Processing Testing Split...")
        cleaned_test, test_report = self.clean_split(test_raw, "Testing Set")

        # 4. Save Cleaned Datasets under data/processed/
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        train_out_path = self.processed_dir / output_train_filename
        test_out_path = self.processed_dir / output_test_filename

        print(f"[+] Exporting cleaned training data -> {train_out_path}")
        cleaned_train.to_csv(train_out_path, index=False)

        print(f"[+] Exporting cleaned testing data  -> {test_out_path}")
        cleaned_test.to_csv(test_out_path, index=False)

        # 5. Compile & Save Cleaning Report under results/tables/
        report_df = pd.DataFrame([train_report, test_report])
        final_report_path = report_path or (ProjectPaths.RESULTS_TABLES / "data_cleaning_report.csv")
        final_report_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(final_report_path, index=False)
        print(f"[+] Exporting cleaning report       -> {final_report_path}")

        # Console Summary
        print("-" * 75)
        print("CLEANING AUDIT REPORT SUMMARY:")
        for rep in [train_report, test_report]:
            print(f"\n--- {rep['split']} ---")
            print(f"  * Original Rows       : {rep['original_rows']:,}")
            print(f"  * Cleaned Rows        : {rep['cleaned_rows']:,}")
            print(f"  * Exact Duplicates    : {rep['exact_duplicates_found']:,} (Removed: {rep['duplicates_removed']:,})")
            print(f"  * Conflicting Labels  : {rep['conflicting_label_feature_groups']:,} feature groups")
            print(f"  * Infinite Handled    : {rep['infinite_values_handled']:,}")
            print(f"  * Service Normalized  : {rep['service_unspecified_normalized']:,} instances ('-' -> 'none')")
            print(f"  * Non-binary Fixed    : {rep['is_ftp_login_binarized']:,} (is_ftp_login > 1 binarized)")

        print("=" * 75)
        print("Data Cleaning Pipeline Completed Successfully.")
        print("=" * 75)

        return {
            "cleaned_train": cleaned_train,
            "cleaned_test": cleaned_test,
            "train_out_path": train_out_path,
            "test_out_path": test_out_path,
            "report_path": report_path,
            "report_df": report_df,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Reproducible data cleaning pipeline for TGCF-IDS."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(ProjectPaths.DATA_RAW),
        help="Path to raw dataset directory.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(ProjectPaths.DATA_PROCESSED),
        help="Path to save cleaned dataset files.",
    )
    parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        default=False,
        help="Whether to remove exact duplicate rows from the processed dataset (default: False to preserve standard benchmark evaluation splits).",
    )
    args = parser.parse_args()

    cleaner = DatasetCleaner(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        remove_exact_duplicates=args.remove_duplicates,
    )
    cleaner.run_pipeline()


if __name__ == "__main__":
    main()
