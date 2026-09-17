#!/usr/bin/env python3
"""
TGCF-IDS: Dataset Acquisition Verification and Inspection Tool for UNSW-NB15.
Performs rigorous data integrity verification, statistical reporting, and EDA plotting.
NO data modification, scaling, encoding, balancing, or leakage is performed.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on sys.path for robust cross-platform imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.paths import ProjectPaths


class DatasetInspector:
    """
    Validates, inspects, and analyzes raw UNSW-NB15 dataset splits without mutation.
    """

    EXPECTED_TRAIN_NAME = "UNSW_NB15_training-set.csv"
    EXPECTED_TEST_NAME = "UNSW_NB15_testing-set.csv"

    # Known targets and non-feature metadata columns in UNSW-NB15
    TARGET_COLUMNS = ["label", "attack_cat"]
    LEAKAGE_METADATA_COLUMNS = ["id"]

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        train_filename: Optional[str] = None,
        test_filename: Optional[str] = None,
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else ProjectPaths.DATA_RAW
        self.train_path = self.raw_dir / (train_filename or self.EXPECTED_TRAIN_NAME)
        self.test_path = self.raw_dir / (test_filename or self.EXPECTED_TEST_NAME)

        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def verify_file_existence(self) -> Tuple[bool, List[str]]:
        """
        Verify presence of raw training and testing CSV files.
        Returns (all_present, list_of_missing_files).
        """
        missing = []
        if not self.train_path.exists():
            missing.append(str(self.train_path))
        if not self.test_path.exists():
            missing.append(str(self.test_path))

        return len(missing) == 0, missing

    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load datasets without modifying original records.
        Fails clearly if files are missing.
        """
        exists, missing = self.verify_file_existence()
        if not exists:
            err_msg = (
                f"\n[CRITICAL ERROR] Required UNSW-NB15 dataset file(s) missing:\n"
                + "\n".join([f"  [-] {m}" for m in missing])
                + f"\n\nPlease place '{self.EXPECTED_TRAIN_NAME}' and '{self.EXPECTED_TEST_NAME}'"
                f" into: {self.raw_dir.resolve()}\n"
            )
            raise FileNotFoundError(err_msg)

        # Verify non-empty files
        for p in [self.train_path, self.test_path]:
            if p.stat().st_size < 1024:
                raise ValueError(
                    f"\n[CRITICAL ERROR] File '{p.name}' is too small ({p.stat().st_size} bytes) and appears invalid or incomplete.\n"
                    f"Please ensure the full CSV file is downloaded into: {self.raw_dir.resolve()}\n"
                )

        print(f"[+] Loading training set from: {self.train_path.name}...")
        self.train_df = pd.read_csv(self.train_path, low_memory=False)

        print(f"[+] Loading testing set from : {self.test_path.name}...")
        self.test_df = pd.read_csv(self.test_path, low_memory=False)

        if len(self.train_df) == 0 or len(self.test_df) == 0:
            raise ValueError("Training or testing dataset contains 0 rows.")

        return self.train_df, self.test_df

    def inspect_structure(self, df: pd.DataFrame, split_name: str) -> Dict:
        """
        Extract structural properties: row/col counts, data types, missing/inf values.
        """
        num_rows, num_cols = df.shape
        cols = list(df.columns)
        dtypes = df.dtypes.to_dict()

        # Missing values
        missing_series = df.isnull().sum()
        missing_dict = missing_series[missing_series > 0].to_dict()

        # Infinite values in numerical columns
        num_df = df.select_dtypes(include=[np.number])
        inf_counts = {}
        for col in num_df.columns:
            inf_cnt = int(np.isinf(num_df[col]).sum())
            if inf_cnt > 0:
                inf_counts[col] = inf_cnt

        # Duplicate rows (excluding 'id' if present)
        cols_to_check = [c for c in cols if c != "id"]
        duplicate_count = int(df.duplicated(subset=cols_to_check).sum())

        return {
            "split": split_name,
            "rows": num_rows,
            "columns": num_cols,
            "column_names": cols,
            "dtypes": dtypes,
            "missing_values": missing_dict,
            "infinite_values": inf_counts,
            "duplicates": duplicate_count,
        }

    def inspect_categorical_columns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Inspect unique values and cardinalities of categorical features.
        """
        cat_cols = df.select_dtypes(include=["object", "category", "string", "str"]).columns.tolist()
        cat_info = {}
        for col in cat_cols:
            unique_vals = df[col].dropna().unique().tolist()
            val_counts = df[col].value_counts(dropna=False).head(10).to_dict()
            cat_info[col] = {
                "num_unique": len(unique_vals),
                "unique_values_sample": unique_vals[:15],
                "top_10_distribution": val_counts,
            }
        return cat_info

    def inspect_targets(self, df: pd.DataFrame, split_name: str) -> Dict:
        """
        Analyze label (binary target) and attack_cat (multi-class target) distributions.
        """
        target_info = {}

        if "label" in df.columns:
            label_counts = df["label"].value_counts(dropna=False).to_dict()
            label_pcts = (df["label"].value_counts(normalize=True, dropna=False) * 100).round(2).to_dict()
            target_info["label"] = {
                "counts": label_counts,
                "percentages": label_pcts,
            }

        if "attack_cat" in df.columns:
            # Note: strip whitespace from raw representation purely for clean inspection
            raw_attack_series = df["attack_cat"].fillna("Normal").astype(str).str.strip()
            attack_counts = raw_attack_series.value_counts(dropna=False).to_dict()
            attack_pcts = (raw_attack_series.value_counts(normalize=True, dropna=False) * 100).round(2).to_dict()
            target_info["attack_cat"] = {
                "unique_categories": sorted(raw_attack_series.unique().tolist()),
                "num_categories": len(raw_attack_series.unique()),
                "counts": attack_counts,
                "percentages": attack_pcts,
            }

        return target_info

    def compute_numerical_statistics(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute descriptive numerical feature statistics (mean, std, min, quantiles, max, skewness)
        and compile into a structured comparative DataFrame.
        """
        # Exclude metadata / targets from numerical statistical table
        exclude_cols = set(self.TARGET_COLUMNS + self.LEAKAGE_METADATA_COLUMNS)
        num_cols = [
            c for c in train_df.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols
        ]

        stats_records = []
        for col in num_cols:
            tr_s = train_df[col].dropna()
            te_s = test_df[col].dropna() if col in test_df.columns else pd.Series(dtype=float)

            stats_records.append({
                "feature": col,
                "train_mean": float(tr_s.mean()),
                "train_std": float(tr_s.std()),
                "train_min": float(tr_s.min()),
                "train_p25": float(tr_s.quantile(0.25)),
                "train_median": float(tr_s.median()),
                "train_p75": float(tr_s.quantile(0.75)),
                "train_max": float(tr_s.max()),
                "train_skew": float(tr_s.skew()),
                "test_mean": float(te_s.mean()) if len(te_s) > 0 else np.nan,
                "test_std": float(te_s.std()) if len(te_s) > 0 else np.nan,
                "test_min": float(te_s.min()) if len(te_s) > 0 else np.nan,
                "test_median": float(te_s.median()) if len(te_s) > 0 else np.nan,
                "test_max": float(te_s.max()) if len(te_s) > 0 else np.nan,
                "test_skew": float(te_s.skew()) if len(te_s) > 0 else np.nan,
            })

        stats_df = pd.DataFrame(stats_records)
        return stats_df

    def save_statistics_table(
        self, stats_df: pd.DataFrame, output_path: Optional[Path] = None
    ) -> Path:
        """
        Export statistical table to CSV.
        """
        target_path = output_path or (ProjectPaths.RESULTS_TABLES / "dataset_statistics.csv")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(target_path, index=False)
        print(f"[+] Saved dataset numerical statistics to: {target_path}")
        return target_path

    def generate_eda_plots(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        figures_dir: Optional[Path] = None,
    ) -> Tuple[Path, Path]:
        """
        Generate academic publication-grade distribution plots for class balance and attack categories.
        """
        fig_dir = figures_dir or (ProjectPaths.RESULTS_FIGURES / "eda")
        fig_dir.mkdir(parents=True, exist_ok=True)

        sns.set_theme(style="whitegrid", palette="deep")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "figure.titlesize": 15,
        })

        # ---------------------------------------------------------
        # Plot 1: Binary Class Distribution (Normal vs Attack)
        # ---------------------------------------------------------
        tr_label_counts = train_df["label"].value_counts().rename({0: "Benign (0)", 1: "Attack (1)"})
        te_label_counts = test_df["label"].value_counts().rename({0: "Benign (0)", 1: "Attack (1)"})

        label_comp_df = pd.DataFrame({
            "Class": ["Benign (0)", "Attack (1)", "Benign (0)", "Attack (1)"],
            "Count": [
                tr_label_counts.get("Benign (0)", 0),
                tr_label_counts.get("Attack (1)", 0),
                te_label_counts.get("Benign (0)", 0),
                te_label_counts.get("Attack (1)", 0),
            ],
            "Split": ["Training Set", "Training Set", "Testing Set", "Testing Set"],
        })

        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
        barplot = sns.barplot(
            data=label_comp_df,
            x="Class",
            y="Count",
            hue="Split",
            palette=["#3498db", "#e74c3c"],
            ax=ax,
        )

        for p in barplot.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(
                    f"{int(height):,}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    xytext=(0, 4),
                    textcoords="offset points",
                )

        ax.set_title("UNSW-NB15: Binary Class Distribution (Train vs Test)", pad=15)
        ax.set_ylabel("Number of Flow Records")
        ax.set_xlabel("Class Label")
        ax.legend(title="Dataset Split", frameon=True)
        plt.tight_layout()

        class_dist_path = fig_dir / "class_distribution.png"
        fig.savefig(class_dist_path, dpi=300)
        plt.close(fig)
        print(f"[+] Saved EDA plot: {class_dist_path}")

        # ---------------------------------------------------------
        # Plot 2: Multi-Class Attack Category Distribution
        # ---------------------------------------------------------
        tr_atk = train_df["attack_cat"].fillna("Normal").astype(str).str.strip().value_counts()
        te_atk = test_df["attack_cat"].fillna("Normal").astype(str).str.strip().value_counts()

        all_cats = sorted(list(set(tr_atk.index).union(set(te_atk.index))))
        # Sort so Normal is at top or bottom, others sorted by total frequency
        all_cats = sorted(all_cats, key=lambda c: (tr_atk.get(c, 0) + te_atk.get(c, 0)), reverse=True)

        atk_records = []
        for cat in all_cats:
            atk_records.append({
                "Attack_Category": cat,
                "Count": tr_atk.get(cat, 0),
                "Split": "Training Set",
            })
            atk_records.append({
                "Attack_Category": cat,
                "Count": te_atk.get(cat, 0),
                "Split": "Testing Set",
            })

        atk_comp_df = pd.DataFrame(atk_records)

        fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
        sns.barplot(
            data=atk_comp_df,
            x="Attack_Category",
            y="Count",
            hue="Split",
            palette=["#2ecc71", "#9b59b6"],
            ax=ax,
        )

        ax.set_yscale("log")
        ax.set_title("UNSW-NB15: Attack Category Distribution (Log Scale)", pad=15)
        ax.set_ylabel("Number of Flow Records (Log Scale)")
        ax.set_xlabel("Attack Category")
        plt.xticks(rotation=35, ha="right")
        ax.legend(title="Dataset Split", frameon=True)
        plt.tight_layout()

        attack_dist_path = fig_dir / "attack_distribution.png"
        fig.savefig(attack_dist_path, dpi=300)
        plt.close(fig)
        print(f"[+] Saved EDA plot: {attack_dist_path}")

        return class_dist_path, attack_dist_path

    def run_full_inspection(self) -> Dict:
        """
        Execute full comprehensive inspection pipeline and print detailed console report.
        """
        print("=" * 75)
        print("     TGCF-IDS : UNSW-NB15 Dataset Acquisition & Inspection Report    ")
        print("=" * 75)

        # 1. File existence
        exists, missing = self.verify_file_existence()
        print(f"1. File Existence Check:")
        print(f"   - Training File : {self.train_path} -> {'EXISTS' if self.train_path.exists() else 'MISSING'}")
        print(f"   - Testing File  : {self.test_path} -> {'EXISTS' if self.test_path.exists() else 'MISSING'}")

        if not exists:
            self.load_datasets()  # Will raise FileNotFoundError with detailed instruction

        train_df, test_df = self.load_datasets()

        # 2 & 3. Rows and Columns
        tr_struct = self.inspect_structure(train_df, "Train")
        te_struct = self.inspect_structure(test_df, "Test")

        print("-" * 75)
        print("2 & 3. Dataset Dimensions:")
        print(f"   - Training Set : {tr_struct['rows']:,} rows × {tr_struct['columns']} columns")
        print(f"   - Testing Set  : {te_struct['rows']:,} rows × {te_struct['columns']} columns")
        print(f"   - Combined Total: {tr_struct['rows'] + te_struct['rows']:,} rows")

        # 4 & 5. Columns and Dtypes
        print("-" * 75)
        print("4 & 5. Column Names and Data Types (Training Set):")
        for i, col in enumerate(train_df.columns, 1):
            dtype = train_df[col].dtype
            print(f"   [{i:02d}] {col:<24}: {str(dtype):<10}")

        # 6, 7 & 8. Missing, Infinite, and Duplicate Values
        print("-" * 75)
        print("6, 7 & 8. Data Integrity (Missing / Infinite / Duplicates):")
        print(f"   - Training Missing Values  : {tr_struct['missing_values'] or 'None (0 missing)'}")
        print(f"   - Testing Missing Values   : {te_struct['missing_values'] or 'None (0 missing)'}")
        print(f"   - Training Infinite Values : {tr_struct['infinite_values'] or 'None (0 infinite)'}")
        print(f"   - Testing Infinite Values  : {te_struct['infinite_values'] or 'None (0 infinite)'}")
        print(f"   - Training Duplicate Flows : {tr_struct['duplicates']:,}")
        print(f"   - Testing Duplicate Flows  : {te_struct['duplicates']:,}")

        # 9. Unique values in Categorical Columns
        print("-" * 75)
        print("9. Categorical Columns Analysis:")
        tr_cat = self.inspect_categorical_columns(train_df)
        for col, info in tr_cat.items():
            if col not in self.TARGET_COLUMNS:
                print(f"   - Column '{col}': {info['num_unique']} unique values | Sample: {info['unique_values_sample'][:8]}")

        # 10, 11 & 12. Targets: label and attack_cat
        print("-" * 75)
        print("10, 11 & 12. Target Distributions (Binary 'label' & Multi-class 'attack_cat'):")
        tr_targets = self.inspect_targets(train_df, "Train")
        te_targets = self.inspect_targets(test_df, "Test")

        print("   [Binary 'label' (0 = Benign, 1 = Attack)]:")
        print(f"     Train -> Benign: {tr_targets['label']['counts'].get(0, 0):,} ({tr_targets['label']['percentages'].get(0, 0)}%) | Attack: {tr_targets['label']['counts'].get(1, 0):,} ({tr_targets['label']['percentages'].get(1, 0)}%)")
        print(f"     Test  -> Benign: {te_targets['label']['counts'].get(0, 0):,} ({te_targets['label']['percentages'].get(0, 0)}%) | Attack: {te_targets['label']['counts'].get(1, 0):,} ({te_targets['label']['percentages'].get(1, 0)}%)")

        print("\n   [Multi-class 'attack_cat']:")
        print(f"     Categories Found ({tr_targets['attack_cat']['num_categories']}): {tr_targets['attack_cat']['unique_categories']}")
        for cat, cnt in tr_targets['attack_cat']['counts'].items():
            pct = tr_targets['attack_cat']['percentages'].get(cat, 0.0)
            te_cnt = te_targets['attack_cat']['counts'].get(cat, 0)
            te_pct = te_targets['attack_cat']['percentages'].get(cat, 0.0)
            print(f"     - {cat:<18}: Train={cnt:>7,} ({pct:>5.2f}%) | Test={te_cnt:>7,} ({te_pct:>5.2f}%)")

        # 13. Numerical feature statistics
        print("-" * 75)
        print("13. Computing Numerical Feature Statistics...")
        stats_df = self.compute_numerical_statistics(train_df, test_df)
        csv_path = self.save_statistics_table(stats_df)

        # Plots
        print("-" * 75)
        print("[*] Generating EDA figures...")
        c_plot, a_plot = self.generate_eda_plots(train_df, test_df)

        print("=" * 75)
        print("Inspection Completed Successfully.")
        print(f"  - Statistics Table : {csv_path}")
        print(f"  - Class Plot       : {c_plot}")
        print(f"  - Attack Plot      : {a_plot}")
        print("=" * 75)

        return {
            "train_structure": tr_struct,
            "test_structure": te_struct,
            "train_targets": tr_targets,
            "test_targets": te_targets,
            "statistics_df": stats_df,
            "stats_table_path": csv_path,
            "class_plot_path": c_plot,
            "attack_plot_path": a_plot,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and verify UNSW-NB15 dataset integrity for TGCF-IDS."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(ProjectPaths.DATA_RAW),
        help="Path to raw data directory containing UNSW-NB15 CSV files.",
    )
    args = parser.parse_args()

    inspector = DatasetInspector(raw_dir=Path(args.data_dir))
    inspector.run_full_inspection()


if __name__ == "__main__":
    main()
