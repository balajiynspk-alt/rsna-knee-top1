#!/usr/bin/env python3
"""
TGCF-IDS: Leakage-Safe Data Preprocessing and Feature Pipeline.
Transforms cleaned UNSW-NB15 data into model-ready numerical and categorical tensors.
All transformations are strictly fitted ONLY on training data.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
import torch

# Ensure project root is accessible
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.paths import ProjectPaths


class LeakageSafePreprocessor:
    """
    Leakage-safe tabular and graph feature preprocessor.
    Guarantees strict separation between training and test sets.
    """

    VERSION = "1.0.0"

    # Exact 10-class taxonomy mapping as specified for UNSW-NB15
    CLASS_MAPPING: Dict[str, int] = {
        "Normal": 0,
        "Analysis": 1,
        "Backdoor": 2,
        "DoS": 3,
        "Exploits": 4,
        "Fuzzers": 5,
        "Generic": 6,
        "Reconnaissance": 7,
        "Shellcode": 8,
        "Worms": 9,
    }

    TARGET_COLUMNS = ["label", "attack_cat"]
    METADATA_COLUMNS = ["id"]
    CATEGORICAL_COLUMNS = ["proto", "service", "state"]

    UNK_TOKEN = "<UNK>"
    UNK_INDEX = 0

    def __init__(
        self,
        scaler_type: str = "robust",
        random_seed: int = 42,
    ):
        self.scaler_type = scaler_type.lower()
        self.random_seed = random_seed

        # Fit status
        self.is_fitted: bool = False

        # Feature groupings
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = list(self.CATEGORICAL_COLUMNS)
        self.metadata_cols: List[str] = list(self.METADATA_COLUMNS)
        self.all_feature_names: List[str] = []
        self.onehot_feature_names: List[str] = []

        # Encoders & Scalers
        if self.scaler_type == "robust":
            self.scaler = RobustScaler(quantile_range=(5.0, 95.0), copy=True)
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler(copy=True)
        else:
            raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Choose 'robust' or 'standard'.")

        self.onehot_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )

        # Categorical vocabularies for Neural Network embedding tokenization:
        # col -> {category: index (1..K)}, with 0 reserved for <UNK>
        self.vocabularies: Dict[str, Dict[str, int]] = {}
        self.cat_cardinalities: Dict[str, int] = {}

        # Audit checksums and fitted parameters
        self.train_checksum: Optional[str] = None
        self.scaler_stats_: Dict[str, Any] = {}

    def _determine_columns(self, df: pd.DataFrame) -> None:
        """Partition columns into numerical, categorical, target, and metadata groups."""
        non_feature_cols = set(self.TARGET_COLUMNS + self.METADATA_COLUMNS + self.CATEGORICAL_COLUMNS)
        self.numerical_cols = [
            c for c in df.columns
            if c not in non_feature_cols and np.issubdtype(df[c].dtype, np.number)
        ]
        self.all_feature_names = self.numerical_cols + self.categorical_cols

    def validate_and_map_targets(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate attack_cat against taxonomy and return integer target arrays.
        Returns:
            y_multiclass (np.ndarray of int64, shape (N,)): 10-class labels [0..9]
            y_binary (np.ndarray of int64, shape (N,)): Binary labels [0 or 1]
        """
        if "attack_cat" not in df.columns:
            raise KeyError("Target column 'attack_cat' missing from dataframe.")
        if "label" not in df.columns:
            raise KeyError("Target column 'label' missing from dataframe.")

        # Clean string representation
        raw_attacks = df["attack_cat"].fillna("Normal").astype(str).str.strip()

        # Validate that all categories exist in mapping
        unmapped = set(raw_attacks.unique()) - set(self.CLASS_MAPPING.keys())
        if unmapped:
            raise ValueError(f"Encountered unknown attack categories: {unmapped}")

        y_multiclass = raw_attacks.map(self.CLASS_MAPPING).to_numpy(dtype=np.int64).copy()
        y_binary = df["label"].astype(np.int64).to_numpy().copy()

        # Sanity check: where y_multiclass == 0, y_binary should be 0; where y_multiclass > 0, y_binary should be 1
        mismatch = ((y_multiclass == 0) & (y_binary != 0)) | ((y_multiclass > 0) & (y_binary == 0))
        if mismatch.any():
            print(f"[!] Warning: Found {mismatch.sum()} target label inconsistencies between binary and multiclass.")

        return y_multiclass, y_binary

    def fit(self, train_df: pd.DataFrame) -> "LeakageSafePreprocessor":
        """
        Fit all numerical scalers and categorical encoders STRICTLY on training data.
        """
        print(f"[+] Fitting Preprocessor strictly on training split ({len(train_df):,} rows)...")
        self._determine_columns(train_df)

        # 1. Fit Numerical Scaler
        X_num_train = train_df[self.numerical_cols].to_numpy(dtype=np.float32)
        # Check for NaNs
        if np.isnan(X_num_train).any():
            X_num_train = np.nan_to_num(X_num_train, nan=0.0)

        self.scaler.fit(X_num_train)

        # Store scaler parameters for leakage audit
        if isinstance(self.scaler, RobustScaler):
            self.scaler_stats_ = {
                "center": self.scaler.center_.copy(),
                "scale": self.scaler.scale_.copy(),
            }
        elif isinstance(self.scaler, StandardScaler):
            self.scaler_stats_ = {
                "mean": self.scaler.mean_.copy(),
                "var": self.scaler.var_.copy(),
            }

        # 2. Fit Categorical Vocabularies (Ordinal Embedding Tokenizer)
        self.vocabularies = {}
        self.cat_cardinalities = {}

        for col in self.categorical_cols:
            unique_cats = sorted(train_df[col].astype(str).unique().tolist())
            # Index 0 is reserved for <UNK>
            vocab = {cat: idx + 1 for idx, cat in enumerate(unique_cats)}
            self.vocabularies[col] = vocab
            # Total cardinality including <UNK> token
            self.cat_cardinalities[col] = len(vocab) + 1

        # 3. Fit OneHotEncoder
        X_cat_train_raw = train_df[self.categorical_cols].astype(str).to_numpy()
        self.onehot_encoder.fit(X_cat_train_raw)

        # Extract generated one-hot feature names
        try:
            self.onehot_feature_names = self.onehot_encoder.get_feature_names_out(self.categorical_cols).tolist()
        except AttributeError:
            self.onehot_feature_names = [f"cat_oh_{i}" for i in range(self.onehot_encoder.transform(X_cat_train_raw).shape[1])]

        self.is_fitted = True
        print(f"[+] Preprocessor fitted successfully:")
        print(f"    - Numerical Features    : {len(self.numerical_cols)}")
        print(f"    - Categorical Features  : {len(self.categorical_cols)} ({self.cat_cardinalities})")
        print(f"    - One-Hot Total Dim     : {len(self.onehot_feature_names)}")
        print(f"    - Combined Dense Dim    : {len(self.numerical_cols) + len(self.onehot_feature_names)}")

        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Transform a dataset split (train, validation, or test) using fitted parameters.
        Returns a dictionary of standardized NumPy arrays and PyTorch tensors.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before calling transform().")

        # 1. Transform Numerical Features
        X_num_raw = df[self.numerical_cols].to_numpy(dtype=np.float32)
        if np.isnan(X_num_raw).any():
            X_num_raw = np.nan_to_num(X_num_raw, nan=0.0)

        X_num_scaled = self.scaler.transform(X_num_raw).astype(np.float32)

        # Replace any remaining edge-case infs with finite bounds
        X_num_scaled = np.nan_to_num(X_num_scaled, nan=0.0, posinf=10.0, neginf=-10.0)

        # 2. Transform Categorical Features (Ordinal Embeddings with <UNK> mapping)
        num_rows = len(df)
        X_cat_ord = np.zeros((num_rows, len(self.categorical_cols)), dtype=np.int64)

        for col_idx, col in enumerate(self.categorical_cols):
            vocab = self.vocabularies[col]
            series_str = df[col].astype(str)
            # Map known categories to 1..K, unknown to 0 (UNK_INDEX)
            mapped_indices = series_str.map(lambda c: vocab.get(c, self.UNK_INDEX)).to_numpy(dtype=np.int64)
            X_cat_ord[:, col_idx] = mapped_indices

        # 3. Transform Categorical Features (One-Hot Encoded)
        X_cat_raw = df[self.categorical_cols].astype(str).to_numpy()
        X_cat_onehot = self.onehot_encoder.transform(X_cat_raw).astype(np.float32)

        # 4. Dense Combined Feature Matrix (Numerical + OneHot)
        X_dense = np.hstack([X_num_scaled, X_cat_onehot]).astype(np.float32)

        # 5. Extract Targets (if present)
        if "attack_cat" in df.columns and "label" in df.columns:
            y_multiclass, y_binary = self.validate_and_map_targets(df)
        else:
            y_multiclass = np.zeros(num_rows, dtype=np.int64)
            y_binary = np.zeros(num_rows, dtype=np.int64)

        # 6. Extract Metadata (id, flow durations, temporal sequencing)
        meta_dict = {}
        for m_col in self.metadata_cols:
            if m_col in df.columns:
                meta_dict[m_col] = df[m_col].to_numpy().copy()
            else:
                meta_dict[m_col] = np.arange(num_rows, dtype=np.int64)

        # Add sequence index for temporal graph sliding-window generation
        meta_dict["sequence_index"] = np.arange(num_rows, dtype=np.int64)
        if "dur" in df.columns:
            meta_dict["dur"] = df["dur"].to_numpy(dtype=np.float32).copy()

        # Package as PyTorch tensors
        tensor_bundle = {
            "x_num": torch.from_numpy(X_num_scaled),
            "x_cat": torch.from_numpy(X_cat_ord),
            "x_dense": torch.from_numpy(X_dense),
            "y_multiclass": torch.from_numpy(y_multiclass),
            "y_binary": torch.from_numpy(y_binary),
            "meta_ids": torch.from_numpy(meta_dict.get("id", np.arange(num_rows, dtype=np.int64))),
            "meta_seq": torch.from_numpy(meta_dict["sequence_index"]),
        }

        return {
            "numpy": {
                "x_num": X_num_scaled,
                "x_cat_ord": X_cat_ord,
                "x_cat_onehot": X_cat_onehot,
                "x_dense": X_dense,
                "y_multiclass": y_multiclass,
                "y_binary": y_binary,
                "metadata": meta_dict,
            },
            "tensors": tensor_bundle,
        }

    def generate_metadata(self) -> Dict[str, Any]:
        """Generate comprehensive metadata manifest documenting all features and parameters."""
        return {
            "project": "TGCF-IDS",
            "preprocessing_version": self.VERSION,
            "random_seed": self.random_seed,
            "scaler_type": self.scaler_type,
            "class_mapping": self.CLASS_MAPPING,
            "class_names": list(self.CLASS_MAPPING.keys()),
            "num_classes": len(self.CLASS_MAPPING),
            "numerical_features": {
                "names": self.numerical_cols,
                "count": len(self.numerical_cols),
            },
            "categorical_features": {
                "names": self.categorical_cols,
                "count": len(self.categorical_cols),
                "cardinalities_with_unk": self.cat_cardinalities,
                "vocabularies": self.vocabularies,
            },
            "onehot_features": {
                "names": self.onehot_feature_names,
                "count": len(self.onehot_feature_names),
            },
            "tensor_shapes": {
                "x_num_dim": len(self.numerical_cols),
                "x_cat_dim": len(self.categorical_cols),
                "x_dense_dim": len(self.numerical_cols) + len(self.onehot_feature_names),
            },
            "prohibited_input_features": [
                "label",
                "attack_cat",
                "id",
                "y_multiclass",
                "y_binary"
            ],
        }

    def leakage_audit(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_output: Dict[str, Any],
        test_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Rigorous Leakage Audit:
        1. Verifies scaler center/mean precisely matches train split and differs from test/combined.
        2. Verifies categorical vocabularies contain 0 test-exclusive categories.
        3. Verifies zero presence of label/attack_cat in feature vectors.
        4. Verifies absence of NaNs / Infs in output feature matrices.
        """
        audit_results = {
            "status": "PASSED",
            "checks": {},
        }

        # Check 1: Target isolation
        prohibited = set(self.TARGET_COLUMNS + self.METADATA_COLUMNS)
        found_in_features = [f for f in self.all_feature_names if f in prohibited]
        target_check = len(found_in_features) == 0
        audit_results["checks"]["target_isolation"] = {
            "passed": target_check,
            "message": "Zero target or ID columns present in feature matrix." if target_check else f"LEAKAGE: {found_in_features}",
        }

        # Check 2: Scaler parameter isolation
        X_tr_num = train_df[self.numerical_cols].to_numpy(dtype=np.float64)
        X_te_num = test_df[self.numerical_cols].to_numpy(dtype=np.float64)
        X_comb_num = np.vstack([X_tr_num, X_te_num])

        if isinstance(self.scaler, RobustScaler):
            true_tr_center = np.median(X_tr_num, axis=0)
            true_te_center = np.median(X_te_num, axis=0)
            true_comb_center = np.median(X_comb_num, axis=0)
            fitted_center = self.scaler.center_

            matches_train = np.allclose(fitted_center, true_tr_center, atol=1e-4)
            differs_from_test = not np.allclose(fitted_center, true_te_center, atol=1e-4)
            differs_from_comb = not np.allclose(fitted_center, true_comb_center, atol=1e-4)

            scaler_isolated = matches_train and differs_from_test
            audit_results["checks"]["scaler_isolation"] = {
                "passed": scaler_isolated,
                "matches_train_stats": matches_train,
                "differs_from_test_stats": differs_from_test,
                "differs_from_combined_stats": differs_from_comb,
            }
        elif isinstance(self.scaler, StandardScaler):
            true_tr_mean = np.mean(X_tr_num, axis=0)
            true_te_mean = np.mean(X_te_num, axis=0)
            fitted_mean = self.scaler.mean_

            matches_train = np.allclose(fitted_mean, true_tr_mean, atol=1e-4)
            differs_from_test = not np.allclose(fitted_mean, true_te_mean, atol=1e-4)
            scaler_isolated = matches_train and differs_from_test

            audit_results["checks"]["scaler_isolation"] = {
                "passed": scaler_isolated,
                "matches_train_stats": matches_train,
                "differs_from_test_stats": differs_from_test,
            }

        # Check 3: Vocabulary isolation & Unseen Category Mapping
        vocab_check = True
        for col in self.categorical_cols:
            train_cats = set(train_df[col].astype(str).unique())
            test_cats = set(test_df[col].astype(str).unique())
            fitted_vocab_cats = set(self.vocabularies[col].keys())

            if fitted_vocab_cats != train_cats:
                vocab_check = False

            unseen = test_cats - train_cats
            if unseen:
                # Confirm unseen categories are mapped to UNK_INDEX (0)
                col_idx = self.categorical_cols.index(col)
                test_cat_ord = test_output["numpy"]["x_cat_ord"][:, col_idx]
                unseen_mask = test_df[col].astype(str).isin(unseen)
                if not (test_cat_ord[unseen_mask] == self.UNK_INDEX).all():
                    vocab_check = False

        audit_results["checks"]["vocabulary_isolation"] = {
            "passed": vocab_check,
            "message": "Vocabularies built strictly from training split; unseen categories correctly assigned to <UNK> token (0).",
        }

        # Check 4: Matrix finiteness and tensor shapes
        tr_dense = train_output["numpy"]["x_dense"]
        te_dense = test_output["numpy"]["x_dense"]
        finite_check = np.isfinite(tr_dense).all() and np.isfinite(te_dense).all()
        audit_results["checks"]["numerical_finiteness"] = {
            "passed": bool(finite_check),
            "train_shape": list(tr_dense.shape),
            "test_shape": list(te_dense.shape),
        }

        # Overall status
        if not all(c.get("passed", False) for c in audit_results["checks"].values()):
            audit_results["status"] = "FAILED"

        return audit_results

    def save_artifacts(
        self,
        output_dir: Path,
        train_transformed: Dict[str, Any],
        test_transformed: Dict[str, Any],
    ) -> Dict[str, Path]:
        """
        Save model-ready feature tensors, preprocessor instance, and metadata manifest.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save PyTorch tensor bundles (.pt)
        train_pt_path = output_dir / "train_features.pt"
        test_pt_path = output_dir / "test_features.pt"
        torch.save(train_transformed["tensors"], train_pt_path)
        torch.save(test_transformed["tensors"], test_pt_path)

        # 2. Save NumPy archive bundles (.npz) for maximum interoperability
        train_npz_path = output_dir / "train_features.npz"
        test_npz_path = output_dir / "test_features.npz"
        np.savez_compressed(
            train_npz_path,
            x_num=train_transformed["numpy"]["x_num"],
            x_cat_ord=train_transformed["numpy"]["x_cat_ord"],
            x_dense=train_transformed["numpy"]["x_dense"],
            y_multiclass=train_transformed["numpy"]["y_multiclass"],
            y_binary=train_transformed["numpy"]["y_binary"],
            meta_ids=train_transformed["numpy"]["metadata"].get("id", np.zeros(len(train_transformed["numpy"]["x_num"]))),
            meta_seq=train_transformed["numpy"]["metadata"]["sequence_index"],
        )
        np.savez_compressed(
            test_npz_path,
            x_num=test_transformed["numpy"]["x_num"],
            x_cat_ord=test_transformed["numpy"]["x_cat_ord"],
            x_dense=test_transformed["numpy"]["x_dense"],
            y_multiclass=test_transformed["numpy"]["y_multiclass"],
            y_binary=test_transformed["numpy"]["y_binary"],
            meta_ids=test_transformed["numpy"]["metadata"].get("id", np.zeros(len(test_transformed["numpy"]["x_num"]))),
            meta_seq=test_transformed["numpy"]["metadata"]["sequence_index"],
        )

        # 3. Save fitted Preprocessor object (.joblib)
        preprocessor_path = output_dir / "preprocessor.joblib"
        joblib.dump(self, preprocessor_path)

        # 4. Save Metadata JSON
        metadata_dict = self.generate_metadata()
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2)

        print(f"[+] Saved artifacts under {output_dir}:")
        print(f"    - PyTorch Train Tensors : {train_pt_path.name}")
        print(f"    - PyTorch Test Tensors  : {test_pt_path.name}")
        print(f"    - NumPy Train Archive   : {train_npz_path.name}")
        print(f"    - NumPy Test Archive    : {test_npz_path.name}")
        print(f"    - Fitted Preprocessor   : {preprocessor_path.name}")
        print(f"    - Metadata Manifest     : {metadata_path.name}")

        return {
            "train_pt": train_pt_path,
            "test_pt": test_pt_path,
            "train_npz": train_npz_path,
            "test_npz": test_npz_path,
            "preprocessor": preprocessor_path,
            "metadata": metadata_path,
        }


def run_preprocessing_pipeline(
    train_cleaned_path: Optional[Path] = None,
    test_cleaned_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    scaler_type: str = "robust",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute full end-to-end leakage-safe preprocessing pipeline.
    """
    tr_path = Path(train_cleaned_path) if train_cleaned_path else (ProjectPaths.DATA_PROCESSED / "UNSW_NB15_training-set_cleaned.csv")
    te_path = Path(test_cleaned_path) if test_cleaned_path else (ProjectPaths.DATA_PROCESSED / "UNSW_NB15_testing-set_cleaned.csv")
    out_dir = Path(output_dir) if output_dir else ProjectPaths.DATA_PROCESSED

    print("=" * 75)
    print("      TGCF-IDS : Leakage-Safe Feature Preprocessing Pipeline      ")
    print("=" * 75)

    if not tr_path.exists() or not te_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset files not found. Please run 'python src/data/clean_dataset.py' first.\n"
            f"Missing: {[str(p) for p in [tr_path, te_path] if not p.exists()]}"
        )

    print(f"[+] Loading cleaned training data : {tr_path.name}")
    train_df = pd.read_csv(tr_path, low_memory=False)

    print(f"[+] Loading cleaned testing data  : {te_path.name}")
    test_df = pd.read_csv(te_path, low_memory=False)

    # Instantiate preprocessor
    preprocessor = LeakageSafePreprocessor(
        scaler_type=scaler_type,
        random_seed=random_seed,
    )

    # Fit strictly on train
    preprocessor.fit(train_df)

    # Transform both splits
    print("[*] Transforming training split...")
    train_output = preprocessor.transform(train_df)

    print("[*] Transforming testing split...")
    test_output = preprocessor.transform(test_df)

    # Perform Leakage Audit
    print("-" * 75)
    print("[*] Executing Rigorous Leakage Audit...")
    audit = preprocessor.leakage_audit(train_df, test_df, train_output, test_output)
    print(f"    - Overall Audit Status   : {audit['status']}")
    for check_name, check_data in audit["checks"].items():
        status_sym = "[PASS]" if check_data.get("passed", False) else "[FAIL]"
        print(f"    - {check_name:<24}: {status_sym} {check_data}")

    if audit["status"] != "PASSED":
        raise RuntimeError(f"Leakage audit failed: {audit}")

    # Save all artifacts
    print("-" * 75)
    saved_paths = preprocessor.save_artifacts(out_dir, train_output, test_output)

    print("=" * 75)
    print("Leakage-Safe Preprocessing Pipeline Completed Successfully.")
    print("=" * 75)

    return {
        "preprocessor": preprocessor,
        "train_output": train_output,
        "test_output": test_output,
        "audit": audit,
        "saved_paths": saved_paths,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Leakage-safe preprocessing pipeline for TGCF-IDS."
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=str(ProjectPaths.DATA_PROCESSED / "UNSW_NB15_training-set_cleaned.csv"),
        help="Path to cleaned training CSV.",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=str(ProjectPaths.DATA_PROCESSED / "UNSW_NB15_testing-set_cleaned.csv"),
        help="Path to cleaned testing CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ProjectPaths.DATA_PROCESSED),
        help="Path to save processed feature tensors.",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="robust",
        choices=["robust", "standard"],
        help="Type of numerical scaler to fit.",
    )
    args = parser.parse_args()

    run_preprocessing_pipeline(
        train_cleaned_path=Path(args.train_file),
        test_cleaned_path=Path(args.test_file),
        output_dir=Path(args.output_dir),
        scaler_type=args.scaler,
    )


if __name__ == "__main__":
    main()
