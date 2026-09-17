#!/usr/bin/env python3
"""
TGCF-IDS: Traditional Machine Learning Baselines Module.
Evaluates Random Forest, Logistic Regression, and XGBoost on leakage-safe preprocessed features
across multiple random seeds with rigorous multi-class metrics, FPR/FNR, and confusion matrices.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Optional XGBoost import with safe fallback
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Ensure project root is accessible
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.paths import ProjectPaths


class TraditionalBaselinesEvaluator:
    """
    Evaluator for standard traditional ML baselines on preprocessed tabular intrusion detection features.
    """

    CLASS_NAMES = [
        "Normal", "Analysis", "Backdoor", "DoS", "Exploits",
        "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
    ]
    NUM_CLASSES = 10

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        figures_dir: Optional[Path] = None,
        seeds: List[int] = [42, 123, 456],
        n_jobs: int = -1,
    ):
        self.data_dir = Path(data_dir) if data_dir else ProjectPaths.DATA_PROCESSED
        self.results_dir = Path(results_dir) if results_dir else (ProjectPaths.RESULTS_TABLES)
        self.figures_dir = Path(figures_dir) if figures_dir else (ProjectPaths.RESULTS_FIGURES / "baselines")
        self.seeds = seeds
        self.n_jobs = n_jobs

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    def load_preprocessed_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load preprocessed feature matrices and targets from npz archives."""
        train_npz = self.data_dir / "train_features.npz"
        test_npz = self.data_dir / "test_features.npz"

        if not train_npz.exists() or not test_npz.exists():
            raise FileNotFoundError(
                f"Preprocessed feature files not found under {self.data_dir}.\n"
                f"Please run 'python src/preprocessing/preprocessor.py' first."
            )

        print(f"[+] Loading preprocessed training archive : {train_npz.name}")
        train_data = np.load(train_npz)
        self.X_train = train_data["x_dense"]
        self.y_train = train_data["y_multiclass"]

        print(f"[+] Loading preprocessed testing archive  : {test_npz.name}")
        test_data = np.load(test_npz)
        self.X_test = test_data["x_dense"]
        self.y_test = test_data["y_multiclass"]

        print(f"[+] Loaded Features: X_train={self.X_train.shape}, y_train={self.y_train.shape}")
        print(f"[+] Loaded Features: X_test={self.X_test.shape}, y_test={self.y_test.shape}")

        return self.X_train, self.y_train, self.X_test, self.y_test

    def compute_evaluation_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute multi-class macro, weighted, per-class metrics, and binary-equivalent FPR/FNR.
        """
        acc = accuracy_score(y_true, y_pred)
        macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Per-class metrics
        per_class_p = precision_score(y_true, y_pred, average=None, labels=list(range(self.NUM_CLASSES)), zero_division=0)
        per_class_r = recall_score(y_true, y_pred, average=None, labels=list(range(self.NUM_CLASSES)), zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(self.NUM_CLASSES)), zero_division=0)

        # Confusion matrix (10x10)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.NUM_CLASSES)))

        # Multi-class macro FPR and FNR (One-vs-Rest)
        fpr_list = []
        fnr_list = []
        for i in range(self.NUM_CLASSES):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)

            fpr_i = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr_i = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            fpr_list.append(fpr_i)
            fnr_list.append(fnr_i)

        macro_fpr = float(np.mean(fpr_list))
        macro_fnr = float(np.mean(fnr_list))

        # Binary equivalent (Class 0: Benign vs Classes 1-9: Attack)
        y_true_bin = (y_true > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)
        cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        tn_b, fp_b, fn_b, tp_b = cm_bin.ravel()

        binary_fpr = fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0.0
        binary_fnr = fn_b / (fn_b + tp_b) if (fn_b + tp_b) > 0 else 0.0
        binary_acc = (tp_b + tn_b) / len(y_true_bin)

        return {
            "accuracy": float(acc),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "macro_fpr": macro_fpr,
            "macro_fnr": macro_fnr,
            "binary_fpr": float(binary_fpr),
            "binary_fnr": float(binary_fnr),
            "binary_accuracy": float(binary_acc),
            "per_class_precision": per_class_p.tolist(),
            "per_class_recall": per_class_r.tolist(),
            "per_class_f1": per_class_f1.tolist(),
            "confusion_matrix": cm,
        }

    def train_and_evaluate_model(
        self, model_name: str, seed: int
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Instantiate model with specific random seed, fit on training split, and evaluate on test split.
        """
        if model_name == "logistic_regression":
            model = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=seed,
            )
        elif model_name == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=25,
                min_samples_split=5,
                random_state=seed,
                n_jobs=self.n_jobs,
            )
        elif model_name == "xgboost":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost is not installed.")
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=self.n_jobs,
                tree_method="hist",
                eval_metric="mlogloss",
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        print(f"[*] Training {model_name.upper()} [Seed={seed}] on {len(self.X_train):,} samples...")
        model.fit(self.X_train, self.y_train)

        print(f"[*] Evaluating {model_name.upper()} [Seed={seed}] on {len(self.X_test):,} test samples...")
        y_pred = model.predict(self.X_test)

        metrics = self.compute_evaluation_metrics(self.y_test, y_pred)
        return metrics, y_pred

    def plot_confusion_matrix(
        self, cm: np.ndarray, model_display_name: str, output_filename: str
    ) -> Path:
        """
        Generate high-resolution publication-quality 10x10 normalized confusion matrix heatmap.
        """
        # Normalize by true label row sums
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

        fig, ax = plt.subplots(figsize=(10, 8.5), dpi=300)
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.CLASS_NAMES,
            yticklabels=self.CLASS_NAMES,
            cbar_kws={"label": "Recall (Normalized Proportion)"},
            linewidths=0.5,
            ax=ax,
        )

        ax.set_title(f"Confusion Matrix: {model_display_name} (Multi-Class Test Split)", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("True Ground-Truth Class", fontsize=11)
        plt.xticks(rotation=40, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        out_path = self.figures_dir / output_filename
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def run_all_baselines(self) -> Dict[str, Any]:
        """
        Execute full benchmark across all models and random seeds.
        Computes Mean ± Standard Deviation, generates confusion matrices, and exports summary tables.
        """
        print("=" * 75)
        print("      TGCF-IDS : Traditional Machine Learning Baseline Benchmark       ")
        print("=" * 75)

        self.load_preprocessed_data()

        models_to_run = ["logistic_regression", "random_forest"]
        if HAS_XGBOOST:
            models_to_run.append("xgboost")
        else:
            print("[!] Notice: XGBoost not available, running Logistic Regression and Random Forest.")

        all_results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models_to_run}
        last_cms: Dict[str, np.ndarray] = {}

        # 1. Run across all seeds
        for model_name in models_to_run:
            print(f"\n==================== Running Model: {model_name.upper()} ====================")
            for seed in self.seeds:
                metrics, y_pred = self.train_and_evaluate_model(model_name, seed)
                metrics["seed"] = seed
                metrics["model"] = model_name
                all_results[model_name].append(metrics)
                last_cms[model_name] = metrics["confusion_matrix"]
                print(f"    -> Seed {seed:3d} | Acc: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Weighted F1: {metrics['weighted_f1']:.4f} | Bin FPR: {metrics['binary_fpr']:.4f}")

        # 2. Compile Aggregated Summary Tables
        summary_rows = []
        raw_run_rows = []
        per_class_summary_rows = []

        for model_name, seed_metrics_list in all_results.items():
            for m in seed_metrics_list:
                raw_run_rows.append({
                    "model": model_name,
                    "seed": m["seed"],
                    "accuracy": m["accuracy"],
                    "macro_precision": m["macro_precision"],
                    "macro_recall": m["macro_recall"],
                    "macro_f1": m["macro_f1"],
                    "weighted_f1": m["weighted_f1"],
                    "macro_fpr": m["macro_fpr"],
                    "macro_fnr": m["macro_fnr"],
                    "binary_fpr": m["binary_fpr"],
                    "binary_fnr": m["binary_fnr"],
                })

            accs = [m["accuracy"] for m in seed_metrics_list]
            m_precs = [m["macro_precision"] for m in seed_metrics_list]
            m_recs = [m["macro_recall"] for m in seed_metrics_list]
            m_f1s = [m["macro_f1"] for m in seed_metrics_list]
            w_f1s = [m["weighted_f1"] for m in seed_metrics_list]
            m_fprs = [m["macro_fpr"] for m in seed_metrics_list]
            m_fnrs = [m["macro_fnr"] for m in seed_metrics_list]
            b_fprs = [m["binary_fpr"] for m in seed_metrics_list]
            b_fnrs = [m["binary_fnr"] for m in seed_metrics_list]

            summary_rows.append({
                "Model": model_name.replace("_", " ").title(),
                "Accuracy": f"{np.mean(accs):.4f} ± {np.std(accs):.4f}",
                "Macro Precision": f"{np.mean(m_precs):.4f} ± {np.std(m_precs):.4f}",
                "Macro Recall": f"{np.mean(m_recs):.4f} ± {np.std(m_recs):.4f}",
                "Macro F1": f"{np.mean(m_f1s):.4f} ± {np.std(m_f1s):.4f}",
                "Weighted F1": f"{np.mean(w_f1s):.4f} ± {np.std(w_f1s):.4f}",
                "Macro FPR": f"{np.mean(m_fprs):.4f} ± {np.std(m_fprs):.4f}",
                "Macro FNR": f"{np.mean(m_fnrs):.4f} ± {np.std(m_fnrs):.4f}",
                "Binary FPR": f"{np.mean(b_fprs):.4f} ± {np.std(b_fprs):.4f}",
                "Binary FNR": f"{np.mean(b_fnrs):.4f} ± {np.std(b_fnrs):.4f}",
                "Acc_mean": np.mean(accs),
                "Acc_std": np.std(accs),
                "MacroF1_mean": np.mean(m_f1s),
                "MacroF1_std": np.std(m_f1s),
                "WeightedF1_mean": np.mean(w_f1s),
                "WeightedF1_std": np.std(w_f1s),
            })

            # Per-class aggregation across seeds
            per_class_f1_matrix = np.array([m["per_class_f1"] for m in seed_metrics_list])
            per_class_rec_matrix = np.array([m["per_class_recall"] for m in seed_metrics_list])
            per_class_prec_matrix = np.array([m["per_class_precision"] for m in seed_metrics_list])

            for c_idx, c_name in enumerate(self.CLASS_NAMES):
                per_class_summary_rows.append({
                    "Model": model_name.replace("_", " ").title(),
                    "Class ID": c_idx,
                    "Class Name": c_name,
                    "Precision": f"{np.mean(per_class_prec_matrix[:, c_idx]):.4f} ± {np.std(per_class_prec_matrix[:, c_idx]):.4f}",
                    "Recall": f"{np.mean(per_class_rec_matrix[:, c_idx]):.4f} ± {np.std(per_class_rec_matrix[:, c_idx]):.4f}",
                    "F1 Score": f"{np.mean(per_class_f1_matrix[:, c_idx]):.4f} ± {np.std(per_class_f1_matrix[:, c_idx]):.4f}",
                    "F1_mean": float(np.mean(per_class_f1_matrix[:, c_idx])),
                })

        summary_df = pd.DataFrame(summary_rows)
        raw_runs_df = pd.DataFrame(raw_run_rows)
        per_class_df = pd.DataFrame(per_class_summary_rows)

        # 3. Save Tables
        summary_table_path = self.results_dir / "traditional_baselines.csv"
        raw_runs_path = self.results_dir / "traditional_baselines_raw_runs.csv"
        per_class_path = self.results_dir / "traditional_baselines_per_class.csv"

        summary_df.to_csv(summary_table_path, index=False)
        raw_runs_df.to_csv(raw_runs_path, index=False)
        per_class_df.to_csv(per_class_path, index=False)

        print("-" * 75)
        print(f"[+] Saved baseline summary table     -> {summary_table_path}")
        print(f"[+] Saved per-class baseline metrics -> {per_class_path}")
        print(f"[+] Saved individual seed runs       -> {raw_runs_path}")

        # 4. Generate Confusion Matrix Figures
        print("-" * 75)
        print("[*] Rendering Confusion Matrix Visualizations...")
        for model_name, cm in last_cms.items():
            disp_name = model_name.replace("_", " ").title()
            fig_name = f"{model_name}_confusion_matrix.png"
            fig_path = self.plot_confusion_matrix(cm, disp_name, fig_name)
            print(f"    - Saved: {fig_path.name}")

        # Print Final Comparison Console Table
        print("\n" + "=" * 75)
        print("          TRADITIONAL MACHINE LEARNING BASELINE BENCHMARK RESULTS       ")
        print("=" * 75)
        print(summary_df[["Model", "Accuracy", "Macro F1", "Weighted F1", "Binary FPR", "Binary FNR"]].to_string(index=False))
        print("=" * 75)

        return {
            "summary_df": summary_df,
            "per_class_df": per_class_df,
            "raw_runs_df": raw_runs_df,
            "summary_path": summary_table_path,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run traditional machine learning baselines for TGCF-IDS."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(ProjectPaths.DATA_PROCESSED),
        help="Directory containing preprocessed feature archives.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="List of random seeds to evaluate over.",
    )
    args = parser.parse_args()

    evaluator = TraditionalBaselinesEvaluator(
        data_dir=Path(args.data_dir),
        seeds=args.seeds,
    )
    evaluator.run_all_baselines()


if __name__ == "__main__":
    main()
