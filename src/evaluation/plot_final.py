#!/usr/bin/env python3
"""
TGCF-IDS: Publication-Quality Final Test Set Visualizations.

Produces high-resolution figures in results/figures/final/:
1. final_confusion_matrix.png: Raw count multi-class confusion matrix.
2. final_confusion_matrix_normalized.png: Row-normalized confusion matrix (True Positives / Recall).
3. final_per_class_metrics.png: Grouped bar chart of Precision, Recall, and F1 per class with support annotations.
4. final_roc_pr_summary.png: Multi-class ROC-AUC and PR-AUC distributions across all 10 attack classes.
5. final_comprehensive_dashboard.png: 4-panel publication composite dashboard labeled "FINAL TEST RESULTS".
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.paths import ProjectPaths


class FinalFigureGenerator:
    """
    Renders publication-ready figures for final test set evaluation results.
    """

    CLASS_NAMES = [
        "Normal", "Analysis", "Backdoor", "DoS", "Exploits",
        "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
    ]

    PALETTE = {
        "primary": "#1D3557",
        "secondary": "#457B9D",
        "accent": "#2A9D8F",
        "highlight": "#E76F51",
        "warning": "#E63946",
        "precision": "#2A9D8F",
        "recall": "#E76F51",
        "f1": "#1D3557",
    }

    def __init__(
        self,
        final_dir: Optional[Union[str, Path]] = None,
        figures_dir: Optional[Union[str, Path]] = None,
    ):
        self.final_dir = Path(final_dir or ProjectPaths.RESULTS_FINAL)
        self.figures_dir = Path(figures_dir or ProjectPaths.RESULTS_FIGURES / "final")
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_json_path = self.final_dir / "final_metrics.json"
        self.report_csv_path = self.final_dir / "classification_report.csv"
        self.cm_raw_csv_path = self.final_dir / "confusion_matrix.csv"
        self.cm_norm_csv_path = self.final_dir / "confusion_matrix_normalized.csv"

        self._set_publication_style()

    def _set_publication_style(self) -> None:
        """Configure crisp academic plotting aesthetics."""
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "figure.titlesize": 15,
            "figure.titleweight": "bold",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        })

    def load_data(self) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load final metrics JSON and CSV tables."""
        if not self.metrics_json_path.exists():
            raise FileNotFoundError(f"Missing {self.metrics_json_path}. Run final_evaluation.py first.")
        if not self.report_csv_path.exists():
            raise FileNotFoundError(f"Missing {self.report_csv_path}.")
        if not self.cm_raw_csv_path.exists():
            raise FileNotFoundError(f"Missing {self.cm_raw_csv_path}.")
        if not self.cm_norm_csv_path.exists():
            raise FileNotFoundError(f"Missing {self.cm_norm_csv_path}.")

        with open(self.metrics_json_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        report_df = pd.read_csv(self.report_csv_path)
        cm_raw_df = pd.read_csv(self.cm_raw_csv_path, index_col=0)
        cm_norm_df = pd.read_csv(self.cm_norm_csv_path, index_col=0)

        return metrics, report_df, cm_raw_df, cm_norm_df

    def plot_confusion_matrix_raw(self, cm_df: pd.DataFrame) -> Path:
        """1. Raw Count Multi-Class Confusion Matrix Heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8.5))

        sns.heatmap(
            cm_df,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            cbar_kws={"label": "Number of Test Flows"},
            linewidths=0.5,
            linecolor="#E2E8F0",
            ax=ax,
        )

        ax.set_title("TGCF-IDS: Multi-Class Confusion Matrix (Raw Flow Counts)\n[FINAL TEST RESULTS — FROZEN MODEL]", pad=15)
        ax.set_xlabel("Predicted Class", fontweight="bold")
        ax.set_ylabel("True Class (Ground Truth)", fontweight="bold")
        plt.xticks(rotation=30, ha="right")
        plt.yticks(rotation=0)

        out_path = self.figures_dir / "final_confusion_matrix.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_confusion_matrix_normalized(self, cm_norm_df: pd.DataFrame) -> Path:
        """2. Normalized Confusion Matrix Heatmap (Recall / Class Accuracy)."""
        fig, ax = plt.subplots(figsize=(10, 8.5))

        sns.heatmap(
            cm_norm_df,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            cbar=True,
            cbar_kws={"label": "Normalized Ratio (Recall)"},
            linewidths=0.5,
            linecolor="#E2E8F0",
            ax=ax,
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_title("TGCF-IDS: Normalized Multi-Class Confusion Matrix (True Class Recall)\n[FINAL TEST RESULTS — FROZEN MODEL]", pad=15)
        ax.set_xlabel("Predicted Class", fontweight="bold")
        ax.set_ylabel("True Class (Ground Truth)", fontweight="bold")
        plt.xticks(rotation=30, ha="right")
        plt.yticks(rotation=0)

        out_path = self.figures_dir / "final_confusion_matrix_normalized.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_per_class_metrics(self, report_df: pd.DataFrame) -> Path:
        """3. Grouped Bar Chart of Precision, Recall, and F1 Per Class."""
        fig, ax = plt.subplots(figsize=(13, 6.5))

        classes = report_df["Class Name"].tolist()
        x = np.arange(len(classes))
        width = 0.26

        p_bars = ax.bar(x - width, report_df["Precision"], width, label="Precision", color=self.PALETTE["precision"], alpha=0.9, edgecolor="#1D3557")
        r_bars = ax.bar(x, report_df["Recall"], width, label="Recall", color=self.PALETTE["recall"], alpha=0.9, edgecolor="#1D3557")
        f_bars = ax.bar(x + width, report_df["F1 Score"], width, label="F1-Score", color=self.PALETTE["f1"], alpha=0.9, edgecolor="#1D3557")

        for i, (p, r, f1, sup) in enumerate(zip(report_df["Precision"], report_df["Recall"], report_df["F1 Score"], report_df["Support"])):
            max_val = max(p, r, f1)
            ax.annotate(f"N={sup:,}", (i, max_val + 0.04), ha="center", fontsize=7.5, fontweight="bold", color="#4A5568")

        ax.set_title("TGCF-IDS: Per-Class Precision, Recall, and F1-Score on Final Test Set\n[FINAL TEST RESULTS — FROZEN MODEL]", pad=15)
        ax.set_xlabel("UNSW-NB15 Traffic Class", fontweight="bold")
        ax.set_ylabel("Score", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=10.5)
        ax.set_ylim(0.0, 1.15)
        ax.legend(frameon=True, loc="upper right")

        out_path = self.figures_dir / "final_per_class_metrics.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_roc_pr_summary(self, metrics: Dict[str, Any]) -> Path:
        """4. Multi-Class ROC-AUC and PR-AUC Horizontal Bar Chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        roc_dict = metrics.get("per_class_roc_auc", {})
        pr_dict = metrics.get("per_class_pr_auc", {})

        classes = list(roc_dict.keys())
        roc_scores = [roc_dict[c] for c in classes]
        pr_scores = [pr_dict.get(c, 0.0) for c in classes]

        y_pos = np.arange(len(classes))

        # ROC-AUC Bar Chart
        ax1.barh(y_pos, roc_scores, color=self.PALETTE["secondary"], edgecolor="#1D3557", alpha=0.85)
        ax1.axvline(metrics["overall_metrics"]["macro_roc_auc"], color="#E63946", linestyle="--", linewidth=1.5,
                    label=f"Macro Mean: {metrics['overall_metrics']['macro_roc_auc']:.4f}")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(classes, fontweight="bold")
        ax1.set_xlabel("One-vs-Rest ROC-AUC", fontweight="bold")
        ax1.set_title("(a) Multi-Class ROC-AUC Scores", pad=10)
        ax1.set_xlim(0.5, 1.02)
        ax1.legend(loc="lower right")

        # PR-AUC Bar Chart
        ax2.barh(y_pos, pr_scores, color=self.PALETTE["accent"], edgecolor="#1D3557", alpha=0.85)
        ax2.axvline(metrics["overall_metrics"]["macro_pr_auc"], color="#E63946", linestyle="--", linewidth=1.5,
                    label=f"Macro Mean: {metrics['overall_metrics']['macro_pr_auc']:.4f}")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(classes, fontweight="bold")
        ax2.set_xlabel("Average Precision (PR-AUC)", fontweight="bold")
        ax2.set_title("(b) Multi-Class Precision-Recall AUC Scores", pad=10)
        ax2.set_xlim(0.0, 1.02)
        ax2.legend(loc="lower right")

        fig.suptitle("TGCF-IDS: Area Under Curve (AUC) Characteristics on Final Test Set\n[FINAL TEST RESULTS — FROZEN MODEL]", y=1.02, fontsize=14, fontweight="bold")

        out_path = self.figures_dir / "final_roc_pr_summary.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_comprehensive_dashboard(
        self,
        metrics: Dict[str, Any],
        report_df: pd.DataFrame,
        cm_norm_df: pd.DataFrame,
    ) -> Path:
        """5. 4-Panel Publication Composite Dashboard."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # (a) Overall Summary Metrics Bar Chart
        core_labels = ["Accuracy", "Weighted F1", "Macro F1", "Macro Precision", "Macro Recall", "Macro ROC-AUC"]
        ov = metrics["overall_metrics"]
        core_vals = [
            ov["accuracy"],
            ov["weighted_f1"],
            ov["macro_f1"],
            ov["macro_precision"],
            ov["macro_recall"],
            ov["macro_roc_auc"],
        ]
        x_pos = np.arange(len(core_labels))
        bars = ax1.bar(x_pos, core_vals, 0.45, color=self.PALETTE["secondary"], edgecolor="#1D3557", alpha=0.85)
        for b in bars:
            height = b.get_height()
            ax1.annotate(f"{height:.4f}", (b.get_x() + b.get_width() / 2, height + 0.015),
                         ha="center", fontsize=9.5, fontweight="bold")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(core_labels, rotation=25, ha="right", fontsize=9.5, fontweight="bold")
        ax1.set_ylabel("Score", fontweight="bold")
        ax1.set_ylim(0.0, 1.10)
        ax1.set_title("(a) Primary Global Test Set Performance Metrics", pad=10)

        # (b) Normalized Confusion Matrix Heatmap
        sns.heatmap(
            cm_norm_df,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=True,
            ax=ax2,
            annot_kws={"size": 8},
        )
        ax2.set_title("(b) Normalized Multi-Class Confusion Matrix (Recall)", pad=10)
        ax2.set_xlabel("Predicted", fontweight="bold")
        ax2.set_ylabel("True", fontweight="bold")
        ax2.tick_params(axis="x", rotation=30)

        # (c) Per-Class F1-Score Breakdown
        classes = report_df["Class Name"].tolist()
        f1_scores = report_df["F1 Score"].tolist()
        ax3.bar(np.arange(len(classes)), f1_scores, color=self.PALETTE["accent"], edgecolor="#1D3557", width=0.55)
        ax3.axhline(ov["macro_f1"], color="#E63946", linestyle="--", linewidth=1.5,
                    label=f"Macro F1: {ov['macro_f1']:.4f}")
        for i, val in enumerate(f1_scores):
            ax3.annotate(f"{val:.3f}", (i, val + 0.02), ha="center", fontsize=8, fontweight="bold")
        ax3.set_xticks(np.arange(len(classes)))
        ax3.set_xticklabels(classes, rotation=25, ha="right", fontsize=9)
        ax3.set_ylabel("F1 Score", fontweight="bold")
        ax3.set_ylim(0.0, 1.12)
        ax3.set_title("(c) Multi-Class F1-Score Breakdown", pad=10)
        ax3.legend(loc="lower right")

        # (d) Operational Security Rates & Computational Efficiency
        sec_labels = ["False Alarm Rate (FPR)", "Missed Attack Rate (FNR)"]
        sec_vals = [ov["binary_fpr"] * 100.0, ov["binary_fnr"] * 100.0]
        ax4.bar(np.arange(len(sec_labels)), sec_vals, 0.4, color=["#2A9D8F", "#E63946"], edgecolor="#1D3557", alpha=0.85)
        for i, v in enumerate(sec_vals):
            ax4.annotate(f"{v:.2f}%", (i, v + 0.05), ha="center", fontsize=10, fontweight="bold")
        ax4.set_xticks(np.arange(len(sec_labels)))
        ax4.set_xticklabels(sec_labels, fontsize=10, fontweight="bold")
        ax4.set_ylabel("Error Rate (%)", fontweight="bold")
        ax4.set_ylim(0.0, max(max(sec_vals) * 1.35, 2.0))
        ax4.set_title(f"(d) Security Rates (Latency: {metrics['computational_efficiency']['latency_ms_per_1000_flows']:.2f} ms/1k flows)", pad=10)

        fig.suptitle("TGCF-IDS : COMPREHENSIVE FINAL TEST SET EVALUATION DASHBOARD\n[FINAL TEST RESULTS — FROZEN MODEL — ZERO TEST LEAKAGE]",
                     fontsize=15, fontweight="bold", y=0.99)

        out_path = self.figures_dir / "final_comprehensive_dashboard.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def generate_all_figures(self) -> List[Path]:
        """Generate all final evaluation figures."""
        print("[*] Generating FINAL TEST RESULTS Visualizations...")
        metrics, report_df, cm_raw_df, cm_norm_df = self.load_data()

        generated = []
        p1 = self.plot_confusion_matrix_raw(cm_raw_df)
        print(f"    [+] Saved: {p1.name}")
        generated.append(p1)

        p2 = self.plot_confusion_matrix_normalized(cm_norm_df)
        print(f"    [+] Saved: {p2.name}")
        generated.append(p2)

        p3 = self.plot_per_class_metrics(report_df)
        print(f"    [+] Saved: {p3.name}")
        generated.append(p3)

        p4 = self.plot_roc_pr_summary(metrics)
        print(f"    [+] Saved: {p4.name}")
        generated.append(p4)

        p5 = self.plot_comprehensive_dashboard(metrics, report_df, cm_norm_df)
        print(f"    [+] Saved: {p5.name}")
        generated.append(p5)

        return generated


def main():
    generator = FinalFigureGenerator()
    generator.generate_all_figures()


if __name__ == "__main__":
    main()
