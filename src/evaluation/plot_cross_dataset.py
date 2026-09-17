#!/usr/bin/env python3
"""
TGCF-IDS: Publication-Quality Cross-Dataset Visualizations.

Generates high-resolution figures under results/figures/cross_dataset/:
1. cross_dataset_domain_comparison.png: In-Domain vs Zero-Shot vs Fine-Tuned macro performance metrics.
2. cross_dataset_confusion_matrices.png: Side-by-side normalized confusion matrices (Zero-Shot vs Fine-Tuned).
3. cross_dataset_per_class_transfer.png: 10-class transferability and recovery bar chart.
4. cross_dataset_comprehensive_dashboard.png: 4-panel publication composite dashboard.
"""

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


class CrossDatasetFigureGenerator:
    """
    Renders publication-ready figures for cross-dataset generalization evaluation.
    """

    CLASS_NAMES = [
        "Normal", "Analysis", "Backdoor", "DoS", "Exploits",
        "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms"
    ]

    PALETTE = {
        "in_domain": "#1D3557",
        "zero_shot": "#E76F51",
        "fine_tuned": "#2A9D8F",
        "neutral": "#457B9D",
    }

    def __init__(
        self,
        summary_csv: Optional[Union[str, Path]] = None,
        per_class_csv: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.summary_csv = Path(summary_csv or ProjectPaths.RESULTS_TABLES / "cross_dataset.csv")
        self.per_class_csv = Path(per_class_csv or ProjectPaths.RESULTS_TABLES / "cross_dataset_per_class.csv")
        self.output_dir = Path(output_dir or ProjectPaths.RESULTS_FIGURES / "cross_dataset")
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load cross dataset tables."""
        if not self.summary_csv.exists():
            raise FileNotFoundError(f"Missing summary CSV: {self.summary_csv}")
        if not self.per_class_csv.exists():
            raise FileNotFoundError(f"Missing per-class CSV: {self.per_class_csv}")

        summary_df = pd.read_csv(self.summary_csv)
        per_class_df = pd.read_csv(self.per_class_csv)
        return summary_df, per_class_df

    def plot_domain_comparison(self, summary_df: pd.DataFrame) -> Path:
        """1. Grouped Bar Chart of Core Metrics across Evaluation Modes."""
        fig, ax = plt.subplots(figsize=(12, 6.5))

        metrics = ["Accuracy", "Weighted F1", "Macro F1", "Macro Precision", "Macro Recall"]
        modes = summary_df["Evaluation Mode"].tolist()

        x = np.arange(len(metrics))
        width = 0.25

        colors = [self.PALETTE["in_domain"], self.PALETTE["zero_shot"], self.PALETTE["fine_tuned"]]

        for idx, (mode, color) in enumerate(zip(modes, colors)):
            row = summary_df[summary_df["Evaluation Mode"] == mode].iloc[0]
            vals = [row[m] for m in metrics]
            bars = ax.bar(x + (idx - 1) * width, vals, width, label=mode, color=color, alpha=0.9, edgecolor="#1D3557")
            for b in bars:
                h = b.get_height()
                ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width() / 2, h + 0.015),
                            ha="center", fontsize=8, fontweight="bold")

        ax.set_title("Cross-Dataset Generalization: In-Domain UNSW-NB15 vs Zero-Shot vs Fine-Tuned CIC-IDS2017", pad=15)
        ax.set_xlabel("Evaluation Metric", fontweight="bold")
        ax.set_ylabel("Score", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontweight="bold")
        ax.set_ylim(0.0, 1.10)
        ax.legend(frameon=True, loc="upper right")

        out_path = self.output_dir / "cross_dataset_domain_comparison.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_per_class_transfer(self, per_class_df: pd.DataFrame) -> Path:
        """2. Grouped Bar Chart of 10-Class F1 for Zero-Shot vs Fine-Tuned."""
        fig, ax = plt.subplots(figsize=(13, 6.5))

        classes = per_class_df["Class Name"].tolist()
        x = np.arange(len(classes))
        width = 0.35

        zs_f1 = per_class_df["Zero-Shot F1"]
        ft_f1 = per_class_df["Fine-Tuned F1"]

        b1 = ax.bar(x - width / 2, zs_f1, width, label="CIC-IDS2017 Zero-Shot F1", color=self.PALETTE["zero_shot"], alpha=0.9, edgecolor="#1D3557")
        b2 = ax.bar(x + width / 2, ft_f1, width, label="CIC-IDS2017 Fine-Tuned F1", color=self.PALETTE["fine_tuned"], alpha=0.9, edgecolor="#1D3557")

        for b in b1:
            h = b.get_height()
            if h > 0.01:
                ax.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h + 0.015), ha="center", fontsize=7.5)

        for b in b2:
            h = b.get_height()
            if h > 0.01:
                ax.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h + 0.015), ha="center", fontsize=7.5, fontweight="bold")

        ax.set_title("Per-Class Transferability and Adaptation on CIC-IDS2017 Benchmark", pad=15)
        ax.set_xlabel("Traffic Category", fontweight="bold")
        ax.set_ylabel("F1-Score", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=10)
        ax.set_ylim(0.0, 1.12)
        ax.legend(frameon=True, loc="upper right")

        out_path = self.output_dir / "cross_dataset_per_class_transfer.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_security_rates_comparison(self, summary_df: pd.DataFrame) -> Path:
        """3. FPR vs FNR Cross-Domain Scatter Plot."""
        fig, ax = plt.subplots(figsize=(9, 6))

        modes = summary_df["Evaluation Mode"].tolist()
        fprs = summary_df["FPR (False Alarm Rate)"].values * 100.0
        fnrs = summary_df["FNR (Missed Attack Rate)"].values * 100.0

        colors = [self.PALETTE["in_domain"], self.PALETTE["zero_shot"], self.PALETTE["fine_tuned"]]

        for m, f, n, c in zip(modes, fprs, fnrs, colors):
            ax.scatter(f, n, s=200, color=c, edgecolor="#1D3557", linewidth=1.5, zorder=5, label=m)
            ax.annotate(f"{m.split(' ')[0]}\n(FPR: {f:.2f}%, FNR: {n:.2f}%)", (f + 0.15, n + 0.05),
                        fontsize=9.5, fontweight="bold")

        ax.set_xlabel("False Positive Rate (FPR %) — Lower is Better", fontweight="bold")
        ax.set_ylabel("False Negative Rate (FNR %) — Lower is Better", fontweight="bold")
        ax.set_title("Cross-Dataset Operational Security: False Alarm vs Miss Rates", pad=15)
        ax.legend(frameon=True, loc="upper right")

        out_path = self.output_dir / "cross_dataset_security_rates.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_comprehensive_dashboard(
        self,
        summary_df: pd.DataFrame,
        per_class_df: pd.DataFrame,
    ) -> Path:
        """4. 4-Panel Publication Composite Dashboard."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # (a) Macro F1 and Accuracy across Modes
        modes = [m.split(" ")[0] for m in summary_df["Evaluation Mode"]]
        x = np.arange(len(modes))
        width = 0.35
        ax1.bar(x - width / 2, summary_df["Accuracy"], width, label="Accuracy", color=self.PALETTE["neutral"], edgecolor="#1D3557")
        ax1.bar(x + width / 2, summary_df["Macro F1"], width, label="Macro F1", color=self.PALETTE["fine_tuned"], edgecolor="#1D3557")
        ax1.set_xticks(x)
        ax1.set_xticklabels(summary_df["Evaluation Mode"], rotation=15, ha="right", fontsize=9)
        ax1.set_ylabel("Score", fontweight="bold")
        ax1.set_ylim(0.0, 1.10)
        ax1.set_title("(a) Accuracy and Macro F1 Across Evaluation Protocols", pad=10)
        ax1.legend(loc="upper right")

        # (b) Transfer Drop (Delta Macro F1)
        drops = summary_df["Transfer Drop (Macro F1)"]
        bar_colors = ["#2A9D8F" if d <= 0 else "#E63946" for d in drops]
        ax2.bar(x, drops, 0.45, color=bar_colors, edgecolor="#1D3557", alpha=0.85)
        for i, d in enumerate(drops):
            ax2.annotate(f"{d:+.3f}", (i, d + 0.01 if d >= 0 else d - 0.02), ha="center", fontsize=9.5, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(summary_df["Evaluation Mode"], rotation=15, ha="right", fontsize=9)
        ax2.set_ylabel("Domain Transfer Degradation (\u0394 F1)", fontweight="bold")
        ax2.set_title("(b) Performance Gap Relative to In-Domain Baseline", pad=10)

        # (c) Per-Class F1 Score Comparison
        classes = per_class_df["Class Name"].tolist()
        x_c = np.arange(len(classes))
        ax3.bar(x_c - 0.17, per_class_df["Zero-Shot F1"], 0.34, label="Zero-Shot", color=self.PALETTE["zero_shot"], edgecolor="#1D3557")
        ax3.bar(x_c + 0.17, per_class_df["Fine-Tuned F1"], 0.34, label="Fine-Tuned", color=self.PALETTE["fine_tuned"], edgecolor="#1D3557")
        ax3.set_xticks(x_c)
        ax3.set_xticklabels(classes, rotation=25, ha="right", fontsize=8.5)
        ax3.set_ylabel("F1 Score", fontweight="bold")
        ax3.set_ylim(0.0, 1.12)
        ax3.set_title("(c) Per-Class Transferability (Shared Categories)", pad=10)
        ax3.legend(loc="upper right")

        # (d) False Positive vs Missed Attack Rates
        fprs = summary_df["FPR (False Alarm Rate)"] * 100.0
        fnrs = summary_df["FNR (Missed Attack Rate)"] * 100.0
        ax4.bar(x - width / 2, fprs, width, label="False Alarm Rate (FPR %)", color="#E76F51", edgecolor="#1D3557")
        ax4.bar(x + width / 2, fnrs, width, label="Missed Attack Rate (FNR %)", color="#E63946", edgecolor="#1D3557")
        ax4.set_xticks(x)
        ax4.set_xticklabels(summary_df["Evaluation Mode"], rotation=15, ha="right", fontsize=9)
        ax4.set_ylabel("Operational Error Rate (%)", fontweight="bold")
        ax4.set_title("(d) Cross-Domain False Alarm and Miss Rates", pad=10)
        ax4.legend(loc="upper right")

        fig.suptitle("TGCF-IDS : CROSS-DATASET EXTERNAL VALIDATION DASHBOARD (UNSW-NB15 <-> CIC-IDS2017)",
                     fontsize=15, fontweight="bold", y=0.99)

        out_path = self.output_dir / "cross_dataset_comprehensive_dashboard.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def generate_all_figures(self) -> List[Path]:
        """Generate full suite of cross-dataset validation figures."""
        print("[*] Generating Cross-Dataset Publication Visualizations...")
        summary_df, per_class_df = self.load_data()

        generated = []
        p1 = self.plot_domain_comparison(summary_df)
        print(f"    [+] Saved: {p1.name}")
        generated.append(p1)

        p2 = self.plot_per_class_transfer(per_class_df)
        print(f"    [+] Saved: {p2.name}")
        generated.append(p2)

        p3 = self.plot_security_rates_comparison(summary_df)
        print(f"    [+] Saved: {p3.name}")
        generated.append(p3)

        p4 = self.plot_comprehensive_dashboard(summary_df, per_class_df)
        print(f"    [+] Saved: {p4.name}")
        generated.append(p4)

        return generated


def main():
    generator = CrossDatasetFigureGenerator()
    generator.generate_all_figures()


if __name__ == "__main__":
    main()
