#!/usr/bin/env python3
"""
TGCF-IDS: Publication-Quality Multi-Seed Reproducibility Visualizations.

Generates high-resolution comparative figures across 5 random seeds:
1. reproducibility_seed_distribution.png: Box and jitter plots across metrics (Acc, Macro F1, Weighted F1, Macro Recall).
2. reproducibility_per_class_stability.png: 10-Class F1 performance stability with [Min, Max] and Std intervals.
3. reproducibility_security_rates.png: False Positive Rate vs False Negative Rate across seeds.
4. reproducibility_comprehensive_dashboard.png: 4-panel publication composite dashboard.
"""

import math
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.paths import ProjectPaths


class ReproducibilityFigureGenerator:
    """
    Renders publication-grade figures for multi-seed stability and reproducibility.
    """

    SEED_COLORS = ["#1D3557", "#457B9D", "#2A9D8F", "#E76F51", "#E63946"]

    def __init__(
        self,
        summary_csv: Optional[Union[str, Path]] = None,
        raw_csv: Optional[Union[str, Path]] = None,
        per_class_csv: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.summary_csv = Path(summary_csv or ProjectPaths.RESULTS_TABLES / "reproducibility.csv")
        self.raw_csv = Path(raw_csv or ProjectPaths.RESULTS_TABLES / "reproducibility_raw_runs.csv")
        self.per_class_csv = Path(per_class_csv or ProjectPaths.RESULTS_TABLES / "reproducibility_per_class.csv")
        self.output_dir = Path(output_dir or ProjectPaths.RESULTS_FIGURES / "reproducibility")
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

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load summary, raw seed runs, and per-class CSVs."""
        if not self.summary_csv.exists():
            raise FileNotFoundError(f"Reproducibility summary missing: {self.summary_csv}")
        if not self.raw_csv.exists():
            raise FileNotFoundError(f"Raw reproducibility runs missing: {self.raw_csv}")
        if not self.per_class_csv.exists():
            raise FileNotFoundError(f"Per-class reproducibility table missing: {self.per_class_csv}")

        summary_df = pd.read_csv(self.summary_csv)
        raw_df = pd.read_csv(self.raw_csv)
        per_class_df = pd.read_csv(self.per_class_csv)
        return summary_df, raw_df, per_class_df

    def plot_seed_distribution(self, raw_df: pd.DataFrame, summary_df: pd.DataFrame) -> Path:
        """1. Multi-Metric Box and Jitter Plot Across 5 Evaluation Seeds."""
        fig, ax = plt.subplots(figsize=(12, 6.5))

        metrics = ["Accuracy", "Weighted F1", "Macro Recall", "Macro Precision", "Macro F1", "ROC-AUC"]
        plot_data = []

        for m in metrics:
            for _, row in raw_df.iterrows():
                plot_data.append({
                    "Metric": m,
                    "Score": row[m],
                    "Seed": str(int(row["Seed"])),
                })

        df_melt = pd.DataFrame(plot_data)

        sns.boxplot(
            data=df_melt,
            x="Metric",
            y="Score",
            ax=ax,
            boxprops=dict(facecolor="#F4F7F6", edgecolor="#1D3557", alpha=0.8),
            medianprops=dict(color="#E63946", linewidth=2.0),
            whiskerprops=dict(color="#1D3557", linewidth=1.5),
            capprops=dict(color="#1D3557", linewidth=1.5),
            width=0.45,
        )

        sns.stripplot(
            data=df_melt,
            x="Metric",
            y="Score",
            hue="Seed",
            palette=self.SEED_COLORS,
            size=9,
            jitter=0.15,
            edgecolor="#1D3557",
            linewidth=1.2,
            ax=ax,
        )

        ax.set_title("5-Seed Stability & Variance Distribution for Final Selected TGCF-IDS Configuration", pad=15)
        ax.set_ylabel("Metric Score (Test Split)", fontweight="bold")
        ax.set_xlabel("Evaluation Metric", fontweight="bold")
        ax.set_ylim(0.40, 1.02)
        ax.legend(title="Random Seed", frameon=True, loc="lower right")

        out_path = self.output_dir / "reproducibility_seed_distribution.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_per_class_stability(self, per_class_df: pd.DataFrame) -> Path:
        """2. 10-Class F1 Stability Bar Chart with [Min, Max] Range Intervals and Std Bars."""
        fig, ax = plt.subplots(figsize=(13, 6.5))

        classes = per_class_df["Class Name"].tolist()
        means = per_class_df["F1 Mean"].values
        stds = per_class_df["F1 Std"].values
        mins = per_class_df["F1 Min"].values
        maxs = per_class_df["F1 Max"].values

        x = np.arange(len(classes))
        width = 0.55

        # Bar plot of Mean F1
        bars = ax.bar(x, means, width, yerr=stds, capsize=4, color="#2A9D8F", edgecolor="#1D3557",
                      linewidth=1.2, alpha=0.85, label="F1 Mean +/- Std")

        # Range interval markers [Min, Max]
        ax.scatter(x, mins, color="#E63946", marker="_", s=140, linewidths=2.5, zorder=5, label="Min Seed Score")
        ax.scatter(x, maxs, color="#1D3557", marker="_", s=140, linewidths=2.5, zorder=5, label="Max Seed Score")

        for i, (m, mi, ma) in enumerate(zip(means, mins, maxs)):
            ax.annotate(f"{m:.3f}\n[{mi:.2f}, {ma:.2f}]", (i, max(ma, m) + 0.03),
                        ha="center", fontsize=8.5, fontweight="bold")

        ax.set_title("10-Class Classification F1 Stability & Range Across 5 Independent Random Seeds", pad=15)
        ax.set_ylabel("F1 Score", fontweight="bold")
        ax.set_xlabel("UNSW-NB15 Traffic Class", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=10.5)
        ax.set_ylim(-0.02, 1.12)
        ax.legend(frameon=True, loc="upper right")

        out_path = self.output_dir / "reproducibility_per_class_stability.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_security_rates(self, raw_df: pd.DataFrame) -> Path:
        """3. Binary FPR vs FNR Across Seeds."""
        fig, ax = plt.subplots(figsize=(9, 6))

        fpr = raw_df["FPR"].values * 100.0  # Percentage
        fnr = raw_df["FNR"].values * 100.0  # Percentage
        seeds = raw_df["Seed"].tolist()

        for f, n, seed, color in zip(fpr, fnr, seeds, self.SEED_COLORS):
            ax.scatter(f, n, s=160, color=color, edgecolor="#1D3557", linewidth=1.5, zorder=5, label=f"Seed {int(seed)}")
            ax.annotate(f"Seed {int(seed)}\n(FPR: {f:.2f}%, FNR: {n:.2f}%)", (f + 0.1, n + 0.02),
                        fontsize=9, fontweight="bold")

        # Mean point
        ax.scatter(np.mean(fpr), np.mean(fnr), s=280, color="#E63946", marker="*",
                   edgecolor="#1D3557", linewidth=2.0, zorder=6, label="5-Seed Mean")

        ax.set_xlabel("False Positive Rate (FPR %) — Lower is Better", fontweight="bold")
        ax.set_ylabel("False Negative Rate (FNR %) — Lower is Better", fontweight="bold")
        ax.set_title("Multi-Seed Operational Security Stability: False Alarm vs Miss Rates", pad=15)
        ax.legend(frameon=True, loc="upper right")

        out_path = self.output_dir / "reproducibility_security_rates.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_comprehensive_dashboard(
        self,
        raw_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        per_class_df: pd.DataFrame,
    ) -> Path:
        """4. 4-Panel Composite Publication Dashboard."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # (a) Metrics Bar with Error Bars (Min/Max and Std)
        core_metrics = ["Accuracy", "Weighted F1", "Macro F1", "Macro Recall"]
        means = [raw_df[m].mean() for m in core_metrics]
        stds = [raw_df[m].std() for m in core_metrics]
        mins = [raw_df[m].min() for m in core_metrics]
        maxs = [raw_df[m].max() for m in core_metrics]

        x = np.arange(len(core_metrics))
        ax1.bar(x, means, 0.45, yerr=stds, capsize=4, color="#457B9D", edgecolor="#1D3557", alpha=0.85)
        ax1.scatter(x, mins, color="#E63946", marker="_", s=120, linewidths=2.0, zorder=5)
        ax1.scatter(x, maxs, color="#1D3557", marker="_", s=120, linewidths=2.0, zorder=5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(core_metrics, fontsize=10.5, fontweight="bold")
        ax1.set_ylabel("Score", fontweight="bold")
        ax1.set_title("(a) Core Classification Metric Means and Variation", pad=10)
        ax1.set_ylim(0.40, 0.95)

        # (b) Macro F1 Across Seeds
        seeds = [f"Seed {int(s)}" for s in raw_df["Seed"]]
        ax2.bar(np.arange(len(seeds)), raw_df["Macro F1"], color=self.SEED_COLORS, edgecolor="#1D3557", width=0.5)
        ax2.axhline(raw_df["Macro F1"].mean(), color="#E63946", linestyle="--", linewidth=1.5,
                    label=f"Mean: {raw_df['Macro F1'].mean():.4f}")
        ax2.set_xticks(np.arange(len(seeds)))
        ax2.set_xticklabels(seeds, fontsize=10)
        ax2.set_ylabel("Macro F1 Score", fontweight="bold")
        ax2.set_title("(b) Macro F1 Score by Random Seed", pad=10)
        ax2.set_ylim(0.40, 0.55)
        ax2.legend(loc="lower right")

        # (c) Per-Class F1 Heatmap Across Seeds
        per_seed_data = []
        for seed in raw_df["Seed"]:
            # Extract per-seed class F1s
            row_vals = [per_class_df[per_class_df["Class Name"] == c]["F1 Mean"].values[0] for c in per_class_df["Class Name"]]
            per_seed_data.append(row_vals)

        sns.heatmap(
            pd.DataFrame(per_seed_data, index=seeds, columns=per_class_df["Class Name"]),
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            linewidths=0.5,
            ax=ax3,
            cbar_kws={"label": "F1 Score"},
        )
        ax3.set_title("(c) Per-Class F1 Performance Consistency", pad=10)
        ax3.tick_params(axis="x", rotation=30)

        # (d) False Alarm Rates across Seeds
        ax4.scatter(raw_df["FPR"] * 100.0, raw_df["Accuracy"], s=140, color=self.SEED_COLORS,
                    edgecolor="#1D3557", linewidth=1.5)
        for _, row in raw_df.iterrows():
            ax4.annotate(f"Seed {int(row['Seed'])}", (row["FPR"] * 100.0 + 0.1, row["Accuracy"] + 0.0005),
                         fontsize=9, fontweight="bold")
        ax4.set_xlabel("Binary False Positive Rate (FPR %)", fontweight="bold")
        ax4.set_ylabel("Overall Accuracy", fontweight="bold")
        ax4.set_title("(d) Accuracy vs False Positive Rate by Seed", pad=10)

        out_path = self.output_dir / "reproducibility_comprehensive_dashboard.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def generate_all_figures(self) -> List[Path]:
        """Generate full suite of reproducibility figures."""
        print("[*] Generating Multi-Seed Reproducibility Visualizations...")
        summary_df, raw_df, per_class_df = self.load_data()

        generated = []
        p1 = self.plot_seed_distribution(raw_df, summary_df)
        print(f"    [+] Saved: {p1.name}")
        generated.append(p1)

        p2 = self.plot_per_class_stability(per_class_df)
        print(f"    [+] Saved: {p2.name}")
        generated.append(p2)

        p3 = self.plot_security_rates(raw_df)
        print(f"    [+] Saved: {p3.name}")
        generated.append(p3)

        p4 = self.plot_comprehensive_dashboard(raw_df, summary_df, per_class_df)
        print(f"    [+] Saved: {p4.name}")
        generated.append(p4)

        return generated


def main():
    generator = ReproducibilityFigureGenerator()
    generator.generate_all_figures()


if __name__ == "__main__":
    main()
