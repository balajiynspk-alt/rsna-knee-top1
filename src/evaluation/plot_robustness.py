#!/usr/bin/env python3
"""
TGCF-IDS: Publication-Quality Robustness Degradation Visualizations.

Generates high-resolution figures under results/figures/robustness/:
1. robustness_degradation_curves.png: 6-panel degradation curves for all 6 perturbation experiments.
2. robustness_security_impact.png: False Positive Rate (FPR) and False Negative Rate (FNR) degradation dynamics.
3. robustness_macro_f1_comparison.png: Overlay comparison of Macro F1 retention across perturbation modes.
4. robustness_comprehensive_dashboard.png: 4-panel composite publication dashboard.
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


class RobustnessFigureGenerator:
    """
    Renders publication-grade degradation curves and sensitivity plots.
    """

    PALETTE = {
        "acc": "#1D3557",
        "macro_f1": "#E63946",
        "recall": "#2A9D8F",
        "precision": "#457B9D",
        "fpr": "#E76F51",
        "fnr": "#9B5DE5",
    }

    MODE_COLORS = [
        "#1D3557", "#457B9D", "#2A9D8F", "#E76F51", "#E63946", "#9B5DE5"
    ]

    def __init__(
        self,
        robustness_csv: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.robustness_csv = Path(robustness_csv or ProjectPaths.RESULTS_TABLES / "robustness.csv")
        self.output_dir = Path(output_dir or ProjectPaths.RESULTS_FIGURES / "robustness")
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

    def load_data(self) -> pd.DataFrame:
        """Load robustness experiment results table."""
        if not self.robustness_csv.exists():
            raise FileNotFoundError(f"Missing robustness table: {self.robustness_csv}")
        return pd.read_csv(self.robustness_csv)

    def plot_degradation_curves_grid(self, df: pd.DataFrame) -> Path:
        """1. 6-Panel Degradation Curves for All Perturbation Modes."""
        modes = df["Perturbation Mode"].unique()
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
        axes = axes.flatten()

        for idx, mode in enumerate(modes):
            ax = axes[idx]
            df_m = df[df["Perturbation Mode"] == mode]

            ax.plot(df_m["Perturbation Strength"], df_m["Accuracy"], marker="o", linewidth=2.2,
                    color=self.PALETTE["acc"], label="Accuracy")
            ax.plot(df_m["Perturbation Strength"], df_m["Weighted F1"], marker="s", linewidth=2.0,
                    color=self.PALETTE["precision"], linestyle="--", label="Weighted F1")
            ax.plot(df_m["Perturbation Strength"], df_m["Macro F1"], marker="^", linewidth=2.2,
                    color=self.PALETTE["macro_f1"], label="Macro F1")
            ax.plot(df_m["Perturbation Strength"], df_m["Macro Recall"], marker="d", linewidth=2.0,
                    color=self.PALETTE["recall"], linestyle=":", label="Macro Recall")

            ax.set_title(f"({chr(97 + idx)}) {mode}", pad=10, fontsize=11.5)
            ax.set_xlabel("Perturbation Strength", fontweight="bold")
            ax.set_ylabel("Metric Score", fontweight="bold")
            ax.set_ylim(-0.02, 1.02)
            if idx == 0:
                ax.legend(frameon=True, loc="lower left", fontsize=9)

        fig.suptitle("TGCF-IDS: Multi-Metric Robustness & Performance Degradation Curves Under Adversarial Perturbations",
                     fontsize=15, fontweight="bold", y=0.98)

        out_path = self.output_dir / "robustness_degradation_curves.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_macro_f1_comparison(self, df: pd.DataFrame) -> Path:
        """2. Overlay Comparison of Macro F1 Retention across all 6 Modalities."""
        fig, ax = plt.subplots(figsize=(11, 6.5))

        modes = df["Perturbation Mode"].unique()
        for mode, color in zip(modes, self.MODE_COLORS):
            df_m = df[df["Perturbation Mode"] == mode]
            # Normalize strength to [0, 1] for visual alignment
            max_s = df_m["Perturbation Strength"].max()
            norm_strength = df_m["Perturbation Strength"] / (max_s if max_s > 0 else 1.0)

            ax.plot(norm_strength, df_m["Macro F1"], marker="o", linewidth=2.4, color=color,
                    label=mode.split("(")[0].strip())

        ax.set_title("Cross-Modality Robustness Comparison: Macro F1 Retention Under Stress", pad=15)
        ax.set_xlabel("Normalized Perturbation Intensity [0.0 = Clean -> 1.0 = Max Stress]", fontweight="bold")
        ax.set_ylabel("Macro F1-Score", fontweight="bold")
        ax.set_ylim(0.0, 0.65)
        ax.legend(frameon=True, loc="upper right", fontsize=9.5)

        out_path = self.output_dir / "robustness_macro_f1_comparison.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_security_impact(self, df: pd.DataFrame) -> Path:
        """3. False Positive Rate vs False Negative Rate Degradation Dynamics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        modes = df["Perturbation Mode"].unique()
        for mode, color in zip(modes, self.MODE_COLORS):
            df_m = df[df["Perturbation Mode"] == mode]
            short_name = mode.split("(")[0].strip()

            ax1.plot(df_m["Perturbation Strength"], df_m["FPR (False Alarm Rate)"] * 100.0,
                     marker="o", linewidth=2.0, color=color, label=short_name)
            ax2.plot(df_m["Perturbation Strength"], df_m["FNR (Missed Attack Rate)"] * 100.0,
                     marker="s", linewidth=2.0, color=color, label=short_name)

        ax1.set_title("(a) False Alarm Rate (FPR %) vs Perturbation", pad=10)
        ax1.set_xlabel("Perturbation Strength", fontweight="bold")
        ax1.set_ylabel("FPR (%) — Lower is Better", fontweight="bold")
        ax1.legend(frameon=True, loc="upper left", fontsize=8.5)

        ax2.set_title("(b) Missed Attack Rate (FNR %) vs Perturbation", pad=10)
        ax2.set_xlabel("Perturbation Strength", fontweight="bold")
        ax2.set_ylabel("FNR (%) — Lower is Better", fontweight="bold")
        ax2.legend(frameon=True, loc="upper left", fontsize=8.5)

        fig.suptitle("TGCF-IDS: Operational Security Metrics Under Perturbations",
                     fontsize=14, fontweight="bold", y=1.02)

        out_path = self.output_dir / "robustness_security_impact.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_comprehensive_dashboard(self, df: pd.DataFrame) -> Path:
        """4. 4-Panel Publication Composite Dashboard."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # (a) Feature Masking Degradation
        df_fm = df[df["Perturbation Mode"].str.contains("Feature Masking")]
        ax1.plot(df_fm["Perturbation Strength"], df_fm["Accuracy"], marker="o", color=self.PALETTE["acc"], linewidth=2.2, label="Accuracy")
        ax1.plot(df_fm["Perturbation Strength"], df_fm["Macro F1"], marker="^", color=self.PALETTE["macro_f1"], linewidth=2.2, label="Macro F1")
        ax1.plot(df_fm["Perturbation Strength"], df_fm["Macro Recall"], marker="d", color=self.PALETTE["recall"], linewidth=2.0, label="Macro Recall")
        ax1.set_title("(a) Resilience to Feature Masking (Sensor Loss)", pad=10)
        ax1.set_xlabel("Feature Masking Ratio", fontweight="bold")
        ax1.set_ylabel("Score", fontweight="bold")
        ax1.set_ylim(0.0, 1.0)
        ax1.legend(loc="lower left", fontsize=9)

        # (b) Numerical Gaussian Noise Degradation
        df_nn = df[df["Perturbation Mode"].str.contains("Numerical Noise")]
        ax2.plot(df_nn["Perturbation Strength"], df_nn["Accuracy"], marker="o", color=self.PALETTE["acc"], linewidth=2.2, label="Accuracy")
        ax2.plot(df_nn["Perturbation Strength"], df_nn["Macro F1"], marker="^", color=self.PALETTE["macro_f1"], linewidth=2.2, label="Macro F1")
        ax2.plot(df_nn["Perturbation Strength"], df_nn["Macro Recall"], marker="d", color=self.PALETTE["recall"], linewidth=2.0, label="Macro Recall")
        ax2.set_title("(b) Resilience to Gaussian Numerical Jitter", pad=10)
        ax2.set_xlabel("Noise Standard Deviation (std)", fontweight="bold")
        ax2.set_ylabel("Score", fontweight="bold")
        ax2.set_ylim(0.0, 1.0)
        ax2.legend(loc="lower left", fontsize=9)

        # (c) Graph Edge Deletion & Addition Impact
        df_del = df[df["Perturbation Mode"].str.contains("Edge Deletion")]
        df_add = df[df["Perturbation Mode"].str.contains("Edge Addition")]
        ax3.plot(df_del["Perturbation Strength"], df_del["Macro F1"], marker="o", color="#1D3557", linewidth=2.2, label="Edge Deletion (Macro F1)")
        ax3.plot(df_add["Perturbation Strength"], df_add["Macro F1"], marker="s", color="#E76F51", linewidth=2.2, label="Edge Addition (Macro F1)")
        ax3.set_title("(c) Graph Topology Perturbations (Link Drops & Cross-Talk)", pad=10)
        ax3.set_xlabel("Edge Perturbation Ratio", fontweight="bold")
        ax3.set_ylabel("Macro F1-Score", fontweight="bold")
        ax3.set_ylim(0.35, 0.60)
        ax3.legend(loc="lower left", fontsize=9)

        # (d) Security Error Rates at Max Perturbation
        max_strengths = []
        for mode in df["Perturbation Mode"].unique():
            df_m = df[df["Perturbation Mode"] == mode]
            row_max = df_m.iloc[-1]
            max_strengths.append({
                "Mode": mode.split("(")[0].strip(),
                "FPR": row_max["FPR (False Alarm Rate)"] * 100.0,
                "FNR": row_max["FNR (Missed Attack Rate)"] * 100.0,
            })
        df_max = pd.DataFrame(max_strengths)
        x = np.arange(len(df_max))
        width = 0.35
        ax4.bar(x - width / 2, df_max["FPR"], width, label="False Alarm Rate (FPR %)", color="#E76F51", edgecolor="#1D3557")
        ax4.bar(x + width / 2, df_max["FNR"], width, label="Missed Attack Rate (FNR %)", color="#E63946", edgecolor="#1D3557")
        ax4.set_xticks(x)
        ax4.set_xticklabels(df_max["Mode"], rotation=25, ha="right", fontsize=8.5)
        ax4.set_ylabel("Error Rate (%) at Max Stress", fontweight="bold")
        ax4.set_title("(d) Worst-Case Security Error Rates Across Modes", pad=10)
        ax4.legend(loc="upper left", fontsize=9)

        fig.suptitle("TGCF-IDS : COMPREHENSIVE CONTROLLED ROBUSTNESS & ADVERSARIAL SENSITIVITY DASHBOARD",
                     fontsize=15, fontweight="bold", y=0.99)

        out_path = self.output_dir / "robustness_comprehensive_dashboard.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def generate_all_figures(self) -> List[Path]:
        """Generate all robustness visualization figures."""
        print("[*] Generating Robustness Degradation Visualizations...")
        df = self.load_data()

        generated = []
        p1 = self.plot_degradation_curves_grid(df)
        print(f"    [+] Saved: {p1.name}")
        generated.append(p1)

        p2 = self.plot_macro_f1_comparison(df)
        print(f"    [+] Saved: {p2.name}")
        generated.append(p2)

        p3 = self.plot_security_impact(df)
        print(f"    [+] Saved: {p3.name}")
        generated.append(p3)

        p4 = self.plot_comprehensive_dashboard(df)
        print(f"    [+] Saved: {p4.name}")
        generated.append(p4)

        return generated


def main():
    generator = RobustnessFigureGenerator()
    generator.generate_all_figures()


if __name__ == "__main__":
    main()
