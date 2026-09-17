#!/usr/bin/env python3
"""
TGCF-IDS: Publication-Quality Computational Efficiency & Complexity Visualizations.

Generates high-resolution figures under results/figures/efficiency/:
1. efficiency_pareto_accuracy_vs_latency.png: Pareto frontier comparing Macro F1 vs Inference Latency.
2. efficiency_memory_and_params.png: Parameter Count vs GPU Peak Memory Footprint.
3. efficiency_training_throughput.png: Training Epoch Duration vs Real-Time Inference Throughput.
4. efficiency_comprehensive_dashboard.png: 4-panel publication composite dashboard.
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


class EfficiencyFigureGenerator:
    """
    Renders publication-grade computational efficiency and resource consumption charts.
    """

    MODEL_COLORS = {
        "MLP": "#457B9D",
        "Transformer": "#2A9D8F",
        "GraphSAGE": "#E76F51",
        "GraphSAGE + Transformer": "#9B5DE5",
        "TGCF-IDS": "#E63946",
    }

    def __init__(
        self,
        efficiency_csv: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.efficiency_csv = Path(efficiency_csv or ProjectPaths.RESULTS_TABLES / "efficiency.csv")
        self.output_dir = Path(output_dir or ProjectPaths.RESULTS_FIGURES / "efficiency")
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
        """Load efficiency benchmark table."""
        if not self.efficiency_csv.exists():
            raise FileNotFoundError(f"Missing efficiency table: {self.efficiency_csv}")
        return pd.read_csv(self.efficiency_csv)

    def plot_pareto_frontier(self, df: pd.DataFrame) -> Path:
        """1. Pareto Frontier: Macro F1 vs Inference Latency."""
        fig, ax = plt.subplots(figsize=(10, 6.5))

        for _, row in df.iterrows():
            m_name = row["Model Architecture"]
            c = self.MODEL_COLORS.get(m_name, "#1D3557")
            lat = row["Inference Latency (ms/1k flows)"]
            f1 = row["Macro F1"]
            size = np.sqrt(row["Parameters"]) * 0.45

            ax.scatter(lat, f1, s=size, color=c, edgecolor="#1D3557", linewidth=1.5, zorder=5, label=m_name)
            ax.annotate(
                f"{m_name}\n(F1: {f1:.3f}, {lat:.1f}ms)",
                (lat + 0.15, f1 + 0.005),
                fontsize=9.5,
                fontweight="bold",
            )

        ax.set_title("Pareto Frontier: Classification Macro F1 vs Inference Latency (Bubble size = Parameter count)", pad=15)
        ax.set_xlabel("Inference Latency (ms / 1,000 flows) — Lower is Better", fontweight="bold")
        ax.set_ylabel("Macro F1-Score — Higher is Better", fontweight="bold")
        ax.set_ylim(0.20, 0.60)
        ax.legend(frameon=True, loc="lower right", fontsize=9.5)

        out_path = self.output_dir / "efficiency_pareto_accuracy_vs_latency.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_memory_and_params(self, df: pd.DataFrame) -> Path:
        """2. Parameter Count vs Peak GPU Memory Footprint."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        models = df["Model Architecture"]
        colors = [self.MODEL_COLORS.get(m, "#1D3557") for m in models]
        x = np.arange(len(models))

        # Parameters
        bars1 = ax1.bar(x, df["Parameters"] / 1000.0, color=colors, edgecolor="#1D3557", width=0.55, alpha=0.85)
        for b in bars1:
            h = b.get_height()
            ax1.annotate(f"{h:.1f}k", (b.get_x() + b.get_width() / 2, h + 3.0), ha="center", fontsize=9, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=25, ha="right", fontsize=9.5)
        ax1.set_ylabel("Parameters (Thousands)", fontweight="bold")
        ax1.set_title("(a) Total Trainable Parameter Count", pad=10)

        # GPU Memory
        bars2 = ax2.bar(x, df["GPU Peak Memory (MB)"], color=colors, edgecolor="#1D3557", width=0.55, alpha=0.85)
        for b in bars2:
            h = b.get_height()
            ax2.annotate(f"{h:.1f} MB", (b.get_x() + b.get_width() / 2, h + 5.0), ha="center", fontsize=9, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=25, ha="right", fontsize=9.5)
        ax2.set_ylabel("Peak VRAM Allocation (MB)", fontweight="bold")
        ax2.set_title("(b) Peak GPU Memory Footprint", pad=10)

        fig.suptitle("TGCF-IDS: Structural Footprint and Hardware Resource Consumption", fontsize=14, fontweight="bold", y=1.02)

        out_path = self.output_dir / "efficiency_memory_and_params.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_training_throughput(self, df: pd.DataFrame) -> Path:
        """3. Training Epoch Duration vs Real-Time Inference Throughput."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        models = df["Model Architecture"]
        colors = [self.MODEL_COLORS.get(m, "#1D3557") for m in models]
        x = np.arange(len(models))

        # Epoch Time
        bars1 = ax1.bar(x, df["Epoch Time (s/epoch)"], color=colors, edgecolor="#1D3557", width=0.55, alpha=0.85)
        for b in bars1:
            h = b.get_height()
            ax1.annotate(f"{h:.2f}s", (b.get_x() + b.get_width() / 2, h + 0.1), ha="center", fontsize=9, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=25, ha="right", fontsize=9.5)
        ax1.set_ylabel("Time (seconds / epoch)", fontweight="bold")
        ax1.set_title("(a) Training Epoch Execution Time", pad=10)

        # Throughput
        bars2 = ax2.bar(x, df["Throughput (flows/sec)"] / 1000.0, color=colors, edgecolor="#1D3557", width=0.55, alpha=0.85)
        for b in bars2:
            h = b.get_height()
            ax2.annotate(f"{h:.1f}k", (b.get_x() + b.get_width() / 2, h + 1.0), ha="center", fontsize=9, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=25, ha="right", fontsize=9.5)
        ax2.set_ylabel("Throughput (Thousand Flows / sec)", fontweight="bold")
        ax2.set_title("(b) Inference Throughput (flows/sec)", pad=10)

        fig.suptitle("TGCF-IDS: Operational Throughput and Computational Scalability", fontsize=14, fontweight="bold", y=1.02)

        out_path = self.output_dir / "efficiency_training_throughput.png"
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

        models = df["Model Architecture"]
        colors = [self.MODEL_COLORS.get(m, "#1D3557") for m in models]
        x = np.arange(len(models))

        # (a) Macro F1 vs Inference Latency
        for _, row in df.iterrows():
            m_name = row["Model Architecture"]
            c = self.MODEL_COLORS.get(m_name, "#1D3557")
            ax1.scatter(row["Inference Latency (ms/1k flows)"], row["Macro F1"], s=180, color=c, edgecolor="#1D3557", linewidth=1.5)
            ax1.annotate(m_name.split("+")[0].strip(), (row["Inference Latency (ms/1k flows)"] + 0.15, row["Macro F1"] + 0.005),
                         fontsize=9, fontweight="bold")
        ax1.set_xlabel("Inference Latency (ms / 1k flows)", fontweight="bold")
        ax1.set_ylabel("Macro F1-Score", fontweight="bold")
        ax1.set_title("(a) Accuracy vs Latency Tradeoff", pad=10)

        # (b) Model Parameters & Size
        ax2.bar(x, df["Parameters"] / 1000.0, color=colors, edgecolor="#1D3557", width=0.5)
        for i, val in enumerate(df["Parameters"] / 1000.0):
            ax2.annotate(f"{val:.1f}k", (i, val + 2.0), ha="center", fontsize=8.5, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax2.set_ylabel("Trainable Parameters (k)", fontweight="bold")
        ax2.set_title("(b) Model Parameter Footprint", pad=10)

        # (c) Peak VRAM
        ax3.bar(x, df["GPU Peak Memory (MB)"], color=colors, edgecolor="#1D3557", width=0.5)
        for i, val in enumerate(df["GPU Peak Memory (MB)"]):
            ax3.annotate(f"{val:.0f} MB", (i, val + 5.0), ha="center", fontsize=8.5, fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax3.set_ylabel("Peak VRAM (MB)", fontweight="bold")
        ax3.set_title("(c) Peak Hardware Memory Footprint", pad=10)

        # (d) Throughput
        ax4.bar(x, df["Throughput (flows/sec)"] / 1000.0, color=colors, edgecolor="#1D3557", width=0.5)
        for i, val in enumerate(df["Throughput (flows/sec)"] / 1000.0):
            ax4.annotate(f"{val:.1f}k/s", (i, val + 1.0), ha="center", fontsize=8.5, fontweight="bold")
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax4.set_ylabel("Throughput (k flows/sec)", fontweight="bold")
        ax4.set_title("(d) Real-Time Inference Throughput", pad=10)

        fig.suptitle("TGCF-IDS : COMPUTATIONAL EFFICIENCY & HARDWARE COMPLEXITY DASHBOARD",
                     fontsize=15, fontweight="bold", y=0.99)

        out_path = self.output_dir / "efficiency_comprehensive_dashboard.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def generate_all_figures(self) -> List[Path]:
        """Generate all efficiency visualization figures."""
        print("[*] Generating Computational Efficiency Visualizations...")
        df = self.load_data()

        generated = []
        p1 = self.plot_pareto_frontier(df)
        print(f"    [+] Saved: {p1.name}")
        generated.append(p1)

        p2 = self.plot_memory_and_params(df)
        print(f"    [+] Saved: {p2.name}")
        generated.append(p2)

        p3 = self.plot_training_throughput(df)
        print(f"    [+] Saved: {p3.name}")
        generated.append(p3)

        p4 = self.plot_comprehensive_dashboard(df)
        print(f"    [+] Saved: {p4.name}")
        generated.append(p4)

        return generated


def main():
    generator = EfficiencyFigureGenerator()
    generator.generate_all_figures()


if __name__ == "__main__":
    main()
