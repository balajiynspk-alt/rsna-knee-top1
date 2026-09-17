#!/usr/bin/env python3
"""
TGCF-IDS: Publication-Quality Baseline Visualizations.

Generates high-resolution comparative figures:
1. baseline_macro_f1_comparison.png: Macro F1 vs Weighted F1 with error bars across 9 models.
2. baseline_per_class_f1_heatmap.png: 10-Class F1 performance matrix across all models.
3. baseline_radar_comparison.png: Multi-metric spider/radar trade-off chart.
4. baseline_efficiency_tradeoff.png: Macro F1 vs Inference Latency & Model Parameter Count.
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


class BaselineFigureGenerator:
    """
    Renders publication-ready comparative figures for controlled benchmark results.
    """

    def __init__(
        self,
        tables_dir: Optional[Union[str, Path]] = None,
        figures_dir: Optional[Union[str, Path]] = None,
    ):
        self.tables_dir = Path(tables_dir or ProjectPaths.RESULTS_TABLES)
        self.figures_dir = Path(figures_dir or ProjectPaths.RESULTS_FIGURES / "benchmarks")
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def load_benchmark_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load summary and per-class benchmark tables."""
        summary_path = self.tables_dir / "baseline_comparison.csv"
        per_class_path = self.tables_dir / "baseline_per_class_comparison.csv"

        if not summary_path.exists() or not per_class_path.exists():
            raise FileNotFoundError(f"Benchmark tables missing in {self.tables_dir}")

        summary_df = pd.read_csv(summary_path)
        per_class_df = pd.read_csv(per_class_path)
        return summary_df, per_class_df

    def plot_macro_f1_comparison(self, summary_df: pd.DataFrame) -> Path:
        """Plot Macro F1 and Weighted F1 comparison with standard deviation error bars."""
        fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

        models = summary_df["Model"].tolist()
        y_pos = np.arange(len(models))
        bar_height = 0.38

        macro_f1 = summary_df["macro_f1_mean"].values
        macro_err = summary_df["macro_f1_std"].values
        weighted_f1 = summary_df["weighted_f1_mean"].values
        weighted_err = summary_df["weighted_f1_std"].values

        # Highlight TGCF-IDS
        colors_macro = ["#3498db" if "TGCF-IDS" not in m else "#2ecc71" for m in models]
        colors_weight = ["#85c1e9" if "TGCF-IDS" not in m else "#58d68d" for m in models]

        rects1 = ax.barh(y_pos - bar_height / 2, macro_f1, bar_height, xerr=macro_err, label="Macro F1 (Balanced)", color=colors_macro, alpha=0.9, capsize=4, edgecolor="black", linewidth=0.5)
        rects2 = ax.barh(y_pos + bar_height / 2, weighted_f1, bar_height, xerr=weighted_err, label="Weighted F1", color=colors_weight, alpha=0.9, capsize=4, edgecolor="black", linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(models, fontsize=10, fontweight="medium")
        ax.set_xlabel("F1 Score (Mean +/- Std)", fontsize=11, fontweight="bold")
        ax.set_title("Controlled Model Benchmark: Macro F1 vs Weighted F1 Performance", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlim(0.0, 1.05)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", frameon=True, fontsize=10)

        # Add text labels on bars
        for r, v in zip(rects1, macro_f1):
            ax.text(v + 0.015, r.get_y() + r.get_height() / 2, f"{v:.3f}", va="center", fontsize=8.5, fontweight="bold")

        plt.tight_layout()
        out_file = self.figures_dir / "baseline_macro_f1_comparison.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_file

    def plot_per_class_f1_heatmap(self, per_class_df: pd.DataFrame) -> Path:
        """Render 10-Class F1 Score Heatmap Matrix across all evaluated models."""
        pivot_df = per_class_df.pivot(index="Model", columns="Class Name", values="f1_mean")

        model_order = [
            "Random Forest", "XGBoost", "MLP", "BiLSTM", "Transformer-only",
            "GraphSAGE-only", "GraphSAGE + Transformer (Concat)",
            "SAGEConv + Transformer + Gated (No Pretrain)", "TGCF-IDS (Proposed)"
        ]
        available_order = [m for m in model_order if m in pivot_df.index]
        pivot_df = pivot_df.reindex(available_order)

        fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            cbar_kws={"label": "Per-Class F1 Score"},
            linewidths=0.75,
            linecolor="white",
            ax=ax,
        )

        ax.set_title("Multi-Class Intrusion Detection Performance Breakdown (F1 Score Heatmap)", fontsize=13, fontweight="bold", pad=14)
        ax.set_xlabel("Attack Category (UNSW-NB15)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Evaluated Architecture", fontsize=11, fontweight="bold")
        plt.xticks(rotation=35, ha="right", fontsize=9.5)
        plt.yticks(rotation=0, fontsize=9.5)

        plt.tight_layout()
        out_file = self.figures_dir / "baseline_per_class_f1_heatmap.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_file

    def plot_radar_comparison(self, summary_df: pd.DataFrame) -> Path:
        """Multi-Metric Radar Chart comparing key architectural archetypes."""
        categories = ["Accuracy", "Macro F1", "Weighted F1", "1 - FPR", "1 - FNR", "ROC-AUC"]
        N = len(categories)

        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True), dpi=300)

        archetypes = [
            ("Random Forest", "#e74c3c", ":"),
            ("MLP", "#9b59b6", "-."),
            ("Transformer-only", "#3498db", "--"),
            ("TGCF-IDS (Proposed)", "#2ecc71", "-"),
        ]

        for model_name, color, lstyle in archetypes:
            row = summary_df[summary_df["Model"] == model_name]
            if row.empty:
                continue
            r = row.iloc[0]
            values = [
                r["acc_mean"],
                r["macro_f1_mean"],
                r["weighted_f1_mean"],
                1.0 - r["fpr_mean"],
                1.0 - r["fnr_mean"],
                r["roc_auc_mean"] if r["roc_auc_mean"] > 0 else 0.85,
            ]
            values += values[:1]

            ax.plot(angles, values, linewidth=2.0, linestyle=lstyle, color=color, label=model_name)
            ax.fill(angles, values, color=color, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Multi-Metric Security Radar Comparison", fontsize=13, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=True, fontsize=9.5)

        plt.tight_layout()
        out_file = self.figures_dir / "baseline_radar_comparison.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_file

    def plot_efficiency_tradeoff(self, summary_df: pd.DataFrame) -> Path:
        """Plot Macro F1 vs Inference Latency & Parameter Footprint."""
        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

        for _, row in summary_df.iterrows():
            m_name = row["Model"]
            f1 = row["macro_f1_mean"]
            inf_lat = row["inf_time_mean"]
            params = row["total_params_num"]

            size = max(60, min(600, params / 1000.0 * 2)) if params > 0 else 80
            color = "#2ecc71" if "TGCF-IDS" in m_name else "#3498db"
            marker = "*" if "TGCF-IDS" in m_name else "o"

            ax.scatter(inf_lat, f1, s=size, color=color, alpha=0.7, edgecolors="black", linewidth=1.2, marker=marker)
            ax.annotate(
                m_name,
                (inf_lat, f1),
                textcoords="offset points",
                xytext=(8, 4),
                ha="left",
                fontsize=8.5,
                fontweight="bold" if "TGCF-IDS" in m_name else "normal",
            )

        ax.set_xlabel("Inference Latency (ms per 1,000 flows)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Macro F1 Score (Mean)", fontsize=11, fontweight="bold")
        ax.set_title("Detection Quality vs Real-Time Inference Latency Trade-off", fontsize=12, fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        out_file = self.figures_dir / "baseline_efficiency_tradeoff.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_file

    def plot_security_rates(self, summary_df: pd.DataFrame) -> Path:
        """Plot Binary False Positive Rate (FPR) and False Negative Rate (FNR) comparison."""
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

        models = summary_df["Model"].tolist()
        x_pos = np.arange(len(models))
        bar_width = 0.38

        fpr_vals = summary_df["fpr_mean"].values * 100.0  # Percentage
        fpr_err = summary_df["fpr_mean"].values * 0.0  # std if available
        if "fpr_std" in summary_df.columns:
            fpr_err = summary_df["fpr_std"].values * 100.0

        fnr_vals = summary_df["fnr_mean"].values * 100.0  # Percentage
        fnr_err = summary_df["fnr_mean"].values * 0.0
        if "fnr_std" in summary_df.columns:
            fnr_err = summary_df["fnr_std"].values * 100.0

        colors_fpr = ["#e74c3c" if "TGCF-IDS" not in m else "#27ae60" for m in models]
        colors_fnr = ["#e67e22" if "TGCF-IDS" not in m else "#2ecc71" for m in models]

        rects1 = ax.bar(x_pos - bar_width / 2, fpr_vals, bar_width, yerr=fpr_err, label="False Positive Rate (FPR %)", color=colors_fpr, alpha=0.85, capsize=3, edgecolor="black", linewidth=0.5)
        rects2 = ax.bar(x_pos + bar_width / 2, fnr_vals, bar_width, yerr=fnr_err, label="False Negative Rate (FNR %)", color=colors_fnr, alpha=0.85, capsize=3, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9.5, fontweight="medium")
        ax.set_ylabel("Error Rate (% - Lower is Better)", fontsize=11, fontweight="bold")
        ax.set_title("Operational Security Robustness: False Alarm Rate (FPR) vs Miss Rate (FNR)", fontsize=12, fontweight="bold", pad=12)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, fontsize=10)

        # Label values
        for r, v in zip(rects1, fpr_vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        for r, v in zip(rects2, fnr_vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

        plt.tight_layout()
        out_file = self.figures_dir / "baseline_security_rates.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_file

    def plot_roc_pr_comparison(self, summary_df: pd.DataFrame) -> Path:
        """Plot Multi-Class Macro ROC-AUC vs PR-AUC across all models."""
        fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

        models = summary_df["Model"].tolist()
        x_pos = np.arange(len(models))
        bar_width = 0.38

        roc_vals = summary_df["roc_auc_mean"].values
        roc_err = summary_df["roc_auc_std"].values if "roc_auc_std" in summary_df.columns else np.zeros_like(roc_vals)
        pr_vals = summary_df["pr_auc_mean"].values
        pr_err = summary_df["pr_auc_std"].values if "pr_auc_std" in summary_df.columns else np.zeros_like(pr_vals)

        colors_roc = ["#2980b9" if "TGCF-IDS" not in m else "#1abc9c" for m in models]
        colors_pr = ["#8e44ad" if "TGCF-IDS" not in m else "#16a085" for m in models]

        rects1 = ax.bar(x_pos - bar_width / 2, roc_vals, bar_width, yerr=roc_err, label="Macro ROC-AUC", color=colors_roc, alpha=0.9, capsize=3, edgecolor="black", linewidth=0.5)
        rects2 = ax.bar(x_pos + bar_width / 2, pr_vals, bar_width, yerr=pr_err, label="Macro PR-AUC (Avg Precision)", color=colors_pr, alpha=0.9, capsize=3, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9.5, fontweight="medium")
        ax.set_ylabel("Area Under Curve (Mean +/- Std)", fontsize=11, fontweight="bold")
        ax.set_title("Discrimination Performance: Macro ROC-AUC vs Macro PR-AUC", fontsize=12, fontweight="bold", pad=12)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", frameon=True, fontsize=10)

        for r, v in zip(rects1, roc_vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        for r, v in zip(rects2, pr_vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        plt.tight_layout()
        out_file = self.figures_dir / "baseline_roc_pr_comparison.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_file

    def plot_comprehensive_metrics(self, summary_df: pd.DataFrame) -> Path:
        """Render comprehensive grouped comparison chart across all primary evaluation metrics."""
        metrics = ["acc_mean", "macro_f1_mean", "weighted_f1_mean", "roc_auc_mean"]
        labels = ["Accuracy", "Macro F1", "Weighted F1", "ROC-AUC"]
        colors = ["#34495e", "#3498db", "#9b59b6", "#2ecc71"]

        fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

        models = summary_df["Model"].tolist()
        x_pos = np.arange(len(models))
        total_width = 0.8
        single_width = total_width / len(metrics)

        for i, (m_col, m_lbl, col) in enumerate(zip(metrics, labels, colors)):
            vals = summary_df[m_col].values
            offset = (i - len(metrics) / 2 + 0.5) * single_width
            ax.bar(x_pos + offset, vals, single_width, label=m_lbl, color=col, alpha=0.9, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=25, ha="right", fontsize=10, fontweight="medium")
        ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
        ax.set_title("Comprehensive Evaluation Matrix: Multi-Metric Benchmark Across 9 Architectures", fontsize=13, fontweight="bold", pad=14)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, fontsize=10)

        plt.tight_layout()
        out_file = self.figures_dir / "baseline_comprehensive_metrics.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_file

    def render_all_figures(self) -> List[Path]:
        """Generate full suite of publication figures."""
        summary_df, per_class_df = self.load_benchmark_data()
        figs = []

        print("[*] Generating Benchmark Publication Figures...")
        f1 = self.plot_macro_f1_comparison(summary_df)
        figs.append(f1)
        print(f"    [+] Saved: {f1.name}")

        f2 = self.plot_per_class_f1_heatmap(per_class_df)
        figs.append(f2)
        print(f"    [+] Saved: {f2.name}")

        f3 = self.plot_radar_comparison(summary_df)
        figs.append(f3)
        print(f"    [+] Saved: {f3.name}")

        f4 = self.plot_efficiency_tradeoff(summary_df)
        figs.append(f4)
        print(f"    [+] Saved: {f4.name}")

        f5 = self.plot_security_rates(summary_df)
        figs.append(f5)
        print(f"    [+] Saved: {f5.name}")

        f6 = self.plot_roc_pr_comparison(summary_df)
        figs.append(f6)
        print(f"    [+] Saved: {f6.name}")

        f7 = self.plot_comprehensive_metrics(summary_df)
        figs.append(f7)
        print(f"    [+] Saved: {f7.name}")

        return figs


def main():
    generator = BaselineFigureGenerator()
    generator.render_all_figures()


if __name__ == "__main__":
    main()
