#!/usr/bin/env python3
"""
TGCF-IDS: Publication-Quality Hyperparameter Search Visualizations.

Generates high-resolution comparative figures for the controlled hyperparameter study:
1. hp_macro_f1_distributions.png: Validation Macro-F1 across hyperparameter dimensions.
2. hp_parameter_sensitivity.png: Multi-metric sensitivity profiles (Macro-F1, Macro-Recall, FPR).
3. hp_pareto_f1_vs_recall.png: Validation Macro-F1 vs Macro-Recall Pareto frontier with optimal pick.
4. hp_radar_best_vs_baseline.png: Radar comparison of optimal config vs baseline reference.
5. hp_comprehensive_dashboard.png: 4-panel publication dashboard summarizing validation tuning.
"""

import json
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


class HyperparameterFigureGenerator:
    """
    Renders publication-grade figures for hyperparameter search results.
    """

    def __init__(
        self,
        search_csv: Optional[Union[str, Path]] = None,
        best_json: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.search_csv = Path(search_csv or ProjectPaths.RESULTS_TABLES / "hyperparameter_search.csv")
        self.best_json = Path(best_json or ProjectPaths.RESULTS_TABLES / "best_configuration.json")
        self.output_dir = Path(output_dir or ProjectPaths.RESULTS_FIGURES / "hyperparameters")
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

    def load_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load search CSV and best config JSON."""
        if not self.search_csv.exists():
            raise FileNotFoundError(f"Search table not found at: {self.search_csv}")
        df = pd.read_csv(self.search_csv)

        best_cfg = {}
        if self.best_json.exists():
            with open(self.best_json, "r", encoding="utf-8") as f:
                best_cfg = json.load(f)

        return df, best_cfg

    def plot_macro_f1_by_dimension(self, df: pd.DataFrame) -> Path:
        """1. Bar / Scatter chart of Validation Macro F1 across tested hyperparameter dimensions."""
        fig, ax = plt.subplots(figsize=(13, 6))

        tested_dims = df["tested_dimension"].unique().tolist()
        palette = sns.color_palette("tab10", len(tested_dims))

        sns.barplot(
            data=df,
            x="tested_dimension",
            y="val_macro_f1",
            palette=palette,
            edgecolor="#1D3557",
            alpha=0.85,
            capsize=0.1,
            ax=ax,
        )

        sns.stripplot(
            data=df,
            x="tested_dimension",
            y="val_macro_f1",
            color="#1D3557",
            size=6,
            jitter=0.2,
            ax=ax,
        )

        ax.set_title("Validation Macro F1 Variation Across 12 Hyperparameter Dimensions", pad=15)
        ax.set_ylabel("Validation Macro F1 Score", fontweight="bold")
        ax.set_xlabel("Tested Hyperparameter Axis", fontweight="bold")
        plt.xticks(rotation=35, ha="right", fontsize=10)

        out_path = self.output_dir / "hp_macro_f1_distributions.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_pareto_f1_vs_recall(self, df: pd.DataFrame, best_cfg: Dict[str, Any]) -> Path:
        """2. Validation Macro-F1 vs Macro-Recall Pareto frontier with optimal configuration highlighted."""
        fig, ax = plt.subplots(figsize=(10, 6.5))

        f1 = df["val_macro_f1"].values
        recall = df["val_macro_recall"].values
        scores = df["selection_score"].values
        exp_ids = df["experiment_id"].values

        scatter = ax.scatter(
            recall,
            f1,
            c=scores,
            cmap="viridis",
            s=120,
            edgecolor="#1D3557",
            linewidth=1.2,
            alpha=0.9,
            zorder=4,
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Validation Selection Score (0.55 F1 + 0.35 Rec + 0.10 Spec)", fontweight="bold")

        # Highlight best model
        best_id = best_cfg.get("experiment_id", "")
        best_row = df[df["experiment_id"] == best_id]
        if not best_row.empty:
            b_r = best_row.iloc[0]["val_macro_recall"]
            b_f1 = best_row.iloc[0]["val_macro_f1"]
            b_desc = best_row.iloc[0]["description"]
            ax.scatter(b_r, b_f1, s=280, color="#E63946", marker="*", edgecolor="#1D3557",
                       linewidth=1.8, zorder=6, label=f"Optimal: {best_id}")
            ax.annotate(
                f"Selected Optimal ({best_id})\n{b_desc}\n(Macro-F1: {b_f1:.4f}, Recall: {b_r:.4f})",
                xy=(b_r, b_f1),
                xytext=(b_r - 0.015, b_f1 - 0.008),
                fontsize=9.5,
                fontweight="bold",
                color="#1D3557",
                arrowprops=dict(arrowstyle="->", color="#E63946", lw=1.5),
            )

        ax.set_xlabel("Validation Macro Recall — Higher is Better", fontweight="bold")
        ax.set_ylabel("Validation Macro F1 Score — Higher is Better", fontweight="bold")
        ax.set_title("Hyperparameter Search: Validation Macro-F1 vs Macro-Recall Optimization Landscape", pad=15)
        ax.legend(loc="upper left", frameon=True)

        out_path = self.output_dir / "hp_pareto_f1_vs_recall.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_radar_best_vs_baseline(self, df: pd.DataFrame, best_cfg: Dict[str, Any]) -> Path:
        """3. Radar comparison of Selected Optimal vs Baseline Reference."""
        categories = [
            "Validation Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Weighted F1",
            "1 - FPR (Specificity)",
        ]
        N = len(categories)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Baseline reference
        base_row = df[df["tested_dimension"] == "baseline"]
        if not base_row.empty:
            b_vals = [
                base_row.iloc[0]["val_accuracy"],
                base_row.iloc[0]["val_macro_precision"],
                base_row.iloc[0]["val_macro_recall"],
                base_row.iloc[0]["val_macro_f1"],
                base_row.iloc[0]["val_weighted_f1"],
                1.0 - base_row.iloc[0]["val_fpr"],
            ]
            b_vals += b_vals[:1]
            ax.plot(angles, b_vals, linewidth=2.0, linestyle="--", color="#E76F51", label="Default Baseline Config")
            ax.fill(angles, b_vals, color="#E76F51", alpha=0.1)

        # Optimal selected
        best_id = best_cfg.get("experiment_id", "")
        best_row = df[df["experiment_id"] == best_id]
        if not best_row.empty:
            o_vals = [
                best_row.iloc[0]["val_accuracy"],
                best_row.iloc[0]["val_macro_precision"],
                best_row.iloc[0]["val_macro_recall"],
                best_row.iloc[0]["val_macro_f1"],
                best_row.iloc[0]["val_weighted_f1"],
                1.0 - best_row.iloc[0]["val_fpr"],
            ]
            o_vals += o_vals[:1]
            ax.plot(angles, o_vals, linewidth=2.5, linestyle="-", color="#1D3557", label=f"Selected Optimal ({best_id})")
            ax.fill(angles, o_vals, color="#1D3557", alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10.5, fontweight="bold")
        ax.set_ylim(0.4, 1.0)
        ax.set_title("Validation Radar Profile: Optimal Hyperparameters vs Baseline", pad=25, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True)

        out_path = self.output_dir / "hp_radar_best_vs_baseline.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_comprehensive_dashboard(self, df: pd.DataFrame, best_cfg: Dict[str, Any]) -> Path:
        """4. 4-Panel Comprehensive Hyperparameter Tuning Dashboard."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # (a) Top 10 Configurations by Selection Score
        top10 = df.head(10).sort_values(by="selection_score", ascending=True)
        ax1.barh(np.arange(len(top10)), top10["selection_score"], color="#2A9D8F", edgecolor="#1D3557", height=0.6)
        ax1.set_yticks(np.arange(len(top10)))
        ax1.set_yticklabels(top10["experiment_id"] + " (" + top10["tested_dimension"] + ")", fontsize=9)
        ax1.set_xlabel("Selection Score (0.55 F1 + 0.35 Rec + 0.10 Spec)", fontweight="bold")
        ax1.set_title("(a) Top 10 Configurations by Validation Selection Score", pad=10)

        # (b) Macro F1 vs Macro Recall
        scatter = ax2.scatter(df["val_macro_recall"], df["val_macro_f1"], c=df["val_fpr"] * 100.0,
                              cmap="coolwarm_r", s=100, edgecolor="#1D3557")
        cbar = fig.colorbar(scatter, ax=ax2)
        cbar.set_label("Validation FPR (%) — Lower is Better")
        ax2.set_xlabel("Validation Macro Recall", fontweight="bold")
        ax2.set_ylabel("Validation Macro F1", fontweight="bold")
        ax2.set_title("(b) Validation Macro-F1 vs Macro-Recall (Colored by FPR)", pad=10)

        # (c) Training Time by Architecture Depth
        sns.barplot(data=df, x="tested_dimension", y="train_time_sec", ax=ax3, color="#4A6984", edgecolor="#1D3557")
        ax3.set_xlabel("Tested Dimension", fontweight="bold")
        ax3.set_ylabel("Training Time (Seconds)", fontweight="bold")
        ax3.set_title("(c) Computational Cost per Experiment Run", pad=10)
        ax3.tick_params(axis="x", rotation=35)

        # (d) False Positive Rate vs Minority Recall
        ax4.scatter(df["val_fpr"] * 100.0, df["val_rare_minority_recall"] * 100.0, s=110,
                    color="#E76F51", edgecolor="#1D3557")
        ax4.set_xlabel("Validation False Positive Rate (FPR %)", fontweight="bold")
        ax4.set_ylabel("Rare Minority Recall (%)", fontweight="bold")
        ax4.set_title("(d) Security Trade-Off: Rare Minority Recall vs False Positive Rate", pad=10)

        out_path = self.output_dir / "hp_comprehensive_dashboard.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def generate_all_figures(self) -> List[Path]:
        """Generate full suite of hyperparameter figures."""
        print("[*] Generating Hyperparameter Search Publication Figures...")
        df, best_cfg = self.load_data()

        generated = []
        p1 = self.plot_macro_f1_by_dimension(df)
        print(f"    [+] Saved: {p1.name}")
        generated.append(p1)

        p2 = self.plot_pareto_f1_vs_recall(df, best_cfg)
        print(f"    [+] Saved: {p2.name}")
        generated.append(p2)

        p3 = self.plot_radar_best_vs_baseline(df, best_cfg)
        print(f"    [+] Saved: {p3.name}")
        generated.append(p3)

        p4 = self.plot_comprehensive_dashboard(df, best_cfg)
        print(f"    [+] Saved: {p4.name}")
        generated.append(p4)

        return generated


def main():
    generator = HyperparameterFigureGenerator()
    generator.generate_all_figures()


if __name__ == "__main__":
    main()
