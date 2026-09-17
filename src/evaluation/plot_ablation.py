#!/usr/bin/env python3
"""
TGCF-IDS: Publication-Quality Ablation Visualizations.

Generates high-resolution comparative figures for the 9-experiment ablation study:
1. ablation_macro_f1_bar.png: Macro F1 and Weighted F1 across 9 configurations with error bars.
2. ablation_minority_recall_fpr.png: Minority-Class Recall vs False Positive Rate security trade-off.
3. ablation_component_contributions.png: Component contribution delta chart relative to Full TGCF-IDS.
4. ablation_radar_chart.png: Multi-metric spider chart across 6 core operational axes.
5. ablation_per_class_heatmap.png: Full 10-class x 9-model F1 score performance matrix.
6. ablation_efficiency_tradeoff.png: Macro F1 vs Inference Latency and Model Parameter Count.
7. ablation_comprehensive_dashboard.png: 4-panel publication composite dashboard.
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


class AblationFigureGenerator:
    """
    Renders publication-grade figures for systematic ablation study results.
    """

    EXPERIMENT_COLORS = [
        "#4A6984",  # A. GraphSAGE only
        "#E76F51",  # B. Transformer only
        "#2A9D8F",  # C. GraphSAGE + Transformer
        "#F4A261",  # D. Remove temporal edges
        "#E63946",  # E. Remove contrastive pretraining
        "#9B5DE5",  # F. Remove feature tokenizer
        "#00BBF9",  # G. Replace learned fusion with concat
        "#D4A373",  # H. Remove edge attributes
        "#1D3557",  # I. Full TGCF-IDS (Proposed)
    ]

    def __init__(
        self,
        summary_csv: Optional[Union[str, Path]] = None,
        per_class_csv: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.summary_csv = Path(summary_csv or ProjectPaths.RESULTS_TABLES / "ablation.csv")
        self.per_class_csv = Path(per_class_csv or ProjectPaths.RESULTS_TABLES / "ablation_per_class.csv")
        self.output_dir = Path(output_dir or ProjectPaths.RESULTS_FIGURES / "ablation")
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
        """Load ablation summary and per-class CSV tables."""
        if not self.summary_csv.exists():
            raise FileNotFoundError(f"Summary ablation table not found at: {self.summary_csv}")
        if not self.per_class_csv.exists():
            raise FileNotFoundError(f"Per-class ablation table not found at: {self.per_class_csv}")

        summary_df = pd.read_csv(self.summary_csv)
        per_class_df = pd.read_csv(self.per_class_csv)
        return summary_df, per_class_df

    def plot_macro_and_weighted_f1_bar(self, summary_df: pd.DataFrame) -> Path:
        """1. Dual bar chart for Macro F1 and Weighted F1 across all 9 ablation variants."""
        fig, ax = plt.subplots(figsize=(12, 6.5))

        experiments = summary_df["Experiment"].tolist()
        # Clean labels for plotting
        short_labels = [
            "A. GraphSAGE\nOnly",
            "B. Transformer\nOnly",
            "C. GraphSAGE +\nTransformer",
            "D. Remove\nTemp. Edges",
            "E. Remove\nPretraining",
            "F. Remove\nTokenizer",
            "G. Concat\nFusion",
            "H. Remove\nEdge Attr",
            "I. Full\nTGCF-IDS",
        ]

        x = np.arange(len(experiments))
        width = 0.38

        macro_f1 = summary_df["macro_f1_mean"].values
        macro_std = summary_df["macro_f1_std"].values
        weighted_f1 = summary_df["weighted_f1_mean"].values
        weighted_std = summary_df["weighted_f1_std"].values

        rects1 = ax.bar(
            x - width / 2,
            macro_f1,
            width,
            yerr=macro_std,
            capsize=4,
            label="Macro F1 Score",
            color="#2A9D8F",
            edgecolor="#1D3557",
            linewidth=1.2,
            alpha=0.9,
        )
        rects2 = ax.bar(
            x + width / 2,
            weighted_f1,
            width,
            yerr=weighted_std,
            capsize=4,
            label="Weighted F1 Score",
            color="#E76F51",
            edgecolor="#1D3557",
            linewidth=1.2,
            alpha=0.9,
        )

        ax.set_ylabel("F1 Score (Test Set)", fontweight="bold")
        ax.set_title("Ablation Study: Macro F1 vs Weighted F1 Performance Across Component Variants", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=10)
        ax.set_ylim(0.35, 0.92)
        ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", loc="upper left")

        # Value annotations on top of bars
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        out_path = self.output_dir / "ablation_macro_f1_bar.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_minority_recall_vs_fpr(self, summary_df: pd.DataFrame) -> Path:
        """2. Security Trade-Off: Rare Minority-Class Recall vs False Positive Rate (FPR)."""
        fig, ax = plt.subplots(figsize=(10, 6.5))

        fpr = summary_df["fpr_mean"].values * 100.0  # Percentage
        recall = summary_df["minority_recall_mean"].values * 100.0  # Percentage
        experiments = summary_df["Experiment"].tolist()

        short_names = [
            "A. GraphSAGE only",
            "B. Transformer only",
            "C. GraphSAGE + Transformer",
            "D. Remove temp. edges",
            "E. Remove pretrain",
            "F. Remove tokenizer",
            "G. Concat fusion",
            "H. Remove edge attr",
            "I. Full TGCF-IDS (Proposed)",
        ]

        for i, (f, r, name, color) in enumerate(zip(fpr, recall, short_names, self.EXPERIMENT_COLORS)):
            is_full = (i == len(experiments) - 1)
            marker = "*" if is_full else "o"
            size = 220 if is_full else 130
            ax.scatter(f, r, s=size, color=color, edgecolor="#1D3557", linewidth=1.5, marker=marker, zorder=5, label=name)

            offset_x = 0.5
            offset_y = 0.3
            if "Full" in name:
                ax.annotate(f"★ {name}\n(FPR: {f:.2f}%, Recall: {r:.2f}%)", xy=(f, r), xytext=(f + 0.8, r - 0.4),
                            fontsize=9.5, fontweight="bold", color="#1D3557",
                            arrowprops=dict(arrowstyle="->", color="#1D3557", lw=1.2))
            else:
                ax.annotate(name.split(". ")[0], xy=(f, r), xytext=(f + offset_x, r + offset_y),
                            fontsize=9, fontweight="bold", color="#333333")

        ax.set_xlabel("False Positive Rate (FPR %) — Lower is Better", fontweight="bold")
        ax.set_ylabel("Rare Minority-Class Recall (%) — Higher is Better", fontweight="bold")
        ax.set_title("Operational Security Trade-off: Rare Minority Recall vs False Positive Rate", pad=15)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=9.5)

        out_path = self.output_dir / "ablation_minority_recall_fpr.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_component_contributions(self, summary_df: pd.DataFrame) -> Path:
        """3. Component Contribution Delta Chart: Performance change when component is removed."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        full_row = summary_df[summary_df["Experiment"].str.contains("Full TGCF-IDS")].iloc[0]
        full_macro_f1 = full_row["macro_f1_mean"]
        full_fpr = full_row["fpr_mean"]

        ablation_rows = summary_df[~summary_df["Experiment"].str.contains("Full TGCF-IDS")].copy()

        # Delta Macro F1: (Ablated - Full) -> Negative means full is better
        delta_f1 = (ablation_rows["macro_f1_mean"].values - full_macro_f1) * 100.0  # percentage points
        # Delta FPR: (Ablated - Full) -> Positive means removing component increases FPR (bad)
        delta_fpr = (ablation_rows["fpr_mean"].values - full_fpr) * 100.0  # percentage points

        names = [
            "A. GraphSAGE only\n(No Transformer)",
            "B. Transformer only\n(No Graph Branch)",
            "C. Concat Fusion\n(No Pretrain)",
            "D. Remove Temp.\nEdges",
            "E. Remove Contrastive\nPretrain",
            "F. Remove Feature\nTokenizer",
            "G. Concat Fusion\n(With Pretrain)",
            "H. Remove Edge\nAttributes",
        ]

        y_pos = np.arange(len(names))

        # Panel 1: Delta Macro F1
        f1_colors = ["#E63946" if d < 0 else "#2A9D8F" for d in delta_f1]
        bars1 = ax1.barh(y_pos, delta_f1, color=f1_colors, edgecolor="#1D3557", height=0.6, alpha=0.85)
        ax1.axvline(0, color="black", linestyle="-", linewidth=1.2)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names, fontsize=9.5)
        ax1.invert_yaxis()
        ax1.set_xlabel(r"$\Delta$ Macro F1 (Percentage Points vs Full Model)", fontweight="bold")
        ax1.set_title(r"Impact on Macro F1 ($\Delta$ from Full TGCF-IDS)", pad=12)

        for bar, val in zip(bars1, delta_f1):
            offset = 0.2 if val >= 0 else -0.2
            ha = "left" if val >= 0 else "right"
            ax1.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.2f}%",
                     va="center", ha=ha, fontsize=8.5, fontweight="bold")

        # Panel 2: Delta False Positive Rate (FPR)
        fpr_colors = ["#E63946" if d > 0 else "#2A9D8F" for d in delta_fpr]
        bars2 = ax2.barh(y_pos, delta_fpr, color=fpr_colors, edgecolor="#1D3557", height=0.6, alpha=0.85)
        ax2.axvline(0, color="black", linestyle="-", linewidth=1.2)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([])
        ax2.invert_yaxis()
        ax2.set_xlabel(r"$\Delta$ False Positive Rate (pp — Lower is Better)", fontweight="bold")
        ax2.set_title(r"Impact on False Alarm Rate ($\Delta$ FPR)", pad=12)

        for bar, val in zip(bars2, delta_fpr):
            offset = 0.2 if val >= 0 else -0.2
            ha = "left" if val >= 0 else "right"
            ax2.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.2f}%",
                     va="center", ha=ha, fontsize=8.5, fontweight="bold")

        out_path = self.output_dir / "ablation_component_contributions.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_radar_chart(self, summary_df: pd.DataFrame) -> Path:
        """4. Spider / Radar chart comparing core variants across 6 operational dimensions."""
        categories = [
            "Accuracy",
            "Macro F1",
            "Weighted F1",
            "1 - FPR (Specificity)",
            "Rare Minority Recall",
            "Macro ROC-AUC",
        ]
        N = len(categories)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        key_experiments = [
            ("I. Full TGCF-IDS", "#1D3557", "-", 2.5),
            ("E. Remove contrastive pretraining", "#E63946", "--", 1.8),
            ("D. Remove temporal edges", "#F4A261", "-.", 1.8),
            ("A. GraphSAGE only", "#4A6984", ":", 1.5),
            ("B. Transformer only", "#E76F51", ":", 1.5),
        ]

        for exp_key, color, linestyle, lw in key_experiments:
            matching = summary_df[summary_df["Experiment"].str.startswith(exp_key.split()[0])]
            if matching.empty:
                continue
            row = matching.iloc[0]

            values = [
                row["acc_mean"],
                row["macro_f1_mean"],
                row["weighted_f1_mean"],
                1.0 - row["fpr_mean"],
                row["minority_recall_mean"] * 3.0,  # Scaled for visual radar comparability
                row["roc_auc_mean"],
            ]
            values += values[:1]

            ax.plot(angles, values, linewidth=lw, linestyle=linestyle, color=color, label=exp_key)
            ax.fill(angles, values, color=color, alpha=0.08)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10.5, fontweight="bold")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Multi-Metric Radar Comparison: Key Ablation Configurations", pad=25, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=True)

        out_path = self.output_dir / "ablation_radar_chart.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_per_class_f1_heatmap(self, per_class_df: pd.DataFrame) -> Path:
        """5. Full 10-Class F1-score heatmap matrix across all 9 ablation models."""
        pivot_df = per_class_df.pivot(index="Experiment", columns="Class Name", values="f1_mean")

        # Sort columns logically: Normal, High-Freq Attacks, Minority Attacks
        class_order = [
            "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
            "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"
        ]
        available_classes = [c for c in class_order if c in pivot_df.columns]
        pivot_df = pivot_df[available_classes]

        fig, ax = plt.subplots(figsize=(13, 7))
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            linewidths=0.8,
            cbar_kws={"label": "F1 Score"},
            ax=ax,
            annot_kws={"size": 9.5, "weight": "bold"},
        )

        ax.set_title("10-Class F1 Score Heatmap Matrix Across All 9 Ablation Configurations", pad=15)
        ax.set_ylabel("Ablation Experiment", fontweight="bold")
        ax.set_xlabel("UNSW-NB15 Traffic Class", fontweight="bold")
        plt.xticks(rotation=30, ha="right")

        out_path = self.output_dir / "ablation_per_class_heatmap.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_efficiency_tradeoff(self, summary_df: pd.DataFrame) -> Path:
        """6. Efficiency Trade-Off: Macro F1 vs Inference Latency & Parameter Count."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        latencies = summary_df["inf_time_mean"].values
        macro_f1 = summary_df["macro_f1_mean"].values
        params = summary_df["total_params_num"].values / 1e3  # in Thousands

        short_names = [
            "A. GraphSAGE only",
            "B. Transformer only",
            "C. GraphSAGE + Transformer",
            "D. Remove temp. edges",
            "E. Remove pretrain",
            "F. Remove tokenizer",
            "G. Concat fusion",
            "H. Remove edge attr",
            "I. Full TGCF-IDS",
        ]

        # Panel 1: Macro F1 vs Latency
        for l, f, name, color in zip(latencies, macro_f1, short_names, self.EXPERIMENT_COLORS):
            is_full = ("Full" in name)
            ax1.scatter(l, f, s=220 if is_full else 120, color=color, edgecolor="#1D3557",
                        linewidth=1.5, marker="*" if is_full else "o", label=name)
            ax1.annotate(name.split(". ")[0], (l + 0.15, f + 0.002), fontsize=9, fontweight="bold")

        ax1.set_xlabel("Inference Latency (ms / 1,000 Flows) — Lower is Better", fontweight="bold")
        ax1.set_ylabel("Macro F1 Score — Higher is Better", fontweight="bold")
        ax1.set_title("Performance vs Inference Latency", pad=12)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Panel 2: Macro F1 vs Parameter Count
        for p, f, name, color in zip(params, macro_f1, short_names, self.EXPERIMENT_COLORS):
            is_full = ("Full" in name)
            ax2.scatter(p, f, s=220 if is_full else 120, color=color, edgecolor="#1D3557",
                        linewidth=1.5, marker="*" if is_full else "o")
            ax2.annotate(name.split(". ")[0], (p + 3.0, f + 0.002), fontsize=9, fontweight="bold")

        ax2.set_xlabel("Model Parameters (Thousands / K) — Lower is Better", fontweight="bold")
        ax2.set_ylabel("Macro F1 Score — Higher is Better", fontweight="bold")
        ax2.set_title("Performance vs Model Complexity (Parameters)", pad=12)
        ax2.grid(True, linestyle="--", alpha=0.5)

        ax1.legend(bbox_to_anchor=(2.25, 1.0), loc="upper left", frameon=True, fontsize=9.5)

        out_path = self.output_dir / "ablation_efficiency_tradeoff.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_comprehensive_dashboard(self, summary_df: pd.DataFrame, per_class_df: pd.DataFrame) -> Path:
        """7. 4-Panel Composite Publication Dashboard."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # (a) Macro F1 Comparison
        x = np.arange(len(summary_df))
        labels = [f"Exp {chr(65 + i)}" for i in range(len(summary_df))]
        ax1.bar(x, summary_df["macro_f1_mean"].values, yerr=summary_df["macro_f1_std"].values,
                capsize=3, color=self.EXPERIMENT_COLORS, edgecolor="#1D3557", alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9.5)
        ax1.set_ylabel("Macro F1 Score", fontweight="bold")
        ax1.set_title("(a) Macro F1 Score Across 9 Ablation Variants", pad=10)
        ax1.set_ylim(0.35, 0.55)

        # (b) Minority Recall vs FPR
        fpr = summary_df["fpr_mean"].values * 100.0
        recall = summary_df["minority_recall_mean"].values * 100.0
        for i, (f, r, color) in enumerate(zip(fpr, recall, self.EXPERIMENT_COLORS)):
            ax2.scatter(f, r, s=150 if i == 8 else 90, color=color, edgecolor="#1D3557",
                        marker="*" if i == 8 else "o")
            ax2.annotate(chr(65 + i), (f + 0.5, r + 0.3), fontsize=9, fontweight="bold")
        ax2.set_xlabel("False Positive Rate (FPR %)", fontweight="bold")
        ax2.set_ylabel("Rare Minority-Class Recall (%)", fontweight="bold")
        ax2.set_title("(b) Security Trade-off: Minority Recall vs False Alarm Rate", pad=10)

        # (c) Component Contribution Deltas
        full_macro_f1 = summary_df[summary_df["Experiment"].str.contains("Full TGCF-IDS")]["macro_f1_mean"].values[0]
        ablation_rows = summary_df[~summary_df["Experiment"].str.contains("Full TGCF-IDS")]
        delta_f1 = (ablation_rows["macro_f1_mean"].values - full_macro_f1) * 100.0
        abl_labels = [f"Exp {chr(65 + i)}" for i in range(len(delta_f1))]
        c_colors = ["#E63946" if d < 0 else "#2A9D8F" for d in delta_f1]
        ax3.barh(np.arange(len(delta_f1)), delta_f1, color=c_colors, edgecolor="#1D3557", height=0.55)
        ax3.axvline(0, color="black", linestyle="-", linewidth=1.0)
        ax3.set_yticks(np.arange(len(delta_f1)))
        ax3.set_yticklabels(abl_labels, fontsize=9.5)
        ax3.invert_yaxis()
        ax3.set_xlabel(r"$\Delta$ Macro F1 (Percentage Points vs Full Model)", fontweight="bold")
        ax3.set_title(r"(c) Macro F1 Component Impact ($\Delta$ from Full TGCF-IDS)", pad=10)

        # (d) Accuracy vs Weighted F1
        acc = summary_df["acc_mean"].values
        wf1 = summary_df["weighted_f1_mean"].values
        for i, (a, w, color) in enumerate(zip(acc, wf1, self.EXPERIMENT_COLORS)):
            ax4.scatter(a, w, s=150 if i == 8 else 90, color=color, edgecolor="#1D3557",
                        marker="*" if i == 8 else "o")
            ax4.annotate(chr(65 + i), (a + 0.003, w + 0.001), fontsize=9, fontweight="bold")
        ax4.set_xlabel("Overall Accuracy", fontweight="bold")
        ax4.set_ylabel("Weighted F1 Score", fontweight="bold")
        ax4.set_title("(d) Accuracy vs Weighted F1 Concordance", pad=10)

        out_path = self.output_dir / "ablation_comprehensive_dashboard.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def generate_all_figures(self) -> List[Path]:
        """Generate full suite of publication ablation figures."""
        print("[*] Generating Ablation Publication Figures...")
        summary_df, per_class_df = self.load_data()

        generated = []
        p1 = self.plot_macro_and_weighted_f1_bar(summary_df)
        print(f"    [+] Saved: {p1.name}")
        generated.append(p1)

        p2 = self.plot_minority_recall_vs_fpr(summary_df)
        print(f"    [+] Saved: {p2.name}")
        generated.append(p2)

        p3 = self.plot_component_contributions(summary_df)
        print(f"    [+] Saved: {p3.name}")
        generated.append(p3)

        p4 = self.plot_radar_chart(summary_df)
        print(f"    [+] Saved: {p4.name}")
        generated.append(p4)

        p5 = self.plot_per_class_f1_heatmap(per_class_df)
        print(f"    [+] Saved: {p5.name}")
        generated.append(p5)

        p6 = self.plot_efficiency_tradeoff(summary_df)
        print(f"    [+] Saved: {p6.name}")
        generated.append(p6)

        p7 = self.plot_comprehensive_dashboard(summary_df, per_class_df)
        print(f"    [+] Saved: {p7.name}")
        generated.append(p7)

        return generated


def main():
    generator = AblationFigureGenerator()
    generator.generate_all_figures()


if __name__ == "__main__":
    main()
