#!/usr/bin/env python3
"""
TGCF-IDS: Rigorous Scientific Ablation Suite.

Conducts controlled multi-seed ablation experiments across 9 systematic configurations:
A. GraphSAGE only: Edge-Aware Temporal GraphSAGE -> Classifier (No Transformer branch)
B. Transformer only: FeatureTokenizer -> FeatureTransformer -> Classifier (No Graph branch)
C. GraphSAGE + Transformer: Dual-branch with Concat Fusion (No Pretraining)
D. Remove temporal edges: TGCF-IDS without temporal delta edge feature (static edge relations)
E. Remove contrastive pretraining: Full TGCF-IDS with Gated Fusion trained from scratch
F. Remove feature tokenizer: TGCF-IDS with direct linear projection instead of 42 token embeddings
G. Replace learned fusion with concatenation: TGCF-IDS with Concat Fusion + Pretraining
H. Remove edge attributes: TGCF-IDS without edge features (standard topology-only SAGEConv)
I. Full TGCF-IDS: All components (Tokenizer + Temporal GraphSAGE + Contrastive Pretrain + Gated Fusion)

All experiments share identical:
- Data splits (175,341 train flows / 82,332 test flows)
- Graph temporal windows (176 train / 83 test snapshots)
- Preprocessing and feature normalizations
- Random seeds (42, 123, 456)
- Optimization parameters (CosineAnnealingLR, AMP, Class Weights)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.paths import ProjectPaths
from src.models.ablation_models import (
    TransformerOnlyClassifier,
    GraphSAGEOnlyClassifier,
    AblationTGCFIDS,
)
from src.models.tgcf_ids import TGCFIDS


class AblationBenchmarkSuite:
    """
    Unified Controlled Ablation Benchmarking Suite.
    """

    CLASS_NAMES = [
        "Normal", "Analysis", "Backdoor", "DoS", "Exploits",
        "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
    ]
    NUM_CLASSES = 10
    RARE_MINORITY_CLASSES = [1, 2, 8, 9]  # Analysis, Backdoor, Shellcode, Worms

    EXPERIMENT_ORDER = [
        "A. GraphSAGE only",
        "B. Transformer only",
        "C. GraphSAGE + Transformer",
        "D. Remove temporal edges",
        "E. Remove contrastive pretraining",
        "F. Remove feature tokenizer",
        "G. Replace learned fusion with concatenation",
        "H. Remove edge attributes",
        "I. Full TGCF-IDS",
    ]

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        graphs_dir: Optional[Union[str, Path]] = None,
        results_dir: Optional[Union[str, Path]] = None,
        pretrained_ckpt: Optional[Union[str, Path]] = None,
        seeds: List[int] = [42, 123, 456],
        epochs: int = 15,
        batch_size: int = 4,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir or ProjectPaths.DATA_PROCESSED)
        self.graphs_dir = Path(graphs_dir or ProjectPaths.DATA_GRAPHS)
        self.results_tables_dir = Path(results_dir or ProjectPaths.RESULTS_TABLES)
        self.pretrained_ckpt = Path(pretrained_ckpt or ProjectPaths.RESULTS_CHECKPOINTS / "graph_pretrained.pt")
        self.seeds = seeds
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.results_tables_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_amp = (self.device.type == "cuda")

        self.train_graphs: Optional[List[Data]] = None
        self.test_graphs: Optional[List[Data]] = None
        self.class_weights: Optional[torch.Tensor] = None

    def _set_seed(self, seed: int) -> None:
        """Enforce deterministic reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_graph_archives(self) -> None:
        """Load pre-constructed PyG graph snapshot archives."""
        train_path = self.graphs_dir / "train_graphs.pt"
        test_path = self.graphs_dir / "test_graphs.pt"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError("Graph archive files missing in data/graphs/")

        print(f"[+] Loading graph archives: {train_path.name}, {test_path.name}")
        self.train_graphs = torch.load(train_path, weights_only=False, map_location="cpu")
        self.test_graphs = torch.load(test_path, weights_only=False, map_location="cpu")
        print(f"[+] Graph Snapshots: Train={len(self.train_graphs)}, Test={len(self.test_graphs)}")

        # Calculate smooth class weights
        targets = []
        for g in self.train_graphs:
            if hasattr(g, "y_multiclass") and g.y_multiclass is not None:
                targets.append(g.y_multiclass)
            elif hasattr(g, "y") and g.y is not None:
                targets.append(g.y)
        all_y = torch.cat(targets).numpy()
        counts = np.bincount(all_y, minlength=self.NUM_CLASSES)
        total = len(all_y)
        weights = total / (self.NUM_CLASSES * np.maximum(counts, 1.0))
        # Log-smoothed weights
        smoothed = np.log1p(weights)
        smoothed = smoothed / smoothed.mean()
        self.class_weights = torch.tensor(smoothed, dtype=torch.float32, device=self.device)

    def _compute_minority_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute macro recall on minority and attack classes."""
        recalls = recall_score(y_true, y_pred, average=None, labels=list(range(self.NUM_CLASSES)), zero_division=0)
        rare_recall = float(np.mean([recalls[c] for c in self.RARE_MINORITY_CLASSES]))
        all_attacks_recall = float(np.mean([recalls[c] for c in range(1, self.NUM_CLASSES)]))
        return {
            "minority_recall_rare4": rare_recall,
            "minority_recall_all_attacks": all_attacks_recall,
            "per_class_recall": recalls.tolist(),
        }

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: Optional[np.ndarray],
        train_time: float,
        inf_time: float,
        total_params: Optional[int] = None,
        trainable_params: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compute full suite of evaluation metrics."""
        acc = float(accuracy_score(y_true, y_pred))
        macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        per_p = precision_score(y_true, y_pred, average=None, labels=list(range(self.NUM_CLASSES)), zero_division=0)
        per_r = recall_score(y_true, y_pred, average=None, labels=list(range(self.NUM_CLASSES)), zero_division=0)
        per_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(self.NUM_CLASSES)), zero_division=0)

        # Binary security rates: Normal (0) vs Attack (1-9)
        y_bin_true = (y_true > 0).astype(int)
        y_bin_pred = (y_pred > 0).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_bin_true, y_bin_pred, labels=[0, 1]).ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        # Minority Recall
        minority_res = self._compute_minority_metrics(y_true, y_pred)

        # Multiclass ROC-AUC and PR-AUC
        roc_auc, pr_auc = 0.0, 0.0
        if y_probs is not None:
            try:
                y_true_bin = label_binarize(y_true, classes=list(range(self.NUM_CLASSES)))
                roc_auc = float(roc_auc_score(y_true_bin, y_probs, multi_class="ovr", average="macro"))
                pr_auc = float(average_precision_score(y_true_bin, y_probs, average="macro"))
            except Exception:
                pass

        return {
            "accuracy": acc,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "fpr": fpr,
            "fnr": fnr,
            "minority_recall_rare4": minority_res["minority_recall_rare4"],
            "minority_recall_all_attacks": minority_res["minority_recall_all_attacks"],
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "train_time_sec": train_time,
            "inf_latency_ms_per_1k": inf_time,
            "total_params": total_params if total_params is not None else 0,
            "trainable_params": trainable_params if trainable_params is not None else 0,
            "per_class_precision": per_p.tolist(),
            "per_class_recall": per_r.tolist(),
            "per_class_f1": per_f1.tolist(),
        }

    def _build_model(self, exp_name: str) -> nn.Module:
        """Construct model instance for specific ablation experiment."""
        if exp_name == "A. GraphSAGE only":
            model = GraphSAGEOnlyClassifier(
                in_channels=194,
                edge_dim=6,
                hidden_dim=64,
                out_channels=64,
                num_layers=2,
                dropout=0.1,
                num_classes=10,
            )
        elif exp_name == "B. Transformer only":
            model = TransformerOnlyClassifier(
                num_numerical=39,
                cat_cardinalities=[134, 14, 10],
                token_dim=64,
                transformer_heads=4,
                transformer_layers=2,
                transformer_ffn_dim=256,
                num_classes=10,
            )
        elif exp_name == "C. GraphSAGE + Transformer":
            # Concat fusion, no pretraining
            model = AblationTGCFIDS(
                num_numerical=39,
                cat_cardinalities=[134, 14, 10],
                token_dim=64,
                graph_in_channels=194,
                graph_edge_dim=6,
                fusion_strategy="concat",
                num_classes=10,
            )
        elif exp_name == "D. Remove temporal edges":
            # Zero out temporal interval delta (feature index 0) on edges, with pretraining + gated fusion
            model = AblationTGCFIDS(
                num_numerical=39,
                cat_cardinalities=[134, 14, 10],
                token_dim=64,
                graph_in_channels=194,
                graph_edge_dim=6,
                fusion_strategy="gated",
                remove_temporal_edges=True,
                num_classes=10,
            )
            if self.pretrained_ckpt.exists():
                model.load_pretrained_graph_encoder(self.pretrained_ckpt)
        elif exp_name == "E. Remove contrastive pretraining":
            # Full architecture with gated fusion, trained from scratch without pretraining
            model = AblationTGCFIDS(
                num_numerical=39,
                cat_cardinalities=[134, 14, 10],
                token_dim=64,
                graph_in_channels=194,
                graph_edge_dim=6,
                fusion_strategy="gated",
                num_classes=10,
            )
        elif exp_name == "F. Remove feature tokenizer":
            # Direct linear projection instead of 42 token embeddings
            model = AblationTGCFIDS(
                num_numerical=39,
                cat_cardinalities=[134, 14, 10],
                token_dim=64,
                graph_in_channels=194,
                graph_edge_dim=6,
                fusion_strategy="gated",
                remove_feature_tokenizer=True,
                num_classes=10,
            )
            if self.pretrained_ckpt.exists():
                model.load_pretrained_graph_encoder(self.pretrained_ckpt)
        elif exp_name == "G. Replace learned fusion with concatenation":
            # Concat fusion with pretraining
            model = AblationTGCFIDS(
                num_numerical=39,
                cat_cardinalities=[134, 14, 10],
                token_dim=64,
                graph_in_channels=194,
                graph_edge_dim=6,
                fusion_strategy="concat",
                num_classes=10,
            )
            if self.pretrained_ckpt.exists():
                model.load_pretrained_graph_encoder(self.pretrained_ckpt)
        elif exp_name == "H. Remove edge attributes":
            # Topology-only SAGEConv (all edge attributes zeroed)
            model = AblationTGCFIDS(
                num_numerical=39,
                cat_cardinalities=[134, 14, 10],
                token_dim=64,
                graph_in_channels=194,
                graph_edge_dim=6,
                fusion_strategy="gated",
                remove_edge_attrs=True,
                num_classes=10,
            )
            if self.pretrained_ckpt.exists():
                model.load_pretrained_graph_encoder(self.pretrained_ckpt)
        elif exp_name == "I. Full TGCF-IDS":
            # Full proposed architecture
            model = TGCFIDS(
                num_numerical=39,
                cat_cardinalities=[134, 14, 10],
                token_dim=64,
                graph_in_channels=194,
                graph_edge_dim=6,
                fusion_dim=128,
                fusion_strategy="gated",
                num_classes=10,
            )
            if self.pretrained_ckpt.exists():
                model.load_pretrained_graph_encoder(self.pretrained_ckpt)
        else:
            raise ValueError(f"Unknown experiment: {exp_name}")

        return model

    def _train_and_eval_experiment(self, exp_name: str, seed: int) -> Dict[str, Any]:
        """Train and evaluate single ablation model run."""
        self._set_seed(seed)
        model = self._build_model(exp_name).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        train_loader = DataLoader(self.train_graphs, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_graphs, batch_size=self.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Training Loop
        model.train()
        t0 = time.perf_counter()
        for epoch in range(self.epochs):
            for batch in train_loader:
                batch = batch.to(self.device)
                targets = batch.y_multiclass if hasattr(batch, "y_multiclass") and batch.y_multiclass is not None else batch.y
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    out = model(data=batch)
                    logits = out.logits if hasattr(out, "logits") else out
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        train_time = time.perf_counter() - t0

        # Evaluation Loop
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        t_eval0 = time.perf_counter()
        total_samples = 0

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                targets = batch.y_multiclass if hasattr(batch, "y_multiclass") and batch.y_multiclass is not None else batch.y
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    out = model(data=batch)
                    logits = out.logits if hasattr(out, "logits") else out
                    probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                total_samples += len(targets)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        eval_time = time.perf_counter() - t_eval0
        inf_latency_per_1k = (eval_time / (total_samples / 1000.0)) * 1000.0  # ms per 1k flows

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        y_probs = np.concatenate(all_probs)

        return self._compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            train_time=train_time,
            inf_time=inf_latency_per_1k,
            total_params=total_params,
            trainable_params=trainable_params,
        )

    def run_ablation_suite(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute full controlled multi-seed ablation benchmark across all 9 experiments."""
        print("=" * 80)
        print("           TGCF-IDS : Systematic Controlled Ablation Benchmark           ")
        print("=" * 80)
        print(f"[*] Execution Device     : {self.device} (AMP Enabled: {self.use_amp})")
        print(f"[*] Evaluated Seeds      : {self.seeds}")
        print(f"[*] Epochs per Deep Run  : {self.epochs}")

        self.load_graph_archives()

        summary_rows = []
        raw_runs = []
        per_class_rows = []

        for exp_name in self.EXPERIMENT_ORDER:
            print("\n" + "=" * 80)
            print(f"[*] Running Ablation Experiment: {exp_name}")
            print("=" * 80)

            seed_metrics: Dict[str, List[float]] = {
                "accuracy": [],
                "macro_precision": [],
                "macro_recall": [],
                "macro_f1": [],
                "weighted_f1": [],
                "fpr": [],
                "fnr": [],
                "minority_recall_rare4": [],
                "minority_recall_all_attacks": [],
                "roc_auc": [],
                "pr_auc": [],
                "train_time_sec": [],
                "inf_latency_ms_per_1k": [],
            }
            per_class_p: List[List[float]] = []
            per_class_r: List[List[float]] = []
            per_class_f1: List[List[float]] = []

            total_p, train_p = None, None

            for seed in self.seeds:
                print(f"  [+] Starting Seed: {seed:>3} ... ", end="", flush=True)
                res = self._train_and_eval_experiment(exp_name, seed)

                total_p = res["total_params"]
                train_p = res["trainable_params"]

                for k in seed_metrics:
                    seed_metrics[k].append(res[k])

                per_class_p.append(res["per_class_precision"])
                per_class_r.append(res["per_class_recall"])
                per_class_f1.append(res["per_class_f1"])

                raw_runs.append({
                    "Experiment": exp_name,
                    "Seed": seed,
                    **{k: res[k] for k in seed_metrics},
                })

                print(
                    f"Done! | Acc: {res['accuracy']:.4f} | "
                    f"Macro F1: {res['macro_f1']:.4f} | "
                    f"W-F1: {res['weighted_f1']:.4f} | "
                    f"Rare Minority Recall: {res['minority_recall_rare4']:.4f} | "
                    f"FPR: {res['fpr']:.4f}"
                )

            # Compute mean and standard deviations
            means = {k: float(np.mean(v)) for k, v in seed_metrics.items()}
            stds = {k: float(np.std(v)) for k, v in seed_metrics.items()}

            summary_rows.append({
                "Experiment": exp_name,
                "Accuracy": f"{means['accuracy']:.4f} +/- {stds['accuracy']:.4f}",
                "Macro Precision": f"{means['macro_precision']:.4f} +/- {stds['macro_precision']:.4f}",
                "Macro Recall": f"{means['macro_recall']:.4f} +/- {stds['macro_recall']:.4f}",
                "Macro F1": f"{means['macro_f1']:.4f} +/- {stds['macro_f1']:.4f}",
                "Weighted F1": f"{means['weighted_f1']:.4f} +/- {stds['weighted_f1']:.4f}",
                "Binary FPR": f"{means['fpr']:.4f} +/- {stds['fpr']:.4f}",
                "Binary FNR": f"{means['fnr']:.4f} +/- {stds['fnr']:.4f}",
                "Minority Recall (Rare 4)": f"{means['minority_recall_rare4']:.4f} +/- {stds['minority_recall_rare4']:.4f}",
                "Minority Recall (All Attacks)": f"{means['minority_recall_all_attacks']:.4f} +/- {stds['minority_recall_all_attacks']:.4f}",
                "ROC-AUC (Macro)": f"{means['roc_auc']:.4f} +/- {stds['roc_auc']:.4f}",
                "PR-AUC (Macro)": f"{means['pr_auc']:.4f} +/- {stds['pr_auc']:.4f}",
                "Train Time (s)": f"{means['train_time_sec']:.1f} +/- {stds['train_time_sec']:.1f}",
                "Inf Latency (ms/1k)": f"{means['inf_latency_ms_per_1k']:.2f} +/- {stds['inf_latency_ms_per_1k']:.2f}",
                "Total Params": f"{total_p:,}",
                "Trainable Params": f"{train_p:,}",
                # Numeric fields for figure plotting
                "acc_mean": means["accuracy"],
                "acc_std": stds["accuracy"],
                "macro_f1_mean": means["macro_f1"],
                "macro_f1_std": stds["macro_f1"],
                "weighted_f1_mean": means["weighted_f1"],
                "weighted_f1_std": stds["weighted_f1"],
                "minority_recall_mean": means["minority_recall_rare4"],
                "minority_recall_std": stds["minority_recall_rare4"],
                "fpr_mean": means["fpr"],
                "fnr_mean": means["fnr"],
                "roc_auc_mean": means["roc_auc"],
                "pr_auc_mean": means["pr_auc"],
                "train_time_mean": means["train_time_sec"],
                "inf_time_mean": means["inf_latency_ms_per_1k"],
                "total_params_num": total_p,
            })

            # Per-class mean +/- std across seeds
            per_class_p_arr = np.array(per_class_p)
            per_class_r_arr = np.array(per_class_r)
            per_class_f1_arr = np.array(per_class_f1)

            for c_idx, c_name in enumerate(self.CLASS_NAMES):
                per_class_rows.append({
                    "Experiment": exp_name,
                    "Class ID": c_idx,
                    "Class Name": c_name,
                    "Precision": f"{per_class_p_arr[:, c_idx].mean():.4f} +/- {per_class_p_arr[:, c_idx].std():.4f}",
                    "Recall": f"{per_class_r_arr[:, c_idx].mean():.4f} +/- {per_class_r_arr[:, c_idx].std():.4f}",
                    "F1 Score": f"{per_class_f1_arr[:, c_idx].mean():.4f} +/- {per_class_f1_arr[:, c_idx].std():.4f}",
                    "f1_mean": float(per_class_f1_arr[:, c_idx].mean()),
                    "f1_std": float(per_class_f1_arr[:, c_idx].std()),
                    "recall_mean": float(per_class_r_arr[:, c_idx].mean()),
                    "recall_std": float(per_class_r_arr[:, c_idx].std()),
                })

        summary_df = pd.DataFrame(summary_rows)
        per_class_df = pd.DataFrame(per_class_rows)
        raw_df = pd.DataFrame(raw_runs)

        # Print summary table
        print("\n" + "=" * 80)
        print("          CONTROLLED MULTI-SEED ABLATION STUDY (Mean +/- Std)          ")
        print("=" * 80)
        display_cols = [
            "Experiment",
            "Accuracy",
            "Macro F1",
            "Weighted F1",
            "Binary FPR",
            "Minority Recall (Rare 4)",
            "ROC-AUC (Macro)",
        ]
        print(summary_df[display_cols].to_string(index=False))
        print("=" * 80)

        # Save to disk
        out_summary = self.results_tables_dir / "ablation.csv"
        out_per_class = self.results_tables_dir / "ablation_per_class.csv"
        out_raw = self.results_tables_dir / "ablation_raw_runs.csv"

        summary_df.to_csv(out_summary, index=False)
        per_class_df.to_csv(out_per_class, index=False)
        raw_df.to_csv(out_raw, index=False)

        print(f"[+] Saved Summary Table -> {out_summary}")
        print(f"[+] Saved Per-Class Table -> {out_per_class}")
        print(f"[+] Saved Raw Runs -> {out_raw}")

        return summary_df, per_class_df, raw_df


def main():
    parser = argparse.ArgumentParser(description="TGCF-IDS Controlled Ablation Suite")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per deep run")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (graph snapshots)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Random seeds")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    suite = AblationBenchmarkSuite(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seeds=args.seeds,
        device=args.device,
    )
    suite.run_ablation_suite()


if __name__ == "__main__":
    main()
