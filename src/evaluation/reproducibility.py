#!/usr/bin/env python3
"""
TGCF-IDS: Strict Multi-Seed Reproducibility & Stability Verification Framework.

Evaluates the final selected TGCF-IDS configuration across 5 distinct random seeds:
Seeds: [42, 123, 2024, 2025, 2026]

Rigorous Deterministic Controls:
- Python standard library random
- NumPy random generator
- PyTorch CPU random state
- PyTorch CUDA & cuDNN deterministic execution (benchmark=False, deterministic=True)
- PyG DataLoader generator & worker_init_fn seeding
- Zero cherry-picking and full statistical reporting (Mean, Std, Min, Max).
"""

import argparse
import json
import os
import random
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
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.paths import ProjectPaths
from src.models.tgcf_ids import TGCFIDS


def seed_everything(seed: int) -> torch.Generator:
    """
    Rigorously control all stochastic sources across Python, NumPy, PyTorch, CUDA, and DataLoaders.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def seed_worker(worker_id: int) -> None:
    """Worker initialization function for PyTorch DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ReproducibilityEvaluator:
    """
    Multi-Seed Scientific Reproducibility Framework for TGCF-IDS.
    """

    CLASS_NAMES = [
        "Normal", "Analysis", "Backdoor", "DoS", "Exploits",
        "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
    ]
    NUM_CLASSES = 10
    EVALUATION_SEEDS = [42, 123, 2024, 2025, 2026]

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        graphs_dir: Optional[Union[str, Path]] = None,
        results_dir: Optional[Union[str, Path]] = None,
        pretrained_ckpt: Optional[Union[str, Path]] = None,
        epochs: int = 15,
        batch_size: int = 4,
        device: Optional[str] = None,
    ):
        self.config_path = Path(config_path or ProjectPaths.CONFIGS_DIR / "best_hyperparameters.yaml")
        self.graphs_dir = Path(graphs_dir or ProjectPaths.DATA_GRAPHS)
        self.results_tables_dir = Path(results_dir or ProjectPaths.RESULTS_TABLES)
        self.pretrained_ckpt = Path(pretrained_ckpt or ProjectPaths.RESULTS_CHECKPOINTS / "graph_pretrained.pt")
        self.epochs = epochs
        self.batch_size = batch_size

        self.results_tables_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_amp = (self.device.type == "cuda")

        self.hyperparameters = self._load_hyperparameters()
        self.train_graphs: Optional[List[Data]] = None
        self.test_graphs: Optional[List[Data]] = None
        self.class_weights: Optional[torch.Tensor] = None

    def _load_hyperparameters(self) -> Dict[str, Any]:
        """Load selected optimal hyperparameters."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data.get("hyperparameters", data)

        # Fallback default optimal configuration from Step 19
        return {
            "learning_rate": 0.002,
            "hidden_dimension": 64,
            "embedding_dimension": 64,
            "transformer_layers": 2,
            "transformer_heads": 4,
            "gnn_layers": 2,
            "dropout": 0.1,
            "knn_k": 10,
            "temporal_window": 30,
            "contrastive_temperature": 0.1,
            "feature_masking_ratio": 0.15,
            "edge_dropout": 0.15,
        }

    def load_data(self) -> None:
        """Load full training and testing graph snapshot collections."""
        train_path = self.graphs_dir / "train_graphs.pt"
        test_path = self.graphs_dir / "test_graphs.pt"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError("Graph archive files missing in data/graphs/")

        print(f"[+] Loading graph archives: {train_path.name}, {test_path.name}")
        self.train_graphs = torch.load(train_path, weights_only=False, map_location="cpu")
        self.test_graphs = torch.load(test_path, weights_only=False, map_location="cpu")
        print(f"[+] Graph Snapshots Loaded: Train={len(self.train_graphs)}, Test={len(self.test_graphs)}")

        # Calculate smooth class weights on training data
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
        smoothed = np.log1p(weights)
        smoothed = smoothed / smoothed.mean()
        self.class_weights = torch.tensor(smoothed, dtype=torch.float32, device=self.device)

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        train_time: float,
        inf_time: float,
    ) -> Dict[str, Any]:
        """Compute full suite of multi-class and security rates."""
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

        # ROC-AUC and PR-AUC
        roc_auc, pr_auc = 0.0, 0.0
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
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "train_time_sec": train_time,
            "inf_latency_ms_per_1k": inf_time,
            "per_class_precision": per_p.tolist(),
            "per_class_recall": per_r.tolist(),
            "per_class_f1": per_f1.tolist(),
        }

    def train_and_eval_seed(self, seed: int) -> Dict[str, Any]:
        """Train model under deterministic seed controls and evaluate on full test split."""
        generator = seed_everything(seed)

        hp = self.hyperparameters
        model = TGCFIDS(
            num_numerical=39,
            cat_cardinalities=[134, 14, 10],
            token_dim=hp["embedding_dimension"],
            transformer_heads=hp["transformer_heads"],
            transformer_layers=hp["transformer_layers"],
            transformer_ffn_dim=hp["embedding_dimension"] * 4,
            transformer_dropout=hp["dropout"],
            graph_in_channels=194,
            graph_edge_dim=6,
            graph_hidden_dim=hp["hidden_dimension"],
            graph_out_channels=hp["embedding_dimension"],
            graph_layers=hp["gnn_layers"],
            graph_dropout=hp["dropout"],
            fusion_dim=hp["hidden_dimension"] * 2,
            fusion_strategy="gated",
            classifier_hidden_dim=hp["hidden_dimension"],
            classifier_dropout=hp["dropout"] * 1.5,
            num_classes=self.NUM_CLASSES,
        ).to(self.device)

        # Load contrastive pretrained encoder weights
        if self.pretrained_ckpt.exists():
            model.load_pretrained_graph_encoder(self.pretrained_ckpt)

        train_loader = DataLoader(
            self.train_graphs,
            batch_size=self.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        test_loader = DataLoader(
            self.test_graphs,
            batch_size=self.batch_size,
            shuffle=False,
        )

        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hp["learning_rate"],
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Training
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

        # Test Evaluation
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
        inf_latency_per_1k = (eval_time / (total_samples / 1000.0)) * 1000.0

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        y_probs = np.concatenate(all_probs)

        return self._compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            train_time=train_time,
            inf_time=inf_latency_per_1k,
        )

    def run_reproducibility_benchmark(
        self,
        seeds: Optional[List[int]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute full 5-seed reproducibility benchmark without cherry-picking.
        """
        eval_seeds = seeds or self.EVALUATION_SEEDS

        print("=" * 85)
        print("     TGCF-IDS : Strict Multi-Seed Reproducibility & Stability Protocol     ")
        print("=" * 85)
        print(f"[*] Execution Device    : {self.device} (AMP Enabled: {self.use_amp})")
        print(f"[*] Evaluated Seeds     : {eval_seeds}")
        print(f"[*] Training Epochs     : {self.epochs}")
        print(f"[*] Hyperparameters     : lr={self.hyperparameters['learning_rate']}, hidden_dim={self.hyperparameters['hidden_dimension']}, emb_dim={self.hyperparameters['embedding_dimension']}")

        self.load_data()

        raw_records = []
        per_class_records = []

        metrics_list: Dict[str, List[float]] = {
            "accuracy": [],
            "macro_precision": [],
            "macro_recall": [],
            "macro_f1": [],
            "weighted_f1": [],
            "roc_auc": [],
            "fpr": [],
            "fnr": [],
            "train_time_sec": [],
            "inf_latency_ms_per_1k": [],
        }

        per_class_f1_matrix = []
        per_class_p_matrix = []
        per_class_r_matrix = []

        for seed in eval_seeds:
            print(f"\n[*] Evaluating Seed: {seed:>5} ... ", end="", flush=True)
            res = self.train_and_eval_seed(seed)

            for k in metrics_list:
                metrics_list[k].append(res[k])

            per_class_f1_matrix.append(res["per_class_f1"])
            per_class_p_matrix.append(res["per_class_precision"])
            per_class_r_matrix.append(res["per_class_recall"])

            raw_records.append({
                "Seed": seed,
                "Accuracy": res["accuracy"],
                "Macro Precision": res["macro_precision"],
                "Macro Recall": res["macro_recall"],
                "Macro F1": res["macro_f1"],
                "Weighted F1": res["weighted_f1"],
                "ROC-AUC": res["roc_auc"],
                "FPR": res["fpr"],
                "FNR": res["fnr"],
                "Train Time (s)": res["train_time_sec"],
                "Inf Latency (ms/1k)": res["inf_latency_ms_per_1k"],
            })

            print(
                f"Done! | Acc: {res['accuracy']:.4f} | "
                f"Macro F1: {res['macro_f1']:.4f} | "
                f"W-F1: {res['weighted_f1']:.4f} | "
                f"FPR: {res['fpr']:.4f} | "
                f"ROC-AUC: {res['roc_auc']:.4f}"
            )

        # Statistical Calculations across all 5 seeds: Mean, Std, Min, Max
        stats_rows = []
        metric_keys = [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Weighted F1",
            "ROC-AUC",
            "FPR",
            "FNR",
            "Train Time (s)",
            "Inf Latency (ms/1k)",
        ]
        key_mapping = {
            "Accuracy": "accuracy",
            "Macro Precision": "macro_precision",
            "Macro Recall": "macro_recall",
            "Macro F1": "macro_f1",
            "Weighted F1": "weighted_f1",
            "ROC-AUC": "roc_auc",
            "FPR": "fpr",
            "FNR": "fnr",
            "Train Time (s)": "train_time_sec",
            "Inf Latency (ms/1k)": "inf_latency_ms_per_1k",
        }

        for name in metric_keys:
            vals = np.array(metrics_list[key_mapping[name]])
            stats_rows.append({
                "Metric": name,
                "Mean": float(np.mean(vals)),
                "Std": float(np.std(vals)),
                "Min": float(np.min(vals)),
                "Max": float(np.max(vals)),
                "Formatted (Mean +/- Std)": f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}",
                "Range [Min, Max]": f"[{np.min(vals):.4f}, {np.max(vals):.4f}]",
            })

        # Per-class summary
        f1_mat = np.array(per_class_f1_matrix)
        p_mat = np.array(per_class_p_matrix)
        r_mat = np.array(per_class_r_matrix)

        for c_idx, c_name in enumerate(self.CLASS_NAMES):
            c_f1 = f1_mat[:, c_idx]
            c_p = p_mat[:, c_idx]
            c_r = r_mat[:, c_idx]
            per_class_records.append({
                "Class ID": c_idx,
                "Class Name": c_name,
                "F1 Mean": float(np.mean(c_f1)),
                "F1 Std": float(np.std(c_f1)),
                "F1 Min": float(np.min(c_f1)),
                "F1 Max": float(np.max(c_f1)),
                "Precision Mean": float(np.mean(c_p)),
                "Recall Mean": float(np.mean(c_r)),
                "Formatted F1": f"{np.mean(c_f1):.4f} +/- {np.std(c_f1):.4f}",
            })

        summary_df = pd.DataFrame(stats_rows)
        raw_df = pd.DataFrame(raw_records)
        per_class_df = pd.DataFrame(per_class_records)

        # Print Summary Table
        print("\n" + "=" * 85)
        print("          5-SEED SCIENTIFIC REPRODUCIBILITY RESULTS (Mean, Std, Min, Max)       ")
        print("=" * 85)
        print(summary_df[["Metric", "Formatted (Mean +/- Std)", "Range [Min, Max]"]].to_string(index=False))
        print("=" * 85)

        # Save to disk
        out_summary = self.results_tables_dir / "reproducibility.csv"
        out_raw = self.results_tables_dir / "reproducibility_raw_runs.csv"
        out_per_class = self.results_tables_dir / "reproducibility_per_class.csv"

        summary_df.to_csv(out_summary, index=False)
        raw_df.to_csv(out_raw, index=False)
        per_class_df.to_csv(out_per_class, index=False)

        print(f"[+] Saved Reproducibility Summary -> {out_summary}")
        print(f"[+] Saved Raw Multi-Seed Runs   -> {out_raw}")
        print(f"[+] Saved Per-Class Multi-Seed  -> {out_per_class}")

        return summary_df, raw_df, per_class_df


def main():
    parser = argparse.ArgumentParser(description="TGCF-IDS Strict Multi-Seed Reproducibility")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per seed run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024, 2025, 2026], help="Evaluation seeds")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    evaluator = ReproducibilityEvaluator(
        epochs=args.epochs,
        device=args.device,
    )
    evaluator.run_reproducibility_benchmark(seeds=args.seeds)


if __name__ == "__main__":
    main()
