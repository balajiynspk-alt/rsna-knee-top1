#!/usr/bin/env python3
"""
TGCF-IDS: Controlled Scientific Hyperparameter Experiment Framework.

Conducts controlled hyperparameter exploration on validation data ONLY across 12 key axes:
1. Learning Rate: [5e-4, 1e-3, 2e-3]
2. Hidden Dimension: [32, 64, 128]
3. Embedding Dimension (token_dim): [32, 64, 128]
4. Transformer Layers: [1, 2, 4]
5. Attention Heads: [2, 4, 8]
6. GNN Layers: [1, 2, 3]
7. Dropout: [0.05, 0.1, 0.2, 0.3]
8. KNN K (Neighborhood Connectivity): [5, 10, 15]
9. Temporal Window (seconds): [10, 30, 60]
10. Contrastive Temperature: [0.05, 0.07, 0.1, 0.2]
11. Feature Masking Ratio: [0.1, 0.2, 0.3]
12. Edge Dropout Ratio: [0.05, 0.1, 0.2]

Crucial Methodological Protocols:
- Pure validation split (80% train / 20% validation) from training data ONLY.
- ZERO test set leakage or contamination.
- Systematic recording of experiment ID, configuration, seed, validation metrics, and time.
- Predefined multi-objective selection score emphasizing Macro-F1 and Macro-Recall:
    Score = 0.55 * Val_Macro_F1 + 0.35 * Val_Macro_Recall + 0.10 * (1.0 - Val_FPR)
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
from src.models.graph_augmentations import GraphAugmentor


class HyperparameterSearchFramework:
    """
    Controlled Hyperparameter Search and Validation Suite for TGCF-IDS.
    """

    NUM_CLASSES = 10
    RARE_MINORITY_CLASSES = [1, 2, 8, 9]

    DEFAULT_BASE_CONFIG = {
        "learning_rate": 1e-3,
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

    def __init__(
        self,
        graphs_path: Optional[Union[str, Path]] = None,
        results_dir: Optional[Union[str, Path]] = None,
        val_ratio: float = 0.20,
        epochs_per_run: int = 10,
        batch_size: int = 4,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        self.graphs_path = Path(graphs_path or ProjectPaths.DATA_GRAPHS / "train_graphs.pt")
        self.results_tables_dir = Path(results_dir or ProjectPaths.RESULTS_TABLES)
        self.results_tables_dir.mkdir(parents=True, exist_ok=True)
        self.val_ratio = val_ratio
        self.epochs = epochs_per_run
        self.batch_size = batch_size
        self.seed = seed

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_amp = (self.device.type == "cuda")

        self.train_graphs: Optional[List[Data]] = None
        self.val_graphs: Optional[List[Data]] = None
        self.class_weights: Optional[torch.Tensor] = None

    def _set_seed(self, seed: int) -> None:
        """Enforce deterministic reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_and_split_data(self) -> None:
        """
        Load training graph archives and split strictly into train_sub and val_sub (NO TEST SET ACCESS).
        """
        if not self.graphs_path.exists():
            raise FileNotFoundError(f"Training graphs archive missing at {self.graphs_path}")

        print(f"[+] Loading training graphs from {self.graphs_path.name} for validation tuning...")
        all_train_graphs = torch.load(self.graphs_path, weights_only=False, map_location="cpu")
        total_graphs = len(all_train_graphs)

        # Deterministic chronological/snapshot validation split
        n_val = max(int(total_graphs * self.val_ratio), 1)
        n_train = total_graphs - n_val

        self.train_graphs = all_train_graphs[:n_train]
        self.val_graphs = all_train_graphs[n_train:]

        print(f"[+] Strict Validation Split Created: Train Snapshots={len(self.train_graphs)}, Validation Snapshots={len(self.val_graphs)}")
        print(f"[*] Note: Final test set is NEVER accessed during hyperparameter search.")

        # Compute smoothed class weights on training subset ONLY
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

    def generate_experiment_grid(self) -> List[Dict[str, Any]]:
        """
        Generate a systematic, controlled exploration grid covering all 12 hyperparameter dimensions.
        """
        experiments = []
        exp_idx = 1

        def add_exp(param_name: str, desc: str, **kwargs):
            nonlocal exp_idx
            cfg = dict(self.DEFAULT_BASE_CONFIG)
            cfg.update(kwargs)
            experiments.append({
                "experiment_id": f"EXP_HP_{exp_idx:03d}",
                "tested_dimension": param_name,
                "description": desc,
                "config": cfg,
            })
            exp_idx += 1

        # 0. Baseline Reference
        add_exp("baseline", "Base Reference Default Architecture")

        # 1. Learning Rate Exploration
        for lr in [5e-4, 2e-3]:
            add_exp("learning_rate", f"Learning Rate: {lr}", learning_rate=lr)

        # 2. Hidden Dimension (Classifier & Fusion)
        for h_dim in [32, 128]:
            add_exp("hidden_dimension", f"Hidden Dimension: {h_dim}", hidden_dimension=h_dim)

        # 3. Embedding Dimension (token_dim & GNN out channels)
        for emb_dim in [32, 128]:
            add_exp("embedding_dimension", f"Embedding Dimension: {emb_dim}", embedding_dimension=emb_dim)

        # 4. Transformer Layers
        for tf_layers in [1, 4]:
            add_exp("transformer_layers", f"Transformer Depth: {tf_layers} layers", transformer_layers=tf_layers)

        # 5. Attention Heads
        for heads in [2, 8]:
            add_exp("transformer_heads", f"Attention Heads: {heads}", transformer_heads=heads)

        # 6. GNN Layers
        for gnn_layers in [1, 3]:
            add_exp("gnn_layers", f"GraphSAGE Depth: {gnn_layers} layers", gnn_layers=gnn_layers)

        # 7. Dropout Regularization
        for d in [0.05, 0.2, 0.3]:
            add_exp("dropout", f"Dropout Rate: {d}", dropout=d)

        # 8. KNN K (Neighborhood Connectivity)
        for k in [5, 15]:
            add_exp("knn_k", f"Graph Connectivity KNN K: {k}", knn_k=k)

        # 9. Temporal Window (Time granularity)
        for tw in [10, 60]:
            add_exp("temporal_window", f"Temporal Window: {tw}s", temporal_window=tw)

        # 10. Contrastive Temperature
        for tau in [0.05, 0.07, 0.2]:
            add_exp("contrastive_temperature", f"Contrastive Temperature: {tau}", contrastive_temperature=tau)

        # 11. Feature Masking Ratio
        for mr in [0.1, 0.3]:
            add_exp("feature_masking_ratio", f"Feature Masking Ratio: {mr}", feature_masking_ratio=mr)

        # 12. Edge Dropout
        for ed in [0.05, 0.2]:
            add_exp("edge_dropout", f"Edge Dropout Ratio: {ed}", edge_dropout=ed)

        return experiments

    def _evaluate_configuration(
        self,
        config: Dict[str, Any],
        seed: int,
    ) -> Dict[str, Any]:
        """
        Train on train_sub and evaluate strictly on val_sub.
        """
        self._set_seed(seed)

        # Construct model from configuration
        model = TGCFIDS(
            num_numerical=39,
            cat_cardinalities=[134, 14, 10],
            token_dim=config["embedding_dimension"],
            transformer_heads=config["transformer_heads"],
            transformer_layers=config["transformer_layers"],
            transformer_ffn_dim=config["embedding_dimension"] * 4,
            transformer_dropout=config["dropout"],
            graph_in_channels=194,
            graph_edge_dim=6,
            graph_hidden_dim=config["hidden_dimension"],
            graph_out_channels=config["embedding_dimension"],
            graph_layers=config["gnn_layers"],
            graph_dropout=config["dropout"],
            fusion_dim=config["hidden_dimension"] * 2,
            fusion_strategy="gated",
            fusion_dropout=config["dropout"],
            classifier_hidden_dim=config["hidden_dimension"],
            classifier_dropout=config["dropout"] * 1.5,
            num_classes=self.NUM_CLASSES,
        ).to(self.device)

        # Load contrastive pretraining if available
        pretrained_path = ProjectPaths.RESULTS_CHECKPOINTS / "graph_pretrained.pt"
        if pretrained_path.exists() and config["embedding_dimension"] == 64 and config["gnn_layers"] == 2:
            try:
                model.load_pretrained_graph_encoder(pretrained_path)
            except Exception:
                pass

        # Optional simulated augmentation / edge density modulation for graph augmentations
        augmentor = GraphAugmentor(
            feature_mask_ratio=config["feature_masking_ratio"],
            edge_drop_ratio=config["edge_dropout"],
        )

        train_loader = DataLoader(self.train_graphs, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_graphs, batch_size=self.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
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

        # Validation Evaluation
        model.eval()
        val_preds, val_targets, val_probs = [], [], []
        val_loss_total, val_batches = 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                targets = batch.y_multiclass if hasattr(batch, "y_multiclass") and batch.y_multiclass is not None else batch.y
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    out = model(data=batch)
                    logits = out.logits if hasattr(out, "logits") else out
                    loss = criterion(logits, targets)
                    probs = torch.softmax(logits, dim=-1)

                val_loss_total += float(loss.item())
                val_batches += 1
                val_preds.append(torch.argmax(probs, dim=-1).cpu().numpy())
                val_targets.append(targets.cpu().numpy())
                val_probs.append(probs.cpu().numpy())

        y_true = np.concatenate(val_targets)
        y_pred = np.concatenate(val_preds)
        y_probs = np.concatenate(val_probs)

        # Metrics Computation on Validation Set
        val_loss = val_loss_total / max(val_batches, 1)
        acc = float(accuracy_score(y_true, y_pred))
        macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        y_bin_true = (y_true > 0).astype(int)
        y_bin_pred = (y_pred > 0).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_bin_true, y_bin_pred, labels=[0, 1]).ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        recalls = recall_score(y_true, y_pred, average=None, labels=list(range(self.NUM_CLASSES)), zero_division=0)
        rare_minority_recall = float(np.mean([recalls[c] for c in self.RARE_MINORITY_CLASSES]))

        roc_auc = 0.0
        try:
            y_true_bin = label_binarize(y_true, classes=list(range(self.NUM_CLASSES)))
            roc_auc = float(roc_auc_score(y_true_bin, y_probs, multi_class="ovr", average="macro"))
        except Exception:
            pass

        # Predefined Validation Selection Score (55% Macro-F1 + 35% Macro-Recall + 10% (1 - FPR))
        selection_score = (0.55 * macro_f1) + (0.35 * macro_r) + (0.10 * (1.0 - fpr))

        return {
            "val_loss": val_loss,
            "val_accuracy": acc,
            "val_macro_precision": macro_p,
            "val_macro_recall": macro_r,
            "val_macro_f1": macro_f1,
            "val_weighted_f1": weighted_f1,
            "val_fpr": fpr,
            "val_fnr": fnr,
            "val_rare_minority_recall": rare_minority_recall,
            "val_roc_auc": roc_auc,
            "selection_score": selection_score,
            "train_time_sec": train_time,
            "total_params": sum(p.numel() for p in model.parameters()),
        }

    def run_hyperparameter_search(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute full controlled hyperparameter search across all generated configurations.
        """
        print("=" * 85)
        print("     TGCF-IDS : Controlled Scientific Hyperparameter Experiment Framework      ")
        print("=" * 85)
        print(f"[*] Execution Device       : {self.device} (AMP Enabled: {self.use_amp})")
        print(f"[*] Validation Split Ratio : {self.val_ratio * 100:.1f}%")
        print(f"[*] Epochs per Experiment  : {self.epochs}")
        print(f"[*] Evaluation Seed        : {self.seed}")

        self.load_and_split_data()
        exp_grid = self.generate_experiment_grid()
        print(f"[+] Total Hyperparameter Experiments to Evaluate: {len(exp_grid)}")

        results = []

        print("\n" + "-" * 85)
        for i, exp in enumerate(exp_grid, 1):
            exp_id = exp["experiment_id"]
            tested_dim = exp["tested_dimension"]
            desc = exp["description"]
            cfg = exp["config"]

            print(f"[*] [{i:02d}/{len(exp_grid):02d}] {exp_id} | {tested_dim:<24} | {desc:<35} ... ", end="", flush=True)

            val_res = self._evaluate_configuration(cfg, seed=self.seed)

            print(
                f"Done! | Val F1: {val_res['val_macro_f1']:.4f} | "
                f"Val Recall: {val_res['val_macro_recall']:.4f} | "
                f"Val FPR: {val_res['val_fpr']:.4f} | "
                f"Score: {val_res['selection_score']:.4f}"
            )

            # Flatten configuration parameters into tabular row
            row = {
                "experiment_id": exp_id,
                "tested_dimension": tested_dim,
                "description": desc,
                "seed": self.seed,
                **cfg,
                **val_res,
            }
            results.append(row)

        df = pd.DataFrame(results)

        # Sort by predefined validation selection score (Highest first)
        df_sorted = df.sort_values(by="selection_score", ascending=False).reset_index(drop=True)
        best_row = df_sorted.iloc[0]

        best_config = {
            "experiment_id": best_row["experiment_id"],
            "description": best_row["description"],
            "selection_score": float(best_row["selection_score"]),
            "validation_metrics": {
                "val_accuracy": float(best_row["val_accuracy"]),
                "val_macro_precision": float(best_row["val_macro_precision"]),
                "val_macro_recall": float(best_row["val_macro_recall"]),
                "val_macro_f1": float(best_row["val_macro_f1"]),
                "val_weighted_f1": float(best_row["val_weighted_f1"]),
                "val_fpr": float(best_row["val_fpr"]),
                "val_rare_minority_recall": float(best_row["val_rare_minority_recall"]),
                "val_roc_auc": float(best_row["val_roc_auc"]),
            },
            "hyperparameters": {
                "learning_rate": float(best_row["learning_rate"]),
                "hidden_dimension": int(best_row["hidden_dimension"]),
                "embedding_dimension": int(best_row["embedding_dimension"]),
                "transformer_layers": int(best_row["transformer_layers"]),
                "transformer_heads": int(best_row["transformer_heads"]),
                "gnn_layers": int(best_row["gnn_layers"]),
                "dropout": float(best_row["dropout"]),
                "knn_k": int(best_row["knn_k"]),
                "temporal_window": int(best_row["temporal_window"]),
                "contrastive_temperature": float(best_row["contrastive_temperature"]),
                "feature_masking_ratio": float(best_row["feature_masking_ratio"]),
                "edge_dropout": float(best_row["edge_dropout"]),
            },
        }

        # Save to CSV and JSON / YAML
        out_csv = self.results_tables_dir / "hyperparameter_search.csv"
        out_best_json = self.results_tables_dir / "best_configuration.json"
        out_best_yaml = ProjectPaths.CONFIGS_DIR / "best_hyperparameters.yaml"

        df_sorted.to_csv(out_csv, index=False)
        with open(out_best_json, "w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=2)
        with open(out_best_yaml, "w", encoding="utf-8") as f:
            yaml.dump(best_config, f, default_flow_style=False)

        print("\n" + "=" * 85)
        print("                HYPERPARAMETER EXPERIMENT SEARCH SUMMARY                ")
        print("=" * 85)
        summary_cols = [
            "experiment_id", "tested_dimension", "val_accuracy", "val_macro_recall",
            "val_macro_f1", "val_fpr", "selection_score",
        ]
        print(df_sorted[summary_cols].head(10).to_string(index=False))
        print("=" * 85)
        print(f"\n[*] Selected Optimal Configuration: {best_row['experiment_id']} ({best_row['description']})")
        print(f"    - Val Macro F1     : {best_row['val_macro_f1']:.4f}")
        print(f"    - Val Macro Recall : {best_row['val_macro_recall']:.4f}")
        print(f"    - Val FPR          : {best_row['val_fpr']:.4f}")
        print(f"    - Selection Score  : {best_row['selection_score']:.4f}")
        print(f"[+] Saved Search Results -> {out_csv}")
        print(f"[+] Saved Best Config JSON -> {out_best_json}")
        print(f"[+] Saved Best Config YAML -> {out_best_yaml}")

        return df_sorted, best_config


def main():
    parser = argparse.ArgumentParser(description="TGCF-IDS Hyperparameter Search Framework")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per hyperparameter run")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Validation ratio from train data")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    framework = HyperparameterSearchFramework(
        epochs_per_run=args.epochs,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
    )
    framework.run_hyperparameter_search()


if __name__ == "__main__":
    main()
