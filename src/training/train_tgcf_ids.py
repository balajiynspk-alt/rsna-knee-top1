#!/usr/bin/env python3
"""
TGCF-IDS: Supervised End-to-End Training and Validation Pipeline.

Stages:
STAGE 1: Load self-supervised contrastively pretrained Temporal GraphSAGE encoder.
STAGE 2: Jointly train FeatureTokenizer + FeatureTransformer + TemporalGraphSAGE + CrossModalFusion + IntrusionClassifier.

Features:
- AdamW optimizer with CosineAnnealingLR scheduler
- Gradient clipping (max_norm=1.0)
- Automatic Mixed Precision (AMP) on CUDA
- Validation-driven early stopping and checkpointing
- Smoothed class-weighted CrossEntropyLoss
- Comprehensive tracking: Train/Val loss, Accuracy, Macro/Weighted F1, Precision, Recall, FPR, FNR, Gating statistics
- Strict isolation: Test data is NEVER used during training or model selection.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import yaml

# Ensure project root is accessible
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.tgcf_ids import TGCFIDS, TGCFIDSOutput
from src.utils.paths import ProjectPaths


class TGCFIDSTrainer:
    """
    End-to-End Supervised Trainer for TGCF-IDS.
    """

    CLASS_NAMES = [
        "Normal", "Analysis", "Backdoor", "DoS", "Exploits",
        "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
    ]
    NUM_CLASSES = 10

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        graphs_path: Optional[Union[str, Path]] = None,
        pretrained_checkpoint: Optional[Union[str, Path]] = None,
        load_pretrained: Optional[bool] = None,
        val_ratio: float = 0.15,
        token_dim: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        graph_hidden_dim: int = 64,
        graph_layers: int = 2,
        fusion_dim: int = 128,
        fusion_strategy: str = "gated",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip: float = 1.0,
        batch_size: int = 4,
        epochs: int = 35,
        early_stopping_patience: int = 10,
        use_amp: bool = True,
        use_class_weights: bool = True,
        freeze_pretrained: bool = False,
        seed: int = 42,
        device: Optional[str] = None,
        results_dir: Optional[Path] = None,
    ):
        self.seed = seed
        self._set_seed(self.seed)

        # Load YAML config if supplied
        self.config: Dict[str, Any] = {}
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

        # Merge config parameters with explicit arguments
        data_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})
        pretrain_cfg = self.config.get("pretrained", {})
        train_cfg = self.config.get("training", {})

        self.graphs_path = Path(graphs_path or data_cfg.get("graphs_path", ProjectPaths.DATA_GRAPHS / "train_graphs.pt"))
        self.pretrained_ckpt = pretrained_checkpoint or pretrain_cfg.get("checkpoint_path", ProjectPaths.RESULTS_CHECKPOINTS / "graph_pretrained.pt")
        self.load_pretrained = load_pretrained if load_pretrained is not None else pretrain_cfg.get("load_graph_encoder", True)
        self.freeze_pretrained = freeze_pretrained or pretrain_cfg.get("freeze_graph_encoder", False)

        self.val_ratio = data_cfg.get("val_ratio", val_ratio)
        self.token_dim = model_cfg.get("token_dim", token_dim)
        self.transformer_heads = model_cfg.get("transformer_heads", transformer_heads)
        self.transformer_layers = model_cfg.get("transformer_layers", transformer_layers)
        self.graph_hidden_dim = model_cfg.get("graph_hidden_dim", graph_hidden_dim)
        self.graph_layers = model_cfg.get("graph_layers", graph_layers)
        self.fusion_dim = model_cfg.get("fusion_dim", fusion_dim)
        self.fusion_strategy = model_cfg.get("fusion_strategy", fusion_strategy)

        self.lr = train_cfg.get("learning_rate", learning_rate)
        self.weight_decay = train_cfg.get("weight_decay", weight_decay)
        self.gradient_clip = train_cfg.get("gradient_clip", gradient_clip)
        self.batch_size = train_cfg.get("batch_size", batch_size)
        self.epochs = train_cfg.get("epochs", epochs)
        self.patience = train_cfg.get("early_stopping_patience", early_stopping_patience)
        self.use_class_weights = train_cfg.get("use_class_weights", use_class_weights)

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_amp = (use_amp and train_cfg.get("use_amp", True)) and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Results Directories
        if results_dir:
            base_res = Path(results_dir)
            self.checkpoints_dir = base_res / "checkpoints"
            self.figures_dir = base_res / "figures"
            self.tables_dir = base_res / "tables"
        else:
            self.checkpoints_dir = ProjectPaths.RESULTS_CHECKPOINTS
            self.figures_dir = ProjectPaths.RESULTS_FIGURES
            self.tables_dir = ProjectPaths.RESULTS_TABLES

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        # Build Model
        self.model = TGCFIDS(
            num_numerical=39,
            cat_cardinalities=[134, 14, 10],
            token_dim=self.token_dim,
            transformer_heads=self.transformer_heads,
            transformer_layers=self.transformer_layers,
            graph_in_channels=194,
            graph_edge_dim=6,
            graph_hidden_dim=self.graph_hidden_dim,
            graph_out_channels=self.graph_hidden_dim,
            graph_layers=self.graph_layers,
            fusion_dim=self.fusion_dim,
            fusion_strategy=self.fusion_strategy,
            classifier_hidden_dim=self.fusion_dim,
            num_classes=self.NUM_CLASSES,
        ).to(self.device)

        # STAGE 1: Load Pretrained Graph Encoder if configured
        if self.load_pretrained and Path(self.pretrained_ckpt).exists():
            try:
                self.model.load_pretrained_graph_encoder(
                    self.pretrained_ckpt,
                    freeze=self.freeze_pretrained,
                )
            except Exception as e:
                print(f"[!] Warning: Could not load pretrained graph encoder from {self.pretrained_ckpt}: {e}")

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-5,
        )

        self.criterion: Optional[nn.Module] = None
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_precision": [],
            "val_macro_recall": [],
            "val_macro_f1": [],
            "val_weighted_f1": [],
            "val_binary_fpr": [],
            "val_binary_fnr": [],
            "mean_gate_val": [],
            "lr": [],
        }

    def _set_seed(self, seed: int) -> None:
        """Enforce determinism."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _compute_class_weights(self, train_graphs: List[Data]) -> torch.Tensor:
        """Compute smoothed inverse square-root class weights strictly on training partition."""
        all_labels = []
        for g in train_graphs:
            all_labels.append(g.y_multiclass.cpu().numpy())
        y_train = np.concatenate(all_labels)

        class_counts = np.bincount(y_train, minlength=self.NUM_CLASSES)
        class_counts = np.maximum(class_counts, 1)

        raw_weights = 1.0 / np.sqrt(class_counts)
        norm_weights = raw_weights / raw_weights.sum() * self.NUM_CLASSES
        return torch.tensor(norm_weights, dtype=torch.float32, device=self.device)

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train one epoch across all training graph snapshots."""
        self.model.train()
        total_loss = 0.0
        gate_vals_sum = 0.0
        total_nodes = 0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.model(data=batch)
                loss = self.criterion(out.logits, batch.y_multiclass)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
                self.optimizer.step()

            total_loss += loss.item()
            gate_vals_sum += out.gate_values.detach().mean().item() * batch.num_nodes
            total_nodes += batch.num_nodes
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_gate = gate_vals_sum / max(total_nodes, 1)
        return avg_loss, avg_gate

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance across graph snapshots."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_binary_targets = []
        gate_vals_list = []
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.model(data=batch)
                loss = self.criterion(out.logits, batch.y_multiclass)

            total_loss += loss.item()
            preds = torch.argmax(out.logits, dim=-1).cpu().numpy()
            targets = batch.y_multiclass.cpu().numpy()
            bin_targets = batch.y_binary.cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)
            all_binary_targets.append(bin_targets)
            gate_vals_list.append(out.gate_values.cpu().numpy())
            num_batches += 1

        val_loss = total_loss / max(num_batches, 1)
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        y_bin_true = np.concatenate(all_binary_targets)
        y_bin_pred = (y_pred > 0).astype(np.int64)

        # Multiclass metrics
        acc = accuracy_score(y_true, y_pred)
        macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Binary confusion matrix for FPR / FNR
        bin_cm = confusion_matrix(y_bin_true, y_bin_pred, labels=[0, 1])
        tn, fp, fn, tp = bin_cm.ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        all_gates = np.vstack(gate_vals_list) if len(gate_vals_list) > 0 else np.array([0.5])
        mean_gate = float(np.mean(all_gates))

        return {
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(acc, 4),
            "val_macro_precision": round(macro_p, 4),
            "val_macro_recall": round(macro_r, 4),
            "val_macro_f1": round(macro_f1, 4),
            "val_weighted_f1": round(weighted_f1, 4),
            "val_binary_fpr": round(fpr, 4),
            "val_binary_fnr": round(fnr, 4),
            "mean_gate_val": round(mean_gate, 4),
        }

    def train_pipeline(
        self,
        max_train_windows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute full training pipeline, validation tracking, checkpointing, and curves generation."""
        if not self.graphs_path.exists():
            raise FileNotFoundError(f"Training graph snapshots missing at: {self.graphs_path}")

        print("=" * 75)
        print("         TGCF-IDS : Supervised End-to-End Model Training Pipeline        ")
        print("=" * 75)
        print(f"[*] Execution Device     : {self.device} (AMP Enabled: {self.use_amp})")
        print(f"[*] Hyperparameters      : LR={self.lr}, Epochs={self.epochs}, BatchSize={self.batch_size}, Clip={self.gradient_clip}")
        print(f"[*] Model Architecture   : TokenDim={self.token_dim}, Heads={self.transformer_heads}, GNN_Hidden={self.graph_hidden_dim}, FusionDim={self.fusion_dim} ({self.fusion_strategy})")

        # Load Graph Snapshots
        all_graphs: List[Data] = torch.load(self.graphs_path, weights_only=False)
        if max_train_windows:
            all_graphs = all_graphs[:max_train_windows]

        # Partition into Train vs Validation snapshots (strict temporal ordering or deterministic split)
        n_total = len(all_graphs)
        n_val = max(1, int(n_total * self.val_ratio))
        n_train = n_total - n_val

        # Temporal split: earlier snapshots for training, subsequent snapshots for validation model selection
        train_graphs = all_graphs[:n_train]
        val_graphs = all_graphs[n_train:]

        print(f"[+] Loaded {n_total} Total Graph Snapshots:")
        print(f"    - Training Partitions   : {len(train_graphs)} snapshots ({len(train_graphs)*1000:,} flows approx)")
        print(f"    - Validation Partitions : {len(val_graphs)} snapshots ({len(val_graphs)*1000:,} flows approx)")

        # Compute smoothed class weights on training split
        if self.use_class_weights:
            class_weights = self._compute_class_weights(train_graphs)
            print(f"[+] Computed Class Weights on Train Split: {np.round(class_weights.cpu().numpy(), 3).tolist()}")
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_graphs, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=self.batch_size, shuffle=False)

        print("-" * 75)
        print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Val Loss':^10} | {'Val Acc':^8} | {'Val MacroF1':^11} | {'Val W-F1':^8} | {'FPR':^7} | {'Gate':^6} | {'Status':^7}")
        print("-" * 75)

        best_macro_f1 = -1.0
        best_epoch = 0
        best_val_metrics = {}
        best_checkpoint_path = self.checkpoints_dir / "best_tgcf_ids.pt"
        last_checkpoint_path = self.checkpoints_dir / "last_tgcf_ids.pt"
        patience_counter = 0

        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            train_loss, train_gate = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(round(train_loss, 4))
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_accuracy"].append(val_metrics["val_accuracy"])
            self.history["val_macro_precision"].append(val_metrics["val_macro_precision"])
            self.history["val_macro_recall"].append(val_metrics["val_macro_recall"])
            self.history["val_macro_f1"].append(val_metrics["val_macro_f1"])
            self.history["val_weighted_f1"].append(val_metrics["val_weighted_f1"])
            self.history["val_binary_fpr"].append(val_metrics["val_binary_fpr"])
            self.history["val_binary_fnr"].append(val_metrics["val_binary_fnr"])
            self.history["mean_gate_val"].append(val_metrics["mean_gate_val"])
            self.history["lr"].append(current_lr)

            # Checkpointing
            val_f1 = val_metrics["val_macro_f1"]
            is_best = val_f1 > best_macro_f1

            if is_best:
                best_macro_f1 = val_f1
                best_epoch = epoch
                best_val_metrics = val_metrics
                patience_counter = 0
                status_str = "[BEST]"

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_macro_f1": best_macro_f1,
                    "val_metrics": val_metrics,
                    "config": self.config,
                }, best_checkpoint_path)
            else:
                patience_counter += 1
                status_str = f"({patience_counter}/{self.patience})"

            # Always save last checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "val_macro_f1": val_f1,
            }, last_checkpoint_path)

            print(
                f"{epoch:^6} | {train_loss:^10.4f} | {val_metrics['val_loss']:^10.4f} | "
                f"{val_metrics['val_accuracy']:^8.4f} | {val_metrics['val_macro_f1']:^11.4f} | "
                f"{val_metrics['val_weighted_f1']:^8.4f} | {val_metrics['val_binary_fpr']:^7.4f} | "
                f"{val_metrics['mean_gate_val']:^6.2f} | {status_str:^7}"
            )

            # Early Stopping Check
            if patience_counter >= self.patience:
                print("-" * 75)
                print(f"[!] Early stopping triggered at epoch {epoch}. Restoring best checkpoint from epoch {best_epoch}.")
                break

        total_training_time = time.time() - t0
        print("-" * 75)
        print(f"[+] Supervised Training Completed in {total_training_time:.2f} seconds.")
        print(f"[+] Best Validation Macro F1: {best_macro_f1:.4f} at Epoch {best_epoch}")

        # Restore best weights
        if best_checkpoint_path.exists():
            best_ckpt = torch.load(best_checkpoint_path, weights_only=False)
            self.model.load_state_dict(best_ckpt["model_state_dict"])

        # Save History Table & Metrics JSON
        history_df = pd.DataFrame(self.history)
        history_csv_path = self.tables_dir / "tgcf_ids_training_history.csv"
        history_df.to_csv(history_csv_path, index=False)
        print(f"[+] Saved Training History Table -> {history_csv_path}")

        summary_metrics = {
            "model": "TGCF-IDS (Dual-Branch Feature-Transformer + Temporal GraphSAGE)",
            "fusion_strategy": self.fusion_strategy,
            "best_epoch": best_epoch,
            "training_time_seconds": round(total_training_time, 2),
            "best_val_macro_f1": best_macro_f1,
            "val_metrics": best_val_metrics,
        }
        metrics_json_path = self.tables_dir / "tgcf_ids_val_metrics.json"
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2)
        print(f"[+] Saved Validation Metrics JSON -> {metrics_json_path}")

        # Plot Training Curves
        self._plot_training_curves()

        print("=" * 75)
        return {
            "model": self.model,
            "best_epoch": best_epoch,
            "best_val_metrics": best_val_metrics,
            "best_checkpoint_path": best_checkpoint_path,
            "history": self.history,
        }

    def _plot_training_curves(self) -> None:
        """Plot and save multi-panel training progression curves."""
        plt.figure(figsize=(14, 10), dpi=200)
        epochs = self.history["epoch"]

        # 1. Loss Curves
        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.history["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2.0)
        plt.plot(epochs, self.history["val_loss"], label="Val Loss", color="#2980b9", linewidth=2.0, linestyle="--")
        plt.title("Cross-Entropy Loss Progression", fontsize=11, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)

        # 2. Macro F1 & Accuracy
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.history["val_macro_f1"], label="Val Macro F1", color="#27ae60", linewidth=2.0)
        plt.plot(epochs, self.history["val_accuracy"], label="Val Accuracy", color="#8e44ad", linewidth=1.8, linestyle="--")
        plt.plot(epochs, self.history["val_weighted_f1"], label="Val Weighted F1", color="#f39c12", linewidth=1.5, linestyle=":")
        plt.title("Validation Performance Metrics", fontsize=11, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)

        # 3. Binary Error Rates (FPR & FNR)
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.history["val_binary_fpr"], label="False Alarm Rate (FPR)", color="#c0392b", linewidth=2.0)
        plt.plot(epochs, self.history["val_binary_fnr"], label="Miss Rate (FNR)", color="#d35400", linewidth=1.8, linestyle="--")
        plt.title("Binary Security Risk Metrics (FPR / FNR)", fontsize=11, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Rate")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)

        # 4. Learned Gate Dynamics
        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.history["mean_gate_val"], label="Mean Feature Gate Weight", color="#16a085", linewidth=2.0)
        plt.axhline(0.5, color="#7f8c8d", linestyle=":", label="Balanced Equilibrium (0.5)")
        plt.title("Cross-Modal Gating Dynamics (Feature vs Graph)", fontsize=11, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Gate Activation")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)

        plt.tight_layout()
        out_fig = self.figures_dir / "tgcf_ids_training_curves.png"
        plt.savefig(out_fig, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved Training Curves Plot -> {out_fig}")


def main():
    parser = argparse.ArgumentParser(description="TGCF-IDS: Supervised End-to-End Training")
    parser.add_argument("--config", type=str, default="configs/tgcf_ids_train.yaml", help="Path to YAML training config")
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs (default: 35)")
    parser.add_argument("--batch-size", type=int, default=4, help="Graph snapshots per batch (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--max-train-windows", type=int, default=None, help="Subset of train windows for verification")

    args = parser.parse_args()

    trainer = TGCFIDSTrainer(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        early_stopping_patience=args.patience,
    )
    trainer.train_pipeline(max_train_windows=args.max_train_windows)


if __name__ == "__main__":
    main()
