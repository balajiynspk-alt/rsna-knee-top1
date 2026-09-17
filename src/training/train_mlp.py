#!/usr/bin/env python3
"""
TGCF-IDS: PyTorch MLP Training and Evaluation Pipeline.
Features:
- Mixed precision training (AMP) on CUDA
- Class-weighted CrossEntropyLoss
- AdamW optimizer with CosineAnnealingLR scheduler
- Metric tracking (Loss, Accuracy, Macro Precision/Recall/F1, Weighted F1)
- Model checkpointing and Early Stopping
- Publication figures (Training curves, Confusion matrix) and CSV reports
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
    classification_report,
)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root is accessible
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.mlp import PyTorchMLP
from src.utils.paths import ProjectPaths


class MLPTrainer:
    """
    End-to-end trainer and evaluator for the PyTorch MLP baseline.
    """

    CLASS_NAMES = [
        "Normal", "Analysis", "Backdoor", "DoS", "Exploits",
        "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
    ]
    NUM_CLASSES = 10

    def __init__(
        self,
        in_features: int = 194,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        epochs: int = 50,
        early_stopping_patience: int = 12,
        val_split_ratio: float = 0.15,
        use_class_weights: bool = True,
        use_amp: bool = True,
        seed: int = 42,
        device: Optional[str] = None,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ):
        self.in_features = in_features
        self.hidden_dims = hidden_dims or [256, 128]
        self.dropout = dropout
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = early_stopping_patience
        self.val_split_ratio = val_split_ratio
        self.use_class_weights = use_class_weights
        self.seed = seed

        # Deterministic setup
        self._set_seed(self.seed)

        # Device detection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Mixed precision
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Paths
        self.data_dir = Path(data_dir) if data_dir else ProjectPaths.DATA_PROCESSED
        if results_dir:
            base_res = Path(results_dir)
            self.checkpoints_dir = base_res / "checkpoints"
            self.figures_dir = base_res / "figures" / "baselines"
            self.tables_dir = base_res / "tables"
        else:
            self.checkpoints_dir = ProjectPaths.RESULTS_CHECKPOINTS
            self.figures_dir = ProjectPaths.RESULTS_FIGURES / "baselines"
            self.tables_dir = ProjectPaths.RESULTS_TABLES

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        # Model instance
        self.model = PyTorchMLP(
            in_features=self.in_features,
            hidden_dims=self.hidden_dims,
            num_classes=self.NUM_CLASSES,
            dropout=self.dropout,
        ).to(self.device)

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
            "lr": [],
        }

    def _set_seed(self, seed: int) -> None:
        """Enforce determinism across PyTorch, NumPy, and random."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def prepare_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load preprocessed PyTorch tensors, split training data into train/val sets,
        compute class weights strictly from the training partition, and construct DataLoaders.
        """
        train_pt_path = self.data_dir / "train_features.pt"
        test_pt_path = self.data_dir / "test_features.pt"

        if not train_pt_path.exists() or not test_pt_path.exists():
            raise FileNotFoundError(
                f"Feature files not found in {self.data_dir}. Run preprocessing first."
            )

        print(f"[+] Loading PyTorch tensor bundles from: {self.data_dir}")
        train_tensors = torch.load(train_pt_path, weights_only=True)
        test_tensors = torch.load(test_pt_path, weights_only=True)

        X_all_train = train_tensors["x_dense"]
        y_all_train = train_tensors["y_multiclass"]

        X_test = test_tensors["x_dense"]
        y_test = test_tensors["y_multiclass"]

        # Stratified train/validation split derived strictly from official training data
        indices = np.arange(len(X_all_train))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=self.val_split_ratio,
            stratify=y_all_train.numpy(),
            random_state=self.seed,
        )

        X_train_split = X_all_train[train_idx]
        y_train_split = y_all_train[train_idx]

        X_val_split = X_all_train[val_idx]
        y_val_split = y_all_train[val_idx]

        print(f"[+] Data Partitions:")
        print(f"    - Training Split    : {len(X_train_split):,} samples")
        print(f"    - Validation Split  : {len(X_val_split):,} samples (for early stopping & tuning)")
        print(f"    - Final Test Split  : {len(X_test):,} samples (held-out for evaluation only)")

        # Compute smoothed class weights on training split: w_c = N / (K * N_c)
        if self.use_class_weights:
            class_counts = torch.bincount(y_train_split, minlength=self.NUM_CLASSES).float()
            total_train_samples = len(y_train_split)
            # Smooth inverse frequency weights with square root dampening to prevent gradient explosion on tiny classes
            weights = torch.sqrt(total_train_samples / (class_counts + 1e-5))
            weights = weights / weights.sum() * self.NUM_CLASSES
            weights = torch.clamp(weights, min=0.2, max=25.0)
            self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
            print(f"[+] Computed Class Weights (smoothed inverse freq): {weights.numpy().round(3).tolist()}")
        else:
            self.criterion = nn.CrossEntropyLoss()

        # DataLoaders
        train_ds = TensorDataset(X_train_split, y_train_split)
        val_ds = TensorDataset(X_val_split, y_val_split)
        test_ds = TensorDataset(X_test, y_test)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch with mixed precision."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            batch_size = batch_x.size(0)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / total_samples

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on validation or test DataLoader."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            batch_size = batch_x.size(0)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        acc = accuracy_score(y_true, y_pred)
        macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Per-class metrics
        per_class_p = precision_score(y_true, y_pred, average=None, labels=list(range(self.NUM_CLASSES)), zero_division=0)
        per_class_r = recall_score(y_true, y_pred, average=None, labels=list(range(self.NUM_CLASSES)), zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(self.NUM_CLASSES)), zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.NUM_CLASSES)))

        # Binary equivalent (0: Benign vs 1-9: Attack)
        y_true_bin = (y_true > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)
        cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        tn_b, fp_b, fn_b, tp_b = cm_bin.ravel()

        binary_fpr = fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0.0
        binary_fnr = fn_b / (fn_b + tp_b) if (fn_b + tp_b) > 0 else 0.0

        return {
            "loss": total_loss / total_samples,
            "accuracy": float(acc),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "binary_fpr": float(binary_fpr),
            "binary_fnr": float(binary_fnr),
            "per_class_precision": per_class_p.tolist(),
            "per_class_recall": per_class_r.tolist(),
            "per_class_f1": per_class_f1.tolist(),
            "confusion_matrix": cm,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def train_pipeline(self) -> Dict[str, Any]:
        """
        Execute full training loop with early stopping, learning rate scheduling,
        best checkpoint saving, and test set evaluation.
        """
        print("=" * 75)
        print("          TGCF-IDS : PyTorch Multi-Layer Perceptron (MLP) Training        ")
        print("=" * 75)
        print(f"[*] Hardware Execution Device : {self.device} (AMP Enabled: {self.use_amp})")
        print(f"[*] Model Architecture        : Input({self.in_features}) -> FC({self.hidden_dims[0]}) -> LN -> GELU -> Drop({self.dropout}) -> FC({self.hidden_dims[1]}) -> GELU -> Drop({self.dropout}) -> FC({self.NUM_CLASSES})")

        train_loader, val_loader, test_loader = self.prepare_data_loaders()

        best_val_macro_f1 = -1.0
        best_epoch = 0
        patience_counter = 0
        checkpoint_path = self.checkpoints_dir / "best_mlp.pt"

        start_time = time.time()
        print("-" * 75)
        print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Val Loss':^10} | {'Val Acc':^8} | {'Val MacroF1':^11} | {'Val W-F1':^8} | {'LR':^9} | {'Status':^8}")
        print("-" * 75)

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update learning rate
            self.scheduler.step()

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_macro_precision"].append(val_metrics["macro_precision"])
            self.history["val_macro_recall"].append(val_metrics["macro_recall"])
            self.history["val_macro_f1"].append(val_metrics["macro_f1"])
            self.history["val_weighted_f1"].append(val_metrics["weighted_f1"])
            self.history["lr"].append(current_lr)

            # Check early stopping criterion based on Validation Macro F1
            val_f1 = val_metrics["macro_f1"]
            is_best = val_f1 > best_val_macro_f1

            status_str = ""
            if is_best:
                best_val_macro_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                status_str = "[BEST]"

                # Save best checkpoint
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_macro_f1": best_val_macro_f1,
                    "val_metrics": val_metrics,
                    "config": {
                        "in_features": self.in_features,
                        "hidden_dims": self.hidden_dims,
                        "dropout": self.dropout,
                        "num_classes": self.NUM_CLASSES,
                    },
                }, checkpoint_path)
            else:
                patience_counter += 1
                status_str = f"({patience_counter}/{self.patience})"

            print(f"{epoch:^6d} | {train_loss:^10.4f} | {val_metrics['loss']:^10.4f} | {val_metrics['accuracy']:^8.4f} | {val_metrics['macro_f1']:^11.4f} | {val_metrics['weighted_f1']:^8.4f} | {current_lr:^9.2e} | {status_str:^8}")

            if patience_counter >= self.patience:
                print("-" * 75)
                print(f"[!] Early stopping triggered at epoch {epoch}. Restoring best model from epoch {best_epoch}.")
                break

        total_time = time.time() - start_time
        print("-" * 75)
        print(f"[+] Training Completed in {total_time:.2f} seconds.")
        print(f"[+] Best Validation Macro F1: {best_val_macro_f1:.4f} at Epoch {best_epoch}")

        # ---------------------------------------------------------------------
        # Final Test Evaluation (Load Best Checkpoint)
        # ---------------------------------------------------------------------
        print("\n" + "=" * 75)
        print("          EVALUATING BEST CHECKPOINT ON HELD-OUT TEST SPLIT           ")
        print("=" * 75)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = self.evaluate(test_loader)

        print(f"[+] Test Accuracy        : {test_metrics['accuracy']:.4f}")
        print(f"[+] Test Macro Precision : {test_metrics['macro_precision']:.4f}")
        print(f"[+] Test Macro Recall    : {test_metrics['macro_recall']:.4f}")
        print(f"[+] Test Macro F1        : {test_metrics['macro_f1']:.4f}")
        print(f"[+] Test Weighted F1     : {test_metrics['weighted_f1']:.4f}")
        print(f"[+] Test Binary FPR (FAR): {test_metrics['binary_fpr']:.4f}")
        print(f"[+] Test Binary FNR (Miss): {test_metrics['binary_fnr']:.4f}")

        # ---------------------------------------------------------------------
        # Generate Classification Report & Tables
        # ---------------------------------------------------------------------
        report_rows = []
        for c_idx, c_name in enumerate(self.CLASS_NAMES):
            report_rows.append({
                "Class ID": c_idx,
                "Class Name": c_name,
                "Precision": round(test_metrics["per_class_precision"][c_idx], 4),
                "Recall": round(test_metrics["per_class_recall"][c_idx], 4),
                "F1 Score": round(test_metrics["per_class_f1"][c_idx], 4),
            })

        class_report_df = pd.DataFrame(report_rows)
        class_report_path = self.tables_dir / "mlp_classification_report.csv"
        class_report_df.to_csv(class_report_path, index=False)
        print(f"\n[+] Saved MLP Classification Report -> {class_report_path}")

        # Save summary JSON
        metrics_summary = {
            "model": "PyTorch MLP",
            "in_features": self.in_features,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "best_epoch": best_epoch,
            "training_time_seconds": round(total_time, 2),
            "test_accuracy": round(test_metrics["accuracy"], 4),
            "test_macro_precision": round(test_metrics["macro_precision"], 4),
            "test_macro_recall": round(test_metrics["macro_recall"], 4),
            "test_macro_f1": round(test_metrics["macro_f1"], 4),
            "test_weighted_f1": round(test_metrics["weighted_f1"], 4),
            "test_binary_fpr": round(test_metrics["binary_fpr"], 4),
            "test_binary_fnr": round(test_metrics["binary_fnr"], 4),
        }
        json_path = self.tables_dir / "mlp_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"[+] Saved MLP Summary Metrics       -> {json_path}")

        # ---------------------------------------------------------------------
        # Generate Figures (Training Curves & Confusion Matrix)
        # ---------------------------------------------------------------------
        curves_path = self.plot_training_curves()
        cm_path = self.plot_confusion_matrix(test_metrics["confusion_matrix"])

        print("=" * 75)
        print("MLP Baseline Training and Evaluation Pipeline Completed.")
        print("=" * 75)

        return {
            "model": self.model,
            "test_metrics": test_metrics,
            "class_report_df": class_report_df,
            "history": self.history,
            "checkpoint_path": checkpoint_path,
            "curves_path": curves_path,
            "cm_path": cm_path,
        }

    def plot_training_curves(self) -> Path:
        """Plot loss, accuracy, and macro F1 curves across training epochs."""
        epochs = self.history["epoch"]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

        # 1. Loss Curves
        ax1.plot(epochs, self.history["train_loss"], "o-", color="#e74c3c", label="Train Loss", lw=2, ms=4)
        ax1.plot(epochs, self.history["val_loss"], "s-", color="#3498db", label="Val Loss", lw=2, ms=4)
        ax1.set_title("CrossEntropy Loss Progression", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.set_ylabel("Loss", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # 2. Accuracy
        ax2.plot(epochs, self.history["val_accuracy"], "^-", color="#2ecc71", label="Val Accuracy", lw=2, ms=4)
        ax2.set_title("Validation Accuracy", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("Accuracy", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        # 3. Macro & Weighted F1
        ax3.plot(epochs, self.history["val_macro_f1"], "d-", color="#9b59b6", label="Val Macro F1", lw=2, ms=4)
        ax3.plot(epochs, self.history["val_weighted_f1"], "v-", color="#f39c12", label="Val Weighted F1", lw=2, ms=4)
        ax3.set_title("Validation F1 Score Dynamics", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Epoch", fontsize=10)
        ax3.set_ylabel("F1 Score", fontsize=10)
        ax3.grid(True, linestyle="--", alpha=0.6)
        ax3.legend()

        plt.suptitle("PyTorch MLP Baseline: Training Dynamics on UNSW-NB15", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        out_path = self.figures_dir / "mlp_training_curves.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[+] Saved Training Curves Plot     -> {out_path}")
        return out_path

    def plot_confusion_matrix(self, cm: np.ndarray) -> Path:
        """Plot normalized multi-class confusion matrix."""
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

        fig, ax = plt.subplots(figsize=(10, 8.5), dpi=300)
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Purples",
            xticklabels=self.CLASS_NAMES,
            yticklabels=self.CLASS_NAMES,
            cbar_kws={"label": "Normalized Recall (Per Class)"},
            linewidths=0.5,
            ax=ax,
        )

        ax.set_title("Confusion Matrix: PyTorch MLP Baseline (Held-out Test Split)", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("True Ground-Truth Class", fontsize=11)
        plt.xticks(rotation=40, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        out_path = self.figures_dir / "mlp_confusion_matrix.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[+] Saved Confusion Matrix Plot    -> {out_path}")
        return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate PyTorch MLP baseline on TGCF-IDS preprocessed features."
    )
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128], help="Hidden layer dimensions.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=40, help="Maximum epochs.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    trainer = MLPTrainer(
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        seed=args.seed,
    )
    trainer.train_pipeline()


if __name__ == "__main__":
    main()
