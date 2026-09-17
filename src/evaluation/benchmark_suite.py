#!/usr/bin/env python3
"""
TGCF-IDS: Controlled Baseline Benchmark Framework.

Evaluates 9 model architectures under an identical test split across multiple random seeds:
1. Random Forest (RF)
2. XGBoost
3. MLP (3-layer PyTorch)
4. BiLSTM
5. Transformer-only
6. GraphSAGE-only
7. GraphSAGE + Transformer (Concat)
8. SAGEConv + Transformer + Gated (No Pretrain)
9. TGCF-IDS (Contrastive Pretrained + Gated Fusion)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.paths import ProjectPaths
from src.models.mlp import PyTorchMLP
from src.models.bilstm import BiLSTMClassifier
from src.models.ablation_models import (
    TransformerOnlyClassifier,
    GraphSAGEOnlyClassifier,
)
from src.models.tgcf_ids import TGCFIDS


class ControlledBenchmarkFramework:
    """
    Unified Controlled Benchmarking Framework for TGCF-IDS and 8 Baseline Models.
    """

    CLASS_NAMES = [
        "Normal", "Analysis", "Backdoor", "DoS", "Exploits",
        "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
    ]
    NUM_CLASSES = 10

    MODEL_ORDER = [
        "Random Forest",
        "XGBoost",
        "MLP",
        "BiLSTM",
        "Transformer-only",
        "GraphSAGE-only",
        "GraphSAGE + Transformer (Concat)",
        "SAGEConv + Transformer + Gated (No Pretrain)",
        "TGCF-IDS (Proposed)",
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

        # Cache for loaded datasets
        self.X_train_dense: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test_dense: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        self.train_graphs: Optional[List[Data]] = None
        self.test_graphs: Optional[List[Data]] = None

    def _set_seed(self, seed: int) -> None:
        """Enforce deterministic reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_all_data(self) -> None:
        """Load tabular npz matrices and graph snapshot archives."""
        train_npz = self.data_dir / "train_features.npz"
        test_npz = self.data_dir / "test_features.npz"

        if not train_npz.exists() or not test_npz.exists():
            raise FileNotFoundError(f"Processed npz files missing in {self.data_dir}")

        train_data = np.load(train_npz)
        test_data = np.load(test_npz)

        self.X_train_dense = train_data["x_dense"]
        self.y_train = train_data["y_multiclass"]
        self.X_test_dense = test_data["x_dense"]
        self.y_test = test_data["y_multiclass"]

        train_g_file = self.graphs_dir / "train_graphs.pt"
        test_g_file = self.graphs_dir / "test_graphs.pt"

        if train_g_file.exists() and test_g_file.exists():
            print(f"[+] Loading graph archives: {train_g_file.name}, {test_g_file.name}")
            self.train_graphs = torch.load(train_g_file, weights_only=False)
            self.test_graphs = torch.load(test_g_file, weights_only=False)
        else:
            print("[!] Warning: Graph archives not found on disk; graph models will be skipped or simulated.")

        print(f"[+] Tabular Data Loaded: X_train={self.X_train_dense.shape}, X_test={self.X_test_dense.shape}")
        if self.train_graphs:
            print(f"[+] Graph Snapshots Loaded: Train={len(self.train_graphs)}, Test={len(self.test_graphs)}")

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute multi-class, per-class, binary security, ROC-AUC, and PR-AUC metrics.
        """
        acc = accuracy_score(y_true, y_pred)
        macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        per_p = precision_score(y_true, y_pred, average=None, labels=list(range(self.NUM_CLASSES)), zero_division=0)
        per_r = recall_score(y_true, y_pred, average=None, labels=list(range(self.NUM_CLASSES)), zero_division=0)
        per_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(self.NUM_CLASSES)), zero_division=0)

        # Binary confusion matrix (Class 0: Benign vs Classes 1-9: Attack)
        y_true_bin = (y_true > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)
        cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        tn_b, fp_b, fn_b, tp_b = cm_bin.ravel()

        fpr = float(fp_b / (fp_b + tn_b)) if (fp_b + tn_b) > 0 else 0.0
        fnr = float(fn_b / (fn_b + tp_b)) if (fn_b + tp_b) > 0 else 0.0

        # ROC-AUC and PR-AUC (One-vs-Rest)
        roc_auc_macro = 0.0
        pr_auc_macro = 0.0

        if y_proba is not None:
            try:
                y_bin_onehot = label_binarize(y_true, classes=list(range(self.NUM_CLASSES)))
                roc_auc_macro = float(roc_auc_score(y_bin_onehot, y_proba, multi_class="ovr", average="macro"))
                pr_auc_macro = float(average_precision_score(y_bin_onehot, y_proba, average="macro"))
            except Exception:
                roc_auc_macro = 0.0
                pr_auc_macro = 0.0

        return {
            "accuracy": float(acc),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "binary_fpr": float(fpr),
            "binary_fnr": float(fnr),
            "roc_auc_macro": float(roc_auc_macro),
            "pr_auc_macro": float(pr_auc_macro),
            "per_class_precision": per_p.tolist(),
            "per_class_recall": per_r.tolist(),
            "per_class_f1": per_f1.tolist(),
        }

    def _get_class_weights_from_train(self) -> torch.Tensor:
        """Compute inverse square-root class weights from training labels."""
        counts = np.bincount(self.y_train, minlength=self.NUM_CLASSES)
        counts = np.maximum(counts, 1)
        raw = 1.0 / np.sqrt(counts)
        norm = raw / raw.sum() * self.NUM_CLASSES
        return torch.tensor(norm, dtype=torch.float32, device=self.device)

    def count_parameters(self, model: nn.Module) -> Tuple[int, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    # -------------------------------------------------------------------------
    # Model 1 & 2: Traditional Tree Baselines
    # -------------------------------------------------------------------------
    def evaluate_random_forest(self, seed: int) -> Dict[str, Any]:
        """Evaluate Random Forest."""
        self._set_seed(seed)
        model = RandomForestClassifier(n_estimators=100, max_depth=25, min_samples_split=5, random_state=seed, n_jobs=-1)

        t0 = time.time()
        model.fit(self.X_train_dense, self.y_train)
        train_time = time.time() - t0

        t_inf_start = time.time()
        y_pred = model.predict(self.X_test_dense)
        y_proba = model.predict_proba(self.X_test_dense)
        inf_time_total = time.time() - t_inf_start
        inf_ms_per_1k = (inf_time_total / len(self.X_test_dense)) * 1000.0 * 1000.0

        metrics = self.compute_metrics(self.y_test, y_pred, y_proba)
        metrics.update({
            "model": "Random Forest",
            "seed": seed,
            "training_time_s": train_time,
            "inference_ms_per_1k": inf_ms_per_1k,
            "total_params": 0,
            "trainable_params": 0,
        })
        return metrics

    def evaluate_xgboost(self, seed: int) -> Dict[str, Any]:
        """Evaluate XGBoost."""
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed.")
        self._set_seed(seed)
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="mlogloss",
        )

        t0 = time.time()
        model.fit(self.X_train_dense, self.y_train)
        train_time = time.time() - t0

        t_inf_start = time.time()
        y_pred = model.predict(self.X_test_dense)
        y_proba = model.predict_proba(self.X_test_dense)
        inf_time_total = time.time() - t_inf_start
        inf_ms_per_1k = (inf_time_total / len(self.X_test_dense)) * 1000.0 * 1000.0

        metrics = self.compute_metrics(self.y_test, y_pred, y_proba)
        metrics.update({
            "model": "XGBoost",
            "seed": seed,
            "training_time_s": train_time,
            "inference_ms_per_1k": inf_ms_per_1k,
            "total_params": 0,
            "trainable_params": 0,
        })
        return metrics

    # -------------------------------------------------------------------------
    # Deep Learning Model Runners (PyG Graph & Tabular)
    # -------------------------------------------------------------------------
    def _train_and_eval_nn_graph(
        self,
        model: nn.Module,
        model_name: str,
        seed: int,
        epochs: int = 15,
        lr: float = 1e-3,
    ) -> Dict[str, Any]:
        """
        Generic training and evaluation engine for neural network architectures operating over graph/flow batches.
        """
        self._set_seed(seed)
        model = model.to(self.device)
        total_p, train_p = self.count_parameters(model)

        class_weights = self._get_class_weights_from_train()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        train_loader = DataLoader(self.train_graphs, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_graphs, batch_size=self.batch_size, shuffle=False)

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    out = model(data=batch)
                    logits = out.logits if hasattr(out, "logits") else out
                    loss = criterion(logits, batch.y_multiclass)

                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            scheduler.step()

        train_time = time.time() - t0

        # Inference on full test split
        model.eval()
        all_preds = []
        all_probas = []
        all_targets = []
        total_inf_nodes = 0

        t_inf_start = time.time()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    out = model(data=batch)
                    logits = out.logits if hasattr(out, "logits") else out
                    probs = torch.softmax(logits, dim=-1)

                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.append(preds)
                all_probas.append(probs.cpu().numpy())
                all_targets.append(batch.y_multiclass.cpu().numpy())
                total_inf_nodes += batch.num_nodes

        inf_time_total = time.time() - t_inf_start
        inf_ms_per_1k = (inf_time_total / max(total_inf_nodes, 1)) * 1000.0 * 1000.0

        y_pred = np.concatenate(all_preds)
        y_proba = np.vstack(all_probas)
        y_true = np.concatenate(all_targets)

        metrics = self.compute_metrics(y_true, y_pred, y_proba)
        metrics.update({
            "model": model_name,
            "seed": seed,
            "training_time_s": train_time,
            "inference_ms_per_1k": inf_ms_per_1k,
            "total_params": total_p,
            "trainable_params": train_p,
        })
        return metrics

    def evaluate_mlp(self, seed: int) -> Dict[str, Any]:
        """Model 3: 3-Layer MLP Baseline."""
        self._set_seed(seed)
        model = PyTorchMLP(in_features=194, hidden_dims=[256, 128], num_classes=self.NUM_CLASSES, dropout=0.2)
        return self._train_and_eval_nn_graph(model, "MLP", seed, epochs=self.epochs, lr=self.lr)

    def evaluate_bilstm(self, seed: int) -> Dict[str, Any]:
        """Model 4: BiLSTM Baseline."""
        self._set_seed(seed)
        model = BiLSTMClassifier(in_features=194, hidden_dim=64, num_layers=2, num_classes=self.NUM_CLASSES, dropout=0.2)
        return self._train_and_eval_nn_graph(model, "BiLSTM", seed, epochs=self.epochs, lr=self.lr)

    def evaluate_transformer_only(self, seed: int) -> Dict[str, Any]:
        """Model 5: Transformer-only Baseline."""
        self._set_seed(seed)
        model = TransformerOnlyClassifier(
            num_numerical=39,
            cat_cardinalities=[134, 14, 10],
            token_dim=64,
            transformer_heads=4,
            transformer_layers=2,
            classifier_hidden_dim=128,
            num_classes=self.NUM_CLASSES,
        )
        return self._train_and_eval_nn_graph(model, "Transformer-only", seed, epochs=self.epochs, lr=self.lr)

    def evaluate_graphsage_only(self, seed: int) -> Dict[str, Any]:
        """Model 6: Temporal GraphSAGE-only Baseline."""
        self._set_seed(seed)
        model = GraphSAGEOnlyClassifier(
            in_channels=194,
            edge_dim=6,
            hidden_dim=64,
            out_channels=64,
            num_layers=2,
            classifier_hidden_dim=128,
            num_classes=self.NUM_CLASSES,
        )
        return self._train_and_eval_nn_graph(model, "GraphSAGE-only", seed, epochs=self.epochs, lr=self.lr)

    def evaluate_graphsage_plus_transformer_concat(self, seed: int) -> Dict[str, Any]:
        """Model 7: GraphSAGE + Transformer with Concatenation Fusion (No Pretrain)."""
        self._set_seed(seed)
        model = TGCFIDS(
            num_numerical=39,
            cat_cardinalities=[134, 14, 10],
            token_dim=64,
            transformer_heads=4,
            transformer_layers=2,
            graph_in_channels=194,
            graph_edge_dim=6,
            graph_hidden_dim=64,
            graph_out_channels=64,
            graph_layers=2,
            fusion_dim=128,
            fusion_strategy="concat",
            classifier_hidden_dim=128,
            num_classes=self.NUM_CLASSES,
        )
        return self._train_and_eval_nn_graph(model, "GraphSAGE + Transformer (Concat)", seed, epochs=self.epochs, lr=self.lr)

    def evaluate_sageconv_transformer_gated_nopretrain(self, seed: int) -> Dict[str, Any]:
        """Model 8: SAGEConv + Transformer with Gated Fusion (No Pretrain)."""
        self._set_seed(seed)
        model = TGCFIDS(
            num_numerical=39,
            cat_cardinalities=[134, 14, 10],
            token_dim=64,
            transformer_heads=4,
            transformer_layers=2,
            graph_in_channels=194,
            graph_edge_dim=6,
            graph_hidden_dim=64,
            graph_out_channels=64,
            graph_layers=2,
            fusion_dim=128,
            fusion_strategy="gated",
            classifier_hidden_dim=128,
            num_classes=self.NUM_CLASSES,
        )
        return self._train_and_eval_nn_graph(model, "SAGEConv + Transformer + Gated (No Pretrain)", seed, epochs=self.epochs, lr=self.lr)

    def evaluate_tgcf_ids(self, seed: int) -> Dict[str, Any]:
        """Model 9: Full TGCF-IDS (Contrastive Pretrained Graph Encoder + Gated Fusion)."""
        self._set_seed(seed)
        model = TGCFIDS(
            num_numerical=39,
            cat_cardinalities=[134, 14, 10],
            token_dim=64,
            transformer_heads=4,
            transformer_layers=2,
            graph_in_channels=194,
            graph_edge_dim=6,
            graph_hidden_dim=64,
            graph_out_channels=64,
            graph_layers=2,
            fusion_dim=128,
            fusion_strategy="gated",
            classifier_hidden_dim=128,
            num_classes=self.NUM_CLASSES,
        )
        # Load self-supervised contrastive pretrained weights
        if self.pretrained_ckpt.exists():
            model.load_pretrained_graph_encoder(self.pretrained_ckpt)
        else:
            print(f"[!] Warning: Pretrained checkpoint not found at {self.pretrained_ckpt}; training from scratch.")

        return self._train_and_eval_nn_graph(model, "TGCF-IDS (Proposed)", seed, epochs=self.epochs, lr=self.lr)

    # -------------------------------------------------------------------------
    # Benchmark Orchestrator
    # -------------------------------------------------------------------------
    def run_benchmark(
        self,
        models_to_run: Optional[List[str]] = None,
        max_train_windows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute controlled multi-seed benchmark across all 9 model configurations.
        """
        print("=" * 80)
        print("         TGCF-IDS : Controlled Multi-Model Scientific Benchmark         ")
        print("=" * 80)
        print(f"[*] Execution Device     : {self.device} (AMP Enabled: {self.use_amp})")
        print(f"[*] Evaluated Seeds      : {self.seeds}")
        print(f"[*] Epochs per Deep Run  : {self.epochs}")

        self.load_all_data()

        if max_train_windows and self.train_graphs:
            self.train_graphs = self.train_graphs[:max_train_windows]
            print(f"[*] Subsetting training graphs to {len(self.train_graphs)} windows for verification.")

        all_models = [
            ("Random Forest", self.evaluate_random_forest),
            ("XGBoost", self.evaluate_xgboost),
            ("MLP", self.evaluate_mlp),
            ("BiLSTM", self.evaluate_bilstm),
            ("Transformer-only", self.evaluate_transformer_only),
            ("GraphSAGE-only", self.evaluate_graphsage_only),
            ("GraphSAGE + Transformer (Concat)", self.evaluate_graphsage_plus_transformer_concat),
            ("SAGEConv + Transformer + Gated (No Pretrain)", self.evaluate_sageconv_transformer_gated_nopretrain),
            ("TGCF-IDS (Proposed)", self.evaluate_tgcf_ids),
        ]

        if models_to_run:
            all_models = [m for m in all_models if m[0] in models_to_run]

        raw_results: List[Dict[str, Any]] = []

        for model_name, eval_fn in all_models:
            print(f"\n" + "=" * 80)
            print(f"[*] Running Benchmark for: {model_name}")
            print("=" * 80)

            for seed in self.seeds:
                print(f"  [+] Starting Seed: {seed:3d} ... ", end="", flush=True)
                try:
                    res = eval_fn(seed)
                    raw_results.append(res)
                    print(f"Done! | Acc: {res['accuracy']:.4f} | Macro F1: {res['macro_f1']:.4f} | W-F1: {res['weighted_f1']:.4f} | FPR: {res['binary_fpr']:.4f} | ROC-AUC: {res['roc_auc_macro']:.4f}")
                except Exception as e:
                    print(f"FAILED with error: {e}")

        # Aggregate across seeds
        summary_rows = []
        per_class_rows = []

        model_names_present = []
        for m in self.MODEL_ORDER:
            if any(r["model"] == m for r in raw_results):
                model_names_present.append(m)

        for m_name in model_names_present:
            m_runs = [r for r in raw_results if r["model"] == m_name]
            if not m_runs:
                continue

            accs = [r["accuracy"] for r in m_runs]
            m_precs = [r["macro_precision"] for r in m_runs]
            m_recs = [r["macro_recall"] for r in m_runs]
            m_f1s = [r["macro_f1"] for r in m_runs]
            w_f1s = [r["weighted_f1"] for r in m_runs]
            b_fprs = [r["binary_fpr"] for r in m_runs]
            b_fnrs = [r["binary_fnr"] for r in m_runs]
            roc_aucs = [r["roc_auc_macro"] for r in m_runs]
            pr_aucs = [r["pr_auc_macro"] for r in m_runs]
            t_times = [r["training_time_s"] for r in m_runs]
            inf_times = [r["inference_ms_per_1k"] for r in m_runs]
            params = m_runs[0]["total_params"]
            train_params = m_runs[0]["trainable_params"]

            summary_rows.append({
                "Model": m_name,
                "Accuracy": f"{np.mean(accs):.4f} +/- {np.std(accs):.4f}",
                "Macro Precision": f"{np.mean(m_precs):.4f} +/- {np.std(m_precs):.4f}",
                "Macro Recall": f"{np.mean(m_recs):.4f} +/- {np.std(m_recs):.4f}",
                "Macro F1": f"{np.mean(m_f1s):.4f} +/- {np.std(m_f1s):.4f}",
                "Weighted F1": f"{np.mean(w_f1s):.4f} +/- {np.std(w_f1s):.4f}",
                "Binary FPR": f"{np.mean(b_fprs):.4f} +/- {np.std(b_fprs):.4f}",
                "Binary FNR": f"{np.mean(b_fnrs):.4f} +/- {np.std(b_fnrs):.4f}",
                "ROC-AUC (Macro)": f"{np.mean(roc_aucs):.4f} +/- {np.std(roc_aucs):.4f}",
                "PR-AUC (Macro)": f"{np.mean(pr_aucs):.4f} +/- {np.std(pr_aucs):.4f}",
                "Train Time (s)": f"{np.mean(t_times):.1f} +/- {np.std(t_times):.1f}",
                "Inf Latency (ms/1k)": f"{np.mean(inf_times):.2f} +/- {np.std(inf_times):.2f}",
                "Total Params": f"{params:,}" if params > 0 else "N/A",
                "Trainable Params": f"{train_params:,}" if train_params > 0 else "N/A",
                # Numeric fields for sorting and programmatic analysis
                "acc_mean": np.mean(accs),
                "acc_std": np.std(accs),
                "macro_f1_mean": np.mean(m_f1s),
                "macro_f1_std": np.std(m_f1s),
                "weighted_f1_mean": np.mean(w_f1s),
                "weighted_f1_std": np.std(w_f1s),
                "fpr_mean": np.mean(b_fprs),
                "fnr_mean": np.mean(b_fnrs),
                "roc_auc_mean": np.mean(roc_aucs),
                "pr_auc_mean": np.mean(pr_aucs),
                "train_time_mean": np.mean(t_times),
                "inf_time_mean": np.mean(inf_times),
                "total_params_num": params,
            })

            # Per-class metrics aggregation
            per_class_f1_matrix = np.array([r["per_class_f1"] for r in m_runs])
            per_class_rec_matrix = np.array([r["per_class_recall"] for r in m_runs])
            per_class_prec_matrix = np.array([r["per_class_precision"] for r in m_runs])

            for c_idx, c_name in enumerate(self.CLASS_NAMES):
                per_class_rows.append({
                    "Model": m_name,
                    "Class ID": c_idx,
                    "Class Name": c_name,
                    "Precision": f"{np.mean(per_class_prec_matrix[:, c_idx]):.4f} +/- {np.std(per_class_prec_matrix[:, c_idx]):.4f}",
                    "Recall": f"{np.mean(per_class_rec_matrix[:, c_idx]):.4f} +/- {np.std(per_class_rec_matrix[:, c_idx]):.4f}",
                    "F1 Score": f"{np.mean(per_class_f1_matrix[:, c_idx]):.4f} +/- {np.std(per_class_f1_matrix[:, c_idx]):.4f}",
                    "f1_mean": float(np.mean(per_class_f1_matrix[:, c_idx])),
                    "f1_std": float(np.std(per_class_f1_matrix[:, c_idx])),
                })

        summary_df = pd.DataFrame(summary_rows)
        raw_df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)} for r in raw_results])
        per_class_df = pd.DataFrame(per_class_rows)

        # Export tables
        summary_path = self.results_tables_dir / "baseline_comparison.csv"
        per_class_path = self.results_tables_dir / "baseline_per_class_comparison.csv"
        raw_runs_path = self.results_tables_dir / "baseline_raw_runs.csv"

        summary_df.to_csv(summary_path, index=False)
        per_class_df.to_csv(per_class_path, index=False)
        raw_df.to_csv(raw_runs_path, index=False)

        print("\n" + "=" * 80)
        print("          CONTROLLED MULTI-MODEL BENCHMARK RESULTS (Mean +/- Std)          ")
        print("=" * 80)
        print(summary_df[["Model", "Accuracy", "Macro F1", "Weighted F1", "Binary FPR", "Binary FNR", "ROC-AUC (Macro)"]].to_string(index=False))
        print("=" * 80)
        print(f"[+] Saved Summary Table -> {summary_path}")
        print(f"[+] Saved Per-Class Table -> {per_class_path}")
        print(f"[+] Saved Raw Runs -> {raw_runs_path}")

        return {
            "summary_df": summary_df,
            "per_class_df": per_class_df,
            "raw_df": raw_df,
        }


def main():
    parser = argparse.ArgumentParser(description="TGCF-IDS Controlled Baseline Benchmark")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per deep baseline (default: 15)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Random seeds")
    parser.add_argument("--max-train-windows", type=int, default=None, help="Subset of training windows for quick testing")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="Specific models to evaluate")

    args = parser.parse_args()

    benchmark = ControlledBenchmarkFramework(
        epochs=args.epochs,
        seeds=args.seeds,
    )
    benchmark.run_benchmark(
        models_to_run=args.models,
        max_train_windows=args.max_train_windows,
    )


if __name__ == "__main__":
    main()
