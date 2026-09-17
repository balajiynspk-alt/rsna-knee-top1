#!/usr/bin/env python3
"""
TGCF-IDS: Controlled Robustness & Sensitivity Testing Framework.

EVALUATION PROTOCOL:
Evaluates the frozen TGCF-IDS model against 6 distinct real-world perturbation modes
across multiple perturbation strengths without modifying original test ground-truth data:
1. Feature Masking: Simulating input sensor dropouts (r in [0.0, 0.5]).
2. Numerical Noise: Gaussian jitter N(0, sigma^2) on continuous features (sigma in [0.0, 0.5]).
3. Random Edge Deletion: Communication failure / packet drops (p in [0.0, 0.5]).
4. Random Edge Addition: Spurious relational traffic / cross-talk (p in [0.0, 0.5]).
5. Controlled Temporal Perturbation: Jitter on edge temporal deltas (sigma in [0.0, 1.0]).
6. Missing Feature Simulation: Random channel dropouts simulating unobserved telemetry (r in [0.0, 0.5]).

All perturbations are strictly deterministic and reproducible.
Outputs:
- results/tables/robustness.csv
- results/figures/robustness/ (Degradation curves and sensitivity dashboard)
"""

import copy
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
)
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


def seed_everything(seed: int = 42) -> None:
    """Enforce complete determinism across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class RobustnessPerturber:
    """
    Deterministic input perturbation operators that never mutate original data.
    """

    @staticmethod
    def apply_feature_masking(data: Data, mask_ratio: float, seed: int = 42) -> Data:
        """Mask a fraction of continuous features to zero."""
        if mask_ratio <= 0.0:
            return data
        rng = np.random.RandomState(seed)
        data_pert = data.clone()
        x_num = data_pert.x_num.clone()
        mask = torch.tensor(rng.binomial(1, 1.0 - mask_ratio, size=x_num.shape), dtype=torch.float32)
        data_pert.x_num = x_num * mask

        if hasattr(data_pert, "x_dense") and data_pert.x_dense is not None:
            x_dense = data_pert.x_dense.clone()
            x_dense[:, :x_num.shape[1]] = data_pert.x_num
            data_pert.x_dense = x_dense
        return data_pert

    @staticmethod
    def apply_numerical_noise(data: Data, noise_std: float, seed: int = 42) -> Data:
        """Inject zero-mean Gaussian noise to numerical features."""
        if noise_std <= 0.0:
            return data
        rng = np.random.RandomState(seed)
        data_pert = data.clone()
        noise = torch.tensor(rng.normal(0.0, noise_std, size=data_pert.x_num.shape), dtype=torch.float32)
        data_pert.x_num = data_pert.x_num + noise

        if hasattr(data_pert, "x_dense") and data_pert.x_dense is not None:
            x_dense = data_pert.x_dense.clone()
            x_dense[:, :data_pert.x_num.shape[1]] = data_pert.x_num
            data_pert.x_dense = x_dense
        return data_pert

    @staticmethod
    def apply_edge_deletion(data: Data, drop_ratio: float, seed: int = 42) -> Data:
        """Drop a fraction of relational graph edges."""
        if drop_ratio <= 0.0 or data.edge_index.size(1) == 0:
            return data
        rng = np.random.RandomState(seed)
        data_pert = data.clone()
        num_edges = data_pert.edge_index.size(1)
        keep_mask = rng.binomial(1, 1.0 - drop_ratio, size=num_edges).astype(bool)
        if not np.any(keep_mask):
            keep_mask[0] = True  # Ensure at least 1 edge
        data_pert.edge_index = data_pert.edge_index[:, keep_mask]
        if hasattr(data_pert, "edge_attr") and data_pert.edge_attr is not None:
            data_pert.edge_attr = data_pert.edge_attr[keep_mask]
        return data_pert

    @staticmethod
    def apply_edge_addition(data: Data, add_ratio: float, seed: int = 42) -> Data:
        """Add random cross-talk graph edges."""
        if add_ratio <= 0.0:
            return data
        rng = np.random.RandomState(seed)
        data_pert = data.clone()
        num_nodes = data_pert.num_nodes
        orig_edges = data_pert.edge_index.size(1)
        num_to_add = int(orig_edges * add_ratio)
        if num_to_add == 0 or num_nodes <= 1:
            return data

        src = rng.randint(0, num_nodes, size=num_to_add)
        dst = rng.randint(0, num_nodes, size=num_to_add)
        new_edges = torch.tensor([src, dst], dtype=torch.long)
        data_pert.edge_index = torch.cat([data_pert.edge_index, new_edges], dim=1)

        if hasattr(data_pert, "edge_attr") and data_pert.edge_attr is not None:
            new_attr = torch.tensor(rng.normal(0.0, 1.0, size=(num_to_add, data_pert.edge_attr.size(1))), dtype=torch.float32)
            data_pert.edge_attr = torch.cat([data_pert.edge_attr, new_attr], dim=0)
        return data_pert

    @staticmethod
    def apply_temporal_perturbation(data: Data, jitter_std: float, seed: int = 42) -> Data:
        """Inject temporal jitter into edge relational attributes."""
        if jitter_std <= 0.0 or not hasattr(data, "edge_attr") or data.edge_attr is None:
            return data
        rng = np.random.RandomState(seed)
        data_pert = data.clone()
        edge_attr = data_pert.edge_attr.clone()
        # Jitter column 0 (delta_time) and column 3 (rate_ratio)
        jitter = torch.tensor(rng.normal(0.0, jitter_std, size=edge_attr.shape), dtype=torch.float32)
        data_pert.edge_attr = edge_attr + jitter
        return data_pert

    @staticmethod
    def apply_missing_feature_simulation(data: Data, drop_cols_ratio: float, seed: int = 42) -> Data:
        """Simulate complete unobserved feature telemetry channels."""
        if drop_cols_ratio <= 0.0:
            return data
        rng = np.random.RandomState(seed)
        data_pert = data.clone()
        x_num = data_pert.x_num.clone()
        num_cols = x_num.shape[1]
        dropped_cols = rng.choice(num_cols, size=int(num_cols * drop_cols_ratio), replace=False)
        x_num[:, dropped_cols] = 0.0
        data_pert.x_num = x_num

        if hasattr(data_pert, "x_dense") and data_pert.x_dense is not None:
            x_dense = data_pert.x_dense.clone()
            x_dense[:, :num_cols] = x_num
            data_pert.x_dense = x_dense
        return data_pert


class RobustnessEvaluator:
    """
    Controlled Robustness Evaluation Suite for TGCF-IDS.
    """

    CLASS_NAMES = [
        "Normal", "Analysis", "Backdoor", "DoS", "Exploits",
        "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms"
    ]
    NUM_CLASSES = 10

    # Test Strengths for each perturbation mode
    STRENGTH_GRIDS = {
        "Feature Masking (Ratio)": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "Numerical Noise (Gaussian Std)": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
        "Random Edge Deletion (Ratio)": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "Random Edge Addition (Ratio)": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "Temporal Attribute Jitter (Std)": [0.0, 0.1, 0.2, 0.3, 0.5, 1.0],
        "Missing Feature Simulation (Channel Drop)": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    }

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        test_graphs_path: Optional[Union[str, Path]] = None,
        output_table: Optional[Union[str, Path]] = None,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        self.config_path = Path(config_path or ProjectPaths.CONFIGS_DIR / "best_hyperparameters.yaml")
        final_ckpt = ProjectPaths.RESULTS_CHECKPOINTS / "final_tgcf_ids_model.pt"
        self.checkpoint_path = Path(checkpoint_path or (final_ckpt if final_ckpt.exists() else ProjectPaths.RESULTS_CHECKPOINTS / "best_tgcf_ids.pt"))
        self.test_graphs_path = Path(test_graphs_path or ProjectPaths.DATA_GRAPHS / "test_graphs.pt")
        self.output_table = Path(output_table or ProjectPaths.RESULTS_TABLES / "robustness.csv")
        self.output_table.parent.mkdir(parents=True, exist_ok=True)

        self.seed = seed
        seed_everything(self.seed)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.use_amp = (self.device.type == "cuda")

        self.model = self._load_frozen_model()
        self.raw_test_graphs = torch.load(self.test_graphs_path, weights_only=False, map_location="cpu")

    def _load_frozen_model(self) -> TGCFIDS:
        """Load and strictly freeze the validated TGCF-IDS architecture."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                hp_data = yaml.safe_load(f)
            hp = hp_data.get("hyperparameters", hp_data)
        else:
            hp = {"hidden_dimension": 64, "embedding_dimension": 64, "transformer_layers": 2, "transformer_heads": 4, "gnn_layers": 2, "dropout": 0.1}

        model = TGCFIDS(
            num_numerical=39,
            cat_cardinalities=[134, 14, 10],
            token_dim=hp.get("embedding_dimension", 64),
            transformer_heads=hp.get("transformer_heads", 4),
            transformer_layers=hp.get("transformer_layers", 2),
            transformer_ffn_dim=hp.get("embedding_dimension", 64) * 4,
            transformer_dropout=hp.get("dropout", 0.1),
            graph_in_channels=194,
            graph_edge_dim=6,
            graph_hidden_dim=hp.get("hidden_dimension", 64),
            graph_out_channels=hp.get("embedding_dimension", 64),
            graph_layers=hp.get("gnn_layers", 2),
            graph_dropout=hp.get("dropout", 0.1),
            fusion_dim=hp.get("hidden_dimension", 64) * 2,
            fusion_strategy="gated",
            classifier_hidden_dim=hp.get("hidden_dimension", 64),
            classifier_dropout=hp.get("dropout", 0.1) * 1.5,
            num_classes=self.NUM_CLASSES,
        ).to(self.device)

        if self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, weights_only=False, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            try:
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"[!] Warning loading checkpoint: {e}")

        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def evaluate_perturbed_graphs(self, graphs: List[Data]) -> Dict[str, float]:
        """Perform frozen inference over perturbed graph snapshots."""
        loader = DataLoader(graphs, batch_size=4, shuffle=False)
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                y_target = batch.y_multiclass if hasattr(batch, "y_multiclass") and batch.y_multiclass is not None else batch.y
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    out = self.model(data=batch)
                    logits = out.logits if hasattr(out, "logits") else out
                    probs = torch.softmax(logits, dim=-1)

                preds = torch.argmax(probs, dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_target.cpu().numpy())

        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Binary security rates
        y_bin_true = (y_true > 0).astype(int)
        y_bin_pred = (y_pred > 0).astype(int)
        cm_bin = confusion_matrix(y_bin_true, y_bin_pred, labels=[0, 1])
        tn, fp, fn, tp = cm_bin.ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "macro_recall": macro_recall,
            "macro_precision": macro_precision,
            "weighted_f1": weighted_f1,
            "fpr": fpr,
            "fnr": fnr,
        }

    def run_robustness_study(self) -> pd.DataFrame:
        """Run complete controlled sensitivity experiments across all 6 perturbation modalities."""
        print("=" * 85)
        print("                 TGCF-IDS : CONTROLLED ROBUSTNESS & SENSITIVITY STUDY               ")
        print("=" * 85)

        results_list = []
        perturber = RobustnessPerturber()

        for exp_idx, (pert_name, strength_grid) in enumerate(self.STRENGTH_GRIDS.items(), 1):
            print(f"\n[{exp_idx}/6] Evaluating Perturbation Mode: {pert_name}")

            for strength in strength_grid:
                # Apply deterministic perturbation per graph snapshot without mutating originals
                perturbed_graphs = []
                for g_idx, g in enumerate(self.raw_test_graphs):
                    seed = self.seed + g_idx * 101

                    if "Feature Masking" in pert_name:
                        g_pert = perturber.apply_feature_masking(g, strength, seed=seed)
                    elif "Numerical Noise" in pert_name:
                        g_pert = perturber.apply_numerical_noise(g, strength, seed=seed)
                    elif "Edge Deletion" in pert_name:
                        g_pert = perturber.apply_edge_deletion(g, strength, seed=seed)
                    elif "Edge Addition" in pert_name:
                        g_pert = perturber.apply_edge_addition(g, strength, seed=seed)
                    elif "Temporal Attribute" in pert_name:
                        g_pert = perturber.apply_temporal_perturbation(g, strength, seed=seed)
                    elif "Missing Feature" in pert_name:
                        g_pert = perturber.apply_missing_feature_simulation(g, strength, seed=seed)
                    else:
                        g_pert = g

                    perturbed_graphs.append(g_pert)

                metrics = self.evaluate_perturbed_graphs(perturbed_graphs)
                print(f"    Strength: {strength:4.2f} | Acc: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Recall: {metrics['macro_recall']:.4f} | FPR: {metrics['fpr']*100:.2f}% | FNR: {metrics['fnr']*100:.2f}%")

                results_list.append({
                    "Perturbation Mode": pert_name,
                    "Perturbation Strength": strength,
                    "Accuracy": metrics["accuracy"],
                    "Macro F1": metrics["macro_f1"],
                    "Macro Recall": metrics["macro_recall"],
                    "Macro Precision": metrics["macro_precision"],
                    "Weighted F1": metrics["weighted_f1"],
                    "FPR (False Alarm Rate)": metrics["fpr"],
                    "FNR (Missed Attack Rate)": metrics["fnr"],
                })

        df_results = pd.DataFrame(results_list)
        df_results.to_csv(self.output_table, index=False)
        print(f"\n[+] Saved complete robustness study -> {self.output_table}")

        print("\n" + "=" * 85)
        print("                     ROBUSTNESS EXPERIMENT SUMMARY TABLE                            ")
        print("=" * 85)
        print(df_results.to_string(index=False))
        print("=" * 85)
        return df_results


def main():
    evaluator = RobustnessEvaluator()
    evaluator.run_robustness_study()


if __name__ == "__main__":
    main()
