#!/usr/bin/env python3
"""
TGCF-IDS: Multi-Dimensional Computational Efficiency & Complexity Benchmark.

Rigorous benchmarking across 5 core architectures:
1. MLP (Tabular baseline)
2. Transformer (Feature-Transformer only)
3. GraphSAGE (Edge-Aware Relational GNN only)
4. GraphSAGE + Transformer (Concatenation fusion baseline)
5. TGCF-IDS (Full dual-branch temporal graph contrastive feature-transformer)

Evaluates:
- Parameter Count
- Model Size (MB)
- Peak GPU Memory (MB)
- CPU Resident Memory (MB)
- Training Time & Epoch Time (sec)
- Inference Latency (ms / 1,000 flows)
- Throughput (flows / sec)
- Classification Performance (Accuracy, Macro F1, Weighted F1)

Strict Protocol:
- Evaluated on identical hardware (CUDA with AMP) and standardized batch settings.
- Outputs saved to results/tables/efficiency.csv.
"""

import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
try:
    import psutil
except ImportError:
    psutil = None
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.paths import ProjectPaths
from src.models.mlp import PyTorchMLP
from src.models.ablation_models import (
    TransformerOnlyClassifier,
    GraphSAGEOnlyClassifier,
    AblationTGCFIDS,
)
from src.models.tgcf_ids import TGCFIDS


def seed_everything(seed: int = 42) -> None:
    """Enforce complete determinism."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MLPWrapper(nn.Module):
    """Wraps PyTorchMLP to accept PyG batch data."""
    def __init__(self, in_features: int = 194, hidden_dims: List[int] = [256, 128], num_classes: int = 10):
        super().__init__()
        self.mlp = PyTorchMLP(in_features=in_features, hidden_dims=hidden_dims, num_classes=num_classes)

    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        x = data.x_dense if hasattr(data, "x_dense") and data.x_dense is not None else data.x
        return self.mlp(x)


class EfficiencyBenchmarkSuite:
    """
    Standardized computational efficiency benchmark.
    """

    MODELS = [
        "MLP",
        "Transformer",
        "GraphSAGE",
        "GraphSAGE + Transformer",
        "TGCF-IDS",
    ]

    def __init__(
        self,
        train_graphs_path: Optional[Union[str, Path]] = None,
        test_graphs_path: Optional[Union[str, Path]] = None,
        output_table: Optional[Union[str, Path]] = None,
        batch_size: int = 4,
        epochs: int = 3,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        self.train_graphs_path = Path(train_graphs_path or ProjectPaths.DATA_GRAPHS / "train_graphs.pt")
        self.test_graphs_path = Path(test_graphs_path or ProjectPaths.DATA_GRAPHS / "test_graphs.pt")
        self.output_table = Path(output_table or ProjectPaths.RESULTS_TABLES / "efficiency.csv")
        self.output_table.parent.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        seed_everything(self.seed)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.use_amp = (self.device.type == "cuda")

        print(f"[*] Efficiency Benchmark Initialized on Device: {self.device} (AMP: {self.use_amp})")
        self.train_graphs = torch.load(self.train_graphs_path, weights_only=False, map_location="cpu")
        self.test_graphs = torch.load(self.test_graphs_path, weights_only=False, map_location="cpu")

    def _instantiate_model(self, model_name: str) -> nn.Module:
        """Create clean instance of target model architecture."""
        if model_name == "MLP":
            model = MLPWrapper(in_features=194, hidden_dims=[256, 128], num_classes=10)
        elif model_name == "Transformer":
            model = TransformerOnlyClassifier(
                num_numerical=39,
                cat_cardinalities=[134, 14, 10],
                token_dim=64,
                transformer_heads=4,
                transformer_layers=2,
                transformer_ffn_dim=256,
                transformer_dropout=0.1,
                classifier_hidden_dim=64,
                num_classes=10,
            )
        elif model_name == "GraphSAGE":
            model = GraphSAGEOnlyClassifier(
                in_channels=194,
                edge_dim=6,
                hidden_dim=64,
                out_channels=64,
                num_layers=2,
                dropout=0.1,
                classifier_hidden_dim=64,
                num_classes=10,
            )
        elif model_name == "GraphSAGE + Transformer":
            model = AblationTGCFIDS(
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
                classifier_hidden_dim=64,
                num_classes=10,
            )
        elif model_name == "TGCF-IDS":
            model = TGCFIDS(
                num_numerical=39,
                cat_cardinalities=[134, 14, 10],
                token_dim=64,
                transformer_heads=4,
                transformer_layers=2,
                transformer_ffn_dim=256,
                transformer_dropout=0.1,
                graph_in_channels=194,
                graph_edge_dim=6,
                graph_hidden_dim=64,
                graph_out_channels=64,
                graph_layers=2,
                graph_dropout=0.1,
                fusion_dim=128,
                fusion_strategy="gated",
                classifier_hidden_dim=64,
                num_classes=10,
            )
            # Load pretrained encoder if available
            pretrain_path = ProjectPaths.RESULTS_CHECKPOINTS / "graph_pretrained.pt"
            if pretrain_path.exists():
                model.load_pretrained_graph_encoder(pretrain_path)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return model.to(self.device)

    def benchmark_model(self, model_name: str) -> Dict[str, Any]:
        """
        Run training, memory tracking, and inference timing for a single model.
        """
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model = self._instantiate_model(model_name)

        # 1. Parameter count and model size
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

        # 2. Training benchmark
        train_loader = DataLoader(self.train_graphs, batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        model.train()
        train_t0 = time.perf_counter()

        for epoch in range(self.epochs):
            for batch in train_loader:
                batch = batch.to(self.device)
                y_targets = batch.y_multiclass if hasattr(batch, "y_multiclass") and batch.y_multiclass is not None else batch.y
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    out = model(data=batch) if not isinstance(model, TransformerOnlyClassifier) else model(x_num=batch.x_num, x_cat=batch.x_cat)
                    logits = out.logits if hasattr(out, "logits") else out
                    loss = criterion(logits, y_targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        train_time_total = time.perf_counter() - train_t0
        epoch_time_sec = train_time_total / max(self.epochs, 1)

        # Peak GPU memory
        peak_gpu_mem_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if self.device.type == "cuda" else 0.0

        # CPU memory
        cpu_mem_mb = (psutil.Process().memory_info().rss / (1024 * 1024)) if psutil is not None else 0.0

        # 3. Inference latency and throughput benchmark
        model.eval()
        test_loader = DataLoader(self.test_graphs, batch_size=self.batch_size, shuffle=False)

        all_preds = []
        all_targets = []
        total_samples = 0

        # Warmup
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    _ = model(data=batch) if not isinstance(model, TransformerOnlyClassifier) else model(x_num=batch.x_num, x_cat=batch.x_cat)
                break

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inf_t0 = time.perf_counter()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                y_targets = batch.y_multiclass if hasattr(batch, "y_multiclass") and batch.y_multiclass is not None else batch.y
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    out = model(data=batch) if not isinstance(model, TransformerOnlyClassifier) else model(x_num=batch.x_num, x_cat=batch.x_cat)
                    logits = out.logits if hasattr(out, "logits") else out
                preds = torch.argmax(logits, dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_targets.cpu().numpy())
                total_samples += len(y_targets)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        inf_time_total = time.perf_counter() - inf_t0

        throughput_fps = total_samples / max(inf_time_total, 1e-6)
        latency_ms_per_1k = (inf_time_total / (total_samples / 1000.0)) * 1000.0

        # 4. Classification performance
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        return {
            "Model Architecture": model_name,
            "Parameters": int(param_count),
            "Model Size (MB)": float(model_size_mb),
            "GPU Peak Memory (MB)": float(peak_gpu_mem_mb),
            "CPU Memory (MB)": float(cpu_mem_mb),
            "Total Training Time (s)": float(train_time_total),
            "Epoch Time (s/epoch)": float(epoch_time_sec),
            "Inference Latency (ms/1k flows)": float(latency_ms_per_1k),
            "Throughput (flows/sec)": float(throughput_fps),
            "Accuracy": float(acc),
            "Macro F1": float(macro_f1),
            "Weighted F1": float(weighted_f1),
        }

    def run_benchmark_suite(self) -> pd.DataFrame:
        """Run full benchmark across all 5 architectures."""
        print("=" * 85)
        print("             TGCF-IDS : COMPUTATIONAL EFFICIENCY & COMPLEXITY BENCHMARK             ")
        print("=" * 85)

        results = []
        for idx, model_name in enumerate(self.MODELS, 1):
            print(f"\n[{idx}/5] Benchmarking Architecture: {model_name}...")
            res = self.benchmark_model(model_name)
            results.append(res)
            print(f"    - Params: {res['Parameters']:,} | Size: {res['Model Size (MB)']:.2f} MB | Peak GPU: {res['GPU Peak Memory (MB)']:.1f} MB")
            print(f"    - Train Time: {res['Total Training Time (s)']:.2f}s ({res['Epoch Time (s/epoch)']:.2f}s/epoch)")
            print(f"    - Latency: {res['Inference Latency (ms/1k flows)']:.2f} ms/1k flows | Throughput: {res['Throughput (flows/sec)']:.1f} flows/s")
            print(f"    - Performance: Acc = {res['Accuracy']:.4f} | Macro F1 = {res['Macro F1']:.4f} | Weighted F1 = {res['Weighted F1']:.4f}")

        df = pd.DataFrame(results)
        df.to_csv(self.output_table, index=False)
        print(f"\n[+] Saved efficiency benchmark table -> {self.output_table}")

        print("\n" + "=" * 85)
        print("                      COMPUTATIONAL EFFICIENCY SUMMARY                              ")
        print("=" * 85)
        print(df.to_string(index=False))
        print("=" * 85)
        return df


def main():
    suite = EfficiencyBenchmarkSuite()
    suite.run_benchmark_suite()


if __name__ == "__main__":
    main()
