#!/usr/bin/env python3
"""
TGCF-IDS: Self-Supervised Temporal Graph Contrastive Pretraining.

Learns robust, structural network-flow representations without attack labels by maximizing
mutual information between stochastic augmented graph views (InfoNCE / NT-Xent Loss).

Architecture:
    View 1 (G1) -> Temporal GraphSAGE -> Projection Head -> z1
    View 2 (G2) -> Temporal GraphSAGE -> Projection Head -> z2
    Loss: NT-Xent(z1, z2, temperature)

Outputs:
- Pretrained encoder checkpoint: results/checkpoints/graph_pretrained.pt
- Training curves:              results/figures/contrastive/contrastive_training_curves.png
- Embedding visualizations:     results/figures/contrastive/contrastive_embeddings_tsne.png
- Loss history:                 results/tables/contrastive_loss_history.csv
- Metrics JSON:                 results/tables/contrastive_metrics.json
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
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Ensure project root is accessible
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.temporal_graphsage import TemporalGraphSAGE
from src.models.projection_head import ContrastiveProjectionHead
from src.models.graph_augmentations import GraphAugmentor
from src.utils.paths import ProjectPaths


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (InfoNCE / NT-Xent).

    Computes symmetrical contrastive loss between representations of corresponding nodes across two views:
        L_{i,j} = -log( exp(sim(z_{1,i}, z_{2,i}) / tau) / sum_{k != i} exp(sim(z_{1,i}, z_k) / tau) )
    """

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: Normalized embeddings from View 1 of shape [N, D].
            z2: Normalized embeddings from View 2 of shape [N, D].

        Returns:
            loss: Scalar NT-Xent contrastive loss.
        """
        N = z1.size(0)
        if N <= 1:
            return torch.tensor(0.0, device=z1.device, requires_grad=True)

        # L2-normalize
        z1_norm = F.normalize(z1, p=2, dim=-1)
        z2_norm = F.normalize(z2, p=2, dim=-1)

        # Full concatenated representation [2N, D]
        z = torch.cat([z1_norm, z2_norm], dim=0)

        # Full cosine similarity matrix [2N, 2N]
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Create self-contrast mask (exclude diagonal i == i)
        diag_mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(diag_mask, -1e4)

        # Targets: index of positive pair for i in [0..N-1] is i + N, and for i in [N..2N-1] is i - N
        pos_indices = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(0, N, device=z.device),
        ])

        loss = F.cross_entropy(sim_matrix, pos_indices)
        return loss


class GraphContrastiveModel(nn.Module):
    """
    Combined Self-Supervised Graph Contrastive Model:
        Encoder (TemporalGraphSAGE) + Projection Head (ContrastiveProjectionHead)
    """

    def __init__(
        self,
        in_channels: int = 194,
        edge_dim: int = 6,
        hidden_dim: int = 64,
        proj_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        aggr: str = "mean",
    ):
        super().__init__()
        self.encoder = TemporalGraphSAGE(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            aggr=aggr,
            use_residual=True,
            use_layer_norm=True,
        )

        self.projection_head = ContrastiveProjectionHead(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            out_dim=proj_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract normalized projection embedding z for contrastive loss computation.
        """
        h = self.encoder(x=x, edge_index=edge_index, edge_attr=edge_attr)
        z = self.projection_head(h, normalize=True)
        return z

    def get_latent_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract unprojected latent representations h from the encoder for downstream tasks.
        """
        return self.encoder(x=x, edge_index=edge_index, edge_attr=edge_attr)


class ContrastiveTrainer:
    """
    End-to-end trainer for self-supervised graph contrastive pretraining.
    """

    CLASS_NAMES = [
        "Normal", "Analysis", "Backdoor", "DoS", "Exploits",
        "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
    ]

    def __init__(
        self,
        in_channels: int = 194,
        edge_dim: int = 6,
        hidden_dim: int = 64,
        proj_dim: int = 64,
        num_layers: int = 2,
        temperature: float = 0.2,
        feature_mask_ratio: float = 0.15,
        feature_noise_std: float = 0.05,
        edge_drop_ratio: float = 0.15,
        temporal_jitter_std: float = 0.02,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 4,
        epochs: int = 25,
        seed: int = 42,
        device: Optional[str] = None,
        graphs_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ):
        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.num_layers = num_layers
        self.temperature = temperature
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

        # Seed setup
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Paths
        self.graphs_dir = Path(graphs_dir) if graphs_dir else ProjectPaths.DATA_GRAPHS
        if results_dir:
            base_res = Path(results_dir)
            self.checkpoints_dir = base_res / "checkpoints"
            self.figures_dir = base_res / "figures" / "contrastive"
            self.tables_dir = base_res / "tables"
        else:
            self.checkpoints_dir = ProjectPaths.RESULTS_CHECKPOINTS
            self.figures_dir = ProjectPaths.RESULTS_FIGURES / "contrastive"
            self.tables_dir = ProjectPaths.RESULTS_TABLES

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        # Augmentor & Loss
        self.augmentor = GraphAugmentor(
            feature_mask_ratio=feature_mask_ratio,
            feature_noise_std=feature_noise_std,
            edge_drop_ratio=edge_drop_ratio,
            temporal_jitter_std=temporal_jitter_std,
            random_seed=seed,
        )
        self.criterion = NTXentLoss(temperature=temperature)

        # Model instance
        self.model = GraphContrastiveModel(
            in_channels=self.in_channels,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            proj_dim=self.proj_dim,
            num_layers=self.num_layers,
            dropout=0.1,
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

        self.use_amp = (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "contrastive_loss": [],
            "lr": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch over all graph snapshots."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_graphs in dataloader:
            # Batch graphs is a PyG Batch of multiple graph snapshots
            # Generate two augmented views of the batched graph
            view1, view2 = self.augmentor.create_augmented_pair(batch_graphs)

            view1 = view1.to(self.device)
            view2 = view2.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                z1 = self.model(x=view1.x_dense, edge_index=view1.edge_index, edge_attr=view1.edge_attr)
                z2 = self.model(x=view2.x_dense, edge_index=view2.edge_index, edge_attr=view2.edge_attr)
                loss = self.criterion(z1, z2)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def train_pipeline(self) -> Dict[str, Any]:
        """Execute full contrastive pretraining pipeline, checkpointing, and evaluation visualizations."""
        train_graphs_path = self.graphs_dir / "train_graphs.pt"
        test_graphs_path = self.graphs_dir / "test_graphs.pt"

        if not train_graphs_path.exists():
            raise FileNotFoundError(f"Training graphs not found at: {train_graphs_path}")

        print("=" * 75)
        print("    TGCF-IDS : Self-Supervised Graph Contrastive Pretraining (InfoNCE)   ")
        print("=" * 75)
        print(f"[*] Execution Device     : {self.device} (AMP Enabled: {self.use_amp})")
        print(f"[*] Pretraining Setup    : HiddenDim={self.hidden_dim}, ProjDim={self.proj_dim}, Layers={self.num_layers}, Tau={self.temperature}")
        print(f"[*] Augmentations        : MaskRatio={self.augmentor.mask_ratio}, NoiseStd={self.augmentor.noise_std}, EdgeDrop={self.augmentor.drop_ratio}")

        # Load Graph Snapshots
        train_graphs = torch.load(train_graphs_path, weights_only=False)
        dataloader = DataLoader(train_graphs, batch_size=self.batch_size, shuffle=True)

        print(f"[+] Loaded {len(train_graphs)} Training Graph Snapshots (Batched at {self.batch_size} graphs/batch)")
        print("-" * 75)
        print(f"{'Epoch':^8} | {'Contrastive Loss':^20} | {'Learning Rate':^15} | {'Elapsed (s)':^12}")
        print("-" * 75)

        t_start = time.time()
        best_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            ep_t0 = time.time()
            loss_val = self.train_epoch(dataloader)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            self.history["epoch"].append(epoch)
            self.history["contrastive_loss"].append(loss_val)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - ep_t0
            is_best = loss_val < best_loss
            if is_best:
                best_loss = loss_val

            status = "[BEST]" if is_best else ""
            print(f"{epoch:^8} | {loss_val:^20.4f} | {current_lr:^15.2e} | {elapsed:^12.2f} {status}")

        total_time = time.time() - t_start
        print("-" * 75)
        print(f"[+] Pretraining Completed in {total_time:.2f} seconds. Lowest NT-Xent Loss: {best_loss:.4f}")

        # Save Pretrained Checkpoint
        checkpoint_path = self.checkpoints_dir / "graph_pretrained.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "encoder_state_dict": self.model.encoder.state_dict(),
            "in_channels": self.in_channels,
            "hidden_dim": self.hidden_dim,
            "proj_dim": self.proj_dim,
            "num_layers": self.num_layers,
            "best_loss": best_loss,
            "epochs": self.epochs,
            "history": self.history,
        }, checkpoint_path)
        print(f"[+] Saved Pretrained Checkpoint -> {checkpoint_path}")

        # Save Loss History Table
        loss_df = pd.DataFrame(self.history)
        loss_csv_path = self.tables_dir / "contrastive_loss_history.csv"
        loss_df.to_csv(loss_csv_path, index=False)
        print(f"[+] Saved Loss History Table   -> {loss_csv_path}")

        # Plot Contrastive Training Curves
        self._plot_training_curves()

        # Generate Latent Embedding Visualization on Test Graph Snapshots
        if test_graphs_path.exists():
            test_graphs = torch.load(test_graphs_path, weights_only=False)
            self._visualize_latent_embeddings(test_graphs[:5])

        # Summary Metrics JSON
        metrics = {
            "model": "Temporal Graph Contrastive Pretraining",
            "loss_function": "NT-Xent (InfoNCE)",
            "temperature": self.temperature,
            "hidden_dim": self.hidden_dim,
            "proj_dim": self.proj_dim,
            "num_layers": self.num_layers,
            "epochs": self.epochs,
            "best_contrastive_loss": round(best_loss, 4),
            "total_pretraining_time_seconds": round(total_time, 2),
        }
        metrics_json_path = self.tables_dir / "contrastive_metrics.json"
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[+] Saved Summary Metrics JSON  -> {metrics_json_path}")
        print("=" * 75)

        return {
            "model": self.model,
            "best_loss": best_loss,
            "checkpoint_path": checkpoint_path,
            "history": self.history,
        }

    def _plot_training_curves(self) -> None:
        """Plot and save NT-Xent loss progression curve."""
        plt.figure(figsize=(9, 5), dpi=200)
        epochs = self.history["epoch"]
        losses = self.history["contrastive_loss"]

        plt.plot(epochs, losses, marker="o", color="#8e44ad", linewidth=2.0, label="NT-Xent Loss")
        plt.title("Self-Supervised Graph Contrastive Pretraining Loss", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("InfoNCE / NT-Xent Loss", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True, fontsize=10)
        plt.tight_layout()

        out_path = self.figures_dir / "contrastive_training_curves.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved Training Curves Plot -> {out_path}")

    def _visualize_latent_embeddings(self, sample_graphs: List[Data], max_samples: int = 1500) -> None:
        """
        Extract unprojected latent representations from the pretrained encoder,
        run 2D t-SNE, and visualize cluster emergence (color-coded by attack category post-hoc).
        """
        self.model.eval()
        latent_list = []
        labels_list = []

        with torch.no_grad():
            for g in sample_graphs:
                g = g.to(self.device)
                h = self.model.get_latent_embeddings(
                    x=g.x_dense,
                    edge_index=g.edge_index,
                    edge_attr=g.edge_attr,
                )
                latent_list.append(h.cpu().numpy())
                labels_list.append(g.y_multiclass.cpu().numpy())

        all_latents = np.vstack(latent_list)[:max_samples]
        all_labels = np.concatenate(labels_list)[:max_samples]

        # Run t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=self.seed, n_iter_without_progress=300)
        emb_2d = tsne.fit_transform(all_latents)

        plt.figure(figsize=(10, 8), dpi=200)
        palette = [
            "#2ecc71", "#e74c3c", "#9b59b6", "#3498db", "#e67e22",
            "#1abc9c", "#f39c12", "#34495e", "#d35400", "#c0392b",
        ]

        present_classes = sorted(list(set(all_labels)))
        for cls_id in present_classes:
            mask = (all_labels == cls_id)
            cls_name = self.CLASS_NAMES[cls_id] if cls_id < len(self.CLASS_NAMES) else f"Class {cls_id}"
            plt.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                label=f"{cls_id}: {cls_name} (N={mask.sum()})",
                color=palette[cls_id % len(palette)],
                alpha=0.65,
                s=20,
                edgecolors="none",
            )

        plt.title(
            "Self-Supervised Pretrained Graph Embeddings (2D t-SNE Projection)\n"
            "(Unsupervised Representation Colored by Attack Taxonomy)",
            fontsize=11,
            fontweight="bold",
            pad=12,
        )
        plt.xlabel("t-SNE Dimension 1", fontsize=10)
        plt.ylabel("t-SNE Dimension 2", fontsize=10)
        plt.legend(
            title="Flow Category (Post-Hoc Verification)",
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
            frameon=True,
            fontsize=8,
            title_fontsize=9,
        )
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        out_path = self.figures_dir / "contrastive_embeddings_tsne.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved Embedding Visualization -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="TGCF-IDS: Self-Supervised Graph Contrastive Pretraining")
    parser.add_argument("--epochs", type=int, default=20, help="Pretraining epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=4, help="Graph snapshots per mini-batch (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--temp", type=float, default=0.2, help="NT-Xent temperature parameter (default: 0.2)")
    parser.add_argument("--hidden-dim", type=int, default=64, help="GraphSAGE hidden dim (default: 64)")
    parser.add_argument("--proj-dim", type=int, default=64, help="Projection head dim (default: 64)")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GraphSAGE layers (default: 2)")

    args = parser.parse_args()

    trainer = ContrastiveTrainer(
        in_channels=194,
        edge_dim=6,
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        num_layers=args.num_layers,
        temperature=args.temp,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    trainer.train_pipeline()


if __name__ == "__main__":
    main()
