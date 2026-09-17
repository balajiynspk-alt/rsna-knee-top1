#!/usr/bin/env python3
"""
TGCF-IDS: Temporal Network-Flow Graph Construction Pipeline.

Constructs sparse PyTorch Geometric (PyG) graphs from consecutive network-flow records
without label leakage, combining:
1. Temporal proximity edges (local sequence adjacency)
2. Shared source/destination host relationship edges
3. Protocol and service similarity edges
4. Port and connection state relationship edges
5. Continuous feature-space K-Nearest Neighbors (KNN) edges

Outputs:
- PyG Data objects saved under data/graphs/
- Graph topology diagnostics saved under results/tables/
- Network visualization figures saved under results/figures/graphs/
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
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data

# Ensure project root is accessible
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.paths import ProjectPaths


class TemporalGraphBuilder:
    """
    Constructs leakage-safe temporal graph snapshots from tabular network-flow sequences.
    """

    CLASS_NAMES = [
        "Normal", "Analysis", "Backdoor", "DoS", "Exploits",
        "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
    ]

    def __init__(
        self,
        window_size: int = 1000,
        stride: Optional[int] = None,
        knn_k: int = 5,
        temporal_radius: int = 2,
        proto_match_weight: float = 1.0,
        service_match_weight: float = 1.0,
        random_seed: int = 42,
    ):
        """
        Args:
            window_size: Number of consecutive flow records per temporal graph snapshot.
            stride: Stride between consecutive windows (defaults to window_size for non-overlapping windows).
            knn_k: Number of nearest neighbors in feature space per node.
            temporal_radius: Number of forward/backward temporal steps to connect adjacent flows.
            proto_match_weight: Weight modifier for protocol matching.
            service_match_weight: Weight modifier for service matching.
            random_seed: Random seed for reproducibility.
        """
        self.window_size = window_size
        self.stride = stride or window_size
        self.knn_k = knn_k
        self.temporal_radius = temporal_radius
        self.proto_match_weight = proto_match_weight
        self.service_match_weight = service_match_weight
        self.seed = random_seed

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def build_window_graph(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        x_dense: torch.Tensor,
        y_multi: Optional[torch.Tensor] = None,
        y_bin: Optional[torch.Tensor] = None,
        meta_ids: Optional[torch.Tensor] = None,
        meta_seq: Optional[torch.Tensor] = None,
        window_idx: int = 0,
    ) -> Data:
        """
        Construct a single sparse PyG Data graph object from a temporal window of nodes.

        Edge Construction Logic (Zero Label Leakage):
        1. Temporal adjacency: connect (i, j) if |i - j| <= temporal_radius.
        2. Feature KNN: connect (i, j) if j in KNN_k(i) using cosine similarity in continuous feature space.
        3. Shared Protocol / Service matching: connect flows with identical non-zero protocol/service.
        4. Edge Attributes (6 dimensions):
           - [0] temporal_dist: |i - j| / W
           - [1] src_relationship: 1.0 / (1.0 + |ct_srv_src_i - ct_srv_src_j|)
           - [2] dst_relationship: 1.0 / (1.0 + |ct_dst_src_ltm_i - ct_dst_src_ltm_j|)
           - [3] proto_similarity: 1.0 if proto_i == proto_j else 0.0
           - [4] port_service_rel: 1.0 if service_i == service_j else 0.0
           - [5] feature_similarity: cosine similarity in [-1.0, 1.0]
        """
        num_nodes = x_num.size(0)
        if num_nodes == 0:
            raise ValueError("Cannot construct graph from empty node batch.")

        src_list: List[int] = []
        dst_list: List[int] = []

        # 1. Temporal Adjacency Edges (|i - j| <= radius)
        if self.temporal_radius > 0:
            for i in range(num_nodes):
                low = max(0, i - self.temporal_radius)
                high = min(num_nodes, i + self.temporal_radius + 1)
                for j in range(low, high):
                    if i != j:
                        src_list.append(i)
                        dst_list.append(j)

        # 2. Feature Space KNN Edges
        k_effective = min(self.knn_k + 1, num_nodes)
        if k_effective > 1:
            x_dense_np = x_dense.detach().cpu().numpy()
            # Normalize for cosine metric
            norms = np.linalg.norm(x_dense_np, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            x_norm = x_dense_np / norms

            nbrs = NearestNeighbors(
                n_neighbors=k_effective,
                metric="cosine",
                algorithm="brute",
            ).fit(x_norm)
            _, indices = nbrs.kneighbors(x_norm)

            for i in range(num_nodes):
                for neighbor in indices[i]:
                    if neighbor != i:
                        src_list.append(i)
                        dst_list.append(neighbor)

        # 3. Protocol & Service Grouping Edges (Sparse sub-sampling if group is large)
        x_cat_np = x_cat.detach().cpu().numpy()
        proto_vals = x_cat_np[:, 0]
        serv_vals = x_cat_np[:, 1]

        # Connect identical non-zero protocols within moderate window distance
        unique_protos = np.unique(proto_vals)
        for p in unique_protos:
            if p == 0:  # Skip unknown
                continue
            members = np.where(proto_vals == p)[0]
            if len(members) > 1 and len(members) < 200:  # Avoid dense clique explosions
                # Connect consecutive members within the same protocol group
                for idx_m in range(len(members) - 1):
                    u, v = members[idx_m], members[idx_m + 1]
                    src_list.extend([u, v])
                    dst_list.extend([v, u])

        # 4. Deduplicate Edges
        raw_edges = set(zip(src_list, dst_list))
        # Remove self loops
        raw_edges = [(u, v) for u, v in raw_edges if u != v]

        if not raw_edges:
            # Fallback: connect consecutive nodes in line topology if disconnected
            raw_edges = [(i, i + 1) for i in range(num_nodes - 1)]
            raw_edges += [(i + 1, i) for i in range(num_nodes - 1)]

        src_arr = np.array([e[0] for e in raw_edges], dtype=np.int64)
        dst_arr = np.array([e[1] for e in raw_edges], dtype=np.int64)
        num_edges = len(raw_edges)

        # 5. Compute Rich 6-Dimensional Edge Attributes
        # Feature columns: ct_srv_src is col 27 in numerical, ct_dst_src_ltm is col 32
        x_num_np = x_num.detach().cpu().numpy()
        ct_srv_src = x_num_np[:, 27] if x_num_np.shape[1] > 27 else np.zeros(num_nodes)
        ct_dst_src = x_num_np[:, 32] if x_num_np.shape[1] > 32 else np.zeros(num_nodes)

        # [0] Temporal Distance (normalized)
        temporal_dist = np.abs(src_arr - dst_arr).astype(np.float32) / max(float(self.window_size), 1.0)

        # [1] Source Context Relationship Similarity
        src_diff = np.abs(ct_srv_src[src_arr] - ct_srv_src[dst_arr])
        src_rel = 1.0 / (1.0 + src_diff)

        # [2] Destination Context Relationship Similarity
        dst_diff = np.abs(ct_dst_src[src_arr] - ct_dst_src[dst_arr])
        dst_rel = 1.0 / (1.0 + dst_diff)

        # [3] Protocol Similarity (1 if identical proto else 0)
        proto_match = (proto_vals[src_arr] == proto_vals[dst_arr]) & (proto_vals[src_arr] != 0)
        proto_sim = proto_match.astype(np.float32)

        # [4] Port/Service Relationship (1 if identical service else 0)
        serv_match = (serv_vals[src_arr] == serv_vals[dst_arr]) & (serv_vals[src_arr] != 0)
        port_rel = serv_match.astype(np.float32)

        # [5] Continuous Cosine Feature Similarity
        x_dense_src = x_dense[src_arr]  # [E, D]
        x_dense_dst = x_dense[dst_arr]  # [E, D]
        dot_product = (x_dense_src * x_dense_dst).sum(dim=-1)
        norm_src = torch.norm(x_dense_src, dim=-1).clamp(min=1e-6)
        norm_dst = torch.norm(x_dense_dst, dim=-1).clamp(min=1e-6)
        feature_sim = (dot_product / (norm_src * norm_dst)).detach().cpu().numpy()

        # Combine into [E, 6] edge attribute matrix
        edge_attr_np = np.column_stack([
            temporal_dist,
            src_rel,
            dst_rel,
            proto_sim,
            port_rel,
            feature_sim,
        ]).astype(np.float32)

        edge_index_tensor = torch.from_numpy(np.vstack([src_arr, dst_arr])).long()
        edge_attr_tensor = torch.from_numpy(edge_attr_np)

        # Fallback targets and metadata if none provided
        if y_multi is None:
            y_multi = torch.zeros(num_nodes, dtype=torch.long)
        if y_bin is None:
            y_bin = torch.zeros(num_nodes, dtype=torch.long)
        if meta_ids is None:
            meta_ids = torch.arange(num_nodes, dtype=torch.long)
        if meta_seq is None:
            meta_seq = torch.arange(num_nodes, dtype=torch.long)

        graph_data = Data(
            x_num=x_num,
            x_cat=x_cat,
            x_dense=x_dense,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            y_multiclass=y_multi,
            y_binary=y_bin,
            meta_ids=meta_ids,
            meta_seq=meta_seq,
            window_idx=window_idx,
            num_nodes=num_nodes,
        )

        return graph_data

    def build_dataset_graphs(
        self,
        tensor_bundle: Dict[str, torch.Tensor],
        max_windows: Optional[int] = None,
    ) -> List[Data]:
        """
        Slice a complete split's feature tensors into temporal window graphs.
        """
        total_samples = tensor_bundle["x_num"].size(0)
        x_num = tensor_bundle["x_num"]
        x_cat = tensor_bundle["x_cat"]
        x_dense = tensor_bundle["x_dense"]
        y_multi = tensor_bundle.get("y_multiclass", torch.zeros(total_samples, dtype=torch.long))
        y_bin = tensor_bundle.get("y_binary", torch.zeros(total_samples, dtype=torch.long))
        meta_ids = tensor_bundle.get("meta_ids", torch.arange(total_samples, dtype=torch.long))
        meta_seq = tensor_bundle.get("meta_sequence_index", torch.arange(total_samples, dtype=torch.long))

        graphs: List[Data] = []
        window_idx = 0

        start_idx = 0
        while start_idx < total_samples:
            end_idx = min(start_idx + self.window_size, total_samples)

            # Extract window slice
            w_x_num = x_num[start_idx:end_idx].clone()
            w_x_cat = x_cat[start_idx:end_idx].clone()
            w_x_dense = x_dense[start_idx:end_idx].clone()
            w_y_multi = y_multi[start_idx:end_idx].clone()
            w_y_bin = y_bin[start_idx:end_idx].clone()
            w_meta_ids = meta_ids[start_idx:end_idx].clone()
            w_meta_seq = meta_seq[start_idx:end_idx].clone()

            graph = self.build_window_graph(
                x_num=w_x_num,
                x_cat=w_x_cat,
                x_dense=w_x_dense,
                y_multi=w_y_multi,
                y_bin=w_y_bin,
                meta_ids=w_meta_ids,
                meta_seq=w_meta_seq,
                window_idx=window_idx,
            )
            graphs.append(graph)
            window_idx += 1

            if max_windows is not None and len(graphs) >= max_windows:
                break

            start_idx += self.stride

        return graphs

    @staticmethod
    def compute_graph_diagnostics(graphs: List[Data]) -> pd.DataFrame:
        """
        Calculate comprehensive topological diagnostics for a list of graph windows:
        - Nodes
        - Edges
        - Average Degree
        - Median Degree
        - Min/Max Degree
        - Number of Connected Components
        - Number of Isolated Nodes
        - Attack Flow Ratio
        """
        records: List[Dict[str, Any]] = []

        for g in graphs:
            N = g.num_nodes
            E = g.edge_index.size(1)
            edge_idx_np = g.edge_index.cpu().numpy()

            # Degree computation
            degrees = np.bincount(edge_idx_np[0], minlength=N)
            avg_deg = float(np.mean(degrees))
            med_deg = float(np.median(degrees))
            min_deg = int(np.min(degrees))
            max_deg = int(np.max(degrees))
            isolated_count = int(np.sum(degrees == 0))

            # Connected components via Scipy
            adj_data = np.ones(E, dtype=np.int32)
            adj_sparse = coo_matrix((adj_data, (edge_idx_np[0], edge_idx_np[1])), shape=(N, N))
            n_components, _ = connected_components(adj_sparse, directed=False)

            # Attack density
            y_bin_np = g.y_binary.cpu().numpy()
            attack_ratio = float(np.mean(y_bin_np)) if len(y_bin_np) > 0 else 0.0

            records.append({
                "window_idx": g.window_idx,
                "num_nodes": N,
                "num_edges": E,
                "avg_degree": round(avg_deg, 2),
                "median_degree": round(med_deg, 2),
                "min_degree": min_deg,
                "max_degree": max_deg,
                "isolated_nodes": isolated_count,
                "connected_components": n_components,
                "attack_flow_ratio": round(attack_ratio, 4),
            })

        return pd.DataFrame(records)

    @classmethod
    def visualize_graph_window(
        cls,
        graph: Data,
        output_path: Union[str, Path],
        max_nodes_to_draw: int = 150,
        title: Optional[str] = None,
    ) -> None:
        """
        Render a publication-ready network topology diagram for a graph snapshot.
        Nodes are colored by attack class and sized by flow rate.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        N = min(graph.num_nodes, max_nodes_to_draw)
        edge_idx_np = graph.edge_index.cpu().numpy()
        y_multi_np = graph.y_multiclass.cpu().numpy()

        # Build NetworkX subgraph
        G = nx.Graph()
        for i in range(N):
            G.add_node(i, label=int(y_multi_np[i]))

        for u, v in zip(edge_idx_np[0], edge_idx_np[1]):
            if u < N and v < N and u != v:
                G.add_edge(int(u), int(v))

        plt.figure(figsize=(10, 8), dpi=200)

        # Spring layout with fixed seed
        pos = nx.spring_layout(G, k=0.25, iterations=50, seed=42)

        # Color palette for attack classes
        palette = [
            "#2ecc71", "#e74c3c", "#9b59b6", "#3498db", "#e67e22",
            "#1abc9c", "#f39c12", "#34495e", "#d35400", "#c0392b",
        ]
        node_colors = [palette[y_multi_np[node] % len(palette)] for node in G.nodes()]

        # Draw edges with transparency
        nx.draw_networkx_edges(
            G, pos,
            alpha=0.18,
            edge_color="#7f8c8d",
            width=0.8,
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=80,
            alpha=0.9,
            linewidths=0.8,
            edgecolors="#ffffff",
        )

        # Custom legend
        present_classes = sorted(list(set(y_multi_np[:N])))
        handles = [
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                label=f"{cls_id}: {cls.CLASS_NAMES[cls_id]}",
                markerfacecolor=palette[cls_id % len(palette)],
                markersize=8,
            )
            for cls_id in present_classes if cls_id < len(cls.CLASS_NAMES)
        ]

        plt.legend(
            handles=handles,
            title="Flow Category",
            loc="upper right",
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#bdc3c7",
            fontsize=8,
            title_fontsize=9,
        )

        window_title = title or f"TGCF-IDS Temporal Flow Graph (Window #{graph.window_idx}, Nodes={N}, Edges={G.number_of_edges()})"
        plt.title(window_title, fontsize=11, fontweight="bold", pad=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()


def run_temporal_graph_pipeline(
    window_size: int = 1000,
    stride: int = 1000,
    knn_k: int = 5,
    temporal_radius: int = 2,
    max_train_windows: Optional[int] = None,
    max_test_windows: Optional[int] = None,
) -> None:
    """
    Execute end-to-end temporal graph construction pipeline on official preprocessed UNSW-NB15 bundles.
    """
    print("=" * 75)
    print("        TGCF-IDS : Temporal Network-Flow Graph Construction Pipeline      ")
    print("=" * 75)

    data_dir = ProjectPaths.DATA_PROCESSED
    graphs_dir = ProjectPaths.DATA_GRAPHS
    figures_dir = ProjectPaths.RESULTS_FIGURES / "graphs"
    tables_dir = ProjectPaths.RESULTS_TABLES

    graphs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    train_pt = data_dir / "train_features.pt"
    test_pt = data_dir / "test_features.pt"

    if not train_pt.exists() or not test_pt.exists():
        raise FileNotFoundError(f"Processed feature bundles missing in {data_dir}. Run preprocessing first.")

    print(f"[*] Window Configuration  : Size={window_size}, Stride={stride}, KNN_k={knn_k}, Temporal_Radius={temporal_radius}")
    print(f"[+] Loading Preprocessed Tensor Bundles...")

    train_bundle = torch.load(train_pt, weights_only=True)
    test_bundle = torch.load(test_pt, weights_only=True)

    builder = TemporalGraphBuilder(
        window_size=window_size,
        stride=stride,
        knn_k=knn_k,
        temporal_radius=temporal_radius,
    )

    t0 = time.time()
    print(f"\n[+] Building Training Temporal Graphs from {train_bundle['x_num'].size(0):,} flows...")
    train_graphs = builder.build_dataset_graphs(train_bundle, max_windows=max_train_windows)
    print(f"[+] Constructed {len(train_graphs)} Training Graph Snapshots in {time.time() - t0:.2f}s")

    t1 = time.time()
    print(f"[+] Building Testing Temporal Graphs from {test_bundle['x_num'].size(0):,} flows...")
    test_graphs = builder.build_dataset_graphs(test_bundle, max_windows=max_test_windows)
    print(f"[+] Constructed {len(test_graphs)} Testing Graph Snapshots in {time.time() - t1:.2f}s")

    # Save Graph Lists to disk
    train_save_path = graphs_dir / "train_graphs.pt"
    test_save_path = graphs_dir / "test_graphs.pt"

    print(f"\n[+] Saving Graph Objects...")
    torch.save(train_graphs, train_save_path)
    torch.save(test_graphs, test_save_path)
    print(f"    - Saved Training Graphs -> {train_save_path}")
    print(f"    - Saved Testing Graphs  -> {test_save_path}")

    # Compute and Save Graph Diagnostics
    print(f"\n[+] Computing Graph Diagnostics...")
    train_diag_df = builder.compute_graph_diagnostics(train_graphs)
    test_diag_df = builder.compute_graph_diagnostics(test_graphs)

    train_diag_df["split"] = "train"
    test_diag_df["split"] = "test"
    full_diag_df = pd.concat([train_diag_df, test_diag_df], ignore_index=True)

    diag_csv_path = tables_dir / "graph_diagnostics.csv"
    full_diag_df.to_csv(diag_csv_path, index=False)
    print(f"    - Saved Graph Diagnostics Table -> {diag_csv_path}")

    # Diagnostic Summary Printout
    print("\n---------------------------------------------------------------------------")
    print(f"Graph Snapshot Summary Statistics (Window Size = {window_size}):")
    print(f" - Total Graph Windows (Train / Test) : {len(train_graphs)} / {len(test_graphs)}")
    print(f" - Mean Nodes per Window              : {full_diag_df['num_nodes'].mean():.1f}")
    print(f" - Mean Edges per Window              : {full_diag_df['num_edges'].mean():.1f}")
    print(f" - Average Node Degree                : {full_diag_df['avg_degree'].mean():.2f}")
    print(f" - Mean Connected Components          : {full_diag_df['connected_components'].mean():.2f}")
    print(f" - Mean Isolated Nodes per Window     : {full_diag_df['isolated_nodes'].mean():.2f}")
    print("---------------------------------------------------------------------------")

    # Visualizations
    print(f"\n[+] Generating Visualizations under {figures_dir}...")

    # Visualize 2 sample graph snapshots
    if len(train_graphs) > 0:
        builder.visualize_graph_window(
            train_graphs[0],
            figures_dir / "temporal_graph_window_0.png",
            title="TGCF-IDS Temporal Graph Window #0 (Train Split)",
        )
    if len(train_graphs) > 1:
        builder.visualize_graph_window(
            train_graphs[1],
            figures_dir / "temporal_graph_window_1.png",
            title="TGCF-IDS Temporal Graph Window #1 (Train Split)",
        )

    # Plot Degree Distribution & Connected Components
    plt.figure(figsize=(12, 5), dpi=200)

    plt.subplot(1, 2, 1)
    sns.histplot(full_diag_df["avg_degree"], kde=True, color="#2980b9", bins=20)
    plt.title("Average Degree Distribution Across Windows", fontsize=11, fontweight="bold")
    plt.xlabel("Average Node Degree", fontsize=10)
    plt.ylabel("Number of Graph Snapshots", fontsize=10)

    plt.subplot(1, 2, 2)
    sns.histplot(full_diag_df["connected_components"], kde=True, color="#e67e22", bins=20)
    plt.title("Connected Components Distribution", fontsize=11, fontweight="bold")
    plt.xlabel("Number of Connected Components", fontsize=10)
    plt.ylabel("Number of Graph Snapshots", fontsize=10)

    plt.tight_layout()
    plt.savefig(figures_dir / "graph_topology_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"    - Saved Network Snapshot #0        -> {figures_dir / 'temporal_graph_window_0.png'}")
    print(f"    - Saved Network Snapshot #1        -> {figures_dir / 'temporal_graph_window_1.png'}")
    print(f"    - Saved Topology Diagnostics Plot -> {figures_dir / 'graph_topology_diagnostics.png'}")

    print("=" * 75)
    print("Temporal Graph Construction and Diagnostics Completed Successfully.")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="TGCF-IDS: Temporal Graph Construction Pipeline")
    parser.add_argument("--window-size", type=int, default=1000, help="Nodes per temporal window (default: 1000)")
    parser.add_argument("--stride", type=int, default=1000, help="Stride between windows (default: 1000)")
    parser.add_argument("--knn-k", type=int, default=5, help="Number of feature KNN neighbors (default: 5)")
    parser.add_argument("--temporal-radius", type=int, default=2, help="Temporal adjacency step radius (default: 2)")
    parser.add_argument("--max-train-windows", type=int, default=None, help="Cap number of train windows")
    parser.add_argument("--max-test-windows", type=int, default=None, help="Cap number of test windows")

    args = parser.parse_args()
    run_temporal_graph_pipeline(
        window_size=args.window_size,
        stride=args.stride,
        knn_k=args.knn_k,
        temporal_radius=args.temporal_radius,
        max_train_windows=args.max_train_windows,
        max_test_windows=args.max_test_windows,
    )


if __name__ == "__main__":
    main()
