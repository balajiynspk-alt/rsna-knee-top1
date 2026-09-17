#!/usr/bin/env python3
"""
TGCF-IDS: Comprehensive Model Explainability & Attribution Framework.

RIGOROUS SCIENTIFIC TAXONOMY & SEPARATION:
1. Feature Attribution (Integrated Gradients & Gradient x Input):
   - Computes path-integral feature attributions attributing prediction scores to input features.
2. Graph & Subgraph Attribution (Edge Saliency & Neighbor Impact):
   - Attributes predictions to topological neighborhood flows, communication links, and edge attributes.
3. Transformer Self-Attention Introspection:
   - Visualizes multi-head attention routing between feature tokens.
   - IMPORTANT: Attention weights reflect representational routing, NOT direct causal explanations.
4. Temporal Flow Sequence Analysis:
   - Traces temporal intervals, burst sequences, and temporal edge dynamics.
5. Prediction Confidence & Uncertainty:
   - Softmax probability margins and Shannon entropy.
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.paths import ProjectPaths
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


class CaseStudyExplanation(NamedTuple):
    """Container holding multi-modal explanation artifacts for a single prediction."""
    sample_index: int
    true_label_id: int
    true_label_name: str
    pred_label_id: int
    pred_label_name: str
    confidence: float
    entropy: float
    top3_classes: List[Tuple[str, float]]
    feature_attributions: Dict[str, float]       # Integrated Gradients
    top_positive_features: List[Tuple[str, float]]
    top_negative_features: List[Tuple[str, float]]
    attention_matrix: np.ndarray                 # [Heads, Tokens, Tokens]
    neighbor_attributions: List[Dict[str, Any]]  # Influential graph neighbors
    temporal_flow_sequence: List[Dict[str, Any]] # Sequential context
    gate_values: np.ndarray                      # Modality fusion gate


class IntegratedGradientsExplainer:
    """
    Computes Integrated Gradients for TGCF-IDS tabular feature tokens.
    """

    FEATURE_NAMES = [
        "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "Sload", "Dload",
        "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
        "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt",
        "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
        "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "proto", "service", "state"
    ]

    def __init__(self, model: TGCFIDS, device: torch.device):
        self.model = model
        self.device = device

    def attribute_sample(
        self,
        data: Data,
        node_idx: int,
        target_class: int,
        steps: int = 30,
    ) -> np.ndarray:
        """
        Compute Integrated Gradients for node_idx along linear path from baseline to input.
        """
        self.model.eval()
        x_num = data.x_num.to(self.device).clone()
        x_cat = data.x_cat.to(self.device).clone()
        edge_index = data.edge_index.to(self.device)
        edge_attr = data.edge_attr.to(self.device)
        x_dense = data.x_dense.to(self.device) if hasattr(data, "x_dense") and data.x_dense is not None else x_num

        baseline = torch.zeros_like(x_num)
        target_val = x_num[node_idx].unsqueeze(0)
        baseline_val = baseline[node_idx].unsqueeze(0)
        diff = target_val - baseline_val

        grads = []
        for step in range(1, steps + 1):
            alpha = float(step) / steps
            interpolated = baseline.clone()
            interpolated[node_idx] = baseline_val + alpha * diff
            interpolated.requires_grad_(True)

            out = self.model(
                x_num=interpolated,
                x_cat=x_cat,
                edge_index=edge_index,
                edge_attr=edge_attr,
                x_dense=x_dense,
            )
            logits = out.logits if hasattr(out, "logits") else out
            score = logits[node_idx, target_class]
            self.model.zero_grad()
            if interpolated.grad is not None:
                interpolated.grad.zero_()
            score.backward(retain_graph=True)
            grads.append(interpolated.grad[node_idx].detach().cpu())

        avg_grads = torch.stack(grads).mean(dim=0)
        ig_scores = (diff.detach().cpu().squeeze(0) * avg_grads).numpy()
        return ig_scores


class GraphSaliencyExplainer:
    """
    Computes edge and neighborhood saliency attributions for graph structure.
    """

    def __init__(self, model: TGCFIDS, device: torch.device):
        self.model = model
        self.device = device

    def attribute_subgraph(
        self,
        data: Data,
        target_node: int,
        target_class: int,
    ) -> List[Dict[str, Any]]:
        """
        Attributes importance to incoming and outgoing edges for target_node.
        """
        self.model.eval()
        edge_attr = data.edge_attr.to(self.device).clone()
        edge_attr.requires_grad_(True)
        edge_index = data.edge_index.to(self.device)

        out = self.model(
            x_num=data.x_num.to(self.device),
            x_cat=data.x_cat.to(self.device),
            edge_index=edge_index,
            edge_attr=edge_attr,
            x_dense=data.x_dense.to(self.device) if hasattr(data, "x_dense") else None,
        )
        logits = out.logits if hasattr(out, "logits") else out
        score = logits[target_node, target_class]
        self.model.zero_grad()
        score.backward()

        edge_grads = edge_attr.grad.abs().sum(dim=-1).detach().cpu().numpy()
        src_nodes = edge_index[0].cpu().numpy()
        dst_nodes = edge_index[1].cpu().numpy()

        connected_neighbors = []
        for e_idx in range(len(edge_grads)):
            if dst_nodes[e_idx] == target_node or src_nodes[e_idx] == target_node:
                neighbor = src_nodes[e_idx] if dst_nodes[e_idx] == target_node else dst_nodes[e_idx]
                if neighbor != target_node:
                    connected_neighbors.append({
                        "edge_index": e_idx,
                        "neighbor_id": int(neighbor),
                        "importance_score": float(edge_grads[e_idx]),
                        "direction": "incoming" if dst_nodes[e_idx] == target_node else "outgoing",
                    })

        # Sort by attribution importance
        connected_neighbors.sort(key=lambda x: x["importance_score"], reverse=True)
        return connected_neighbors[:8]


class TGCFIDSExplainabilitySuite:
    """
    High-level explainability orchestrator generating multi-modal case studies.
    """

    CLASS_NAMES = [
        "Normal", "Analysis", "Backdoor", "DoS", "Exploits",
        "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms"
    ]

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        test_graphs_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        self.config_path = Path(config_path or ProjectPaths.CONFIGS_DIR / "best_hyperparameters.yaml")
        final_ckpt = ProjectPaths.RESULTS_CHECKPOINTS / "final_tgcf_ids_model.pt"
        self.checkpoint_path = Path(checkpoint_path or (final_ckpt if final_ckpt.exists() else ProjectPaths.RESULTS_CHECKPOINTS / "best_tgcf_ids.pt"))
        self.test_graphs_path = Path(test_graphs_path or ProjectPaths.DATA_GRAPHS / "test_graphs.pt")
        self.output_dir = Path(output_dir or ProjectPaths.RESULTS_FIGURES / "explainability")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seed = seed
        seed_everything(self.seed)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self._load_model()
        self.ig_explainer = IntegratedGradientsExplainer(self.model, self.device)
        self.graph_explainer = GraphSaliencyExplainer(self.model, self.device)
        self.test_graphs = torch.load(self.test_graphs_path, weights_only=False, map_location="cpu")

    def _load_model(self) -> TGCFIDS:
        """Load frozen model."""
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
            num_classes=10,
        ).to(self.device)

        if self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, weights_only=False, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            try:
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"[!] Warning loading checkpoint: {e}")

        model.eval()
        return model

    def explain_case_study(
        self,
        graph_idx: int,
        node_idx: int,
    ) -> CaseStudyExplanation:
        """
        Generate complete multi-modal explanation for a specific flow event.
        """
        g = self.test_graphs[graph_idx]
        y_true = int(g.y_multiclass[node_idx].item() if hasattr(g, "y_multiclass") else g.y[node_idx].item())

        # Forward pass with attention extraction
        with torch.no_grad():
            out = self.model(
                x_num=g.x_num.to(self.device),
                x_cat=g.x_cat.to(self.device),
                edge_index=g.edge_index.to(self.device),
                edge_attr=g.edge_attr.to(self.device),
                x_dense=g.x_dense.to(self.device) if hasattr(g, "x_dense") else None,
                return_attention=True,
            )
            logits = out.logits[node_idx].unsqueeze(0)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_id = int(np.argmax(probs))
            conf = float(probs[pred_id])
            entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))
            gate_vals = out.gate_values[node_idx].cpu().numpy()

        # Top 3 classes
        top3_ids = np.argsort(probs)[::-1][:3]
        top3_classes = [(self.CLASS_NAMES[i], float(probs[i])) for i in top3_ids]

        # 1. Feature Attribution (Integrated Gradients)
        ig_scores = self.ig_explainer.attribute_sample(g, node_idx=node_idx, target_class=pred_id, steps=25)
        num_names = IntegratedGradientsExplainer.FEATURE_NAMES[:len(ig_scores)]
        feat_attr_dict = {name: float(score) for name, score in zip(num_names, ig_scores)}

        # Sort features
        sorted_feats = sorted(feat_attr_dict.items(), key=lambda x: x[1], reverse=True)
        top_pos = [f for f in sorted_feats if f[1] > 0][:5]
        top_neg = [f for f in sorted_feats if f[1] < 0][-5:]

        # 2. Transformer Attention Matrix
        attn_matrix = None
        if out.attention_weights is not None and len(out.attention_weights) > 0:
            last_layer_attn = out.attention_weights[-1]
            if last_layer_attn is not None:
                attn_matrix = last_layer_attn[node_idx].cpu().numpy()  # [Heads, Tokens, Tokens]

        # 3. Graph Saliency & Influential Neighbors
        neighbors = self.graph_explainer.attribute_subgraph(g, target_node=node_idx, target_class=pred_id)

        # 4. Temporal Flow Sequence Context
        seq = []
        for offset in range(-3, 4):
            idx_seq = node_idx + offset
            if 0 <= idx_seq < g.num_nodes:
                seq.append({
                    "offset": offset,
                    "flow_idx": idx_seq,
                    "is_target": (idx_seq == node_idx),
                    "dur": float(g.x_num[idx_seq, 0].item()),
                    "sbytes": float(g.x_num[idx_seq, 1].item()),
                    "dbytes": float(g.x_num[idx_seq, 2].item()),
                })

        return CaseStudyExplanation(
            sample_index=node_idx,
            true_label_id=y_true,
            true_label_name=self.CLASS_NAMES[y_true],
            pred_label_id=pred_id,
            pred_label_name=self.CLASS_NAMES[pred_id],
            confidence=conf,
            entropy=entropy,
            top3_classes=top3_classes,
            feature_attributions=feat_attr_dict,
            top_positive_features=top_pos,
            top_negative_features=top_neg,
            attention_matrix=attn_matrix,
            neighbor_attributions=neighbors,
            temporal_flow_sequence=seq,
            gate_values=gate_vals,
        )

    def run_explainability_case_studies(self) -> List[CaseStudyExplanation]:
        """
        Selects representative test cases across key attack categories and produces explanations.
        """
        print("=" * 85)
        print("              TGCF-IDS : MULTI-MODAL EXPLAINABILITY & ATTRIBUTION SUITE             ")
        print("=" * 85)

        target_attacks = [
            ("DoS", 3),
            ("Reconnaissance", 7),
            ("Exploits", 4),
            ("Normal", 0),
        ]

        explanations = []
        for attack_name, attack_id in target_attacks:
            found = False
            for g_idx, g in enumerate(self.test_graphs):
                y = g.y_multiclass if hasattr(g, "y_multiclass") else g.y
                indices = (y == attack_id).nonzero(as_tuple=True)[0]
                if len(indices) > 0:
                    node_idx = int(indices[0].item())
                    exp = self.explain_case_study(g_idx, node_idx)
                    explanations.append(exp)
                    found = True
                    print(f"\n[+] Generated Explanation for: {attack_name} (Node {node_idx} in Graph {g_idx})")
                    print(f"    - Predicted: {exp.pred_label_name} (Confidence: {exp.confidence*100:.2f}%, Entropy: {exp.entropy:.3f})")
                    print("    - Top Positive Features (Integrated Gradients):")
                    for name, score in exp.top_positive_features[:3]:
                        print(f"        * {name:<18}: +{score:.4f}")
                    print("    - Influential Graph Neighbors:")
                    for n in exp.neighbor_attributions[:3]:
                        print(f"        * Neighbor Flow {n['neighbor_id']} ({n['direction']}): Saliency = {n['importance_score']:.4f}")
                    break
            if not found:
                print(f"[!] Could not find instance for: {attack_name}")

        return explanations


def main():
    suite = TGCFIDSExplainabilitySuite()
    explanations = suite.run_explainability_case_studies()


if __name__ == "__main__":
    main()
