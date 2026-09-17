#!/usr/bin/env python3
"""
TGCF-IDS: Temporal Graph Contrastive Feature-Transformer Intrusion Detection System.

Complete unified architecture integrating:
1. Feature Branch:
   - FeatureTokenizer: Projects 39 numerical + 3 categorical features into 42 tokens in R^D
   - FeatureTransformer: Multi-Head Self-Attention encoder + Global pooling -> h_feat in R^{N x D_feat}
2. Graph Branch:
   - Edge-Aware Temporal GraphSAGE: Relational message passing with 6D edge attributes -> h_graph in R^{N x D_graph}
   - Contrastive Pretrained Initialization support
3. CrossModalFusion:
   - Dynamic Learned Gating / Concat / Add -> h_fused in R^{N x D_fused}
4. 10-Class Intrusion Classifier:
   - Linear -> LayerNorm -> GELU -> Dropout -> Linear -> Logits in R^{N x 10}

Outputs:
- logits: [N, 10]
- feature_repr: [N, D_feat]
- graph_repr: [N, D_graph]
- fused_repr: [N, D_fused]
- gate_values: [N, D_fused]
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from src.models.feature_tokenizer import FeatureTokenizer
from src.models.feature_transformer import FeatureTransformer, FeatureTransformerOutput
from src.models.temporal_graphsage import TemporalGraphSAGE
from src.models.cross_modal_fusion import CrossModalFusion, FusionOutput
from src.models.classifier import IntrusionClassifier


class TGCFIDSOutput(NamedTuple):
    """Complete container for TGCF-IDS model outputs."""
    logits: torch.Tensor                      # Raw classification logits: [N, 10]
    feature_repr: torch.Tensor                # Feature Transformer representation: [N, D_feat]
    graph_repr: torch.Tensor                  # Temporal GraphSAGE representation: [N, D_graph]
    fused_repr: torch.Tensor                  # CrossModalFused representation: [N, D_fused]
    gate_values: torch.Tensor                 # Dynamic gate values in [0, 1]: [N, D_fused] or [N, 1]
    feature_contrib: torch.Tensor             # Feature modality contribution: [N, D_fused]
    graph_contrib: torch.Tensor               # Graph modality contribution: [N, D_fused]
    attention_weights: Optional[List[torch.Tensor]] = None  # Introspection attention maps


class TGCFIDS(nn.Module):
    """
    TGCF-IDS: Full Dual-Branch Temporal Graph Contrastive Feature-Transformer Architecture.

    Args:
        num_numerical: Number of continuous numerical features (default: 39).
        cat_cardinalities: Cardinalities of categorical features (e.g., [134, 14, 10] with <UNK>).
        token_dim: Feature tokenizer token dimension (default: 64).
        transformer_heads: Number of attention heads in Feature Transformer (default: 4).
        transformer_layers: Number of Transformer encoder layers (default: 2).
        transformer_ffn_dim: FFN hidden dimension (default: 256).
        transformer_dropout: Dropout probability in Transformer (default: 0.1).
        transformer_pooling: Pooling strategy ('mean', 'cls', 'max', 'attention').
        graph_in_channels: Input node dimension for GraphSAGE (default: 194 for x_dense).
        graph_edge_dim: Number of edge attributes (default: 6).
        graph_hidden_dim: GraphSAGE hidden dimension (default: 64).
        graph_out_channels: GraphSAGE output embedding dimension (default: 64).
        graph_layers: Number of GraphSAGE layers (default: 2).
        graph_dropout: Dropout probability in GraphSAGE (default: 0.1).
        graph_aggr: Aggregation operator in GraphSAGE ('mean', 'max', 'sum').
        fusion_dim: Dimension of fused cross-modal space (default: 128).
        fusion_strategy: Cross-modal fusion strategy ('gated', 'concat', 'add').
        fusion_gate_type: Gating granularity ('vector' or 'scalar').
        fusion_dropout: Dropout probability in fusion module (default: 0.1).
        classifier_hidden_dim: Hidden dimension of classification head (default: 128).
        classifier_dropout: Dropout probability in classifier head (default: 0.2).
        num_classes: Number of intrusion classes (default: 10).
    """

    CLASS_NAMES = [
        "Normal", "Analysis", "Backdoor", "DoS", "Exploits",
        "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
    ]
    NUM_CLASSES = 10

    def __init__(
        self,
        num_numerical: int = 39,
        cat_cardinalities: Union[List[int], Dict[str, int]] = None,
        token_dim: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_ffn_dim: Optional[int] = None,
        transformer_dropout: float = 0.1,
        transformer_pooling: str = "mean",
        graph_in_channels: int = 194,
        graph_edge_dim: int = 6,
        graph_hidden_dim: int = 64,
        graph_out_channels: int = 64,
        graph_layers: int = 2,
        graph_dropout: float = 0.1,
        graph_aggr: str = "mean",
        fusion_dim: int = 128,
        fusion_strategy: str = "gated",
        fusion_gate_type: str = "vector",
        fusion_dropout: float = 0.1,
        classifier_hidden_dim: int = 128,
        classifier_dropout: float = 0.2,
        num_classes: int = 10,
    ):
        super().__init__()
        if cat_cardinalities is None:
            # Default UNSW-NB15 cardinalities with <UNK> token at index 0
            cat_cardinalities = [134, 14, 10]

        self.num_numerical = num_numerical
        self.token_dim = token_dim
        self.graph_in_channels = graph_in_channels
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes

        # -------------------------------------------------------------
        # 1. Feature Branch: FeatureTokenizer + FeatureTransformer
        # -------------------------------------------------------------
        self.tokenizer = FeatureTokenizer(
            num_numerical=num_numerical,
            cat_cardinalities=cat_cardinalities,
            embedding_dim=token_dim,
            bias=True,
            use_identity_emb=True,
            dropout=0.0,
        )

        self.transformer = FeatureTransformer(
            embedding_dim=token_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            ffn_dim=transformer_ffn_dim or (token_dim * 4),
            dropout=transformer_dropout,
            pooling=transformer_pooling,
            use_final_norm=True,
        )

        # -------------------------------------------------------------
        # 2. Graph Branch: Edge-Aware Temporal GraphSAGE
        # -------------------------------------------------------------
        self.graph_encoder = TemporalGraphSAGE(
            in_channels=graph_in_channels,
            edge_dim=graph_edge_dim,
            hidden_dim=graph_hidden_dim,
            out_channels=graph_out_channels,
            num_layers=graph_layers,
            dropout=graph_dropout,
            aggr=graph_aggr,
            use_residual=True,
            use_layer_norm=True,
        )

        # -------------------------------------------------------------
        # 3. Cross-Modal Fusion Engine
        # -------------------------------------------------------------
        self.fusion = CrossModalFusion(
            feat_dim=token_dim,
            graph_dim=graph_out_channels,
            fused_dim=fusion_dim,
            strategy=fusion_strategy,
            gate_type=fusion_gate_type,
            dropout=fusion_dropout,
            use_residual=True,
        )

        # -------------------------------------------------------------
        # 4. 10-Class Intrusion Classifier Head
        # -------------------------------------------------------------
        self.classifier = IntrusionClassifier(
            in_features=fusion_dim,
            hidden_dim=classifier_hidden_dim,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )

    def forward(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        x_dense: Optional[torch.Tensor] = None,
        data: Optional[Union[Data, Batch]] = None,
        return_attention: bool = False,
    ) -> TGCFIDSOutput:
        """
        Forward pass through full TGCF-IDS dual-branch pipeline.

        Accepts either a PyG Data/Batch snapshot object OR individual feature tensors.

        Args:
            x_num: Scaled numerical features [N, 39]
            x_cat: Ordinal categorical indices [N, 3]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge attributes [E, 6]
            x_dense: Optional dense combined features [N, 194] (derived automatically if None)
            data: Optional PyG Data or Batch object
            return_attention: If True, returns Transformer attention weights

        Returns:
            TGCFIDSOutput with logits, feature_repr, graph_repr, fused_repr, gate_values, contributions.
        """
        # Unpack from PyG Data / Batch object if provided
        if data is not None:
            if x_num is None and hasattr(data, "x_num"):
                x_num = data.x_num
            if x_cat is None and hasattr(data, "x_cat"):
                x_cat = data.x_cat
            if edge_index is None and hasattr(data, "edge_index"):
                edge_index = data.edge_index
            if edge_attr is None and hasattr(data, "edge_attr"):
                edge_attr = data.edge_attr
            if x_dense is None and hasattr(data, "x_dense"):
                x_dense = data.x_dense
            elif x_dense is None and hasattr(data, "x_graph"):
                x_dense = data.x_graph

            # Fallback if single combined data.x is present
            if (x_num is None or x_cat is None) and hasattr(data, "x"):
                if data.x.shape[-1] >= self.tokenizer.total_features:
                    x_num = data.x[:, :self.tokenizer.num_numerical]
                    x_cat = data.x[:, self.tokenizer.num_numerical:self.tokenizer.total_features]
                elif x_dense is None:
                    x_dense = data.x

        if x_num is None or x_cat is None:
            raise ValueError("TGCF-IDS requires x_num and x_cat inputs.")
        if edge_index is None or edge_attr is None:
            raise ValueError("TGCF-IDS requires edge_index and edge_attr inputs.")

        # Derive x_dense if not explicitly supplied
        if x_dense is None:
            x_dense = x_num

        # -------------------------------------------------------------
        # Branch 1: Feature Tokenizer + Transformer
        # -------------------------------------------------------------
        tokens = self.tokenizer(x_num=x_num, x_cat=x_cat)           # [N, 42, token_dim]
        tf_out = self.transformer(tokens, return_attention=return_attention)
        h_feat = tf_out.flow_repr                                     # [N, token_dim]

        # -------------------------------------------------------------
        # Branch 2: Edge-Aware Temporal GraphSAGE
        # -------------------------------------------------------------
        h_graph = self.graph_encoder(
            x=x_dense,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )                                                             # [N, graph_out_channels]

        # -------------------------------------------------------------
        # Cross-Modal Fusion
        # -------------------------------------------------------------
        fusion_out = self.fusion(h_feat=h_feat, h_graph=h_graph)     # [N, fusion_dim]
        h_fused = fusion_out.fused_repr

        # -------------------------------------------------------------
        # 10-Class Classification Head
        # -------------------------------------------------------------
        logits = self.classifier(h_fused)                             # [N, 10]

        return TGCFIDSOutput(
            logits=logits,
            feature_repr=h_feat,
            graph_repr=h_graph,
            fused_repr=h_fused,
            gate_values=fusion_out.gate_values,
            feature_contrib=fusion_out.feature_contrib,
            graph_contrib=fusion_out.graph_contrib,
            attention_weights=tf_out.attention_weights,
        )

    def load_pretrained_graph_encoder(
        self,
        checkpoint_path: Union[str, Path],
        freeze: bool = False,
    ) -> None:
        """
        Load self-supervised contrastively pretrained weights into Temporal GraphSAGE branch.

        Args:
            checkpoint_path: Path to graph_pretrained.pt checkpoint.
            freeze: Whether to freeze GraphSAGE weights during initial supervised training.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found at: {path}")

        ckpt = torch.load(path, weights_only=False, map_location=next(self.parameters()).device)
        if "encoder_state_dict" in ckpt:
            self.graph_encoder.load_state_dict(ckpt["encoder_state_dict"])
            print(f"[+] Successfully loaded pretrained encoder weights from {path}")
        elif "model_state_dict" in ckpt:
            # Extract encoder sub-keys
            encoder_keys = {
                k.replace("encoder.", ""): v
                for k, v in ckpt["model_state_dict"].items()
                if k.startswith("encoder.")
            }
            self.graph_encoder.load_state_dict(encoder_keys)
            print(f"[+] Successfully loaded pretrained encoder state from {path}")
        else:
            raise KeyError("Checkpoint missing 'encoder_state_dict' or 'model_state_dict'.")

        if freeze:
            for param in self.graph_encoder.parameters():
                param.requires_grad = False
            print("[*] GraphSAGE branch parameters FROZEN.")

    @classmethod
    def from_metadata_file(
        cls,
        metadata_path: Union[str, Path],
        token_dim: int = 64,
        fusion_dim: int = 128,
        **kwargs,
    ) -> "TGCFIDS":
        """
        Factory method to construct TGCF-IDS directly from preprocessing metadata.json.
        """
        path = Path(metadata_path)
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        num_numerical = meta.get("numerical_features", {}).get("count", 39)
        cat_cards = meta.get("categorical_features", {}).get("cardinalities_with_unk", {})
        graph_in_channels = meta.get("tensor_shapes", {}).get("x_dense_dim", 194)

        return cls(
            num_numerical=num_numerical,
            cat_cardinalities=cat_cards,
            token_dim=token_dim,
            graph_in_channels=graph_in_channels,
            fusion_dim=fusion_dim,
            **kwargs,
        )

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Compute total and trainable parameter counts across all sub-modules.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        tokenizer_params = sum(p.numel() for p in self.tokenizer.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        graph_params = sum(p.numel() for p in self.graph_encoder.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "submodule_parameters": {
                "FeatureTokenizer": tokenizer_params,
                "FeatureTransformer": transformer_params,
                "TemporalGraphSAGE": graph_params,
                "CrossModalFusion": fusion_params,
                "IntrusionClassifier": classifier_params,
            },
        }

    def print_summary(self) -> None:
        """Print clean formatted parameter breakdown."""
        summary = self.get_model_summary()
        print("=" * 75)
        print("                 TGCF-IDS : Model Architecture Summary                   ")
        print("=" * 75)
        print(f"{'Module / Sub-Network':<35} | {'Parameters':<18} | {'% of Total':<12}")
        print("-" * 75)
        for name, count in summary["submodule_parameters"].items():
            pct = (count / summary["total_parameters"]) * 100.0 if summary["total_parameters"] > 0 else 0.0
            print(f"{name:<35} | {count:>15,}  | {pct:>10.2f}%")
        print("-" * 75)
        print(f"{'Total Parameters':<35} | {summary['total_parameters']:>15,}")
        print(f"{'Trainable Parameters':<35} | {summary['trainable_parameters']:>15,}")
        print(f"{'Non-Trainable Parameters':<35} | {summary['non_trainable_parameters']:>15,}")
        print("=" * 75)
