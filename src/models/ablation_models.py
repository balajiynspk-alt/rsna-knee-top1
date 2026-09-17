#!/usr/bin/env python3
"""
TGCF-IDS: Ablation and Baseline Component Architectures.

Provides modular architectures for controlled scientific ablation study:
A. GraphSAGE only (GraphSAGEOnlyClassifier)
B. Transformer only (TransformerOnlyClassifier)
C. GraphSAGE + Transformer (AblationTGCFIDS, concat, no pretrain)
D. Remove temporal edges (AblationTGCFIDS, remove_temporal_edges=True)
E. Remove contrastive pretraining (AblationTGCFIDS, gated, no pretrain)
F. Remove feature tokenizer (AblationTGCFIDS, remove_feature_tokenizer=True)
G. Replace learned fusion with concatenation (AblationTGCFIDS, fusion_strategy='concat', with pretrain)
H. Remove edge attributes (AblationTGCFIDS, remove_edge_attrs=True)
I. Full TGCF-IDS (AblationTGCFIDS / TGCFIDS with all components)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from src.models.feature_tokenizer import FeatureTokenizer
from src.models.feature_transformer import FeatureTransformer
from src.models.temporal_graphsage import TemporalGraphSAGE
from src.models.cross_modal_fusion import CrossModalFusion
from src.models.classifier import IntrusionClassifier


class TransformerOnlyClassifier(nn.Module):
    """
    Transformer-Only Baseline Architecture.
    FeatureTokenizer -> FeatureTransformer (Pre-LN Self-Attention) -> Classifier (10 Logits).
    """

    def __init__(
        self,
        num_numerical: int = 39,
        cat_cardinalities: Optional[Union[List[int], Dict[str, int]]] = None,
        token_dim: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_ffn_dim: int = 256,
        transformer_dropout: float = 0.1,
        transformer_pooling: str = "mean",
        classifier_hidden_dim: int = 128,
        classifier_dropout: float = 0.2,
        num_classes: int = 10,
    ):
        super().__init__()
        if cat_cardinalities is None:
            cat_cardinalities = [134, 14, 10]

        self.tokenizer = FeatureTokenizer(
            num_numerical=num_numerical,
            cat_cardinalities=cat_cardinalities,
            embedding_dim=token_dim,
        )
        self.transformer = FeatureTransformer(
            embedding_dim=token_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            ffn_dim=transformer_ffn_dim,
            dropout=transformer_dropout,
            pooling=transformer_pooling,
        )
        self.classifier = IntrusionClassifier(
            in_features=token_dim,
            hidden_dim=classifier_hidden_dim,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )

    def forward(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
        data: Optional[Union[Data, Batch]] = None,
    ) -> torch.Tensor:
        if data is not None:
            if x_num is None and hasattr(data, "x_num"):
                x_num = data.x_num
            if x_cat is None and hasattr(data, "x_cat"):
                x_cat = data.x_cat
            if (x_num is None or x_cat is None) and hasattr(data, "x"):
                if data.x.shape[-1] >= self.tokenizer.total_features:
                    x_num = data.x[:, :self.tokenizer.num_numerical]
                    x_cat = data.x[:, self.tokenizer.num_numerical:self.tokenizer.total_features]

        if x_num is None or x_cat is None:
            raise ValueError("TransformerOnlyClassifier requires x_num and x_cat.")

        tokens = self.tokenizer(x_num=x_num, x_cat=x_cat)
        tf_out = self.transformer(tokens)
        logits = self.classifier(tf_out.flow_repr)
        return logits


class GraphSAGEOnlyClassifier(nn.Module):
    """
    Temporal GraphSAGE-Only Baseline Architecture.
    Edge-Aware Temporal GraphSAGE -> Classifier (10 Logits).
    """

    def __init__(
        self,
        in_channels: int = 194,
        edge_dim: int = 6,
        hidden_dim: int = 64,
        out_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        aggr: str = "mean",
        classifier_hidden_dim: int = 128,
        classifier_dropout: float = 0.2,
        num_classes: int = 10,
    ):
        super().__init__()
        self.graph_encoder = TemporalGraphSAGE(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            aggr=aggr,
        )
        self.classifier = IntrusionClassifier(
            in_features=out_channels,
            hidden_dim=classifier_hidden_dim,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        data: Optional[Union[Data, Batch]] = None,
    ) -> torch.Tensor:
        if data is not None:
            if x is None and hasattr(data, "x_dense"):
                x = data.x_dense
            elif x is None and hasattr(data, "x_graph"):
                x = data.x_graph
            elif x is None and hasattr(data, "x"):
                x = data.x

            if edge_index is None and hasattr(data, "edge_index"):
                edge_index = data.edge_index
            if edge_attr is None and hasattr(data, "edge_attr"):
                edge_attr = data.edge_attr

        if x is None or edge_index is None or edge_attr is None:
            raise ValueError("GraphSAGEOnlyClassifier requires x, edge_index, and edge_attr.")

        h_graph = self.graph_encoder(x=x, edge_index=edge_index, edge_attr=edge_attr)
        logits = self.classifier(h_graph)
        return logits


class AblationTGCFIDS(nn.Module):
    """
    Highly Configurable Dual-Branch Architecture for Systematic Ablation Experiments.

    Supports systematic removal or modification of individual components:
    - remove_temporal_edges: Zeroes out the temporal interval delta feature on edges.
    - remove_edge_attrs: Zeroes out all edge attributes (standard topology-only SAGEConv).
    - remove_feature_tokenizer: Bypasses per-feature tokenization with a direct linear projection.
    - fusion_strategy: 'gated' (learned dynamic gating) vs 'concat' (direct vector concatenation).
    - pretraining: Load self-supervised contrastive weights into Temporal GraphSAGE encoder.
    """

    def __init__(
        self,
        num_numerical: int = 39,
        cat_cardinalities: Optional[Union[List[int], Dict[str, int]]] = None,
        token_dim: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_ffn_dim: int = 256,
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
        remove_temporal_edges: bool = False,
        remove_edge_attrs: bool = False,
        remove_feature_tokenizer: bool = False,
    ):
        super().__init__()
        if cat_cardinalities is None:
            cat_cardinalities = [134, 14, 10]

        self.num_numerical = num_numerical
        self.token_dim = token_dim
        self.graph_in_channels = graph_in_channels
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes

        self.remove_temporal_edges = remove_temporal_edges
        self.remove_edge_attrs = remove_edge_attrs
        self.remove_feature_tokenizer = remove_feature_tokenizer

        # -------------------------------------------------------------
        # Feature Branch
        # -------------------------------------------------------------
        if not remove_feature_tokenizer:
            self.tokenizer = FeatureTokenizer(
                num_numerical=num_numerical,
                cat_cardinalities=cat_cardinalities,
                embedding_dim=token_dim,
                bias=True,
                use_identity_emb=True,
                dropout=0.0,
            )
            self.linear_projector = None
        else:
            # Ablation: Remove feature tokenizer -> Direct linear projection of dense features
            self.tokenizer = None
            self.linear_projector = nn.Sequential(
                nn.Linear(graph_in_channels, token_dim),
                nn.LayerNorm(token_dim),
                nn.GELU(),
            )

        self.transformer = FeatureTransformer(
            embedding_dim=token_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            ffn_dim=transformer_ffn_dim,
            dropout=transformer_dropout,
            pooling=transformer_pooling,
            use_final_norm=True,
        )

        # -------------------------------------------------------------
        # Graph Branch
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
        # Fusion Engine
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
        # Classifier Head
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
    ) -> torch.Tensor:
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
            elif x_dense is None and hasattr(data, "x"):
                x_dense = data.x

            if (x_num is None or x_cat is None) and hasattr(data, "x"):
                if data.x.shape[-1] >= self.num_numerical + 3:
                    x_num = data.x[:, :self.num_numerical]
                    x_cat = data.x[:, self.num_numerical:self.num_numerical + 3]

        if x_dense is None and x_num is not None:
            x_dense = x_num
        if edge_attr is not None:
            if self.remove_edge_attrs:
                # Ablation H: Remove all edge attributes
                edge_attr = torch.zeros_like(edge_attr)
            elif self.remove_temporal_edges:
                # Ablation D: Remove temporal delta feature (feature index 0)
                edge_attr = edge_attr.clone()
                edge_attr[:, 0] = 0.0

        # Branch 1: Feature Representation
        if not self.remove_feature_tokenizer:
            if x_num is None or x_cat is None:
                raise ValueError("Requires x_num and x_cat when tokenizer is enabled.")
            tokens = self.tokenizer(x_num=x_num, x_cat=x_cat)
            tf_out = self.transformer(tokens)
            h_feat = tf_out.flow_repr
        else:
            # Linear projection of dense vector expanded to sequence of length 1 for transformer
            proj = self.linear_projector(x_dense).unsqueeze(1)  # [N, 1, token_dim]
            tf_out = self.transformer(proj)
            h_feat = tf_out.flow_repr

        # Branch 2: Graph Representation
        h_graph = self.graph_encoder(
            x=x_dense,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # Cross-Modal Fusion
        fusion_out = self.fusion(h_feat=h_feat, h_graph=h_graph)
        h_fused = fusion_out.fused_repr

        # Classifier
        logits = self.classifier(h_fused)
        return logits

    def load_pretrained_graph_encoder(
        self,
        checkpoint_path: Union[str, Path],
        freeze: bool = False,
    ) -> None:
        """Load pretrained contrastive graph encoder weights."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found at: {path}")

        ckpt = torch.load(path, weights_only=False, map_location=next(self.parameters()).device)
        if "encoder_state_dict" in ckpt:
            self.graph_encoder.load_state_dict(ckpt["encoder_state_dict"])
            print(f"[+] Loaded pretrained encoder weights from {path}")
        elif "model_state_dict" in ckpt:
            encoder_keys = {
                k.replace("encoder.", ""): v
                for k, v in ckpt["model_state_dict"].items()
                if k.startswith("encoder.")
            }
            self.graph_encoder.load_state_dict(encoder_keys)
            print(f"[+] Loaded pretrained encoder state from {path}")
        else:
            raise KeyError("Checkpoint missing 'encoder_state_dict' or 'model_state_dict'.")

        if freeze:
            for param in self.graph_encoder.parameters():
                param.requires_grad = False
            print("[*] GraphSAGE branch parameters FROZEN.")
