#!/usr/bin/env python3
"""
TGCF-IDS Incremental Temporal Graph Engine.
Constructs and maintains bounded spatial-temporal graph snapshots.
"""

import math
import time
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

from production.graph.node_manager import NodeManager, EdgeManager

logger = logging.getLogger("production.graph.temporal_graph")


class TemporalGraphEngine:
    """
    Stateful incremental graph builder for streaming real-time flows.
    """

    def __init__(
        self,
        duration_seconds: float = 60.0,
        max_nodes: int = 2000,
        max_edges: int = 20000,
        node_ttl_seconds: float = 120.0,
    ):
        self.window_duration = duration_seconds
        self.node_manager = NodeManager(node_ttl_seconds=node_ttl_seconds, max_nodes=max_nodes)
        self.edge_manager = EdgeManager(max_edges=max_edges)
        self._last_eviction = time.time()

    def _extract_6d_edge_features(self, flow: Dict[str, Any]) -> np.ndarray:
        """Extracts and normalizes 6D edge attributes: [dur, sbytes, dbytes, spkts, dpkts, rate]."""
        dur = float(flow.get("dur", 0.0))
        sbytes = float(flow.get("sbytes", 0.0))
        dbytes = float(flow.get("dbytes", 0.0))
        spkts = float(flow.get("spkts", 0.0))
        dpkts = float(flow.get("dpkts", 0.0))
        rate = float(flow.get("rate", 0.0))

        # Log1p transformation
        raw_vec = np.array([
            math.log1p(max(0.0, dur)),
            math.log1p(max(0.0, sbytes)),
            math.log1p(max(0.0, dbytes)),
            math.log1p(max(0.0, spkts)),
            math.log1p(max(0.0, dpkts)),
            math.log1p(max(0.0, rate)),
        ], dtype=np.float32)
        return raw_vec

    def ingest_flow(self, flow: Dict[str, Any], dense_feat: Optional[np.ndarray] = None):
        """Ingests a finalized flow into the current graph state."""
        ts = float(flow.get("timestamp", time.time()))
        src_ip = str(flow.get("src_ip", flow.get("saddr", "127.0.0.1")))
        dst_ip = str(flow.get("dst_ip", flow.get("daddr", "127.0.0.1")))

        self.node_manager.get_or_create_node(src_ip, dense_feat=dense_feat, timestamp=ts)
        self.node_manager.get_or_create_node(dst_ip, dense_feat=dense_feat, timestamp=ts)

        edge_attr = self._extract_6d_edge_features(flow)
        self.edge_manager.add_or_update_edge(src_ip, dst_ip, edge_attr, timestamp=ts)

        # Periodic memory cleanup
        if (time.time() - self._last_eviction) > 10.0:
            evicted_indices = self.node_manager.evict_expired_nodes(now=time.time())
            self._last_eviction = time.time()

    def get_graph_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Exports the current active snapshot:
        Returns:
            x_dense: [N_nodes, 194]
            edge_index: [2, E]
            edge_attr: [E, 6]
        """
        x_dense, compact_mapping = self.node_manager.export_node_matrix()
        edge_index, edge_attr = self.edge_manager.export_edge_tensors(compact_mapping)
        return x_dense, edge_index, edge_attr

    def get_stats(self) -> Dict[str, Any]:
        return {
            "active_nodes": len(self.node_manager.ip_to_idx),
            "active_edges": len(self.edge_manager.edges),
            "max_nodes": self.node_manager.max_nodes,
            "max_edges": self.edge_manager.max_edges,
        }
