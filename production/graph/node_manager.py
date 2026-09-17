#!/usr/bin/env python3
"""
TGCF-IDS Production Graph Node and Edge State Managers.
"""

import time
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import torch


class NodeManager:
    """Tracks active IP host nodes with 194-dim dense feature states and TTL expiration."""

    def __init__(self, node_ttl_seconds: float = 120.0, max_nodes: int = 2000):
        self.node_ttl = node_ttl_seconds
        self.max_nodes = max_nodes
        self.ip_to_idx: Dict[str, int] = {}
        self.idx_to_ip: Dict[int, str] = {}
        self.node_last_seen: Dict[int, float] = {}
        self.node_features: Dict[int, np.ndarray] = {}  # 194-dim dense vector
        self._next_idx = 0

    def get_or_create_node(self, ip: str, dense_feat: Optional[np.ndarray] = None, timestamp: Optional[float] = None) -> int:
        now = timestamp or time.time()
        if ip in self.ip_to_idx:
            idx = self.ip_to_idx[ip]
            self.node_last_seen[idx] = now
            if dense_feat is not None:
                self.node_features[idx] = dense_feat
            return idx

        # If at capacity, evict oldest node
        if len(self.ip_to_idx) >= self.max_nodes:
            self.evict_expired_nodes(now=now, force_one=True)

        idx = self._next_idx
        self._next_idx += 1
        self.ip_to_idx[ip] = idx
        self.idx_to_ip[idx] = ip
        self.node_last_seen[idx] = now
        self.node_features[idx] = dense_feat if dense_feat is not None else np.zeros(194, dtype=np.float32)
        return idx

    def evict_expired_nodes(self, now: float, force_one: bool = False) -> Set[int]:
        """Removes nodes exceeding TTL."""
        expired = set()
        if force_one and self.node_last_seen:
            oldest_idx = min(self.node_last_seen.items(), key=lambda x: x[1])[0]
            expired.add(oldest_idx)

        for idx, last_seen in list(self.node_last_seen.items()):
            if (now - last_seen) > self.node_ttl:
                expired.add(idx)

        for idx in expired:
            ip = self.idx_to_ip.pop(idx, None)
            if ip:
                self.ip_to_idx.pop(ip, None)
            self.node_last_seen.pop(idx, None)
            self.node_features.pop(idx, None)

        return expired

    def export_node_matrix(self) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Exports node tensor in index order."""
        if not self.idx_to_ip:
            return torch.zeros((1, 194), dtype=torch.float32), {"127.0.0.1": 0}

        sorted_indices = sorted(self.idx_to_ip.keys())
        compact_mapping = {self.idx_to_ip[old_idx]: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
        feature_matrix = np.array([self.node_features.get(old_idx, np.zeros(194, dtype=np.float32)) for old_idx in sorted_indices], dtype=np.float32)
        return torch.from_numpy(feature_matrix), compact_mapping


class EdgeManager:
    """Tracks directed flow edges with 6D continuous transmission metrics."""

    def __init__(self, max_edges: int = 20000):
        self.max_edges = max_edges
        self.edges: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def add_or_update_edge(self, src_ip: str, dst_ip: str, edge_attr_6d: np.ndarray, timestamp: float):
        key = (src_ip, dst_ip)
        if len(self.edges) >= self.max_edges and key not in self.edges:
            # Remove oldest edge
            oldest_key = min(self.edges.items(), key=lambda x: x[1]["timestamp"])[0]
            del self.edges[oldest_key]

        self.edges[key] = {
            "attr": edge_attr_6d,
            "timestamp": timestamp,
        }

    def remove_nodes(self, evicted_ips: Set[str]):
        """Purges any edges connected to evicted host nodes."""
        to_delete = [k for k in self.edges if k[0] in evicted_ips or k[1] in evicted_ips]
        for k in to_delete:
            del self.edges[k]

    def export_edge_tensors(self, ip_to_compact_idx: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exports edge_index and edge_attr matching compact node indices."""
        src_indices = []
        dst_indices = []
        attrs = []

        for (src_ip, dst_ip), data in self.edges.items():
            if src_ip in ip_to_compact_idx and dst_ip in ip_to_compact_idx:
                src_indices.append(ip_to_compact_idx[src_ip])
                dst_indices.append(ip_to_compact_idx[dst_ip])
                attrs.append(data["attr"])

        if not src_indices:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 6), dtype=torch.float32)

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        edge_attr = torch.tensor(np.array(attrs, dtype=np.float32), dtype=torch.float32)
        return edge_index, edge_attr
