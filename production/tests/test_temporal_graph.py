"""Tests for Incremental Temporal Graph Engine."""

import pytest
import torch
import time

from production.graph.temporal_graph import TemporalGraphEngine


def test_graph_engine_initialization():
    """Test initializing temporal graph engine."""
    engine = TemporalGraphEngine(duration_seconds=60.0, max_nodes=100, max_edges=500)
    assert engine is not None
    stats = engine.get_stats()
    assert stats["active_nodes"] == 0
    assert stats["active_edges"] == 0


def test_graph_snapshot_generation(sample_flow_dict):
    """Test that adding flows updates graph and generates valid PyTorch tensors."""
    engine = TemporalGraphEngine(duration_seconds=60.0)
    
    # Ingest sample flow
    engine.ingest_flow(sample_flow_dict)
    
    stats = engine.get_stats()
    assert stats["active_nodes"] == 2  # src_ip and dst_ip
    assert stats["active_edges"] == 1  # 1 directed edge
    
    # Generate snapshot tensors
    x_dense, edge_index, edge_attr = engine.get_graph_tensors()
    
    assert isinstance(x_dense, torch.Tensor)
    assert isinstance(edge_index, torch.Tensor)
    assert isinstance(edge_attr, torch.Tensor)
    
    assert x_dense.shape[0] == 2
    assert x_dense.shape[1] == 194  # 194-dim node feature matrix
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == 1
    assert edge_attr.shape == (1, 6)  # 6-dim edge features


def test_graph_ttl_eviction(sample_flow_dict):
    """Test that stale nodes are pruned after TTL expiration."""
    engine = TemporalGraphEngine(node_ttl_seconds=1.0)
    
    flow1 = sample_flow_dict.copy()
    flow1["timestamp"] = 100.0
    engine.ingest_flow(flow1)
    assert engine.get_stats()["active_nodes"] == 2
    
    # Advance time beyond TTL
    engine.node_manager.evict_expired_nodes(now=105.0)
    assert engine.get_stats()["active_nodes"] == 0
