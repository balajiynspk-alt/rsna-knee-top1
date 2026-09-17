"""Tests for Bidirectional Flow Manager."""

import pytest
import time
from production.capture.flow_manager import FlowManager
from production.capture.interface import PacketEvent


def test_flow_aggregation_forward_backward():
    """Test that forward and backward packets are aggregated into a single bidirectional flow."""
    fm = FlowManager(idle_timeout_seconds=5.0, active_timeout_seconds=10.0, max_tracked_flows=100)
    
    # Packet 1: 192.168.1.10:1234 -> 10.0.0.1:80 (TCP, 100 bytes)
    p1 = PacketEvent(
        timestamp=1000.0,
        src_ip="192.168.1.10",
        dst_ip="10.0.0.1",
        src_port=1234,
        dst_port=80,
        protocol="tcp",
        packet_length=100,
        tcp_flags=0x02,  # SYN
        ttl=64
    )
    
    # Packet 2: 10.0.0.1:80 -> 192.168.1.10:1234 (TCP, 200 bytes, SYN-ACK)
    p2 = PacketEvent(
        timestamp=1000.05,
        src_ip="10.0.0.1",
        dst_ip="192.168.1.10",
        src_port=80,
        dst_port=1234,
        protocol="tcp",
        packet_length=200,
        tcp_flags=0x12,  # SYN-ACK
        ttl=58
    )
    
    # Packet 3: 192.168.1.10:1234 -> 10.0.0.1:80 (TCP, 500 bytes, FIN)
    p3 = PacketEvent(
        timestamp=1000.10,
        src_ip="192.168.1.10",
        dst_ip="10.0.0.1",
        src_port=1234,
        dst_port=80,
        protocol="tcp",
        packet_length=500,
        tcp_flags=0x01,  # FIN (triggers flow closure)
        ttl=64
    )
    
    emitted = fm.process_packet(p1)
    assert len(emitted) == 0
    assert len(fm.active_flows) == 1
    
    emitted = fm.process_packet(p2)
    assert len(emitted) == 0
    assert len(fm.active_flows) == 1
    
    # FIN packet emits finalized flow
    emitted = fm.process_packet(p3)
    assert len(emitted) == 1
    
    flow_record = emitted[0]
    assert flow_record["src_ip"] == "192.168.1.10"
    assert flow_record["dst_ip"] == "10.0.0.1"
    assert flow_record["spkts"] == 2
    assert flow_record["dpkts"] == 1
    assert flow_record["sbytes"] == 600
    assert flow_record["dbytes"] == 200
    assert flow_record["proto"] == "tcp"
    assert flow_record["dur"] > 0.0


def test_idle_timeout_expiration():
    """Test that idle flows are expired when idle_timeout is exceeded."""
    fm = FlowManager(idle_timeout_seconds=1.0, active_timeout_seconds=10.0)
    
    p1 = PacketEvent(
        timestamp=100.0,
        src_ip="1.1.1.1",
        dst_ip="2.2.2.2",
        src_port=1000,
        dst_port=80,
        protocol="tcp",
        packet_length=60
    )
    fm.process_packet(p1)
    assert len(fm.active_flows) == 1
    
    # Advance time at timestamp 102.0 (2 seconds later > 1.0 idle timeout)
    expired = fm.sweep_timeouts(current_time=102.0)
    assert len(expired) == 1
    assert len(fm.active_flows) == 0


def test_lru_eviction():
    """Test that oldest flow is evicted when max_tracked_flows is reached."""
    fm = FlowManager(max_tracked_flows=2)
    
    p1 = PacketEvent(timestamp=1.0, src_ip="1.1.1.1", dst_ip="2.2.2.2", src_port=1, dst_port=80, protocol="tcp", packet_length=50)
    p2 = PacketEvent(timestamp=2.0, src_ip="1.1.1.2", dst_ip="2.2.2.2", src_port=2, dst_port=80, protocol="tcp", packet_length=50)
    p3 = PacketEvent(timestamp=3.0, src_ip="1.1.1.3", dst_ip="2.2.2.2", src_port=3, dst_port=80, protocol="tcp", packet_length=50)
    
    fm.process_packet(p1)
    fm.process_packet(p2)
    assert len(fm.active_flows) == 2
    
    # Adding third packet should evict the least recently used flow (p1)
    evicted = fm.process_packet(p3)
    assert len(evicted) == 1
    assert evicted[0]["src_ip"] == "1.1.1.1"
    assert len(fm.active_flows) == 2
