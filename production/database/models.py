#!/usr/bin/env python3
"""
TGCF-IDS Production Relational Database Models (SQLAlchemy).
"""

import time
import uuid
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    Text,
    DateTime,
    Index,
    JSON,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class FlowEventModel(Base):
    """Database representation of an analyzed network flow."""
    __tablename__ = "flow_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(Float, index=True, default=time.time)
    src_ip = Column(String(45), index=True, nullable=False)
    dst_ip = Column(String(45), index=True, nullable=False)
    src_port = Column(Integer, nullable=False)
    dst_port = Column(Integer, nullable=False)
    protocol = Column(String(10), nullable=False)
    service = Column(String(20), default="-")
    state = Column(String(10), default="CON")
    duration = Column(Float, default=0.0)
    source_bytes = Column(Float, default=0.0)
    dest_bytes = Column(Float, default=0.0)
    predicted_class = Column(String(30), index=True, nullable=False)
    confidence = Column(Float, nullable=False)
    is_malicious = Column(Boolean, index=True, default=False)
    verdict = Column(String(20), nullable=False)
    severity = Column(String(15), default="INFO")
    latency_ms = Column(Float, default=0.0)


class AlertRecordModel(Base):
    """Database representation of security intrusion alerts."""
    __tablename__ = "intrusion_alerts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    alert_id = Column(String(36), index=True, unique=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(Float, index=True, default=time.time)
    incident_id = Column(String(36), index=True, nullable=True)
    src_ip = Column(String(45), index=True, nullable=False)
    dst_ip = Column(String(45), index=True, nullable=False)
    src_port = Column(Integer, nullable=False)
    dst_port = Column(Integer, nullable=False)
    protocol = Column(String(10), nullable=False)
    attack_type = Column(String(30), index=True, nullable=False)
    confidence = Column(Float, nullable=False)
    confidence_percent = Column(String(10), default="0%")
    severity = Column(String(15), index=True, default="HIGH")
    status = Column(String(20), index=True, default="ACTIVE") # ACTIVE, ACKNOWLEDGED, RESOLVED, FALSE_POSITIVE
    event_count = Column(Integer, default=1)
    first_seen = Column(Float, default=time.time)
    last_seen = Column(Float, default=time.time)
    model_version = Column(String(50), default="TGCF-IDS-production-compatible-v1")


class SystemMetricLogModel(Base):
    """System health, CPU, memory, and throughput telemetry records."""
    __tablename__ = "system_telemetry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(Float, index=True, default=time.time)
    throughput_flows_sec = Column(Float, default=0.0)
    average_latency_ms = Column(Float, default=0.0)
    active_graph_nodes = Column(Integer, default=0)
    active_graph_edges = Column(Integer, default=0)
    cpu_percent = Column(Float, default=0.0)
    ram_used_mb = Column(Float, default=0.0)
    gpu_memory_mb = Column(Float, default=0.0)
