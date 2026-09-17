#!/usr/bin/env python3
"""
TGCF-IDS Production Data Schemas and Validation Contracts.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator
import time


class RawFlowRecord(BaseModel):
    """Raw network flow record as ingested from packet capture, NetFlow, or CSV."""
    flow_id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    src_ip: str = "127.0.0.1"
    dst_ip: str = "127.0.0.1"
    src_port: int = 0
    dst_port: int = 0
    proto: str = "tcp"
    service: str = "-"
    state: str = "CON"
    dur: float = 0.0
    sbytes: float = 0.0
    dbytes: float = 0.0
    spkts: float = 0.0
    dpkts: float = 0.0
    rate: float = 0.0
    sttl: float = 64.0
    dttl: float = 64.0
    sload: float = 0.0
    dload: float = 0.0
    sloss: float = 0.0
    dloss: float = 0.0
    sinpkt: float = 0.0
    dinpkt: float = 0.0
    sjit: float = 0.0
    djit: float = 0.0
    swin: float = 0.0
    stcpb: float = 0.0
    dtcpb: float = 0.0
    dwin: float = 0.0
    tcprtt: float = 0.0
    synack: float = 0.0
    ackdat: float = 0.0
    smean: float = 0.0
    dmean: float = 0.0
    trans_depth: float = 0.0
    response_body_len: float = 0.0
    ct_srv_src: float = 1.0
    ct_state_ttl: float = 0.0
    ct_dst_ltm: float = 1.0
    ct_src_dport_ltm: float = 1.0
    ct_dst_sport_ltm: float = 1.0
    ct_dst_src_ltm: float = 1.0
    is_ftp_login: float = 0.0
    ct_ftp_cmd: float = 0.0
    ct_flw_http_mthd: float = 0.0
    ct_src_ltm: float = 1.0
    ct_srv_dst: float = 1.0
    is_sm_ips_ports: float = 0.0

    model_config = {
        "extra": "ignore"
    }


class PredictionResult(BaseModel):
    """Structured inference result for an individual network flow."""
    event_id: str
    timestamp: float
    flow_id: Optional[str] = None
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    predicted_class: str
    class_id: int
    confidence: float
    confidence_percent: str
    is_malicious: bool
    verdict: str
    severity: str
    probabilities: Dict[str, float]
    inference_latency_ms: float
    model_version: str
    gate_feature_weight: Optional[float] = None
    gate_graph_weight: Optional[float] = None


class BatchPredictionResult(BaseModel):
    """Batch container for multi-flow inference."""
    batch_size: int
    predictions: List[PredictionResult]
    total_latency_ms: float
    throughput_flows_sec: float
    attack_count: int
    normal_count: int
