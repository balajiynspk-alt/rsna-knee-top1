#!/usr/bin/env python3
"""
TGCF-IDS Alert Generation, Deduplication, and Incident Correlation Engine.
"""

import time
import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from pydantic import BaseModel, Field

logger = logging.getLogger("production.detection.alerts")


class IntrusionAlert(BaseModel):
    """Normalized security intrusion alert model."""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    event_id: str
    flow_id: Optional[str] = None
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    attack_type: str
    confidence: float
    confidence_percent: str
    severity: str          # "INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"
    status: str = "ACTIVE" # "ACTIVE", "ACKNOWLEDGED", "RESOLVED", "FALSE_POSITIVE"
    incident_id: Optional[str] = None
    event_count: int = 1
    first_seen: float = Field(default_factory=time.time)
    last_seen: float = Field(default_factory=time.time)
    model_version: str = "TGCF-IDS-production-compatible-v1"


class AlertEngine:
    """
    Deduplicates repeated attack alerts and groups related events into security incidents.
    """

    def __init__(self, deduplication_window_seconds: float = 60.0):
        self.dedup_window = deduplication_window_seconds
        self.active_incidents: Dict[Tuple[str, str, str], IntrusionAlert] = {}
        self.generated_alerts: List[IntrusionAlert] = []

    def process_prediction(self, pred_dict: Dict[str, Any]) -> Optional[IntrusionAlert]:
        """Evaluates prediction and emits a deduplicated/aggregated alert if malicious."""
        if not pred_dict.get("is_malicious", False):
            return None

        src_ip = str(pred_dict.get("src_ip", "127.0.0.1"))
        dst_ip = str(pred_dict.get("dst_ip", "127.0.0.1"))
        attack_type = str(pred_dict.get("predicted_class", "Attack"))
        ts = float(pred_dict.get("timestamp", time.time()))

        dedup_key = (src_ip, dst_ip, attack_type)

        if dedup_key in self.active_incidents:
            existing_alert = self.active_incidents[dedup_key]
            # If within deduplication window, aggregate into existing incident
            if (ts - existing_alert.last_seen) <= self.dedup_window:
                existing_alert.event_count += 1
                existing_alert.last_seen = ts
                existing_alert.confidence = max(existing_alert.confidence, float(pred_dict.get("confidence", 0.0)))
                existing_alert.confidence_percent = f"{existing_alert.confidence * 100:.2f}%"
                return None  # Aggregated, don't spam new alert

        # Generate new alert
        alert = IntrusionAlert(
            event_id=str(pred_dict.get("event_id", uuid.uuid4())),
            flow_id=str(pred_dict.get("flow_id", "")),
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=int(pred_dict.get("src_port", 0)),
            dst_port=int(pred_dict.get("dst_port", 0)),
            protocol=str(pred_dict.get("protocol", "TCP")),
            attack_type=attack_type,
            confidence=float(pred_dict.get("confidence", 0.0)),
            confidence_percent=str(pred_dict.get("confidence_percent", "0%")),
            severity=str(pred_dict.get("severity", "HIGH")),
            incident_id=f"INC-{uuid.uuid4().hex[:8].upper()}",
            first_seen=ts,
            last_seen=ts,
            model_version=str(pred_dict.get("model_version", "TGCF-IDS-production-compatible-v1")),
        )

        self.active_incidents[dedup_key] = alert
        self.generated_alerts.append(alert)
        return alert

    def sweep_expired_incidents(self, current_time: Optional[float] = None):
        """Cleans up inactive incidents beyond the aggregation window."""
        now = current_time or time.time()
        expired = [k for k, alert in self.active_incidents.items() if (now - alert.last_seen) > self.dedup_window]
        for k in expired:
            del self.active_incidents[k]
