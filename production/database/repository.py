#!/usr/bin/env python3
"""
TGCF-IDS Production Repository Layer (CRUD Operations).
"""

from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc

from production.database.models import FlowEventModel, AlertRecordModel, SystemMetricLogModel
from production.model.schema import PredictionResult
from production.detection.alert_engine import IntrusionAlert


class ProductionRepository:
    """Repository handling persistence of flows, alerts, and metrics."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def save_prediction(self, pred: PredictionResult):
        with self.session_factory() as session:
            rec = FlowEventModel(
                id=pred.event_id,
                timestamp=pred.timestamp,
                src_ip=pred.src_ip,
                dst_ip=pred.dst_ip,
                src_port=pred.src_port,
                dst_port=pred.dst_port,
                protocol=pred.protocol,
                predicted_class=pred.predicted_class,
                confidence=pred.confidence,
                is_malicious=pred.is_malicious,
                verdict=pred.verdict,
                severity=pred.severity,
                latency_ms=pred.inference_latency_ms,
            )
            session.add(rec)
            session.commit()

    def save_alert(self, alert: IntrusionAlert):
        with self.session_factory() as session:
            existing = session.query(AlertRecordModel).filter_by(alert_id=alert.alert_id).first()
            if existing:
                existing.event_count = alert.event_count
                existing.last_seen = alert.last_seen
                existing.confidence = alert.confidence
                existing.confidence_percent = alert.confidence_percent
            else:
                rec = AlertRecordModel(
                    alert_id=alert.alert_id,
                    timestamp=alert.timestamp,
                    incident_id=alert.incident_id,
                    src_ip=alert.src_ip,
                    dst_ip=alert.dst_ip,
                    src_port=alert.src_port,
                    dst_port=alert.dst_port,
                    protocol=alert.protocol,
                    attack_type=alert.attack_type,
                    confidence=alert.confidence,
                    confidence_percent=alert.confidence_percent,
                    severity=alert.severity,
                    status=alert.status,
                    event_count=alert.event_count,
                    first_seen=alert.first_seen,
                    last_seen=alert.last_seen,
                    model_version=alert.model_version,
                )
                session.add(rec)
            session.commit()

    def get_recent_alerts(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with self.session_factory() as session:
            alerts = (
                session.query(AlertRecordModel)
                .order_by(desc(AlertRecordModel.last_seen))
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [
                {
                    "alert_id": a.alert_id,
                    "timestamp": a.timestamp,
                    "incident_id": a.incident_id,
                    "src_ip": a.src_ip,
                    "dst_ip": a.dst_ip,
                    "src_port": a.src_port,
                    "dst_port": a.dst_port,
                    "protocol": a.protocol,
                    "attack_type": a.attack_type,
                    "confidence_percent": a.confidence_percent,
                    "severity": a.severity,
                    "status": a.status,
                    "event_count": a.event_count,
                    "first_seen": a.first_seen,
                    "last_seen": a.last_seen,
                }
                for a in alerts
            ]

    def get_statistics(self) -> Dict[str, Any]:
        with self.session_factory() as session:
            total_flows = session.query(FlowEventModel).count()
            attack_flows = session.query(FlowEventModel).filter_by(is_malicious=True).count()
            normal_flows = total_flows - attack_flows
            total_alerts = session.query(AlertRecordModel).count()
            return {
                "total_flows": total_flows,
                "attack_flows": attack_flows,
                "normal_flows": normal_flows,
                "total_alerts": total_alerts,
            }
