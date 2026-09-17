"""Tests for Detection and Alert Engine."""

import pytest
import time
from production.detection.alert_engine import AlertEngine, IntrusionAlert


def test_alert_generation():
    """Test generating an alert from an attack prediction dictionary."""
    engine = AlertEngine(deduplication_window_seconds=10.0)
    
    pred_dict = {
        "event_id": "test-event-1",
        "timestamp": time.time(),
        "flow_id": "flow-100",
        "src_ip": "192.168.1.100",
        "dst_ip": "10.0.0.1",
        "src_port": 4444,
        "dst_port": 80,
        "protocol": "TCP",
        "predicted_class": "Exploits",
        "class_id": 4,
        "confidence": 0.92,
        "confidence_percent": "92.00%",
        "is_malicious": True,
        "verdict": "ATTACK ALERT",
        "severity": "HIGH",
        "model_version": "v1"
    }
    
    alert = engine.process_prediction(pred_dict)
    assert alert is not None
    assert alert.attack_type == "Exploits"
    assert alert.severity == "HIGH"
    assert alert.confidence == 0.92
    assert alert.src_ip == "192.168.1.100"
    assert alert.dst_ip == "10.0.0.1"


def test_normal_traffic_no_alert():
    """Test that Normal traffic does not trigger an alert."""
    engine = AlertEngine()
    
    pred_dict = {
        "event_id": "test-normal-1",
        "timestamp": time.time(),
        "src_ip": "192.168.1.50",
        "dst_ip": "10.0.0.1",
        "predicted_class": "Normal",
        "is_malicious": False,
        "confidence": 0.99
    }
    
    alert = engine.process_prediction(pred_dict)
    assert alert is None


def test_alert_deduplication():
    """Test that rapid duplicate alerts for the same source/dest/attack are deduplicated."""
    engine = AlertEngine(deduplication_window_seconds=5.0)
    
    pred_dict = {
        "event_id": "test-event-2",
        "timestamp": 1000.0,
        "src_ip": "192.168.1.200",
        "dst_ip": "10.0.0.5",
        "src_port": 5000,
        "dst_port": 22,
        "protocol": "TCP",
        "predicted_class": "Generic",
        "class_id": 6,
        "confidence": 0.85,
        "confidence_percent": "85.00%",
        "is_malicious": True,
        "severity": "MEDIUM"
    }
    
    # First alert should be created
    alert1 = engine.process_prediction(pred_dict)
    assert alert1 is not None
    assert alert1.event_count == 1
    
    # Immediate second identical alert at 1001.0 should be aggregated into existing incident
    pred_dict2 = pred_dict.copy()
    pred_dict2["timestamp"] = 1001.0
    alert2 = engine.process_prediction(pred_dict2)
    assert alert2 is None  # Suppressed due to deduplication
    
    # Check that the existing incident count increased
    key = ("192.168.1.200", "10.0.0.5", "Generic")
    assert engine.active_incidents[key].event_count == 2
