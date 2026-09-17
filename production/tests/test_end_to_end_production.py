"""End-to-End Integration and API Tests for TGCF-IDS Production System."""

import pytest
from fastapi.testclient import TestClient

from production.api.main import app
from production.detection.detector import RealTimeIntrusionDetector


@pytest.fixture(scope="module")
def api_client():
    """FastAPI test client with active lifespan."""
    with TestClient(app) as client:
        yield client


def test_api_health_endpoint(api_client):
    """Test /api/v1/health endpoint."""
    response = api_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["HEALTHY", "DEGRADED", "UNHEALTHY", "healthy", "operational"]


def test_api_ready_endpoint(api_client):
    """Test /api/v1/ready endpoint."""
    response = api_client.get("/api/v1/ready")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert data["ready"] is True


def test_api_model_info_endpoint(api_client):
    """Test /api/v1/model metadata endpoint."""
    response = api_client.get("/api/v1/model")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "TGCF-IDS"
    assert data["num_classes"] == 10
    assert len(data["classes"]) == 10


def test_api_predict_flow_endpoint(api_client, sample_flow_dict):
    """Test /api/v1/predict/flow REST endpoint with raw flow payload."""
    response = api_client.post("/api/v1/predict/flow", json=sample_flow_dict)
    assert response.status_code == 200
    result = response.json()
    
    assert "class_id" in result
    assert "predicted_class" in result
    assert "confidence" in result
    assert "severity" in result
    assert "inference_latency_ms" in result
    assert result["class_id"] >= 0


def test_api_predict_batch_endpoint(api_client, sample_flow_dict):
    """Test /api/v1/predict/batch REST endpoint with multiple flows."""
    batch = [sample_flow_dict.copy() for _ in range(3)]
    response = api_client.post("/api/v1/predict/batch", json=batch)
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 3


def test_api_statistics_endpoint(api_client):
    """Test /api/v1/statistics endpoint."""
    response = api_client.get("/api/v1/statistics")
    assert response.status_code == 200
    stats = response.json()
    assert "graph_state" in stats
    assert "inference" in stats


def test_detector_end_to_end(sample_flow_dict):
    """Test the full detection pipeline orchestrator with database persistence."""
    detector = RealTimeIntrusionDetector(batch_size=16)
    
    # Process a single raw flow
    pred, alert = detector.process_flow_record(sample_flow_dict)
    
    assert pred is not None
    assert 0 <= pred.class_id < 10
    assert pred.confidence > 0.0
    assert pred.predicted_class in detector.inference_engine.adapter.CLASS_NAMES
