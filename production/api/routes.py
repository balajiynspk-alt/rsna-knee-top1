#!/usr/bin/env python3
"""
TGCF-IDS Production FastAPI REST Routes and WebSocket Event Stream.
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel

from production.model.schema import RawFlowRecord, PredictionResult, BatchPredictionResult


class ConnectionManager:
    """Manages active real-time WebSocket clients."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)
        for dead in dead_connections:
            self.disconnect(dead)


ws_manager = ConnectionManager()
api_router = APIRouter()


def register_routes(app, detector, repo, health_mgr):
    """Binds detector and repository instances to API endpoints."""

    @api_router.get("/health")
    def health():
        return health_mgr.get_health_status()

    @api_router.get("/ready")
    def ready():
        return health_mgr.get_readiness_status()

    @api_router.get("/model")
    def model_info():
        return {
            "model_name": "TGCF-IDS",
            "model_version": detector.inference_engine.loader.MODEL_VERSION,
            "architecture": "Dual-Branch Temporal Graph Contrastive Feature-Transformer",
            "checkpoint": detector.inference_engine.loader.checkpoint_path.name,
            "device": str(detector.inference_engine.device),
            "num_classes": 10,
            "classes": detector.inference_engine.adapter.CLASS_NAMES,
            "performance_stats": detector.inference_engine.get_performance_stats(),
        }

    @api_router.post("/predict/flow", response_model=PredictionResult)
    def predict_single_flow(flow: RawFlowRecord):
        flow_dict = flow.model_dump()
        pred, alert = detector.process_flow_record(flow_dict)
        if repo:
            repo.save_prediction(pred)
            if alert:
                repo.save_alert(alert)
        return pred

    @api_router.post("/predict/batch", response_model=List[PredictionResult])
    def predict_batch_flows(flows: List[RawFlowRecord]):
        flow_dicts = [f.model_dump() for f in flows]
        results = detector.process_flow_batch(flow_dicts)
        preds = []
        for pred, alert in results:
            preds.append(pred)
            if repo:
                repo.save_prediction(pred)
                if alert:
                    repo.save_alert(alert)
        return preds

    @api_router.get("/alerts")
    def get_alerts(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
        if repo:
            return repo.get_recent_alerts(limit=limit, offset=offset)
        return []

    @api_router.get("/statistics")
    def get_stats():
        stats = repo.get_statistics() if repo else {"total_flows": 0, "attack_flows": 0, "normal_flows": 0, "total_alerts": 0}
        stats["graph_state"] = detector.graph_engine.get_stats()
        stats["inference"] = detector.inference_engine.get_performance_stats()
        return stats

    app.include_router(api_router, prefix="/api/v1")
