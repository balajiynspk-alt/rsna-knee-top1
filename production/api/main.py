#!/usr/bin/env python3
"""
TGCF-IDS Production FastAPI Application Server.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from production.detection.detector import RealTimeIntrusionDetector
from production.database.database import DatabaseManager
from production.database.repository import ProductionRepository
from production.monitoring.logging import HealthCheckManager
from production.api.routes import register_routes, ws_manager

logger = logging.getLogger("production.api")

detector = None
db_mgr = None
repo = None
health_mgr = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, db_mgr, repo, health_mgr
    logger.info("[*] Starting TGCF-IDS Production Server...")
    detector = RealTimeIntrusionDetector()
    db_mgr = DatabaseManager()
    repo = ProductionRepository(db_mgr.get_session)
    health_mgr = HealthCheckManager(detector=detector, db_manager=db_mgr)

    # Bridge internal event hub to WebSocket broadcaster
    loop = asyncio.get_event_loop()

    def on_internal_event(event_data):
        asyncio.run_coroutine_threadsafe(ws_manager.broadcast(event_data), loop)

    detector.event_hub.subscribe(on_internal_event)
    register_routes(app, detector, repo, health_mgr)
    logger.info("[+] TGCF-IDS API Server & WebSocket Hub ready.")
    yield
    logger.info("[-] Shutting down TGCF-IDS Production Server...")


app = FastAPI(
    title="TGCF-IDS Production API",
    description="Real-time Network Intrusion Detection using Temporal Graph Contrastive Feature-Transformers",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep alive and receive client heartbeats
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.get("/", response_class=HTMLResponse)
def root_dashboard():
    """Serves pure HTML/JS production status dashboard."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>TGCF-IDS Production NIDS</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 24px; background: #0F172A; color: #F8FAFC; }
            .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 16px; margin-bottom: 24px; }
            .card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
            .card { background: #1E293B; padding: 20px; border-radius: 8px; border: 1px solid #334155; }
            .card h3 { margin: 0 0 8px 0; font-size: 13px; color: #94A3B8; text-transform: uppercase; }
            .card .val { font-size: 28px; font-weight: bold; color: #38BDF8; }
            .feed-card { background: #1E293B; border-radius: 8px; border: 1px solid #334155; padding: 16px; }
            table { width: 100%; border-collapse: collapse; font-size: 13px; }
            th { text-align: left; padding: 10px; color: #94A3B8; border-bottom: 1px solid #334155; }
            td { padding: 10px; border-bottom: 1px solid #1E293B; }
            .badge-normal { background: #065F46; color: #34D399; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
            .badge-attack { background: #991B1B; color: #F87171; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="header">
            <div>
                <h1 style="margin:0; font-size:22px;">TGCF-IDS Production Monitor</h1>
                <span style="color:#10B981; font-size:12px;">● SYSTEM ONLINE | FROZEN RESEARCH CORE v1.0.0</span>
            </div>
            <div>
                <a href="/docs" style="color:#38BDF8; text-decoration:none; font-weight:bold;">Swagger API Docs →</a>
            </div>
        </div>
        <div class="card-grid">
            <div class="card"><h3>Total Analyzed Flows</h3><div class="val" id="cnt-flows">0</div></div>
            <div class="card"><h3>Active Security Alerts</h3><div class="val" style="color:#F87171;" id="cnt-alerts">0</div></div>
            <div class="card"><h3>Inference Throughput</h3><div class="val" id="val-throughput">90,731 /s</div></div>
            <div class="card"><h3>Active Graph Nodes</h3><div class="val" id="val-nodes">0</div></div>
        </div>
        <div class="feed-card">
            <h3 style="margin-top:0; color:#94A3B8;">Real-Time Intrusion Event Feed (WebSocket Live)</h3>
            <table id="event-table">
                <thead>
                    <tr><th>TIME</th><th>FLOW PAIR</th><th>PROTO</th><th>VERDICT</th><th>CATEGORY</th><th>CONFIDENCE</th><th>LATENCY</th></tr>
                </thead>
                <tbody id="event-body">
                    <tr><td colspan="7" style="text-align:center; color:#64748B;">Connecting to live WebSocket stream...</td></tr>
                </tbody>
            </table>
        </div>
        <script>
            let flowCount = 0;
            let alertCount = 0;
            const ws = new WebSocket((window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host + '/ws/events');
            ws.onopen = () => {
                document.getElementById('event-body').innerHTML = '';
            };
            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                if (data.type === 'flow_prediction') {
                    const p = data.prediction;
                    flowCount++;
                    if (p.is_malicious) alertCount++;
                    document.getElementById('cnt-flows').innerText = flowCount;
                    document.getElementById('cnt-alerts').innerText = alertCount;
                    
                    const tr = document.createElement('tr');
                    const badge = p.is_malicious ? '<span class="badge-attack">ATTACK</span>' : '<span class="badge-normal">NORMAL</span>';
                    tr.innerHTML = `<td>${new Date().toLocaleTimeString()}</td><td>${p.src_ip}:${p.src_port} → ${p.dst_ip}:${p.dst_port}</td><td>${p.protocol}</td><td>${badge}</td><td>${p.predicted_class}</td><td>${p.confidence_percent}</td><td>${p.inference_latency_ms.toFixed(2)} ms</td>`;
                    
                    const tbody = document.getElementById('event-body');
                    tbody.insertBefore(tr, tbody.firstChild);
                    if (tbody.children.length > 25) tbody.removeChild(tbody.lastChild);
                }
            };
        </script>
    </body>
    </html>
    """
