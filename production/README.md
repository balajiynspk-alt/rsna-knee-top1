# TGCF-IDS Real-Time Production System

High-throughput, real-time intrusion detection production engine built around the frozen research model **TGCF-IDS** (*Temporal Graph Contrastive Feature-Transformer Intrusion Detection System*).

---

## 🏛️ Architecture Overview

The production system operates in a completely isolated `production/` hierarchy without modifying any research baselines (`src/`, `configs/`, `data/`, `results/` remain strictly read-only).

```
Network Packets / PCAP
        │
        ▼
┌─────────────────────────┐
│ FlowManager & Capture   │  ◄── Live Socket Sniffer / PCAP Replay / Synthetic Stream
│ (Bidirectional 5-Tuple) │
└───────────┬─────────────┘
            │ Flow Completed / Timed Out
            ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│ Feature Sanitization &  │     │ Incremental Temporal    │
│ Robust Scaling Pipeline │     │ Graph Engine (194D)     │
└───────────┬─────────────┘     └────────────┬────────────┘
            │                                │
            └───────────────┬────────────────┘
                            ▼
            ┌───────────────────────────────┐
            │   TGCF-IDS Production Model   │
            │   - Tabular Transformer       │
            │   - Temporal GraphSAGE        │
            │   - Contrastive Fusion Gate   │
            └───────────────┬───────────────┘
                            ▼
            ┌───────────────────────────────┐
            │ Detection & Alert Engine      │
            │ (Deduplication & Correlation) │
            └───────────────┬───────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│ SQLite/Postgres Events  │     │ FastAPI WebSocket &     │
│ & Telemetry Storage     │     │ Real-Time SOC Dashboard │
└─────────────────────────┘     └─────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Run Production Engine (Web Server + Real-Time Stream)
```powershell
python -m production run --host 0.0.0.0 --port 8000
```
Open your browser at **http://localhost:8000** to view the live Cyber Operations Dashboard with real-time attack detection charts and network topology.

### 2. Validate Research Model Parity & Integrity
```powershell
python -m production validate-model
```
Checks SHA256 integrity (`f6a85511...`) and verifies tensor output shapes and gating values.

### 3. Run Throughput and Latency Benchmarks
```powershell
python -m production benchmark --flows 1000 --batch-size 32
```
Outputs p50, p95, p99 latency metrics and flows-per-second (FPS) throughput.

### 4. Health Check
```powershell
python -m production health
```

---

## 📡 REST API & WebSocket Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web Dashboard (Interactive Cyber Threat Operations Map) |
| `GET` | `/api/v1/health` | Service health status, GPU/CPU metrics, and model state |
| `POST` | `/api/v1/predict` | Synchronous classification of a single network flow record |
| `POST` | `/api/v1/predict/batch` | Micro-batched classification of multiple flows |
| `GET` | `/api/v1/alerts` | Paginated security alerts with severity & IP filters |
| `GET` | `/api/v1/metrics` | Real-time throughput (FPS), alert counts, and queue depth |
| `WS` | `/ws/live-stream` | Real-time WebSocket feed broadcasting live flows & threat alerts |

---

## 🧪 Testing

Execute the complete production test suite:
```powershell
pytest production/tests/ -v
```

---

## 🐳 Docker Deployment

Build and run using Docker Compose:
```bash
docker-compose -f production/docker-compose.yml up --build -d
```
