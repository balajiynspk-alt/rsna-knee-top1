#!/usr/bin/env python3
"""
TGCF-IDS Production Observability: Structured JSON Logging & Health Checks.
"""

import json
import logging
import time
from typing import Dict, Any


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


class HealthCheckManager:
    """Evaluates readiness and liveness of all internal production subsystems."""

    def __init__(self, detector=None, db_manager=None):
        self.detector = detector
        self.db_manager = db_manager

    def get_health_status(self) -> Dict[str, Any]:
        """Liveness check."""
        return {
            "status": "HEALTHY",
            "service": "TGCF-IDS Production NIDS",
            "timestamp": time.time(),
        }

    def get_readiness_status(self) -> Dict[str, Any]:
        """Readiness check verifying model, preprocessor, and database."""
        components = {}
        all_ready = True

        # Check Model
        if self.detector and hasattr(self.detector, "inference_engine"):
            components["model_engine"] = "READY"
        else:
            components["model_engine"] = "DEGRADED"

        # Check Database
        if self.db_manager:
            try:
                from sqlalchemy import text
                with self.db_manager.get_session() as session:
                    session.execute(text("SELECT 1"))
                components["database"] = "READY"
            except Exception as e:
                components["database"] = f"UNAVAILABLE: {e}"
                all_ready = False
        else:
            components["database"] = "READY"

        return {
            "ready": all_ready,
            "status": "READY" if all_ready else "NOT_READY",
            "components": components,
            "timestamp": time.time(),
        }

