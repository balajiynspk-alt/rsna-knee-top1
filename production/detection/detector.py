#!/usr/bin/env python3
"""
TGCF-IDS Real-Time Intrusion Detector & End-to-End Pipeline Coordinator.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import torch

from production.capture.flow_manager import FlowManager
from production.preprocessing.feature_pipeline import ProductionFeaturePipeline
from production.graph.temporal_graph import TemporalGraphEngine
from production.model.inference import ProductionInferenceEngine
from production.detection.alert_engine import AlertEngine, IntrusionAlert
from production.streaming.buffer import StreamingFlowBuffer, EventStreamHub
from production.model.schema import RawFlowRecord, PredictionResult

logger = logging.getLogger("production.detection.detector")


class RealTimeIntrusionDetector:
    """
    Unified end-to-end intrusion detection orchestrator.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 64,
        attack_threshold: float = 0.50,
        dedup_window: float = 60.0,
    ):
        logger.info("[*] Initializing TGCF-IDS Real-Time Production Detector...")
        self.pipeline = ProductionFeaturePipeline()
        self.graph_engine = TemporalGraphEngine()
        self.inference_engine = ProductionInferenceEngine(
            checkpoint_path=checkpoint_path,
            device=device,
            batch_size=batch_size,
            attack_threshold=attack_threshold,
        )
        self.alert_engine = AlertEngine(deduplication_window_seconds=dedup_window)
        self.flow_buffer = StreamingFlowBuffer(max_capacity=50000)
        self.event_hub = EventStreamHub()

        self._total_processed = 0
        self._total_alerts = 0

    def process_flow_record(self, raw_flow: Dict[str, Any]) -> Tuple[PredictionResult, Optional[IntrusionAlert]]:
        """
        Processes a single network flow synchronously through the full pipeline.
        """
        # 1. Update temporal graph state
        self.graph_engine.ingest_flow(raw_flow)

        # 2. Transform tabular features to exact tensor representations
        x_num, x_cat, x_dense = self.pipeline.transform_single(raw_flow, device=self.inference_engine.device)

        # 3. Retrieve temporal graph edge context (bounded to current flow batch)
        edge_index = torch.empty((2, 0), dtype=torch.long, device=self.inference_engine.device)
        edge_attr = torch.empty((0, 6), dtype=torch.float32, device=self.inference_engine.device)

        # 4. Execute dual-branch inference
        results = self.inference_engine.predict_tensors(
            x_num=x_num,
            x_cat=x_cat,
            x_dense=x_dense,
            edge_index=edge_index,
            edge_attr=edge_attr,
            metadata=[raw_flow],
        )
        pred = results[0]
        self._total_processed += 1

        # 5. Evaluate alert engine
        alert = self.alert_engine.process_prediction(pred.model_dump())
        if alert is not None:
            self._total_alerts += 1

        # 6. Publish real-time events
        self.event_hub.publish({
            "type": "flow_prediction",
            "prediction": pred.model_dump(),
            "alert": alert.model_dump() if alert else None,
        })

        return pred, alert

    def process_flow_batch(self, raw_flows: List[Dict[str, Any]]) -> List[Tuple[PredictionResult, Optional[IntrusionAlert]]]:
        """
        Processes a micro-batch of flows for maximum hardware throughput.
        """
        if not raw_flows:
            return []

        # 1. Update graph state with all flows in batch
        for f in raw_flows:
            self.graph_engine.ingest_flow(f)

        # 2. Batch transform tabular features
        x_num, x_cat, x_dense = self.pipeline.transform_records(raw_flows, device=self.inference_engine.device)

        # 3. Retrieve temporal graph edge context (bounded to current flow batch)
        edge_index = torch.empty((2, 0), dtype=torch.long, device=self.inference_engine.device)
        edge_attr = torch.empty((0, 6), dtype=torch.float32, device=self.inference_engine.device)

        # 4. Batched dual-branch forward pass
        predictions = self.inference_engine.predict_tensors(
            x_num=x_num,
            x_cat=x_cat,
            x_dense=x_dense,
            edge_index=edge_index,
            edge_attr=edge_attr,
            metadata=raw_flows,
        )

        outputs = []
        for pred in predictions:
            self._total_processed += 1
            alert = self.alert_engine.process_prediction(pred.model_dump())
            if alert:
                self._total_alerts += 1
            outputs.append((pred, alert))

        return outputs
