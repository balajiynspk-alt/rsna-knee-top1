#!/usr/bin/env python3
"""
TGCF-IDS High-Throughput Production Inference Runtime.
"""

import logging
import time
from typing import Dict, List, Optional, Any

import pandas as pd
import torch

from production.model.model_adapter import TGCFIDSProductionAdapter
from production.model.model_loader import ProductionModelLoader
from production.model.schema import RawFlowRecord, PredictionResult, BatchPredictionResult

logger = logging.getLogger("production.model.inference")


class ProductionInferenceEngine:
    """
    Manages model execution, micro-batching, and device synchronization.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 64,
        attack_threshold: float = 0.50,
    ):
        self.loader = ProductionModelLoader(checkpoint_path=checkpoint_path, device=device)
        self.model, self.manifest = self.loader.build_and_load_model()
        self.adapter = TGCFIDSProductionAdapter(
            model=self.model,
            device=self.loader.device,
            model_version=self.loader.MODEL_VERSION,
            attack_threshold=attack_threshold,
        )
        self.batch_size = batch_size
        self._total_inferred = 0
        self._total_time_sec = 0.0

    @property
    def device(self) -> torch.device:
        return self.loader.device

    def predict_tensors(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        x_dense: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[PredictionResult]:
        """Runs batched inference on prepared tensors."""
        t_start = time.perf_counter()
        results = self.adapter.predict_tensor_batch(
            x_num=x_num,
            x_cat=x_cat,
            x_dense=x_dense,
            edge_index=edge_index,
            edge_attr=edge_attr,
            flow_metadata=metadata,
        )
        elapsed = time.perf_counter() - t_start
        self._total_inferred += len(results)
        self._total_time_sec += elapsed
        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Returns runtime throughput and cumulative latency statistics."""
        avg_throughput = (
            self._total_inferred / max(1e-6, self._total_time_sec)
            if self._total_time_sec > 0
            else 0.0
        )
        return {
            "total_flows_inferred": self._total_inferred,
            "cumulative_time_sec": round(self._total_time_sec, 4),
            "average_throughput_flows_sec": round(avg_throughput, 2),
            "device": str(self.device),
            "model_version": self.loader.MODEL_VERSION,
        }
