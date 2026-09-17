#!/usr/bin/env python3
"""
TGCF-IDS Production Model Adapter & Inference Wrapper.
Isolates the frozen research model from streaming production pipelines.
"""

import time
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn.functional as F

from src.models.tgcf_ids import TGCFIDS, TGCFIDSOutput
from production.model.schema import RawFlowRecord, PredictionResult, BatchPredictionResult


class TGCFIDSProductionAdapter:
    """
    Thread-safe production adapter for TGCF-IDS real-time inference.
    """

    CLASS_NAMES = [
        "Normal",
        "Analysis",
        "Backdoor",
        "DoS",
        "Exploits",
        "Fuzzers",
        "Generic",
        "Reconnaissance",
        "Shellcode",
        "Worms"
    ]

    SEVERITY_MAPPING = {
        "Normal": "INFO",
        "Analysis": "MEDIUM",
        "Backdoor": "CRITICAL",
        "DoS": "HIGH",
        "Exploits": "HIGH",
        "Fuzzers": "MEDIUM",
        "Generic": "MEDIUM",
        "Reconnaissance": "LOW",
        "Shellcode": "CRITICAL",
        "Worms": "CRITICAL",
    }

    def __init__(
        self,
        model: TGCFIDS,
        device: torch.device,
        model_version: str = "TGCF-IDS-production-compatible-v1",
        attack_threshold: float = 0.50,
        alert_threshold: float = 0.70,
    ):
        self.model = model
        self.device = device
        self.model_version = model_version
        self.attack_threshold = attack_threshold
        self.alert_threshold = alert_threshold

    @torch.no_grad()
    def predict_tensor_batch(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        x_dense: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        flow_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[PredictionResult]:
        """
        Executes forward inference on preprocessed tensor batches.
        """
        batch_size = x_num.shape[0]
        x_num = x_num.to(self.device)
        x_cat = x_cat.to(self.device)

        if x_dense is None:
            x_dense = torch.zeros((batch_size, 194), dtype=torch.float32, device=self.device)
        else:
            x_dense = x_dense.to(self.device)

        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        else:
            edge_index = edge_index.to(self.device)

        if edge_attr is None:
            edge_attr = torch.empty((0, 6), dtype=torch.float32, device=self.device)
        else:
            edge_attr = edge_attr.to(self.device)

        t_start = time.perf_counter()
        out: TGCFIDSOutput = self.model(
            x_num=x_num,
            x_cat=x_cat,
            x_dense=x_dense,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        t_total_ms = (time.perf_counter() - t_start) * 1000.0
        per_flow_latency = t_total_ms / max(1, batch_size)

        probs = F.softmax(out.logits, dim=-1).cpu().numpy()
        results = []

        # Extract gate weights if present
        gate_vals = out.gate_values.cpu().numpy() if out.gate_values is not None else None

        for idx in range(batch_size):
            p_vec = probs[idx]
            pred_idx = int(np.argmax(p_vec))
            pred_class = self.CLASS_NAMES[pred_idx]
            confidence = float(p_vec[pred_idx])

            # Operational alert decision
            is_malicious = (pred_idx != 0) and (confidence >= self.attack_threshold)
            verdict = "ATTACK ALERT" if is_malicious else "BENIGN NORMAL"
            severity = self.SEVERITY_MAPPING.get(pred_class, "INFO") if is_malicious else "INFO"

            meta = flow_metadata[idx] if flow_metadata and idx < len(flow_metadata) else {}
            event_id = str(uuid.uuid4())
            ts = float(meta.get("timestamp", time.time()))

            prob_dict = {self.CLASS_NAMES[c]: float(p_vec[c]) for c in range(len(self.CLASS_NAMES))}

            g_feat = float(np.mean(gate_vals[idx])) if gate_vals is not None else 0.5
            g_graph = 1.0 - g_feat

            res = PredictionResult(
                event_id=event_id,
                timestamp=ts,
                flow_id=meta.get("flow_id", f"flow_{idx}"),
                src_ip=str(meta.get("src_ip", meta.get("saddr", "127.0.0.1"))),
                dst_ip=str(meta.get("dst_ip", meta.get("daddr", "127.0.0.1"))),
                src_port=int(meta.get("src_port", meta.get("sport", 0))),
                dst_port=int(meta.get("dst_port", meta.get("dsport", 0))),
                protocol=str(meta.get("proto", "TCP")).upper(),
                predicted_class=pred_class,
                class_id=pred_idx,
                confidence=confidence,
                confidence_percent=f"{confidence * 100:.2f}%",
                is_malicious=is_malicious,
                verdict=verdict,
                severity=severity,
                probabilities=prob_dict,
                inference_latency_ms=per_flow_latency,
                model_version=self.model_version,
                gate_feature_weight=round(g_feat, 4),
                gate_graph_weight=round(g_graph, 4),
            )
            results.append(res)

        return results
