#!/usr/bin/env python3
"""
TGCF-IDS Production Categorical Encoder and Numerical Scaler Wrappers.
Consumes frozen research preprocessing artifacts without modifying research files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

from src.utils.paths import ProjectPaths

logger = logging.getLogger("production.preprocessing.encoders")


class ProductionCategoricalEncoder:
    """
    Ordinal categorical encoder using exact frozen research vocabularies with <UNK> (0) fallback.
    """

    def __init__(self, metadata_path: Optional[str] = None):
        self.metadata_path = Path(metadata_path or ProjectPaths.DATA_PROCESSED / "metadata.json")
        self.vocabularies = self._load_vocabularies()
        self.cardinalities = [
            len(self.vocabularies.get("proto", {})) + 2,
            len(self.vocabularies.get("service", {})) + 2,
            len(self.vocabularies.get("state", {})) + 2,
        ]

    def _load_vocabularies(self) -> Dict[str, Dict[str, int]]:
        if not self.metadata_path.exists():
            logger.warning(f"Metadata manifest not found at {self.metadata_path}. Using fallback vocabularies.")
            return {"proto": {}, "service": {}, "state": {}}
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            return meta.get("categorical_features", {}).get("vocabularies", {})

    def encode_record(self, proto: str, service: str, state: str) -> List[int]:
        """Encodes proto, service, state into integer tokens with <UNK>=0 fallback."""
        p_idx = self.vocabularies.get("proto", {}).get(proto.lower(), 0)
        s_idx = self.vocabularies.get("service", {}).get(service.lower(), 0)
        st_idx = self.vocabularies.get("state", {}).get(state.upper(), self.vocabularies.get("state", {}).get(state.lower(), 0))
        return [p_idx, s_idx, st_idx]


class ProductionRobustScaler:
    """
    Applies exact RobustScaler IQR formula from training split: (x - Q2) / (Q3 - Q1).
    """

    def __init__(self, metadata_path: Optional[str] = None):
        self.metadata_path = Path(metadata_path or ProjectPaths.DATA_PROCESSED / "metadata.json")
        self.centers, self.scales = self._load_scaling_parameters()

    def _load_scaling_parameters(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        centers = {}
        scales = {}
        if not self.metadata_path.exists():
            return centers, scales
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            stats = meta.get("numerical_features", {}).get("statistics", {})
            for col, stat_dict in stats.items():
                centers[col] = float(stat_dict.get("center", stat_dict.get("q2", 0.0)))
                scale = float(stat_dict.get("scale", stat_dict.get("iqr", 1.0)))
                scales[col] = scale if scale != 0 and not np.isnan(scale) else 1.0
        return centers, scales

    def scale_feature(self, col: str, val: float) -> float:
        center = self.centers.get(col, 0.0)
        scale = self.scales.get(col, 1.0)
        return (val - center) / scale
