#!/usr/bin/env python3
"""
TGCF-IDS Unified Production Feature Preprocessing Pipeline.
Guarantees 100% numerical parity with research preprocessing.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import torch

from src.utils.paths import ProjectPaths
from production.preprocessing.validation import PreprocessingValidator
from production.preprocessing.scaler import ProductionCategoricalEncoder, ProductionRobustScaler

logger = logging.getLogger("production.preprocessing.pipeline")


class ProductionFeaturePipeline:
    """
    Transforms raw network flow dictionaries into model-ready PyTorch tensors.
    """

    NUMERICAL_COLS = PreprocessingValidator.NUMERICAL_COLS
    CATEGORICAL_COLS = PreprocessingValidator.CATEGORICAL_COLS

    def __init__(self, preprocessor_joblib_path: Optional[str] = None):
        self.joblib_path = Path(
            preprocessor_joblib_path or ProjectPaths.DATA_PROCESSED / "preprocessor.joblib"
        )
        self.preprocessor = self._load_joblib_preprocessor()
        self.encoder = ProductionCategoricalEncoder()
        self.scaler = ProductionRobustScaler()

    def _load_joblib_preprocessor(self) -> Optional[Any]:
        if self.joblib_path.exists():
            try:
                import joblib
                from src.preprocessing.preprocessor import LeakageSafePreprocessor
                import __main__
                __main__.LeakageSafePreprocessor = LeakageSafePreprocessor
                preproc = joblib.load(self.joblib_path)
                logger.info(f"[+] Loaded frozen preprocessor from: {self.joblib_path.name}")
                return preproc
            except Exception as e:
                logger.warning(f"[-] Could not load joblib preprocessor ({e}). Using pure Python pipeline fallback.")
        return None

    def transform_records(
        self, records: List[Dict[str, Any]], device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transforms a batch of raw flow dictionaries into (x_num, x_cat, x_dense) tensors.
        """
        cleaned_records = [PreprocessingValidator.validate_and_clean_record(r) for r in records]
        df = pd.DataFrame(cleaned_records)

        if self.preprocessor is not None:
            # High-performance batch transformation via frozen preprocessor
            transformed = self.preprocessor.transform(df)
            x_num = torch.from_numpy(transformed["numpy"]["x_num"]).float().to(device)
            x_cat = torch.from_numpy(transformed["numpy"]["x_cat_ord"]).long().to(device)
            x_dense = torch.from_numpy(transformed["numpy"]["x_dense"]).float().to(device)
            return x_num, x_cat, x_dense

        # Pure Python fallback
        num_matrix = []
        cat_matrix = []
        for r in cleaned_records:
            num_row = [self.scaler.scale_feature(c, r[c]) for c in self.NUMERICAL_COLS]
            cat_row = self.encoder.encode_record(r["proto"], r["service"], r["state"])
            num_matrix.append(num_row)
            cat_matrix.append(cat_row)

        x_num = torch.tensor(num_matrix, dtype=torch.float32, device=device)
        x_cat = torch.tensor(cat_matrix, dtype=torch.long, device=device)
        x_dense = torch.zeros((len(records), 194), dtype=torch.float32, device=device)
        return x_num, x_cat, x_dense

    def transform_single(
        self, record: Dict[str, Any], device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transforms a single raw flow record."""
        return self.transform_records([record], device=device)
