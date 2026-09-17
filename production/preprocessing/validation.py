#!/usr/bin/env python3
"""
TGCF-IDS Production Preprocessing Validator.
Guarantees clean, non-corrupted feature vectors before inference.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class PreprocessingValidator:
    """
    Validates feature completeness, numerical health, and categorical sanity.
    """

    NUMERICAL_COLS = [
        "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
        "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
        "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
        "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
        "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
        "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
        "is_sm_ips_ports"
    ]

    CATEGORICAL_COLS = ["proto", "service", "state"]

    PROHIBITED_COLS = {"label", "attack_cat", "id", "y_multiclass", "y_binary"}

    @classmethod
    def sanitize_numerical(cls, val: Any, fallback: float = 0.0) -> float:
        """Sanitizes individual numerical values, removing NaNs, Infs, and invalid types."""
        if val is None:
            return fallback
        try:
            fval = float(val)
            if math.isnan(fval) or math.isinf(fval):
                return fallback
            return fval
        except (ValueError, TypeError):
            return fallback

    @classmethod
    def sanitize_categorical(cls, val: Any, fallback: str = "-") -> str:
        """Sanitizes categorical string tokens."""
        if val is None:
            return fallback
        sval = str(val).strip().lower()
        if sval in ("", "none", "nan", "null"):
            return fallback
        return sval

    @classmethod
    def validate_and_clean_record(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates a single flow record and returns a sanitized dictionary
        with all 42 required fields guaranteed to be present and well-typed.
        """
        cleaned = {}

        # 1. Numerical features
        for num_col in cls.NUMERICAL_COLS:
            cleaned[num_col] = cls.sanitize_numerical(record.get(num_col, 0.0))

        # 2. Categorical features
        for cat_col in cls.CATEGORICAL_COLS:
            cleaned[cat_col] = cls.sanitize_categorical(record.get(cat_col, "-"))

        # 3. Preserve metadata without leaking prohibited labels
        for k, v in record.items():
            if k not in cls.PROHIBITED_COLS and k not in cleaned:
                cleaned[k] = v

        return cleaned
