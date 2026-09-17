"""Pytest fixtures for production test suite."""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from production.model.schema import RawFlowRecord
from production.preprocessing.feature_pipeline import ProductionFeaturePipeline
from production.model.model_loader import load_production_model
from production.model.model_adapter import TGCFIDSProductionAdapter


@pytest.fixture
def sample_flow_dict():
    """Returns a valid raw flow dictionary matching UNSW-NB15 schema."""
    return {
        "src_ip": "192.168.1.50",
        "dst_ip": "10.0.0.1",
        "src_port": 54321,
        "dst_port": 80,
        "proto": "tcp",
        "state": "FIN",
        "service": "http",
        "dur": 0.125,
        "sbytes": 1024.0,
        "dbytes": 4096.0,
        "spkts": 10.0,
        "dpkts": 12.0,
        "sttl": 64.0,
        "dttl": 58.0,
        "sloss": 0.0,
        "dloss": 0.0,
        "sload": 65536.0,
        "dload": 262144.0,
        "sinpkt": 12.5,
        "dinpkt": 10.4,
        "sjit": 1.2,
        "djit": 0.8,
        "swin": 255.0,
        "dwin": 255.0,
        "stcpb": 1000000.0,
        "dtcpb": 2000000.0,
        "tcprtt": 0.045,
        "synack": 0.025,
        "ackdat": 0.020,
        "smean": 102.0,
        "dmean": 341.0,
        "trans_depth": 1.0,
        "response_body_len": 2048.0,
        "ct_srv_src": 2.0,
        "ct_state_ttl": 0.0,
        "ct_dst_ltm": 1.0,
        "ct_src_dport_ltm": 1.0,
        "ct_dst_sport_ltm": 1.0,
        "ct_dst_src_ltm": 1.0,
        "is_ftp_login": 0.0,
        "ct_ftp_cmd": 0.0,
        "ct_flw_http_mthd": 1.0,
        "ct_src_ltm": 2.0,
        "ct_srv_dst": 2.0,
        "is_sm_ips_ports": 0.0,
    }


@pytest.fixture
def sample_flow_obj(sample_flow_dict):
    """Returns a validated RawFlowRecord Pydantic object."""
    return RawFlowRecord(**sample_flow_dict)


@pytest.fixture(scope="session")
def feature_pipeline():
    """Initializes the production feature pipeline with frozen preprocessor."""
    pipeline = ProductionFeaturePipeline()
    return pipeline


@pytest.fixture(scope="session")
def model_adapter():
    """Loads the TGCF-IDS production model adapter."""
    model = load_production_model(device="cpu")
    adapter = TGCFIDSProductionAdapter(model=model, device=torch.device("cpu"))
    return adapter
