#!/usr/bin/env python3
"""
TGCF-IDS: Pure Backend / CLI Prediction & Inference Tool.

Allows testing and predicting network flow traffic without any front-end or web UI.

Usage:
1. Demo mode (runs predictions on sample benign and attack flows):
   python scripts/predict.py --demo

2. Predict from a CSV file:
   python scripts/predict.py --csv data/sample_flows.csv

3. Predict from a JSON file:
   python scripts/predict.py --json data/sample_flow.json

4. Interactive terminal mode (prompting values in terminal):
   python scripts/predict.py --interactive
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.paths import ProjectPaths
from src.models.tgcf_ids import TGCFIDS
from src.preprocessing.preprocessor import LeakageSafePreprocessor
import __main__
__main__.LeakageSafePreprocessor = LeakageSafePreprocessor


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


class FlowPredictor:
    """Headless CLI & Python API for TGCF-IDS flow classification."""

    def __init__(self, checkpoint_path: Optional[str] = None, metadata_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path or ProjectPaths.RESULTS_CHECKPOINTS / "final_tgcf_ids_model.pt")
        if not self.checkpoint_path.exists():
            self.checkpoint_path = ProjectPaths.RESULTS_CHECKPOINTS / "best_tgcf_ids.pt"

        self.metadata_path = Path(metadata_path or ProjectPaths.DATA_PROCESSED / "metadata.json")

        # Load preprocessor
        preproc_path = ProjectPaths.DATA_PROCESSED / "preprocessor.joblib"
        if preproc_path.exists():
            import joblib
            self.preprocessor = joblib.load(preproc_path)
        else:
            self.preprocessor = None

        # Build & load model
        self.model = self._load_model()

    def _load_metadata(self) -> dict:
        if not self.metadata_path.exists():
            return {}
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_model(self) -> TGCFIDS:
        model = TGCFIDS(
            num_numerical=39,
            cat_cardinalities=[134, 14, 10],
            token_dim=64,
            transformer_heads=4,
            transformer_layers=2,
            transformer_ffn_dim=256,
            graph_in_channels=194,
            graph_edge_dim=6,
            graph_hidden_dim=64,
            graph_out_channels=64,
            graph_layers=2,
            fusion_dim=128,
            fusion_strategy="gated",
            classifier_hidden_dim=64,
            num_classes=10,
        ).to(self.device)

        if self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)

        model.eval()
        return model

    def preprocess_flow(self, flow_dict: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert raw flow attributes into properly scaled PyTorch tensors."""
        # Ensure all required numerical and categorical columns exist
        df = pd.DataFrame([flow_dict])
        if self.preprocessor is not None:
            for col in self.preprocessor.numerical_cols:
                if col not in df.columns:
                    df[col] = 0.0
            for col in self.preprocessor.categorical_cols:
                if col not in df.columns:
                    df[col] = "-"
            transformed = self.preprocessor.transform(df)
            x_num = torch.from_numpy(transformed["numpy"]["x_num"]).float().to(self.device)
            x_cat = torch.from_numpy(transformed["numpy"]["x_cat_ord"]).long().to(self.device)
            x_dense = torch.from_numpy(transformed["numpy"]["x_dense"]).float().to(self.device)
            return x_num, x_cat, x_dense
        else:
            x_num = torch.zeros((1, 39), dtype=torch.float32, device=self.device)
            x_cat = torch.zeros((1, 3), dtype=torch.long, device=self.device)
            x_dense = torch.zeros((1, 194), dtype=torch.float32, device=self.device)
            return x_num, x_cat, x_dense

    @torch.no_grad()
    def predict_single(self, flow_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Predict the intrusion class for a single network flow record."""
        x_num, x_cat, x_dense = self.preprocess_flow(flow_dict)
        edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        edge_attr = torch.empty((0, 6), dtype=torch.float32, device=self.device)

        out = self.model(
            x_num=x_num,
            x_cat=x_cat,
            x_dense=x_dense,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        logits = out.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Class_{pred_idx}"
        confidence = float(probs[pred_idx])
        is_attack = (pred_idx != 0)

        all_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        return {
            "predicted_class": pred_class,
            "is_malicious": is_attack,
            "verdict": "ATTACK ALERT" if is_attack else "BENIGN NORMAL",
            "confidence": confidence,
            "confidence_percent": f"{confidence * 100:.2f}%",
            "all_class_probabilities": all_probs,
        }

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch predict a pandas DataFrame of flows."""
        results = []
        for idx, row in df.iterrows():
            flow_dict = row.to_dict()
            res = self.predict_single(flow_dict)
            results.append({
                "flow_index": idx,
                "verdict": res["verdict"],
                "predicted_class": res["predicted_class"],
                "confidence": res["confidence_percent"],
                "is_malicious": res["is_malicious"],
            })
        return pd.DataFrame(results)


def create_sample_files():
    """Create sample CSV and JSON files for instant testing."""
    sample_dir = ProjectPaths.DATA_DIR
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_flow = {
        "dur": 0.000011,
        "proto": "udp",
        "service": "dns",
        "state": "INT",
        "spkts": 2,
        "dpkts": 0,
        "sbytes": 168,
        "dbytes": 0,
        "rate": 90909.09,
        "sttl": 254,
        "dttl": 0,
        "sload": 61090908.0,
        "dload": 0.0,
        "sloss": 0,
        "dloss": 0,
        "sinpkt": 0.011,
        "dinpkt": 0.0,
        "sjit": 0.0,
        "djit": 0.0,
        "swin": 0,
        "stcpb": 0,
        "dtcpb": 0,
        "dwin": 0,
        "tcprtt": 0.0,
        "synack": 0.0,
        "ackdat": 0.0,
        "smean": 84,
        "dmean": 0,
        "trans_depth": 0,
        "response_body_len": 0,
        "ct_srv_src": 2,
        "ct_state_ttl": 2,
        "ct_dst_ltm": 1,
        "ct_src_dport_ltm": 1,
        "ct_dst_sport_ltm": 1,
        "ct_dst_src_ltm": 2,
        "is_ftp_login": 0,
        "ct_ftp_cmd": 0,
        "ct_flw_http_mthd": 0,
        "ct_src_ltm": 1,
        "ct_srv_dst": 2,
        "is_sm_ips_ports": 0
    }

    json_path = sample_dir / "sample_flow.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sample_flow, f, indent=2)

    # Sample batch CSV
    sample_df = pd.DataFrame([
        # Benign-like flow
        {**sample_flow, "dur": 0.05, "proto": "tcp", "service": "http", "state": "FIN", "sbytes": 350, "dbytes": 1200, "sttl": 62, "dttl": 252},
        # High-rate Generic flood
        {**sample_flow, "dur": 0.000005, "proto": "udp", "service": "dns", "state": "INT", "sbytes": 5000, "rate": 250000.0, "sttl": 254},
        # DoS SYN flood
        {**sample_flow, "dur": 0.001, "proto": "tcp", "service": "-", "state": "CON", "spkts": 50, "dpkts": 0, "rate": 50000.0, "sttl": 64},
    ])
    csv_path = sample_dir / "sample_flows.csv"
    sample_df.to_csv(csv_path, index=False)
    return json_path, csv_path


def run_demo():
    """Run interactive demonstration on sample network flows."""
    print("\n" + "=" * 75)
    print("      TGCF-IDS : HEADLESS INFERENCE / CLI TEST DEMO (NO FRONTEND)      ")
    print("=" * 75)
    predictor = FlowPredictor()

    test_cases = [
        {
            "name": "Test Case 1: Standard Benign Web Flow (HTTP / TCP)",
            "flow": {
                "dur": 0.12, "proto": "tcp", "service": "http", "state": "FIN",
                "spkts": 10, "dpkts": 12, "sbytes": 850, "dbytes": 4500,
                "rate": 183.3, "sttl": 62, "dttl": 252, "sload": 56666.0, "dload": 300000.0,
                "swin": 255, "dwin": 255, "smean": 85, "dmean": 375, "ct_srv_src": 1, "ct_dst_src_ltm": 1
            }
        },
        {
            "name": "Test Case 2: High-Rate Volumetric Flood (Generic / UDP)",
            "flow": {
                "dur": 0.000002, "proto": "udp", "service": "dns", "state": "INT",
                "spkts": 2, "dpkts": 0, "sbytes": 168, "dbytes": 0,
                "rate": 500000.0, "sttl": 254, "dttl": 0, "sload": 336000000.0, "dload": 0.0,
                "smean": 84, "dmean": 0, "ct_srv_src": 20, "ct_dst_src_ltm": 20
            }
        },
        {
            "name": "Test Case 3: Reconnaissance Port Scan Probe",
            "flow": {
                "dur": 0.00001, "proto": "tcp", "service": "-", "state": "INT",
                "spkts": 1, "dpkts": 0, "sbytes": 44, "dbytes": 0,
                "rate": 100000.0, "sttl": 254, "dttl": 0, "ct_dst_sport_ltm": 15, "ct_src_dport_ltm": 15
            }
        }
    ]

    for tc in test_cases:
        print(f"\n[+] {tc['name']}")
        res = predictor.predict_single(tc["flow"])
        print(f"    - Verdict        : \033[1;{31 if res['is_malicious'] else 32}m{res['verdict']}\033[0m")
        print(f"    - Attack Category: {res['predicted_class']}")
        print(f"    - Confidence     : {res['confidence_percent']}")
        print("    - Class Probabilities (Top 3):")
        sorted_probs = sorted(res["all_class_probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
        for cname, p in sorted_probs:
            print(f"        * {cname:<15}: {p * 100:6.2f}%")
    print("\n" + "=" * 75)


def main():
    parser = argparse.ArgumentParser(description="TGCF-IDS Headless / CLI Prediction Tool")
    parser.add_argument("--demo", action="store_true", help="Run quick demo on built-in test flows")
    parser.add_argument("--csv", type=str, help="Path to input CSV file containing network flows")
    parser.add_argument("--json", type=str, help="Path to JSON file containing a single flow record")
    parser.add_argument("--interactive", action="store_true", help="Run interactive terminal prompt")
    parser.add_argument("--output", type=str, help="Optional output CSV path for batch predictions")

    args = parser.parse_args()
    create_sample_files()

    if args.demo or (not args.csv and not args.json and not args.interactive):
        run_demo()
        return

    predictor = FlowPredictor()

    if args.json:
        json_path = Path(args.json)
        if not json_path.exists():
            print(f"[-] Error: JSON file not found: {json_path}")
            return
        with open(json_path, "r", encoding="utf-8") as f:
            flow_dict = json.load(f)
        res = predictor.predict_single(flow_dict)
        print("\n=== TGCF-IDS Prediction Result ===")
        print(json.dumps(res, indent=2))

    elif args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"[-] Error: CSV file not found: {csv_path}")
            return
        df = pd.read_csv(csv_path)
        print(f"[+] Loaded {len(df)} flows from {csv_path}")
        results_df = predictor.predict_dataframe(df)
        print("\n=== Batch Prediction Summary ===")
        print(results_df.to_string(index=False))
        if args.output:
            results_df.to_csv(args.output, index=False)
            print(f"[+] Saved batch predictions to: {args.output}")

    elif args.interactive:
        print("\n=== Interactive Flow Inspector (Terminal CLI) ===")
        print("Enter key flow attributes (or press ENTER for default):")
        dur = float(input("Duration (dur) [default 0.05]: ") or 0.05)
        proto = input("Protocol (proto) [default tcp]: ") or "tcp"
        service = input("Service (service) [default http]: ") or "http"
        state = input("State (state) [default FIN]: ") or "FIN"
        sbytes = float(input("Source Bytes (sbytes) [default 500]: ") or 500)
        dbytes = float(input("Dest Bytes (dbytes) [default 1500]: ") or 1500)
        rate = float(input("Packet Rate (rate) [default 200.0]: ") or 200.0)

        flow = {
            "dur": dur, "proto": proto, "service": service, "state": state,
            "sbytes": sbytes, "dbytes": dbytes, "rate": rate,
            "spkts": 5, "dpkts": 5, "sttl": 64, "dttl": 64
        }
        res = predictor.predict_single(flow)
        print(f"\n[+] Prediction: {res['verdict']} | Class: {res['predicted_class']} | Confidence: {res['confidence_percent']}")


if __name__ == "__main__":
    main()
