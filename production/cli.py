#!/usr/bin/env python3
"""
TGCF-IDS Production Unified Command Line Interface.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from production.model.model_loader import ProductionModelLoader
from production.monitoring.logging import HealthCheckManager
from production.benchmarks.benchmark_runner import run_production_benchmark


def cmd_validate_model(args):
    print("\n[*] Validating Frozen Research Model Checkpoint & Architecture...")
    loader = ProductionModelLoader(verify_hash=True)
    model, manifest = loader.build_and_load_model()
    print("\n" + "=" * 70)
    print("           TGCF-IDS MODEL VALIDATION REPORT           ")
    print("=" * 70)
    for k, v in manifest.items():
        print(f"  {k:<20}: {v}")
    print("=" * 70)
    print("[+] Model integrity check: PASSED (100% Verified).\n")


def cmd_health(args):
    from production.detection.detector import RealTimeIntrusionDetector
    detector = RealTimeIntrusionDetector()
    health_mgr = HealthCheckManager(detector=detector)
    print("\n" + "=" * 60)
    print("       TGCF-IDS PRODUCTION HEALTH REPORT       ")
    print("=" * 60)
    print("Liveness Status :", json.dumps(health_mgr.get_health_status(), indent=2))
    print("Readiness Status:", json.dumps(health_mgr.get_readiness_status(), indent=2))
    print("=" * 60 + "\n")



def cmd_benchmark(args):
    run_production_benchmark(num_flows=args.flows, batch_size=args.batch_size)


def cmd_run(args):
    import uvicorn
    print("\n" + "=" * 75)
    print(f"  Starting TGCF-IDS Production Server on http://{args.host}:{args.port}")
    print(f"  Swagger Docs: http://{args.host}:{args.port}/docs")
    print(f"  Live Dashboard: http://{args.host}:{args.port}/")
    print("=" * 75 + "\n")
    uvicorn.run("production.api.main:app", host=args.host, port=args.port, log_level="info")


def main():
    parser = argparse.ArgumentParser(description="TGCF-IDS Production System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Production command to run")

    # Command: run
    p_run = subparsers.add_parser("run", help="Start FastAPI production server and WebSocket hub")
    p_run.add_argument("--host", default="0.0.0.0", help="Host address")
    p_run.add_argument("--port", type=int, default=8000, help="Port number")

    # Command: validate-model
    p_val = subparsers.add_parser("validate-model", help="Verify frozen research model checkpoint integrity")

    # Command: health
    p_health = subparsers.add_parser("health", help="Check subsystem health and readiness")

    # Command: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run high-throughput inference benchmark")
    p_bench.add_argument("--flows", type=int, default=1000, help="Number of flows to test")
    p_bench.add_argument("--batch-size", type=int, default=64, help="Inference batch size")

    args = parser.parse_args()

    if args.command == "validate-model":
        cmd_validate_model(args)
    elif args.command == "health":
        cmd_health(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
