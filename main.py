#!/usr/bin/env python3
"""
TGCF-IDS: Temporal Graph Contrastive Feature-Transformer Intrusion Detection System
Main Entrypoint and Pipeline Orchestrator.
"""

import argparse
import sys
from pathlib import Path
import yaml

from src.utils.paths import ProjectPaths
from scripts.check_env import run_diagnostics


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file using cross-platform pathlib resolution."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TGCF-IDS: Temporal Graph Contrastive Feature-Transformer NIDS"
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Run environment, CUDA, PyTorch, and PyG diagnostics.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ProjectPaths.DEFAULT_CONFIG),
        help="Path to YAML configuration file.",
    )

    args = parser.parse_args()

    if args.check_env:
        run_diagnostics()
        return

    # Print Project Banner
    print("=" * 70)
    print("  TGCF-IDS : Temporal Graph Contrastive Feature-Transformer NIDS  ")
    print("  Academic Research Foundation Initialized                        ")
    print("=" * 70)

    config_file = Path(args.config)
    try:
        cfg = load_config(config_file)
        print(f"[+] Loaded Configuration from : {config_file}")
        print(f"[+] Dataset Target             : {cfg.get('dataset', {}).get('name', 'N/A')}")
        print(f"[+] Device Requested           : {cfg.get('project', {}).get('device', 'cpu')}")
        print(f"[+] Random Seed                : {cfg.get('project', {}).get('seed', 42)}")
    except Exception as e:
        print(f"[-] Notice: Configuration could not be parsed: {e}")

    # Ensure all foundational directories exist
    ProjectPaths.ensure_directories()
    print("[+] Verified all project directory structures.")
    print("-" * 70)
    print("Pipeline Stages Available:")
    print("  1. Environment Diagnostics  : python main.py --check-env")
    print("  2. Run Tests                : pytest")
    print("  3. Pipeline Execution       : (Scheduled for next development phase)")
    print("=" * 70)


if __name__ == "__main__":
    main()
