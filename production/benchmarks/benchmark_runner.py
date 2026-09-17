#!/usr/bin/env python3
"""
TGCF-IDS Synthetic Traffic Generator and Production Load Testing Engine.
"""

import math
import random
import time
import logging
import numpy as np
from typing import Dict, List, Generator, Any

from production.detection.detector import RealTimeIntrusionDetector

logger = logging.getLogger("production.benchmarks")


def generate_synthetic_flow_batch(count: int = 64) -> List[Dict[str, Any]]:
    """Generates realistic synthetic flow records for throughput benchmarking."""
    flows = []
    protocols = ["tcp", "udp", "icmp"]
    services = ["http", "dns", "ftp", "ssh", "smtp", "-"]
    states = ["FIN", "CON", "INT", "REQ", "RST"]

    for _ in range(count):
        src_ip = f"192.168.1.{random.randint(2, 254)}"
        dst_ip = f"10.0.0.{random.randint(2, 254)}"
        dur = random.uniform(0.0001, 2.5)
        spkts = random.randint(1, 50)
        dpkts = random.randint(0, 50)
        sbytes = random.randint(40, 15000)
        dbytes = random.randint(0, 30000)
        rate = (spkts + dpkts) / dur

        flow = {
            "timestamp": time.time(),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": random.randint(1024, 65535),
            "dst_port": random.choice([80, 443, 53, 22, 21, 8080]),
            "proto": random.choice(protocols),
            "service": random.choice(services),
            "state": random.choice(states),
            "dur": dur,
            "sbytes": float(sbytes),
            "dbytes": float(dbytes),
            "spkts": float(spkts),
            "dpkts": float(dpkts),
            "rate": float(rate),
            "sttl": float(random.choice([64, 128, 254])),
            "dttl": float(random.choice([0, 64, 128, 252])),
            "sload": float((sbytes * 8.0) / dur),
            "dload": float((dbytes * 8.0) / dur),
            "smean": float(sbytes / spkts),
            "dmean": float(dbytes / max(1, dpkts)),
            "ct_srv_src": float(random.randint(1, 10)),
            "ct_dst_ltm": float(random.randint(1, 15)),
        }
        flows.append(flow)
    return flows


def run_production_benchmark(num_flows: int = 1000, batch_size: int = 64) -> Dict[str, Any]:
    """Runs high-throughput load benchmark measuring p50, p95, p99 latency."""
    detector = RealTimeIntrusionDetector(batch_size=batch_size)
    print("\n" + "=" * 80)
    print(f"      TGCF-IDS PRODUCTION BENCHMARK RUNNER ({num_flows} FLOWS, BATCH: {batch_size})      ")
    print("=" * 80)

    batches = []
    remaining = num_flows
    while remaining > 0:
        cur_bs = min(remaining, batch_size)
        batches.append(generate_synthetic_flow_batch(cur_bs))
        remaining -= cur_bs

    # Warmup
    warmup_batch = generate_synthetic_flow_batch(32)
    detector.process_flow_batch(warmup_batch)

    latencies_per_batch = []
    t_start_total = time.perf_counter()

    for b in batches:
        t_b_start = time.perf_counter()
        detector.process_flow_batch(b)
        latencies_per_batch.append((time.perf_counter() - t_b_start) * 1000.0)

    t_total_sec = time.perf_counter() - t_start_total
    throughput = num_flows / max(1e-6, t_total_sec)

    latencies_per_flow = [lat / len(b) for lat, b in zip(latencies_per_batch, batches)]
    p50 = float(np.percentile(latencies_per_flow, 50))
    p95 = float(np.percentile(latencies_per_flow, 95))
    p99 = float(np.percentile(latencies_per_flow, 99))

    report = {
        "total_flows": num_flows,
        "batch_size": batch_size,
        "total_duration_sec": round(t_total_sec, 4),
        "sustained_throughput_flows_sec": round(throughput, 2),
        "per_flow_latency_p50_ms": round(p50, 3),
        "per_flow_latency_p95_ms": round(p95, 3),
        "per_flow_latency_p99_ms": round(p99, 3),
        "device": str(detector.inference_engine.device),
    }

    print(f"[+] Total Duration:      {report['total_duration_sec']} s")
    print(f"[+] Sustained Throughput: {report['sustained_throughput_flows_sec']:,} flows/sec")
    print(f"[+] Per-Flow Latency p50: {report['per_flow_latency_p50_ms']} ms")
    print(f"[+] Per-Flow Latency p95: {report['per_flow_latency_p95_ms']} ms")
    print(f"[+] Per-Flow Latency p99: {report['per_flow_latency_p99_ms']} ms")
    print(f"[+] Inference Device:     {report['device']}")
    print("=" * 80 + "\n")
    return report


if __name__ == "__main__":
    import numpy as np
    run_production_benchmark()
