#!/usr/bin/env python3
"""
TGCF-IDS Real-Time Bidirectional Flow Extraction & Aggregation Engine.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Generator
from collections import OrderedDict

from production.capture.interface import PacketEvent

logger = logging.getLogger("production.capture.flow_manager")


class ActiveFlowState:
    """Internal statistical accumulator for a single bidirectional connection."""

    def __init__(self, key: Tuple[str, str, int, int, str], start_time: float):
        self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.proto = key
        self.start_time = start_time
        self.last_seen = start_time

        self.spkts = 0
        self.dpkts = 0
        self.sbytes = 0
        self.dbytes = 0
        self.sttl_last = 64
        self.dttl_last = 64
        self.state = "CON"
        self.is_closed = False

        self.src_pkts_timestamps: List[float] = []
        self.dst_pkts_timestamps: List[float] = []

    def update(self, pkt: PacketEvent):
        """Updates flow state with a new incoming packet."""
        self.last_seen = pkt.timestamp
        is_forward = (pkt.src_ip == self.src_ip and pkt.src_port == self.src_port)

        if is_forward:
            self.spkts += 1
            self.sbytes += pkt.packet_length
            self.sttl_last = pkt.ttl
            self.src_pkts_timestamps.append(pkt.timestamp)
        else:
            self.dpkts += 1
            self.dbytes += pkt.packet_length
            self.dttl_last = pkt.ttl
            self.dst_pkts_timestamps.append(pkt.timestamp)

        # Check TCP termination flags
        if pkt.tcp_flags & 0x01:  # FIN
            self.state = "FIN"
            self.is_closed = True
        elif pkt.tcp_flags & 0x04:  # RST
            self.state = "RST"
            self.is_closed = True

    def export_flow_record(self) -> Dict[str, Any]:
        """Calculates UNSW-NB15 flow features from raw accumulated packets."""
        dur = max(1e-6, self.last_seen - self.start_time)
        tot_pkts = self.spkts + self.dpkts
        rate = tot_pkts / dur

        sload = (self.sbytes * 8.0) / dur if dur > 0 else 0.0
        dload = (self.dbytes * 8.0) / dur if dur > 0 else 0.0

        smean = self.sbytes / max(1, self.spkts)
        dmean = self.dbytes / max(1, self.dpkts)

        # Inter-packet arrival times
        sinpkt = 0.0
        if len(self.src_pkts_timestamps) > 1:
            diffs = np_diff = [
                self.src_pkts_timestamps[i] - self.src_pkts_timestamps[i - 1]
                for i in range(1, len(self.src_pkts_timestamps))
            ]
            sinpkt = (sum(diffs) / len(diffs)) * 1000.0  # ms

        dinpkt = 0.0
        if len(self.dst_pkts_timestamps) > 1:
            diffs = [
                self.dst_pkts_timestamps[i] - self.dst_pkts_timestamps[i - 1]
                for i in range(1, len(self.dst_pkts_timestamps))
            ]
            dinpkt = (sum(diffs) / len(diffs)) * 1000.0  # ms

        service = "http" if (self.dst_port in (80, 443, 8080) or self.src_port in (80, 443, 8080)) else (
            "dns" if (self.dst_port == 53 or self.src_port == 53) else (
                "ftp" if (self.dst_port in (20, 21)) else (
                    "ssh" if (self.dst_port == 22) else "-"
                )
            )
        )

        return {
            "timestamp": self.start_time,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "proto": self.proto.lower(),
            "service": service,
            "state": self.state,
            "dur": float(dur),
            "sbytes": float(self.sbytes),
            "dbytes": float(self.dbytes),
            "spkts": float(self.spkts),
            "dpkts": float(self.dpkts),
            "rate": float(rate),
            "sttl": float(self.sttl_last),
            "dttl": float(self.dttl_last),
            "sload": float(sload),
            "dload": float(dload),
            "sloss": 0.0,
            "dloss": 0.0,
            "sinpkt": float(sinpkt),
            "dinpkt": float(dinpkt),
            "sjit": 0.0,
            "djit": 0.0,
            "swin": 255.0 if self.proto == "tcp" else 0.0,
            "stcpb": 0.0,
            "dtcpb": 0.0,
            "dwin": 255.0 if self.proto == "tcp" else 0.0,
            "tcprtt": 0.0,
            "synack": 0.0,
            "ackdat": 0.0,
            "smean": float(smean),
            "dmean": float(dmean),
            "trans_depth": 0.0,
            "response_body_len": 0.0,
            "ct_srv_src": 1.0,
            "ct_state_ttl": 0.0,
            "ct_dst_ltm": 1.0,
            "ct_src_dport_ltm": 1.0,
            "ct_dst_sport_ltm": 1.0,
            "ct_dst_src_ltm": 1.0,
            "is_ftp_login": 0.0,
            "ct_ftp_cmd": 0.0,
            "ct_flw_http_mthd": 0.0,
            "ct_src_ltm": 1.0,
            "ct_srv_dst": 1.0,
            "is_sm_ips_ports": 1.0 if (self.src_ip == self.dst_ip and self.src_port == self.dst_port) else 0.0,
        }


class FlowManager:
    """
    Manages active flow table, handles timeouts, and emits finalized flows.
    """

    def __init__(
        self,
        idle_timeout_seconds: float = 30.0,
        active_timeout_seconds: float = 300.0,
        max_tracked_flows: int = 100000,
    ):
        self.idle_timeout = idle_timeout_seconds
        self.active_timeout = active_timeout_seconds
        self.max_tracked_flows = max_tracked_flows
        self.active_flows: OrderedDict[Tuple[str, str, int, int, str], ActiveFlowState] = OrderedDict()

    def _make_key(self, pkt: PacketEvent) -> Tuple[Tuple[str, str, int, int, str], bool]:
        """Produces canonical bidirectional flow key."""
        fwd_key = (pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.protocol)
        rev_key = (pkt.dst_ip, pkt.src_ip, pkt.dst_port, pkt.src_port, pkt.protocol)
        if fwd_key in self.active_flows:
            return fwd_key, True
        if rev_key in self.active_flows:
            return rev_key, False
        return fwd_key, True

    def process_packet(self, pkt: PacketEvent) -> List[Dict[str, Any]]:
        """Ingests a packet, updates active flow, and returns any immediately finalized flows."""
        emitted_flows = []
        key, is_fwd = self._make_key(pkt)

        if key not in self.active_flows:
            # Memory protection: enforce LRU eviction if maximum capacity reached
            if len(self.active_flows) >= self.max_tracked_flows:
                oldest_key, oldest_flow = self.active_flows.popitem(last=False)
                emitted_flows.append(oldest_flow.export_flow_record())

            self.active_flows[key] = ActiveFlowState(key, start_time=pkt.timestamp)

        flow = self.active_flows[key]
        flow.update(pkt)
        self.active_flows.move_to_end(key)

        # Check for immediate TCP closure (FIN/RST)
        if flow.is_closed:
            del self.active_flows[key]
            emitted_flows.append(flow.export_flow_record())

        return emitted_flows

    def sweep_timeouts(self, current_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """Sweeps and finalizes expired flows based on idle and active timeouts."""
        now = current_time or time.time()
        expired_keys = []
        emitted_flows = []

        for key, flow in self.active_flows.items():
            is_idle = (now - flow.last_seen) >= self.idle_timeout
            is_active_expired = (now - flow.start_time) >= self.active_timeout

            if is_idle or is_active_expired:
                expired_keys.append(key)
                emitted_flows.append(flow.export_flow_record())

        for k in expired_keys:
            del self.active_flows[k]

        return emitted_flows
