#!/usr/bin/env python3
"""
TGCF-IDS Packet Capture Abstraction Layer.
"""

from abc import ABC, abstractmethod
from typing import Dict, Generator, Optional, Any
from dataclasses import dataclass
import time


@dataclass
class PacketEvent:
    """Standardized internal representation of a captured network packet."""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str       # "tcp", "udp", "icmp", "other"
    packet_length: int
    tcp_flags: int = 0  # Bitmask: FIN=1, SYN=2, RST=4, PSH=8, ACK=16, URG=32
    ttl: int = 64
    payload: bytes = b""


class PacketSource(ABC):
    """Abstract interface for packet capture backends (Live, PCAP, Synthetic)."""

    @abstractmethod
    def start(self) -> None:
        """Initialize and open the packet stream."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Gracefully close and release network resources."""
        pass

    @abstractmethod
    def packets(self) -> Generator[PacketEvent, None, None]:
        """Yields captured packets in chronological order."""
        pass
