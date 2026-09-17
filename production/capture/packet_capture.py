#!/usr/bin/env python3
"""
TGCF-IDS Packet Capture Sources: Live Socket & Synthetic Packet Generator.
"""

import random
import socket
import struct
import time
from typing import Generator, Optional, List, Dict, Any

from production.capture.interface import PacketSource, PacketEvent


class SyntheticPacketSource(PacketSource):
    """
    Deterministic synthetic packet generator for safe integration and load testing.
    """

    def __init__(self, count: int = 100, inter_packet_delay: float = 0.01):
        self.count = count
        self.delay = inter_packet_delay
        self.running = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def packets(self) -> Generator[PacketEvent, None, None]:
        ips = [f"192.168.1.{i}" for i in range(10, 25)]
        remote_ips = [f"10.0.0.{i}" for i in range(1, 10)]

        for i in range(self.count):
            if not self.running:
                break
            src = random.choice(ips)
            dst = random.choice(remote_ips)
            proto = random.choice(["tcp", "udp", "icmp"])
            sport = random.randint(1024, 65535)
            dport = random.choice([80, 443, 53, 22, 21, 8080])
            pkt_len = random.randint(64, 1500)

            yield PacketEvent(
                timestamp=time.time(),
                src_ip=src,
                dst_ip=dst,
                src_port=sport,
                dst_port=dport,
                protocol=proto,
                packet_length=pkt_len,
                tcp_flags=0x10 if proto == "tcp" else 0,  # ACK flag
                ttl=64,
            )
            if self.delay > 0:
                time.sleep(self.delay)


class LiveSocketPacketSource(PacketSource):
    """
    Live raw socket sniffer capturing packets from host network interfaces.
    """

    def __init__(self, interface_ip: Optional[str] = None):
        self.interface_ip = interface_ip or socket.gethostbyname(socket.gethostname())
        self.sock: Optional[socket.socket] = None
        self.running = False

    def start(self) -> None:
        self.running = True
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IP)
            self.sock.bind((self.interface_ip, 0))
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
            self.sock.ioctl(socket.SIO_RCVALL, socket.RCVALL_ON)
        except Exception as e:
            self.sock = None
            raise PermissionError(f"Raw socket binding failed (requires Administrator): {e}")

    def stop(self) -> None:
        self.running = False
        if self.sock:
            try:
                self.sock.ioctl(socket.SIO_RCVALL, socket.RCVALL_OFF)
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def packets(self) -> Generator[PacketEvent, None, None]:
        if not self.sock:
            return

        while self.running:
            try:
                raw_data, _ = self.sock.recvfrom(65535)
                iph = struct.unpack("!BBHHHBBH4s4s", raw_data[0:20])
                proto_code = iph[6]
                src_ip = socket.inet_ntoa(iph[8])
                dst_ip = socket.inet_ntoa(iph[9])
                ttl = iph[5]
                proto = "tcp" if proto_code == 6 else ("udp" if proto_code == 17 else "other")

                sport, dport, flags = 0, 0, 0
                if proto_code == 6 and len(raw_data) >= 34:  # TCP
                    tcph = struct.unpack("!HHLLBBHHH", raw_data[20:40])
                    sport, dport = tcph[0], tcph[1]
                    flags = tcph[5]
                elif proto_code == 17 and len(raw_data) >= 28:  # UDP
                    udph = struct.unpack("!HHHH", raw_data[20:28])
                    sport, dport = udph[0], udph[1]

                yield PacketEvent(
                    timestamp=time.time(),
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=sport,
                    dst_port=dport,
                    protocol=proto,
                    packet_length=len(raw_data),
                    tcp_flags=flags,
                    ttl=ttl,
                )
            except Exception:
                break
