#!/usr/bin/env python3
"""
TGCF-IDS PCAP Replay & Offline Traffic Ingestion Engine.
"""

import struct
import time
from pathlib import Path
from typing import Generator, Optional, Union

from production.capture.interface import PacketSource, PacketEvent


class PCAPPacketSource(PacketSource):
    """
    Reads standard libpcap binary files and replays packets at configurable playback speeds.
    """

    def __init__(self, pcap_path: Union[str, Path], playback_speed: float = 1.0):
        self.pcap_path = Path(pcap_path)
        self.playback_speed = playback_speed
        self.file_handle = None
        self.running = False

    def start(self) -> None:
        if not self.pcap_path.exists():
            raise FileNotFoundError(f"PCAP file not found: {self.pcap_path}")
        self.file_handle = open(self.pcap_path, "rb")
        self.running = True

        # Read 24-byte global pcap header
        global_header = self.file_handle.read(24)
        if len(global_header) < 24:
            raise ValueError(f"Invalid PCAP file: header too short ({len(global_header)} bytes)")

    def stop(self) -> None:
        self.running = False
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def packets(self) -> Generator[PacketEvent, None, None]:
        if not self.file_handle:
            return

        last_ts = None
        while self.running:
            header = self.file_handle.read(16)
            if len(header) < 16:
                break

            ts_sec, ts_usec, incl_len, orig_len = struct.unpack("=IIII", header)
            packet_data = self.file_handle.read(incl_len)
            current_ts = ts_sec + (ts_usec / 1e6)

            # Timing control for replay speed
            if last_ts is not None and self.playback_speed > 0:
                delta_sec = (current_ts - last_ts) / self.playback_speed
                if 0 < delta_sec < 5.0:
                    time.sleep(delta_sec)
            last_ts = current_ts

            # Parse Ethernet (14 bytes) + IP header (20 bytes)
            if len(packet_data) >= 34:
                eth_type = struct.unpack("!H", packet_data[12:14])[0]
                if eth_type == 0x0800:  # IPv4
                    ip_data = packet_data[14:34]
                    iph = struct.unpack("!BBHHHBBH4s4s", ip_data)
                    proto_code = iph[6]
                    src_ip = ".".join(map(str, iph[8]))
                    dst_ip = ".".join(map(str, iph[9]))
                    ttl = iph[5]
                    proto = "tcp" if proto_code == 6 else ("udp" if proto_code == 17 else "other")

                    sport, dport, flags = 0, 0, 0
                    if proto_code == 6 and len(packet_data) >= 54:
                        tcph = struct.unpack("!HHLLBBHHH", packet_data[34:54])
                        sport, dport, flags = tcph[0], tcph[1], tcph[5]
                    elif proto_code == 17 and len(packet_data) >= 42:
                        udph = struct.unpack("!HHHH", packet_data[34:42])
                        sport, dport = udph[0], udph[1]

                    yield PacketEvent(
                        timestamp=current_ts,
                        src_ip=src_ip,
                        dst_ip=dst_ip,
                        src_port=sport,
                        dst_port=dport,
                        protocol=proto,
                        packet_length=incl_len,
                        tcp_flags=flags,
                        ttl=ttl,
                    )
