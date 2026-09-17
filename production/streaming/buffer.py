#!/usr/bin/env python3
"""
TGCF-IDS Streaming Buffer and Sliding Window Manager.
"""

import collections
import threading
import time
from typing import Dict, List, Optional, Any, Callable


class StreamingFlowBuffer:
    """
    Thread-safe bounded FIFO buffer collecting raw flow events from packet capture.
    """

    def __init__(self, max_capacity: int = 10000):
        self.max_capacity = max_capacity
        self.buffer = collections.deque(maxlen=max_capacity)
        self.lock = threading.Lock()
        self._dropped_count = 0

    def push(self, flow: Dict[str, Any]) -> bool:
        with self.lock:
            if len(self.buffer) >= self.max_capacity:
                self._dropped_count += 1
            self.buffer.append(flow)
            return True

    def drain_batch(self, max_items: int = 64) -> List[Dict[str, Any]]:
        with self.lock:
            batch = []
            while self.buffer and len(batch) < max_items:
                batch.append(self.buffer.popleft())
            return batch

    def __len__(self) -> int:
        with self.lock:
            return len(self.buffer)

    @property
    def dropped_flows(self) -> int:
        return self._dropped_count


class EventStreamHub:
    """
    Publish-subscribe event bus broadcasting predictions, alerts, and system telemetry.
    """

    def __init__(self):
        self.subscribers: List[Callable[[Dict[str, Any]], None]] = []
        self.lock = threading.Lock()

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]):
        with self.lock:
            self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Dict[str, Any]], None]):
        with self.lock:
            if callback in self.subscribers:
                self.subscribers.remove(callback)

    def publish(self, event: Dict[str, Any]):
        with self.lock:
            callbacks = list(self.subscribers)
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                pass
