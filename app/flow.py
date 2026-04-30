"""
flow.py
────────
Week 2 — In-memory flow tracking primitives.

Classes
-------
FlowKey      — 5-tuple that uniquely identifies a network flow.
PacketRecord — Immutable snapshot of one packet's metadata.
FlowRecord   — Accumulates packets belonging to a single flow and
               tracks everything the feature extractor needs.

Flow expiry rules (checked by FlowRecord.is_expired):
  1. FIN flag seen → flow is considered closed immediately.
  2. last_seen older than idle_timeout seconds → idle expiry.
  3. start_time older than active_timeout seconds → hard cap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# FlowKey  (hashable 5-tuple)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FlowKey:
    src_ip:   str
    dst_ip:   str
    src_port: int
    dst_port: int
    protocol: int   # 6 = TCP, 17 = UDP, etc.


# ─────────────────────────────────────────────────────────────────────────────
# PacketRecord  (immutable per-packet snapshot)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PacketRecord:
    timestamp:     float          # Unix epoch seconds
    length:        int            # total wire length (bytes)
    header_length: int            # IP + transport header bytes
    direction:     str            # "fwd" or "bwd"

    # TCP flags (0 or 1)
    flag_syn: int = 0
    flag_ack: int = 0
    flag_fin: int = 0
    flag_rst: int = 0
    flag_psh: int = 0
    flag_urg: int = 0
    flag_cwe: int = 0
    flag_ece: int = 0

    payload_len: int = 0          # application-layer payload bytes
    init_win:    int = 0          # TCP initial window size (first packet only)


# ─────────────────────────────────────────────────────────────────────────────
# FlowRecord  (mutable accumulator)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FlowRecord:
    key:        FlowKey
    start_time: float = field(default_factory=time.time)

    # Packet lists (populated by add_packet)
    fwd_packets: List[PacketRecord] = field(default_factory=list)
    bwd_packets: List[PacketRecord] = field(default_factory=list)

    # Derived state updated incrementally
    last_seen:    float = 0.0
    fin_seen:     bool  = False

    # Initial window sizes (first packet in each direction)
    fwd_init_win: int = 0
    bwd_init_win: int = 0

    # Active / idle sub-flow tracking
    # An "active period" is a burst of packets; idle is the gap between bursts.
    # We track the timestamp of the last packet in the previous burst and
    # the start of the current burst to compute Active / Idle stats.
    _active_start:     float = field(default=0.0, repr=False)
    _last_packet_time: float = field(default=0.0, repr=False)
    active_periods:    List[float] = field(default_factory=list)   # durations
    idle_periods:      List[float] = field(default_factory=list)   # durations

    # Threshold (seconds) between packets to start a new active period
    ACTIVE_TIMEOUT: float = field(default=5.0, repr=False)

    def __post_init__(self):
        self.last_seen         = self.start_time
        self._active_start     = self.start_time
        self._last_packet_time = self.start_time

    # ── Public API ────────────────────────────────────────────────────────────

    def add_packet(self, pkt: PacketRecord) -> None:
        """Append a packet to the flow, updating all derived state."""
        # Detect idle gap → close previous active period, start new one
        gap = pkt.timestamp - self._last_packet_time
        if gap > self.ACTIVE_TIMEOUT and self._last_packet_time > 0:
            active_dur = self._last_packet_time - self._active_start
            if active_dur > 0:
                self.active_periods.append(active_dur)
            self.idle_periods.append(gap)
            self._active_start = pkt.timestamp

        self._last_packet_time = pkt.timestamp
        self.last_seen         = max(self.last_seen, pkt.timestamp)

        if pkt.direction == "fwd":
            if not self.fwd_packets:
                self.fwd_init_win = pkt.init_win
            self.fwd_packets.append(pkt)
        else:
            if not self.bwd_packets:
                self.bwd_init_win = pkt.init_win
            self.bwd_packets.append(pkt)

        if pkt.flag_fin:
            self.fin_seen = True

    def is_expired(
        self,
        idle_timeout:   float = 120.0,
        active_timeout: float = 600.0,
    ) -> bool:
        """
        Return True if the flow should be exported:
          • FIN flag was seen, OR
          • no packet for idle_timeout seconds, OR
          • flow has been open longer than active_timeout seconds.
        """
        if self.fin_seen:
            return True
        now = time.time()
        if (now - self.last_seen) >= idle_timeout:
            return True
        if (now - self.start_time) >= active_timeout:
            return True
        return False

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def all_packets(self) -> List[PacketRecord]:
        return sorted(
            self.fwd_packets + self.bwd_packets,
            key=lambda p: p.timestamp,
        )

    @property
    def duration(self) -> float:
        """Flow duration in microseconds (matches CIC-IDS-2017 convention)."""
        if not (self.fwd_packets or self.bwd_packets):
            return 0.0
        all_ts = [p.timestamp for p in self.fwd_packets + self.bwd_packets]
        return (max(all_ts) - min(all_ts)) * 1_000_000   # seconds → µs
