"""
feature_extractor.py
─────────────────────
Week 2 — Compute the full CIC-IDS-2017 feature vector from a FlowRecord.

Output: a dict with exactly 77 keys whose names match the column names used
by the Data Team notebook and stored in models/feature_columns.json.

All values are Python float / int (both accepted by the Pydantic schema).
Division-by-zero cases always return 0.0.

Feature groups (matches CICFlowMeter output order):
  1.  Basic packet / byte counts                          [5 features]
  2.  Per-direction packet length stats                   [10 features]
  3.  Flow-level byte/packet rates                        [2 features]
  4.  Flow IAT (inter-arrival time) stats                 [4 features]
  5.  Fwd/Bwd IAT stats                                   [10 features]
  6.  TCP flag counts (per direction + totals)            [12 features]
  7.  Header length + per-direction packet rates          [4 features]
  8.  Aggregate packet length stats                       [4 features]
  9.  Ratios, averages, segment sizes                     [5 features]
  10. Bulk / subflow / window / active-data features      [14 features]
  11. Active / Idle period stats                          [8 features]
                                                         ──────────────
                                                         Total: 77
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Sequence

from app.flow import FlowRecord, PacketRecord

# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_div(num: float, den: float) -> float:
    return num / den if den != 0.0 else 0.0


def _stats(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Return (mean, std, max, min) — all 0.0 for empty sequences."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    mn  = float(sum(values)) / len(values)
    mx  = float(max(values))
    mi  = float(min(values))
    if len(values) >= 2:
        std = statistics.stdev(values)
    else:
        std = 0.0
    return mn, std, mx, mi


def _iat(timestamps: Sequence[float]) -> List[float]:
    """Inter-arrival times from a sorted list of timestamps (seconds)."""
    if len(timestamps) < 2:
        return []
    return [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(flow: FlowRecord) -> Dict[str, float]:
    """
    Compute all 77 CIC-IDS-2017 features from a FlowRecord.

    Parameters
    ----------
    flow : FlowRecord

    Returns
    -------
    dict[str, float]  — keys are the exact CIC-IDS-2017 column names.
    """
    fwd: List[PacketRecord] = flow.fwd_packets
    bwd: List[PacketRecord] = flow.bwd_packets
    all_pkts: List[PacketRecord] = sorted(
        fwd + bwd, key=lambda p: p.timestamp
    )

    n_fwd = len(fwd)
    n_bwd = len(bwd)
    n_all = len(all_pkts)

    # ── 1. Basic counts ───────────────────────────────────────────────────────
    fwd_lengths = [p.length for p in fwd]
    bwd_lengths = [p.length for p in bwd]
    all_lengths = [p.length for p in all_pkts]

    total_len_fwd = sum(fwd_lengths)
    total_len_bwd = sum(bwd_lengths)

    # ── Flow duration (µs) ────────────────────────────────────────────────────
    if all_pkts:
        ts_all = [p.timestamp for p in all_pkts]
        flow_duration_us = (max(ts_all) - min(ts_all)) * 1_000_000
    else:
        flow_duration_us = 0.0
    flow_duration_s = flow_duration_us / 1_000_000

    # ── 2. Per-direction packet length stats ──────────────────────────────────
    fwd_pkt_mean, fwd_pkt_std, fwd_pkt_max, fwd_pkt_min = _stats(fwd_lengths)
    bwd_pkt_mean, bwd_pkt_std, bwd_pkt_max, bwd_pkt_min = _stats(bwd_lengths)

    # ── 3. Flow byte/packet rates (per second) ────────────────────────────────
    total_bytes = total_len_fwd + total_len_bwd
    flow_bytes_s  = _safe_div(total_bytes, flow_duration_s)
    flow_pkts_s   = _safe_div(n_all,       flow_duration_s)

    # ── 4. Flow IAT stats ─────────────────────────────────────────────────────
    ts_sorted = sorted(p.timestamp for p in all_pkts)
    flow_iat  = _iat(ts_sorted)  # in seconds → keep as seconds (CICFlowMeter uses µs)
    # CICFlowMeter stores IAT in microseconds
    flow_iat_us = [v * 1_000_000 for v in flow_iat]
    flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = _stats(flow_iat_us)

    # ── 5. Fwd / Bwd IAT stats ────────────────────────────────────────────────
    fwd_ts = sorted(p.timestamp for p in fwd)
    bwd_ts = sorted(p.timestamp for p in bwd)

    fwd_iat_raw = _iat(fwd_ts)
    bwd_iat_raw = _iat(bwd_ts)

    fwd_iat_us = [v * 1_000_000 for v in fwd_iat_raw]
    bwd_iat_us = [v * 1_000_000 for v in bwd_iat_raw]

    fwd_iat_total = sum(fwd_iat_us)
    bwd_iat_total = sum(bwd_iat_us)
    fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = _stats(fwd_iat_us)
    bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = _stats(bwd_iat_us)

    # ── 6. TCP flag counts ────────────────────────────────────────────────────
    def _flag_sum(pkts: List[PacketRecord], attr: str) -> int:
        return sum(getattr(p, attr) for p in pkts)

    fin_count = _flag_sum(all_pkts, "flag_fin")
    syn_count = _flag_sum(all_pkts, "flag_syn")
    rst_count = _flag_sum(all_pkts, "flag_rst")
    psh_count = _flag_sum(all_pkts, "flag_psh")
    ack_count = _flag_sum(all_pkts, "flag_ack")
    urg_count = _flag_sum(all_pkts, "flag_urg")
    cwe_count = _flag_sum(all_pkts, "flag_cwe")
    ece_count = _flag_sum(all_pkts, "flag_ece")

    fwd_psh_flags = _flag_sum(fwd, "flag_psh")
    bwd_psh_flags = _flag_sum(bwd, "flag_psh")
    fwd_urg_flags = _flag_sum(fwd, "flag_urg")
    bwd_urg_flags = _flag_sum(bwd, "flag_urg")

    # ── 7. Header lengths + per-direction rates ───────────────────────────────
    fwd_header_len = sum(p.header_length for p in fwd)
    bwd_header_len = sum(p.header_length for p in bwd)
    fwd_pkts_s     = _safe_div(n_fwd, flow_duration_s)
    bwd_pkts_s     = _safe_div(n_bwd, flow_duration_s)

    # ── 8. Aggregate packet length stats ──────────────────────────────────────
    pkt_len_mean, pkt_len_std, pkt_len_max, pkt_len_min = _stats(all_lengths)
    pkt_len_var = pkt_len_std ** 2

    # ── 9. Ratios, averages, segment sizes ────────────────────────────────────
    down_up_ratio   = _safe_div(n_bwd, n_fwd)
    avg_pkt_size    = _safe_div(total_bytes, n_all)
    avg_fwd_seg     = fwd_pkt_mean    # same as Fwd Packet Length Mean
    avg_bwd_seg     = bwd_pkt_mean    # same as Bwd Packet Length Mean
    fwd_header_len2 = fwd_header_len  # "Fwd Header Length.1" duplicate column

    # ── 10. Bulk / subflow / window / active-data features ───────────────────
    # CICFlowMeter bulk features: set to 0 — requires multi-packet bulk
    # detection which is beyond a single-pass extractor scope.
    fwd_avg_bytes_bulk   = 0.0
    fwd_avg_pkts_bulk    = 0.0
    fwd_avg_bulk_rate    = 0.0
    bwd_avg_bytes_bulk   = 0.0
    bwd_avg_pkts_bulk    = 0.0
    bwd_avg_bulk_rate    = 0.0

    # Subflow: CICFlowMeter subflow = the whole flow for single-subflow cases
    subflow_fwd_pkts  = n_fwd
    subflow_fwd_bytes = total_len_fwd
    subflow_bwd_pkts  = n_bwd
    subflow_bwd_bytes = total_len_bwd

    # Window sizes (captured from first packet in each direction)
    init_win_fwd = flow.fwd_init_win
    init_win_bwd = flow.bwd_init_win

    # act_data_pkt_fwd: fwd packets that carried payload
    act_data_pkt_fwd = sum(1 for p in fwd if p.payload_len > 0)

    # min_seg_size_forward: minimum header length in forward direction
    min_seg_size_fwd = int(min((p.header_length for p in fwd), default=0))

    # ── 11. Active / Idle period stats ────────────────────────────────────────
    # Close the last active period
    active_periods = list(flow.active_periods)
    if flow._last_packet_time > flow._active_start:
        last_active = flow._last_packet_time - flow._active_start
        if last_active > 0:
            active_periods.append(last_active)
    # Convert to µs
    active_us = [v * 1_000_000 for v in active_periods]
    idle_us   = [v * 1_000_000 for v in flow.idle_periods]

    active_mean, active_std, active_max, active_min = _stats(active_us)
    idle_mean,   idle_std,   idle_max,   idle_min   = _stats(idle_us)

    # ── Assemble output dict (exact CIC-IDS-2017 column names) ───────────────
    return {
        # Group 1
        "Flow Duration":                  flow_duration_us,
        "Total Fwd Packets":              n_fwd,
        "Total Backward Packets":         n_bwd,
        "Total Length of Fwd Packets":    total_len_fwd,
        "Total Length of Bwd Packets":    total_len_bwd,
        # Group 2
        "Fwd Packet Length Max":          fwd_pkt_max,
        "Fwd Packet Length Min":          fwd_pkt_min,
        "Fwd Packet Length Mean":         fwd_pkt_mean,
        "Fwd Packet Length Std":          fwd_pkt_std,
        "Bwd Packet Length Max":          bwd_pkt_max,
        "Bwd Packet Length Min":          bwd_pkt_min,
        "Bwd Packet Length Mean":         bwd_pkt_mean,
        "Bwd Packet Length Std":          bwd_pkt_std,
        # Group 3
        "Flow Bytes/s":                   flow_bytes_s,
        "Flow Packets/s":                 flow_pkts_s,
        # Group 4
        "Flow IAT Mean":                  flow_iat_mean,
        "Flow IAT Std":                   flow_iat_std,
        "Flow IAT Max":                   flow_iat_max,
        "Flow IAT Min":                   flow_iat_min,
        # Group 5
        "Fwd IAT Total":                  fwd_iat_total,
        "Fwd IAT Mean":                   fwd_iat_mean,
        "Fwd IAT Std":                    fwd_iat_std,
        "Fwd IAT Max":                    fwd_iat_max,
        "Fwd IAT Min":                    fwd_iat_min,
        "Bwd IAT Total":                  bwd_iat_total,
        "Bwd IAT Mean":                   bwd_iat_mean,
        "Bwd IAT Std":                    bwd_iat_std,
        "Bwd IAT Max":                    bwd_iat_max,
        "Bwd IAT Min":                    bwd_iat_min,
        # Group 6
        "Fwd PSH Flags":                  fwd_psh_flags,
        "Bwd PSH Flags":                  bwd_psh_flags,
        "Fwd URG Flags":                  fwd_urg_flags,
        "Bwd URG Flags":                  bwd_urg_flags,
        # Group 7
        "Fwd Header Length":              fwd_header_len,
        "Bwd Header Length":              bwd_header_len,
        "Fwd Packets/s":                  fwd_pkts_s,
        "Bwd Packets/s":                  bwd_pkts_s,
        # Group 8
        "Min Packet Length":              pkt_len_min,
        "Max Packet Length":              pkt_len_max,
        "Packet Length Mean":             pkt_len_mean,
        "Packet Length Std":              pkt_len_std,
        "Packet Length Variance":         pkt_len_var,
        # Group 6 (flag totals)
        "FIN Flag Count":                 fin_count,
        "SYN Flag Count":                 syn_count,
        "RST Flag Count":                 rst_count,
        "PSH Flag Count":                 psh_count,
        "ACK Flag Count":                 ack_count,
        "URG Flag Count":                 urg_count,
        "CWE Flag Count":                 cwe_count,
        "ECE Flag Count":                 ece_count,
        # Group 9
        "Down/Up Ratio":                  down_up_ratio,
        "Average Packet Size":            avg_pkt_size,
        "Avg Fwd Segment Size":           avg_fwd_seg,
        "Avg Bwd Segment Size":           avg_bwd_seg,
        "Fwd Header Length.1":            fwd_header_len2,
        # Group 10 — bulk
        "Fwd Avg Bytes/Bulk":             fwd_avg_bytes_bulk,
        "Fwd Avg Packets/Bulk":           fwd_avg_pkts_bulk,
        "Fwd Avg Bulk Rate":              fwd_avg_bulk_rate,
        "Bwd Avg Bytes/Bulk":             bwd_avg_bytes_bulk,
        "Bwd Avg Packets/Bulk":           bwd_avg_pkts_bulk,
        "Bwd Avg Bulk Rate":              bwd_avg_bulk_rate,
        # Group 10 — subflow
        "Subflow Fwd Packets":            subflow_fwd_pkts,
        "Subflow Fwd Bytes":              subflow_fwd_bytes,
        "Subflow Bwd Packets":            subflow_bwd_pkts,
        "Subflow Bwd Bytes":              subflow_bwd_bytes,
        # Group 10 — window / data
        "Init_Win_bytes_forward":         init_win_fwd,
        "Init_Win_bytes_backward":        init_win_bwd,
        "act_data_pkt_fwd":               act_data_pkt_fwd,
        "min_seg_size_forward":           min_seg_size_fwd,
        # Group 11
        "Active Mean":                    active_mean,
        "Active Std":                     active_std,
        "Active Max":                     active_max,
        "Active Min":                     active_min,
        "Idle Mean":                      idle_mean,
        "Idle Std":                       idle_std,
        "Idle Max":                       idle_max,
        "Idle Min":                       idle_min,
    }
