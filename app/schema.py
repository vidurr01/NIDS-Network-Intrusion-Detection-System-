"""
schema.py
──────────
Pydantic v2 request model for POST /predict.

Every field uses the exact CIC-IDS-2017 column name as its alias so that
  (a) feature_extractor.py can pass its dict directly, and
  (b) downstream code calls model.model_dump(by_alias=True) to recover
      the original column names before feeding them to the ML pipeline.

All fields default to 0.0 — missing features are treated as zero by the
predictor's _align_features() function anyway, so this is consistent.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict


class FlowFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # ── Group 1: Basic counts ──────────────────────────────────────────────
    flow_duration:            float = Field(0.0, alias="Flow Duration")
    total_fwd_packets:        float = Field(0.0, alias="Total Fwd Packets")
    total_bwd_packets:        float = Field(0.0, alias="Total Backward Packets")
    total_len_fwd:            float = Field(0.0, alias="Total Length of Fwd Packets")
    total_len_bwd:            float = Field(0.0, alias="Total Length of Bwd Packets")

    # ── Group 2: Per-direction packet length stats ─────────────────────────
    fwd_pkt_len_max:          float = Field(0.0, alias="Fwd Packet Length Max")
    fwd_pkt_len_min:          float = Field(0.0, alias="Fwd Packet Length Min")
    fwd_pkt_len_mean:         float = Field(0.0, alias="Fwd Packet Length Mean")
    fwd_pkt_len_std:          float = Field(0.0, alias="Fwd Packet Length Std")
    bwd_pkt_len_max:          float = Field(0.0, alias="Bwd Packet Length Max")
    bwd_pkt_len_min:          float = Field(0.0, alias="Bwd Packet Length Min")
    bwd_pkt_len_mean:         float = Field(0.0, alias="Bwd Packet Length Mean")
    bwd_pkt_len_std:          float = Field(0.0, alias="Bwd Packet Length Std")

    # ── Group 3: Flow byte / packet rates ─────────────────────────────────
    flow_bytes_s:             float = Field(0.0, alias="Flow Bytes/s")
    flow_packets_s:           float = Field(0.0, alias="Flow Packets/s")

    # ── Group 4: Flow IAT stats ────────────────────────────────────────────
    flow_iat_mean:            float = Field(0.0, alias="Flow IAT Mean")
    flow_iat_std:             float = Field(0.0, alias="Flow IAT Std")
    flow_iat_max:             float = Field(0.0, alias="Flow IAT Max")
    flow_iat_min:             float = Field(0.0, alias="Flow IAT Min")

    # ── Group 5: Fwd / Bwd IAT stats ──────────────────────────────────────
    fwd_iat_total:            float = Field(0.0, alias="Fwd IAT Total")
    fwd_iat_mean:             float = Field(0.0, alias="Fwd IAT Mean")
    fwd_iat_std:              float = Field(0.0, alias="Fwd IAT Std")
    fwd_iat_max:              float = Field(0.0, alias="Fwd IAT Max")
    fwd_iat_min:              float = Field(0.0, alias="Fwd IAT Min")
    bwd_iat_total:            float = Field(0.0, alias="Bwd IAT Total")
    bwd_iat_mean:             float = Field(0.0, alias="Bwd IAT Mean")
    bwd_iat_std:              float = Field(0.0, alias="Bwd IAT Std")
    bwd_iat_max:              float = Field(0.0, alias="Bwd IAT Max")
    bwd_iat_min:              float = Field(0.0, alias="Bwd IAT Min")

    # ── Group 6: Per-direction flag counts ────────────────────────────────
    fwd_psh_flags:            float = Field(0.0, alias="Fwd PSH Flags")
    bwd_psh_flags:            float = Field(0.0, alias="Bwd PSH Flags")
    fwd_urg_flags:            float = Field(0.0, alias="Fwd URG Flags")
    bwd_urg_flags:            float = Field(0.0, alias="Bwd URG Flags")

    # ── Group 7: Header lengths + per-direction rates ──────────────────────
    fwd_header_length:        float = Field(0.0, alias="Fwd Header Length")
    bwd_header_length:        float = Field(0.0, alias="Bwd Header Length")
    fwd_packets_s:            float = Field(0.0, alias="Fwd Packets/s")
    bwd_packets_s:            float = Field(0.0, alias="Bwd Packets/s")

    # ── Group 8: Aggregate packet length stats ────────────────────────────
    min_packet_length:        float = Field(0.0, alias="Min Packet Length")
    max_packet_length:        float = Field(0.0, alias="Max Packet Length")
    packet_length_mean:       float = Field(0.0, alias="Packet Length Mean")
    packet_length_std:        float = Field(0.0, alias="Packet Length Std")
    packet_length_variance:   float = Field(0.0, alias="Packet Length Variance")

    # ── Group 6 (cont.): Total TCP flag counts ────────────────────────────
    fin_flag_count:           float = Field(0.0, alias="FIN Flag Count")
    syn_flag_count:           float = Field(0.0, alias="SYN Flag Count")
    rst_flag_count:           float = Field(0.0, alias="RST Flag Count")
    psh_flag_count:           float = Field(0.0, alias="PSH Flag Count")
    ack_flag_count:           float = Field(0.0, alias="ACK Flag Count")
    urg_flag_count:           float = Field(0.0, alias="URG Flag Count")
    cwe_flag_count:           float = Field(0.0, alias="CWE Flag Count")
    ece_flag_count:           float = Field(0.0, alias="ECE Flag Count")

    # ── Group 9: Ratios, averages, segment sizes ──────────────────────────
    down_up_ratio:            float = Field(0.0, alias="Down/Up Ratio")
    average_packet_size:      float = Field(0.0, alias="Average Packet Size")
    avg_fwd_segment_size:     float = Field(0.0, alias="Avg Fwd Segment Size")
    avg_bwd_segment_size:     float = Field(0.0, alias="Avg Bwd Segment Size")
    fwd_header_length2:       float = Field(0.0, alias="Fwd Header Length.1")

    # ── Group 10a: Bulk features ──────────────────────────────────────────
    fwd_avg_bytes_bulk:       float = Field(0.0, alias="Fwd Avg Bytes/Bulk")
    fwd_avg_packets_bulk:     float = Field(0.0, alias="Fwd Avg Packets/Bulk")
    fwd_avg_bulk_rate:        float = Field(0.0, alias="Fwd Avg Bulk Rate")
    bwd_avg_bytes_bulk:       float = Field(0.0, alias="Bwd Avg Bytes/Bulk")
    bwd_avg_packets_bulk:     float = Field(0.0, alias="Bwd Avg Packets/Bulk")
    bwd_avg_bulk_rate:        float = Field(0.0, alias="Bwd Avg Bulk Rate")

    # ── Group 10b: Subflow features ───────────────────────────────────────
    subflow_fwd_packets:      float = Field(0.0, alias="Subflow Fwd Packets")
    subflow_fwd_bytes:        float = Field(0.0, alias="Subflow Fwd Bytes")
    subflow_bwd_packets:      float = Field(0.0, alias="Subflow Bwd Packets")
    subflow_bwd_bytes:        float = Field(0.0, alias="Subflow Bwd Bytes")

    # ── Group 10c: Window / active-data features ──────────────────────────
    init_win_bytes_forward:   float = Field(0.0, alias="Init_Win_bytes_forward")
    init_win_bytes_backward:  float = Field(0.0, alias="Init_Win_bytes_backward")
    act_data_pkt_fwd:         float = Field(0.0, alias="act_data_pkt_fwd")
    min_seg_size_forward:     float = Field(0.0, alias="min_seg_size_forward")

    # ── Group 11: Active / Idle period stats ──────────────────────────────
    active_mean:              float = Field(0.0, alias="Active Mean")
    active_std:               float = Field(0.0, alias="Active Std")
    active_max:               float = Field(0.0, alias="Active Max")
    active_min:               float = Field(0.0, alias="Active Min")
    idle_mean:                float = Field(0.0, alias="Idle Mean")
    idle_std:                 float = Field(0.0, alias="Idle Std")
    idle_max:                 float = Field(0.0, alias="Idle Max")
    idle_min:                 float = Field(0.0, alias="Idle Min")
