"""
tests/test_week2_week3.py
─────────────────────────
Run with:   pytest tests/ -v

Covers:
  Week 2 — FlowRecord, feature_extractor (all 77 features present + sane values)
  Week 3 — predictor (real models), alerting (local log), /alerts endpoint,
            /predict pipeline_used field, is_attack ↔ label consistency

NOTE: Tests require real model files in models/ (produced by Data Team notebook).
      The label set is loaded from the trained LabelEncoder at runtime via
      get_labels() — never from a hardcoded or mock list.
"""

import sys
import os
import json
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from fastapi.testclient import TestClient

# ─────────────────────────────────────────────────────────────────────────────
# Week 2: Flow + Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

from app.flow import FlowKey, FlowRecord, PacketRecord
from app.feature_extractor import extract_features
from app.schema import FlowFeatures

# Load real label set from trained LabelEncoder — not a hardcoded list
from app.predictor import get_labels
KNOWN_LABELS = get_labels()


def _make_packet(direction="fwd", ts=None, length=100, header=40,
                 flag_syn=0, flag_ack=0, flag_fin=0, flag_psh=0,
                 init_win=0, payload=60):
    return PacketRecord(
        timestamp=ts or time.time(),
        length=length,
        header_length=header,
        direction=direction,
        flag_syn=flag_syn,
        flag_ack=flag_ack,
        flag_fin=flag_fin,
        flag_psh=flag_psh,
        payload_len=payload,
        init_win=init_win,
    )


def _make_flow(n_fwd=5, n_bwd=3, close=True) -> FlowRecord:
    key = FlowKey("192.168.1.1", "10.0.0.1", 54321, 80, 6)
    flow = FlowRecord(key=key, start_time=time.time() - 1.5)
    t = flow.start_time
    for i in range(n_fwd):
        pkt = _make_packet(direction="fwd", ts=t + i * 0.1, length=200,
                           flag_syn=(1 if i == 0 else 0),
                           flag_ack=(1 if i > 0 else 0),
                           init_win=(65535 if i == 0 else 0))
        flow.add_packet(pkt)
    for i in range(n_bwd):
        pkt = _make_packet(direction="bwd", ts=t + 0.05 + i * 0.1, length=150,
                           flag_ack=1,
                           init_win=(32768 if i == 0 else 0))
        flow.add_packet(pkt)
    if close:
        fin = _make_packet(direction="fwd", ts=t + 1.4, flag_fin=1)
        flow.add_packet(fin)
    return flow


class TestFlowRecord:
    def test_packet_count(self):
        flow = _make_flow(n_fwd=4, n_bwd=2, close=False)
        assert len(flow.fwd_packets) == 4
        assert len(flow.bwd_packets) == 2

    def test_fin_expires_flow(self):
        flow = _make_flow(n_fwd=3, n_bwd=2, close=True)
        assert flow.fin_seen is True
        assert flow.is_expired(idle_timeout=9999)

    def test_idle_expires_flow(self):
        key = FlowKey("1.2.3.4", "5.6.7.8", 1234, 80, 6)
        flow = FlowRecord(key=key, start_time=time.time() - 200)
        flow.last_seen = time.time() - 150
        assert flow.is_expired(idle_timeout=120)

    def test_init_win_captured(self):
        flow = _make_flow()
        assert flow.fwd_init_win == 65535
        assert flow.bwd_init_win == 32768


class TestFeatureExtractor:
    EXPECTED_FEATURES = [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Fwd Packet Length Max", "Fwd Packet Length Min",
        "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min",
        "Bwd Packet Length Mean", "Bwd Packet Length Std",
        "Flow Bytes/s", "Flow Packets/s",
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
        "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
        "Fwd Header Length", "Bwd Header Length",
        "Fwd Packets/s", "Bwd Packets/s",
        "Min Packet Length", "Max Packet Length",
        "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
        "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
        "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
        "Down/Up Ratio", "Average Packet Size",
        "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1",
        "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
        "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
        "Subflow Fwd Packets", "Subflow Fwd Bytes",
        "Subflow Bwd Packets", "Subflow Bwd Bytes",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward",
        "act_data_pkt_fwd", "min_seg_size_forward",
        "Active Mean", "Active Std", "Active Max", "Active Min",
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    ]

    def test_all_features_present(self):
        flow = _make_flow()
        feats = extract_features(flow)
        missing = [k for k in self.EXPECTED_FEATURES if k not in feats]
        assert missing == [], f"Missing features: {missing}"

    def test_feature_count(self):
        flow = _make_flow()
        feats = extract_features(flow)
        assert len(feats) == len(self.EXPECTED_FEATURES)

    def test_all_values_are_floats(self):
        flow = _make_flow()
        feats = extract_features(flow)
        for k, v in feats.items():
            assert isinstance(v, (int, float)), f"{k} is {type(v)}"

    def test_no_negative_counts(self):
        flow = _make_flow()
        feats = extract_features(flow)
        for k in ("Total Fwd Packets", "Total Backward Packets",
                  "Total Length of Fwd Packets", "Total Length of Bwd Packets"):
            assert feats[k] >= 0, f"{k} should be >= 0"

    def test_duration_positive(self):
        flow = _make_flow()
        feats = extract_features(flow)
        assert feats["Flow Duration"] > 0

    def test_init_win_preserved(self):
        flow = _make_flow()
        feats = extract_features(flow)
        assert feats["Init_Win_bytes_forward"] == 65535
        assert feats["Init_Win_bytes_backward"] == 32768

    def test_pydantic_schema_accepts_features(self):
        """Extracted features (CIC alias keys) must be accepted by FlowFeatures."""
        flow = _make_flow()
        feats = extract_features(flow)
        obj = FlowFeatures(**feats)
        assert obj.flow_duration == feats["Flow Duration"]

    def test_empty_flow_does_not_crash(self):
        key = FlowKey("1.1.1.1", "2.2.2.2", 1111, 2222, 17)
        flow = FlowRecord(key=key)
        feats = extract_features(flow)
        assert feats["Total Fwd Packets"] == 0
        assert feats["Flow Duration"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Week 3: Predictor (real models)
# ─────────────────────────────────────────────────────────────────────────────

from app.predictor import predict


class TestPredictor:
    def test_returns_required_keys(self):
        result = predict({})
        for key in ("label", "confidence", "is_attack", "shap_values"):
            assert key in result, f"Missing key: {key}"

    def test_label_in_known_labels(self):
        result = predict({})
        assert result["label"] in KNOWN_LABELS, (
            f"Unexpected label '{result['label']}'. "
            f"Known: {KNOWN_LABELS}"
        )

    def test_is_attack_consistent_with_label(self):
        for _ in range(30):
            result = predict({})
            if result["label"] == "BENIGN":
                assert result["is_attack"] is False
            else:
                assert result["is_attack"] is True

    def test_confidence_in_range(self):
        for _ in range(10):
            result = predict({})
            assert 0.0 <= result["confidence"] <= 1.0

    def test_pipeline_used_field_present(self):
        result = predict({})
        assert "pipeline_used" in result


# ─────────────────────────────────────────────────────────────────────────────
# Week 3: Alerting
# ─────────────────────────────────────────────────────────────────────────────

from app.alerting import build_alert, send_alert


class TestAlerting:
    def _attack_pred(self, label="DDoS"):
        return {
            "label":         label,
            "confidence":    0.95,
            "is_attack":     True,
            "shap_values":   {"Flow Bytes/s": 0.4},
            "pipeline_used": "xgboost",
        }

    def test_build_alert_structure(self):
        alert = build_alert(self._attack_pred())
        for key in ("alert_id", "timestamp", "severity", "label",
                    "confidence", "is_attack"):
            assert key in alert

    def test_severity_ddos_is_high(self):
        alert = build_alert(self._attack_pred("DDoS"))
        assert alert["severity"] == "HIGH"

    def test_severity_portscan_is_medium(self):
        alert = build_alert(self._attack_pred("PortScan"))
        assert alert["severity"] == "MEDIUM"

    def test_benign_does_not_alert(self):
        benign_pred = {"label": "BENIGN", "confidence": 0.99,
                       "is_attack": False, "shap_values": {}}
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            import app.alerting as alert_mod
            original = alert_mod.SIEM_LOG_FILE
            alert_mod.SIEM_LOG_FILE = tmp.name
            send_alert(benign_pred)
            alert_mod.SIEM_LOG_FILE = original
        content = open(tmp.name).read()
        assert content == ""

    def test_attack_writes_to_log(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False,
                                        mode="w") as tmp:
            tmp_path = tmp.name

        import app.alerting as alert_mod
        original = alert_mod.SIEM_LOG_FILE
        alert_mod.SIEM_LOG_FILE = tmp_path
        send_alert(self._attack_pred("DDoS"))
        alert_mod.SIEM_LOG_FILE = original

        lines = open(tmp_path).read().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["label"] == "DDoS"
        assert data["severity"] == "HIGH"


# ─────────────────────────────────────────────────────────────────────────────
# Week 3: API Integration (/predict + /alerts)
# ─────────────────────────────────────────────────────────────────────────────

from app.main import app as fastapi_app

client = TestClient(fastapi_app)


class TestAPIWeek3:
    def test_predict_returns_pipeline_used(self):
        r = client.post("/predict", json={})
        assert r.status_code == 200
        assert "pipeline_used" in r.json()

    def test_predict_latency_present(self):
        r = client.post("/predict", json={})
        assert r.json()["latency_ms"] >= 0

    def test_predict_with_extracted_features(self):
        """Full Week-2 → Week-3 pipeline: real flow → extract → predict."""
        flow = _make_flow(n_fwd=10, n_bwd=7)
        feats = extract_features(flow)
        r = client.post("/predict", json=feats)
        assert r.status_code == 200
        body = r.json()
        assert body["label"] in KNOWN_LABELS

    def test_alerts_endpoint_returns_list(self):
        r = client.get("/alerts")
        assert r.status_code == 200
        assert "alerts" in r.json()

    def test_alerts_n_parameter(self):
        r = client.get("/alerts?n=5")
        assert r.status_code == 200
        assert len(r.json()["alerts"]) <= 5

    def test_health_still_works(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_labels_endpoint(self):
        """Labels must come from the real encoder, not a hardcoded list."""
        r = client.get("/labels")
        assert r.status_code == 200
        labels = r.json()["labels"]
        assert "BENIGN" in labels
        assert "Unknown/Novel Attack" in labels
