"""
alerting.py
────────────
Week 3 — SIEM alerting helpers.

Two responsibilities:
  1. build_alert(prediction)  → construct a structured alert dict.
  2. send_alert(prediction)   → write the alert to a local JSONL log
                                 and optionally POST to a webhook.

Severity mapping (mirrors ui_dashboard.py HIGH_SEVERITY_LABELS /
MEDIUM_SEVERITY_LABELS so the UI and backend agree):
  HIGH   — DDoS variants, DoS variants, Heartbleed, Web Brute Force
  MEDIUM — PortScan, Patator, XSS, SQLi, Infiltration, Bot
  LOW    — anything else that is not BENIGN / Unknown/Novel Attack
  INFO   — Unknown/Novel Attack (anomaly — severity unknown by definition)

BENIGN flows are silently ignored — nothing is written or sent.

Configuration (environment variables):
  SIEM_LOG_FILE   path to the JSONL alert log   (default: siem_alerts.jsonl)
  SIEM_WEBHOOK    HTTP URL to POST alerts to     (default: "" → disabled)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

import httpx

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
SIEM_LOG_FILE: str = os.getenv("SIEM_LOG_FILE", "siem_alerts.jsonl")
SIEM_WEBHOOK:  str = os.getenv("SIEM_WEBHOOK",  "")

# ── Severity lookup tables (must stay in sync with ui_dashboard.py) ───────────
_HIGH_LABELS: frozenset[str] = frozenset({
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "Heartbleed",
    "Web Attack \u2013 Brute Force",   # em-dash variant (CIC-IDS-2017 CSV)
    "Web Attack – Brute Force",         # en-dash variant (just in case)
})

_MEDIUM_LABELS: frozenset[str] = frozenset({
    "PortScan",
    "FTP-Patator",
    "SSH-Patator",
    "Web Attack \u2013 XSS",
    "Web Attack – XSS",
    "Web Attack \u2013 SQL Injection",
    "Web Attack – SQL Injection",
    "Infiltration",
    "Bot",
})


def _severity(label: str) -> str:
    if label in _HIGH_LABELS:
        return "HIGH"
    if label in _MEDIUM_LABELS:
        return "MEDIUM"
    if label == "Unknown/Novel Attack":
        return "INFO"
    if label == "BENIGN":
        return "NONE"
    return "LOW"


# ── Public API ────────────────────────────────────────────────────────────────

def build_alert(prediction: dict[str, Any]) -> dict[str, Any]:
    """
    Build a structured alert dict from a predictor result.

    Parameters
    ----------
    prediction : dict  — output of predictor.predict()

    Returns
    -------
    dict with keys:
        alert_id, timestamp, severity, label,
        confidence, is_attack, shap_values, pipeline_used
    """
    label = prediction.get("label", "UNKNOWN")
    return {
        "alert_id":     str(uuid.uuid4()),
        "timestamp":    time.time(),
        "severity":     _severity(label),
        "label":        label,
        "confidence":   prediction.get("confidence", 0.0),
        "is_attack":    prediction.get("is_attack", False),
        "shap_values":  prediction.get("shap_values", {}),
        "pipeline_used": prediction.get("pipeline_used", ""),
    }


def send_alert(
    prediction: dict[str, Any],
    features:   dict[str, Any] | None = None,
) -> None:
    """
    Fire a SIEM alert for attack predictions.

    BENIGN flows are silently skipped.
    Unknown/Novel Attack alerts are written at INFO severity.

    Steps:
      1. Build the alert dict.
      2. Append it as a JSON line to SIEM_LOG_FILE.
      3. If SIEM_WEBHOOK is set, POST the alert via httpx (best-effort,
         errors are logged but never re-raised so the API response is
         never blocked by a slow or unavailable webhook).

    Parameters
    ----------
    prediction : dict  — output of predictor.predict()
    features   : dict  — original feature dict (attached to webhook payload
                         for downstream SIEM correlation; not stored locally
                         to keep the log file small).
    """
    if not prediction.get("is_attack", False):
        return  # BENIGN — nothing to do

    alert = build_alert(prediction)

    # 1. Write to local JSONL log
    try:
        with open(SIEM_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(alert) + "\n")
    except OSError as exc:
        log.error("Failed to write alert to '%s': %s", SIEM_LOG_FILE, exc)

    # 2. Optionally POST to webhook
    if SIEM_WEBHOOK:
        payload = {**alert}
        if features:
            payload["features"] = features
        try:
            resp = httpx.post(SIEM_WEBHOOK, json=payload, timeout=3.0)
            resp.raise_for_status()
            log.info("SIEM webhook accepted alert %s (HTTP %d)",
                     alert["alert_id"], resp.status_code)
        except Exception as exc:
            log.warning("SIEM webhook delivery failed for alert %s: %s",
                        alert["alert_id"], exc)
