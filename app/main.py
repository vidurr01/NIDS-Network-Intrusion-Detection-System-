"""
main.py — IDS Prediction API
==============================
Week 1:  Walking skeleton (/health, /predict, /sample)
Week 2:  /predict calls real feature extractor (flows from sniffer)
Week 3:  XGBoost + Isolation Forest pipeline; SIEM alerting on attacks

Endpoints:
  POST /predict    — accept 77 CIC-IDS-2017 features, return prediction + alert
  GET  /health     — liveness check
  GET  /sample     — returns a real CIC-IDS-2017 test row saved by the notebook
  GET  /alerts     — tail the last N alerts from the local log
  GET  /labels     — list all known attack/benign class names
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.schema import FlowFeatures
from app.predictor import predict, get_labels, MODEL_DIR
from app.alerting import send_alert, SIEM_LOG_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(
    title="IDS Prediction API",
    description=(
        "Intrusion Detection System API.\n\n"
        "**Week 3:** XGBoost + Isolation Forest pipeline trained on "
        "CIC-IDS-2017; SIEM alerting on detected attacks."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Utility"])
def health():
    """Liveness check."""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/labels", tags=["Utility"])
def labels():
    """
    Return all class names the model can predict.
    Derived at runtime from the trained LabelEncoder — never a hardcoded list.
    """
    return {"labels": get_labels()}


@app.get("/sample", tags=["Utility"])
def sample_payload():
    """
    Return a real CIC-IDS-2017 test-set row saved by the Data Team notebook
    (section 2.7 exports models/sample_flow.json).

    Paste the 'payload' object into POST /predict to test end-to-end.
    """
    sample_path = MODEL_DIR / "sample_flow.json"
    if not sample_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "models/sample_flow.json not found. "
                "Re-run Data Team notebook section 2.7 to export it."
            ),
        )
    with open(sample_path, "r", encoding="utf-8") as fh:
        flow = json.load(fh)
    return {
        "description": "Real CIC-IDS-2017 test row — paste 'payload' into POST /predict",
        "payload": flow,
    }


@app.post("/predict", tags=["Prediction"])
def predict_endpoint(features: FlowFeatures):
    """
    Accept one network-flow feature vector and return a prediction.

    Pipeline:
    1. XGBoost → label + confidence.
    2. confidence < threshold → Isolation Forest anomaly check.
    3. anomaly detected → label = "Unknown/Novel Attack".
    4. attack detected → SIEM alert fired (webhook + local log).

    Response example:
    ```json
    {
      "label":         "DDoS",
      "confidence":    0.93,
      "is_attack":     true,
      "shap_values":   {"Flow Bytes/s": 0.42, ...},
      "pipeline_used": "xgboost",
      "latency_ms":    1.23
    }
    ```
    """
    t0 = time.perf_counter()

    feature_dict = features.model_dump(by_alias=True)
    try:
        result = predict(feature_dict)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)

    log.info(
        "Prediction: label=%-30s  confidence=%.2f  pipeline=%s",
        result["label"], result["confidence"], result.get("pipeline_used", "-"),
    )

    send_alert(result, feature_dict)
    return result


@app.get("/alerts", tags=["Utility"])
def recent_alerts(n: int = Query(default=20, ge=1, le=500)):
    """Return the last `n` alerts from the local JSONL log."""
    log_path = Path(SIEM_LOG_FILE)
    if not log_path.exists():
        return {"alerts": [], "message": "No alerts logged yet."}

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    tail  = lines[-n:]

    alerts = []
    for line in reversed(tail):
        try:
            alerts.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    return {"count": len(alerts), "alerts": alerts}
