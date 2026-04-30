"""
predictor.py
─────────────
Week 3 — Real prediction pipeline.

Decision logic:
  1. Run XGBoost classifier → get predicted label + class probabilities.
  2. If max_probability < CONFIDENCE_THRESHOLD → call Isolation Forest.
  3. If Isolation Forest returns -1 (anomaly) → label = "Unknown/Novel Attack".
  4. send_alert() in main.py fires a SIEM webhook if an attack is detected.

Model files (produced by Data Team notebook section 2.7):
  models/xgb_nids.joblib          — trained XGBClassifier  (joblib)
  models/iso_forest.joblib         — trained IsolationForest (joblib)
  models/label_encoder.joblib      — sklearn LabelEncoder   (joblib)
  models/feature_columns.json      — ordered list of feature names (JSON)
  models/sample_flow.json          — one real CIC-IDS-2017 test row (JSON)
                                     used by GET /sample endpoint

Models are loaded lazily on the first call to predict() or get_labels().
A RuntimeError is raised at that point if any required file is missing,
rather than at import time, so the API can start without model files and
return a helpful 503 via the endpoint rather than crashing on startup.
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD: float = float(os.getenv("IDS_CONFIDENCE_THRESHOLD", "0.80"))
MODEL_DIR = Path(__file__).parent.parent / "models"

_xgb_model                  = None
_iso_forest                 = None
_label_encoder              = None
_feature_columns: list[str] = []
_shap_explainer             = None
_models_loaded: bool        = False


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_models() -> None:
    global _xgb_model, _iso_forest, _label_encoder, _feature_columns

    required = {
        "xgb_nids":        MODEL_DIR / "xgb_nids.joblib",
        "iso_forest":      MODEL_DIR / "iso_forest.joblib",
        "label_encoder":   MODEL_DIR / "label_encoder.joblib",
        "feature_columns": MODEL_DIR / "feature_columns.json",
    }

    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise RuntimeError(
            f"Required model files not found: {', '.join(missing)}.\n"
            f"Run the Data Team notebook (section 2.7) and place outputs "
            f"in '{MODEL_DIR}' before starting the API."
        )

    import joblib
    _xgb_model     = joblib.load(required["xgb_nids"])
    _iso_forest    = joblib.load(required["iso_forest"])
    _label_encoder = joblib.load(required["label_encoder"])

    with open(required["feature_columns"], "r", encoding="utf-8") as fh:
        _feature_columns = json.load(fh)

    log.info(
        "Models loaded from '%s' — %d classes, %d features",
        MODEL_DIR, len(_label_encoder.classes_), len(_feature_columns),
    )


def _ensure_models() -> None:
    """Trigger model loading on first use (lazy).  Raises RuntimeError if files missing."""
    global _models_loaded
    if not _models_loaded:
        _load_models()
        _models_loaded = True


# ── Public helpers ────────────────────────────────────────────────────────────

def get_labels() -> list[str]:
    """
    All known class names from the trained LabelEncoder plus the anomaly label.
    Used by tests instead of a hardcoded list so tests never drift from reality.
    """
    _ensure_models()
    return list(_label_encoder.classes_) + ["Unknown/Novel Attack"]


def _align_features(features: dict[str, Any]) -> np.ndarray:
    row = [float(features.get(col, 0.0)) for col in _feature_columns]
    return np.array(row, dtype=np.float32).reshape(1, -1)


def _get_shap_explainer(X_row: np.ndarray):
    """
    Lazy-init SHAP PermutationExplainer, matching the notebook's choice
    (section 1.9: TreeExplainer was abandoned due to XGBoost compat issues).
    """
    global _shap_explainer
    if _shap_explainer is None:
        try:
            import shap
            background      = shap.maskers.Independent(X_row)
            _shap_explainer = shap.Explainer(_xgb_model.predict_proba, background)
        except Exception as exc:
            log.warning("SHAP explainer init failed: %s", exc)
    return _shap_explainer


def _extract_class_shap(sv: Any, pred_idx: int):
    """
    Robustly extract per-feature SHAP values for the predicted class,
    handling the three output shapes that shap==0.45.1 can produce:

      A. list of arrays  (old-style list-per-class)
           sv[pred_idx]         shape (n_samples, n_features)
           sv[pred_idx][0]      shape (n_features,)

      B. 3-D numpy array        (Explanation.values from newer SHAP)
           sv.shape == (n_samples, n_features, n_classes)
           sv[0, :, pred_idx]   shape (n_features,)

      C. 2-D numpy array        (binary / single-output fallback)
           sv.shape == (n_samples, n_features)
           sv[0]                shape (n_features,)

    Returns None if the shape is unrecognised so the caller skips SHAP
    gracefully rather than raising an unhandled exception.
    """
    # Unwrap shap.Explanation object if present
    if hasattr(sv, "values"):
        sv = sv.values

    if isinstance(sv, list):
        # Case A
        try:
            arr = np.array(sv[pred_idx])
            return arr[0] if arr.ndim == 2 else arr
        except (IndexError, TypeError):
            return None

    if isinstance(sv, np.ndarray):
        if sv.ndim == 3:
            # Case B: (samples, features, classes)
            return sv[0, :, pred_idx]
        if sv.ndim == 2:
            # Case C: (samples, features)
            return sv[0]

    return None


# ── Main inference pipeline ───────────────────────────────────────────────────

def predict(features: dict[str, Any]) -> dict[str, Any]:
    """
    Full two-stage inference pipeline.

    Parameters
    ----------
    features : dict — CIC-IDS-2017 column-name keys (from model_dump by_alias).

    Returns
    -------
    dict — label, confidence, is_attack, shap_values, pipeline_used
    """
    _ensure_models()

    X = _align_features(features)

    # Stage 1 — XGBoost
    proba     = _xgb_model.predict_proba(X)[0]
    max_proba = float(proba.max())
    pred_idx  = int(proba.argmax())
    label     = _label_encoder.inverse_transform([pred_idx])[0]
    pipeline  = "xgboost"

    # Stage 2 — Isolation Forest fallback
    if max_proba < CONFIDENCE_THRESHOLD:
        iso_pred = int(_iso_forest.predict(X)[0])
        pipeline = (
            f"xgboost+isolation_forest"
            f"(conf={max_proba:.3f}<{CONFIDENCE_THRESHOLD})"
        )
        if iso_pred == -1:
            label     = "Unknown/Novel Attack"
            max_proba = 1.0 - max_proba

    is_attack = (label != "BENIGN")

    # SHAP — mirrors notebook explain_prediction() (section 2.1)
    shap_values: dict[str, float] = {}
    if label != "Unknown/Novel Attack":
        try:
            import shap  # noqa: F401
            explainer = _get_shap_explainer(X)
            if explainer is not None:
                sv         = explainer.shap_values(X)
                class_shap = _extract_class_shap(sv, pred_idx)
                if class_shap is not None and len(class_shap) == len(_feature_columns):
                    top5 = sorted(
                        zip(_feature_columns, class_shap.tolist()),
                        key=lambda x: abs(x[1]),
                        reverse=True,
                    )[:5]
                    shap_values = {k: round(v, 4) for k, v in top5}
                else:
                    log.warning(
                        "SHAP shape mismatch (got %s, expected %d features) — skipping.",
                        getattr(class_shap, "shape", type(class_shap)),
                        len(_feature_columns),
                    )
        except Exception as exc:
            log.warning("SHAP inference failed: %s", exc)

    return {
        "label":         label,
        "confidence":    round(max_proba, 4),
        "is_attack":     is_attack,
        "shap_values":   shap_values,
        "pipeline_used": pipeline,
    }
