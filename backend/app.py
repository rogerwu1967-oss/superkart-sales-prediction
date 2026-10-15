"""SuperKart sales prediction API (Flask).

Endpoints
  GET  /                 welcome message
  GET  /health           liveness check
  GET  /v1/info          model fingerprint (MD5) + library versions -> used to verify the deployment
  POST /v1/predict       single prediction, JSON body with the 10 features
  POST /v1/predictbatch  batch prediction, CSV file in form field "file"
"""
import hashlib, math, platform, logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from flask import Flask, jsonify, request

superkart_api = Flask("SuperKart")

# ---- load the serialized pipeline (preprocessing + model) ----
MODEL_PATH = Path(__file__).resolve().with_name("superkart_model.joblib")
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
MODEL_BYTES = MODEL_PATH.read_bytes()
MODEL_MD5 = hashlib.md5(MODEL_BYTES).hexdigest()
model = joblib.load(MODEL_PATH)
MODEL_NAME = type(model.steps[-1][1]).__name__ if hasattr(model, "steps") else type(model).__name__

RUNTIME = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit-learn": sklearn.__version__,
    "xgboost": xgboost.__version__,
}
# Printed at start-up so `docker logs backend` shows which model / versions the container runs
logging.basicConfig(level=logging.INFO)
logging.getLogger("SuperKart").info("Loaded %s | model_md5=%s | %s", MODEL_NAME, MODEL_MD5, RUNTIME)

# Features expected by the model (same names as in training; the pipeline selects them by name)
FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
    "Store_Age_Years", "Product_Type_Category",
]
NUMERIC_FEATURES = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years"]


def _validate_record(data):
    """Return (DataFrame, None) for a valid record or (None, error message)."""
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object."
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return None, f"Missing required features: {missing}"
    row = {f: data[f] for f in FEATURES}
    for f in FEATURES:
        if row[f] is None or (isinstance(row[f], str) and not row[f].strip()):
            return None, f"Feature '{f}' cannot be empty."
    for f in NUMERIC_FEATURES:
        try:
            value = float(row[f])
        except (TypeError, ValueError):
            return None, f"Feature '{f}' must be numeric."
        if not math.isfinite(value):
            return None, f"Feature '{f}' must be finite."
        row[f] = value
    return pd.DataFrame([row], columns=FEATURES), None


@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart System"


@superkart_api.get("/health")
def health():
    return jsonify({"status": "ok"})


@superkart_api.get("/v1/info")
def info():
    return jsonify({
        "name": "SuperKart",
        "model": MODEL_NAME,
        "model_md5": MODEL_MD5,
        "model_size_kb": round(len(MODEL_BYTES) / 1024, 1),
        **RUNTIME,
        "features": FEATURES,
        "endpoints": ["/health", "/v1/info", "/v1/predict", "/v1/predictbatch"],
    })


@superkart_api.post("/v1/predict")
def predict_sales():
    frame, error = _validate_record(request.get_json(silent=True))
    if error:
        return jsonify({"error": error}), 400
    try:
        prediction = float(model.predict(frame)[0])
        if not math.isfinite(prediction):
            return jsonify({"error": "Model returned a non-finite prediction."}), 500
        # Negative values are NOT clamped: they would indicate a model/version mismatch that must be fixed.
        return jsonify({"Sales": round(prediction, 2)})
    except Exception:
        superkart_api.logger.exception("Single prediction failed")
        return jsonify({"error": "Prediction failed; check backend logs."}), 500


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify({"error": "Upload a CSV file using form field 'file'."}), 400
    try:
        input_data = pd.read_csv(uploaded)
    except Exception as exc:
        return jsonify({"error": f"Could not read CSV: {exc}"}), 400
    missing = [f for f in FEATURES if f not in input_data.columns]
    if missing:
        return jsonify({"error": f"CSV is missing required columns: {missing}"}), 400
    if input_data.empty:
        return jsonify({"error": "CSV contains no data rows."}), 400
    try:
        predictions = np.asarray(model.predict(input_data[FEATURES]), dtype=float)
        if not np.isfinite(predictions).all():
            return jsonify({"error": "Model returned a non-finite prediction."}), 500
        # Row-indexed JSON: {"0": 2842.4, "1": ...}
        return jsonify({str(i): round(float(v), 2) for i, v in enumerate(predictions)})
    except Exception:
        superkart_api.logger.exception("Batch prediction failed")
        return jsonify({"error": "Batch prediction failed; check backend logs."}), 500


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
