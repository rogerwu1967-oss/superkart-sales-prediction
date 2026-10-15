"""SuperKart sales prediction API (V5).

Run from this directory with: gunicorn --bind 0.0.0.0:7860 app:superkart_api
"""
from __future__ import annotations

import hashlib
import math
import os
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from flask import Flask, jsonify, request

superkart_api = Flask("SuperKart")
MODEL_PATH = Path(__file__).resolve().with_name("superkart_model.joblib")
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
    "Store_Age_Years", "Product_Type_Category",
]
NUMERIC_FEATURES = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years"]


def _validate_record(data):
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object."
    missing = [name for name in FEATURES if name not in data]
    if missing:
        return None, f"Missing required features: {missing}"
    row = {name: data[name] for name in FEATURES}
    for name in FEATURES:
        if row[name] is None or (isinstance(row[name], str) and not row[name].strip()):
            return None, f"Feature '{name}' cannot be empty or null."
    for name in NUMERIC_FEATURES:
        try:
            value = float(row[name])
        except (TypeError, ValueError):
            return None, f"Feature '{name}' must be numeric."
        if not math.isfinite(value):
            return None, f"Feature '{name}' must be finite."
        row[name] = value
    return pd.DataFrame([row], columns=FEATURES), None


@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart System"


@superkart_api.get("/health")
def health():
    return jsonify({"status": "ok"})


@superkart_api.get("/v1/info")
def info():
    raw = MODEL_PATH.read_bytes()
    return jsonify({
        "name": "SuperKart",
        "version": "v5",
        "model": type(model.steps[-1][1]).__name__ if hasattr(model, "steps") else type(model).__name__,
        "model_md5": hashlib.md5(raw).hexdigest(),
        "model_sha256": hashlib.sha256(raw).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "features": FEATURES,
        "endpoints": ["/health", "/v1/info", "/v1/predict", "/v1/predictbatch"],
    })


@superkart_api.post("/v1/predict")
def predict_sales():
    data = request.get_json(silent=True)
    frame, error = _validate_record(data)
    if error:
        return jsonify({"error": error}), 400
    try:
        prediction = float(model.predict(frame)[0])
        if not math.isfinite(prediction):
            return jsonify({"error": "Model returned a non-finite prediction."}), 500
        # Do not clamp negative results: returning them exposes model/version/schema problems.
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
    missing = [name for name in FEATURES if name not in input_data.columns]
    if missing:
        return jsonify({"error": f"CSV is missing required columns: {missing}"}), 400
    if input_data.empty:
        return jsonify({"error": "CSV contains no data rows."}), 400
    try:
        predictions = np.asarray(model.predict(input_data[FEATURES]), dtype=float)
        if not np.isfinite(predictions).all():
            return jsonify({"error": "Model returned a non-finite prediction."}), 500
        # Keep the row-indexed JSON format expected by the V4/V5 notebook.
        return jsonify({str(i): round(float(value), 2) for i, value in enumerate(predictions)})
    except Exception:
        superkart_api.logger.exception("Batch prediction failed")
        return jsonify({"error": "Batch prediction failed; check backend logs."}), 500


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
