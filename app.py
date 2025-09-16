#!/usr/bin/env python3
"""
app.py - simple Flask API to serve heart disease predictions.
POST /predict with JSON:
{
  "features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
}
or
{
  "columns": ["age","sex",...],
  "data": [[...], [...]]
}
"""
from flask import Flask, request, jsonify, render_template
import joblib
import os
import pandas as pd       
import numpy as np

MODEL_PATH = os.environ.get("MODEL_PATH", "models/heart_model.joblib")
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():   # <-- changed from 'home' to 'index'
    return render_template("index.html")

# Load model at startup
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return {
        "message": "✅ Heart Disease Prediction API is running. Use POST on /predict with JSON input."
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def validate_input_json(payload: dict):
    if "features" in payload:
        features = payload["features"]
        X = np.array(features)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X
    elif "data" in payload and "columns" in payload:
        data = payload["data"]
        return np.array(data)
    else:
        raise ValueError("Invalid input format. Use 'features' or 'columns'+'data'.")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Check if frontend sent "columns" + "data"
        if "columns" in data and "data" in data:
            df = pd.DataFrame(data["data"], columns=data["columns"])
        # Or raw JSON format (single record)
        else:
            df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        # Always return arrays for frontend compatibility
        return jsonify({
            "prediction": [int(prediction)],
            "probability": [float(probability)]
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
