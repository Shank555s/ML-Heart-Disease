#!/usr/bin/env python3
"""
app.py - Flask API + frontend for Heart Disease prediction using CatBoost.
"""

from flask import Flask, request, jsonify, render_template
from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np
import os

MODEL_PATH = os.environ.get("MODEL_PATH", "models/heart_model.cbm")
app = Flask(__name__)

# Load CatBoost model at startup
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

# Define categorical columns (as per training)
CATEGORICAL_FEATURES = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Frontend can send "columns" + "data" or a single dict of features
        if "columns" in data and "data" in data:
            df = pd.DataFrame(data["data"], columns=data["columns"])
        else:
            df = pd.DataFrame([data])

        # Ensure categorical columns are integers
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(int)

        # Prepare CatBoost Pool
        pool = Pool(df, cat_features=[c for c in CATEGORICAL_FEATURES if c in df.columns])

        # Predict
        prediction = model.predict(pool)
        probability = model.predict_proba(pool)[:, 1]

        return jsonify({
            "prediction": prediction.astype(int).tolist(),
            "probability": probability.astype(float).tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
