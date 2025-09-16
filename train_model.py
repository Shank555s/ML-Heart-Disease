#!/usr/bin/env python3
"""
train_model.py
- Loads heart.csv
- Builds a sklearn pipeline: imputer -> scaler -> classifier
- Uses GridSearchCV for simple hyperparam tuning
- Saves best model to models/heart_model.joblib
- Prints metrics (accuracy, classification report, roc_auc)
"""
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("heart_1000.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Loaded data shape: {df.shape}")
    return df


def split_features_target(df: pd.DataFrame):
    # assume the target column is 'target' or 'HeartDisease' — adapt if needed
    if "target" in df.columns:
        y = df["target"]
        X = df.drop(columns=["target"])
    elif "heart_disease" in df.columns:
        y = df["heart_disease"]
        X = df.drop(columns=["heart_disease"])
    else:
        # fallback: assume last column is target
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y


def build_pipeline():
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ]
    pipe = Pipeline(steps=steps)
    return pipe


def main():
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)

    # train test split — stratified to keep class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = build_pipeline()

    param_grid = {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [None, 6, 10],
        "clf__min_samples_split": [2, 5],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1
    )

    logger.info("Starting GridSearchCV...")
    grid.fit(X_train, y_train)
    logger.info(f"Best params: {grid.best_params_}")
    best_model = grid.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {acc:.4f}")
    logger.info("Classification report:\n" + classification_report(y_test, y_pred))

    if y_proba is not None:
        roc = roc_auc_score(y_test, y_proba)
        logger.info(f"Test ROC AUC: {roc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")

    model_path = MODEL_DIR / "heart_model.joblib"
    joblib.dump(best_model, model_path)
    logger.info(f"Saved model to {model_path}")

    # Save a small metadata file
    metadata = {
        "model_file": str(model_path),
        "scikit_learn_version": joblib.__version__,
        "notes": "RandomForest pipeline trained with GridSearchCV",
    }
    pd.Series(metadata).to_frame("value").to_csv(MODEL_DIR / "model_metadata.csv")
    logger.info("Saved model metadata.")


if __name__ == "__main__":
    main()
