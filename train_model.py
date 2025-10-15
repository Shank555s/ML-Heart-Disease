#!/usr/bin/env python3
"""
train_model.py - Train CatBoost model for Heart Disease prediction
- Handles categorical features automatically
- Stratified train-test split
- Saves trained model to models/heart_model.cbm
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import joblib

# Paths
DATA_PATH = Path("heart_1000.csv")  # your 1000-record CSV
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "heart_model.cbm"

# Load dataset
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# Features and target
target_col = "target"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical features (CatBoost can handle integer-coded categorical)
categorical_features = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Create CatBoost Pool (special object for CatBoost)
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
test_pool = Pool(X_test, y_test, cat_features=categorical_features)

# Initialize CatBoost classifier
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric="AUC",
    random_seed=42,
    verbose=100,
    class_weights=[1, max(1, y_train.value_counts()[0]/y_train.value_counts()[1])]
)

# Train model
model.fit(train_pool, eval_set=test_pool, use_best_model=True)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test ROC AUC: {roc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
model.save_model(MODEL_PATH)
print(f"Saved CatBoost model to {MODEL_PATH}")

# Save feature names and categorical info
meta = {
    "features": list(X.columns),
    "categorical_features": categorical_features,
    "target": target_col,
}
pd.Series(meta).to_frame("value").to_csv(MODEL_DIR / "model_metadata.csv")
print("Saved model metadata.")
