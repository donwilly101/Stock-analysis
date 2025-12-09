
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI

# ===== Load Artifacts =====
model = joblib.load("artifacts/xgb_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
feature_cols = joblib.load("artifacts/feature_cols.pkl")

app = FastAPI(
    title="Stock Trend Prediction API",
    description="Predict 0=Downtrend, 1=Sideways, 2=Uptrend using the tuned XGBoost model",
    version="1.0.0",
)

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Model API active",
        "n_features": len(feature_cols),
        "features_required": feature_cols,
    }


@app.post("/predict")
def predict(features: dict):
    import csv
    from datetime import datetime
    import os

    # Convert input dict to DataFrame
    df = pd.DataFrame([features])

    # Reorder columns to match training
    try:
        df = df[feature_cols]
    except KeyError as e:
        return {
            "status": "error",
            "message": f"Feature mismatch: {e}",
            "expected_features": feature_cols,
        }

    # Scale inputs
    X_scaled = scaler.transform(df.values)

    # Predict probabilities
    probs = model.predict_proba(X_scaled)[0]
    pred_class = int(np.argmax(probs))

    label_map = {0: "Downtrend", 1: "Sideways", 2: "Uptrend"}
    trend_label = label_map[pred_class]

    # ===== Simple logging to CSV for monitoring =====
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", "predictions_log.csv")
    write_header = not os.path.exists(log_file)

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "prediction_class": pred_class,
        "trend_label": trend_label,
        "prob_class_0": float(probs[0]),
        "prob_class_1": float(probs[1]),
        "prob_class_2": float(probs[2]),
    }
    # add input features to the row
    for k, v in features.items():
        row[f"feat_{k}"] = v

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    # ================================================

    return {
        "status": "success",
        "prediction_class": pred_class,
        "trend_label": trend_label,
        "probabilities": {
            "class_0": float(probs[0]),
            "class_1": float(probs[1]),
            "class_2": float(probs[2]),
        },
    }

    # Scale inputs
    X_scaled = scaler.transform(df.values)

    # Predict probabilities
    probs = model.predict_proba(X_scaled)[0]
    pred_class = int(np.argmax(probs))

    label_map = {0: "Downtrend", 1: "Sideways", 2: "Uptrend"}

    return {
        "status": "success",
        "prediction_class": pred_class,
        "trend_label": label_map[pred_class],
        "probabilities": {
            "class_0": float(probs[0]),
            "class_1": float(probs[1]),
            "class_2": float(probs[2]),
        },
    }
