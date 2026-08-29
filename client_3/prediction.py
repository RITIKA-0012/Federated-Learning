import os
import sys
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================
# CONFIGURATION & PATHS
# ==========================
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "train")

model_path = os.path.join(output_dir, "xgboost_heart_model.json")
scaler_path = os.path.join(output_dir, "xgboost_heart_scaler.pkl")
meta_path = os.path.join(output_dir, "xgboost_metadata.pkl")
confidence_plot_path = os.path.join(output_dir, "xgboost_prediction_confidence.png")

IMPORTANT_FEATURES = [
    'chest pain',
    'Maximum Heart Rate',
    'Major Blood Vessels',
    'ST Depression',
    'Thalassemia Result',
    'age',
    'Cholesterol Level',
    'Resting Blood Pressure'
]


def load_artifacts():
    """Load model, scaler, and metadata. If not found, trigger training first."""
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(meta_path)):
        print("[INFO] Model artifacts not found. Starting training pipeline first...")
        from trained import train_model
        success = train_model()
        if not success:
            raise FileNotFoundError("Failed to train and create XGBoost model artifacts.")

    model = XGBClassifier()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)
    meta = joblib.load(meta_path)
    return model, scaler, meta


def predict_sample(input_dict):
    """Programmatic prediction helper function for a given dictionary of patient features."""
    model, scaler, meta = load_artifacts()
    features = meta.get("features", IMPORTANT_FEATURES)
    defaults = meta.get("defaults", {})

    row = {}
    for feat in features:
        row[feat] = float(input_dict.get(feat, defaults.get(feat, 0.0)))

    df = pd.DataFrame([row])[features]
    scaled = scaler.transform(df)
    prob = float(model.predict_proba(scaled)[0][1])
    is_high_risk = prob >= 0.50

    return {
        "model": "XGBoost Classifier",
        "probability": prob,
        "risk_percentage": prob * 100,
        "safe_percentage": (1.0 - prob) * 100,
        "is_high_risk": is_high_risk,
        "predicted_class": 1 if is_high_risk else 0
    }


def live_prediction():
    """Interactive CLI disease prediction using the trained XGBoost model."""
    model, scaler, meta = load_artifacts()
    features = meta.get("features", IMPORTANT_FEATURES)
    defaults = meta.get("defaults", {})
    accuracy = meta.get("accuracy", 97.07)

    print("\n====================================================")
    print("      XGBOOST LIVE DISEASE PREDICTION SYSTEM        ")
    print("====================================================\n")
    print(f"[INFO] Model Used: XGBoost Classifier (Trained Accuracy: {accuracy:.2f}%)")
    print(f"[INFO] Predicting heart disease risk using the {len(features)} most important features.\n")

    feature_hints = {
        "chest pain": "0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic",
        "Maximum Heart Rate": "Max heart rate achieved (bpm, e.g. 71 - 202)",
        "Major Blood Vessels": "Number of major vessels (0-3) colored by fluoroscopy",
        "ST Depression": "ST depression induced by exercise relative to rest (e.g. 0.0 - 6.2)",
        "Thalassemia Result": "1: normal, 2: fixed defect, 3: reversible defect",
        "age": "Age in years (e.g. 29 - 77)",
        "Cholesterol Level": "Serum cholesterol in mg/dl (e.g. 126 - 564)",
        "Resting Blood Pressure": "Resting blood pressure in mm Hg (e.g. 94 - 200)"
    }

    patient_data = {}
    print("Please enter patient details below (press Enter to accept default values):")
    for feat in features:
        hint = feature_hints.get(feat, "")
        hint_text = f" ({hint})" if hint else ""
        def_val = defaults.get(feat, 0.0)

        try:
            val_str = input(f"-> Enter {feat}{hint_text} [default: {def_val}]: ").strip()
            if not val_str:
                patient_data[feat] = float(def_val)
            else:
                patient_data[feat] = float(val_str)
        except (EOFError, ValueError):
            patient_data[feat] = float(def_val)

    df_input = pd.DataFrame([patient_data])[features]
    scaled_input = scaler.transform(df_input)
    prob = float(model.predict_proba(scaled_input)[0][1])
    risk_pct = prob * 100
    safe_pct = (1.0 - prob) * 100
    is_high_risk = prob >= 0.50

    print("\n====================================================")
    print("                 PREDICTION RESULT                  ")
    print("====================================================")
    if is_high_risk:
        print("[!] HIGH RISK / DISEASE DETECTED")
        print("Predicted Class: 1 (High Risk of Heart Disease)")
    else:
        print("[+] LOW RISK / NO DISEASE DETECTED")
        print("Predicted Class: 0 (Low Risk / Healthy)")

    print(f"\nHeart Disease Risk Probability: {risk_pct:.2f}%")
    print(f"Healthy / Low Risk Confidence:  {safe_pct:.2f}%")
    print(f"Model:                          XGBoost Classifier")
    print("====================================================")

    # Save Confidence Plot
    plt.figure(figsize=(8, 5))
    labels = ["Heart Disease Risk (Class 1)", "Healthy / Low Risk (Class 0)"]
    values = [prob, 1.0 - prob]
    colors = ["#d62728" if is_high_risk else "#ff9896", "#2ca02c" if not is_high_risk else "#98df8a"]

    bars = plt.bar(labels, values, color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height*100:.2f}%", ha="center", fontweight='bold')

    plt.axhline(y=0.50, color="red", linestyle="--", label="Decision Threshold (50.00%)")
    plt.title("XGBoost Heart Disease Risk Prediction Confidence")
    plt.ylabel("Probability")
    plt.ylim(0, 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(confidence_plot_path)
    plt.close()
    print(f"[INFO] Prediction confidence graph saved to: '{confidence_plot_path}'\n")


if __name__ == "__main__":
    live_prediction()
