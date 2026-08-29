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

# All 20 clinical features for heart_disease_2.csv
FEATURES = [
    'Age',
    'Gender',
    'Blood Pressure',
    'Cholesterol Level',
    'Exercise Habits',
    'Smoking',
    'Family Heart Disease',
    'Diabetes',
    'BMI',
    'High Blood Pressure',
    'Low HDL Cholesterol',
    'High LDL Cholesterol',
    'Alcohol Consumption',
    'Stress Level',
    'Sleep Hours',
    'Sugar Consumption',
    'Triglyceride Level',
    'Fasting Blood Sugar',
    'CRP Level',
    'Homocysteine Level'
]

CATEGORICAL_MAPPINGS = {
    'Gender': {'male': 1, 'female': 0, '1': 1, '0': 0},
    'Exercise Habits': {'low': 0, 'medium': 1, 'high': 2, '0': 0, '1': 1, '2': 2},
    'Smoking': {'no': 0, 'yes': 1, '0': 0, '1': 1},
    'Family Heart Disease': {'no': 0, 'yes': 1, '0': 0, '1': 1},
    'Diabetes': {'no': 0, 'yes': 1, '0': 0, '1': 1},
    'High Blood Pressure': {'no': 0, 'yes': 1, '0': 0, '1': 1},
    'Low HDL Cholesterol': {'no': 0, 'yes': 1, '0': 0, '1': 1},
    'High LDL Cholesterol': {'no': 0, 'yes': 1, '0': 0, '1': 1},
    'Alcohol Consumption': {'low': 0, 'medium': 1, 'high': 2, '0': 0, '1': 1, '2': 2},
    'Stress Level': {'low': 0, 'medium': 1, 'high': 2, '0': 0, '1': 1, '2': 2},
    'Sugar Consumption': {'low': 0, 'medium': 1, 'high': 2, '0': 0, '1': 1, '2': 2},
}


def encode_feature_value(feature_name, value, default_val=0.0):
    """Helper to convert string or numeric values into normalized floats."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return float(default_val)

    if feature_name in CATEGORICAL_MAPPINGS:
        val_str = str(value).strip().lower()
        if val_str in CATEGORICAL_MAPPINGS[feature_name]:
            return float(CATEGORICAL_MAPPINGS[feature_name][val_str])

    try:
        return float(value)
    except (ValueError, TypeError):
        return float(default_val)


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
    features = meta.get("features", FEATURES)
    defaults = meta.get("defaults", {})

    row = {}
    for feat in features:
        raw_val = input_dict.get(feat, None)
        def_val = defaults.get(feat, 0.0)
        row[feat] = encode_feature_value(feat, raw_val if raw_val is not None else def_val, def_val)

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
    features = meta.get("features", FEATURES)
    defaults = meta.get("defaults", {})
    accuracy = meta.get("accuracy", 80.0)

    print("\n====================================================")
    print("      XGBOOST LIVE DISEASE PREDICTION SYSTEM        ")
    print("====================================================\n")
    print(f"[INFO] Model Used: XGBoost Classifier (Trained Accuracy: {accuracy:.2f}%)")
    print(f"[INFO] Predicting heart disease risk using {len(features)} clinical & lifestyle features.\n")

    feature_hints = {
        "Age": "Age in years (e.g. 18 - 80)",
        "Gender": "Male / Female or 1 / 0",
        "Blood Pressure": "Resting systolic blood pressure (e.g. 90 - 180 mmHg)",
        "Cholesterol Level": "Total serum cholesterol (e.g. 150 - 300 mg/dL)",
        "Exercise Habits": "Low / Medium / High (or 0 / 1 / 2)",
        "Smoking": "Yes / No (or 1 / 0)",
        "Family Heart Disease": "Yes / No (or 1 / 0)",
        "Diabetes": "Yes / No (or 1 / 0)",
        "BMI": "Body Mass Index (e.g. 15.0 - 45.0)",
        "High Blood Pressure": "Yes / No (or 1 / 0)",
        "Low HDL Cholesterol": "Yes / No (or 1 / 0)",
        "High LDL Cholesterol": "Yes / No (or 1 / 0)",
        "Alcohol Consumption": "Low / Medium / High (or 0 / 1 / 2)",
        "Stress Level": "Low / Medium / High (or 0 / 1 / 2)",
        "Sleep Hours": "Average daily sleep hours (e.g. 4.0 - 10.0)",
        "Sugar Consumption": "Low / Medium / High (or 0 / 1 / 2)",
        "Triglyceride Level": "Blood triglycerides (e.g. 100 - 500 mg/dL)",
        "Fasting Blood Sugar": "Fasting blood sugar (e.g. 70 - 200 mg/dL)",
        "CRP Level": "C-Reactive Protein level (e.g. 0.5 - 20.0 mg/L)",
        "Homocysteine Level": "Homocysteine level (e.g. 5.0 - 25.0 µmol/L)"
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
                patient_data[feat] = encode_feature_value(feat, val_str, def_val)
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
