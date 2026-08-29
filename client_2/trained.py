import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================
# CONFIGURATION & PATHS
# ==========================
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "data", "heart_disease_2.csv")
output_dir = os.path.join(base_dir, "train")
os.makedirs(output_dir, exist_ok=True)

# Required artifact file paths inside train/
model_save_path = os.path.join(output_dir, "xgboost_heart_model.json")
scaler_save_path = os.path.join(output_dir, "xgboost_heart_scaler.pkl")
meta_save_path = os.path.join(output_dir, "xgboost_metadata.pkl")
feat_imp_path = os.path.join(output_dir, "xgboost_feature_importance.png")
pr_curve_path = os.path.join(output_dir, "xgboost_precision_recall.png")
cm_path = os.path.join(output_dir, "xgboost_prediction_confusion_matrix.png")

# All 20 clinical & lifestyle features for heart_disease_2.csv
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

# Categorical encodings
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


def train_model():
    print("====================================================")
    print("        XGBOOST MODEL TRAINING PIPELINE             ")
    print("====================================================\n")

    # 1. Load Data
    if not os.path.exists(csv_path):
        print(f"[ERROR] Dataset not found at: {csv_path}")
        return False

    data = pd.read_csv(csv_path)
    print(f"[INFO] Successfully loaded dataset: '{csv_path}' ({len(data)} rows, {len(data.columns)} columns)")

    # 2. Preprocess & Encode Features
    available_features = [f for f in FEATURES if f in data.columns]
    if not available_features:
        available_features = [c for c in data.columns if c not in ['Heart Disease Status', 'Heart Disease', 'Status', 'target']]

    print(f"[INFO] Processing {len(available_features)} clinical features for training.")

    for col in available_features:
        if col in CATEGORICAL_MAPPINGS:
            # Clean string mapping
            str_series = data[col].astype(str).str.strip().str.lower()
            data[col] = str_series.map(CATEGORICAL_MAPPINGS[col])
        
        # Ensure strictly numeric
        data[col] = pd.to_numeric(data[col], errors="coerce")
        # Impute missing values with column median
        med_val = data[col].median()
        if pd.isna(med_val):
            med_val = 0.0
        data[col] = data[col].fillna(med_val)

    # 3. Align Target
    if "Heart Disease Status" in data.columns:
        data["target"] = (data["Heart Disease Status"].astype(str).str.strip().str.lower() == "yes").astype(int)
        print("[INFO] Target 'Heart Disease Status' aligned: 1 = Yes (High Risk / Disease Detected), 0 = No (Low Risk / Healthy)")
    elif "Heart Disease" in data.columns:
        data["target"] = (data["Heart Disease"].astype(str).str.strip().str.lower() == "presence").astype(int)
        print("[INFO] Target 'Heart Disease' aligned: 1 = Presence (High Risk), 0 = Absence (Low Risk)")
    elif "Status" in data.columns:
        data["target"] = (data["Status"] == 0).astype(int)
        print("[INFO] Target 'Status' aligned: 1 = High Risk, 0 = Low Risk")
    else:
        target_col = data.columns[-1]
        data["target"] = pd.to_numeric(data[target_col], errors="coerce").fillna(0).astype(int)
        print(f"[INFO] Target column '{target_col}' aligned.")

    X = data[available_features]
    y = data["target"]

    # 4. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] 80/20 Split -> Training samples: {len(X_train)} (80%), Testing samples: {len(X_test)} (20%)")

    # 5. Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Apply SMOTE for Balancing
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    # 7. Build and Train XGBoost Classifier
    print("\nTraining XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=2,
        gamma=0.1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        objective="binary:logistic",
        eval_metric="logloss"
    )

    model.fit(
        X_train_res,
        y_train_res,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )

    # 8. Evaluate Model
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred) * 100
    print("\n--- XGBOOST TEST EVALUATION ---")
    print(f"Model Accuracy: {acc:.2f}%")
    print("\nClassification Report:\n", classification_report(
        y_test, y_pred, target_names=["Low Risk (No Disease)", "High Risk (Disease Detected)"], zero_division=0
    ))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # 9. Save Model, Scaler & Metadata
    feature_defaults = X.median().to_dict()
    model.save_model(model_save_path)
    joblib.dump(scaler, scaler_save_path)
    joblib.dump({
        "features": available_features,
        "defaults": feature_defaults,
        "categorical_mappings": CATEGORICAL_MAPPINGS,
        "accuracy": acc,
        "test_samples": len(X_test),
        "train_samples": len(X_train)
    }, meta_save_path)

    print(f"\n[SAVED] Model file:              '{model_save_path}'")
    print(f"[SAVED] Scaler file:             '{scaler_save_path}'")
    print(f"[SAVED] Metadata file:           '{meta_save_path}'")

    # 10. Save Confusion Matrix Plot (xgboost_prediction_confusion_matrix.png)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title("XGBoost Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.xticks([0, 1], ["Low Risk (0)", "High Risk (1)"])
    plt.yticks([0, 1], ["Low Risk (0)", "High Risk (1)"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"[SAVED] Confusion Matrix Plot:   '{cm_path}'")

    # 11. Save Precision-Recall Plot (xgboost_precision_recall.png)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions[:-1], label="Precision", color="#1f77b4", linewidth=2)
    plt.plot(thresholds, recalls[:-1], label="Recall", color="#ff7f0e", linewidth=2)
    plt.title("XGBoost Precision-Recall vs Threshold")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(pr_curve_path)
    plt.close()
    print(f"[SAVED] Precision-Recall Plot:   '{pr_curve_path}'")

    # 12. Save Feature Importance Plot (xgboost_feature_importance.png)
    plt.figure(figsize=(10, 8))
    sorted_idx = np.argsort(model.feature_importances_)
    plt.barh([available_features[i] for i in sorted_idx], model.feature_importances_[sorted_idx], color="#008080")
    plt.title("XGBoost Feature Importance Ranking")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Clinical Feature")
    plt.tight_layout()
    plt.savefig(feat_imp_path)
    plt.close()
    print(f"[SAVED] Feature Importance Plot: '{feat_imp_path}'")

    print("\n[SUCCESS] Training pipeline completed successfully. All artifacts saved in 'train/' folder.\n")
    return True


if __name__ == "__main__":
    train_model()
