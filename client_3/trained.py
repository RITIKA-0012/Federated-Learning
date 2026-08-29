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
csv_path = os.path.join(base_dir, "data", "heart_disease_3.csv")
output_dir = os.path.join(base_dir, "train")
os.makedirs(output_dir, exist_ok=True)

# Required artifact file paths inside train/
model_save_path = os.path.join(output_dir, "xgboost_heart_model.json")
scaler_save_path = os.path.join(output_dir, "xgboost_heart_scaler.pkl")
meta_save_path = os.path.join(output_dir, "xgboost_metadata.pkl")
feat_imp_path = os.path.join(output_dir, "xgboost_feature_importance.png")
pr_curve_path = os.path.join(output_dir, "xgboost_precision_recall.png")
cm_path = os.path.join(output_dir, "xgboost_prediction_confusion_matrix.png")

# 13 clinical features for heart_disease_3.csv (Statlog dataset)
FEATURES = [
    'Age',
    'Sex',
    'Chest pain type',
    'BP',
    'Cholesterol',
    'FBS over 120',
    'EKG results',
    'Max HR',
    'Exercise angina',
    'ST depression',
    'Slope of ST',
    'Number of vessels fluro',
    'Thallium'
]


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

    # 2. Handle missing values
    num_cols = data.select_dtypes(include=np.number).columns
    for col in num_cols:
        if data[col].isna().any():
            data[col].fillna(data[col].median(), inplace=True)

    cat_cols = data.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        if data[col].isna().any():
            data[col].fillna(data[col].mode()[0], inplace=True)

    # 3. Align Target
    # In heart_disease_3.csv: 'Heart Disease' column has 'Presence' (1 = High Risk) and 'Absence' (0 = Low Risk)
    if "Heart Disease" in data.columns:
        data["target"] = (data["Heart Disease"].astype(str).str.strip().str.lower() == "presence").astype(int)
        print("[INFO] Target 'Heart Disease' aligned: 1 = Presence (High Risk / Disease Detected), 0 = Absence (Low Risk / No Disease)")
    elif "Status" in data.columns:
        data["target"] = (data["Status"] == 0).astype(int)
        print("[INFO] Target 'Status' aligned: 1 = High Risk (Disease Detected), 0 = Low Risk (No Disease)")
    else:
        target_col = data.columns[-1]
        if data[target_col].dtype in [object, "string"]:
            data["target"] = (data[target_col].astype(str).str.strip().str.lower().isin(["presence", "1", "yes", "positive", "disease", "true"])).astype(int)
        else:
            data["target"] = data[target_col].astype(int)
        print(f"[INFO] Target column '{target_col}' aligned.")

    # 4. Feature Selection: Use all 13 clinical features
    available_features = [f for f in FEATURES if f in data.columns]
    if not available_features:
        # Fallback to all non-target columns if names differ
        available_features = [col for col in data.columns if col not in ["Heart Disease", "Status", "target"]]
    
    print(f"[INFO] Using {len(available_features)} clinical features for training: {available_features}")

    X = data[available_features]
    y = data["target"]

    # 5. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] 80/20 Split -> Training samples: {len(X_train)} (80%), Testing samples: {len(X_test)} (20%)")

    # 6. Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. Apply SMOTE for Balancing
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    # 8. Build and Train XGBoost Classifier
    print("\nTraining XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.01,
        max_depth=3,
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

    # 9. Evaluate Model
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred) * 100
    print("\n--- XGBOOST TEST EVALUATION ---")
    print(f"Model Accuracy: {acc:.2f}%")
    print("\nClassification Report:\n", classification_report(
        y_test, y_pred, target_names=["Low Risk (Absence)", "High Risk (Presence)"], zero_division=0
    ))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # 10. Save Model, Scaler & Metadata
    feature_defaults = X.median().to_dict()
    model.save_model(model_save_path)
    joblib.dump(scaler, scaler_save_path)
    joblib.dump({
        "features": available_features,
        "defaults": feature_defaults,
        "accuracy": acc,
        "test_samples": len(X_test),
        "train_samples": len(X_train)
    }, meta_save_path)

    print(f"\n[SAVED] Model file:              '{model_save_path}'")
    print(f"[SAVED] Scaler file:             '{scaler_save_path}'")
    print(f"[SAVED] Metadata file:           '{meta_save_path}'")

    # 11. Save Confusion Matrix Plot (xgboost_prediction_confusion_matrix.png)
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

    # 12. Save Precision-Recall Plot (xgboost_precision_recall.png)
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

    # 13. Save Feature Importance Plot (xgboost_feature_importance.png)
    plt.figure(figsize=(10, 6))
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
