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
csv_path = os.path.join(base_dir, "data", "heart_disease_uci.csv")
output_dir = os.path.join(base_dir, "train")
os.makedirs(output_dir, exist_ok=True)

# Required artifact file paths inside train/
model_save_path = os.path.join(output_dir, "xgboost_heart_model.json")
scaler_save_path = os.path.join(output_dir, "xgboost_heart_scaler.pkl")
meta_save_path = os.path.join(output_dir, "xgboost_metadata.pkl")
feat_imp_path = os.path.join(output_dir, "xgboost_feature_importance.png")
pr_curve_path = os.path.join(output_dir, "xgboost_precision_recall.png")
cm_path = os.path.join(output_dir, "xgboost_prediction_confusion_matrix.png")

# Top 8 important features identified from heart_disease_1.csv
IMPORTANT_FEATURES = [
    'cp', 'thalch', 'ca', 'oldpeak', 'thal', 'age', 'chol', 'trestbps'
]

FEATURE_ENCODINGS = {
    "cp": {
        "typical angina": 0,
        "atypical angina": 1,
        "non-anginal": 2,
        "asymptomatic": 3,
    },
    "thal": {
        "normal": 0,
        "fixed defect": 1,
        "reversable defect": 2,
    },
}


def prepare_features(data):
    features = data[IMPORTANT_FEATURES].copy()
    for column, encoding in FEATURE_ENCODINGS.items():
        features[column] = features[column].map(encoding)
    return features.astype(float)


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
            data[col] = data[col].fillna(data[col].median())

    cat_cols = data.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].mode()[0])

    # 3. Align the UCI target: num=0 is no disease, num=1..4 is disease.
    if "num" not in data.columns:
        print("[ERROR] Dataset must contain the 'num' target column.")
        return False
    data["target"] = (data["num"] > 0).astype(int)
    print("[INFO] Target 'num' aligned: 1 = Disease Detected, 0 = No Disease")

    # 4. Feature Selection: Use important clinical features
    X = prepare_features(data)
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

    # 9. Evaluate Model
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred) * 100
    print("\n--- XGBOOST TEST EVALUATION ---")
    print(f"Model Accuracy: {acc:.2f}%")
    print("\nClassification Report:\n", classification_report(
        y_test, y_pred, target_names=["Low Risk (No Disease)", "High Risk (Disease)"], zero_division=0
    ))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # 10. Save Model, Scaler & Metadata
    feature_defaults = X.median().to_dict()
    model.save_model(model_save_path)
    joblib.dump(scaler, scaler_save_path)
    joblib.dump({
        "features": IMPORTANT_FEATURES,
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
    plt.figure(figsize=(9, 5))
    sorted_idx = np.argsort(model.feature_importances_)
    plt.barh([IMPORTANT_FEATURES[i] for i in sorted_idx], model.feature_importances_[sorted_idx], color="#008080")
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
