import pandas as pd
import numpy as np
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from imblearn.combine import SMOTETomek

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

label_encoders = {}
scaler = StandardScaler()
feature_order = []

csv_data = "data/heart_disease.csv"
output_dir = "train"

os.makedirs(output_dir, exist_ok=True)

try:

    # ==========================
    # LOAD DATA
    # ==========================

    data = pd.read_csv(csv_data)

    # Fill missing numeric
    num_cols = data.select_dtypes(include=np.number).columns
    for col in num_cols:
        data[col].fillna(data[col].median(), inplace=True)

    # Fill missing categorical
    cat_cols = data.select_dtypes(include="object").columns
    for col in cat_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Target conversion
    data["Heart Disease Status"] = (data["Heart Disease Status"] == "Yes").astype(int)

    # Stress level mapping - FIX: handle unmapped values
    stress_map = {"Low": 1, "Medium": 2, "High": 3}
    data["Stress Level"] = data["Stress Level"].map(stress_map)
    data["Stress Level"].fillna(2, inplace=True)  # Fill any unmapped with Medium

    # Encode categorical columns
    for col in data.columns:
        if data[col].dtype == 'object' or isinstance(data[col].iloc[0], str):
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le

    X = data.drop("Heart Disease Status", axis=1)
    y = data["Heart Disease Status"]
    
    # ADD: Check for and handle any remaining NaN values
    print(f"\nNaN count per column:\n{X.isna().sum()}")
    
    # Fill any remaining NaN with median (numeric) or 0
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)
    
    print(f"\nNaN count after cleanup: {X.isna().sum().sum()}")

    feature_order = X.columns.tolist()

    print(f"\nTraining with {len(feature_order)} features")

    # ==========================
    # TRAIN TEST SPLIT
    # ==========================

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.15,
        random_state=42,
        stratify=y_train_full
    )

    # ==========================
    # SCALING
    # ==========================

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("Original distribution:", np.bincount(y_train))
    
    # ADD: Debug check
    print(f"NaN in X_train_scaled: {np.isnan(X_train_scaled).sum()}")

    # ==========================
    # BALANCING DATA
    # ==========================

    smote = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    print("Balanced distribution:", np.bincount(y_train_res))

    # ==========================
    # CLASS WEIGHTS
    # ==========================

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = dict(enumerate(class_weights))

    print("Class Weights:", class_weights)

    # ==========================
    # MODEL
    # ==========================

    model = Sequential([
        tf.keras.Input(shape=(X_train_scaled.shape[1],)),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation="relu"),
        Dropout(0.1),

        Dense(1, activation="sigmoid")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0007)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    es = EarlyStopping(
        monitor="val_auc",
        patience=20,
        mode="max",
        restore_best_weights=True
    )

    print("\nTraining Model...\n")

    history = model.fit(
        X_train_res,
        y_train_res,
        validation_data=(X_val_scaled, y_val),
        epochs=150,
        batch_size=32,
        callbacks=[es],
        class_weight=class_weights,
        verbose=1
    )

    # ==========================
    # EVALUATION
    # ==========================

    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

    print("\n==========================")
    print("MODEL EVALUATION")
    print("==========================")

    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix\n")
    print(confusion_matrix(y_test, y_pred))

    # ==========================
    # SAVE MODEL
    # ==========================

    model_path = os.path.join(output_dir, "tf_heart_model.keras")
    model.save(model_path)

    print("\nModel saved to:", model_path)

    # ==========================
    # PLOTS
    # ==========================

    plt.figure()
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy.png"))
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()

    print("Training graphs saved.")

    # ==========================
    # LIVE PREDICTION
    # ==========================

    print("\nEnter details for prediction\n")

    new_input = {}

    for col in feature_order:

        if col in label_encoders:

            le = label_encoders[col]
            print(f"{col} options: {list(le.classes_)}")
            val = input(f"Enter {col}: ")

            try:
                new_input[col] = le.transform([val])[0]
            except:
                new_input[col] = 0

        else:
            new_input[col] = float(input(f"Enter {col}: "))

    new_df = pd.DataFrame([new_input])
    new_df = new_df[feature_order]

    new_scaled = scaler.transform(new_df)

    prob = model.predict(new_scaled)[0][0]

    risk = prob
    no_risk = 1 - prob
    threshold = 0.5

    print("\n====================== RESULT ======================")

    if risk > threshold:
        print("⚠ High Risk of Heart Disease")
    else:
        print("✅ Low Risk")

    print(f"Risk Probability: {risk*100:.2f}%")
    print(f"No Disease Confidence: {no_risk*100:.2f}%")

    print("====================================================")

    # ==========================
    # PREDICTION CONFIDENCE GRAPH
    # ==========================

    plt.figure(figsize=(8,5))

    labels = [
        "Risk of Disease (Class 1)",
        "Confidence in No Disease (Class 0)"
    ]

    values = [risk, no_risk]

    colors = ["#d62728", "#98df8a"]

    bars = plt.bar(labels, values, color=colors)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.02,
            f"{height*100:.2f}%",
            ha="center"
        )

    plt.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        label=f"Decision Threshold ({threshold*100:.2f}%)"
    )

    plt.title("Prediction Confidence for Your Input")
    plt.ylabel("Probability")
    plt.ylim(0,1.1)

    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.legend()

    pred_graph = os.path.join(output_dir, "prediction_confidence.png")

    plt.savefig(pred_graph)
    plt.close()

    print(f"\nPrediction graph saved to: {pred_graph}")

except Exception as e:
    print(f"Error: {e}")