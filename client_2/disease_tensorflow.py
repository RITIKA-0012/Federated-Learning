import pandas as pd
import numpy as np
import os
import sys
import warnings
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ==========================
# GLOBAL CONFIGURATION
# ==========================
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_data = os.path.join(base_dir, "data", "heart_disease_1.csv")
output_dir = os.path.join(base_dir, "train")
TARGET_COL = "Status"  # In heart_disease_1.csv the target column is 'Status'
FORCE_RETRAIN = False

os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, "tf_global_disease_model.keras")
prep_path = os.path.join(output_dir, "preprocessor.pkl")


def save_dataset_csv(df, filename):
    save_path = os.path.join(output_dir, filename)
    df.to_csv(save_path, index=False)
    return save_path


def train_and_save_model():
    print("\n====================================================")
    print("       HEART DISEASE MODEL TRAINING PIPELINE        ")
    print("====================================================\n")
    
    # 1. Load Data
    try:
        data = pd.read_csv(csv_data)
        print(f"[INFO] Successfully loaded dataset: '{csv_data}' ({len(data)} rows, {len(data.columns)} columns)")
    except FileNotFoundError:
        print(f"[ERROR] Dataset not found at: {csv_data}")
        return False
        
    global TARGET_COL
    if TARGET_COL not in data.columns:
        TARGET_COL = data.columns[-1]
        print(f"[WARNING] Target column not found. Defaulting to last column: '{TARGET_COL}'")

    # 2. Clean Data (Handle Missing Values across all columns)
    num_cols = data.select_dtypes(include=np.number).columns
    for col in num_cols:
        if data[col].isna().any():
            data[col].fillna(data[col].median(), inplace=True)

    cat_cols = data.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        if data[col].isna().any():
            data[col].fillna(data[col].mode()[0], inplace=True)

    # 3. Align & Encode Target
    # In heart_disease_1.csv, Status 0 represents Disease / High Risk, while Status 1 represents No Disease / Low Risk.
    # We invert Status so that:
    #   Class 1 = High Risk / Disease Detected
    #   Class 0 = Low Risk / No Disease Detected (Healthy)
    if TARGET_COL.lower() == "status":
        unique_vals = set(data[TARGET_COL].unique())
        if unique_vals.issubset({0, 1}):
            data[TARGET_COL] = (data[TARGET_COL] == 0).astype(int)
            print("[INFO] Target 'Status' aligned: 1 = High Risk (Disease Detected), 0 = Low Risk (No Disease)")
        else:
            target_le = LabelEncoder()
            data[TARGET_COL] = target_le.fit_transform(data[TARGET_COL].astype(str))
    elif data[TARGET_COL].dtype == 'object' or isinstance(data[TARGET_COL].iloc[0], str):
        if set(data[TARGET_COL].unique()).issubset({"Yes", "No"}):
            data[TARGET_COL] = (data[TARGET_COL] == "Yes").astype(int)
        elif set(data[TARGET_COL].unique()).issubset({"Presence", "Absence"}):
            data[TARGET_COL] = (data[TARGET_COL] == "Presence").astype(int)
        else:
            target_le = LabelEncoder()
            data[TARGET_COL] = target_le.fit_transform(data[TARGET_COL].astype(str))

    # 4. Encode Categorical Features
    label_encoders = {}
    features = data.drop(TARGET_COL, axis=1)
    
    for col in features.columns:
        if features[col].dtype == 'object' or isinstance(features[col].iloc[0], str):
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col].astype(str))
            label_encoders[col] = le

    X_full = features
    y = data[TARGET_COL]

    # ==========================
    # 5. AUTOMATED FEATURE SELECTION
    # ==========================
    # Read and evaluate all columns from the dataset to find the most important features
    print(f"\n[INFO] Evaluating feature importance across all {len(X_full.columns)} predictor columns...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_full, y)
    
    importances = rf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    
    # Select top features (features contributing to ~90% predictive power, top 8 features)
    num_features_to_keep = 8  # chest pain, Max HR, Major Vessels, ST Depression, Thalassemia, age, Cholesterol, Resting BP
    top_indices = sorted_indices[:num_features_to_keep]
    important_features = [X_full.columns[i] for i in top_indices]
    
    print("\n--- FEATURE IMPORTANCE RANKING ---")
    for rank, idx in enumerate(sorted_indices, 1):
        feat = X_full.columns[idx]
        score = importances[idx]
        selected = "[SELECTED] (Used for prediction)" if feat in important_features else "[EXCLUDED] (Low predictive power)"
        print(f"  {rank:2d}. {feat:<25} (Importance: {score:.4f}) -> {selected}")

    # Train model using only the important features selected from the full dataset
    X = X_full[important_features]
    feature_order = important_features.copy()

    # 6. Train/Test Split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n[INFO] Original training class distribution: {y_train_full.value_counts().sort_index().to_dict()}")
    train_df = pd.concat([
        X_train_full.reset_index(drop=True),
        y_train_full.reset_index(drop=True).rename(TARGET_COL)
    ], axis=1)

    min_class_count = train_df[TARGET_COL].value_counts().min()
    balanced_frames = []
    for class_label in sorted(train_df[TARGET_COL].unique()):
        class_df = train_df[train_df[TARGET_COL] == class_label]
        balanced_frames.append(class_df.sample(n=min_class_count, random_state=42))

    balanced_train_df = pd.concat(balanced_frames, axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    X_train_bal = balanced_train_df.drop(TARGET_COL, axis=1)
    y_train_bal = balanced_train_df[TARGET_COL]

    print(f"[INFO] Balanced training class distribution: {y_train_bal.value_counts().sort_index().to_dict()}")
    save_dataset_csv(balanced_train_df, "training_data.csv")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_bal, y_train_bal, test_size=0.15, random_state=42, stratify=y_train_bal
    )

    # 7. Scale the Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 8. Class Weights
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))

    # 9. Model Architecture
    model = Sequential([
        tf.keras.Input(shape=(X_train_scaled.shape[1],)),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation="relu"),
        Dropout(0.2),
        
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name='auc')]
    )

    es = EarlyStopping(monitor="val_auc", patience=20, mode="max", restore_best_weights=True)

    print("\nTraining Deep Neural Network Model...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=150, batch_size=32, callbacks=[es],
        class_weight=class_weights_dict, verbose=1
    )

    # 10. Save Evaluation Graphs
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_graph.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_graph.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auc_graph.png'))
    plt.close()

    # 11. Evaluate on Test Data
    y_val_prob = model.predict(X_val_scaled, verbose=0).reshape(-1)
    best_threshold = 0.5
    best_score = -1
    for thr in np.linspace(0.3, 0.7, 41):
        preds = (y_val_prob >= thr).astype(int)
        score = accuracy_score(y_val, preds)
        if score > best_score:
            best_score = score
            best_threshold = thr

    y_prob = model.predict(X_test_scaled, verbose=0).reshape(-1)
    y_pred = (y_prob >= best_threshold).astype(int)
    print(f"\n[INFO] Optimal Decision Threshold: {best_threshold:.2f}")
    print("\n--- MODEL EVALUATION ON TEST SET ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Low Risk (No Disease)", "High Risk (Disease)"]))

    prediction_output = pd.DataFrame({
        "actual": y_test.reset_index(drop=True).astype(int),
        "predicted": y_pred.astype(int),
        "probability": y_prob,
    })
    save_dataset_csv(prediction_output, "prediction_output.csv")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.xticks([0, 1], ['Low Risk (0)', 'High Risk (1)'])
    plt.yticks([0, 1], ['Low Risk (0)', 'High Risk (1)'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # 12. Save Model & Preprocessors
    model.save(model_path)
    
    # Store feature medians as safe fallbacks
    feature_defaults = X[feature_order].median().to_dict()

    preprocessor_data = {
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_order': feature_order,
        'feature_defaults': feature_defaults,
        'decision_threshold': float(best_threshold)
    }
    with open(prep_path, 'wb') as f:
        pickle.dump(preprocessor_data, f)
        
    print(f"\n[SUCCESS] Model, graphs, and preprocessors saved to '{output_dir}'.")
    return True


def predict_sample(input_values):
    """Programmatic prediction helper function for given feature values dict."""
    model = load_model(model_path)
    with open(prep_path, 'rb') as f:
        preprocessor_data = pickle.load(f)

    scaler = preprocessor_data['scaler']
    feature_order = preprocessor_data['feature_order']
    threshold = preprocessor_data.get('decision_threshold', 0.5)

    input_df = pd.DataFrame([input_values])[feature_order]
    scaled = scaler.transform(input_df)
    prob = float(model.predict(scaled, verbose=0)[0][0])
    is_high_risk = prob >= threshold
    return {
        "probability": prob,
        "risk_percentage": prob * 100,
        "safe_percentage": (1 - prob) * 100,
        "is_high_risk": is_high_risk,
        "predicted_class": 1 if is_high_risk else 0,
        "threshold": threshold
    }


def live_prediction():
    print("\n====================================================")
    print("           LIVE DISEASE PREDICTION SYSTEM            ")
    print("====================================================\n")

    # Load Model and Preprocessor
    model = load_model(model_path)
    with open(prep_path, 'rb') as f:
        preprocessor_data = pickle.load(f)

    scaler = preprocessor_data['scaler']
    label_encoders = preprocessor_data.get('label_encoders', {})
    feature_order = preprocessor_data['feature_order']
    feature_defaults = preprocessor_data.get('feature_defaults', {})
    threshold = float(preprocessor_data.get('decision_threshold', 0.5))

    print(f"[INFO] The model only requires the {len(feature_order)} most important features to make accurate predictions.")
    print("Please enter the patient's values below (press Enter to accept default values):\n")

    # Clear descriptions/hints for important features
    feature_hints = {
        "chest pain": "0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic",
        "Maximum Heart Rate": "Max heart rate achieved during exercise (e.g. 71 - 202 bpm)",
        "Major Blood Vessels": "Number of major vessels (0-3) colored by fluoroscopy",
        "ST Depression": "ST depression induced by exercise relative to rest (e.g. 0.0 - 6.2)",
        "Thalassemia Result": "1: normal, 2: fixed defect, 3: reversible defect",
        "age": "Age in years (e.g. 29 - 77)",
        "Cholesterol Level": "Serum cholesterol in mg/dl (e.g. 126 - 564)",
        "Resting Blood Pressure": "Resting blood pressure in mm Hg (e.g. 94 - 200)",
        "Exercise-Induced Angina": "1: Yes, 0: No",
        "ST Segment Slope": "0: upsloping, 1: flat, 2: downsloping",
        "gender": "1: Male, 0: Female"
    }

    new_input = {}
    for col in feature_order:
        hint = feature_hints.get(col, "")
        hint_str = f" ({hint})" if hint else ""
        default_val = feature_defaults.get(col, 0.0)

        if col in label_encoders:
            le = label_encoders[col]
            print(f"-> {col}{hint_str}")
            print(f"   Available options: {list(le.classes_)}")
            val = input(f"   Enter {col} [default {default_val}]: ").strip()
            if not val:
                new_input[col] = default_val
            else:
                try:
                    new_input[col] = le.transform([val])[0]
                except Exception:
                    print(f"   Unrecognized value for '{col}', using default {default_val}.")
                    new_input[col] = default_val
        else:
            val = input(f"-> Enter {col}{hint_str} [default {default_val}]: ").strip()
            if not val:
                new_input[col] = float(default_val)
            else:
                try:
                    new_input[col] = float(val)
                except ValueError:
                    print(f"   Invalid input, using default: {default_val}")
                    new_input[col] = float(default_val)

    # Format and scale input
    new_df = pd.DataFrame([new_input])[feature_order]
    new_scaled = scaler.transform(new_df)
    
    prob = float(model.predict(new_scaled, verbose=0)[0][0])

    risk_prob = prob * 100
    safe_prob = (1 - prob) * 100

    print("\n====================================================")
    print("                      RESULT                        ")
    print("====================================================")
    
    if prob >= threshold:
        print("[!] HIGH RISK / DISEASE DETECTED")
        print("Predicted Class: 1 (High Risk of Heart Disease)")
    else:
        print("[+] LOW RISK / NO DISEASE DETECTED")
        print("Predicted Class: 0 (Low Risk / Healthy)")

    print(f"\nHeart Disease Risk Probability: {risk_prob:.2f}%")
    print(f"Healthy / Low Risk Confidence:  {safe_prob:.2f}%")
    print(f"Decision Threshold:             {threshold*100:.2f}%")
    print("====================================================")

    # ==========================
    # SAVE LIVE PREDICTION GRAPH
    # ==========================
    plt.figure(figsize=(8, 5))
    labels = ["Risk of Disease (Class 1)", "Low Risk / Healthy (Class 0)"]
    values = [prob, 1 - prob]
    colors = ["#d62728" if prob >= threshold else "#ff9896", 
              "#2ca02c" if prob < threshold else "#98df8a"]

    bars = plt.bar(labels, values, color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height*100:.2f}%", ha="center", fontweight='bold')

    plt.axhline(y=threshold, color="red", linestyle="--", label=f"Decision Threshold ({threshold*100:.2f}%)")
    plt.title("Heart Disease Risk Prediction Confidence")
    plt.ylabel("Probability")
    plt.ylim(0, 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()

    pred_graph = os.path.join(output_dir, "prediction_confidence.png")
    plt.savefig(pred_graph)
    plt.close()
    print(f"\n[INFO] Prediction confidence graph saved to: {pred_graph}")


# ==========================
# MAIN EXECUTION PIPELINE
# ==========================
if __name__ == "__main__":
    is_interactive = sys.stdin.isatty()
    # Check if --train-only argument is passed
    train_only = "--train-only" in sys.argv

    if FORCE_RETRAIN or not (os.path.exists(model_path) and os.path.exists(prep_path)):
        success = train_and_save_model()
        if success and not train_only and is_interactive:
            live_prediction()
    else:
        print(f"[INFO] Existing trained model found in '{output_dir}'. Skipping training...")
        if not train_only and is_interactive:
            live_prediction()