import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
data = pd.read_csv("heart_disease.csv")

# Handle missing values (categorical with mode, numeric with mean)
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].mean(), inplace=True)

# Convert Stress Level to numeric
data['Stress Level'] = data['Stress Level'].map({'Low': 1, 'Medium': 2, 'High': 3}).fillna(2)

# Encode categorical columns
label_encoders = {}
for col in data.columns:
    if data[col].dtype == 'object' and col not in ['Stress Level']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

# -----------------------------
# 2️⃣ Select 12 important features
# -----------------------------
selected_features = [
    'Age', 'Gender', 'Blood Pressure', 'Smoking', 'Stress Level', 'BMI',
    'Cholesterol Level', 'Family Heart Disease', 'Diabetes',
    'Alcohol Consumption', 'Sugar Consumption', 'CRP Level'
]
X = data[selected_features]
y = data['Heart Disease Status']

# -----------------------------
# 3️⃣ Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fill missing numeric values again (safety)
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# -----------------------------
# 4️⃣ Scale data
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 5️⃣ Apply SMOTE for balancing
# -----------------------------
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# -----------------------------
# 6️⃣ Build and train XGBoost model
# -----------------------------
scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=3,
    gamma=0.2,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.2,
    reg_lambda=1.0,
    random_state=42,
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# -----------------------------
# 7️⃣ Evaluate model
# -----------------------------
y_pred = model.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model & scaler
joblib.dump(model, "xgboost_heart_12param_balanced.pkl")
joblib.dump(scaler, "heart_scaler_12param_balanced.pkl")

# -----------------------------
# 8️⃣ Prediction input (12 parameters)
# -----------------------------
print("\n--- Enter details for prediction ---")
age = float(input("Enter Age (in years): "))
gender = float(input("Enter Gender (0 = Female, 1 = Male): "))
bp = float(input("Enter Blood Pressure (in mmHg): "))
smoking = float(input("Enter Smoking (0 = Non-smoker, 1 = Smoker): "))
print("\nStress Level Options:\n1 = Low  |  2 = Medium  |  3 = High")
stress = float(input("Enter Stress Level (1/2/3): "))
bmi = float(input("Enter BMI (Body Mass Index): "))
chol = float(input("Enter Cholesterol Level: "))
family = float(input("Family Heart Disease (0 = No, 1 = Yes): "))
diabetes = float(input("Diabetes (0 = No, 1 = Yes): "))
alcohol = float(input("Alcohol Consumption (0 = Low, 1 = Medium, 2 = High): "))
sugar = float(input("Sugar Consumption (0 = Low, 1 = Medium, 2 = High): "))
crp = float(input("Enter CRP Level: "))

new_data = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'Blood Pressure': bp,
    'Smoking': smoking,
    'Stress Level': stress,
    'BMI': bmi,
    'Cholesterol Level': chol,
    'Family Heart Disease': family,
    'Diabetes': diabetes,
    'Alcohol Consumption': alcohol,
    'Sugar Consumption': sugar,
    'CRP Level': crp
}])

new_data_scaled = scaler.transform(new_data)
probability = model.predict_proba(new_data_scaled)[0][1] * 100
prediction = 1 if probability > 45 else 0

if prediction == 1:
    print(f"\n⚠️ High risk: Likely heart disease detected. (Confidence: {probability:.2f}%)")
else:
    print(f"\n✅ Low risk: No heart disease detected. (Confidence: {100 - probability:.2f}%)")


y_prob = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(8,5))
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.title("Precision-Recall vs Threshold")
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 9️⃣ Feature importance visualization
# -----------------------------
plt.figure(figsize=(10, 6))
plt.barh(selected_features, model.feature_importances_, color='teal')
plt.title("Feature Importance - Heart Disease Prediction (XGBoost)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
