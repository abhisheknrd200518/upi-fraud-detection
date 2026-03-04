import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ==============================
# LOAD CORRECT DATASET
# ==============================

data = pd.read_csv("upi_transactions_2024.csv")

# Clean column names
data.columns = data.columns.str.strip().str.lower()

print("Columns in Dataset:")
print(data.columns)

# ==============================
# CHECK TARGET COLUMN
# ==============================

# Change this if your target column name is different
if "fraud_flag" in data.columns:
    target_column = "fraud_flag"
elif "class" in data.columns:
    target_column = "class"
else:
    raise ValueError("❌ No target column found (fraud_flag or class)")

# ==============================
# HANDLE TIMESTAMP (IF EXISTS)
# ==============================

if "timestamp" in data.columns:
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    data["hour"] = data["timestamp"].dt.hour
    data["month"] = data["timestamp"].dt.month
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["hour_of_day"] = data["hour"]
    data["is_weekend"] = data["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    data = data.drop(columns=["timestamp"])

# ==============================
# DROP TRANSACTION ID IF EXISTS
# ==============================

if "transaction id" in data.columns:
    data = data.drop(columns=["transaction id"])

# ==============================
# ENCODE CATEGORICAL FEATURES
# ==============================

label_encoders = {}

for column in data.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

# ==============================
# SPLIT FEATURES & TARGET
# ==============================

X = data.drop(target_column, axis=1)
y = data[target_column]

# Save feature names (important for Streamlit)
joblib.dump(X.columns.tolist(), "feature_names.pkl")

# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# TRAIN XGBOOST MODEL
# ==============================

model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# EVALUATE MODEL
# ==============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Model Training Completed")
print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# SAVE MODEL & ENCODERS
# ==============================

joblib.dump(model, "xgboost_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("\n✅ Model and encoders saved successfully!")