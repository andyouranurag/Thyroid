# thyroid_model_training.py
"""
Updated training script with feature-selection support.
- Trains LogisticRegression, SVM, RandomForest, XGBoost
- Calibrates probabilities
- Optionally uses a recommended selected feature set (recall-optimized)
- Saves models, scaler, encoders, selected_features, and model_scores
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, recall_score

# ---------------- CONFIG ----------------
CSV_PATH = "Thyroid_Diff.csv"
TARGET = "Response"

# Toggle: set to True to train on the recommended selected features (smaller set).
# If False, script uses the full feature list from the dataset (except 'Recurred' if present).
USE_SELECTED_FEATURES = True

# Recommended feature set (from analysis) — optimized for recall
RECOMMENDED_FEATURES = [
    'Risk','N','T','Age','Physical Examination',
    'Pathology','Adenopathy','Thyroid Function',
    'Gender','Hx Smoking','Stage','Focality'
]

# Directory to save trained artifacts
OUT_DIR = "saved_models"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Load data ----------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"{CSV_PATH} not found. Place your dataset in the working directory.")

df = pd.read_csv(CSV_PATH)
print(f"Loaded dataset: {CSV_PATH} — shape: {df.shape}")

# Drop unnamed columns if present
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Determine features to use
if USE_SELECTED_FEATURES:
    # validate recommended features exist in dataset
    missing = [f for f in RECOMMENDED_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Some recommended features missing from dataset: {missing}")
    FEATURES = RECOMMENDED_FEATURES.copy()
    # save recommended features file for downstream use
    joblib.dump(FEATURES, os.path.join(OUT_DIR, "selected_features.pkl"))
    print("Using RECOMMENDED features (saved to selected_features.pkl):")
else:
    # use all columns except target and Recurred if present
    FEATURES = [c for c in df.columns if c not in [TARGET, 'Recurred']]
    joblib.dump(FEATURES, os.path.join(OUT_DIR, "selected_features.pkl"))
    print("Using FULL feature set (saved to selected_features.pkl):")

print(FEATURES)

# ---------------- Prepare X, y ----------------
X = df[FEATURES].copy()
y = df[TARGET].copy()

# ---------------- Encoding categorical features ----------------
encoders = {}
categorical_cols = []

# Determine categorical columns heuristically (non-numeric)
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        categorical_cols.append(col)

# Also treat low-cardinality non-numeric-like columns as categorical (defensive)
for col in X.columns:
    if col not in categorical_cols:
        # if dtype numeric we keep numeric; else if small unique values and not Age, treat as categorical
        if X[col].dtype not in [np.float64, np.int64] and X[col].nunique() <= 20 and col.lower() != "age":
            categorical_cols.append(col)

# Ensure Age is numeric (and not treated as categorical)
if 'Age' in categorical_cols:
    categorical_cols.remove('Age')

print("Categorical columns detected:", categorical_cols)

# Fill missing values: numeric -> median, categorical -> mode
for col in X.select_dtypes(include=[np.number]).columns:
    if X[col].isnull().any():
        X[col].fillna(X[col].median(), inplace=True)
for col in categorical_cols:
    if X[col].isnull().any():
        X[col].fillna(X[col].mode()[0], inplace=True)

# Label-encode categorical columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y_enc = target_encoder.fit_transform(y.astype(str))
encoders[TARGET] = target_encoder

# ---------------- Scale numeric features ----------------
scaler = StandardScaler()
if 'Age' in X.columns:
    X['Age'] = scaler.fit_transform(X[['Age']])

# ---------------- Train/test split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)
print("Train/test shapes:", X_train.shape, X_test.shape)

# ---------------- Define models ----------------
# Use class_weight='balanced' where appropriate to help cost-sensitivity.
models_base = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced', multi_class='multinomial', random_state=42),
    "SVM": SVC(probability=True, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=200, random_state=42)
}

trained_calibrated = {}
model_performance = {}

# ---------------- Train, calibrate & evaluate ----------------
print("\nTraining and calibrating models (this can take some minutes)...")
for name, base_model in models_base.items():
    print(f"\n--- {name} ---")
    # fit base model
    base_model.fit(X_train, y_train)
    # calibrate
    cal = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    cal.fit(X_train, y_train)
    trained_calibrated[name] = cal

    # evaluate
    y_pred = cal.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average='macro')
    model_performance[name] = {"accuracy": float(acc), "recall_macro": float(rec)}

    print("Accuracy: {:.4f}, Recall (macro): {:.4f}".format(acc, rec))
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# ---------------- Save everything ----------------
print("\nSaving models and artifacts to:", OUT_DIR)
for name, model in trained_calibrated.items():
    fname = f"{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, os.path.join(OUT_DIR, fname))

joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
joblib.dump(encoders, os.path.join(OUT_DIR, "encoders.pkl"))
joblib.dump(model_performance, os.path.join(OUT_DIR, "model_scores.pkl"))

# Save selected features (already saved prior), keep copy in OUT_DIR
joblib.dump(FEATURES, os.path.join(OUT_DIR, "selected_features.pkl"))

print("Saved files:")
print("- models:", [f for f in os.listdir(OUT_DIR) if f.endswith(".pkl")])
print("\nModel performance summary (accuracy & recall_macro):")
for m, stats in model_performance.items():
    print(f"  {m}: accuracy={stats['accuracy']:.3f}, recall_macro={stats['recall_macro']:.3f}")

best_by_recall = max(model_performance.items(), key=lambda kv: kv[1]['recall_macro'])
print(f"\nBest model by recall_macro: {best_by_recall[0]} (recall={best_by_recall[1]['recall_macro']:.3f})")

print("\n✅ Training pipeline complete.")
