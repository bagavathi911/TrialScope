"""
generate_model.py
-----------------
Trains a sample cancer clinical trial prediction model and saves:
  - model_pipeline.pkl  (full sklearn Pipeline: preprocessor + classifier)
  - feature_columns.json (ordered list of raw input columns)
  - label_encoder.pkl   (for any label encoding if needed)

Run once to create the model artifacts before starting the Flask app.
"""

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Feature definitions ────────────────────────────────────────────────────────
CANCER_TYPES     = ["lung", "breast", "colorectal", "prostate", "leukemia",
                    "melanoma", "pancreatic", "ovarian", "bladder", "renal"]
DRUG_TYPES       = ["chemotherapy", "immunotherapy", "targeted_therapy",
                    "hormone_therapy", "radiation", "combination"]
PHASES           = ["I", "II", "III", "IV"]
SPONSOR_TYPES    = ["industry", "academic", "government"]

CATEGORICAL_COLS = ["cancer_type", "drug_type", "phase", "sponsor_type"]
NUMERICAL_COLS   = ["enrollment_size", "trial_duration_months"]
ALL_FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS   # ORDER MATTERS

# ── Synthetic dataset ──────────────────────────────────────────────────────────
N = 3000

cancer_type          = np.random.choice(CANCER_TYPES, N)
drug_type            = np.random.choice(DRUG_TYPES, N)
phase                = np.random.choice(PHASES, N, p=[0.15, 0.35, 0.40, 0.10])
enrollment_size      = np.random.randint(20, 2000, N)
trial_duration_months = np.random.randint(6, 120, N)
sponsor_type         = np.random.choice(SPONSOR_TYPES, N, p=[0.55, 0.30, 0.15])

# Rule-based success probability (domain knowledge baked in)
success_prob = np.full(N, 0.40)

# Phase effect
success_prob += np.where(phase == "I", -0.10,
               np.where(phase == "II", 0.00,
               np.where(phase == "III", 0.10, 0.05)))

# Drug type effect
success_prob += np.where(drug_type == "targeted_therapy", 0.12,
               np.where(drug_type == "immunotherapy", 0.08,
               np.where(drug_type == "combination", 0.06,
               np.where(drug_type == "hormone_therapy", 0.04,
               np.where(drug_type == "chemotherapy", -0.02, -0.05)))))

# Cancer type effect
success_prob += np.where(cancer_type == "breast", 0.10,
               np.where(cancer_type == "prostate", 0.08,
               np.where(cancer_type == "leukemia", 0.05,
               np.where(cancer_type == "pancreatic", -0.12,
               np.where(cancer_type == "lung", -0.05, 0.00)))))

# Enrollment / duration effects
success_prob += np.clip((enrollment_size - 200) / 5000, -0.05, 0.05)
success_prob += np.clip((trial_duration_months - 24) / 500, -0.03, 0.03)

# Sponsor effect
success_prob += np.where(sponsor_type == "industry", 0.05,
               np.where(sponsor_type == "academic", 0.00, -0.03))

success_prob = np.clip(success_prob + np.random.normal(0, 0.10, N), 0.05, 0.95)
outcome = (success_prob >= 0.50).astype(int)

df = pd.DataFrame({
    "cancer_type":           cancer_type,
    "drug_type":             drug_type,
    "phase":                 phase,
    "enrollment_size":       enrollment_size.astype(float),
    "trial_duration_months": trial_duration_months.astype(float),
    "sponsor_type":          sponsor_type,
    "outcome":               outcome,
})

X = df[ALL_FEATURE_COLS]
y = df["outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ── Preprocessing pipeline ─────────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ("num", StandardScaler(), NUMERICAL_COLS),
], remainder="drop")

full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )),
])

full_pipeline.fit(X_train, y_train)

# ── Evaluation ─────────────────────────────────────────────────────────────────
y_pred = full_pipeline.predict(X_test)
print("\n=== Model Evaluation ===")
print(classification_report(y_test, y_pred, target_names=["Failure", "Success"]))

# ── Save artifacts ─────────────────────────────────────────────────────────────
joblib.dump(full_pipeline, "model/model_pipeline.pkl")

with open("model/feature_columns.json", "w") as f:
    json.dump({"feature_columns": ALL_FEATURE_COLS,
               "categorical_cols": CATEGORICAL_COLS,
               "numerical_cols": NUMERICAL_COLS,
               "cancer_types": CANCER_TYPES,
               "drug_types": DRUG_TYPES,
               "phases": PHASES,
               "sponsor_types": SPONSOR_TYPES}, f, indent=2)

print("\n✅  Artifacts saved:")
print("   model/model_pipeline.pkl")
print("   model/feature_columns.json")
