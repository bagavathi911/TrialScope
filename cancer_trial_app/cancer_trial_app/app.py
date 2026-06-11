

import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load artifacts once at startup ────────────────────────────────────────────
MODEL_PATH    = os.path.join(BASE_DIR, "model", "model_pipeline.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_columns.json")

pipeline = joblib.load(MODEL_PATH)

with open(FEATURES_PATH) as f:
    meta = json.load(f)

FEATURE_COLUMNS  = meta["feature_columns"]   # ordered list – must match training
CATEGORICAL_COLS = meta["categorical_cols"]
NUMERICAL_COLS   = meta["numerical_cols"]
CANCER_TYPES     = meta["cancer_types"]
DRUG_TYPES       = meta["drug_types"]
PHASES           = meta["phases"]
SPONSOR_TYPES    = meta["sponsor_types"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_input_df(form) -> pd.DataFrame:
    """
    Convert raw form data into a single-row DataFrame whose column
    names and dtypes exactly match what the Pipeline was trained on.
    """
    row = {
        "cancer_type":           str(form["cancer_type"]).strip().lower(),
        "drug_type":             str(form["drug_type"]).strip().lower(),
        "phase":                 str(form["phase"]).strip().upper(),
        "sponsor_type":          str(form["sponsor_type"]).strip().lower(),
        "enrollment_size":       float(form["enrollment_size"]),
        "trial_duration_months": float(form["trial_duration_months"]),
    }
    # Guarantee column order matches training
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def classify_risk(prob: float) -> dict:
    """
    Return a dict with label, risk_level, colour code, and emoji
    based on the success probability threshold rules.
    """
    if prob > 0.5:
        label = "LIKELY TO SUCCEED"
        if prob >= 0.70:
            risk_level, colour, badge = "Low Risk",    "#22c55e", "low"
        else:
            risk_level, colour, badge = "Medium Risk", "#f59e0b", "medium"
    else:
        label = "HIGH RISK OF FAILURE"
        if prob >= 0.40:
            risk_level, colour, badge = "Medium Risk", "#f59e0b", "medium"
        else:
            risk_level, colour, badge = "High Risk",   "#ef4444", "high"

    return {
        "label":      label,
        "risk_level": risk_level,
        "colour":     colour,
        "badge":      badge,
        "prob_pct":   round(prob * 100, 1),
        "prob_raw":   round(prob, 4),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        cancer_types  = CANCER_TYPES,
        drug_types    = DRUG_TYPES,
        phases        = PHASES,
        sponsor_types = SPONSOR_TYPES,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        df_input  = build_input_df(request.form)
        prob      = float(pipeline.predict_proba(df_input)[0, 1])  # P(success)
        result    = classify_risk(prob)

        # Pass user inputs back to the template for display
        user_input = {
            "cancer_type":           request.form["cancer_type"].title(),
            "drug_type":             request.form["drug_type"].replace("_", " ").title(),
            "phase":                 f"Phase {request.form['phase']}",
            "enrollment_size":       int(float(request.form["enrollment_size"])),
            "trial_duration_months": int(float(request.form["trial_duration_months"])),
            "sponsor_type":          request.form["sponsor_type"].title(),
        }

        return render_template("result.html", result=result, user_input=user_input)

    except (KeyError, ValueError) as exc:
        return render_template("index.html",
                               error=f"Input error: {exc}",
                               cancer_types  = CANCER_TYPES,
                               drug_types    = DRUG_TYPES,
                               phases        = PHASES,
                               sponsor_types = SPONSOR_TYPES), 400


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint for programmatic access."""
    try:
        data      = request.get_json(force=True)
        df_input  = build_input_df(data)
        prob      = float(pipeline.predict_proba(df_input)[0, 1])
        result    = classify_risk(prob)
        return jsonify({"status": "ok", "prediction": result})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": True})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
