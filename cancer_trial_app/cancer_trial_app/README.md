# TrialScope — Cancer Clinical Trial Outcome Predictor

A production-ready Flask web application that predicts the success
probability of cancer clinical trials using a trained Gradient Boosting
Machine (GBM) classifier wrapped in a **sklearn Pipeline**.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (creates model/model_pipeline.pkl + model/feature_columns.json)
python generate_model.py

# 3. Start the Flask dev server
python app.py
# → http://localhost:5000
```

For production:
```bash
gunicorn app:app --workers 2 --bind 0.0.0.0:8000
```

---

## Project Structure

```
cancer_trial_app/
├── app.py                  # Flask app — routes, preprocessing, prediction
├── generate_model.py       # Training script — run once to create artifacts
├── requirements.txt
├── Procfile                # For Heroku / Railway / Render
├── model/
│   ├── model_pipeline.pkl  # Full sklearn Pipeline (preprocessor + classifier)
│   └── feature_columns.json# Column metadata (order, categories)
└── templates/
    ├── index.html          # Input form
    └── result.html         # Prediction result page
```

---

## Architecture

### Training / Serving Parity (CRITICAL)

All preprocessing is baked into the **sklearn Pipeline** saved as
`model_pipeline.pkl`. The Flask app calls `pipeline.predict_proba(df)`
directly — no manual encoding, no manual scaling. This guarantees that
training and deployment transformations are always identical.

```
User Input → DataFrame (same column order as training)
           → Pipeline.predict_proba()
               ├── ColumnTransformer
               │   ├── OneHotEncoder  (categorical cols)
               │   └── StandardScaler (numerical cols)
               └── GradientBoostingClassifier
           → P(success) → Risk label
```

### Prediction Logic

| Probability | Label                  | Risk Level  |
|-------------|------------------------|-------------|
| > 0.70      | LIKELY TO SUCCEED      | Low Risk    |
| 0.50 – 0.70 | LIKELY TO SUCCEED      | Medium Risk |
| 0.40 – 0.50 | HIGH RISK OF FAILURE   | Medium Risk |
| < 0.40      | HIGH RISK OF FAILURE   | High Risk   |

### API Endpoint

A JSON endpoint is available for programmatic access:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cancer_type": "breast",
    "drug_type": "targeted_therapy",
    "phase": "III",
    "enrollment_size": 500,
    "trial_duration_months": 48,
    "sponsor_type": "industry"
  }'
```

Response:
```json
{
  "status": "ok",
  "prediction": {
    "label": "LIKELY TO SUCCEED",
    "risk_level": "Low Risk",
    "badge": "low",
    "prob_pct": 72.4,
    "prob_raw": 0.7240
  }
}
```

---

## Replacing with Your Own Model

1. Train your model in Colab using the same `FEATURE_COLUMNS` order.
2. Wrap preprocessing + classifier in a `sklearn.pipeline.Pipeline`.
3. Save with `joblib.dump(pipeline, "model/model_pipeline.pkl")`.
4. Save `feature_columns.json` with the metadata dict (see `generate_model.py`).
5. The Flask app will load and use it without any other changes.

---

## Key Design Decisions

- **Single Pipeline artifact** — eliminates training/serving skew completely.
- **Stateless routes** — each prediction is independent; safe for horizontal scaling.
- **JSON API** — `/api/predict` enables integration with dashboards or notebooks.
- **`handle_unknown="ignore"`** on the encoder — unseen categories default to
  the zero vector instead of crashing.
- **`remainder="drop"`** — extra columns in input are silently ignored.
