"""Job: one function that loads everything and scores a new product — used by the API, by tests, and by any batch script later. 
    Never duplicate this logic inside the API itself."""

# src/7_prediction/predict.py
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import (
    TEXT_VECTORIZER_PATH, TEXT_MODEL_PATH, STRUCTURED_MODEL_PATH, ENCODERS_PATH,
    CAT_COLS, COL_AGE
)

_vectorizer = None
_text_model = None
_structured_model = None
_encoders = None

def _load_all():
    global _vectorizer, _text_model, _structured_model, _encoders
    if _vectorizer is None:
        _vectorizer = joblib.load(TEXT_VECTORIZER_PATH)
        _text_model = joblib.load(TEXT_MODEL_PATH)
        _structured_model = joblib.load(STRUCTURED_MODEL_PATH)
        _encoders = joblib.load(ENCODERS_PATH)

def predict_risk(reviews: list[str], division: str, department: str, class_name: str, age: float = 35) -> dict:
    """
    reviews: list of review text strings for one product.
    division/department/class_name: raw category strings as they appear in the source data.
    Returns a risk score, per-review sentiment, and whether the product has enough data to trust the score.
    """

    if len(reviews) < 3:
        return {"status": "insufficient_data", "review_count": len(reviews)}

    _load_all()

    X_text = _vectorizer.transform(reviews)
    review_risk_scores = _text_model.predict_proba(X_text)[:, 1]

    def encode_safe(col_name, value, encoder):
        classes = list(encoder.classes_)
        return encoder.transform([value])[0] if value in classes else -1

    row = pd.DataFrame([{
        CAT_COLS[0]: encode_safe(CAT_COLS[0], division, _encoders[CAT_COLS[0]]),
        CAT_COLS[1]: encode_safe(CAT_COLS[1], department, _encoders[CAT_COLS[1]]),
        CAT_COLS[2]: encode_safe(CAT_COLS[2], class_name, _encoders[CAT_COLS[2]]),
        COL_AGE: age,
        "review_length": np.mean([len(r) for r in reviews]),
        "review_word_count": np.mean([len(r.split()) for r in reviews]),
        "text_risk_score": review_risk_scores.mean(),
    }])

    risk_score = float(_structured_model.predict_proba(row)[0, 1])
    risk_label = "high" if risk_score >= 0.6 else "medium" if risk_score >= 0.35 else "low"

    return {
        "status": "ok",
        "risk_score": round(risk_score, 3),
        "risk_label": risk_label,
        "review_count": len(reviews),
        "avg_text_sentiment_risk": round(float(review_risk_scores.mean()), 3),
        "most_negative_reviews": [reviews[i] for i in np.argsort(-review_risk_scores)[:3]],
    }