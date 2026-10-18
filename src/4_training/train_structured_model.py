"""Job: combine structured features + the text model's output into one final risk classifier."""

import sys
from pathlib import Path
import pandas as pd
import joblib
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import (
    TRAIN_PATH, CANDIDATE_STRUCTURED_MODEL_PATH,
    CANDIDATE_TEXT_VECTORIZER_PATH, CANDIDATE_TEXT_MODEL_PATH, RANDOM_STATE, FEATURE_COLS
)
from src.training_utils import add_text_risk_score

def train_structured_model():
    train_df = pd.read_parquet(TRAIN_PATH)
    vectorizer = joblib.load(CANDIDATE_TEXT_VECTORIZER_PATH)
    text_model = joblib.load(CANDIDATE_TEXT_MODEL_PATH)

    train_df = add_text_risk_score(train_df, vectorizer, text_model)

    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(train_df["risk"] == 0).sum() / (train_df["risk"] == 1).sum(),
        random_state=RANDOM_STATE, eval_metric="logloss"
    )
    model.fit(train_df[FEATURE_COLS], train_df["risk"])

    joblib.dump(model, CANDIDATE_STRUCTURED_MODEL_PATH)
    print(f"Candidate structured model trained on {len(train_df)} rows using features: {FEATURE_COLS}")

if __name__ == "__main__":
    train_structured_model()