"""Job: score the pipeline on the untouched validation set, write eval_metrics.json with a pass/fail gate — 
    this file is what CI will read later to decide whether to block a bad model."""

import sys
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import (
    VAL_PATH, CANDIDATE_TEXT_VECTORIZER_PATH, CANDIDATE_TEXT_MODEL_PATH,
    CANDIDATE_STRUCTURED_MODEL_PATH, EVAL_METRICS_FILE, FEATURE_COLS
)
from src.training_utils import add_text_risk_score

MIN_RECALL = 0.55  # the gate — tune once you've seen your first real numbers

def evaluate():
    val_df = pd.read_parquet(VAL_PATH)
    vectorizer = joblib.load(CANDIDATE_TEXT_VECTORIZER_PATH)
    text_model = joblib.load(CANDIDATE_TEXT_MODEL_PATH)
    structured_model = joblib.load(CANDIDATE_STRUCTURED_MODEL_PATH)

    val_df = add_text_risk_score(val_df, vectorizer, text_model)
    preds = structured_model.predict(val_df[FEATURE_COLS])
    probs = structured_model.predict_proba(val_df[FEATURE_COLS])[:, 1]

    metrics = {
        "precision": float(precision_score(val_df["risk"], preds)),
        "recall": float(recall_score(val_df["risk"], preds)),
        "f1": float(f1_score(val_df["risk"], preds)),
        "roc_auc": float(roc_auc_score(val_df["risk"], probs)),
    }
    metrics["passed"] = bool(metrics["recall"] >= MIN_RECALL)

    with open(EVAL_METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    evaluate()