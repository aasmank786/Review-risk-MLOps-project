"""Job: train the sentiment/text-risk model on review text alone."""

import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import TRAIN_PATH, COL_TEXT, CANDIDATE_TEXT_VECTORIZER_PATH, CANDIDATE_TEXT_MODEL_PATH, RANDOM_STATE

def train_text_model():
    train_df = pd.read_parquet(TRAIN_PATH)

    # NOTE: deliberately NOT using stop_words="english" here. sklearn's English stop list
    # includes "not", "no", "never" — removing them before n-grams are formed turns
    # "not very good" into just "good", destroying the exact signal a sentiment model
    # needs. TF-IDF's own weighting already downweights uninformative common words.
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(train_df[COL_TEXT])

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_text, train_df["risk"])

    CANDIDATE_TEXT_VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, CANDIDATE_TEXT_VECTORIZER_PATH)
    joblib.dump(model, CANDIDATE_TEXT_MODEL_PATH)
    print(f"Candidate text model trained on {X_text.shape[0]} reviews, {X_text.shape[1]} features")

if __name__ == "__main__":
    train_text_model()