"""Since both training and evaluation need add_text_risk_score, pull it into a shared module instead of copy-pasting:"""

import pandas as pd
from src.config import COL_TEXT

def add_text_risk_score(df: pd.DataFrame, vectorizer, text_model) -> pd.DataFrame:
    X_text = vectorizer.transform(df[COL_TEXT])
    df = df.copy()
    df["text_risk_score"] = text_model.predict_proba(X_text)[:, 1]
    return df