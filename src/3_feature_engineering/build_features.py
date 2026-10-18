"""Job: create the target label, split by product (not by row — this matters), 
    build structured features, and save train/val/test."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import (
    VALIDATED_DATA, FEATURES_DIR, TRAIN_PATH, VAL_PATH, TEST_PATH,
    COL_RATING, COL_RECOMMENDED, COL_TEXT, COL_CLOTHING_ID, CAT_COLS,
    COL_AGE, CANDIDATE_ENCODERS_PATH, RANDOM_STATE
)

def build_features():
    df = pd.read_csv(VALIDATED_DATA)

    # Proxy target — documented assumption: no real "returned" label exists in this
    # dataset, so risk = low rating OR not recommended.
    # NOTE: Recommended IND can itself be missing for some rows. (df[COL_RECOMMENDED] == 0)
    # evaluates to False for NaN, so those rows fall back to being judged on rating alone —
    # intentional, but worth knowing rather than assuming every row used both signals.
    df["risk"] = ((df[COL_RATING] <= 2) | (df[COL_RECOMMENDED] == 0)).astype(int)

    # Structured features that don't depend on any fitted statistic — safe to compute before the split
    df["review_length"] = df[COL_TEXT].str.len()
    df["review_word_count"] = df[COL_TEXT].str.split().str.len()
    for col in CAT_COLS:
        df[col] = df[col].fillna("Unknown")

    # Split by PRODUCT, not by row — a product's reviews all land in one split.
    # Do this BEFORE fitting anything statistical (age median, category encoders) — fitting
    # those on the full dataset first would leak val/test information into train, the same
    # class of bug we caught earlier in the store-sales notebook.
    product_ids = df[COL_CLOTHING_ID].unique()
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(product_ids)
    n = len(product_ids)
    train_ids = set(product_ids[: int(n * 0.7)])
    val_ids = set(product_ids[int(n * 0.7): int(n * 0.85)])
    test_ids = set(product_ids[int(n * 0.85):])

    train_df = df[df[COL_CLOTHING_ID].isin(train_ids)].copy()
    val_df = df[df[COL_CLOTHING_ID].isin(val_ids)].copy()
    test_df = df[df[COL_CLOTHING_ID].isin(test_ids)].copy()

    # Fit the age imputer's median on TRAIN ONLY, then apply the same value everywhere
    train_age_median = train_df[COL_AGE].median()
    for split_df in (train_df, val_df, test_df):
        split_df[COL_AGE] = split_df[COL_AGE].fillna(train_age_median)

    # Fit category encoders on TRAIN ONLY. "Unknown" is guaranteed to be one of the classes
    # because we fillna'd it above — this is what lets serving-time map any never-seen
    # category to "Unknown" instead of an out-of-distribution value the model never learned.
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        encoders[col] = le
        for split_df in (val_df, test_df):
            known = set(le.classes_)
            split_df[col] = split_df[col].astype(str).where(split_df[col].astype(str).isin(known), "Unknown")
            split_df[col] = le.transform(split_df[col])
    CANDIDATE_ENCODERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, CANDIDATE_ENCODERS_PATH)
    joblib.dump(train_age_median, CANDIDATE_ENCODERS_PATH.parent / "age_median.joblib")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(TRAIN_PATH, index=False)
    val_df.to_parquet(VAL_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)

    print(f"Train: {len(train_df)} rows ({len(train_ids)} products)")
    print(f"Val:   {len(val_df)} rows ({len(val_ids)} products)")
    print(f"Test:  {len(test_df)} rows ({len(test_ids)} products)")
    print(f"Risk rate — train: {train_df['risk'].mean():.3f}, val: {val_df['risk'].mean():.3f}")

if __name__ == "__main__":
    build_features()