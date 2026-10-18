"""Job: enforce the schema and quality rules explicitly, split into valid rows that continue downstream, 
    and log what got dropped and why — never silently."""

import sys

from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import (
    RAW_DATA, VALIDATED_DATA, COL_RATING, COL_RECOMMENDED,
    COL_TEXT, COL_CLOTHING_ID
)

REQUIRED_COLS = [COL_RATING, COL_RECOMMENDED, COL_TEXT, COL_CLOTHING_ID]

def validate() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA)
    n_start = len(df)

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Rule 1: rating must be 1-5
    df = df[df[COL_RATING].between(1, 5)]

    # Rule 2: drop rows with no review text at all (empty string counts as missing here)
    df = df.dropna(subset=[COL_TEXT])
    df = df[df[COL_TEXT].str.strip().str.len() > 0]

    # Rule 3: drop exact duplicate reviews
    df = df.drop_duplicates(subset=[COL_CLOTHING_ID, COL_TEXT])

    # Rule 4: drop rows with no product id
    df = df.dropna(subset=[COL_CLOTHING_ID])

    n_end = len(df)
    print(f"Validation: {n_start} -> {n_end} rows ({n_start - n_end} dropped)")

    VALIDATED_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(VALIDATED_DATA, index=False)
    print(f"Saved validated data to {VALIDATED_DATA}")
    return df

if __name__ == "__main__":
    validate()