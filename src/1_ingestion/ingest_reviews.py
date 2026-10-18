"""Job: load the raw CSV, do the absolute minimum sanity check, hand it downstream untouched otherwise. Ingestion should never silently clean data — that's the next stage's job, 
done deliberately, not accidentally."""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import RAW_DATA

def ingest() -> pd.DataFrame:
    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"Expected raw data at {RAW_DATA}. Download the dataset and place it there first."
        )
    df = pd.read_csv(RAW_DATA)
    print(f"Ingested {len(df)} rows, {df.shape[1]} columns from {RAW_DATA.name}")
    return df

if __name__ == "__main__":
    ingest()