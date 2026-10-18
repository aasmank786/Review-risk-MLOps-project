"""This is the single place that defines paths and column names — so if something moves, you fix it in one file, not eight."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = ROOT / "data" / "raw" / "Womens Clothing E-Commerce Reviews.csv"
VALIDATED_DATA = ROOT / "data" / "processed" / "reviews_validated.csv"

FEATURES_DIR = ROOT / "data" / "features"
TRAIN_PATH = FEATURES_DIR / "train.parquet"
VAL_PATH = FEATURES_DIR / "val.parquet"
TEST_PATH = FEATURES_DIR / "test.parquet"

MODELS_DIR = ROOT / "models"
CANDIDATE_DIR = MODELS_DIR / "candidate"
ARCHIVE_DIR = MODELS_DIR / "archive"

# LIVE paths — what predict.py / the API actually load. Only promote_model.py writes here.
TEXT_VECTORIZER_PATH = MODELS_DIR / "text_vectorizer.joblib"
TEXT_MODEL_PATH = MODELS_DIR / "text_model.joblib"
STRUCTURED_MODEL_PATH = MODELS_DIR / "structured_model.joblib"
ENCODERS_PATH = MODELS_DIR / "encoders.joblib"

# CANDIDATE paths — where training and evaluation read/write, before promotion decides
# whether a candidate is good enough to become the new LIVE version.
CANDIDATE_TEXT_VECTORIZER_PATH = CANDIDATE_DIR / "text_vectorizer.joblib"
CANDIDATE_TEXT_MODEL_PATH = CANDIDATE_DIR / "text_model.joblib"
CANDIDATE_STRUCTURED_MODEL_PATH = CANDIDATE_DIR / "structured_model.joblib"
CANDIDATE_ENCODERS_PATH = CANDIDATE_DIR / "encoders.joblib"

APP_CONFIG_DIR = ROOT / "app_config"
MODEL_VERSION_FILE = APP_CONFIG_DIR / "model_version.json"
EVAL_METRICS_FILE = ROOT / "eval_metrics.json"
PROMOTE_REPORT_FILE = ROOT / "promote_report.json"

# Raw dataset's actual column names — change here if your CSV differs
COL_RATING = "Rating"
COL_RECOMMENDED = "Recommended IND"
COL_TEXT = "Review Text"
COL_TITLE = "Title"
COL_CLOTHING_ID = "Clothing ID"
COL_DIVISION = "Division Name"
COL_DEPARTMENT = "Department Name"
COL_CLASS = "Class Name"
COL_AGE = "Age"

CAT_COLS = [COL_DIVISION, COL_DEPARTMENT, COL_CLASS]
FEATURE_COLS = CAT_COLS + [COL_AGE, "review_length", "review_word_count", "text_risk_score"]

RANDOM_STATE = 42