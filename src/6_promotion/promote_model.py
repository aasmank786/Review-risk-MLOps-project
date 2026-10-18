"""Job: only "promote" this model to production if it beats whatever's currently there (or nothing is there yet).
"""
# src/6_promotion/promote_model.py
import sys
from pathlib import Path
import json
import shutil
from datetime import datetime, timezone

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import (
    EVAL_METRICS_FILE, MODEL_VERSION_FILE, PROMOTE_REPORT_FILE,
    CANDIDATE_DIR, MODELS_DIR, ARCHIVE_DIR,
    TEXT_VECTORIZER_PATH, TEXT_MODEL_PATH, STRUCTURED_MODEL_PATH, ENCODERS_PATH,
    CANDIDATE_TEXT_VECTORIZER_PATH, CANDIDATE_TEXT_MODEL_PATH,
    CANDIDATE_STRUCTURED_MODEL_PATH, CANDIDATE_ENCODERS_PATH,
)

# (live_path, candidate_path) pairs — this is the only place these get paired up
LIVE_CANDIDATE_PAIRS = [
    (TEXT_VECTORIZER_PATH, CANDIDATE_TEXT_VECTORIZER_PATH),
    (TEXT_MODEL_PATH, CANDIDATE_TEXT_MODEL_PATH),
    (STRUCTURED_MODEL_PATH, CANDIDATE_STRUCTURED_MODEL_PATH),
    (ENCODERS_PATH, CANDIDATE_ENCODERS_PATH),
]

def promote():
    with open(EVAL_METRICS_FILE) as f:
        new_metrics = json.load(f)

    if not new_metrics["passed"]:
        report = {"action": "skipped", "reason": "failed evaluation gate"}
        with open(PROMOTE_REPORT_FILE, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        return

    current_recall = -1
    if MODEL_VERSION_FILE.exists():
        with open(MODEL_VERSION_FILE) as f:
            current = json.load(f)
            current_recall = current.get("eval_recall", -1)

    if new_metrics["recall"] <= current_recall:
        report = {"action": "skipped", "reason": "did not beat current champion",
                   "new_recall": new_metrics["recall"], "current_recall": current_recall}
        with open(PROMOTE_REPORT_FILE, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        return  # candidate files are left in models/candidate/ untouched — live is never touched here

    # Passed the gate AND beat the current champion — NOW it's safe to touch the live path.
    version_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    timestamped_archive = ARCHIVE_DIR / version_id
    timestamped_archive.mkdir(parents=True, exist_ok=True)

    for live_path, candidate_path in LIVE_CANDIDATE_PAIRS:
        if live_path.exists():
            shutil.copy2(live_path, timestamped_archive / live_path.name)  # back up the outgoing model
        live_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_path, live_path)  # candidate becomes live, only now

    version_info = {
        "model_version": version_id,
        "eval_recall": new_metrics["recall"],
        "eval_f1": new_metrics["f1"],
        "eval_roc_auc": new_metrics["roc_auc"],
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    MODEL_VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_VERSION_FILE, "w") as f:
        json.dump(version_info, f, indent=2)

    report = {"action": "promoted", "new_recall": new_metrics["recall"],
              "previous_recall": current_recall, "archived_previous_version_at": str(timestamped_archive)}
    with open(PROMOTE_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    promote()