import sys
import importlib.util
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

spec = importlib.util.spec_from_file_location("predict", ROOT / "src" / "7_prediction" / "predict.py")
predict_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_module)

app = FastAPI(title="Review Risk API")

@app.get("/")
def root():
    return {
        "message": "Review Risk API is running",
        "docs": "/docs",
        "health": "/health"
    }

class PredictRequest(BaseModel):
    reviews: list[str] = Field(..., max_length=200)
    division: str
    department: str
    class_name: str
    age: float = 35

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    if not req.reviews:
        raise HTTPException(status_code=400, detail="reviews list cannot be empty")
    try:
        return predict_module.predict_risk(
            reviews=req.reviews, division=req.division,
            department=req.department, class_name=req.class_name, age=req.age
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))