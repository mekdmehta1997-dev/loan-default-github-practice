from pathlib import Path
from functools import lru_cache
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Dummy Loan Default Prediction API")

class Loan(BaseModel):
    income: float = Field(gt=0)
    loan_amount: float = Field(gt=0)
    credit_score: int = Field(ge=300, le=850)

@lru_cache
def get_model():
    path = ROOT / "model" / "model.pkl"
    if not path.exists():
        raise RuntimeError("Model missing. Run: python -m src.train")
    return joblib.load(path)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(loan: Loan):
    features = pd.DataFrame([loan.model_dump()])
    prediction = int(get_model().predict(features)[0])
    return {"default_prediction": prediction}
