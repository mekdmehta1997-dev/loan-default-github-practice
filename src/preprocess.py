from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["income", "loan_amount", "credit_score"]

def load_data(path=None):
    df = pd.read_csv(path or ROOT / "data" / "loans.csv")
    return df[FEATURES], df["default"]
