from fastapi.testclient import TestClient
from api.app import app
from src.preprocess import load_data

client = TestClient(app)

def test_dataset():
    X, y = load_data()
    assert len(X) == len(y) and len(X) > 0
    assert list(X.columns) == ["income", "loan_amount", "credit_score"]

def test_health():
    assert client.get("/health").json() == {"status": "ok"}

def test_validation():
    response = client.post("/predict", json={"income": -1, "loan_amount": 1000, "credit_score": 700})
    assert response.status_code == 422
