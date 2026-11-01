from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.preprocess import load_data

ROOT = Path(__file__).resolve().parents[1]

def train():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipeline.fit(X_train, y_train)
    (ROOT / "model").mkdir(exist_ok=True)
    joblib.dump(pipeline, ROOT / "model" / "model.pkl")
    return pipeline, X_test, y_test

if __name__ == "__main__":
    model, X_test, y_test = train()
    print(f"Model trained. Test accuracy: {model.score(X_test, y_test):.3f}")
