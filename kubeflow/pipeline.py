"""Dummy Kubeflow Pipeline: requires kfp>=2 in the environment compiling this file.
This is a standalone pipeline demonstration, separate from the FastAPI Docker image.
"""
from kfp import dsl, compiler

@dsl.component(base_image="python:3.11-slim", packages_to_install=["pandas==2.2.3", "scikit-learn==1.6.1"])
def train_dummy_model() -> float:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import random
    random.seed(42)
    rows = []
    for _ in range(200):
        income = random.randint(20000, 140000)
        loan = random.randint(3000, 90000)
        score = random.randint(450, 850)
        rows.append([income, loan, score, int(loan / income > 0.65 or score < 580)])
    df = pd.DataFrame(rows, columns=["income", "loan_amount", "credit_score", "default"])
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="default"), df["default"], random_state=42, stratify=df["default"])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    return float(accuracy_score(y_test, model.predict(X_test)))

@dsl.pipeline(name="dummy-loan-training")
def loan_pipeline():
    train_dummy_model()

if __name__ == "__main__":
    compiler.Compiler().compile(loan_pipeline, "kubeflow/loan_pipeline.yaml")
