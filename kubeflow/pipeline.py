
from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model, Metrics



@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def preprocess(
    train_data: Output[Dataset],
    test_data: Output[Dataset]
):
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )

    df = pd.DataFrame(
        X, columns=[f"feature_{i}" for i in range(5)]
    )
    df["default"] = y

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["default"]
    )

    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)

    print(f"Training records: {len(train_df)}")
    print(f"Testing records: {len(test_df)}")




@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def train(
    train_data: Input[Dataset],
    model: Output[Model]
):
    import pandas as pd
    import joblib
    from sklearn.linear_model import LogisticRegression

    # Load training data only
    df = pd.read_csv(train_data.path)

    X_train = df.drop(columns=["default"])
    y_train = df["default"]

    # Train the model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Save trained model as a Kubeflow artifact
    joblib.dump(clf, model.path)

    print("Model training completed")
    print(f"Training records: {len(df)}")




@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def evaluate(
    test_data: Input[Dataset],
    model: Input[Model],
    metrics: Output[Metrics]
):
    import pandas as pd
    import joblib
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

    # Load unseen testing data
    df = pd.read_csv(test_data.path)

    X_test = df.drop(columns=["default"])
    y_test = df["default"]

    # Load trained model
    clf = joblib.load(model.path)

    # Predict loan defaults
    predictions = clf.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    # Log metrics to Kubeflow
    metrics.log_metric("accuracy", float(accuracy))
    metrics.log_metric("precision", float(precision))
    metrics.log_metric("recall", float(recall))
    metrics.log_metric("f1_score", float(f1))

    print(f"Testing records: {len(df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")




@dsl.pipeline(
    name="loan-default-training-pipeline"
)
def loan_default_pipeline():

    # Step 1: Preprocess and split data
    preprocess_task = preprocess()

    # Step 2: Train using 80% training data
    train_task = train(
        train_data=preprocess_task.outputs["train_data"]
    )

    # Step 3: Evaluate using 20% testing data
    evaluate(
        test_data=preprocess_task.outputs["test_data"],
        model=train_task.outputs["model"]
    )



if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=loan_default_pipeline,
        package_path="kubeflow/loan_pipeline.yaml"
    )
