from sklearn.metrics import accuracy_score, classification_report
from src.train import train

if __name__ == "__main__":
    model, X_test, y_test = train()
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions, zero_division=0))
