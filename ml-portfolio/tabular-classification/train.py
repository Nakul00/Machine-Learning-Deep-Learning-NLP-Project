import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def main():
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, random_state=SEED))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipe, "artifacts/model.pkl")
    with open("artifacts/metrics.txt", "w") as f:
        f.write(f"accuracy={acc:.4f}\n")
        f.write(classification_report(y_test, preds))

    print(f"Saved artifacts/model.pkl with accuracy={acc:.4f}")

if __name__ == "__main__":
    main()
