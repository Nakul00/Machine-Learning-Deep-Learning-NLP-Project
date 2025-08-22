import os, random, numpy as np, joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

CATEGORIES = ["rec.autos", "sci.med"]

def main():
    data = fetch_20newsgroups(subset="train", categories=CATEGORIES, remove=("headers","footers","quotes"))
    X_train, X_valid, y_train, y_valid = train_test_split(
        data.data, data.target, test_size=0.2, random_state=SEED, stratify=data.target
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_valid)
    acc = accuracy_score(y_valid, preds)
    report = classification_report(y_valid, preds, target_names=CATEGORIES)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipe, "artifacts/model.pkl")
    with open("artifacts/metrics.txt", "w") as f:
        f.write(f"accuracy={acc:.4f}\n{report}")

    print(f"Saved artifacts/model.pkl with accuracy={acc:.4f}")

if __name__ == "__main__":
    main()
