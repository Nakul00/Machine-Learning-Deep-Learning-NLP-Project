import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, classification_report

CATEGORIES = ["rec.autos", "sci.med"]

def main():
    test = fetch_20newsgroups(subset="test", categories=CATEGORIES, remove=("headers","footers","quotes"))
    model = joblib.load("artifacts/model.pkl")
    preds = model.predict(test.data)
    print(f"Accuracy: {accuracy_score(test.target, preds):.4f}")
    print(classification_report(test.target, preds, target_names=CATEGORIES))

if __name__ == "__main__":
    main()
