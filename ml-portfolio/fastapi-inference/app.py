from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import joblib, os

MODEL_PATH = os.path.join("..", "tabular-classification", "artifacts", "model.pkl")

app = FastAPI(title="Iris Inference API")

class IrisRequest(BaseModel):
    features: conlist(float, min_items=4, max_items=4)

@app.on_event("startup")
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model file not found. Run ../tabular-classification/train.py first.")
    global model
    model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(req: IrisRequest):
    try:
        pred = model.predict([req.features])[0]
        proba = getattr(model, "predict_proba", lambda X: None)([req.features])
        return {"prediction": int(pred), "probabilities": (proba[0].tolist() if proba is not None else None)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
