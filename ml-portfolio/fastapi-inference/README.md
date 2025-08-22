# FastAPI Inference (Iris)

Serve the Iris classifier from `../tabular-classification/artifacts/model.pkl`.

## Run

```bash
pip install -r ../requirements.txt

# Train and save the model (if not already)
python ../tabular-classification/train.py

# Start API
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Example request

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
  "features": [5.1, 3.5, 1.4, 0.2]
}'
```
