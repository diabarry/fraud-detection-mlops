import time
from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import Predictor
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

app = FastAPI()
predictor = Predictor()

# --- Metrics for Grafana ---
# 1. To measure response time (latency)
PRED_LATENCY = Histogram("prediction_duration_seconds", "API response time")

# 2. To count ALL predictions (Essential for dashboard charting!)
PREDICTIONS_TOTAL = Counter("predictions_total", "Total number of processed transactions")

# 3. To count only fraud cases
FRAUD_COUNT = Counter("fraud_detected_total", "Total number of detected frauds")

class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.post("/predict")
async def predict(data: Transaction):
    # Increment the global counter on every call
    PREDICTIONS_TOTAL.inc()

    # Measure execution time
    with PRED_LATENCY.time():
        result = predictor.predict(data.model_dump())

    # Increment the fraud counter if the model predicts fraud
    # Note: check if predictor returns 1/0 or True/False
    if result.get("is_fraud") or result.get("prediction") == 1:
        FRAUD_COUNT.inc()
        
    return result

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")