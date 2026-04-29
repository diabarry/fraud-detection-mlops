from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import time
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

app = FastAPI(title="Real-time Fraud Detection Service")

# --- Monitoring Metrics ---
PREDICTION_COUNT = Counter("prediction_total", "Total number of predictions")
FRAUD_DETECTED = Counter("fraud_detected_total", "Total number of detected frauds")
LATENCY = Histogram("prediction_latency_seconds", "API response latency")

# --- Artifact Loading ---
# Important: I load the model AND the scaler saved during training
try:
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
except Exception as e:
    print(f"Error loading artifacts: {e}")

# --- Data Schema (Credit Card PCA dataset) ---
class Transaction(BaseModel):
    # The 28 PCA components + Time + Amount
    Time: float = Field(..., example=406.0)
    V1: float = Field(..., example=-1.3598)
    V2: float = Field(..., example=-0.0727)
    V3: float = Field(..., example=2.5363)
    V4: float = Field(..., example=1.3781)
    V5: float = Field(..., example=-0.3383)
    V6: float = Field(..., example=0.4623)
    V7: float = Field(..., example=0.2396)
    V8: float = Field(..., example=0.0987)
    V9: float = Field(..., example=0.3637)
    V10: float = Field(..., example=0.0907)
    V11: float = Field(..., example=-0.5515)
    V12: float = Field(..., example=-0.6178)
    V13: float = Field(..., example=-0.9913)
    V14: float = Field(..., example=-0.3111)
    V15: float = Field(..., example=1.4681)
    V16: float = Field(..., example=-0.4704)
    V17: float = Field(..., example=0.2079)
    V18: float = Field(..., example=0.0257)
    V19: float = Field(..., example=0.4039)
    V20: float = Field(..., example=0.2514)
    V21: float = Field(..., example=-0.0183)
    V22: float = Field(..., example=0.2778)
    V23: float = Field(..., example=-0.1104)
    V24: float = Field(..., example=0.0669)
    V25: float = Field(..., example=0.1285)
    V26: float = Field(..., example=-0.1891)
    V27: float = Field(..., example=0.1335)
    V28: float = Field(..., example=-0.0210)
    Amount: float = Field(..., example=149.62)

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "fraud-detection", "version": "1.0.0"}

@app.get("/metrics")
def metrics():
    """Endpoint for Prometheus scraping"""
    return Response(generate_latest(), media_type="text/plain")

@app.post("/predict")
async def predict(data: Transaction):
    start_time = time.time()
    PREDICTION_COUNT.inc()
    
    try:
        # 1. Convert to DataFrame
        input_df = pd.DataFrame([data.model_dump()])
        
        # 2. Preprocessing (Applying the scaler to 'Amount' only)
        # Note: 'Time' should be dropped if because model doesn't use it
        if 'Time' in input_df.columns:
            input_df = input_df.drop(columns=['Time'])
            
        input_df["Amount"] = scaler.transform(input_df[["Amount"]])
        
        # 3. Prediction
        proba = model.predict_proba(input_df)[0][1]
        is_fraud = 1 if proba > 0.5 else 0 # Adjustable threshold
        
        if is_fraud:
            FRAUD_DETECTED.inc()
            
        # 4. Latency Logging
        LATENCY.observe(time.time() - start_time)
        
        return {
            "is_fraud": is_fraud,
            "probability": round(float(proba), 4),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))