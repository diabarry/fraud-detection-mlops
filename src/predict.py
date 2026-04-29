import time
import joblib
import numpy as np
import pandas as pd
import yaml
import logging

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self):
        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)
        # Load serialized model and scaler artifacts
        self.model = joblib.load(config["model"]["model_path"])
        self.scaler = joblib.load(config["model"]["scaler_path"])
        self.target_name = config["data"]["target"]

    def predict(self, features_dict):
        """
        Executes the full inference pipeline for a single transaction.
        """
        start_time = time.time()
        logger.info("New prediction request received.")
        try:
            # 1. Convert input dictionary to DataFrame
            df = pd.DataFrame([features_dict])
            
            # 2. Data Preprocessing (Must match training steps)
            if 'Time' in df.columns:
                df = df.drop(columns=['Time'])
            
            # Apply the fitted scaler to the 'Amount' feature
            df["Amount"] = self.scaler.transform(df[["Amount"]])
            
            # 3. Model Inference
            # Get the probability of the positive class (Fraud)
            proba = self.model.predict_proba(df)[0][1]
            
            # Apply decision threshold (default 0.5)
            is_fraud = int(proba > 0.5)
            duration = time.time() - start_time
                
            if is_fraud:
                logger.warning(f"🚨 FRAUD DETECTED! Probability: {proba:.4f} | Latency: {duration:.4f}s")
            else:
                logger.info(f"✅ Normal transaction. Probability: {proba:.4f} | Latency: {duration:.4f}s")
                
            return {"is_fraud": is_fraud, "probability": float(proba)}
            
        except Exception as e:
            logger.error(f"❌ Error during prediction: {str(e)}")
            raise e