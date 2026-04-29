import pandas as pd
import yaml
import logging

class Ingestion:
    def __init__(self):
        logging.info("--- Data Ingestion started ---")
        with open("config.yml", "r") as f:
            self.config = yaml.safe_load(f)

    def load_data(self):
        try:
            df = pd.read_csv(self.config["data"]["path"])
            logging.info(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logging.error(f"❌ Error during loading: {e}")
            raise e