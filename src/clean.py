import pandas as pd
import joblib
import yaml
import logging

logger = logging.getLogger(__name__)

class Cleaner:
    def __init__(self):
        pass

    def clean_data(self, df, is_training=True):
        df = df.copy()
        logger.info("Starting data cleaning process...")
        
        # Drop the 'Time' column as it's typically not used for this specific model
        if 'Time' in df.columns:
            df = df.drop(['Time'], axis=1)
        
        return df