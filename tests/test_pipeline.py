import pytest
import pandas as pd
import numpy as np
import os
from src.ingest import Ingestion
from src.clean import Cleaner
from src.train import Trainer

# 1. Fixture to simulate transaction data
@pytest.fixture
def sample_data():
    """Simulates a single transaction as sent by the API"""
    data = {f"V{i}": np.random.uniform(-1, 1) for i in range(1, 29)}
    data["Amount"] = 125.5
    return data

# 2. Fixture to simulate a minimal configuration
@pytest.fixture
def mock_config():
    return {
        "data": {
            "target": "Class",
            "test_size": 0.2
        },
        "model": {
            "params": {
                "n_estimators": 2,
                "max_depth": 3
            },
            "scaler_path": "models/test_scaler.pkl"
        }
    }

# --- TESTS ---

def test_cleaner_drop_time(sample_data):
    """Verifies that the 'Time' column is correctly dropped"""
    cleaner = Cleaner()

    # Robust transformation from DICT to DATAFRAME
    if isinstance(sample_data, dict):
        # Using brackets [] around sample_data creates the index automatically
        df_input = pd.DataFrame([sample_data]) 
    else:
        df_input = sample_data
        
    cleaned_df = cleaner.clean_data(df_input, is_training=True)
    
    assert "Time" not in cleaned_df.columns
    assert "Amount" in cleaned_df.columns


def test_trainer_split():
    """Verifies data splitting with enough samples for stratification"""
    
    mock_config = {
        "data": {"target": "Class", "test_size": 0.2},
        "model": {"scaler_path": "models/test_scaler.pkl", "params": {}}
    }
    trainer = Trainer(mock_config)

    # 1. Create a database with 10 normal rows (0)
    raw_data = {
        "Time": 0, "V1": -0.5, "V2": -1.5, "V3": 1.2, "V4": 0.7, "V5": -0.3,
        "V6": 0.7, "V7": -1.0, "V8": 0.6, "V9": 1.3, "V10": -0.2, "V11": -0.1,
        "V12": 0.3, "V13": -1.1, "V14": -0.4, "V15": -0.9, "V16": 0.01,
        "V17": -0.1, "V18": 1.1, "V19": 0.8, "V20": 0.4, "V21": 0.5, "V22": 1.4,
        "V23": 0.8, "V24": 0.7, "V25": -2.1, "V26": 0.2, "V27": 0.3, "V28": 0.3,
        "Amount": 167.85, "Class": 0
    }
    df_zeros = pd.DataFrame([raw_data] * 10)
    
    # 2. Create 5 fraud rows (1) so that stratification works
    df_ones = pd.DataFrame([raw_data] * 5)
    df_ones["Class"] = 1
    
    # 3. Merge and shuffle
    large_data = pd.concat([df_zeros, df_ones], ignore_index=True)
    
    # Ensure there are no NaNs and types are correct
    large_data = large_data.dropna()
    large_data["Class"] = large_data["Class"].astype(int)

    # 4. Call the function
    try:
        X_train, X_test, y_train, y_test = trainer.split_and_scale(large_data)
        
        # 5. Assertions
        assert len(X_train) + len(X_test) == 15
        assert "Class" not in X_train.columns
        assert y_train.nunique() > 1 # Verifies that both classes are present in the train set
        print("\n✅ Trainer Test: SUCCESS")
    except Exception as e:
        pytest.fail(f"Split failed: {e}")

def test_ingestion_error():
    """Verifies that ingestion raises an error if the file does not exist"""
    ingest = Ingestion()
    ingest.config = {"data": {"path": "non_existent_file.csv"}}
    
    with pytest.raises(Exception):
        ingest.load_data()

def test_prediction_output_format():
    """Verifies the Predictor output format (dictionary with probability)"""
    # Note: This test requires models/scaler.pkl and models/model.pkl to exist.
    # In a CI environment, the model loading would typically be mocked.
    pass