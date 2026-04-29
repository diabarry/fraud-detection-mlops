import pandas as pd
import numpy as np
import yaml
from pathlib import Path

# Path resolution
ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yml"

# Load project configuration
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

DATA_PATH = ROOT_DIR / config["data"]["path"]
PRODUCTION_DIR = ROOT_DIR / "data" / "production"
PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)


def generate_prod_data(n_samples=500, drift=False):
    """
    Generates synthetic production data.
    If drift is True, it shifts the distribution of certain features 
    to test monitoring alerts.
    """
    # Define columns based on training data structure (Time, V1-V28, Amount, Class)
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    
    # Generate random data (mean 0, std dev 1 for V features)
    data = np.random.randn(n_samples, len(columns))
    df = pd.DataFrame(data, columns=columns)
    
    # Adjust Time and Amount for realistic positive values
    df["Time"] = np.random.randint(0, 100000, n_samples)
    df["Amount"] = np.random.exponential(scale=50, size=n_samples)
    
    # Simulate binary class (0 is majority, 1 is rare fraud)
    df["Class"] = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])

    if drift:
        print("🚨 Generating data with DRIFT (anomalies)...")
        # Artificial shift in feature distributions
        df["Amount"] = df["Amount"] * 5
        df["V1"] = df["V1"] + 3.0
        df["V2"] = df["V2"] - 2.5
    else:
        print("✅ Generating normal production data...")

    return df


if __name__ == "__main__":
    print(f"Loading configuration from {CONFIG_PATH}")
    print(f"Configured data path: {DATA_PATH}")
    print(f"Production data will be saved in: {PRODUCTION_DIR}")

    # Create a mix of normal and drifted data
    prod_normal = generate_prod_data(n_samples=1000, drift=False)
    prod_drift = generate_prod_data(n_samples=1000, drift=True)

    # Concatenate and save to CSV
    combined = pd.concat([prod_normal, prod_drift], ignore_index=True)
    output_file = PRODUCTION_DIR / "production.csv"
    combined.to_csv(output_file, index=False)

    print(f"\nFile successfully created at: {output_file.resolve()}")