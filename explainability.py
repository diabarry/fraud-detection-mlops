import pandas as pd
import joblib
import yaml
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Fix for Windows console display issues with special characters
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass 

# Define Project Paths
ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yml"
REPORTS_DIR = ROOT_DIR / "monitoring" / "explanations"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

class Explainer:
    """
    Class to handle Model Interpretability using SHAP.
    It provides both Global (Model-wide) and Local (Transaction-specific) explanations.
    """
    def __init__(self):
        # 1. Load Configuration file
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 2. Load Trained Model
        model_path = ROOT_DIR / self.config["model"]["model_path"]
        print(f"Loading model: {model_path}")
        self.model = joblib.load(model_path)
        
        # 3. Load Scaler (Required for data consistency)
        scaler_path = ROOT_DIR / self.config["model"]["scaler_path"]
        print(f"Loading scaler: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        
        self.explainer = None

    def preprocess_data(self, df):
        """
        Replicates the same scaling transformation used during training.
        This ensures SHAP values are calculated on the correct data scale.
        """
        df_copy = df.copy()
        # Apply scaling to 'Amount' column if it exists in the input
        if "Amount" in df_copy.columns:
            # Use transform() (not fit_transform) to use training statistics
            df_copy["Amount"] = self.scaler.transform(df_copy[["Amount"]])
        return df_copy

    def generate_summary_plot(self, X_sample):
        """
        Generates a SHAP Summary Plot to show overall feature importance.
        """
        print("--- Generating Global Summary Plot ---")
        X_scaled = self.preprocess_data(X_sample)
        
        # Use TreeExplainer for tree-based models like XGBoost
        self.explainer = shap.TreeExplainer(self.model)
        shap_values = self.explainer.shap_values(X_scaled)
        
        plt.figure(figsize=(10, 6))
        # The summary plot displays the distribution of the impact each feature has on the model
        shap.summary_plot(shap_values, X_scaled, show=False)
        
        output_path = REPORTS_DIR / "shap_summary.png"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Done: Summary plot saved to {output_path}")

    def explain_single_prediction(self, transaction_df, transaction_id=0):
        """
        Generates a Waterfall Plot to explain why a specific transaction 
        was classified as fraud or normal.
        """
        print(f"--- Explaining Transaction #{transaction_id} ---")
        X_scaled = self.preprocess_data(transaction_df)
        
        # Initialize Explainer for the Waterfall visualization API
        explainer = shap.Explainer(self.model, X_scaled)
        shap_values = explainer(X_scaled)
        
        plt.figure(figsize=(12, 8))
        # Waterfall plot shows how each feature value moves the prediction from the base value
        shap.plots.waterfall(shap_values[transaction_id], show=False)
        
        output_path = REPORTS_DIR / f"explanation_tx_{transaction_id}.png"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Done: Waterfall plot saved to {output_path}")

if __name__ == "__main__":
    try:
        # Load production data for explanation examples
        prod_data_path = ROOT_DIR / "data" / "production" / "production.csv"
        
        if not prod_data_path.exists():
            print(f"Error: Data file not found at {prod_data_path}")
        else:
            df = pd.read_csv(prod_data_path)
            
            # Remove target and non-feature columns
            X = df.drop(columns=["Class", "Time"], errors="ignore")
            
            # Initialize Explainer logic
            exp = Explainer()
            
            # 1. Global Explanation: Analyze the top 100 rows
            exp.generate_summary_plot(X.head(100))
            
            # 2. Local Explanation: Analyze the very first row
            exp.explain_single_prediction(X.head(1))
            
    except Exception as e:
        print(f"An unexpected error occurred: {e}")