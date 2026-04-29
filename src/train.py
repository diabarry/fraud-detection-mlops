import mlflow.xgboost
import logging
import joblib
from xgboost import XGBClassifier
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        self.model = None

    def split_and_scale(self, df):
        """
        Splits the dataset into training and testing sets, then scales the 'Amount' feature.
        """
        X = df.drop(self.config["data"]["target"], axis=1)
        y = df[self.config["data"]["target"]]
        
        # Use stratified split to maintain the fraud/normal ratio in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config["data"]["test_size"], stratify=y, random_state=42
        )

        # FIT only on training data to prevent data leakage
        logger.info("Scaling features (Fit on Train, Transform on Test)...")
        X_train["Amount"] = self.scaler.fit_transform(X_train[["Amount"]])
        X_test["Amount"] = self.scaler.transform(X_test[["Amount"]])
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        Trains the XGBoost classifier with automated MLflow tracking.
        """
        logger.info(f"Starting training with {len(X_train)} samples...")
        
        # Calculate weight to handle class imbalance (Fraud is rare)
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        
        # Enable MLflow autologging for XGBoost to capture params and metrics
        mlflow.xgboost.autolog() 
        
        self.model = XGBClassifier(
            **self.config["model"]["params"],
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False
        )
        
        self.model.fit(X_train, y_train)
        logger.info("Model training complete.")

    def save_artifacts(self, model_path, scaler_path):
        """
        Serializes the model and the fitted scaler to the specified paths.
        """
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Model and Scaler artifacts saved to: {model_path} and {scaler_path}")