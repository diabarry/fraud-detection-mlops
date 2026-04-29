import pandas as pd
import yaml
import mlflow
import joblib
import logging
import os
import great_expectations as ge
from great_expectations.data_context.types.base import DataContextConfig
from great_expectations.datasource.fluent.pandas_datasource import PandasDatasource
from great_expectations.datasource.fluent.batch_request import BatchRequest

# Importing logic blocks (modules)
from src.ingest import Ingestion
from src.clean import Cleaner
from src.train import Trainer
from src.evaluate import Evaluator
from explainability import Explainer

import logging

# Configuration for logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log") # Optional: keeps a written record
    ]
)
logger = logging.getLogger(__name__)


def validate_data(df: pd.DataFrame):
    """
    Data Quality Step (Great Expectations).
    Ensures that incoming data adheres to the defined contract.
    """
    logging.info("--- Step: Data Quality Validation ---")

    project_config = DataContextConfig(
        config_version=2.0,
        expectations_store_name="expectations_store",
        validation_results_store_name="validation_results_store",
        checkpoint_store_name="checkpoint_store",
        stores={
            "expectations_store": {"class_name": "InMemoryStoreBackend"},
            "validation_results_store": {"class_name": "InMemoryStoreBackend"},
            "checkpoint_store": {"class_name": "InMemoryStoreBackend"},
            "evaluation_parameter_store": {"class_name": "InMemoryStoreBackend"},
        },
        data_docs_sites={},
    )

    context = ge.get_context(project_config=project_config)
    datasource = PandasDatasource(name="pandas")
    context.add_datasource(datasource=datasource)
    datasource.add_dataframe_asset(name="asset")

    batch_request = BatchRequest(
        datasource_name="pandas",
        data_asset_name="asset",
        options={"dataframe": df},
    )
    validator = context.get_validator(
        batch_request=batch_request,
        create_expectation_suite_with_name="default",
    )

    # 1. Verify that the target column 'Class' exists
    # 2. Verify that 'Amount' has no null values
    # 3. Verify that 'Class' contains only 0 or 1
    res1 = validator.expect_column_to_exist("Class")
    res2 = validator.expect_column_values_to_not_be_null("Amount")
    res3 = validator.expect_column_values_to_be_in_set("Class", [0, 1])

    if not (res1["success"] and res2["success"] and res3["success"]):
        logging.error("ERROR: Data quality is poor!")
        return False

    logging.info("SUCCESS: Data quality validated.")
    return True

def main():
    # 1. Load configuration
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Ingestion (Via local DVC)
    ingestor = Ingestion()
    df = ingestor.load_data()

    # 3. Raw data validation
    if not validate_data(df):
        return # Halt the pipeline if data is corrupted

    # 4. Start MLflow tracking cycle
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    with mlflow.start_run(run_name="Fraud_Detection_Training"):
        
        # 5. Cleaning and Scaling (Saves scaler.pkl)
        cleaner = Cleaner()
        df_cleaned = cleaner.clean_data(df)

        # 6. Training
        trainer = Trainer(config)
        X_train, X_test, y_train, y_test = trainer.split_and_scale(df_cleaned)
        trainer.train_model(X_train, y_train)
        trainer.save_artifacts(config["model"]["model_path"],config["model"]["scaler_path"]) # Saves model.pkl
        

        # 7. Evaluation & Drift
        evaluator = Evaluator()
        roc, pr = evaluator.evaluate(trainer.model, X_test, y_test, X_train, y_train)

        explainer = Explainer()
        explainer.generate_summary_plot(X_test.head(100)) # test echantillon for explanation

        # 8. Logging results and artifacts
        mlflow.log_params(config["model"]["params"])
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("pr_auc", pr)
        
        # Logging files to make them retrievable via the MLflow UI
        mlflow.log_artifact(config["model"]["model_path"])
        mlflow.log_artifact(config["model"]["scaler_path"])
        mlflow.log_artifact("monitoring/explanations/shap_summary.png")
        
        logging.info(f"SUCCESS: Pipeline completed successfully. PR-AUC: {pr:.4f}")

if __name__ == "__main__":
    main()