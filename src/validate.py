import pandas as pd
import great_expectations as ge
import logging
from great_expectations.data_context.types.base import DataContextConfig
from great_expectations.datasource.fluent.pandas_datasource import PandasDatasource
from great_expectations.datasource.fluent.batch_request import BatchRequest

logger = logging.getLogger(__name__)

def validate_data(df: pd.DataFrame):
    """
    Performs data quality checks using Great Expectations to ensure 
    the dataset conforms to the required schema and constraints.
    """
    logger.info("--- 🛡️ Step: Data Quality Validation (Great Expectations) ---")

    # Configure an ephemeral context using InMemoryStore for rapid validation
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

    # Initialize Great Expectations context and datasource
    context = ge.get_context(project_config=project_config)
    datasource = PandasDatasource(name="pandas")
    context.add_datasource(datasource=datasource)
    datasource.add_dataframe_asset(name="asset")

    # Define the batch request targeting the current DataFrame
    batch_request = BatchRequest(
        datasource_name="pandas",
        data_asset_name="asset",
        options={"dataframe": df},
    )
    
    # Create or retrieve a validator with a temporary suite
    validator = context.get_validator(
        batch_request=batch_request,
        create_expectation_suite_with_name="default",
    )

    # --- Quality Expectations ---
    results = [
        validator.expect_column_to_exist("Amount"),
        validator.expect_column_values_to_not_be_null("Amount"),
        # Validating PCA feature ranges to ensure data integrity
        validator.expect_column_values_to_be_between("V1", min_value=-100, max_value=100),
    ]

    # Aggregate validation results
    success = all(res["success"] for res in results)
    
    if not success:
        logger.error("❌ Validation failure: Dataset does not meet quality requirements!")
    else:
        logger.info("✅ Validation successful: Data schema and integrity verified.")
        
    return success