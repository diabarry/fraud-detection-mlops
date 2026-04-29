# from asyncio.log import logger
import logging
logger = logging.getLogger(__name__)

from evidently import Report
from evidently.core.datasets import BinaryClassification, DataDefinition, Dataset
from evidently.presets import ClassificationPreset, DataDriftPreset
from evidently.metrics import *
from sklearn.metrics import average_precision_score, roc_auc_score

class Evaluator:
    def evaluate(self, model, X_test, y_test, X_train, y_train):
        logger.info("Evaluating model performance on the test set...")
        
        # Calculate prediction probabilities
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        
        logger.info(f"Final Results: ROC-AUC = {roc_auc:.4f} | PR-AUC = {pr_auc:.4f}")
        
        # Check against performance threshold
        if pr_auc < 0.80:
            logger.warning("⚠️ Warning: PR-AUC is below the expected performance threshold!")

        # Prepare reference data (training set) for Evidently
        train_eval = X_train.copy()
        train_eval["Class"] = y_train
        train_eval["prediction"] = model.predict_proba(X_train)[:, 1]

        # Prepare current data (test set) for Evidently
        test_eval = X_test.copy()
        test_eval["Class"] = y_test
        test_eval["prediction"] = y_proba

        # Define classification metadata
        classification_definition = DataDefinition(
            classification=[
                BinaryClassification(
                    target="Class",
                    prediction_probas="prediction",
                )
            ]
        )

        # Create Evidently Dataset objects
        reference_dataset = Dataset.from_pandas(train_eval, data_definition=classification_definition)
        current_dataset = Dataset.from_pandas(test_eval, data_definition=classification_definition)

        # Initialize and run the Drift and Classification report
        drift_report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
        snapshot = drift_report.run(current_data=current_dataset, reference_data=reference_dataset)

        # Save the report as an HTML file
        report_path = "monitoring/drift_report.html"
        snapshot.save_html(report_path)

        return roc_auc, pr_auc