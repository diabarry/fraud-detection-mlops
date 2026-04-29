import yaml
import pandas as pd
import joblib
from pathlib import Path
from evidently import Report
from evidently.core.datasets import BinaryClassification, DataDefinition, Dataset
from evidently.presets import ClassificationPreset, DataDriftPreset

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

RAW_DATA_PATH = ROOT_DIR / config["data"]["path"]
TRAIN_PATH = ROOT_DIR / "data" / "train.csv"
MODEL_PATH = ROOT_DIR / config["model"]["model_path"]
MONITORING_DIR = ROOT_DIR / "monitoring"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)


def load_training_reference():
    if TRAIN_PATH.exists():
        print(f"📥 Loading training reference data from {TRAIN_PATH}")
        return pd.read_csv(TRAIN_PATH)

    raise FileNotFoundError(
        f"Reference file not found: {TRAIN_PATH}."
    )


def run_production_monitoring(production_path):
    production_path = Path(production_path)
    if not production_path.exists():
        raise FileNotFoundError(f"Production file not found: {production_path}")

    print(f"--- Analyzing Drift between training reference and {production_path} ---")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Reference file not found: {TRAIN_PATH}"
        )

    ref_df = load_training_reference()
    prod_df = pd.read_csv(production_path)
    model = joblib.load(MODEL_PATH)

    # Filtering features (excluding target and Time)
    features = [c for c in ref_df.columns if c not in [config["data"]["target"], "Time"]]

    ref_eval = ref_df.copy()
    prod_eval = prod_df.copy()
    
    # Generating predictions for both datasets
    ref_eval["prediction"] = model.predict_proba(ref_eval[features])[:, 1]
    prod_eval["prediction"] = model.predict_proba(prod_eval[features])[:, 1]

    # Defining classification metadata for Evidently
    classification_definition = DataDefinition(
        classification=[
            BinaryClassification(
                target=config["data"]["target"],
                prediction_probas="prediction",
            )
        ]
    )

    reference_dataset = Dataset.from_pandas(ref_eval, data_definition=classification_definition)
    current_dataset = Dataset.from_pandas(prod_eval, data_definition=classification_definition)

    # Initializing the drift report with presets
    drift_report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])

    # Running the analysis
    snapshot = drift_report.run(current_data=current_dataset, reference_data=reference_dataset)

    report_path = MONITORING_DIR / "production_drift_report.html"
    snapshot.save_html(str(report_path))

    print(f"✨ Drift report saved at: {report_path}")


if __name__ == "__main__":
    # Pointing to the production inference data
    run_production_monitoring(ROOT_DIR / "data" / "production" / "production.csv")