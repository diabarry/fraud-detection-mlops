import logging
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

RAW_DATA_PATH = ROOT_DIR / config["data"]["path"]
TRAIN_PATH = ROOT_DIR / "data" / "train.csv"
TEST_PATH = ROOT_DIR / "data" / "test.csv"


class Trainer:
    def __init__(self, config: dict):
        self.config = config

    def split_data(self, df):
        X = df.drop(self.config["data"]["target"], axis=1)
        y = df[self.config["data"]["target"]]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config["data"]["test_size"], stratify=y, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def export_train_test(self):
        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError(f"Fichier de données brutes introuvable : {RAW_DATA_PATH}")

        df = pd.read_csv(RAW_DATA_PATH)
        X_train, X_test, y_train, y_test = self.split_data(df)

        train_df = X_train.copy()
        train_df[self.config["data"]["target"]] = y_train.values

        test_df = X_test.copy()
        test_df[self.config["data"]["target"]] = y_test.values

        train_df.to_csv(TRAIN_PATH, index=False)
        test_df.to_csv(TEST_PATH, index=False)

        logger.info(f"✅ Train exporté : {TRAIN_PATH}")
        logger.info(f"✅ Test exporté : {TEST_PATH}")
        return TRAIN_PATH, TEST_PATH


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    trainer = Trainer(config)
    trainer.export_train_test()
