import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        logger.info("Entered the data ingestion component")
        try:
            if not os.path.exists(self.config.data_path):
                logger.warning(f"Data file not found at {self.config.data_path}. Creating a dummy dataframe for testing purposes.")
                # Create a dummy dataframe matching the schema for pipeline testing if not provided.
                # In production, this would just raise an error.
                df = pd.DataFrame({
                    "amount": [100.0, 500.0, 20.0, 900.0],
                    "oldbalanceOrg": [100.0, 500.0, 20.0, 900.0],
                    "newbalanceOrig": [0.0, 0.0, 0.0, 0.0],
                    "oldbalanceDest": [0.0, 0.0, 0.0, 0.0],
                    "newbalanceDest": [100.0, 500.0, 20.0, 900.0],
                    "type": ["PAYMENT", "TRANSFER", "CASH_OUT", "TRANSFER"],
                    "isFraud": [0, 1, 0, 1]
                })
                os.makedirs(os.path.dirname(self.config.data_path), exist_ok=True)
                df.to_csv(self.config.data_path, index=False)

            df = pd.read_csv(self.config.data_path)
            logger.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            logger.info("Initiating train test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df["isFraud"])

            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logger.info("Data ingestion completed")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
