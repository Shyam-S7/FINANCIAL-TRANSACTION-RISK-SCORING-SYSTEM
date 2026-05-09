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

                logger.warning(f"Data file not found at {self.config.data_path}")

                raise FileNotFoundError(f"CSV file not found: {self.config.data_path}")

            # LOAD CSV
            df = pd.read_csv(self.config.data_path)

            logger.info("Read the dataset as dataframe")

            # LOG DATASET INFO
            logger.info(f"Dataset Shape: {df.shape}")

            logger.info(f"Dataset Columns: {list(df.columns)}")

            logger.info(f"First 5 Rows:\n{df.head().to_string()}")

            logger.info(
                f"Target Distribution:\n{df['isFraud'].value_counts().to_string()}"
            )
            df.drop(columns=["nameOrig", "nameDest"], inplace=True)

            logger.info("Drop nameOrig and nameDest columns")

            # CREATE ARTIFACTS FOLDER
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            logger.info("Initiating train test split")

            # TRAIN TEST SPLIT
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df["isFraud"]
            )

            logger.info(f"Train Shape: {train_set.shape}")

            logger.info(f"Test Shape: {test_set.shape}")

            # SAVE FILES
            train_set.to_csv(self.config.train_data_path, index=False, header=True)

            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logger.info("Train and test CSV files saved")

            logger.info("Data ingestion completed")

            return (self.config.train_data_path, self.config.test_data_path)

        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)


if __name__ == "__main__":

    from src.config.configuration import ConfigurationManager

    try:
        config = ConfigurationManager()

        data_ingestion_config = config.get_data_ingestion_config()

        data_ingestion = DataIngestion(data_ingestion_config)

        data_ingestion.initiate_data_ingestion()

    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)
