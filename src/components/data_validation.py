import sys
import pandas as pd
from src.utils.logger import logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import DataValidationConfig
import os

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self, train_data_path: str) -> bool:
        try:
            validation_status = True
            logger.info("Validating dataset schema")

            data = pd.read_csv(train_data_path)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    logger.warning(f"Column {col} not found in schema")

            # Write status
            os.makedirs(os.path.dirname(self.config.status_file), exist_ok=True)
            with open(self.config.status_file, "w") as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise CustomException(e, sys)
