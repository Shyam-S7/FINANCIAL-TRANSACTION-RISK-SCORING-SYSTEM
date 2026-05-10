import sys
import pandas as pd
import os

from src.utils.logger import logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self, train_data_path: str) -> bool:

        try:
            logger.info("Starting Data Validation")

            validation_status = True

            data = pd.read_csv(train_data_path)

            all_cols = set(data.columns)

            # all_schema already contains columns dictionary
            schema_cols = set(self.config.all_schema.keys())

            logger.info(f"Dataset Columns: {all_cols}")

            logger.info(f"Schema Columns: {schema_cols}")

            missing_cols = schema_cols - all_cols

            extra_cols = all_cols - schema_cols

            if len(missing_cols) > 0:
                validation_status = False
                logger.warning(f"Missing Columns: {missing_cols}")

            if len(extra_cols) > 0:
                validation_status = False
                logger.warning(f"Extra Columns: {extra_cols}")

            os.makedirs(os.path.dirname(self.config.status_file), exist_ok=True)

            with open(self.config.status_file, "w") as f:
                f.write(f"Validation status: {validation_status}")

            logger.info(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    from src.config.configuration import ConfigurationManager

    try:

        config = ConfigurationManager()

        data_validation_config = config.get_data_validation_config()

        data_validation = DataValidation(data_validation_config)

        train_data_path = config.get_data_ingestion_config().train_data_path

        status = data_validation.validate_all_columns(str(train_data_path))

        print(status)

    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)
