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
        logger.info("Starting Data Validation")

        validation_status = True

        data = pd.read_csv(train_data_path)
        data_columns = set(data.columns)

        schema = self.config.all_schema

        # handle string schema
        if isinstance(schema, str):
            import ast
            schema = ast.literal_eval(schema)

        if not isinstance(schema, dict):
            raise Exception("Invalid schema format")

        if "columns" not in schema:
            raise Exception(f"Schema missing 'columns'. Got keys: {schema.keys()}")

        schema_columns = set(schema["columns"].keys())

        missing = schema_columns - data_columns
        extra = data_columns - schema_columns

        if missing:
            logger.error(f"Missing columns: {missing}")
            validation_status = False

        if extra:
            logger.warning(f"Extra columns ignored: {extra}")

        return validation_status

    except Exception as e:
        logger.exception("Validation failed")
        raise CustomException(e, sys)
    


if __name__ == "__main__":
    from src.config.configuration import ConfigurationManager

    try:
        config = ConfigurationManager()

        data_validation_config = config.get_data_validation_config()

        data_validation = DataValidation(data_validation_config)

        train_data_path = config.get_data_ingestion_config().train_data_path

        result = data_validation.validate_all_columns(str(train_data_path))

        logger.info(f"Validation Result: {result}")

    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)
