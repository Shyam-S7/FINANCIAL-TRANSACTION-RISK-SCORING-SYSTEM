from src.config.paths import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)
from src.utils.common import read_yaml, create_directories
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config['artifacts_root']])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config['data_ingestion']
        create_directories([config['root_dir']])

        return DataIngestionConfig(
            root_dir=Path(config['root_dir']),
            data_path=Path(config['data_path']),
            train_data_path=Path(config['train_data_path']),
            test_data_path=Path(config['test_data_path']),
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config['data_validation']
        schema = self.schema['columns']
        create_directories([config['root_dir']])

        return DataValidationConfig(
            root_dir=Path(config['root_dir']),
            status_file=Path(config['status_file']),
            all_schema=schema,
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config['data_transformation']
        create_directories([config['root_dir']])

        return DataTransformationConfig(
            root_dir=Path(config['root_dir']),
            train_data_path=Path(config['train_data_path']),
            test_data_path=Path(config['test_data_path']),
            preprocessor_path=Path(config['preprocessor_path']),
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config['model_trainer']
        params = self.params
        schema = self.schema['target_column']
        create_directories([config['root_dir']])

        return ModelTrainerConfig(
            root_dir=Path(config['root_dir']),
            train_data_path=Path(self.config['data_transformation']['train_data_path']),
            test_data_path=Path(self.config['data_transformation']['test_data_path']),
            model_path=Path(config['model_path']),
            target_column=schema['name'],
            params=params,
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config['model_evaluation']
        schema = self.schema['target_column']
        create_directories([config['root_dir']])

        return ModelEvaluationConfig(
            root_dir=Path(config['root_dir']),
            test_data_path=Path(self.config['data_transformation']['test_data_path']),
            model_path=Path(self.config['model_trainer']['model_path']),
            metrics_file=Path(config['metrics_file']),
            target_column=schema['name'],
        )
