from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_path: Path
    train_data_path: Path
    test_data_path: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: Path
    all_schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    preprocessor_path: Path
    train_arr_path: Path
    test_arr_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_arr_path: Path
    test_arr_path: Path
    model_path: Path
    target_column: str
    params: dict

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_arr_path: Path
    model_path: Path
    metrics_file: Path
    target_column: str
