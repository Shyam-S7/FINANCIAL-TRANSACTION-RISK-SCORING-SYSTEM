import os
import yaml
from src.utils.logger import logger
import joblib
import json
from pathlib import Path

def read_yaml(path_to_yaml: Path) -> dict:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return content
    except Exception as e:
        raise e

def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

def save_object(file_path: Path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Object saved at {file_path}")
    except Exception as e:
        raise e

def load_object(file_path: Path):
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")
        return joblib.load(file_path)
    except Exception as e:
        raise e

def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")
