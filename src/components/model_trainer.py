import os
import sys
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utils.logger import logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import ModelTrainerConfig
from src.utils.common import save_object


class ModelTrainer:
    """
    Purpose:
    Trains multiple models, performs hyperparameter tuning,
    selects the best model, and saves it.
    """

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self):

        try:

            logger.info("Starting model training component")

            # IMPORT HERE TO AVOID CIRCULAR IMPORTS
            from src.config.configuration import ConfigurationManager
            from src.components.data_transformation import DataTransformation
            from src.components.tuning import HyperparameterTuner

            # CONFIG
            config = ConfigurationManager()

            data_transformation_config = config.get_data_transformation_config()

            # DATA TRANSFORMATION
            data_transformation = DataTransformation(data_transformation_config)

            (
                train_data_path,
                test_data_path,
                _,
            ) = data_transformation.initiate_data_transformation()

            logger.info("Data transformation completed")

            # LOAD TRANSFORMED ARRAYS
            train_array = np.load(train_data_path)

            test_array = np.load(test_data_path)

            # SPLIT FEATURES AND TARGET
            X_train = train_array[:, :-1]

            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]

            y_test = test_array[:, -1]

            logger.info("Train and test arrays loaded successfully")

            # MODELS
            models = {
                "LogisticRegression": LogisticRegression(class_weight="balanced"),
                "RandomForestClassifier": RandomForestClassifier(
                    class_weight="balanced"
                ),
                "XGBClassifier": XGBClassifier(),
            }

            # PARAMETERS
            params = {
                "LogisticRegression": self.config.params["LogisticRegression"],
                "RandomForestClassifier": self.config.params["RandomForestClassifier"],
                "XGBClassifier": self.config.params["XGBClassifier"],
            }

            logger.info("Starting hyperparameter tuning")

            tuner = HyperparameterTuner()

            best_models = {}

            # TUNING
            for model_name, model in models.items():

                logger.info(f"Tuning model: {model_name}")

                best_model, best_params = tuner.tune_model(
                    model=model,
                    param_grid=params[model_name],
                    X_train=X_train,
                    y_train=y_train,
                    cv=3,
                    scoring="roc_auc",
                    search_type="grid",
                )

                best_models[model_name] = best_model

                logger.info(f"Best params for {model_name}: " f"{best_params}")

            logger.info("Hyperparameter tuning completed")

            # SELECT BEST MODEL
            best_model_name = None

            best_model = None

            best_score = 0

            for model_name, model in best_models.items():

                score = model.score(X_test, y_test)

                logger.info(f"{model_name} Test Score: {score:.4f}")

                if score > best_score:

                    best_score = score

                    best_model_name = model_name

                    best_model = model

            # VALIDATION
            if best_score < 0.6:

                raise CustomException(
                    "No suitable model found with score > 0.6",
                    sys,
                )

            logger.info(f"Best model found: {best_model_name}")

            logger.info(f"Best model score: {best_score:.4f}")

            # SAVE MODEL
            save_object(
                file_path=self.config.model_path,
                obj=best_model,
            )

            logger.info("Best model saved successfully")

            return self.config.model_path

        except Exception as e:

            logger.exception(e)

            raise CustomException(e, sys)


if __name__ == "__main__":

    from src.config.configuration import (
        ConfigurationManager,
    )

    try:

        config = ConfigurationManager()

        model_trainer_config = config.get_model_trainer_config()

        model_trainer = ModelTrainer(model_trainer_config)

        model_path = model_trainer.initiate_model_trainer()

        logger.info(f"Model saved at: {model_path}")

    except Exception as e:

        logger.exception(e)

        raise CustomException(e, sys)
