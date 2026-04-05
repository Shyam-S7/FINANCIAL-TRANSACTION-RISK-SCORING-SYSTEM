import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from src.utils.logger import logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import ModelTrainerConfig
from src.utils.common import save_object

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, param):
        try:
            report = {}

            for i in range(len(list(models))):
                model_name = list(models.keys())[i]
                model = list(models.values())[i]
                para = param[model_name]

                gs = GridSearchCV(model, para, cv=3, scoring='roc_auc', n_jobs=-1)
                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                y_test_pred = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
                test_model_score = roc_auc_score(y_test, y_test_pred)

                report[model_name] = test_model_score
                logger.info(f"{model_name} best params: {gs.best_params_} with AUC: {test_model_score}")

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self):
        try:
            train_array = np.load(self.config.train_data_path)
            test_array = np.load(self.config.test_data_path)

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "LogisticRegression": LogisticRegression(),
                "RandomForestClassifier": RandomForestClassifier(),
                "XGBClassifier": XGBClassifier()
            }

            params = {
                "LogisticRegression": self.config.params["LogisticRegression"],
                "RandomForestClassifier": self.config.params["RandomForestClassifier"],
                "XGBClassifier": self.config.params["XGBClassifier"]
            }

            model_report: dict = self.evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            # In production, we typically enforce a minimum AUC score, e.g., 0.6
            if best_model_score < 0.6:
                raise CustomException("No suitable model found with AUC > 0.6")
            
            logger.info(f"Best found model on testing dataset: {best_model_name}")

            save_object(
                file_path=self.config.model_path,
                obj=best_model
            )

        except Exception as e:
            raise CustomException(e, sys)
