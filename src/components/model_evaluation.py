import os
import sys
import numpy as np
from src.utils.logger import logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import ModelEvaluationConfig
from src.utils.common import load_object, save_json
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def initiate_model_evaluation(self):
        try:
            logger.info("Loading test data and model for evaluation")
            test_array = np.load(self.config.test_data_path)
            model = load_object(self.config.model_path)

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logger.info("Predicting on test data")
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

            logger.info("Calculating metrics")
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            metrics = {
                "ROC-AUC": roc_auc,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1
            }

            save_json(path=self.config.metrics_file, data=metrics)
            logger.info(f"Evaluation Metrics saved to {self.config.metrics_file}: {metrics}")

        except Exception as e:
            raise CustomException(e, sys)
