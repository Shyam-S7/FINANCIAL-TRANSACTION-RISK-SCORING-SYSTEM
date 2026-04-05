import pandas as pd
from src.utils.logger import logger
from src.exception.custom_exception import CustomException
import sys
from src.utils.common import load_object
from src.components.feature_engineering import FeatureEngineering
from pathlib import Path

class PredictPipeline:
    def __init__(self):
        # Using hardcoded paths for inference service ease
        self.model_path = Path("artifacts/model_trainer/best_model.joblib")
        self.preprocessor_path = Path("artifacts/data_transformation/preprocessor.joblib")

    def predict(self, data: pd.DataFrame):
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            logger.info("Applying feature engineering to input data")
            fe = FeatureEngineering()
            processed_data = fe.create_derived_features(data)

            # transform
            scaled_data = preprocessor.transform(processed_data)

            # predict
            preds = model.predict(scaled_data)
            probs = model.predict_proba(scaled_data)[:, 1] if hasattr(model, "predict_proba") else preds

            return preds, probs
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, amount_val: float, oldbalanceOrg_val: float, newbalanceOrig_val: float, oldbalanceDest_val: float, newbalanceDest_val: float, type_val: str):
        self.amount = amount_val
        self.oldbalanceOrg = oldbalanceOrg_val
        self.newbalanceOrig = newbalanceOrig_val
        self.oldbalanceDest = oldbalanceDest_val
        self.newbalanceDest = newbalanceDest_val
        self.type = type_val

    def get_data_as_dataframe(self):
        try:
            custom_data_dict = {
                "amount": [self.amount],
                "oldbalanceOrg": [self.oldbalanceOrg],
                "newbalanceOrig": [self.newbalanceOrig],
                "oldbalanceDest": [self.oldbalanceDest],
                "newbalanceDest": [self.newbalanceDest],
                "type": [self.type],
            }
            return pd.DataFrame(custom_data_dict)
        except Exception as e:
            raise CustomException(e, sys)
