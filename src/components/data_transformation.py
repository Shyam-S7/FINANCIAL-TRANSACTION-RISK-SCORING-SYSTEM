import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.utils.logger import logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import DataTransformationConfig
from src.components.feature_engineering import FeatureEngineering
from src.utils.common import save_object

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.fe = FeatureEngineering()

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        try:
            logger.info("Reading train and test data")
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            logger.info("Applying feature engineering")
            train_df = self.fe.create_derived_features(train_df)
            test_df = self.fe.create_derived_features(test_df)

            target_column_name = "isFraud"
            
            # Select dynamical categorical and numerical
            numerical_columns = train_df.select_dtypes(exclude="object").columns.tolist()
            if target_column_name in numerical_columns:
                numerical_columns.remove(target_column_name)

            categorical_columns = train_df.select_dtypes(include="object").columns.tolist()

            logger.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(numerical_columns, categorical_columns)

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logger.info(f"Applying preprocessing object on training and testing data")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logger.info(f"Saving preprocessor block")

            save_object(
                file_path=self.config.preprocessor_path,
                obj=preprocessing_obj
            )

            np.save(self.config.train_data_path, train_arr)
            np.save(self.config.test_data_path, test_arr)

            return (
                self.config.train_data_path,
                self.config.test_data_path,
                self.config.preprocessor_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
