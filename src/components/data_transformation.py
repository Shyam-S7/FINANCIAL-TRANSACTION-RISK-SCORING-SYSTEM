import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

from src.utils.logger import logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import DataTransformationConfig
from src.components.feature_engineering import FeatureEngineering
from src.utils.common import save_object


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.fe = FeatureEngineering()

    def get_data_transformer_object(
        self,
        numerical_columns,
        categorical_columns,
    ):
        try:

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns),
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

            # REDUCE DATA SIZE FOR FAST TRAINING
            train_df = train_df.sample(300000, random_state=42)

            logger.info("Applying feature engineering")

            train_df = self.fe.create_derived_features(train_df)

            test_df = self.fe.create_derived_features(test_df)

            # DROP HIGH CARDINALITY COLUMNS
            drop_columns = ["nameOrig", "nameDest"]

            train_df = train_df.drop(columns=drop_columns)

            test_df = test_df.drop(columns=drop_columns)

            logger.info(f"Dropped columns: {drop_columns}")

            target_column_name = "isFraud"

            # NUMERICAL COLUMNS
            numerical_columns = train_df.select_dtypes(
                exclude=["object", "str"]
            ).columns.tolist()

            if target_column_name in numerical_columns:
                numerical_columns.remove(target_column_name)

            # CATEGORICAL COLUMNS
            categorical_columns = train_df.select_dtypes(
                include=["object", "str"]
            ).columns.tolist()

            logger.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object(
                numerical_columns,
                categorical_columns,
            )

            input_feature_train_df = train_df.drop(columns=[target_column_name])

            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])

            target_feature_test_df = test_df[target_column_name]

            logger.info("Applying preprocessing object " "on training and testing data")

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # APPLY SMOTE ONLY ON TRAINING DATA
            logger.info("Applying SMOTE for class balancing")

            smote = SMOTE(random_state=42)

            input_feature_train_arr, target_feature_train_df = smote.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )

            logger.info("SMOTE applied successfully")

            logger.info(
                f"Balanced class distribution:\n"
                f"{pd.Series(target_feature_train_df).value_counts().to_string()}"
            )

            # COMBINE X AND y
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df),
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df),
            ]

            # CREATE DIRECTORY
            os.makedirs(
                os.path.dirname(self.config.preprocessor_path),
                exist_ok=True,
            )

            # SAVE PREPROCESSOR
            logger.info("Saving preprocessor object")

            save_object(
                file_path=self.config.preprocessor_path,
                obj=preprocessing_obj,
            )

            # SAVE TRANSFORMED ARRAYS
            np.save(
                self.config.train_arr_path,
                train_arr,
            )

            np.save(
                self.config.test_arr_path,
                test_arr,
            )

            logger.info("Saved transformed train and test arrays")

            logger.info("Data transformation completed successfully")

            return (
                self.config.train_arr_path,
                self.config.test_arr_path,
                self.config.preprocessor_path,
            )

        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)


if __name__ == "__main__":

    from src.config.configuration import (
        ConfigurationManager,
    )

    try:

        config = ConfigurationManager()

        data_transformation_config = config.get_data_transformation_config()

        data_transformation = DataTransformation(data_transformation_config)

        (
            train_arr_path,
            test_arr_path,
            preprocessor_path,
        ) = data_transformation.initiate_data_transformation()

        logger.info("Data Transformation Completed Successfully")

        logger.info(f"Train array saved at: {train_arr_path}")

        logger.info(f"Test array saved at: {test_arr_path}")

        logger.info(f"Preprocessor saved at: {preprocessor_path}")

    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)
