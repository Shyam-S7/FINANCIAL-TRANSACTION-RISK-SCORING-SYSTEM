import pandas as pd
import numpy as np

class FeatureEngineering:
    """
    Houses all feature engineering methods to keep transformations modular.
    These are methods we want to apply to both training and production data.
    """
    def __init__(self):
        pass

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates meaningful features based on standard financial fraud patterns.
        """
        # Copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Extract features related to balance changes
        df['errorBalanceOrg'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
        df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

        return df
