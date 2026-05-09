import sys
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.utils.logger import logger
from src.exception.custom_exception import CustomException


class HyperparameterTuner:
    """
    A dedicated component for handling hyperparameter tuning across different models.
    Supports both Grid Search and Randomized Search strategies.
    """

    def __init__(self):
        pass

    def tune_model(
        self,
        model,
        param_grid,
        X_train,
        y_train,
        cv=2,
        scoring="roc_auc",
        search_type="random",
        n_iter=10,
    ):
        """
        Tunes a given model using the specified search strategy.

        Args:
            model: The machine learning model to tune.
            param_grid (dict): Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
            X_train: Training features.
            y_train: Training target.
            cv (int): Number of cross-validation folds.
            scoring (str): Metric to evaluate.
            search_type (str): "grid" for GridSearchCV or "random" for RandomizedSearchCV.
            n_iter (int): Number of iterations for RandomizedSearchCV.

        Returns:
            best_estimator_: The fully trained model with optimal parameters.
            best_params_: Dictionary of the best parameters.
        """
        try:
            model_name = model.__class__.__name__
            logger.info(f"Starting {search_type} search for {model_name}")

            if search_type.lower() == "random":
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    random_state=42,
                    verbose=1,
                )
            else:
                search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=1,
                )

            # Fit the search to the training data
            search.fit(X_train, y_train)

            logger.info(f"Successfully tuned {model_name}.")
            logger.info(f"Best parameters: {search.best_params_}")
            logger.info(f"Best {scoring} score: {search.best_score_:.4f}")

            return search.best_estimator_, search.best_params_

        except Exception as e:
            raise CustomException(e, sys)
