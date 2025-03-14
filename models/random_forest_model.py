"""
random_forest_model.py

Full Pipeline for Training and Predicting Player Transfer Prices using Random Forest.

This script uses HalvingGridSearchCV to train and evaluate a Random Forest model on three
preprocessed dataset variants. Performance metrics are saved to a CSV file.
"""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import Pipeline

from model_utils import run_training_pipeline, build_preprocessor


def rf_pipeline_builder(X_train) -> Pipeline:
    """
    Build a scikit-learn pipeline for Random Forest regression.

    This function creates a pipeline that preprocesses the training data and then fits a Random
    Forest regressor wrapped in a TransformedTargetRegressor to apply a log1p transform to the target.

    Args:
        X_train: The training features used to fit the preprocessor.

    Returns:
        Pipeline: A pipeline with preprocessing and Random Forest regression steps.
    """
    preprocessor = build_preprocessor(X_train)
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    regressor = TransformedTargetRegressor(regressor=rf, func=np.log1p, inverse_func=np.expm1)
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


# Define hyperparameter grid for Random Forest.
rf_param_grid = [
    {
        "regressor__regressor__bootstrap": [True],
        "regressor__regressor__max_samples": [None, 0.8, 0.6],
        "regressor__regressor__n_estimators": [50, 100, 300, 500, 1000],
        "regressor__regressor__max_depth": [None, 10, 20, 30],
        "regressor__regressor__min_samples_split": [2, 5, 10],
        "regressor__regressor__min_samples_leaf": [1, 2, 4],
        "regressor__regressor__max_features": [None, "sqrt", "log2"],
        "regressor__regressor__criterion": ["squared_error", "absolute_error"],
        "regressor__regressor__ccp_alpha": [0.0, 0.001, 0.01]
    },
    {
        "regressor__regressor__bootstrap": [False],
        "regressor__regressor__max_samples": [None],  # Must be None when bootstrap is False.
        "regressor__regressor__n_estimators": [50, 100, 300, 500, 1000],
        "regressor__regressor__max_depth": [None, 10, 20, 30],
        "regressor__regressor__min_samples_split": [2, 5, 10],
        "regressor__regressor__min_samples_leaf": [1, 2, 4],
        "regressor__regressor__max_features": [None, "sqrt", "log2"],
        "regressor__regressor__criterion": ["squared_error", "absolute_error"],
        "regressor__regressor__ccp_alpha": [0.0, 0.001, 0.01]
    }
]

if __name__ == "__main__":
    run_training_pipeline(
        model_name="Random Forest",
        pipeline_builder=rf_pipeline_builder,
        search_class=HalvingGridSearchCV,
        param_grid=rf_param_grid,
        use_sample_weight=False,
        model_filename="random_forest_model",
        predictions_subdir="random_forest",
        metrics_filename="performance_metrics_random_forest.csv"
    )