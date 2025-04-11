"""
Train and predict player market values using Random Forest.
Hyperparameters are tuned via HalvingGridSearchCV, applying log transform on targets.
"""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import Pipeline

from model_utils import run_training_pipeline, build_preprocessor


def rf_pipeline_builder(X_train) -> Pipeline:
    """
    Create a pipeline with data preprocessing and RandomForestRegressor
    wrapped in a log-transform for the target.
    """
    preprocessor = build_preprocessor(X_train)
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    regressor = TransformedTargetRegressor(
        regressor=rf, func=np.log1p, inverse_func=np.expm1
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


# Hyperparameter grid
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
        "regressor__regressor__ccp_alpha": [0.0, 0.001, 0.01],
    },
    {
        "regressor__regressor__bootstrap": [False],
        "regressor__regressor__max_samples": [None],
        "regressor__regressor__n_estimators": [50, 100, 300, 500, 1000],
        "regressor__regressor__max_depth": [None, 10, 20, 30],
        "regressor__regressor__min_samples_split": [2, 5, 10],
        "regressor__regressor__min_samples_leaf": [1, 2, 4],
        "regressor__regressor__max_features": [None, "sqrt", "log2"],
        "regressor__regressor__criterion": ["squared_error", "absolute_error"],
        "regressor__regressor__ccp_alpha": [0.0, 0.001, 0.01],
    },
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
        metrics_filename="performance_metrics_random_forest.csv",
    )