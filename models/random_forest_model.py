"""
Full Pipeline for Training and Predicting Player Transfer Prices using Random Forest.

This script uses HalvingGridSearchCV to train and evaluate a Random Forest model on three
preprocessed variants. Performance metrics are saved to a CSV.
"""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import Pipeline

from model_utils import run_training_pipeline, build_preprocessor


def rf_pipeline_builder(X_train) -> Pipeline:
    """Builds the pipeline for Random Forest."""
    preprocessor = build_preprocessor(X_train)
    rf = RandomForestRegressor(random_state=42)
    regressor = TransformedTargetRegressor(regressor=rf, func=np.log1p, inverse_func=np.expm1)
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


rf_param_grid = [
    {
        "regressor__regressor__bootstrap": [True],
        "regressor__regressor__max_samples": [None, 0.8],
        "regressor__regressor__n_estimators": [100, 300],
        "regressor__regressor__max_depth": [None, 10],
        "regressor__regressor__max_features": [None, "sqrt"]
    },
    {
        "regressor__regressor__bootstrap": [False],
        "regressor__regressor__max_samples": [None],
        "regressor__regressor__n_estimators": [100, 300],
        "regressor__regressor__max_depth": [None, 10],
        "regressor__regressor__max_features": [None, "sqrt"]
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
        metrics_filename="performance_metrics_random_forest.csv",
        search_kwargs={"factor": 2}
    )
