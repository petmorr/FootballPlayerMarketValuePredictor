"""
Train and predict player market values using Ridge (as a form of linear regression).
Hyperparameters are tuned via GridSearchCV, applying log transform on targets.
"""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from model_utils import run_training_pipeline, build_preprocessor


def lr_pipeline_builder(X_train) -> Pipeline:
    """
    Create a pipeline with data preprocessing and Ridge regression
    wrapped in a log-transform for the target.
    """
    preprocessor = build_preprocessor(X_train)
    ridge = Ridge(random_state=42)
    regressor = TransformedTargetRegressor(
        regressor=ridge, func=np.log1p, inverse_func=np.expm1
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


# Hyperparameter grid
lr_param_grid = {
    "regressor__regressor__alpha": [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
    "regressor__regressor__solver": ["cholesky", "lsqr"],
    "regressor__regressor__tol": [1e-5, 1e-4, 1e-3, 1e-2],
    "regressor__regressor__max_iter": [None, 1000, 5000, 10000],
}

if __name__ == "__main__":
    run_training_pipeline(
        model_name="Linear Regression",
        pipeline_builder=lr_pipeline_builder,
        search_class=GridSearchCV,
        param_grid=lr_param_grid,
        use_sample_weight=True,
        model_filename="linear_regression_model",
        predictions_subdir="linear_regression",
        metrics_filename="performance_metrics_linear_regression.csv",
    )
