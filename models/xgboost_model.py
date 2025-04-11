"""
Train and predict player market values using GPU-accelerated XGBoost.
Hyperparameters are tuned via HalvingGridSearchCV, applying log transform on targets.
"""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import Pipeline

from model_utils import run_training_pipeline, build_preprocessor, XGBRegressorGPU


def xgb_pipeline_builder(X_train) -> Pipeline:
    """
    Create a pipeline with data preprocessing and a GPU-accelerated XGBRegressor,
    wrapped in a log-transform for the target.
    """
    preprocessor = build_preprocessor(X_train)
    xgb_gpu = XGBRegressorGPU(
        random_state=42, tree_method="hist", device="cuda", verbosity=1, n_jobs=-1
    )
    regressor = TransformedTargetRegressor(
        regressor=xgb_gpu, func=np.log1p, inverse_func=np.expm1
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


# Hyperparameter grid
xgb_param_grid = {
    "regressor__regressor__n_estimators": [100, 300, 500],
    "regressor__regressor__max_depth": [3, 7, 15],
    "regressor__regressor__learning_rate": [0.05, 0.1, 0.2],
    "regressor__regressor__subsample": [0.7, 1.0],
    "regressor__regressor__colsample_bytree": [0.7, 1.0],
    "regressor__regressor__gamma": [0, 0.5, 1],
    "regressor__regressor__min_child_weight": [1, 5],
    "regressor__regressor__reg_alpha": [0, 1],
    "regressor__regressor__reg_lambda": [1, 2],
}

if __name__ == "__main__":
    run_training_pipeline(
        model_name="XGBoost",
        pipeline_builder=xgb_pipeline_builder,
        search_class=HalvingGridSearchCV,
        param_grid=xgb_param_grid,
        use_sample_weight=False,
        model_filename="xgboost_model",
        predictions_subdir="xgboost",
        metrics_filename="performance_metrics_xgboost.csv",
    )