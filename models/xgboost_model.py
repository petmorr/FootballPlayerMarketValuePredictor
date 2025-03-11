"""
Full Pipeline for Training and Predicting Player Transfer Prices using XGBoost.

This script uses XGBoost's XGBRegressorGPU (imported from model_utils) with an expanded comprehensive
hyperparameter grid and RandomizedSearchCV to train and evaluate the model on three preprocessed variants.
GPU acceleration is enabled by setting tree_method='hist' and device='cuda'.
Performance metrics are saved to a CSV.
"""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from model_utils import run_training_pipeline, build_preprocessor, XGBRegressorGPU


def xgb_pipeline_builder(X_train) -> Pipeline:
    """
    Build the pipeline for XGBoost.

    Uses the common preprocessor and wraps our XGBRegressorGPU (defined in model_utils)
    in a TransformedTargetRegressor to apply a log1p transform to the target variable.

    GPU acceleration is enabled by setting tree_method='hist' and device='cuda'.
    """
    preprocessor = build_preprocessor(X_train)
    xgb_gpu = XGBRegressorGPU(
        random_state=42,
        tree_method='hist',
        device='cuda',
        verbosity=1,
        n_jobs=1  # Use a single thread to avoid oversubscription
    )
    regressor = TransformedTargetRegressor(
        regressor=xgb_gpu,
        func=np.log1p,
        inverse_func=np.expm1
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])

xgb_param_grid = {
    "regressor__regressor__n_estimators": [100, 200, 300, 500],
    "regressor__regressor__max_depth": [3, 5, 7, 10, 15],
    "regressor__regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "regressor__regressor__subsample": [0.5, 0.7, 1.0],
    "regressor__regressor__colsample_bytree": [0.5, 0.7, 1.0],
    "regressor__regressor__gamma": [0, 0.1, 0.5, 1],
    "regressor__regressor__min_child_weight": [1, 3, 5, 10],
    "regressor__regressor__reg_alpha": [0, 0.1, 1, 10],
    "regressor__regressor__reg_lambda": [1, 1.5, 2, 3]
}

if __name__ == "__main__":
    run_training_pipeline(
        model_name="XGBoost",
        pipeline_builder=xgb_pipeline_builder,
        search_class=RandomizedSearchCV,
        param_grid=xgb_param_grid,
        use_sample_weight=False,
        model_filename="xgboost_model",
        predictions_subdir="xgboost",
        metrics_filename="performance_metrics_xgboost.csv",
        search_kwargs={"n_iter": 200, "random_state": 42}
    )