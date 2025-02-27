"""
Full Pipeline for Training and Predicting Player Transfer Prices using an Improved Linear Model.

This script trains and evaluates the model on three preprocessed variants:
  - enhanced_feature_engineering
  - feature_engineering
  - no_feature_engineering

Performance metrics are saved to a CSV for comparison.
"""

import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline

from logging_config import configure_logger
from model_utils import (
    load_updated_data,
    compute_sample_weights,
    select_features_and_target,
    build_preprocessor,
    split_data,
    evaluate_model,
    predict_on_file
)

# Configure logger.
logger = configure_logger("linear_regression_model", "linear_regression_model.log")

# Define updated data folder paths for each preprocessing variant.
PREPROC_VARIANTS = {
    "enhanced_feature_engineering": "../data/updated/enhanced_feature_engineering",
    "feature_engineering": "../data/updated/feature_engineering",
    "no_feature_engineering": "../data/updated/no_feature_engineering"
}

# Create a results folder relative to this script.
RESULTS_FOLDER = Path(__file__).parent / "results"
RESULTS_FOLDER.mkdir(exist_ok=True)


def train_and_predict() -> None:
    """
    Train the Improved Linear Regression model (Ridge) and generate predictions for each data variant.

    This function:
      - Loads updated data.
      - Splits data into training, validation, and test sets.
      - Builds a processing pipeline with grid search.
      - Evaluates the model.
      - Saves the best model and performance metrics.
      - Generates predictions concurrently.
    """
    performance_records = []  # List to store performance metrics.
    drop_cols = {"league", "season", "born", "country_code", "squad", "rank", "position"}

    for variant_name, variant_folder in PREPROC_VARIANTS.items():
        logger.info(f"Processing variant: {variant_name} from folder: {variant_folder}")

        # Load the updated data.
        df = load_updated_data(variant_folder)

        # Select features and target.
        X_all, y_all = select_features_and_target(df, drop_cols=drop_cols)
        groups_all = X_all["player"]
        X_all = X_all.drop(columns=["player"], errors="ignore")

        # Split data in a group-aware manner.
        X_train, X_val, X_test, y_train, y_val, y_test, groups_train, _ = split_data(
            X_all, y_all, groups_all, random_state=42
        )
        sample_weights = compute_sample_weights(y_train)

        # Build the pipeline.
        preprocessor = build_preprocessor(X_train)
        ridge = Ridge(random_state=42)
        regressor = TransformedTargetRegressor(
            regressor=ridge, func=np.log1p, inverse_func=np.expm1
        )
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])

        # Configure GridSearchCV using GroupKFold.
        gkf = GroupKFold(n_splits=5)
        param_grid = {
            "regressor__regressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
            "regressor__regressor__solver": ["cholesky", "lsqr"],
            "regressor__regressor__tol": [1e-5, 1e-4, 1e-3, 1e-2],
            "regressor__regressor__max_iter": [None, 1000, 5000, 10000]
        }
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=gkf.split(X_train, y_train, groups_train),
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )

        start_time = time.time()
        logger.info(f"Starting GridSearchCV for variant: {variant_name}")
        grid_search.fit(X_train, y_train, regressor__sample_weight=sample_weights)
        elapsed = time.time() - start_time
        logger.info(f"GridSearchCV for variant {variant_name} completed in {elapsed:.2f} seconds.")
        logger.info(f"Best hyperparameters for variant {variant_name}: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_

        # Evaluate the model.
        val_metrics = evaluate_model(best_model, X_val, y_val, dataset_name=f"Validation ({variant_name} - LR)")
        test_metrics = evaluate_model(best_model, X_test, y_test, dataset_name=f"Test ({variant_name} - LR)")

        performance_records.append({
            "variant": variant_name,
            "dataset": "Validation",
            "model": "Linear Regression",
            "rmse": val_metrics[0],
            "r2": val_metrics[1],
            "mae": val_metrics[2],
            "medae": val_metrics[3],
            "mape": val_metrics[4],
            "training_time": elapsed,
            "best_params": str(grid_search.best_params_)
        })
        performance_records.append({
            "variant": variant_name,
            "dataset": "Test",
            "model": "Linear Regression",
            "rmse": test_metrics[0],
            "r2": test_metrics[1],
            "mae": test_metrics[2],
            "medae": test_metrics[3],
            "mape": test_metrics[4],
            "training_time": elapsed,
            "best_params": str(grid_search.best_params_)
        })

        # Save the trained model.
        model_path = RESULTS_FOLDER / f"linear_regression_model_{variant_name}.pkl"
        joblib.dump(best_model, model_path)
        logger.info(f"Linear Regression model for variant {variant_name} saved to {model_path}")

        # Generate predictions.
        pred_drop_cols = {"player", "league", "season", "born", "country_code", "squad", "rank", "position", "Market Value"}
        predictions_dir = Path("../data/predictions/linear_regression") / variant_name
        predictions_dir.mkdir(parents=True, exist_ok=True)
        file_pattern = os.path.join(variant_folder, "updated_*.parquet")
        files = glob.glob(file_pattern)
        if not files:
            logger.error(f"No files found in {variant_folder}")
        else:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(predict_on_file, best_model, file_path, pred_drop_cols, predictions_dir)
                    for file_path in files
                ]
                for future in as_completed(futures):
                    future.result()
            logger.info(f"Prediction process completed for variant {variant_name}.")

    # Save performance metrics to CSV.
    csv_path = RESULTS_FOLDER / "performance_metrics_linear_regression.csv"
    pd.DataFrame(performance_records).to_csv(csv_path, index=False)
    logger.info(f"Performance metrics saved to {csv_path}")


def main() -> None:
    """Main function to execute the training and prediction pipeline for the linear regression model."""
    logger.info("Starting full training and prediction pipeline for Improved Linear Regression Model")
    train_and_predict()
    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()