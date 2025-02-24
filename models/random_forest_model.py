"""
Full Pipeline for Training and Predicting Player Transfer Prices using Random Forest.

This script:
  - Loads updated Parquet datasets.
  - Selects predictor features and the target ("Market Value"), dropping identifiers.
  - Splits data into training (60%), validation (20%), and test (20%) sets.
  - Builds a preprocessing pipeline.
  - Trains a Random Forest regressor (wrapped in TransformedTargetRegressor with log1p/expm1)
    and tunes its hyperparameters via GridSearchCV.
  - Evaluates the model using RMSE, R², MAE, MedAE, and MAPE.
  - Generates predictions concurrently for each updated file and saves them in the
    "../data/predictions/random_forest" folder.
"""

from logging_config import configure_logger

logging = configure_logger("random_forest_model", "random_forest_model.log")

import os
import glob
import time
from pathlib import Path
import joblib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from model_utils import (load_updated_data, select_features_and_target, build_preprocessor,
                         split_data, evaluate_model, predict_on_file)

# Global paths and constants
UPDATED_DIR = "../data/updated"
PREDICTIONS_DIR = "../data/predictions/random_forest"
Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)
DROP_COLS = {"player", "league", "season", "born", "country_code", "squad", "Market Value"}


def train_and_predict() -> None:
    """
    Trains the Random Forest model with hyperparameter tuning (via GridSearchCV)
    and then generates predictions concurrently for each updated file.
    """
    # Load data
    df = load_updated_data(UPDATED_DIR)
    X, y = select_features_and_target(df, drop_cols={"player", "league", "season", "born", "country_code", "squad"})
    # Build preprocessor
    preprocessor = build_preprocessor(X)
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_state=42)
    # Define the Random Forest model wrapped in a TransformedTargetRegressor (for log transformation)
    rf = RandomForestRegressor(random_state=42)
    regressor = TransformedTargetRegressor(
        regressor=rf,
        func=np.log1p,
        inverse_func=np.expm1
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    # Set up hyperparameter tuning via GridSearchCV
    param_grid = {
        "regressor__regressor__n_estimators": [100, 200],
        "regressor__regressor__max_depth": [None, 10, 20],
        "regressor__regressor__min_samples_split": [2, 5],
        "regressor__regressor__min_samples_leaf": [1, 2],
        "regressor__regressor__max_features": [None, "sqrt", "log2"]
    }
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )
    start_time = time.time()
    logging.info("Starting GridSearchCV for Random Forest regression...")
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start_time
    logging.info(f"GridSearchCV completed in {elapsed:.2f} seconds.")
    logging.info("Best hyperparameters found:")
    logging.info(grid_search.best_params_)
    best_model = grid_search.best_estimator_
    # Evaluate the model on validation and test sets
    logging.info("Evaluating Random Forest Model on Validation Set:")
    evaluate_model(best_model, X_val, y_val, dataset_name="Validation (RF)")
    logging.info("Evaluating Random Forest Model on Test Set:")
    evaluate_model(best_model, X_test, y_test, dataset_name="Test (RF)")
    # Save the final model
    model_path = "random_forest_model.pkl"
    joblib.dump(best_model, model_path)
    logging.info(f"Random Forest model saved to {model_path}")
    # Generate predictions concurrently for each updated file
    file_pattern = os.path.join(UPDATED_DIR, "updated_*.parquet")
    files = glob.glob(file_pattern)
    if not files:
        logging.error(f"No updated files found in {UPDATED_DIR}")
        return
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(predict_on_file, best_model, file_path, DROP_COLS, PREDICTIONS_DIR)
                   for file_path in files]
        for future in as_completed(futures):
            future.result()  # Raise any exceptions encountered
    logging.info("Prediction process completed for all files.")

def main() -> None:
    logging.info("Starting full training and prediction pipeline for Random Forest Model")
    train_and_predict()
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()