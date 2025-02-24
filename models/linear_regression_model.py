"""
Full Pipeline for Training and Predicting Player Transfer Prices using an Improved Linear Model.

This script:
  - Loads updated Parquet datasets.
  - Selects features and the target ("Market Value"), dropping identifiers.
  - Splits the data into training, validation, and test sets.
  - Builds a preprocessing pipeline.
  - Trains a Ridge regression model (wrapped in TransformedTargetRegressor with log1p/expm1)
    using GridSearchCV for hyperparameter tuning.
  - Evaluates the model using RMSE, R², MAE, MedAE, and MAPE.
  - Generates predictions concurrently for each updated file and saves them in the 
    "../data/predictions/linear_regression" folder.
"""

from logging_config import configure_logger

logging = configure_logger("linear_regression_model", "linear_regression_model.log")

import os
import glob
import time
from pathlib import Path
import joblib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from model_utils import (load_updated_data, select_features_and_target, build_preprocessor,
                         split_data, evaluate_model, predict_on_file)

# Global paths and constants
UPDATED_DIR = "../data/updated"
PREDICTIONS_DIR = "../data/predictions/linear_regression"
Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)
DROP_COLS = {"player", "league", "season", "born", "country_code", "squad", "Market Value"}

def train_and_predict() -> None:
    """
    Trains the improved linear model using Ridge regression (with target log transformation)
    and generates predictions concurrently across updated files.
    """
    # Load and prepare data
    df = load_updated_data(UPDATED_DIR)
    X, y = select_features_and_target(df, drop_cols={"player", "league", "season", "born", "country_code", "squad"})
    preprocessor = build_preprocessor(X)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_state=42)

    # Define Ridge model with target transformation (log1p/expm1)
    ridge = Ridge(random_state=42)
    regressor = TransformedTargetRegressor(
        regressor=ridge,
        func=np.log1p,
        inverse_func=np.expm1
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    # Hyperparameter tuning using GridSearchCV
    param_grid = {"regressor__regressor__alpha": [0.1, 1.0, 10.0, 100.0]}
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )
    start_time = time.time()
    logging.info("Starting GridSearchCV for Improved Linear Model...")
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start_time
    logging.info(f"GridSearchCV completed in {elapsed:.2f} seconds.")
    logging.info("Best hyperparameters found:")
    logging.info(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Evaluate the model on validation and test sets
    logging.info("Evaluating Improved Linear Model on Validation Set:")
    evaluate_model(best_model, X_val, y_val, dataset_name="Validation (Improved LR)")
    logging.info("Evaluating Improved Linear Model on Test Set:")
    evaluate_model(best_model, X_test, y_test, dataset_name="Test (Improved LR)")

    # Save the trained model
    model_path = "linear_regression_model.pkl"
    joblib.dump(best_model, model_path)
    logging.info(f"Improved Linear Regression model saved to {model_path}")

    # Generate predictions concurrently for each updated file
    file_pattern = os.path.join(UPDATED_DIR, "updated_*.parquet")
    files = glob.glob(file_pattern)
    if not files:
        logging.error(f"No updated files found in {UPDATED_DIR}")
        return

    # Use ThreadPoolExecutor to run predictions concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(predict_on_file, best_model, file_path, DROP_COLS, PREDICTIONS_DIR)
                   for file_path in files]
        for future in as_completed(futures):
            # This will raise any exceptions encountered during prediction.
            future.result()
    logging.info("Prediction process completed for all files.")

def main() -> None:
    logging.info("Starting full training and prediction pipeline for Improved Linear Regression Model")
    train_and_predict()
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()