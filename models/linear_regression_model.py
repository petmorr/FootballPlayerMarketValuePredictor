#!/usr/bin/env python3
"""
Full Pipeline for Training and Predicting Player Transfer Prices using Linear Regression.

This script:
  - Loads and combines updated Parquet datasets.
  - Selects features and the target ("Market Value").
  - Splits the data into training, validation, and test sets.
  - Trains a Linear Regression model wrapped in a TransformedTargetRegressor (using log1p/expm1)
    so that all predictions are positive.
  - Evaluates the model on validation and test sets.
  - Iterates over each updated file to predict player prices,
    rounds predictions to two decimal places, and saves the final dataset
    (with a new column "predicted_market_value") to a dedicated predictions folder.

Make sure your updated Parquet files follow the naming convention "updated_*.parquet" and
are located in "../data/updated". The predicted files are saved in "../data/predictions/linear_regression".
"""

import glob
import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------
# Paths and Global Constants
# -------------------------------
UPDATED_DIR = "../data/updated"
PREDICTIONS_DIR = "../data/predictions/linear_regression"
# Ensure predictions folder exists
Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)

# Columns to drop when forming features (identifiers and target)
DROP_COLS = {"player", "league", "season", "born", "country_code", "squad", "Market Value"}


# -------------------------------
# Utility Functions
# -------------------------------

def load_updated_data(data_dir: str = UPDATED_DIR) -> pd.DataFrame:
    """
    Load and concatenate all Parquet files from the updated directory.
    Assumes files follow the naming convention 'updated_*.parquet'.
    """
    file_pattern = os.path.join(data_dir, "updated_*.parquet")
    files = glob.glob(file_pattern)
    if not files:
        logger.error(f"No files found in {data_dir} matching pattern updated_*.parquet")
        raise FileNotFoundError(f"No files found in {data_dir} matching pattern updated_*.parquet")

    df_list = []
    for file in files:
        logger.info(f"Loading {file}")
        df = pd.read_parquet(file)
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Combined data shape: {combined_df.shape}")
    return combined_df


def select_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select predictor features and the target variable.
    Assumes target variable is 'Market Value'.
    Drops identifiers and metadata that are not useful for prediction.
    """
    if "Market Value" not in df.columns:
        logger.error("The data does not contain a 'Market Value' column.")
        raise ValueError("The data does not contain a 'Market Value' column.")

    X = df.drop(columns=DROP_COLS, errors="ignore")
    y = df["Market Value"]

    # Drop rows with missing target values.
    mask = y.notnull()
    X, y = X[mask], y[mask]

    logger.info(f"Selected features shape: {X.shape} and target shape: {y.shape}")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that applies StandardScaler to numeric features
    and OneHotEncoder (with sparse_output=False) to categorical features.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info(f"Numeric features: {numeric_features}")
    logger.info(f"Categorical features: {categorical_features}")

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    return preprocessor


def split_data(X: pd.DataFrame, y: pd.Series, random_state: int = 42
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split the data into training (60%), validation (20%), and testing (20%) sets.
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    # 25% of 80% gives 20% for validation.
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25,
                                                      random_state=random_state)
    logger.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, dataset_name: str = "Dataset") -> Tuple[float, float]:
    """
    Evaluate the model on a given dataset and log RMSE and R² scores.
    Returns the RMSE and R².
    """
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    logger.info(f"{dataset_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return rmse, r2


def predict_on_file(model, file_path: str) -> None:
    """
    Load a single updated file, generate predictions, round them to two decimals,
    attach the predictions to the original data, and save the result.
    """
    logger.info(f"Processing file: {file_path}")
    df = pd.read_parquet(file_path)
    df_pred = df.copy()

    # Prepare features (drop the same columns as used in training)
    X_pred = df.drop(columns=DROP_COLS, errors="ignore")
    predictions = model.predict(X_pred)
    # Round predictions to 2 decimal places (currency format)
    predictions = np.round(predictions, 2)
    df_pred["predicted_market_value"] = predictions

    base_name = Path(file_path).name
    output_file = Path(PREDICTIONS_DIR) / f"predicted_{base_name}"
    df_pred.to_parquet(output_file, index=False)
    logger.info(f"Saved predictions to: {output_file}")


# -------------------------------
# Training and Prediction Pipeline
# -------------------------------

def train_and_predict() -> None:
    """
    Train the Linear Regression model (with a log transformation of the target)
    and apply it to each updated file to generate final predictions.
    """
    # Load and combine updated data for training
    df = load_updated_data()
    X, y = select_features_and_target(df)
    preprocessor = build_preprocessor(X)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_state=42)

    # Create a pipeline that transforms the target using log1p/expm1 to ensure positive predictions.
    regressor = TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=np.log1p,
        inverse_func=np.expm1
    )

    pipeline_lr = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    # Train the model on the training set.
    pipeline_lr.fit(X_train, y_train)

    # Evaluate the model on validation and test sets.
    logger.info("Evaluating Linear Regression Model on Validation Set")
    evaluate_model(pipeline_lr, X_val, y_val, dataset_name="Validation (LR)")

    logger.info("Evaluating Linear Regression Model on Test Set")
    evaluate_model(pipeline_lr, X_test, y_test, dataset_name="Test (LR)")

    # Optionally, save the trained model for later use.
    model_path = "linear_regression_model.pkl"
    joblib.dump(pipeline_lr, model_path)
    logger.info(f"Linear Regression model saved to {model_path}")

    # Now, run predictions on each updated file.
    file_pattern = os.path.join(UPDATED_DIR, "updated_*.parquet")
    files = glob.glob(file_pattern)
    if not files:
        logger.error(f"No updated files found in {UPDATED_DIR}")
        return

    for file_path in files:
        predict_on_file(pipeline_lr, file_path)

    logger.info("Prediction process completed for all files.")


# -------------------------------
# Main Entry Point
# -------------------------------

def main() -> None:
    logger.info("Starting full training and prediction pipeline for Linear Regression Model")
    train_and_predict()
    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
