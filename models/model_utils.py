"""
Common utilities for model training and prediction.
"""

from logging_config import configure_logger

logging = configure_logger("model_utils", "model_utils.log")

import glob
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, median_absolute_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------------------------------------------
# Data Loading and Preparation Functions
# -----------------------------------------------------------------------------
def load_updated_data(data_dir: str = "../data/updated") -> pd.DataFrame:
    """
    Loads and concatenates all Parquet files from the specified directory.
    Assumes file names match the pattern 'updated_*.parquet'.
    """
    file_pattern = os.path.join(data_dir, "updated_*.parquet")
    files = glob.glob(file_pattern)
    if not files:
        logging.error(f"No files found in {data_dir} matching pattern updated_*.parquet")
        raise FileNotFoundError(f"No files found in {data_dir} matching pattern updated_*.parquet")
    df_list = [pd.read_parquet(file) for file in files]
    combined_df = pd.concat(df_list, ignore_index=True)
    logging.info(f"Combined data shape: {combined_df.shape}")
    return combined_df


def select_features_and_target(df: pd.DataFrame, drop_cols: set = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Selects predictor features and the target variable.
    Assumes target variable is 'Market Value'. Additional columns (identifiers)
    are dropped to form the feature set.
    """
    if drop_cols is None:
        drop_cols = {"player", "league", "season", "born", "country_code", "squad"}
    if "Market Value" not in df.columns:
        logging.error("The data does not contain a 'Market Value' column.")
        raise ValueError("The data does not contain a 'Market Value' column.")
    X = df.drop(columns=drop_cols.union({"Market Value"}), errors="ignore")
    y = df["Market Value"]
    # Remove rows with missing target values
    mask = y.notnull()
    X, y = X[mask], y[mask]
    logging.info(f"Selected features shape: {X.shape} and target shape: {y.shape}")
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that:
      - Imputes missing numeric values with the median.
      - Scales numeric features.
      - Imputes missing categorical values with a constant ("missing").
      - One-hot encodes categorical features.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    logging.info(f"Numeric features: {numeric_features}")
    logging.info(f"Categorical features: {categorical_features}")
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    return preprocessor

def split_data(X: pd.DataFrame, y: pd.Series, random_state: int = 42
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits the data into training (60%), validation (20%), and test (20%) sets.
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    # Further split the remaining 80%: 25% of 80% (i.e. 20%) for validation.
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25,
                                                      random_state=random_state)
    logging.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model(model, X: pd.DataFrame, y: pd.Series, dataset_name: str = "Dataset"
                   ) -> Tuple[float, float, float, float, float]:
    """
    Evaluates the model on the provided dataset using multiple metrics:
      - RMSE, R², MAE, Median AE, and MAPE.
    Logs and returns the metrics.
    """
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    medae = median_absolute_error(y, predictions)
    mape = mean_absolute_percentage_error(y, predictions) * 100  # expressed as a percentage
    logging.info(f"{dataset_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}, "
                 f"MedAE: {medae:.4f}, MAPE: {mape:.2f}%")
    return rmse, r2, mae, medae, mape


def predict_on_file(model, file_path: str, drop_cols: set, predictions_dir: str) -> None:
    """
    Generates predictions for a single updated file:
      - Loads the file.
      - Prepares the features (using the same dropped columns as in training).
      - Rounds predictions to two decimal places.
      - Saves the DataFrame with an added "predicted_market_value" column.
    """
    logging.info(f"Processing file: {file_path}")
    df = pd.read_parquet(file_path)
    df_pred = df.copy()
    X_pred = df.drop(columns=drop_cols, errors="ignore")
    predictions = model.predict(X_pred)
    predictions = np.round(predictions, 2)
    df_pred["predicted_market_value"] = predictions
    base_name = Path(file_path).name
    output_file = Path(predictions_dir) / f"predicted_{base_name}"
    df_pred.to_parquet(output_file, index=False)
    logging.info(f"Saved predictions to: {output_file}")
