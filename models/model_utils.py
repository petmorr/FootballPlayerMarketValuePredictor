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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------------------------------------------------------
# Data Loading and Preparation Functions
# -----------------------------------------------------------------------------
def load_updated_data(data_dir: str) -> pd.DataFrame:
    """
    Loads and concatenates all Parquet files from the specified folder.
    Use an updated folder (e.g., "../data/updated/enhanced_feature_engineering").
    """
    # Update file pattern to read updated_*.parquet files.
    file_pattern = os.path.join(data_dir, "updated_*.parquet")
    files = glob.glob(file_pattern)
    if not files:
        logging.error(f"No files found in {data_dir} matching pattern updated_*.parquet")
        raise FileNotFoundError(f"No files found in {data_dir} matching pattern updated_*.parquet")
    df_list = [pd.read_parquet(file) for file in files]
    combined_df = pd.concat(df_list, ignore_index=True)
    logging.info(f"Combined data shape: {combined_df.shape}")
    return combined_df

def compute_sample_weights(y: pd.Series, factor: float = 1.5) -> np.array:
    """
    Computes sample weights inversely proportional to the deviation from the median.
    Outliers (those far from the median) receive lower weight.
    """
    median = np.median(y)
    mad = np.median(np.abs(y - median))
    weights = 1 / (1 + np.abs(y - median) / (mad * factor + 1e-6))
    return weights

def select_features_and_target(df: pd.DataFrame, drop_cols: set = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Selects predictor features and the target variable.
    Assumes target variable is 'Market Value'. Additional columns (identifiers)
    are dropped to form the feature set.
    """
    if drop_cols is None:
        drop_cols = {"league", "season", "born", "country_code", "squad"}
    if "Market Value" not in df.columns:
        logging.error("The data does not contain a 'Market Value' column.")
        raise ValueError("The data does not contain a 'Market Value' column.")
    # Keep "player" for grouping, drop others.
    X = df.drop(columns=drop_cols.union({"Market Value"}), errors="ignore")
    y = df["Market Value"]
    mask = y.notnull()
    X, y = X[mask], y[mask]
    logging.info(f"Selected features shape: {X.shape} and target shape: {y.shape}")
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that imputes missing values,
    scales numeric features, and one-hot encodes categorical features.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "player" in numeric_features:
        numeric_features.remove("player")
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

def split_data(X: pd.DataFrame, y: pd.Series, groups: pd.Series, random_state: int = 42
               ) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits the data into training (60%), validation (20%), and test (20%) sets.
    Returns X_train, X_val, X_test, y_train, y_val, y_test, groups_train, groups_test.
    """
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train_val, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_val, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train_val, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

    gss_val = GroupShuffleSplit(test_size=0.25, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups_train_val))
    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
    groups_train = groups_train_val.iloc[train_idx]
    logging.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, groups_train, groups_test

def evaluate_model(model, X: pd.DataFrame, y: pd.Series, dataset_name: str = "Dataset"
                   ) -> Tuple[float, float, float, float, float]:
    """
    Evaluates the model on the provided dataset using RMSE, R², MAE, MedAE, and MAPE.
    """
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    medae = median_absolute_error(y, predictions)
    mape = mean_absolute_percentage_error(y, predictions) * 100
    logging.info(
        f"{dataset_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}, MedAE: {medae:.4f}, MAPE: {mape:.2f}%")
    return rmse, r2, mae, medae, mape

def predict_on_file(model, file_path: str, drop_cols: set, predictions_dir: str) -> None:
    """
    Generates predictions for a single file:
      - Loads the file.
      - Prepares features by dropping specified columns.
      - Rounds predictions to two decimals.
      - Saves the file with an added "predicted_market_value" column.
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