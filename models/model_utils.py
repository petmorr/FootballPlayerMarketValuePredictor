"""
Common utilities for model training and prediction.
"""

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

from logging_config import configure_logger

logger = configure_logger("model_utils", "model_utils.log")


def load_updated_data(data_dir: str) -> pd.DataFrame:
    """
    Load and concatenate all Parquet files from the specified folder.

    Args:
        data_dir (str): Directory containing updated data files.

    Returns:
        pd.DataFrame: Combined DataFrame of updated data.

    Raises:
        FileNotFoundError: If no matching files are found.
    """
    file_pattern = os.path.join(data_dir, "updated_*.parquet")
    files = glob.glob(file_pattern)
    if not files:
        logger.error(f"No files found in {data_dir} matching pattern updated_*.parquet")
        raise FileNotFoundError(f"No files found in {data_dir} matching pattern updated_*.parquet")
    df_list = [pd.read_parquet(file) for file in files]
    combined_df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Combined data shape: {combined_df.shape}")
    return combined_df


def compute_sample_weights(y: pd.Series, factor: float = 1.5) -> np.array:
    """
    Compute sample weights inversely proportional to the deviation from the median.

    Outliers receive lower weight.

    Args:
        y (pd.Series): Target variable.
        factor (float, optional): Scaling factor. Defaults to 1.5.

    Returns:
        np.array: Computed sample weights.
    """
    median = np.median(y)
    mad = np.median(np.abs(y - median))
    weights = 1 / (1 + np.abs(y - median) / (mad * factor + 1e-6))
    return weights


def select_features_and_target(df: pd.DataFrame, drop_cols: set = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select predictor features and the target variable ('Market Value').

    Args:
        df (pd.DataFrame): Input DataFrame.
        drop_cols (set, optional): Columns to drop from the feature set.
                                   Defaults to {"league", "season", "born", "country_code", "squad"}.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature set X and target variable y.

    Raises:
        ValueError: If 'Market Value' column is missing.
    """
    if drop_cols is None:
        drop_cols = {"league", "season", "born", "country_code", "squad"}
    if "Market Value" not in df.columns:
        logger.error("The data does not contain a 'Market Value' column.")
        raise ValueError("The data does not contain a 'Market Value' column.")
    # Keep "player" column for grouping.
    X = df.drop(columns=drop_cols.union({"Market Value"}), errors="ignore")
    y = df["Market Value"]
    mask = y.notnull()
    X, y = X[mask], y[mask]
    logger.info(f"Selected features shape: {X.shape} and target shape: {y.shape}")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer for preprocessing numeric and categorical features.

    Args:
        X (pd.DataFrame): Input feature DataFrame.

    Returns:
        ColumnTransformer: Preprocessing transformer.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "player" in numeric_features:
        numeric_features.remove("player")
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info(f"Numeric features: {numeric_features}")
    logger.info(f"Categorical features: {categorical_features}")

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
    Split the data into training (60%), validation (20%), and test (20%) sets.

    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        groups (pd.Series): Grouping variable (e.g., 'player').
        random_state (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple: X_train, X_val, X_test, y_train, y_val, y_test, groups_train, groups_test.
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
    logger.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, groups_train, groups_test


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, dataset_name: str = "Dataset"
                   ) -> Tuple[float, float, float, float, float]:
    """
    Evaluate the model using various regression metrics.

    Args:
        model: Trained model.
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        dataset_name (str, optional): Label for the dataset. Defaults to "Dataset".

    Returns:
        Tuple[float, float, float, float, float]: RMSE, R², MAE, MedAE, MAPE.
    """
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    medae = median_absolute_error(y, predictions)
    mape = mean_absolute_percentage_error(y, predictions) * 100
    logger.info(
        f"{dataset_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}, MedAE: {medae:.4f}, MAPE: {mape:.2f}%")
    return rmse, r2, mae, medae, mape


def predict_on_file(model, file_path: str, drop_cols: set, predictions_dir: str) -> None:
    """
    Generate predictions for a single file and save the results.

    Args:
        model: Trained model.
        file_path (str): Path to the input Parquet file.
        drop_cols (set): Columns to drop from the input data.
        predictions_dir (str): Directory where predictions will be saved.
    """
    logger.info(f"Processing file: {file_path}")
    df = pd.read_parquet(file_path)
    df_pred = df.copy()
    X_pred = df.drop(columns=drop_cols, errors="ignore")
    predictions = model.predict(X_pred)
    predictions = np.round(predictions, 2)
    df_pred["predicted_market_value"] = predictions
    base_name = Path(file_path).name
    output_file = Path(predictions_dir) / f"predicted_{base_name}"
    df_pred.to_parquet(output_file, index=False)
    logger.info(f"Saved predictions to: {output_file}")
