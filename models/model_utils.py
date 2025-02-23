"""
Common utilities for model training.
"""

import glob
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import logging

# Configure module-level logging.
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_updated_data(data_dir: str = "../data/updated") -> pd.DataFrame:
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

    drop_cols = {"player", "league", "season", "born", "country_code", "squad"}
    X = df.drop(columns=drop_cols.union({"Market Value"}), errors="ignore")
    y = df["Market Value"]

    # Drop rows with missing target values.
    mask = y.notnull()
    X, y = X[mask], y[mask]

    logger.info(f"Selected features shape: {X.shape} and target shape: {y.shape}")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that applies StandardScaler to numeric features
    and OneHotEncoder to categorical features.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info(f"Numeric features: {numeric_features}")
    logger.info(f"Categorical features: {categorical_features}")

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    # Use sparse_output=False (for scikit-learn versions >=1.2)
    categorical_transformer = Pipeline(steps=[
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
    Split the data into training (60%), validation (20%), and testing (20%) sets.
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    # From the remaining 80%, take 25% for validation => 0.25 * 0.8 = 20%
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=random_state
    )
    logger.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, dataset_name: str = "Dataset"
                   ) -> Tuple[float, float]:
    """
    Evaluate the model on a given dataset and log RMSE and R² scores.
    Returns the RMSE and R².
    """
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    logger.info(f"{dataset_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return rmse, r2
