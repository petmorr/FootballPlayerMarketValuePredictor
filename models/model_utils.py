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


def aggregate_player_records(df: pd.DataFrame, weight_col: str = "minutes_played") -> pd.DataFrame:
    """
    Aggregates duplicate records for the same player (and season, if available)
    using minutes_played as the weight for numeric features and concatenates
    squad names for categorical features.
    """
    group_cols = ["player"]
    if "season" in df.columns:
        group_cols.append("season")

    def agg_func(group: pd.DataFrame) -> pd.Series:
        result = {}
        for col in group.columns:
            if col in group_cols:
                result[col] = group.iloc[0][col]
            elif pd.api.types.is_numeric_dtype(group[col]):
                if col == weight_col:
                    result[col] = group[col].sum()
                else:
                    weights = group[weight_col]
                    if weights.sum() > 0:
                        result[col] = np.average(group[col], weights=weights)
                    else:
                        result[col] = group[col].mean()
            else:
                # For 'squad', concatenate unique team names; otherwise use mode.
                if col == "squad":
                    teams = group[col].unique()
                    result[col] = "/".join(sorted(teams))
                else:
                    result[col] = group[col].mode().iloc[0] if not group[col].mode().empty else group[col].iloc[0]
        return pd.Series(result)

    aggregated_df = df.groupby(group_cols, as_index=False).apply(agg_func)
    logging.info(f"Aggregated data shape (after combining duplicates): {aggregated_df.shape}")
    return aggregated_df


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
    # Keep "player" for group-based cross validation; drop others.
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
    # Exclude 'player' from preprocessing if present.
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
    Note: groups are used for cross validation to ensure that records for the same player
    are not split across folds.
    """
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train_val, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_val, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train_val, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

    # Further split the training+validation set for validation
    gss_val = GroupShuffleSplit(test_size=0.25, random_state=random_state)  # 25% of 80% = 20%
    train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups_train_val))
    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
    groups_train = groups_train_val.iloc[train_idx]
    logging.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, groups_train, groups_test

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
    mape = mean_absolute_percentage_error(y, predictions) * 100  # as a percentage
    logging.info(f"{dataset_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}, "
                 f"MedAE: {medae:.4f}, MAPE: {mape:.2f}%")
    return rmse, r2, mae, medae, mape

def predict_on_file(model, file_path: str, drop_cols: set, predictions_dir: str) -> None:
    """
    Generates predictions for a single updated file:
      - Loads the file.
      - Prepares the features (dropping the same columns as in training).
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