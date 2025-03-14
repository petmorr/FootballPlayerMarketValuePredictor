"""
model_utils.py

Common utilities for model training and prediction.

This module includes functions for loading data, computing sample weights,
selecting features and targets, building preprocessors, splitting data,
evaluating models, generating predictions, and supporting GPU-accelerated XGBoost.
"""

import glob
import os
import time
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict, Any

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
    Load and concatenate all Parquet files from the specified directory.

    Args:
        data_dir (str): Directory containing updated Parquet files.

    Returns:
        pd.DataFrame: Combined DataFrame from all files.

    Raises:
        FileNotFoundError: If no files matching the pattern are found.
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
    Compute sample weights inversely proportional to deviation from the median.

    Args:
        y (pd.Series): Target values.
        factor (float): Scaling factor for the median absolute deviation.

    Returns:
        np.array: Computed sample weights.
    """
    median = np.median(y)
    mad = np.median(np.abs(y - median))
    weights = 1 / (1 + np.abs(y - median) / (mad * factor + 1e-6))
    return weights


def select_features_and_target(df: pd.DataFrame, drop_cols: Optional[set] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select predictor features and target ('Market Value') from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        drop_cols (Optional[set]): Set of columns to drop from predictors.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features DataFrame and target Series.

    Raises:
        ValueError: If 'Market Value' column is missing.
    """
    if drop_cols is None:
        drop_cols = {"league", "season", "born", "country_code", "squad"}
    if "Market Value" not in df.columns:
        logger.error("The data does not contain a 'Market Value' column.")
        raise ValueError("The data does not contain a 'Market Value' column.")
    X = df.drop(columns=drop_cols.union({"Market Value"}), errors="ignore")
    y = df["Market Value"]
    mask = y.notnull()
    X, y = X[mask], y[mask]
    logger.info(f"Selected features shape: {X.shape} and target shape: {y.shape}")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer to preprocess numeric and categorical features.

    Args:
        X (pd.DataFrame): The features DataFrame.

    Returns:
        ColumnTransformer: The configured preprocessor.
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
    Split data into training (60%), validation (20%), and test (20%) sets using group-aware splitting.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target values.
        groups (pd.Series): Groups for splitting.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple containing training, validation, and test splits for X, y, and groups.
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
    Evaluate a regression model using various metrics.

    Args:
        model: The trained model.
        X (pd.DataFrame): Features to predict on.
        y (pd.Series): True target values.
        dataset_name (str): Label for the dataset (used for logging).

    Returns:
        Tuple[float, float, float, float, float]: RMSE, R², MAE, MedAE, and MAPE.
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


def predict_on_file(model, file_path: str, drop_cols: set, predictions_dir: Union[str, Path]) -> None:
    """
    Generate predictions for a single Parquet file and save the results.

    Args:
        model: The trained regression model.
        file_path (str): Path to the input Parquet file.
        drop_cols (set): Set of columns to drop from predictors.
        predictions_dir (Union[str, Path]): Directory to save predictions.
    """
    logger.info(f"Processing file for prediction: {file_path}")
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


# --- XGBoost GPU Support ---
import cupy as cp
from xgboost import XGBRegressor, DMatrix

class XGBRegressorGPU:
    """
    GPU-accelerated wrapper for XGBRegressor.

    This wrapper ensures that input data is converted to GPU arrays if available.
    """

    def __init__(self, **kwargs):
        # Enable GPU acceleration by setting tree_method to 'gpu_hist' if not provided.
        if cp is not None and 'tree_method' not in kwargs:
            kwargs['tree_method'] = 'gpu_hist'
        self._model = XGBRegressor(**kwargs)

    def fit(self, X, y, **kwargs):
        """
        Fit the XGBRegressor model.

        Args:
            X: Feature matrix.
            y: Target vector.
            **kwargs: Additional parameters.

        Returns:
            Fitted model.
        """
        if cp is not None:
            X = cp.asarray(X)
            y = cp.asarray(y)
        return self._model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        """
        Predict using the XGBRegressor model.

        Args:
            X: Feature matrix.
            **kwargs: Additional prediction parameters.

        Returns:
            np.array: Predictions.
        """
        X_np = np.asarray(X)
        if cp is not None:
            X_gpu = cp.asarray(X_np)
            dmat = DMatrix(X_gpu)
        else:
            dmat = DMatrix(X_np)
        booster = self._model.get_booster()
        y_pred = booster.predict(dmat, **kwargs)
        return cp.asnumpy(y_pred) if cp is not None else y_pred

    def get_params(self, deep=True):
        """
        Get model parameters.

        Args:
            deep (bool): Whether to include nested objects.

        Returns:
            dict: Model parameters.
        """
        return self._model.get_params(deep=deep)

    def set_params(self, **params):
        """
        Set model parameters.

        Args:
            **params: Parameters to set.

        Returns:
            self
        """
        self._model.set_params(**params)
        return self

    def get_booster(self):
        """
        Get the underlying XGBoost booster.

        Returns:
            Booster: The booster object.
        """
        return self._model.get_booster()
# --- End XGBoost GPU Support ---


def process_variant(variant_name: str,
                    variant_folder: str,
                    model_name: str,
                    pipeline_builder,
                    search_class,
                    param_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]]],
                    use_sample_weight: bool,
                    model_filename: str,
                    predictions_subdir: str,
                    search_kwargs: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Process a single feature-engineering variant by training, evaluating, and generating predictions.

    Args:
        variant_name (str): Name of the variant.
        variant_folder (str): Folder path containing updated data for the variant.
        model_name (str): Name of the model.
        pipeline_builder: Callable that builds a pipeline from training data.
        search_class: Hyperparameter search class.
        param_grid (Union[Dict, List[Dict]]): Hyperparameter grid.
        use_sample_weight (bool): Whether to use sample weights.
        model_filename (str): Filename prefix for saving the model.
        predictions_subdir (str): Subdirectory for saving predictions.
        search_kwargs (Optional[Dict[str, Any]]): Additional search parameters.

    Returns:
        List[Dict[str, Any]]: Performance records for the variant.
    """
    import joblib
    drop_cols = {"league", "season", "born", "country_code", "squad", "rank", "position"}
    logger.info(f"Processing variant: {variant_name} from folder: {variant_folder}")
    df = load_updated_data(variant_folder)
    X_all, y_all = select_features_and_target(df, drop_cols=drop_cols)
    groups_all = X_all["player"]
    X_all = X_all.drop(columns=["player"], errors="ignore")
    X_train, X_val, X_test, y_train, y_val, y_test, groups_train, _ = split_data(X_all, y_all, groups_all,
                                                                                 random_state=42)
    fit_params = {}
    if use_sample_weight:
        sample_weights = compute_sample_weights(y_train)
        fit_params = {"regressor__sample_weight": sample_weights}
    pipeline = pipeline_builder(X_train)
    from sklearn.model_selection import GroupKFold
    cv = GroupKFold(n_splits=5)
    if search_kwargs is None:
        search_kwargs = {}
    # Support both RandomizedSearchCV and others.
    if search_class.__name__ == "RandomizedSearchCV":
        search_obj = search_class(
            estimator=pipeline,
            param_distributions=param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            verbose=1,
            n_jobs=-1,
            **search_kwargs
        )
    else:
        search_obj = search_class(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            verbose=1,
            n_jobs=-1,
            **search_kwargs
        )
    logger.info(f"Starting hyperparameter search for {model_name} on variant: {variant_name}")
    start = time.time()
    search_obj.fit(X_train, y_train, groups=groups_train, **fit_params)
    elapsed = time.time() - start
    logger.info(f"Hyperparameter search for variant {variant_name} completed in {elapsed:.2f} seconds.")
    logger.info(f"Best hyperparameters for variant {variant_name}: {search_obj.best_params_}")
    best_model = search_obj.best_estimator_
    val_metrics = evaluate_model(best_model, X_val, y_val, dataset_name=f"Validation ({variant_name} - {model_name})")
    test_metrics = evaluate_model(best_model, X_test, y_test, dataset_name=f"Test ({variant_name} - {model_name})")
    performance = [
        {
            "variant": variant_name,
            "dataset": "Validation",
            "model": model_name,
            "rmse": val_metrics[0],
            "r2": val_metrics[1],
            "mae": val_metrics[2],
            "medae": val_metrics[3],
            "mape": val_metrics[4],
            "training_time": elapsed,
            "best_params": str(search_obj.best_params_)
        },
        {
            "variant": variant_name,
            "dataset": "Test",
            "model": model_name,
            "rmse": test_metrics[0],
            "r2": test_metrics[1],
            "mae": test_metrics[2],
            "medae": test_metrics[3],
            "mape": test_metrics[4],
            "training_time": elapsed,
            "best_params": str(search_obj.best_params_)
        }
    ]
    model_path = Path(__file__).parent / "results" / f"{model_filename}_{variant_name}.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"{model_name} model for variant {variant_name} saved to {model_path}")
    pred_drop_cols = drop_cols.union({"Market Value"})
    predictions_dir = Path("../data/predictions") / predictions_subdir / variant_name
    predictions_dir.mkdir(parents=True, exist_ok=True)
    file_pattern = os.path.join(variant_folder, "updated_*.parquet")
    files = glob.glob(file_pattern)
    if not files:
        logger.error(f"No files found in {variant_folder}")
    else:
        for file_path in files:
            try:
                predict_on_file(best_model, file_path, pred_drop_cols, predictions_dir)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
        logger.info(f"Prediction process completed for variant {variant_name}.")
    try:
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            logger.info("GPU memory freed for variant " + variant_name)
    except Exception as e:
        logger.error("Error during GPU memory cleanup: " + str(e))
    return performance


def run_training_pipeline(model_name: str,
                          pipeline_builder,
                          search_class,
                          param_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]]],
                          use_sample_weight: bool,
                          model_filename: str,
                          predictions_subdir: str,
                          metrics_filename: str):
    """
    Run the complete training and prediction pipeline for a given model across all preprocessed variants.

    Args:
        model_name (str): e.g., "Random Forest", "Linear Regression", "XGBoost".
        pipeline_builder: Callable that accepts training data and returns a Pipeline.
        search_class: Hyperparameter search class (e.g., GridSearchCV, HalvingGridSearchCV).
        param_grid (Union[Dict, List[Dict]]): Hyperparameter grid.
        use_sample_weight (bool): Whether to compute sample weights.
        model_filename (str): Filename prefix for saving the trained model.
        predictions_subdir (str): Subdirectory (under ../data/predictions) to save predictions.
        metrics_filename (str): Filename for the CSV file with performance metrics.

    Returns:
        None. Performance metrics are saved to a CSV file.
    """
    all_records = []
    PREPROC_VARIANTS = {
        "enhanced_feature_engineering": "../data/updated/enhanced_feature_engineering",
        "feature_engineering": "../data/updated/feature_engineering",
        "no_feature_engineering": "../data/updated/no_feature_engineering"
    }

    # Process each variant sequentially.
    for vn, vf in PREPROC_VARIANTS.items():
        try:
            result = process_variant(vn, vf, model_name, pipeline_builder,
                                     search_class, param_grid, use_sample_weight,
                                     model_filename, predictions_subdir)
            all_records.extend(result)
        except Exception as exc:
            logger.error(f"Variant {vn} generated an exception: {exc}", exc_info=True)

    csv_path = Path(__file__).parent / "results" / metrics_filename
    import pandas as pd
    pd.DataFrame(all_records).to_csv(csv_path, index=False)
    logger.info(f"Performance metrics saved to {csv_path}")