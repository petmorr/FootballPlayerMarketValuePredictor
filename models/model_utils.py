"""
Common utilities for model training and prediction, including data loading,
preprocessing, evaluation, and optional GPU-accelerated XGBoost support.
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
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from logging_config import configure_logger

logger = configure_logger("model_utils", "model_utils.log")


def load_updated_data(data_dir: str) -> pd.DataFrame:
    """
    Load and concatenate all Parquet files from a directory matching 'updated_*.parquet'.

    Args:
        data_dir: Directory containing Parquet files.

    Returns:
        Combined DataFrame.
    """
    pattern = os.path.join(data_dir, "updated_*.parquet")
    files = glob.glob(pattern)
    if not files:
        msg = f"No files found in {data_dir} matching pattern 'updated_*.parquet'"
        logger.error(msg)
        raise FileNotFoundError(msg)
    df_list = [pd.read_parquet(f) for f in files]
    combined_df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Combined data shape: {combined_df.shape}")
    return combined_df


def compute_sample_weights(y: pd.Series, factor: float = 1.5) -> np.ndarray:
    """
    Compute sample weights, inversely proportional to deviation from the median.

    Args:
        y: Target values.
        factor: Scaling factor for weighting.

    Returns:
        Array of sample weights.
    """
    median = np.median(y)
    mad = np.median(np.abs(y - median))
    weights = 1 / (1 + np.abs(y - median) / (mad * factor + 1e-6))
    return weights


def select_features_and_target(
        df: pd.DataFrame, drop_cols: Optional[set] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features from target ('Market Value').

    Args:
        df: DataFrame with all columns.
        drop_cols: Columns to drop from the feature set.

    Returns:
        (Features, Target) as (X, y).
    """
    if drop_cols is None:
        drop_cols = {"league", "season", "born", "country_code", "squad"}
    if "Market Value" not in df.columns:
        msg = "DataFrame lacks 'Market Value' column."
        logger.error(msg)
        raise ValueError(msg)
    X = df.drop(columns=drop_cols.union({"Market Value"}), errors="ignore")
    y = df["Market Value"]
    mask = y.notnull()
    X, y = X[mask], y[mask]
    logger.info(f"Features shape: {X.shape}, target shape: {y.shape}")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Construct a ColumnTransformer for numeric and categorical features.

    Args:
        X: Features DataFrame.

    Returns:
        Configured preprocessor.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "player" in numeric_features:
        numeric_features.remove("player")

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info(f"Numeric features: {numeric_features}")
    logger.info(f"Categorical features: {categorical_features}")

    numeric_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def split_data(
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train (60%), validation (20%), and test (20%) sets. Uses group-aware splitting.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        groups: Series indicating group membership.
        random_state: Random seed.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test, groups_train, groups_test).
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

    logger.info(
        f"Train shape: {X_train.shape}, "
        f"Val shape: {X_val.shape}, "
        f"Test shape: {X_test.shape}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, groups_train, groups_test


def evaluate_model(
        model,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "Dataset"
) -> Tuple[float, float, float, float, float]:
    """
    Evaluate a trained regression model with multiple metrics.

    Args:
        model: Trained model.
        X: Features for prediction.
        y: True target values.
        dataset_name: Label for logging.

    Returns:
        (RMSE, R^2, MAE, Median AE, MAPE).
    """
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    medae = median_absolute_error(y, predictions)
    mape = mean_absolute_percentage_error(y, predictions) * 100
    logger.info(
        f"{dataset_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, "
        f"MAE: {mae:.4f}, MedAE: {medae:.4f}, MAPE: {mape:.2f}%"
    )
    return rmse, r2, mae, medae, mape


def predict_on_file(
        model,
        file_path: str,
        drop_cols: set,
        predictions_dir: Union[str, Path]
) -> None:
    """
    Generate predictions for a single Parquet file and save them.

    Args:
        model: Trained regression model.
        file_path: Path to input Parquet file.
        drop_cols: Columns to drop from features.
        predictions_dir: Directory for output predictions.
    """
    logger.info(f"Predicting file: {file_path}")
    df = pd.read_parquet(file_path)
    df_pred = df.copy()
    X_pred = df.drop(columns=drop_cols, errors="ignore")
    preds = model.predict(X_pred)
    df_pred["predicted_market_value"] = np.round(preds, 2)
    base_name = Path(file_path).name
    output_file = Path(predictions_dir) / f"predicted_{base_name}"
    df_pred.to_parquet(output_file, index=False)
    logger.info(f"Saved predictions to: {output_file}")


import cupy as cp  # GPU acceleration for XGB
from xgboost import XGBRegressor, DMatrix


class XGBRegressorGPU:
    """
    GPU-accelerated XGBRegressor wrapper. Enables GPU if available.
    """

    def __init__(self, **kwargs):
        if cp is not None and "tree_method" not in kwargs:
            kwargs["tree_method"] = "gpu_hist"
        self._model = XGBRegressor(**kwargs)

    def fit(self, X: Any, y: Any, **kwargs) -> "XGBRegressorGPU":
        if cp is not None:
            X = cp.asarray(X)
            y = cp.asarray(y)
        self._model.fit(X, y, **kwargs)
        return self

    def predict(self, X: Any, **kwargs) -> np.ndarray:
        X_host = np.asarray(X)
        if cp is not None:
            X_gpu = cp.asarray(X_host)
            dmat = DMatrix(X_gpu)
            preds = self._model.get_booster().predict(dmat, **kwargs)
            return cp.asnumpy(preds)
        # CPU fallback
        dmat = DMatrix(X_host)
        return self._model.get_booster().predict(dmat, **kwargs)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return self._model.get_params(deep=deep)

    def set_params(self, **params) -> "XGBRegressorGPU":
        self._model.set_params(**params)
        return self

    def get_booster(self):
        return self._model.get_booster()


def process_variant(
        variant_name: str,
        variant_folder: str,
        model_name: str,
        pipeline_builder,
        search_class,
        param_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]]],
        use_sample_weight: bool,
        model_filename: str,
        predictions_subdir: str,
        search_kwargs: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Train, evaluate, and generate predictions for a single feature-engineering variant.

    Args:
        variant_name: Name of the variant.
        variant_folder: Path containing updated Parquet files.
        model_name: Display name of the model.
        pipeline_builder: Function to build a pipeline with preprocessing + model.
        search_class: Hyperparameter search class (e.g., GridSearchCV).
        param_grid: Parameter grid for the search.
        use_sample_weight: Whether to compute sample weights for training.
        model_filename: Prefix for saving the trained model.
        predictions_subdir: Subdirectory for saving predictions under ../data/predictions.
        search_kwargs: Extra arguments for the search class.

    Returns:
        List of performance metric records.
    """
    import joblib
    from sklearn.model_selection import GroupKFold

    drop_cols = {"league", "season", "born", "country_code", "squad", "rank", "position"}
    logger.info(f"Processing variant: {variant_name} at folder: {variant_folder}")

    df = load_updated_data(variant_folder)
    X_all, y_all = select_features_and_target(df, drop_cols=drop_cols)
    groups_all = X_all["player"]
    X_all.drop(columns=["player"], inplace=True, errors="ignore")

    X_train, X_val, X_test, y_train, y_val, y_test, groups_train, _ = split_data(
        X_all, y_all, groups_all, random_state=42
    )

    fit_params = {}
    if use_sample_weight:
        sample_weights = compute_sample_weights(y_train)
        fit_params = {"regressor__sample_weight": sample_weights}

    pipeline = pipeline_builder(X_train)
    cv = GroupKFold(n_splits=5)
    if search_kwargs is None:
        search_kwargs = {}

    # Handle random vs. grid search:
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

    logger.info(f"Starting hyperparameter search: {model_name} on variant: {variant_name}")
    start = time.time()
    search_obj.fit(X_train, y_train, groups=groups_train, **fit_params)
    elapsed = time.time() - start
    logger.info(f"Completed hyperparameter search in {elapsed:.2f}s. Best params: {search_obj.best_params_}")

    best_model = search_obj.best_estimator_
    val_metrics = evaluate_model(best_model, X_val, y_val, f"Validation ({variant_name} - {model_name})")
    test_metrics = evaluate_model(best_model, X_test, y_test, f"Test ({variant_name} - {model_name})")

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
            "best_params": str(search_obj.best_params_),
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
            "best_params": str(search_obj.best_params_),
        },
    ]

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = results_dir / f"{model_filename}_{variant_name}.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"Saved {model_name} model for {variant_name} to {model_path}")

    pred_drop_cols = drop_cols.union({"Market Value"})
    pred_dir = Path("../data/predictions") / predictions_subdir / variant_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    files = glob.glob(os.path.join(variant_folder, "updated_*.parquet"))
    if not files:
        logger.error(f"No files found for predictions in {variant_folder}")
    else:
        for fpath in files:
            try:
                predict_on_file(best_model, fpath, pred_drop_cols, pred_dir)
            except Exception as e:
                logger.error(f"Error predicting on {fpath}: {e}", exc_info=True)
        logger.info(f"Predictions done for {variant_name}")

    # Release GPU memory if possible
    if cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            logger.info(f"Freed GPU memory for variant {variant_name}")
        except Exception as e:
            logger.error(f"GPU cleanup error: {e}")

    return performance


def run_training_pipeline(
        model_name: str,
        pipeline_builder,
        search_class,
        param_grid: Union[Dict[str, List[Any]], List[Dict[str, Any]]],
        use_sample_weight: bool,
        model_filename: str,
        predictions_subdir: str,
        metrics_filename: str
) -> None:
    """
    End-to-end pipeline for training across all FE variants, saving models, predictions, and metrics.

    Args:
        model_name: Display name of the model.
        pipeline_builder: Function that returns a Pipeline.
        search_class: Hyperparameter search class.
        param_grid: Hyperparameter grid.
        use_sample_weight: Whether to compute sample weights in training.
        model_filename: Prefix for model file output.
        predictions_subdir: Subdir for storing prediction outputs.
        metrics_filename: CSV file for storing performance metrics.
    """
    from pathlib import Path

    PREPROC_VARIANTS = {
        "enhanced_feature_engineering": "../data/updated/enhanced_feature_engineering",
        "feature_engineering": "../data/updated/feature_engineering",
        "no_feature_engineering": "../data/updated/no_feature_engineering",
    }

    all_records = []
    for vn, folder in PREPROC_VARIANTS.items():
        try:
            records = process_variant(
                vn,
                folder,
                model_name,
                pipeline_builder,
                search_class,
                param_grid,
                use_sample_weight,
                model_filename,
                predictions_subdir,
            )
            all_records.extend(records)
        except Exception as exc:
            logger.error(f"Variant {vn} raised: {exc}", exc_info=True)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / metrics_filename
    pd.DataFrame(all_records).to_csv(csv_path, index=False)
    logger.info(f"Saved metrics to {csv_path}")
