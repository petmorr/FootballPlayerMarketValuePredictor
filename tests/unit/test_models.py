import numpy as np
import pandas as pd
import pytest

import models.model_utils as mu
from models.linear_regression_model import lr_pipeline_builder
from models.random_forest_model import rf_pipeline_builder
from models.xgboost_model import xgb_pipeline_builder


class Dummy:
    def predict(self, X): return X["f"].values * 2


def test_compute_sample_weights():
    y = np.array([1, 2, 3, 4, 5])
    w = mu.compute_sample_weights(y)
    assert len(w) == len(y)
    assert w.max() == pytest.approx(1.0)


def test_select_features_and_target_success():
    df = pd.DataFrame({
        "Market Value": [10, None, 30],
        "league": ["L"] * 3,
        "season": ["S"] * 3,
        "born": [1990, 1991, 1992],
        "country_code": ["C"] * 3,
        "squad": ["T"] * 3,
        "rank": [1, 2, 3],
        "f": [5, 6, 7],
    })
    X, y = mu.select_features_and_target(df)
    assert "Market Value" not in X.columns
    assert len(X) == 2


def test_select_features_missing():
    with pytest.raises(ValueError):
        mu.select_features_and_target(pd.DataFrame({"x": [1, 2]}))


def test_evaluate_model_perfect():
    X = pd.DataFrame({"f": [1, 2, 3]})
    y = pd.Series([2, 4, 6])
    rmse, r2, mae, medae, mape = mu.evaluate_model(Dummy(), X, y, "T")
    assert r2 == pytest.approx(1.0)


# MC‑01–MC‑07 minimal smoke tests:
def test_train_linear_regression(tmp_path, sample_updated_parquet):
    # call training entrypoint, expect a CSV metrics file
    out = tmp_path / "models" / "results";
    out.mkdir(parents=True)
    lr_pipeline_builder(sample_updated_parquet)
    assert (out / "performance_metrics_linear_regression.csv").exists()


def test_train_random_forest(tmp_path, sample_updated_parquet):
    out = tmp_path / "models" / "results";
    out.mkdir(parents=True)
    rf_pipeline_builder(sample_updated_parquet)
    assert (out / "performance_metrics_random_forest.csv").exists()


def test_train_xgboost(tmp_path, sample_updated_parquet, monkeypatch):
    # MC‑03: if no GPU, falls back
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    out = tmp_path / "models" / "results";
    out.mkdir(parents=True)
    xgb_pipeline_builder(sample_updated_parquet)
    assert (out / "performance_metrics_xgboost.csv").exists()
