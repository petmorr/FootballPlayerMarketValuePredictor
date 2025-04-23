from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


# Test fixtures
@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing with mock market values"""
    np.random.seed(42)

    # Create data with variety of features and data types
    data = {
        'player': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5'],
        'team': ['Team A', 'Team B', 'Team A', 'Team C', 'Team B'],
        'age': [25, 28, 22, 31, 24],
        'position': ['Forward', 'Midfielder', 'Defender', 'Goalkeeper', 'Forward'],
        'goals': [10, 5, 1, 0, 8],
        'assists': [5, 10, 3, 1, 4],
        'appearances': [30, 28, 25, 20, 22],
        'minutes': [2700, 2500, 2250, 1800, 1980],
        'Market Value': [15000000, 25000000, 8000000, 5000000, 12000000]
    }
    return pd.DataFrame(data)


@pytest.fixture
def large_dataframe():
    """Create a larger dataframe for testing with 1000+ rows"""
    np.random.seed(42)

    # Generate 1000 rows of player data
    n_rows = 1000
    players = [f'Player{i}' for i in range(1, n_rows + 1)]
    teams = np.random.choice(['Team A', 'Team B', 'Team C', 'Team D', 'Team E'], n_rows)
    ages = np.random.randint(18, 40, n_rows)
    positions = np.random.choice(['Forward', 'Midfielder', 'Defender', 'Goalkeeper'], n_rows)
    goals = np.random.randint(0, 25, n_rows)
    assists = np.random.randint(0, 15, n_rows)
    appearances = np.random.randint(1, 38, n_rows)
    minutes = appearances * np.random.randint(60, 95, n_rows)
    market_values = np.random.lognormal(16, 1, n_rows)  # Log-normal distribution for market values

    data = {
        'player': players,
        'team': teams,
        'age': ages,
        'position': positions,
        'goals': goals,
        'assists': assists,
        'appearances': appearances,
        'minutes': minutes,
        'Market Value': market_values
    }
    return pd.DataFrame(data)


@pytest.fixture
def small_dataframe():
    """Create a very small dataframe (3-5 players) for edge case testing"""
    np.random.seed(42)

    # Create minimal data with just 3 players
    data = {
        'player': ['Player1', 'Player2', 'Player3'],
        'team': ['Team A', 'Team B', 'Team A'],
        'age': [25, 28, 22],
        'position': ['Forward', 'Midfielder', 'Defender'],
        'goals': [10, 5, 1],
        'assists': [5, 10, 3],
        'appearances': [30, 28, 25],
        'minutes': [2700, 2500, 2250],
        'Market Value': [15000000, 25000000, 8000000]
    }
    return pd.DataFrame(data)


@pytest.fixture
def negative_values_dataframe():
    """Create a dataframe with some negative market values for testing robustness"""
    np.random.seed(42)

    # Create data with some negative values
    data = {
        'player': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5'],
        'team': ['Team A', 'Team B', 'Team A', 'Team C', 'Team B'],
        'age': [25, 28, 22, 31, 24],
        'position': ['Forward', 'Midfielder', 'Defender', 'Goalkeeper', 'Forward'],
        'goals': [10, 5, 1, 0, 8],
        'assists': [5, 10, 3, 1, 4],
        'appearances': [30, 28, 25, 20, 22],
        'minutes': [2700, 2500, 2250, 1800, 1980],
        'Market Value': [15000000, -2000000, 8000000, 0, 12000000]  # Negative and zero values
    }
    return pd.DataFrame(data)


@pytest.fixture
def missing_market_value_dataframe():
    """Create a dataframe without the Market Value column"""
    np.random.seed(42)

    # Create data without the Market Value column
    data = {
        'player': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5'],
        'team': ['Team A', 'Team B', 'Team A', 'Team C', 'Team B'],
        'age': [25, 28, 22, 31, 24],
        'position': ['Forward', 'Midfielder', 'Defender', 'Goalkeeper', 'Forward'],
        'goals': [10, 5, 1, 0, 8],
        'assists': [5, 10, 3, 1, 4],
        'appearances': [30, 28, 25, 20, 22],
        'minutes': [2700, 2500, 2250, 1800, 1980],
        # Market Value column is missing
    }
    return pd.DataFrame(data)


# Test MC-01: Linear Regression with sample weights
def test_linear_regression_sample_weights(sample_dataframe):
    """
    Test Case ID: MC-01
    Title: Linear Regression with sample weights
    """
    # Import directly from model_utils
    from models.model_utils import compute_sample_weights, select_features_and_target

    # Don't mock compute_sample_weights, use the actual implementation
    X, y = select_features_and_target(sample_dataframe)
    weights = compute_sample_weights(y)

    # Check that weights have expected properties:
    # 1. All weights should be between 0 and 1
    assert all(0 <= w <= 1 for w in weights)

    # 2. The weights should add up to a reasonable value (5 weights should sum to about 3-4)
    assert 2 < sum(weights) < 5

    # 3. The median value should have the highest weight
    median_val = np.median(y)
    median_idx = np.abs(y - median_val).argmin()

    # Find values that are closer to median and farther from median
    values_near_median = [i for i, val in enumerate(y) if abs(val - median_val) < 5000000]
    values_far_from_median = [i for i, val in enumerate(y) if abs(val - median_val) > 8000000]

    # Values near median should have higher weights than those far from median
    if values_near_median and values_far_from_median:
        avg_near_weight = sum(weights[i] for i in values_near_median) / len(values_near_median)
        avg_far_weight = sum(weights[i] for i in values_far_from_median) / len(values_far_from_median)
        assert avg_near_weight > avg_far_weight


# Test MC-02: Process variant with random forest
def test_process_variant_random_forest(sample_dataframe):
    """
    Test Case ID: MC-02
    Title: Random Forest with moderate data
    """
    # Import the model_utils module
    from models import model_utils

    # Define a real search class that we can use to mock (not MagicMock)
    class MockGridSearchCV:
        __name__ = "GridSearchCV"

        def __init__(self, estimator, param_grid, cv, scoring, verbose, n_jobs, **kwargs):
            self.best_params_ = {"n_estimators": 10}
            self.best_estimator_ = MagicMock()
            self.estimator = estimator
            self.param_grid = param_grid
            self.cv = cv
            self.scoring = scoring
            self.verbose = verbose
            self.n_jobs = n_jobs
            self.kwargs = kwargs

        def fit(self, X, y, groups=None, **kwargs):
            # Just return self, don't actually fit
            return self

    # Create our patches
    with patch('models.model_utils.load_updated_data', return_value=sample_dataframe), \
            patch('models.model_utils.split_data') as mock_split, \
            patch('models.model_utils.evaluate_model', return_value=(1000000, 0.8, 500000, 400000, 20.0)), \
            patch('joblib.dump'), \
            patch('glob.glob', return_value=["test_file.parquet"]), \
            patch('models.model_utils.predict_on_file'), \
            patch('models.model_utils.logger'), \
            patch('sklearn.model_selection.GroupKFold'), \
            patch('models.model_utils.build_preprocessor'):
        # Setup for mock_split
        # Create simplified X and y for the split function to return
        X_numeric = sample_dataframe[['age', 'goals', 'assists', 'appearances', 'minutes']]
        y = sample_dataframe['Market Value']

        # Set up the mock to return simplified data without categorical columns
        mock_split.return_value = (
            X_numeric.iloc[:3],
            X_numeric.iloc[3:4],
            X_numeric.iloc[4:],
            y.iloc[:3],
            y.iloc[3:4],
            y.iloc[4:],
            pd.Series(['Player1', 'Player2', 'Player3']),
            pd.Series(['Player5'])
        )

        # Define a simple pipeline builder that returns a mock
        def build_rf_pipeline(X):
            return MagicMock()

        # Call process_variant with our mock search class and other mocks
        result = model_utils.process_variant(
            variant_name="test_variant",
            variant_folder="test_folder",
            model_name="Random Forest",
            pipeline_builder=build_rf_pipeline,
            search_class=MockGridSearchCV,  # Use our real class, not a MagicMock
            param_grid={"n_estimators": [10, 20]},
            use_sample_weight=False,
            model_filename="rf_model",
            predictions_subdir="rf_predictions"
        )

        # Check that the function returned the expected results
        assert len(result) == 2
        assert result[0]["variant"] == "test_variant"
        assert result[0]["model"] == "Random Forest"
        assert result[0]["dataset"] == "Validation"
        assert result[1]["dataset"] == "Test"


# Test MC-03: XGBoost GPU detection
def test_xgboost_gpu_detection():
    """
    Test Case ID: MC-03
    Title: XGBoost GPU
    """
    # First test with GPU available (mocked)
    with patch('models.model_utils.cp', MagicMock()):
        from models.model_utils import XGBRegressorGPU

        # Create instance with GPU
        xgb_gpu = XGBRegressorGPU()

        # Check that GPU tree method is used
        assert 'gpu_hist' in str(xgb_gpu._model.get_params())

    # Then test with no GPU available
    with patch('models.model_utils.cp', None):
        from models.model_utils import XGBRegressorGPU

        # Create instance without GPU
        xgb_cpu = XGBRegressorGPU()

        # Should use CPU fallback
        assert 'gpu_hist' not in str(xgb_cpu._model.get_params())


# Test MC-04: Very Small Dataset (3-5 total players)
def test_training_with_small_dataset(small_dataframe):
    """
    Test Case ID: MC-04
    Title: Very Small Dataset (3-5 total players)
    """
    # Import directly from model_utils
    from models.model_utils import select_features_and_target, build_preprocessor, evaluate_model

    # Use the small dataset
    X, y = select_features_and_target(small_dataframe)

    # Build preprocessor and check it handles the tiny dataset
    preprocessor = build_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)

    # Check that the transformation worked on the small dataset
    assert X_transformed.shape[0] == 3  # Should have 3 rows

    # Create a simple mock model
    class MockModel:
        def predict(self, X):
            return np.array([10000000, 20000000, 5000000])

    # Use the mock model to evaluate
    metrics = evaluate_model(MockModel(), X, y)

    # Check that metrics were calculated even on tiny dataset
    assert len(metrics) == 5  # Should have 5 metrics
    assert all(not np.isnan(m) for m in metrics)  # No NaN values


# Test MC-05: Large Dataset (1-2k rows)
def test_training_with_large_dataset(large_dataframe):
    """
    Test Case ID: MC-05
    Title: Large Dataset (1-2k rows)
    """
    # Import directly from model_utils
    from models.model_utils import select_features_and_target, build_preprocessor

    # Use the large dataset
    X, y = select_features_and_target(large_dataframe)

    # Build preprocessor and time how long it takes
    import time
    start_time = time.time()

    preprocessor = build_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)

    end_time = time.time()

    # Check preprocessing completes in reasonable time (should be < 5 seconds for 1000 rows)
    assert end_time - start_time < 5

    # Check that the transformation worked on the large dataset
    assert X_transformed.shape[0] == 1000  # Should have 1000 rows


# Test MC-06: Missing "Market Value" or incomplete, updated data
def test_missing_market_value(missing_market_value_dataframe):
    """
    Test Case ID: MC-06
    Title: Missing "Market Value" or incomplete, updated data
    """
    # Import directly from model_utils
    from models.model_utils import select_features_and_target

    # Test with missing Market Value - should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        select_features_and_target(missing_market_value_dataframe)

    # Check error message
    assert "DataFrame lacks 'Market Value' column" in str(exc_info.value)


# Test MC-07: Negative or zero values in the target
def test_negative_values_in_target(negative_values_dataframe):
    """
    Test Case ID: MC-07
    Title: Negative or zero values in the target
    """
    # Import directly from model_utils
    from models.model_utils import select_features_and_target, build_preprocessor, compute_sample_weights

    # Extract features and target with negative values
    X, y = select_features_and_target(negative_values_dataframe)

    # Verify that negative & zero values are accepted
    assert (y <= 0).any()

    # Test that preprocessing works on negative values
    preprocessor = build_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)

    # Compute sample weights with negative values
    weights = compute_sample_weights(y)

    # All weights should be valid (between 0 and 1)
    assert all(0 <= w <= 1 for w in weights)

    # Create a simple mock model to evaluate
    class MockModel:
        def predict(self, X):
            return np.array([10000000, -1000000, 5000000, 0, 8000000])

    # Use the mock model to evaluate (with patched logger)
    with patch('models.model_utils.logger') as mock_logger:
        from models.model_utils import evaluate_model
        metrics = evaluate_model(MockModel(), X, y)

        # Check that metrics were calculated even with negative values
        assert len(metrics) == 5  # Should have 5 metrics
        assert all(not np.isnan(m) for m in metrics)  # No NaN values


# Additional test for evaluate_model function
def test_evaluate_model(sample_dataframe):
    """Test that model evaluation produces expected metrics"""
    # Import directly from model_utils
    from models.model_utils import evaluate_model

    # Create mock model and data
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([10000000, 20000000, 7000000, 6000000, 13000000])

    X = sample_dataframe.iloc[:, :-1]
    y = sample_dataframe.iloc[:, -1]

    # Calculate expected metrics manually
    expected_predictions = np.array([10000000, 20000000, 7000000, 6000000, 13000000])
    true_values = np.array(y)

    expected_rmse = np.sqrt(np.mean((true_values - expected_predictions) ** 2))

    # Run evaluation with patched logger
    with patch('models.model_utils.logger') as mock_logger:
        rmse, r2, mae, medae, mape = evaluate_model(mock_model, X, y)

    # Check that RMSE is close to expected value
    assert abs(rmse - expected_rmse) < 0.01
