"""
Random Forest Model for predicting Market Value.

This script loads the updated data, builds a preprocessing pipeline,
performs extensive hyperparameter tuning on a Random Forest model using GridSearchCV,
evaluates the tuned model on validation and test sets, and saves the final model.
"""

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import logging
from model_utils import (
    load_updated_data,
    select_features_and_target,
    build_preprocessor,
    split_data,
    evaluate_model,
)


def train_random_forest_model() -> None:
    # Load the combined updated data
    df = load_updated_data(data_dir="../data/updated")
    X, y = select_features_and_target(df)

    # Build the preprocessing pipeline
    preprocessor = build_preprocessor(X)

    # Split data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_state=42)

    # Create the Random Forest pipeline
    pipeline_rf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ])

    # Define an extensive hyperparameter grid
    param_grid = {
        "regressor__n_estimators": [100, 200, 300],
        "regressor__max_depth": [None, 10, 20, 30],
        "regressor__min_samples_split": [2, 5, 10],
        "regressor__min_samples_leaf": [1, 2, 4],
        "regressor__max_features": [None, "sqrt", "log2"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline_rf,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    # Perform hyperparameter tuning on the training set
    grid_search.fit(X_train, y_train)

    logging.info("Best hyperparameters for Random Forest:")
    logging.info(grid_search.best_params_)

    # Evaluate performance on validation and test sets using the best estimator
    logging.info("Evaluating Random Forest Model on Validation Set")
    evaluate_model(grid_search.best_estimator_, X_val, y_val, dataset_name="Validation (RF)")

    logging.info("Evaluating Random Forest Model on Test Set")
    evaluate_model(grid_search.best_estimator_, X_test, y_test, dataset_name="Test (RF)")

    # Save the best model to disk
    model_path = "random_forest_model.pkl"
    joblib.dump(grid_search.best_estimator_, model_path)
    logging.info(f"Random Forest model saved to {model_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Starting training for Random Forest Model")
    train_random_forest_model()
    logging.info("Training for Random Forest Model completed")


if __name__ == "__main__":
    main()
