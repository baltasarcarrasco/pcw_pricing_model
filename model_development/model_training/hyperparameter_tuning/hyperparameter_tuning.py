import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
import joblib
import sys

# add parent directory to path
sys.path.append("../../")
from logs.logger.logger import get_logger

MODEL_OUTPUT_PATH = (
    "C:\\Users\\balta\\OneDrive\\Escritorio\\pcw_pricing_model\\models\\catboost.pkl"
)


# Set up logging
logger = get_logger("../../logs/hyperparameter_tuning.log")

# Load the data
data = pd.read_parquet("../../data/processed/train.parquet")

# Split the dataset into features and target
X = data.drop("competitor_lowest_price", axis=1)
y = data["competitor_lowest_price"]

param_grid = {
    "colsample_bylevel": [0.01, 0.1],
    "depth": [1, 3, 6, 9, 12],
    "boosting_type": ["Ordered", "Plain"],
    "bootstrap_type": ["Bayesian", "Bernoulli", "MVS"],
}

# Log the hyperparameters
logger.info(f"Hyperparameters to tune: {param_grid}")

# Initialize the best hyperparameters
best_params = None

# Perform randomized grid search cross-validation
try:
    model = CatBoostRegressor()
    random_search = RandomizedSearchCV(
        model, param_grid, cv=5, scoring="neg_root_mean_squared_error"
    )
    random_search.fit(X, y)
    # Get the best score and its hyperparameters
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    # Log the results
    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Best score: {best_score}")

    # Save the best model
    joblib.dump(random_search.best_estimator_, MODEL_OUTPUT_PATH)
except Exception as e:
    logger.error(f"Error during randomized grid search: {e}")
