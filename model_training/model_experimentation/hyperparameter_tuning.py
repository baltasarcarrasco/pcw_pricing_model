import os
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="../../logs/hyperparameter_tuning.log",
    level=logging.INFO,
)

# Load the dataset
data = pd.read_csv("path/to/dataset.csv")

# Split the dataset into features and target
X = data.drop("target", axis=1)
y = data["target"]

# Define the hyperparameters to tune
param_grid = {
    "iterations": [100, 200, 300],
    "learning_rate": [0.01, 0.1, 0.5],
    "depth": [4, 6, 8],
}
# Log the hyperparameters
logging.info(f"Hyperparameters to tune: {param_grid}")

# Perform grid search cross-validation
try:
    model = CatBoostRegressor()
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X, y)
except Exception as e:
    logging.error(f"Error during grid search: {e}")

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Log the results
logging.info(f"Best model: {best_model}")
logging.info(f"Best hyperparameters: {best_params}")

# Save the best model
model_file = "models/catboost.pkl"
best_model.save_model(model_file)
