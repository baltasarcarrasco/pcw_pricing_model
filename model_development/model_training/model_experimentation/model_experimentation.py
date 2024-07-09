import pandas as pd
import sys
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import numpy as np

# add parent directory to path
sys.path.append("../../")
from logs.logger.logger import get_logger

# Set up logging
logger = get_logger("../../logs/models_experimentation.log")

# Load the data
data = pd.read_parquet("../../data/processed/train.parquet")

# Split the data into features and target
X = data.drop("competitor_lowest_price", axis=1)
y = data["competitor_lowest_price"]

# Initialize the models
models = [
    LinearRegression(),
    RandomForestRegressor(),
    DecisionTreeRegressor(),
    ElasticNet(),
    GradientBoostingRegressor(),
    XGBRegressor(),
    CatBoostRegressor(),
    LGBMRegressor(),
]

# TODO: Test performance of CatBoostRegressor by not encoding categorical features.

# Define K-Folds cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate over the models
for model in models:
    try:
        model_name = model.__class__.__name__
        # Perform cross-validation
        rmse_scores = cross_val_score(
            model, X, y, cv=kf, scoring="neg_root_mean_squared_error"
        )

        # Log the results
        logger.info(
            f"Cross-validated RMSE scores for {model_name}: {np.mean(rmse_scores)}"
        )
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
