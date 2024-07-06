import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="../../logs/models_experimentation.log",
    level=logging.INFO,
)

# Load the data
data = pd.read_excel("../../data/processed/train.xlsx")

# Split the data into features and target
X = data.drop("competitor_lowest_price", axis=1)
y = data["competitor_lowest_price"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the models
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    ElasticNet(),
    GradientBoostingRegressor(),
    XGBRegressor(),
    CatBoostRegressor(),
    LGBMRegressor(),
]

# Define K-Folds cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate over the models
for model in models:
    try:
        model_name = model.__class__.__name__
        # Perform cross-validation
        rmse_scores = cross_val_score(
            model, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error"
        )

        # Log the results
        logging.info(
            f"Cross-validated RMSE scores for {model_name}: {np.mean(rmse_scores)}"
        )
    except Exception as e:
        logging.error(f"Error evaluating {model_name}: {e}")
