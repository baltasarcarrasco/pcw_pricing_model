from catboost import CatBoostRegressor
from utils.utils import avg_price_diff, market_share, avg_price_diff_score
from logs.logger.logger import get_logger
import pandas as pd

logger = get_logger("logs/model_evaluation.log")

# load model from 'models/catboost_model.pkl'
catboost_model = CatBoostRegressor()
catboost_model.load_model("models/catboost_model.pkl")

# load data from 'data/processed/val.parquet'
val = pd.read_parquet("data/processed/val.parquet")

# Split the dataset into features and target
X_val = val.drop("competitor_lowest_price", axis=1)
y_val = val["competitor_lowest_price"]
try:
    # Predict the target variable on the validation set
    y_pred = catboost_model.predict(X_val)

    # Save the predictions and the true values to a parquet file
    pd.DataFrame(
        {"competitor_lowest_price": y_val, "competitor_lowest_price_pred": y_pred}
    ).to_parquet("data/predictions/val_predictions.parquet")
except Exception as e:
    logger.error(f"Error predicting on the validation set: {e}")

# Compute the average price difference
avgerage_price_difference = avg_price_diff(y_val, y_pred)
logger.info(f"Average price difference: {avgerage_price_difference}")
# Compute the market share
market_share_value = market_share(y_val, y_pred)
logger.info(f"Market share: {market_share_value}")
# Compute the average price difference score
avg_price_diff_score_value = avg_price_diff_score(y_val, y_pred)
logger.info(f"Average price difference score: {avg_price_diff_score_value}")

# Save the results to a json file
results = {
    "average_price_difference": avgerage_price_difference,
    "market_share": market_share_value,
    "average_price_difference_score": avg_price_diff_score_value,
}
pd.DataFrame(results, index=[0]).to_json(
    "data/results/model_evaluation_results.json", orient="records"
)
