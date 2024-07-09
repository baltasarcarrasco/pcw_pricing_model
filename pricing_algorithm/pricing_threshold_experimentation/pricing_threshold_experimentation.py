from model_evaluation.utils.utils import avg_price_diff_score
from logs.logger.logger import get_logger
import pandas as pd

logger = get_logger("logs/pricing_threshold_experimentation.log")

# Read average_price_difference from 'data/results/model_evaluation_results.json'
average_price_difference = pd.read_json("data/results/model_evaluation_results.json")[
    "average_price_difference"
][0]

# Load predictions from 'data/predictions/val_predictions.parquet'
predictions = pd.read_parquet("data/predictions/val_predictions.parquet")

# Add values from 0 to average_price_difference to the predictions
# and compute the average price difference score
# keep track of the best score and the corresponding value
best_score = float("inf")
best_value = None

for value in range(int(average_price_difference)):
    predictions["competitor_lowest_price_pred"] = (
        predictions["competitor_lowest_price_pred"] + 1
    )
    score = avg_price_diff_score(
        predictions["competitor_lowest_price"],
        predictions["competitor_lowest_price_pred"],
        market_share_threshold=0.4,
    )
    if score < best_score:
        best_score = score
        best_value = value

# Log the best value and score
logger.info(f"Best value: {best_value}")
logger.info(f"Best score: {best_score}")
