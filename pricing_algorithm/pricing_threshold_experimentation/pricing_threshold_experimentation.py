from model_development.model_evaluation.utils.utils import (
    avg_price_diff_score,
    market_share,
)
from logs.logger.logger import get_logger
import numpy as np
import pandas as pd

LOG_OUTPUT_PATH = "../../logs/pricing_threshold_experimentation.log"
MODEL_EVALUATION_RESULTS_PATH = "../../data/results/model_evaluation_results.json"
VALIDATION_PREDICTIONS_PATH = "../../data/predictions/val_predictions.parquet"
CONFIG_FILE_PATH = "../config/config.json"

# Set up logging
logger = get_logger(LOG_OUTPUT_PATH)

# Read average_price_difference
average_price_difference = pd.read_json(MODEL_EVALUATION_RESULTS_PATH)[
    "average_price_difference"
][0]

# Load predictions from 'data/predictions/val_predictions.parquet'
predictions = pd.read_parquet(VALIDATION_PREDICTIONS_PATH)

# Add values from 0 to average_price_difference to the predictions
# and compute the average price difference score
# keep track of the best score and the corresponding value
best_score = float("inf")
best_value = None

# Generate 20 points between -average_price_difference and average_price_difference
possible_thresholds = np.linspace(
    -average_price_difference, average_price_difference, num=20
)

# Loop through the possible thresholds
for value in possible_thresholds:
    # Adjust the predictions
    predictions["competitor_lowest_price_pred_adjusted"] = (
        predictions["competitor_lowest_price_pred"] + value
    )
    # Compute the average price difference score
    score = avg_price_diff_score(
        predictions["competitor_lowest_price"],
        predictions["competitor_lowest_price_pred_adjusted"],
        market_share_threshold=0.35,
    )
    # Compute the market share
    market_share_value = market_share(
        predictions["competitor_lowest_price"],
        predictions["competitor_lowest_price_pred_adjusted"],
    )
    # Update the best score and value if necessary
    if score < best_score:
        best_score = score
        best_value = value
        market_share_of_best_value = market_share_value

# Log the best value and score
logger.info(f"Best value: {best_value}")
logger.info(f"Best score: {best_score}")
logger.info(f"Market share: {market_share_of_best_value}")

# Save the best value to a json file
pd.DataFrame({"pricing_algorithm_threshold": [best_value]}).to_json(
    CONFIG_FILE_PATH, orient="records"
)
