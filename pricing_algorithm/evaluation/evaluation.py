from model_development.model_evaluation.utils.utils import (
    avg_price_diff,
    market_share,
)
from pricing_algorithm.pricing_algorithm import PricingAlgorithm
from logs.logger.logger import get_logger
import pandas as pd

LOG_OUTPUT_PATH = "../../logs/pricing_algorithm_evaluation.log"
TEST_DATA_PATH = "../../data/processed/test.parquet"
EVALUATION_RESULTS_PATH = "../../data/results/pricing_algorithm_evaluation_results.json"
TEST_PREDICTIONS_OUTPUT_PATH = "../../data/predictions/test_predictions.parquet"

# Set up logging
logger = get_logger(LOG_OUTPUT_PATH)

# Load the data
val = pd.read_parquet(TEST_DATA_PATH)

# Split the dataset into features and target
X_val = val.drop("competitor_lowest_price", axis=1)
y_val = val["competitor_lowest_price"]

# Create an instance of the PricingAlgorithm class
pricing_algorithm = PricingAlgorithm()
try:
    # Predict the target variable on the validation set
    y_pred = pricing_algorithm.predict(X_val)

    # Save the predictions and the true values to a parquet file
    pd.DataFrame(
        {"competitor_lowest_price": y_val, "competitor_lowest_price_pred": y_pred}
    ).to_parquet(TEST_PREDICTIONS_OUTPUT_PATH)
except Exception as e:
    logger.error(f"Error predicting on the validation set: {e}")

# Compute the average price difference
avgerage_price_difference = avg_price_diff(y_val, y_pred)
logger.info(f"Average price difference: {avgerage_price_difference}")
# Compute the market share
market_share_value = market_share(y_val, y_pred)
logger.info(f"Market share: {market_share_value}")

# Save the results to a json file
results = {
    "average_price_difference": avgerage_price_difference,
    "market_share": market_share_value,
}
pd.DataFrame(results, index=[0]).to_json(EVALUATION_RESULTS_PATH, orient="records")
