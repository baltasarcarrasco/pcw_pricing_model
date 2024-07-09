from pricing_algorithm.pricing_algorithm import PricingAlgorithm
import pandas as pd

TEST_SET_PATH = "../../data/processed/final_test_set.parquet"
PREDICTIONS_OUTPUT_PATH = "../../data/predictions/final_predictions.csv"

# Load data
test_df = pd.read_parquet(TEST_SET_PATH)

# Split the dataset into features and target
X_quote_ids = test_df["quote_id"]
X_test = test_df.drop(["quote_id"], axis=1)

# Create an instance of the PricingAlgorithm class
pricing_algorithm = PricingAlgorithm()
# Predict the target variable on the validation set
y_pred = pricing_algorithm.predict(X_test)

# Save the predictions as proposed price, and their corresponding quote_ids
predictions = pd.DataFrame({"quote_id": X_quote_ids, "proposed_price": y_pred})
predictions.to_csv(PREDICTIONS_OUTPUT_PATH)
