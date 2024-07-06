import numpy as np
from sklearn.metrics import make_scorer


# Custom scoring function example (you can define your own)
def market_share(competitor_lowest_prices, competitor_lowest_prices_pred):
    sold_policies = competitor_lowest_prices_pred < competitor_lowest_prices
    # Compute market share as amount of times the model predicts a lower price than the competitor
    market_share = np.mean(sold_policies)

    return market_share


def avg_price_diff(competitor_lowest_prices, competitor_lowest_prices_pred):
    # Compute the difference between the competitor's lowest price and the model's prediction
    price_diff = competitor_lowest_prices - competitor_lowest_prices_pred
    # Calculate average price difference for the positive differences
    positive_differences = price_diff[price_diff > 0]
    avg_price_diff = np.mean(positive_differences)
    print(avg_price_diff)
    return avg_price_diff


# Convert custom scoring functions to scorers
avg_price_diff_score = make_scorer(avg_price_diff, greater_is_better=False)
market_share_score = make_scorer(market_share)
