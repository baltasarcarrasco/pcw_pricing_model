import numpy as np


def market_share(competitor_lowest_prices, competitor_lowest_prices_pred):
    """
    Compute the market share of the model based on the competitor's lowest prices
    """
    # Compute the amount of times the model predicts a lower price than the competitor
    sold_policies = competitor_lowest_prices_pred < competitor_lowest_prices
    # Compute market share
    market_share = np.mean(sold_policies)

    return market_share


def avg_price_diff(competitor_lowest_prices, competitor_lowest_prices_pred):
    """
    Compute the average price difference between the competitor's lowest price and the model's prediction
    """
    # Compute the difference between the competitor's lowest price and the model's prediction
    price_diff = competitor_lowest_prices - competitor_lowest_prices_pred
    # Keep only the positive differences
    positive_differences = price_diff[price_diff > 0]
    # Compute the average price difference
    avg_price_diff = np.mean(positive_differences)
    return avg_price_diff


def avg_price_diff_score(
    competitor_lowest_prices, competitor_lowest_prices_pred, market_share_threshold=0.3
):
    """
    Compute the average price difference score based on the market share threshold
    """
    if (
        market_share(competitor_lowest_prices, competitor_lowest_prices_pred)
        > market_share_threshold
    ):
        return avg_price_diff(competitor_lowest_prices, competitor_lowest_prices_pred)
    else:
        return float("inf")
