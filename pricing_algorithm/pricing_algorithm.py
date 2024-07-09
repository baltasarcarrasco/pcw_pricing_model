import pandas as pd
import joblib


class PricingAlgorithm:
    def __init__(self):
        self.model = joblib.load("../../models/catboost.pkl")
        # read threshold from config file
        self.threshold = pd.read_json("../config/config.json")[
            "pricing_algorithm_threshold"
        ][0]

    def predict(self, data):
        prediction = self.model.predict(data)
        adjusted_prediction = prediction + self.threshold
        return adjusted_prediction
