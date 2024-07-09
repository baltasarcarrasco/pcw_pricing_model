from catboost import CatBoostRegressor
import json


class PricingAlgorithm:
    def __init__(self):
        model = CatBoostRegressor()
        self.model = model.load_model("models/catboost_model.pkl")
        # read threshold from config file
        config_file = "config/config.json"
        with open(config_file) as f:
            config = json.load(f)
            self.threshold = config["threshold"]

    def predict(self, data):
        prediction = self.model.predict(data)
        adjusted_prediction = prediction + self.threshold
        return adjusted_prediction
