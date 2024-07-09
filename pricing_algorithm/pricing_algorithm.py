import pandas as pd
import joblib
import numpy as np

MODEL_PATH = "../../models/catboost.pkl"
CONFIG_FILE_PATH = "../config/config.json"


class PricingAlgorithm:
    def __init__(self):
        # Load the model
        self.model = joblib.load("MODEL_PATH")
        # Read threshold from config file
        self.threshold = pd.read_json(CONFIG_FILE_PATH)["pricing_algorithm_threshold"][
            0
        ]

    def predict(self, data):
        """
        Predict the target variable on the dataset, and adjust the predictions by adding the threshold value.
        """
        # Make predictions
        prediction = self.model.predict(data)
        # Adjust the predictions
        adjusted_prediction = prediction + self.threshold
        # Ensure the predictions are not negative
        adjusted_prediction = np.where(adjusted_prediction < 0, 0, adjusted_prediction)
        return adjusted_prediction
