import sys
import logging
from utils.utils import (
    load_data,
    clean_column_types,
    treat_missing_values,
    encode_categorical_features,
    generate_features,
    save_data,
)

# add parent directory to path
sys.path.append("../")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/preprocessing.log"),
        logging.StreamHandler(),
    ],
)


class DataPreprocessingPipeline:
    def __init__(self, filepath):
        logging.info("Loading data...")
        logging.info(f"Filepath: {filepath}")
        try:
            self.data = load_data(filepath)
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def preprocess(self, output_path):
        logging.info("Preprocessing data...")
        try:
            logging.info("Treating missing values...")
            self.data = treat_missing_values(self.data)
            logging.info("Cleaning column types...")
            self.data = clean_column_types(self.data)
            logging.info("Encoding categorical features...")
            self.data = encode_categorical_features(self.data)
            logging.info("Generating features...")
            self.data = generate_features(self.data)
            logging.info("Saving data...")
            save_data(self.data, output_path)
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise
