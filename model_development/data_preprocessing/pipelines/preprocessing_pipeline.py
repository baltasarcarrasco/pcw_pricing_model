import sys

# add parent directory to path
sys.path.append("../")
from logs.logger.logger import get_logger
from utils.utils import (
    load_data,
    clean_column_types,
    treat_missing_values,
    encode_categorical_features,
    generate_features,
    scale_numerical_features,
    select_features,
    save_data,
)


# Set up logging
logger = get_logger("../logs/preprocessing_pipeline.log")

# TODO: Implement the class as an sklearn Transformer in order to use it in a sklearn pipeline


class DataPreprocessingPipeline:
    def __init__(self, filepath):
        logger.info("Loading data...")
        logger.info(f"Filepath: {filepath}")
        try:
            self.data = load_data(filepath)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def preprocess(self, output_path, training_set=True):
        logger.info("Preprocessing data...")
        try:
            logger.info("Treating missing values...")
            self.data = treat_missing_values(self.data)
            logger.info("Cleaning column types...")
            self.data = clean_column_types(self.data)
            logger.info("Encoding categorical features...")
            self.data = encode_categorical_features(self.data)
            logger.info("Generating features...")
            self.data = generate_features(self.data)
            logger.info("Scaling features")
            self.data = scale_numerical_features(self.data)
            logger.info("Selecting features...")
            self.data = select_features(self.data, training_set=training_set)
            logger.info("Saving data...")
            save_data(self.data, output_path, training_set=training_set)
            logger.info(f"Data saved successfully to {output_path}")
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise
