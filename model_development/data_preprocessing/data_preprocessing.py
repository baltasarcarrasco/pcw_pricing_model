from pipelines.preprocessing_pipeline import DataPreprocessingPipeline

RAW_TRAIN_DATA_PATH = (
    "C:\\Users\\balta\\OneDrive\\Escritorio\\pcw_pricing_model\\data\\raw\\train.xlsx"
)
RAW_TEST_DATA_PATH = (
    "C:\\Users\\balta\\OneDrive\\Escritorio\\pcw_pricing_model\\data\\raw\\test.xlsx"
)
PROCESS_TRAIN_DATA_OUTPUT_PATH = (
    "C:\\Users\\balta\\OneDrive\\Escritorio\\pcw_pricing_model\\data\\processed\\"
)
PROCESS_TEST_DATA_OUTPUT_PATH = (
    "C:\\Users\\balta\\OneDrive\\Escritorio\\pcw_pricing_model\\data\\processed\\"
)


# Process training data
pipeline = DataPreprocessingPipeline(filepath=RAW_TRAIN_DATA_PATH)
pipeline.preprocess(
    output_path=PROCESS_TRAIN_DATA_OUTPUT_PATH,
    training_set=True,
)

# Process test data
pipeline = DataPreprocessingPipeline(filepath=RAW_TEST_DATA_PATH)
pipeline.preprocess(
    output_path=PROCESS_TEST_DATA_OUTPUT_PATH,
    training_set=False,
)
