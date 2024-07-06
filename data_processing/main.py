from pipelines.preprocessing_pipeline import DataPreprocessingPipeline

# Process training data
pipeline = DataPreprocessingPipeline(
    filepath="C:\\Users\\balta\\OneDrive\\Escritorio\\pcw_pricing_model\\data\\raw\\train.xlsx"
)
pipeline.preprocess(
    output_path="C:\\Users\\balta\\OneDrive\\Escritorio\\pcw_pricing_model\\data\\processed\\train.xlsx",
    training=True,
)

# Process test data
pipeline = DataPreprocessingPipeline(
    filepath="C:\\Users\\balta\\OneDrive\\Escritorio\\pcw_pricing_model\\data\\raw\\test.xlsx"
)
pipeline.preprocess(
    output_path="C:\\Users\\balta\\OneDrive\\Escritorio\\pcw_pricing_model\\data\\processed\\test.xlsx",
    training=False,
)
