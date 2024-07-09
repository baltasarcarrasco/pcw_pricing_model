
# PCW Pricing Model

## Overview

This repository contains the code and resources for a pricing algorithm designed to predict the lowest competitor price for an insurance service. The objective of the algorithm is to minimize the average price difference of the predictions that are below the lowest competitor price, while ensuring that at least 30% of the market share is covered.

## Repository Structure

```plaintext
pcw_pricing_model-main/
├── data/
│   ├── predictions/
│   │   ├── test_predictions.parquet
│   │   ├── val_predictions.parquet
│   ├── processed/
│   │   ├── final_validation_set.parquet
│   │   ├── test.parquet
│   │   ├── train.parquet
│   │   ├── val.parquet
│   ├── results/
│   │   ├── model_evaluation_results.json
│   │   ├── pricing_algorithm_evaluation_results.json
├── eda/
│   ├── exploratory_data_analisis.ipynb
├── logs/
│   ├── __init__.py
│   ├── hyperparameter_tuning.log
│   ├── model_evaluation.log
│   ├── models_experimentation.log
│   ├── preprocessing_pipeline.log
│   ├── pricing_algorithm_evaluation.log
│   ├── pricing_threshold_experimentation.log
│   ├── logger/
│   │   ├── logger.py
├── model_development/
│   ├── preprocessing/
│   │   ├── preprocessing.py
│   ├── model_training/
│   │   ├── hyperparameter_tuning/
│   │   │   ├── catboost_info/
│   │   │   │   ├── catboost_training.json
│   │   │   │   ├── learn_error.tsv
│   │   │   │   ├── time_left.tsv
│   │   │   │   ├── events.out.tfevents
│   │   ├── model_experimentation/
│   │   │   ├── model_experimentation.py
│   │   │   ├── catboost_info/
│   │   │   │   ├── catboost_training.json
│   │   │   │   ├── learn_error.tsv
│   │   │   │   ├── time_left.tsv
│   │   │   │   ├── events.out.tfevents
├── models/
│   ├── catboost.pkl
├── pricing_algorithm/
│   ├── pricing_algorithm.py
│   ├── pricing_threshold.txt
│   ├── config/
│   │   ├── config.json
│   ├── deployment/
│   │   ├── deploy.py
│   ├── evaluation/
│   │   ├── evaluation.py
│   ├── pricing_threshold_experimentation/
│   │   ├── pricing_threshold_experimentation.py
├── .gitattributes
├── .gitignore
├── README.md
├── ideas.txt
├── requirements.txt
```

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- Required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/pcw_pricing_model.git
   cd pcw_pricing_model
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Data Preprocessing

Run the preprocessing script to prepare the data:
```sh
python model_development/preprocessing/preprocessing.py
```

### Model Experimentation

Experiment with different models:
```sh
python model_development/model_training/model_experimentation/model_experimentation.py
```

### Model Training

Train the model using hyperparameter tuning:
```sh
python model_development/model_training/hyperparameter_tuning/hyperparameter_tuning.py
```

### Model Evaluation

Evaluate the performance of the model:
```sh
python model_development/model_training/model_evaluation/model_evaluation.py
```

### Pricing Threshold Experimentation

Experiment with different pricing thresholds:
```sh
python pricing_algorithm/pricing_threshold_experimentation/pricing_threshold_experimentation.py
```

### Pricing Algorithm

Run the pricing algorithm:
```sh
python pricing_algorithm/pricing_algorithm.py
```

### Pricing Algorithm Evaluation

Evaluate the performance of the pricing algorithm:
```sh
python pricing_algorithm/evaluation/evaluation.py
```

### Deployment

The code for deploying the algorithm in AWS Sagemaker is located in the `pricing_algorithm/deployment` directory.

## Logs

Logs for various processes are stored in the `logs` directory, including hyperparameter tuning, model evaluation, and pricing algorithm evaluation.

## Results

Evaluation results are stored in the `data/results` directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For any questions or feedback, please contact baltasarcarrasco0@gmail.com.
