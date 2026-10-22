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

## How to run

```bash
pip install -r requirements.txt
```

The pipeline runs in stages, one script per stage:

1. Preprocessing: `model_development/data_preprocessing/data_preprocessing.py`
2. Model experimentation and tuning: `model_development/model_training/model_experimentation/model_experimentation.py`, `model_development/model_training/hyperparameter_tuning/hyperparameter_tuning.py`
3. Model evaluation: `model_development/model_evaluation/model_evaluation.py`
4. Pricing threshold experimentation and evaluation: `pricing_algorithm/pricing_threshold_experimentation/pricing_threshold_experimentation.py`, `pricing_algorithm/evaluation/evaluation.py`
5. Test-set predictions: `pricing_algorithm/predict/predict_test_set.py`

## Results

These figures come from the saved outputs in `data/results/`:

| Evaluation | Avg. price difference (below lowest competitor) | Market share covered |
|---|---|---|
| Model evaluation | 60.56 | 39.3% |
| Pricing algorithm evaluation | 62.40 | 42.4% |

Both meet the ≥30% market-share constraint. Test-set predictions are in `data/predictions/final_predictions.csv`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For any questions or feedback, please contact baltasarcarrasco0@gmail.com.
