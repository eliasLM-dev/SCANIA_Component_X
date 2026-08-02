# Predictive Maintenance for Heavy Trucks: A Cost Sensitive Approach to Component Failure Prediction

This repository contains the full implementation for a Masters's thesis on predicting component failures in heavy trucks using operational sensor data, evaluated under a real, industry defined cost structure rather than standard classification metrics alone.

Full thesis available at Stockholm University's DiVA portal: [Read the full thesis here](https://su.diva-portal.org/smash/get/diva2:2080706/FULLTEXT01.pdf)

## Overview

This project builds and evaluates a complete machine learning pipeline for predictive maintenance using the SCANIA Component X dataset, a public benchmark of anonymized operational readouts, time to event labels, and vehicle specifications from a real fleet of heavy trucks. The goal is not just to classify failures accurately, but to minimize the actual maintenance cost incurred, using a five class cost matrix defined by the dataset authors.

Several model families are trained and compared, including logistic regression, LSTM and TCN sequence models, and SMOTE augmented classifiers built on TapNet embeddings. The strongest candidates are then verified across 30 random seeds to confirm that performance differences are statistically significant rather than artifacts of a single training run.

## Key Result

The final selected model is a **TCN (Temporal Convolutional Network)**, chosen based on a full multi seed verification process:

- Average test cost of **7.91 per vehicle**, the lowest among all five evaluated models
- Statistically significant improvement over the LSTM alternative (Mann Whitney U test, p = 0.029)
- Outperforms both the naive predict all healthy baseline (8.62) and a published ensemble benchmark (8.37)
- Competitive with the leading published result on this dataset (CMC LightGBM, 7.20), with the remaining gap largely explained by a structural disadvantage from framing the task as binary rather than five class classification

## Repository Structure

- `Notebooks/`:
  - `data_preprocessing.ipynb`: Cleaning, labeling, and feature engineering.
  - `model_experimentation.ipynb`: Training and comparing LR, LSTM, TCN, and SMOTE augmented models.
  - `model_evaluation.ipynb`: Cost sensitive threshold tuning and held out test evaluation.
  - `model_verification.ipynb`: Multi seed robustness verification of the final candidate models.
- `Data/`:
  - Raw and processed train, validation, and test data.
- `Results/`:
  - `Models/`: Saved trained models.
  - `Predictions/`: Saved predictions and evaluation artifacts.
- `utils.py`: Shared functions used across all notebooks.
- `requirements.txt`: List of dependencies required to reproduce the environment.

## Getting Started

The notebooks in this repository already contain their full output, so browsing them directly on GitHub is the fastest way to review the results without running anything locally.

To run the pipeline yourself:

1. Clone the repository and install dependencies:
   `pip install -r requirements.txt`
2. Download the SCANIA Component X dataset (see the Dataset section below) and place it in the `Data/` folder.
3. Run the notebooks in order: `data_preprocessing.ipynb`, `model_experimentation.ipynb`, `model_evaluation.ipynb`, then `model_verification.ipynb`. Each notebook saves the outputs the next one depends on.

Trained models and saved predictions from the original run are already included under `Results/`, so later notebooks can also be run independently without repeating earlier training steps.

## Pipeline Summary

1. **Preprocessing**: raw sensor readouts are cleaned, missing values are imputed per vehicle, cumulative counters are converted to deltas and log transformed, and features are scaled using train only statistics to avoid leakage.
2. **Experimentation**: multiple model families are trained and compared on validation data, including a logistic regression baseline, LSTM and TCN sequence models, and SMOTE augmented classifiers on TapNet embeddings.
3. **Evaluation**: models are compared using both standard classification metrics and the dataset's true cost matrix, with thresholds tuned to minimize actual maintenance cost rather than accuracy alone.
4. **Verification**: the two strongest candidates (LSTM and TCN) are retrained across 30 seeds to confirm the final model choice is statistically robust rather than a result of favorable random initialization.

## Dataset

This project uses the SCANIA Component X dataset, a public predictive maintenance benchmark of anonymized truck operational data. Full details on the dataset and its cost matrix can be found in the accompanying paper: [SCANIA Component X dataset](https://www.nature.com/articles/s41597-025-04802-6).

## Requirements

See `requirements.txt` for the full list of dependencies, including pandas, numpy, scikit learn, PyTorch, sktime, and XGBoost.

## Author

Elias Larsson Medina, Masters's Thesis, Stockholm University, 2026.
