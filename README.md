# SCANIA Component X: Predictive Maintenance Modeling

This repository contains the implementation of my Master's Thesis conducted at **Scania CV AB**. The project develops a Machine Learning framework for **Predictive Maintenance**, specifically focused on binary classification of component failures.

## Project Overview

The goal of this research is to move from reactive/scheduled maintenance to **proactive predictive maintenance**. By analyzing vehicle specifications and operational readouts, we develop models to forecast component failures before they occur.

### Key Features:

- **Feature Engineering**: Processing operational histograms and cumulative counters.
- **RUL Estimation**: Implementing Time-to-Event (TTE) modeling.
- **Scalable Architecture**: Modular Python utilities for data processing and model training.

## Repository Structure

- `notebooks/`:
  - `data_preprocessing.ipynb`: Cleaning and aligning operational readouts with failure labels.
  - `model_experimentation.ipynb`: Testing various architectures (XGBoost, Random Forest, etc.).
- `utils/`:
  - `data_utils.py`: Logic for handling Scania-specific data formats.
  - `models.py`: Model definitions and hyperparameters.
  - `trainer.py`: Training loops and evaluation logic.
- `requirements.txt`: List of dependencies required to reproduce the environment.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/eliasLM-dev/SCANIA_Component_X.git](https://github.com/eliasLM-dev/SCANIA_Component_X.git)
   ```
