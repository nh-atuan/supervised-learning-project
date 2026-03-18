# Data Directory

This folder manages all datasets used in the project for two tasks:

- Regression
- Classification

## Goals

- Do not commit large data files (especially `.csv`) to Git.
- Ensure reproducibility: the full path from raw data to processed data can be reproduced.
- Keep regression and classification data separate to avoid mixing pipelines.

## Structure

- `raw/`: Original downloaded datasets, not preprocessed.
- `processed/`: Preprocessed datasets (clean, split, encode, scale, etc.).
- `external/`: (optional) supplementary data from external sources.

## Datasets

### 1. Regression - Bike Sharing Dataset

- Source: UCI Machine Learning Repository
- Link: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- Description: Predict bike rental counts using weather, season, and time-based features.

### 2. Classification - Forest Cover Type Dataset

- Source: Kaggle
- Link: https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset
- Description: Classify forest cover type using cartographic and environmental variables.

## Important Conventions

- Raw data is NOT included in this repository.
- Follow the instructions in each subfolder to place data in the correct location.
- Keep input file names exactly as documented so notebooks/scripts run consistently.
