# House Price Prediction

Project Status: Work in Progress

This project is a personal exercise used to practice and demonstrate a structured data science modeling workflow. The goal is to predict housing prices in Ames, Iowa, while maintaining disciplined coding and modeling habits.

## Key Objectives

* Demonstrate data preparation and feature engineering using DuckDB.
* Implement a variety of modeling techniques, including traditional regression and machine learning (XGBoost, LightGBM, CatBoost).
* Utilize model stacking to improve prediction performance.
* Maintain a clean, modular project structure that separates data processing, modeling, and validation.

## Modeling Workflow

The project uses a three-stage pipeline to ensure rigorous evaluation and avoid data leakage.

### Phase 1: Training
Dataset: 80% of training data (~1,168 records)

* Baseline: OLS Linear Regression.
* Machine Learning: L1/L2 regression, tree-based models, and gradient boosting (XGBoost, LightGBM, CatBoost).
* Tuning: 10-fold cross-validation for hyperparameter optimization.

### Phase 2: Validation
Dataset: 20% of training data (~292 records)

* Metric: RMSE of the log-transformed sales price.
* Selection: Performance comparison across models to select candidates for stacking or final use.

### Phase 3: Testing
Dataset: Unseen test data

* Final evaluation using the log-RMSE metric to report the selected model's performance.

## Tools Used

* Database: DuckDB
* Modeling: Scikit-learn, XGBoost, LightGBM, CatBoost
* Package Management: uv

## Getting Started

### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for Python and dependency management. For installation instructions, please refer to the [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Setup

In the project directory, run:

```bash
uv venv
uv sync
```

`uv venv` initializes the virtual environment, while `uv sync` reads `pyproject.toml` and `uv.lock`, installs the correct Python version, and synchronizes all dependencies. No manual `pip install` needed.

### Running the Pipeline

Each stage can be run as a single command from the project root:

```bash
uv run python -m s1_data          # Run all data prep scripts (a0-a9)
uv run python -m s2_model         # Run all modeling scripts (a1-a8)
uv run python -m s4_prediction    # Run final predictions
```

To run an individual script within a stage:

```bash
uv run python -m s1_data.a3_contextual_imputation
uv run python -m s2_model.a7_catboost
```

## Project Structure

* `s0_eda/` — Exploratory data analysis and visualization.
* `s1_data/` — Data loading and DuckDB-based processing.
* `s2_model/` — Model definitions, training scripts, and tuning.
* `s3_validation/` — Shared evaluation helpers (imported by other modules).
* `s4_prediction/` — Final prediction generation.
