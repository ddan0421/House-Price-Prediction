# House Price Prediction

Project Status: Work in Progress

This project is a personal exercise used to practice and demonstrate a structured data science modeling workflow. The goal is to predict housing prices in Ames, Iowa, while maintaining disciplined coding and modeling habits.

## Key Objectives

* Demonstrate data preparation and feature engineering using DuckDB.
* Implement a variety of modeling techniques, including traditional regression and machine learning (XGBoost, LightGBM, CatBoost).
* Utilize model stacking to improve prediction performance.
* Maintain a clean, modular project structure that separates data processing, modeling, and validation.

## Data Preparation

All intermediate data lives in a single `data/AmesHousePrice.duckdb` file. Each `s1_data` script reads the upstream tables, applies one transformation step (raw load, outlier removal, imputation, feature engineering, model-specific prep), and writes the result back as a new table via `save_df` in `s1_data/db_utils.py`. Downstream `s2_model` and `s4_prediction` scripts then pull the model-specific train/val/test tables (e.g. `X_train_xgb`, `test_lgbm`) with `load_df`, keeping every stage reproducible from the same DuckDB file.

Outliers are dropped early, in `s1_data/a2_drop_outliers.py`: two partial sales of homes over 4,000 sq ft that sold far below market. Removing them before imputation means every downstream script splits the same rows, and it improved held-out RMSE for every linear and kernel model. See `s0_eda/EDA-charts.py` for the leverage, studentized residual, and Cook's distance diagnostics that identify them.

## Modeling Workflow

The project uses a three-stage pipeline to ensure rigorous evaluation and avoid data leakage.

### Phase 1: Training
Dataset: 80% of training data (1,166 records)

* Baseline: OLS Linear Regression.
* Machine Learning: L1/L2 regression, tree-based models, and gradient boosting (XGBoost, LightGBM, CatBoost).
* Tuning: 10-fold cross-validation throughout, by grid search for the linear, kernel and tree models, and by grid search plus Bayesian optimization for the three boosters.

### Phase 2: Validation
Dataset: 20% of training data (292 records)

* Metric: RMSE of the log-transformed sale price.
* Read once, at the end, as an independent check. Nothing is selected on these rows — the stack picks its members from out-of-fold predictions over the training split, so the validation set stays clean enough to be worth reporting.

### Phase 3: Testing
Dataset: Unseen test features (no SalePrice)

* Base learners are refit on train + val, and their test predictions are combined by the meta-learner to produce the final SalePrice for each test record.

### Stacking

`s2_model/a8_stacking.py` builds a 10-fold out-of-fold prediction matrix over the training split, then fits a non-negative least squares meta-learner on it. The non-negativity constraint matters because base predictions correlate above 0.95 with one another, which makes unconstrained OLS answer with large offsetting coefficients that extrapolate badly. It also drives redundant models to exactly zero weight, so it performs model selection as a side effect. The surviving weights are written to `models/meta_learner_nnls.json` and read back by `s4_prediction`.

The out-of-fold folds deliberately use a different seed from the one the base models were tuned against, so no model is scored on the same partition its hyperparameters were chosen to win.

## Tools Used

* Database: DuckDB
* Modeling: Scikit-learn, XGBoost, LightGBM, CatBoost, statsmodels
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
* `models/` — Fitted base models, tuned CatBoost parameters, and the meta-learner weights that `s4_prediction` reads.
