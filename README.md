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

## Project Structure

* s0_eda/: Exploratory data analysis and visualization.
* s1_data/: Data loading and DuckDB-based processing.
* s2_model/: Model definitions, training scripts, and tuning.
* s3_validation/: Model evaluation and comparison.
* s4_prediction/: Final prediction generation.
