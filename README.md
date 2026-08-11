# House Price Prediction

This project is a personal exercise used to practice and demonstrate a structured data science modeling workflow. The goal is to predict housing prices in Ames, Iowa, while maintaining disciplined coding and modeling habits.

## Key Objectives

* Demonstrate data preparation and feature engineering using DuckDB.
* Implement a variety of modeling techniques, including traditional regression and machine learning (XGBoost, LightGBM, CatBoost).
* Utilize model stacking to improve prediction performance.
* Maintain a clean, modular project structure that separates data processing, modeling, and validation.

## Data Preparation

All intermediate data lives in a single `data/AmesHousePrice.duckdb` file. Each `s1_data` script reads the upstream tables, applies one transformation step (raw load, imputation, feature engineering, model-specific prep), and writes the result back as a new table via `save_df` in `s1_data/db_utils.py`. Downstream `s2_model` and `s4_prediction` scripts then pull the model-specific train/val/test tables (e.g. `X_train_xgb`, `test_lgbm`) with `load_df`, keeping every stage reproducible from the same DuckDB file.

`load_df` returns columns in alphabetical order, and the feature lists in `a1` to `a3` are wrapped in `sorted()`. Trees and boosters sample features by position, so a model's layout would otherwise depend on the order the prep SQL happened to create columns in, and editing that SQL would shift their scores the way a seed change does. A fixed order keeps runs comparable, so a difference between two runs can be attributed to the change under test.

Two partial sales of homes over 4,000 sq ft sold far below market and dominate the influence diagnostics in `s0_eda/EDA-charts.py` (leverage, studentized residual, Cook's distance). They are deliberately **kept** in the training data, because dropping them costs more than it saves: the test set contains a 5,095 sq ft quality-10 partial sale, and with both training examples of that profile gone the model extrapolates past every price it has ever seen. Keeping them leaves `GrLivArea` covered out to 5,642 sq ft.

`Area_vs_Nbhd` divides `GrLivArea` by the median `GrLivArea` of the home's neighbourhood, which separates floor area far above the local norm from floor area itself — the over-improvement signal those two partial sales carry. It is built alongside the other numerical transformations in `a4` through `a8`, and the medians come from the training split alone, so no validation or test row contributes to them.

## Modeling Workflow

The project uses a three-stage pipeline to ensure rigorous evaluation and avoid data leakage.

### Phase 1: Training
Dataset: 80% of training data (1,168 records)

* Baseline: OLS Linear Regression.
* Machine Learning: L1/L2 regression, tree-based models, and gradient boosting (XGBoost, LightGBM, CatBoost).
* Tuning: 10-fold cross-validation throughout, by grid search for the linear, kernel and tree models, and by grid search plus Bayesian optimization for XGBoost and LightGBM. CatBoost is left at library defaults.
* Both booster grid searches score candidates with early-stopped cross-validation rather than a fixed tree count. A smaller `learning_rate` needs more trees to reach the same fit, so each candidate is given the number of rounds it needs; one shared count would reward whichever rate converges soonest instead of the best configuration. Candidates are dispatched to separate processes, since at this data size most of a boosting round is Python overhead inside the CV loop and the GIL allows only one thread to run Python at a time.
* RBF SVR searches gamma as multiples of scikit-learn's `scale` heuristic, which keeps the grid anchored to the data's variance rather than to absolute values. `tol` is pinned at its tightest setting rather than searched, since it is a convergence threshold and not a hyperparameter.
* The boosters train on the full feature set. The `models/selected_features_*.txt` files are written for inspection only; nothing reads them back.

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

Results are reported two ways per model: `oof_rmse` over the 1,168 out-of-fold training rows and `val_rmse` over the 292 validation rows. Read the out-of-fold column when comparing small differences, since four times the rows means roughly half the standard error; the validation column is kept because it is the only figure measured on rows no base model saw in any fold. The stack's own out-of-fold figure comes from refitting the meta-learner ten times on nine-tenths of the out-of-fold matrix, because scoring the weights on the same rows that produced them would be in-sample.

## Results

Submitted to Kaggle's [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition as **dunkindonuts**, a one-person team.

* Public [leaderboard](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/leaderboard) score: **0.11745** root mean squared log error.
* Approximately 100th out of 3,754 teams, or the top 3%, as of August 2026.

## Further Improvements

**Fitted preprocessing lives outside the cross-validation folds.** The `StandardScaler` in the regression, SVR and KNN prep scripts, the `IterativeImputer` that fills `LotFrontage`, and the neighbourhood medians behind `Area_vs_Nbhd` are each fit once on the full 1,168-row training split, before any fold splitting happens. So when `a8_stacking.py` trains a fold on 90% of those rows, the features it sees were scaled and imputed using statistics that included the held-out tenth. The out-of-fold RMSE is therefore slightly optimistic in absolute terms.

The effect is small and mostly harmless for the way these numbers are used. Scaling and imputation statistics barely move when a tenth of the rows are dropped, and the bias applies to every base model about equally, so neither the model ranking nor the non-negative least squares weights are meaningfully distorted. It also does not touch the Kaggle score, where fitting the transformers on train and applying them to test is the correct procedure.

Fixing it properly would mean moving every fitted transformation inside the fold loop, so that each fold derives its own scaler, imputer and neighbourhood medians. That reshapes the boundary between `s1_data` and `s2_model`: the prep scripts would have to become reusable transformers rather than scripts that write finished tables to DuckDB. That is a large restructuring for a small correction to a diagnostic number, so it is skipped for now.

**The booster searches could move to Optuna.** The Bayesian tuning in `a5_xgb.py` and `a6_lgbm.py` uses `bayes_opt`, which optimizes over continuous bounds only. Integer parameters such as `max_depth`, `num_leaves` and `min_child_samples` are therefore declared as ranges and rounded inside the objective, so the surrogate fits a smooth surface over what is really a staircase and spends trials separating values that train identical models. Optuna's `suggest_int` treats them as genuinely discrete.

Two smaller gains come with it. An Optuna trial can carry the boosting round its early-stopped CV selected as an attribute, so the round count is read off the best trial instead of being recovered by a second CV call at the winning parameters. Its TPE sampler also avoids re-proposing points it has already evaluated, which `bayes_opt` does occasionally. Changing sampler changes the search path, so the two Bayesian models would need to be re-measured rather than assumed to improve.

**Models are pickled rather than saved in framework-native formats.** Every base model except CatBoost is written with `pickle`, which is kept for simplicity here: `s1` through `s3` run end to end in one environment, so the artifacts never cross a version or machine boundary. It is not what production would want. Pickle serializes the Python object graph rather than the model, so loading depends on the estimator class still existing in a compatible library version, and it executes code by design. The native formats — `.ubj` or `.json` for XGBoost, `.txt` for LightGBM, `.cbm` for CatBoost — store only the trees, splits and leaf values under a documented schema, load across versions and languages, and carry no executable payload.

The natural place to adopt them is `s4_prediction`, where the surviving base models are refit on train plus val. Those are the deployable artifacts, distinct from the `s2` models fit on the training split alone, and the script currently discards them after predicting. The switch would only ever be partial, since the scikit-learn estimators have no native format and a mixed bundle is unavoidable short of an ONNX conversion. Writing the library versions alongside the exported files would do more for reloading them safely than the format change on its own.

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
uv run python -m s1_data          # Run all data prep scripts (a0-a8)
uv run python -m s2_model         # Run all modeling scripts (a1-a8)
uv run python -m s4_prediction    # Run final predictions
```

To run an individual script within a stage:

```bash
uv run python -m s1_data.a2_contextual_imputation
uv run python -m s2_model.a7_catboost
```

## Project Structure

* `s0_eda/` — Exploratory data analysis and visualization.
* `s1_data/` — Data loading and DuckDB-based processing.
* `s2_model/` — Model definitions, training scripts, and tuning.
* `s3_validation/` — Shared evaluation helpers (imported by other modules).
* `s4_prediction/` — Final prediction generation.