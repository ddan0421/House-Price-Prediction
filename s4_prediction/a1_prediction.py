import os
import pickle
import warnings

import catboost as cb
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone

from s1_data.db_utils import load_df

warnings.filterwarnings("ignore", category=UserWarning)

base_folder = "data"
database = "AmesHousePrice.duckdb"
database_path = os.path.join(base_folder, database)
conn = duckdb.connect(database=database_path, read_only=False)

random_state = 42

################################################# Final Prediction #######################################################
"""
Pipeline:
1. Load the list of surviving base models from a8 elimination
   (models/meta_learner_active_models.txt)
2. Load each base model's train, val, and test data from duckdb
   - Feature-selected test matrices are saved in a1..a7 (e.g. test_reg_lr, test_xgb)
   - Tree models (a4) use full test_ml from s1_data (no extra save)
3. Concatenate train + val for the final fit, then refit each surviving base
   model (using the loaded pkl as a hyperparameter template via sklearn.clone)
4. Predict on test with each refit base model
5. Pass the test base predictions through the saved OLS meta-learner
6. Inverse log-transform and emit data/submission.csv keyed on real Kaggle Id
"""

# -------------------- Surviving models from elimination --------------------
with open("models/meta_learner_active_models.txt") as f:
    active_models = [line.strip() for line in f if line.strip()]
print(f"Active models for stacking ({len(active_models)}): {active_models}")


# -------------------- Targets --------------------
y_train = load_df(conn, "y_train").values.ravel()
y_val = load_df(conn, "y_val").values.ravel()
y_full = np.concatenate([y_train, y_val])


# -------------------- Per-model train / val / test DataFrames --------------------
# Linear regressors (Ridge / Lasso / ElasticNet) — test_reg_lr saved by a1 (lr_features only)
X_train_reg = load_df(conn, "X_train_reg_lr")
X_val_reg = load_df(conn, "X_val_reg_lr")
test_reg = load_df(conn, "test_reg_lr")

# RBF SVR — saved by a2
X_train_svr_rbf = load_df(conn, "X_train_svr_rbf")
X_val_svr_rbf = load_df(conn, "X_val_svr_rbf")
test_svr_rbf = load_df(conn, "test_svr_rbf")

# Linear SVR — saved by a2
X_train_linear_svr = load_df(conn, "X_train_linear_svr")
X_val_linear_svr = load_df(conn, "X_val_linear_svr")
test_linear_svr = load_df(conn, "test_linear_svr")

# KNN — saved by a3
X_train_knn = load_df(conn, "X_train_knn_final")
X_val_knn = load_df(conn, "X_val_knn_final")
test_knn = load_df(conn, "test_knn_final")

# Trees (DT / RF / ET) — full test_ml from s1_data only (a4 has no feature selection / no test save)
X_train_ml = load_df(conn, "X_train_ml")
X_val_ml = load_df(conn, "X_val_ml")
test_ml = load_df(conn, "test_ml")

# XGB (GridSearch + Bayesian) — saved by a5
X_train_xgb = load_df(conn, "X_train_xgb")
X_val_xgb = load_df(conn, "X_val_xgb")
test_xgb = load_df(conn, "test_xgb")

# Categorical column lists (matches a6/a7)
nominal_cat = [
    "MSSubClass_MSZoning", "LotConfig_LandSlope", "Neighborhood_Condition", "BldgType_HouseStyle",
    "Exterior1st_Exterior2nd", "CentralAir_Electrical", "LotShape_LandContour", "RoofStyle_RoofMatl",
    "Heating_HeatingQC", "Alley", "MasVnrType", "Foundation", "GarageType", "PavedDrive", "Fence",
    "MiscFeature", "SaleType", "SaleCondition", "Season_Sold",
]
ordinal_cat = [
    "Utilities", "Functional", "OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtQual",
    "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "KitchenQual", "FireplaceQu",
    "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Street",
]
all_cat_columns = nominal_cat + ordinal_cat

# LightGBM (GridSearch + Bayesian) - saved by a6
# Note: we cast to category AFTER concatenating train + val below. Casting
# beforehand would coerce the columns back to object on concat whenever train
# and val carry different category levels, which LightGBM rejects with
# "pandas dtypes must be int, float or bool".
X_train_lgbm = load_df(conn, "X_train_lgbm")
X_val_lgbm = load_df(conn, "X_val_lgbm")
test_lgbm = load_df(conn, "test_lgbm")
lgbm_cat_columns = [c for c in X_train_lgbm.columns if c in all_cat_columns]

# CatBoost — same columns as X_train_cat; test_cat from s1_data (a9 / cat prep), no subsetting
X_train_cat = load_df(conn, "X_train_cat")
X_val_cat = load_df(conn, "X_val_cat")
test_cat = load_df(conn, "test_cat")
cat_cat_columns = [c for c in X_train_cat.columns if c in all_cat_columns]


# -------------------- Combine train + val for the final fit --------------------
def _combine(a, b):
    return pd.concat([a, b], axis=0, ignore_index=True)


X_full_reg = _combine(X_train_reg, X_val_reg)
X_full_svr_rbf = _combine(X_train_svr_rbf, X_val_svr_rbf)
X_full_linear_svr = _combine(X_train_linear_svr, X_val_linear_svr)
X_full_knn = _combine(X_train_knn, X_val_knn)
X_full_ml = _combine(X_train_ml, X_val_ml)
X_full_xgb = _combine(X_train_xgb, X_val_xgb)
X_full_lgbm = _combine(X_train_lgbm, X_val_lgbm)
X_full_cat = _combine(X_train_cat, X_val_cat)

# LightGBM cast: do it now so each cat column gets the union of categories from
# train+val. Then align the test set's categories to the train+val cats so
# unseen test levels become NaN (LightGBM handles them) and the dtype matches.
for col in lgbm_cat_columns:
    X_full_lgbm[col] = X_full_lgbm[col].astype("category")
    test_lgbm[col] = pd.Categorical(
        test_lgbm[col], categories=X_full_lgbm[col].cat.categories
    )


# -------------------- Per-model train+val and test registry --------------------
# Each entry: (full_X, test_X)
DATA = {
    "xgb":         (X_full_xgb,        test_xgb),
    "xgb_bayes":   (X_full_xgb,        test_xgb),
    "ridge":       (X_full_reg,        test_reg),
    "lasso":       (X_full_reg,        test_reg),
    "enet":        (X_full_reg,        test_reg),
    "svr_rbf":     (X_full_svr_rbf,    test_svr_rbf),
    "svr_linear":  (X_full_linear_svr, test_linear_svr),
    "knn":         (X_full_knn,        test_knn),
    "dt":          (X_full_ml,         test_ml),
    "rf":          (X_full_ml,         test_ml),
    "et":          (X_full_ml,         test_ml),
    "lgbm":        (X_full_lgbm,       test_lgbm),
    "lgbm_bayes":  (X_full_lgbm,       test_lgbm),
    "cat_basic":   (X_full_cat,        test_cat),
}


# -------------------- Load pkl base models (used as hyperparameter templates) --------------------
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


pkl_paths = {
    "xgb":         "models/final_model_xgb.pkl",
    "xgb_bayes":   "models/final_model_xgb_bayes.pkl",
    "ridge":       "models/final_model_ridge.pkl",
    "lasso":       "models/final_model_lasso.pkl",
    "enet":        "models/final_model_enet.pkl",
    "svr_rbf":     "models/final_model_svr_rbf.pkl",
    "svr_linear":  "models/final_model_linear_svr.pkl",
    "knn":         "models/final_model_knn.pkl",
    "dt":          "models/final_model_dt.pkl",
    "rf":          "models/final_model_rf.pkl",
    "et":          "models/final_model_et.pkl",
    "lgbm":        "models/final_model_lgbm.pkl",
    "lgbm_bayes":  "models/final_model_lgbm_bayes.pkl",
}

models_pkl = {name: load_pkl(path) for name, path in pkl_paths.items() if name in active_models}


# -------------------- Refit each surviving base model on train + val --------------------
print(f"\nRefitting {len(active_models)} base models on train + val...")
trained = {}
for name in active_models:
    X_full, _ = DATA[name]
    if name == "cat_basic":
        model = cb.CatBoostRegressor(
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
            cat_features=cat_cat_columns,
            allow_writing_files=False,
        )
        model.fit(X_full, y_full)
    elif name in ("lgbm", "lgbm_bayes"):
        model = clone(models_pkl[name])
        model.fit(X_full, y_full, categorical_feature=lgbm_cat_columns)
    else:
        model = clone(models_pkl[name])
        model.fit(X_full, y_full)
    trained[name] = model
    print(f"  Refit '{name}' done.")


# -------------------- Predict on test with each base model --------------------
n_test = int(conn.execute("SELECT COUNT(*) FROM test").fetchone()[0])
test_preds = np.zeros((n_test, len(active_models)))
for i, name in enumerate(active_models):
    _, X_test = DATA[name]
    test_preds[:, i] = np.asarray(trained[name].predict(X_test)).ravel()


# -------------------- Apply OLS meta-learner --------------------
with open("models/meta_learner_ols.pkl", "rb") as f:
    meta_learner_ols = pickle.load(f)

test_preds_df = pd.DataFrame(test_preds, columns=active_models)
test_preds_const = sm.add_constant(test_preds_df, has_constant="add")
final_preds_log = np.asarray(meta_learner_ols.predict(test_preds_const)).ravel()


# -------------------- Build submission with real Kaggle Id --------------------
"""
The duckdb tables produced by save_df use a synthetic Id (range(len(df))) that
load_df excludes. The real Kaggle test Id is only kept in the raw `test` table
(loaded from CSV in s1_data/a1_load_raw_data.py). Both orderings are sorted
by Id, so positional alignment holds.
"""
test_ids = conn.execute("SELECT Id FROM test ORDER BY Id").fetch_df()["Id"]

submission = pd.DataFrame({
    "Id": test_ids.values,
    "SalePrice": np.exp(final_preds_log),
})

os.makedirs("data", exist_ok=True)
submission.to_csv("data/submission.csv", index=False)
print(f"\nSubmission saved to data/submission.csv ({len(submission)} rows)")

conn.close()
