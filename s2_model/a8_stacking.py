import os
import json
import pickle
import warnings

import catboost as cb
import duckdb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

from s1_data.db_utils import load_df

warnings.filterwarnings("ignore", category=UserWarning)

base_folder = "data"
database = "AmesHousePrice.duckdb"
database_path = os.path.join(base_folder, database)
conn = duckdb.connect(database=database_path, read_only=False)

# Not the seed the base models were tuned with. Reusing it would test each model on
# the same folds it was tuned to win, making it look better than it is.
oof_seed = 7

# -------------------- Targets --------------------
y_train = load_df(conn, "y_train").values.ravel()
y_val = load_df(conn, "y_val").values.ravel()

# -------------------- Per-model train/val DataFrames --------------------
X_train_reg = load_df(conn, "X_train_reg_lr")
X_val_reg = load_df(conn, "X_val_reg_lr")

X_train_svr_rbf = load_df(conn, "X_train_svr_rbf")
X_val_svr_rbf = load_df(conn, "X_val_svr_rbf")

X_train_linear_svr = load_df(conn, "X_train_linear_svr")
X_val_linear_svr = load_df(conn, "X_val_linear_svr")

X_train_knn = load_df(conn, "X_train_knn_final")
X_val_knn = load_df(conn, "X_val_knn_final")

X_train_ml = load_df(conn, "X_train_ml")
X_val_ml = load_df(conn, "X_val_ml")

X_train_xgb = load_df(conn, "X_train_xgb")
X_val_xgb = load_df(conn, "X_val_xgb")

# Categorical column lists (matches a6/a7 nominal_cat + ordinal_cat)
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

# LightGBM expects categorical dtype (lost in duckdb roundtrip, so re-cast)
X_train_lgbm = load_df(conn, "X_train_lgbm")
X_val_lgbm = load_df(conn, "X_val_lgbm")
lgbm_cat_columns = [c for c in X_train_lgbm.columns if c in all_cat_columns]
X_train_lgbm[lgbm_cat_columns] = X_train_lgbm[lgbm_cat_columns].astype("category")
X_val_lgbm[lgbm_cat_columns] = X_val_lgbm[lgbm_cat_columns].astype("category")

# CatBoost uses the full raw cat-encoded train table
X_train_cat = load_df(conn, "X_train_cat")
X_val_cat = load_df(conn, "X_val_cat")
cat_cat_columns = [c for c in X_train_cat.columns if c in all_cat_columns]


# -------------------- Load trained base models --------------------
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


xgb_model = load_pkl("models/final_model_xgb.pkl")
xgb_bayes_model = load_pkl("models/final_model_xgb_bayes.pkl")
ridge_model = load_pkl("models/final_model_ridge.pkl")
lasso_model = load_pkl("models/final_model_lasso.pkl")
enet_model = load_pkl("models/final_model_enet.pkl")
svr_rbf_model = load_pkl("models/final_model_svr_rbf.pkl")
linear_svr_model = load_pkl("models/final_model_linear_svr.pkl")
knn_model = load_pkl("models/final_model_knn.pkl")
dt_model = load_pkl("models/final_model_dt.pkl")
rf_model = load_pkl("models/final_model_rf.pkl")
et_model = load_pkl("models/final_model_et.pkl")
lgbm_model = load_pkl("models/final_model_lgbm.pkl")
lgbm_bayes_model = load_pkl("models/final_model_lgbm_bayes.pkl")

cat_basic_model = cb.CatBoostRegressor()
cat_basic_model.load_model("models/final_model_catboost_basic.cbm")

with open("models/catboost_best_params.json") as f:
    cat_params = json.load(f)


# -------------------- Base model registry --------------------
# Each entry: (name, fitted_model, X_train_df, X_val_df)
base_models = [
    ("xgb",         xgb_model,        X_train_xgb,        X_val_xgb),
    ("xgb_bayes",   xgb_bayes_model,  X_train_xgb,        X_val_xgb),
    ("ridge",       ridge_model,      X_train_reg,        X_val_reg),
    ("lasso",       lasso_model,      X_train_reg,        X_val_reg),
    ("enet",        enet_model,       X_train_reg,        X_val_reg),
    ("svr_rbf",     svr_rbf_model,    X_train_svr_rbf,    X_val_svr_rbf),
    ("svr_linear",  linear_svr_model, X_train_linear_svr, X_val_linear_svr),
    ("knn",         knn_model,        X_train_knn,        X_val_knn),
    ("dt",          dt_model,         X_train_ml,         X_val_ml),
    ("rf",          rf_model,         X_train_ml,         X_val_ml),
    ("et",          et_model,         X_train_ml,         X_val_ml),
    ("lgbm",        lgbm_model,       X_train_lgbm,       X_val_lgbm),
    ("lgbm_bayes",  lgbm_bayes_model, X_train_lgbm,       X_val_lgbm),
    ("cat_basic",   cat_basic_model,  X_train_cat,        X_val_cat),
]


# -------------------- Out-of-fold predictions --------------------
n_train = len(y_train)
oof_preds = np.zeros((n_train, len(base_models)))
kf = KFold(n_splits=10, shuffle=True, random_state=oof_seed)

"""
Each fold fits a clone of the loaded base model on train_idx and predicts val_idx.
cat_basic is the exception: it is rebuilt from cat_params rather than cloned. The
loaded models themselves stay untouched, so the val predictions further down still
use the full-training-set fit from a1..a7.
"""
for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_train))):
    print(f"Processing Fold {fold + 1}...")
    for i, (name, model, X_tr, _) in enumerate(base_models):
        X_fold_tr = X_tr.iloc[train_idx]
        X_fold_va = X_tr.iloc[val_idx]
        y_fold_tr = y_train[train_idx]

        if name == "cat_basic":
            fold_model = cb.CatBoostRegressor(
                **cat_params,
                verbose=False,
                cat_features=cat_cat_columns,
                allow_writing_files=False,
            )
            fold_model.fit(X_fold_tr, y_fold_tr)
        elif name in ("lgbm", "lgbm_bayes"):
            fold_model = clone(model)
            fold_model.fit(X_fold_tr, y_fold_tr, categorical_feature=lgbm_cat_columns)
        else:
            fold_model = clone(model)
            fold_model.fit(X_fold_tr, y_fold_tr)

        oof_preds[val_idx, i] = np.asarray(fold_model.predict(X_fold_va)).ravel()

all_names = [name for name, *_ in base_models]



# -------------------- Meta-learner: non-negative least squares --------------------
oof_x = pd.DataFrame(oof_preds, columns=all_names)


def fit_meta(x, y):
    """Non-negative weights with a free intercept."""
    return LinearRegression(positive=True).fit(x, y)


meta_learner = fit_meta(oof_x, y_train)

weights = pd.Series(meta_learner.coef_, index=all_names)
active = [name for name in all_names if weights[name] > 1e-6]
dropped = [name for name in all_names if name not in active]

print(f"\n[Meta-learner] NNLS kept {len(active)} of {len(all_names)} base models")
for name in sorted(active, key=lambda n: -weights[n]):
    print(f"  {name:<12} {weights[name]:.4f}")
print(f"  {'intercept':<12} {meta_learner.intercept_:.4f}")
print(f"  {'weight sum':<12} {weights.sum():.4f}")
print(f"[Meta-learner] Zero weight: {dropped}")


# -------------------- RMSE comparison --------------------
val_preds = np.zeros((len(y_val), len(base_models)))
for i, (_, model, _, X_va) in enumerate(base_models):
    val_preds[:, i] = np.asarray(model.predict(X_va)).ravel()

val_x = pd.DataFrame(val_preds, columns=all_names)

stack_val_rmse = root_mean_squared_error(y_val, meta_learner.predict(val_x))
summary = pd.DataFrame(
    [(name, root_mean_squared_error(y_val, val_preds[:, i]))
     for i, name in enumerate(all_names)],
    columns=["model", "val_rmse"],
).sort_values("val_rmse").reset_index(drop=True)
summary.loc[len(summary)] = ["stack (nnls)", stack_val_rmse]

print("\n[Comparison] held-out RMSE on y_val")
print(summary.to_string(index=False, float_format=lambda v: f"{v:.5f}"))


# -------------------- Save for s4_prediction --------------------
# Weights are keyed by name so s4 stays correct regardless of column ordering.
with open("models/meta_learner_nnls.json", "w") as f:
    json.dump({"intercept": float(meta_learner.intercept_),
               "weights": {name: float(weights[name]) for name in active}}, f, indent=2)
print("Meta-learner saved to models/meta_learner_nnls.json")

with open("models/meta_learner_active_models.txt", "w") as f:
    for name in active:
        f.write(name + "\n")
print("Surviving model list saved to models/meta_learner_active_models.txt")

conn.close()
