import os
import pickle
import warnings

import catboost as cb
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

from s1_data.db_utils import load_df
from s2_model.models import sm_ols
from s3_validation.model_evaluation import evaluate_model

warnings.filterwarnings("ignore", category=UserWarning)

base_folder = "data"
database = "AmesHousePrice.duckdb"
database_path = os.path.join(base_folder, database)
conn = duckdb.connect(database=database_path, read_only=False)

random_state = 42

################################################# Stacking Models #######################################################
"""
Base models (and the duckdb table each was trained on in a1..a7):

Linear regressors (Ridge / Lasso / ElasticNet)
RBF SVR                                                          
Linear SVR                                   
KNN                                                      
Decision Tree / Random Forest / Extra Trees     
XGBoost (GridSearch + Bayesian)                                
LightGBM (GridSearch + Bayesian)                               
CatBoost (basic)                           
"""

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

cat_basic_model = cb.CatBoostRegressor(cat_features=cat_cat_columns)
cat_basic_model.load_model("models/final_model_catboost_basic.cbm")


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
kf = KFold(n_splits=10, shuffle=True, random_state=random_state)

"""
For each fold we fit a *clone* of the loaded base model on the train_idx subset
and predict on val_idx. The original loaded models stay untouched and are used
directly for val/test prediction below (where they should already be trained
on the full training set).
"""
for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_train))):
    print(f"Processing Fold {fold + 1}...")
    for i, (name, model, X_tr, _) in enumerate(base_models):
        X_fold_tr = X_tr.iloc[train_idx]
        X_fold_va = X_tr.iloc[val_idx]
        y_fold_tr = y_train[train_idx]

        if name == "cat_basic":
            fold_model = cb.CatBoostRegressor(
                loss_function="RMSE",
                random_seed=random_state,
                verbose=False,
                cat_features=cat_cat_columns,
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
oof_df = pd.DataFrame(oof_preds, columns=all_names)
oof_df["Target"] = y_train
print("OOF Predictions:")
print(oof_df)


# -------------------- Validation predictions for all base models --------------------
"""
The loaded base models were already fit on the full training set in a1..a7,
so we use them directly here (the OOF loop above used clones, so these are
still untouched).
"""
val_preds = np.zeros((len(y_val), len(base_models)))
for i, (name, model, _, X_va) in enumerate(base_models):
    val_preds[:, i] = np.asarray(model.predict(X_va)).ravel()


# -------------------- Recursive elimination by validation RMSE --------------------
"""
Greedy backward elimination over the set of base models.

At each step we refit the OLS meta-learner on every (active set minus one model)
and pick the removal that yields the lowest validation RMSE. If that removal
is no worse than the current RMSE (within TOLERANCE), we drop the model
permanently and continue. Otherwise we stop.

OOF predictions and val predictions for each base model are independent of
the meta-learner, so they're computed once above and re-sliced here -- no
need to retrain base models inside the elimination loop.
"""
TOLERANCE = 1e-4  # how much worse a candidate can be and still be "barely changed"
name_to_idx = {name: i for i, name in enumerate(all_names)}


def fit_meta_and_score(active_names):
    """Fit OLS meta-learner on the given active subset; return (val_rmse, meta_model)."""
    cols = [name_to_idx[n] for n in active_names]

    X_oof = pd.DataFrame(oof_preds[:, cols], columns=active_names)
    X_oof_const = sm.add_constant(X_oof, has_constant="add")
    meta = sm_ols(X_oof_const, y_train, verbose=False)

    X_val_meta = pd.DataFrame(val_preds[:, cols], columns=active_names)
    X_val_meta_const = sm.add_constant(X_val_meta, has_constant="add")
    preds = np.asarray(meta.predict(X_val_meta_const)).ravel()
    return root_mean_squared_error(y_val, preds), meta


active = list(all_names)
baseline_rmse, _ = fit_meta_and_score(active)
print(f"\n[Recursive Elimination] Baseline with all {len(active)} models: RMSE = {baseline_rmse:.6f}")

elimination_history = [(list(active), baseline_rmse)]

while len(active) > 1:
    candidate_results = []
    for name in active:
        candidate = [n for n in active if n != name]
        cand_rmse, _ = fit_meta_and_score(candidate)
        candidate_results.append((cand_rmse, name))

    candidate_results.sort()  # lowest RMSE first
    best_rmse, drop_name = candidate_results[0]

    if best_rmse <= baseline_rmse + TOLERANCE:
        delta = best_rmse - baseline_rmse
        sign = "+" if delta >= 0 else ""
        print(f"  Drop '{drop_name}'  RMSE: {baseline_rmse:.6f} -> {best_rmse:.6f} "
              f"({sign}{delta:.6f}); {len(active) - 1} models remain")
        active.remove(drop_name)
        baseline_rmse = best_rmse
        elimination_history.append((list(active), baseline_rmse))
    else:
        print(f"  Stop: best candidate removal ('{drop_name}' -> {best_rmse:.6f}) "
              f"is worse than current {baseline_rmse:.6f} by more than tolerance")
        break

print(f"\n[Recursive Elimination] Final RMSE: {baseline_rmse:.6f}")
print(f"[Recursive Elimination] Surviving {len(active)} models: {active}")


# -------------------- Refit + save final meta-learner --------------------
final_rmse, meta_learner_ols = fit_meta_and_score(active)
print(f"\nStacking Performance (after elimination):")
print(f"Root Mean Squared Error: {final_rmse:.4f}")

with open("models/meta_learner_ols.pkl", "wb") as f:
    pickle.dump(meta_learner_ols, f)
print("Meta-learner saved to models/meta_learner_ols.pkl")

with open("models/meta_learner_active_models.txt", "w") as f:
    for name in active:
        f.write(name + "\n")
print("Surviving model list saved to models/meta_learner_active_models.txt")

conn.close()
