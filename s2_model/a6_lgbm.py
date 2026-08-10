import itertools
import os
import pickle
import numpy as np
import duckdb
import lightgbm as lgb
from bayes_opt import BayesianOptimization, acquisition
from joblib import Parallel, delayed

from s1_data.db_utils import load_df, save_df
from s3_validation.model_evaluation import evaluate_model

base_folder = "data"
database = "AmesHousePrice.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=False)

random_state = 42
seed = 42

X_train_cat = load_df(conn, "X_train_cat")
X_val_cat = load_df(conn, "X_val_cat")
test_cat = load_df(conn, "test_cat")
y_train = load_df(conn, "y_train")
y_val = load_df(conn, "y_val")


nominal_cat = ["MSSubClass_MSZoning", "LotConfig_LandSlope", "Neighborhood_Condition", "BldgType_HouseStyle",
               "Exterior1st_Exterior2nd", "CentralAir_Electrical", "LotShape_LandContour", "RoofStyle_RoofMatl",
               "Heating_HeatingQC", "Alley", "MasVnrType", "Foundation", "GarageType", "PavedDrive", "Fence", 
               "MiscFeature", "SaleType", "SaleCondition", "Season_Sold"]

ordinal_cat = ["Utilities", "Functional", "OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtQual", 
               "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "KitchenQual", "FireplaceQu", 
               "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Street"]


all_cat_columns = nominal_cat + ordinal_cat

cat_columns = [f for f in X_train_cat.columns if f in all_cat_columns]

X_train_cat[cat_columns] = X_train_cat[cat_columns].astype("category")
X_val_cat[cat_columns] = X_val_cat[cat_columns].astype("category")
test_cat[cat_columns] = test_cat[cat_columns].astype("category")


X_train_lgbm = X_train_cat.copy()
X_val_lgbm = X_val_cat.copy()
test_lgbm = test_cat.copy()


############################################## LightGBM Regressor Model ############################################################
max_boost_rounds = 5000


def lgbm_cv(params, n_jobs=-1):
    """
    Return (best CV RMSE, boosting round that achieved it) for one parameter set.

    The Dataset is built per call so candidates can be scored concurrently without sharing
    one across parallel lgb.cv runs.
    """
    dtrain_lgbm = lgb.Dataset(data=X_train_lgbm, label=y_train.values.ravel(),
                              categorical_feature=cat_columns, free_raw_data=False)
    cv_results = lgb.cv(
        {**params,
         "objective": "regression",
         "metric": "rmse",
         "verbosity": -1,
         "feature_pre_filter": False,
         "seed": seed,
         "n_jobs": n_jobs,
         "boosting_type": "gbdt"},
        dtrain_lgbm,
        num_boost_round=max_boost_rounds,
        nfold=10,
        seed=seed,
        shuffle=True,
        stratified=False,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                   lgb.log_evaluation(0)],
    )
    curve = np.array(cv_results["valid rmse-mean"])
    return curve.min(), int(curve.argmin()) + 1


# Scored with the same early-stopped CV as the Bayesian search below. A smaller learning_rate needs
# more trees to reach the same fit, so early stopping gives each candidate the number of rounds it
# needs. One fixed count for all of them would just reward the fastest-learning rate.
grid_fixed = {"subsample": 0.85, "subsample_freq": 1, "colsample_bytree": 0.85,
              "reg_alpha": 0.0, "min_split_gain": 0.0}
param_grid = {
    "learning_rate": [0.03, 0.05],
    "num_leaves": [4, 8, 16, 31],
    "min_child_samples": [10, 15, 20],
    "reg_lambda": [1.1, 3.0],
}

candidates = [{**grid_fixed, **dict(zip(param_grid, values))}
              for values in itertools.product(*param_grid.values())]

# Candidates run in separate processes, each with n_jobs=1. With so few rows, most of a boosting
# round is Python overhead inside lgb.cv rather than C++ work, and the GIL lets only one thread run
# Python at a time, so a process per candidate is what actually keeps the cores busy.
scores = Parallel(n_jobs=-1)(
    delayed(lgbm_cv)(candidate, 1) for candidate in candidates)

best_grid_rmse, best_grid_rounds, best_grid_params = min(
    ((rmse, rounds, candidate) for (rmse, rounds), candidate in zip(scores, candidates)),
    key=lambda item: item[0])

print("10-Fold CV RMSE:", best_grid_rmse)
print("Optimal Parameters:", best_grid_params)
print("Best boosting round from CV:", best_grid_rounds)

final_model_lgbm = lgb.LGBMRegressor(
    **best_grid_params,
    n_estimators=best_grid_rounds,
    objective="regression",
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)
final_model_lgbm.fit(X_train_lgbm, y_train.values.ravel(), categorical_feature=cat_columns)
print("Optimal Estimator:", final_model_lgbm)

selected_features_lgbm = X_train_lgbm.columns[np.array(final_model_lgbm.feature_importances_) > 0]

# Diagnostic only -- the model above trains on all features, and nothing reads this file.
with open("models/selected_features_lgbm.txt", "w") as f:
    for feat in selected_features_lgbm:
        f.write(f"{feat}\n")


# Save the trained model for future use (stacking)
with open("models/final_model_lgbm.pkl", "wb") as f:
    pickle.dump(final_model_lgbm, f)
print("lgbm model saved to models/final_model_lgbm.pkl")




############################################## LGBM Models with Bayesian Optimization ############################################################
def tuned_params(learning_rate, num_leaves, min_child_samples, reg_alpha, reg_lambda,
                 colsample_bytree, subsample, subsample_freq):
    """Map raw BayesianOptimization output onto valid LightGBM parameters."""
    return {
        "learning_rate": learning_rate,
        "num_leaves": int(round(num_leaves)),
        "min_child_samples": int(round(min_child_samples)),
        "reg_alpha": max(reg_alpha, 0),
        "reg_lambda": max(reg_lambda, 0),
        "colsample_bytree": max(min(colsample_bytree, 1), 0),
        "subsample": max(min(subsample, 1), 0),
        "subsample_freq": int(round(subsample_freq)),
    }


def bayesian_opt_lgbm(init_iter=20, n_iters=80, random_state=random_state):
    def hyp_lgbm(learning_rate, num_leaves, min_child_samples, reg_alpha, reg_lambda,
                 colsample_bytree, subsample, subsample_freq):
        best_rmse, _ = lgbm_cv(tuned_params(
            learning_rate, num_leaves, min_child_samples, reg_alpha, reg_lambda,
            colsample_bytree, subsample, subsample_freq))
        return -best_rmse

    pds = {
        "learning_rate": (0.005, 0.3),
        "num_leaves": (4, 200),
        "min_child_samples": (5, 50),
        "reg_alpha": (0, 5),
        "reg_lambda": (0, 5),
        "colsample_bytree": (0.4, 1.0),
        "subsample": (0.5, 1.0),
        "subsample_freq": (1, 7),
    }

    acq = acquisition.UpperConfidenceBound(
        kappa=1.0,
        exploration_decay=0.95,
        exploration_decay_delay=init_iter,
        random_state=random_state,
    )

    optimizer = BayesianOptimization(f=hyp_lgbm, pbounds=pds,
                                     acquisition_function=acq,
                                     random_state=random_state)
    optimizer.maximize(init_points=init_iter, n_iter=n_iters)
    return optimizer


results = bayesian_opt_lgbm()
print("Best Parameters:", results.max["params"])

# Re-run the winning configuration so the round count comes from the same call as the
# score. BayesianOptimization only returns a scalar, so the alternative is caching each
# call's round count under a key rebuilt from the params, which silently mismatches.
best_params = tuned_params(**results.max["params"])
best_rmse, best_n_estimators = lgbm_cv(best_params)
print("Best RMSE Score:", best_rmse)
print(f"Best boosting round from CV: {best_n_estimators}")

best_params.update({
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "feature_pre_filter": False,
    "n_jobs": -1,
    "random_state": random_state,
    "boosting_type": "gbdt",
    "n_estimators": best_n_estimators,
})

lgbm_bayes_model = lgb.LGBMRegressor(**best_params)
lgbm_bayes_model.fit(X_train_lgbm, y_train.values.ravel(), categorical_feature=cat_columns)

with open("models/final_model_lgbm_bayes.pkl", "wb") as f:
    pickle.dump(lgbm_bayes_model, f)
print("LGBM Bayes model saved to models/final_model_lgbm_bayes.pkl")



save_df(conn, X_train_lgbm, "X_train_lgbm")
save_df(conn, X_val_lgbm, "X_val_lgbm")
save_df(conn, test_lgbm, "test_lgbm")


evaluate_model(final_model_lgbm, X_val_lgbm, y_val, "LGBM (GridSearch)")
evaluate_model(lgbm_bayes_model, X_val_lgbm, y_val, "LGBM (Bayes Opt)")

conn.close()