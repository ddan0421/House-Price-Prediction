import os
import pickle
import numpy as np
import duckdb
from sklearn.model_selection import GridSearchCV, KFold
import xgboost as xgb
from bayes_opt import BayesianOptimization

from s1_data.db_utils import load_df, save_df
from s3_validation.model_evaluation import evaluate_model

base_folder = "data"
database = "AmesHousePrice.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=False)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

random_state = 42
seed = 42

X_train_tree_raw = load_df(conn, "X_train_ml")
X_val_tree_raw = load_df(conn, "X_val_ml")
test_tree_raw = load_df(conn, "test_ml")
y_train = load_df(conn, "y_train")
y_val = load_df(conn, "y_val")

############################################## XGBoost Regressor Model ############################################################

X_train_xgb = X_train_tree_raw.copy()
X_val_xgb = X_val_tree_raw.copy()
test_xgb = test_tree_raw.copy()
                           
xgb_model = xgb.XGBRegressor(
    random_state=random_state, 
    objective="reg:squarederror",
    n_jobs=1,
    tree_method="hist",
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.75)

param_grid = {
    "learning_rate": [0.05, 0.07, 0.10], 
    "max_depth": [2, 3, 4],  
    "min_child_weight": [2, 3], 
    "reg_alpha": [0, 0.5],  
    "reg_lambda": [0.281, 1, 3]  
}

gs_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True)

gs_xgb.fit(X_train_xgb, y_train.values.ravel())

print("10-Fold CV RMSE:", -gs_xgb.best_score_)  
print("Optimal Parameters:", gs_xgb.best_params_)
print("Optimal Estimator:", gs_xgb.best_estimator_)

final_model_xgb = gs_xgb.best_estimator_

selected_features_xgb = X_train_xgb.columns[np.array(final_model_xgb.feature_importances_) > 0]

# Diagnostic only -- the model above trains on all features, and nothing reads this file.
with open("models/selected_features_xgb.txt", "w") as f:
    for feat in selected_features_xgb:
        f.write(f"{feat}\n")

# Save the trained model for future use (stacking)
with open("models/final_model_xgb.pkl", "wb") as f:
    pickle.dump(final_model_xgb, f)
print("xgboost model saved to models/final_model_xgb.pkl")


############################################## XGB Models with Bayesian Optimization ############################################################
max_boost_rounds = 5000

dtrain_xgb = xgb.DMatrix(data=X_train_xgb, label=y_train.values.ravel())


def tuned_params(learning_rate, max_depth, min_child_weight, subsample,
                 colsample_bytree, reg_alpha, reg_lambda, gamma):
    """Map raw BayesianOptimization output onto valid XGBoost parameters."""
    return {
        "learning_rate": learning_rate,
        "max_depth": int(round(max_depth)),
        "min_child_weight": min_child_weight,
        "subsample": max(min(subsample, 1), 0),
        "colsample_bytree": max(min(colsample_bytree, 1), 0),
        "reg_alpha": max(reg_alpha, 0),
        "reg_lambda": max(reg_lambda, 0),
        "gamma": max(gamma, 0),
    }


def xgb_cv(params):
    """Return (best CV RMSE, boosting round that achieved it) for one parameter set."""
    cv_results = xgb.cv(
        {**params,
         "objective": "reg:squarederror",
         "eval_metric": "rmse",
         "seed": seed,
         "n_jobs": -1,
         "booster": "gbtree"},
        dtrain_xgb,
        num_boost_round=max_boost_rounds,
        nfold=10,
        seed=seed,
        shuffle=True,
        stratified=False,
        early_stopping_rounds=50,
        metrics="rmse",
    )
    curve = cv_results["test-rmse-mean"]
    return curve.min(), int(curve.idxmin()) + 1


def bayesian_opt_xgb(init_iter=10, n_iters=40, random_state=random_state):
    def hyp_xgb(learning_rate, max_depth, min_child_weight, subsample,
                colsample_bytree, reg_alpha, reg_lambda, gamma):
        best_rmse, _ = xgb_cv(tuned_params(
            learning_rate, max_depth, min_child_weight, subsample,
            colsample_bytree, reg_alpha, reg_lambda, gamma))
        return -best_rmse

    pds = {
        "learning_rate": (0.005, 0.3),
        "max_depth": (2, 10),
        "min_child_weight": (1, 10),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.4, 1.0),
        "reg_alpha": (0, 5),
        "reg_lambda": (0, 5),
        "gamma": (0, 5),
    }

    optimizer = BayesianOptimization(f=hyp_xgb, pbounds=pds, random_state=random_state)
    optimizer.maximize(init_points=init_iter, n_iter=n_iters)
    return optimizer


results = bayesian_opt_xgb()
print("Best Parameters:", results.max["params"])

# Re-run the winning configuration so the round count comes from the same call as the
# score. BayesianOptimization only returns a scalar, so the alternative is caching each
# call's round count under a key rebuilt from the params, which silently mismatches.
best_params = tuned_params(**results.max["params"])
best_rmse, best_n_estimators = xgb_cv(best_params)
print("Best RMSE Score:", best_rmse)
print(f"Best boosting round from CV: {best_n_estimators}")

best_params.update({
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_jobs": -1,
    "random_state": random_state,
    "booster": "gbtree",
    "n_estimators": best_n_estimators,
})

xgb_bayes_model = xgb.XGBRegressor(**best_params)
xgb_bayes_model.fit(X_train_xgb, y_train.values.ravel())

with open("models/final_model_xgb_bayes.pkl", "wb") as f:
    pickle.dump(xgb_bayes_model, f)
print("XGB Bayes model saved to models/final_model_xgb_bayes.pkl")


save_df(conn, X_train_xgb, "X_train_xgb")
save_df(conn, X_val_xgb, "X_val_xgb")
save_df(conn, test_xgb, "test_xgb")


evaluate_model(final_model_xgb, X_val_xgb, y_val, "XGBoost (GridSearch)")
evaluate_model(xgb_bayes_model, X_val_xgb, y_val, "XGBoost (Bayes Opt)")

conn.close()