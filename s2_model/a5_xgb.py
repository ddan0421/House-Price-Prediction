import os
import pickle
import numpy as np
import duckdb
from sklearn.model_selection import GridSearchCV, KFold
import xgboost as xgb
from bayes_opt import BayesianOptimization
import pickle

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

xgb_features = [
    "LotFrontage", "LotArea", "OverallQual", "OverallCond", "MasVnrArea", "BsmtFinSF1", 
    "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", 
    "BsmtFullBath", "FullBath", "HalfBath", "TotRmsAbvGrd", "Fireplaces", "GarageCars", 
    "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch", "Age_House", 
    "Yrs_Since_Remodel", "Age_Garage", "ExterQual_encoded", "BsmtQual_encoded", 
    "BsmtCond_encoded", "BsmtExposure_encoded", "BsmtFinType1_encoded", "KitchenQual_encoded", 
    "FireplaceQu_encoded", "GarageFinish_encoded", "GarageQual_encoded", "GarageCond_encoded", 
    "PoolQC_encoded", "Functional_encoded", "FinishedAreaPct", "Living_Rooms", "Garage_Space", 
    "Garage_AgeCars", "Ratio_Bedroom_Rooms", "Ratio_2ndFlr_Living", "MSSubClass_MSZoning_20_RL", 
    "MSSubClass_MSZoning_50_RL", "MSSubClass_MSZoning_60_RL", "MSSubClass_MSZoning_70_RL", 
    "MSSubClass_MSZoning_120_RL", "MSSubClass_MSZoning_160_RL", "Neighborhood_Condition_CollgCr_Norm", 
    "Neighborhood_Condition_Crawfor_Norm", "Neighborhood_Condition_Gilbert_Norm", 
    "Neighborhood_Condition_NAmes_Norm", "Neighborhood_Condition_NoRidge_Norm", 
    "Neighborhood_Condition_NridgHt_Norm", "Neighborhood_Condition_OldTown_Norm", 
    "Neighborhood_Condition_Somerst_Norm", "Neighborhood_Condition_StoneBr_Norm", 
    "GarageType_Attchd", "GarageType_BuiltIn", "GarageType_Detchd", "CentralAir_Electrical_N_SBrkr", 
    "CentralAir_Electrical_Y_SBrkr", "PavedDrive_N", "PavedDrive_Y", "SaleCondition_Abnorml", 
    "SaleCondition_Normal", "SaleCondition_Partial", "Foundation_CBlock", "Foundation_PConc", 
    "Season_Sold_Spring", "Season_Sold_Summer"
]


X_train_xgb = X_train_tree_raw[xgb_features]
X_val_xgb = X_val_tree_raw[xgb_features]
test_xgb = test_tree_raw[xgb_features]
                           
xgb_model = xgb.XGBRegressor(random_state=random_state, objective="reg:squarederror")

param_grid = {
    "n_estimators": [180, 200],  
    "learning_rate": [0.07, 0.10], 
    "max_depth": [2, 3, 4],  
    "min_child_weight": [2, 3], 
    "subsample": [0.78, 0.8,],  
    "colsample_bytree": [0.728, 0.75],  
    "reg_alpha": [0, 0.5],  
    "reg_lambda": [0.281, 1]  
}

gs_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    refit=True)

gs_xgb.fit(X_train_xgb, y_train.values.ravel())

print("10-Fold CV RMSE:", -gs_xgb.best_score_)  
print("Optimal Parameters:", gs_xgb.best_params_)
print("Optimal Estimator:", gs_xgb.best_estimator_)

final_model_xgb = gs_xgb.best_estimator_

selected_features_xgb = X_train_xgb.columns[np.array(final_model_xgb.feature_importances_) > 0]

with open("models/selected_features_xgb.txt", "w") as f:
    for feat in selected_features_xgb:
        f.write(f"{feat}\n")

# Save the trained model for future use (stacking)
with open("models/final_model_xgb.pkl", "wb") as f:
    pickle.dump(final_model_xgb, f)
print("xgboost model saved to models/final_model_xgb.pkl")


############################################## XGB Models with Bayesian Optimization ############################################################
# Best boosting round (from xgb.cv early stopping) recorded per BO call so we can refit
best_rounds_per_call = {}
MAX_BOOST_ROUNDS = 5000


def bayesian_opt_xgb(X, y, init_iter=10, n_iters=40, random_state=random_state, seed=seed):
    dtrain = xgb.DMatrix(data=X, label=y)

    def hyp_xgb(learning_rate, max_depth, min_child_weight, subsample,
                colsample_bytree, reg_alpha, reg_lambda, gamma):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "seed": seed,
            "n_jobs": -1,
            "booster": "gbtree",
            "learning_rate": learning_rate,
            "max_depth": int(round(max_depth)),
            "min_child_weight": min_child_weight,
            "subsample": max(min(subsample, 1), 0),
            "colsample_bytree": max(min(colsample_bytree, 1), 0),
            "reg_alpha": max(reg_alpha, 0),
            "reg_lambda": max(reg_lambda, 0),
            "gamma": max(gamma, 0),
        }

        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=MAX_BOOST_ROUNDS,
            nfold=10,
            seed=seed,
            shuffle=True,
            stratified=False,
            early_stopping_rounds=50,
            metrics="rmse",
        )
        best_rmse = cv_results["test-rmse-mean"].min()
        best_round = int(cv_results["test-rmse-mean"].idxmin()) + 1
        # Key params by their float repr so we can look up best_round after BO finishes
        key = (
            round(learning_rate, 6), int(round(max_depth)), round(min_child_weight, 6),
            round(subsample, 6), round(colsample_bytree, 6),
            round(reg_alpha, 6), round(reg_lambda, 6), round(gamma, 6),
        )
        best_rounds_per_call[key] = best_round
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


results = bayesian_opt_xgb(X_train_xgb, y_train.values.ravel())
print("Best Parameters:", results.max["params"])
print("Best RMSE Score:", -results.max["target"])

best_params = results.max["params"].copy()
best_params["max_depth"] = int(round(best_params["max_depth"]))

# Look up the best boosting round captured during the optimal BO call
best_key = (
    round(best_params["learning_rate"], 6), best_params["max_depth"],
    round(best_params["min_child_weight"], 6), round(best_params["subsample"], 6),
    round(best_params["colsample_bytree"], 6), round(best_params["reg_alpha"], 6),
    round(best_params["reg_lambda"], 6), round(best_params["gamma"], 6),
)
best_n_estimators = best_rounds_per_call.get(best_key, MAX_BOOST_ROUNDS)
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