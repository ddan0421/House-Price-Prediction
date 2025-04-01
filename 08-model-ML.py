# Tree-based models
# use non-scaled data
import pandas as pd
import numpy as np
import duckdb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from bayes_opt import BayesianOptimization
import pickle
import models
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


X_train = pd.read_csv("data/model_data/X_train_ml.csv")
X_val = pd.read_csv("data/model_data/X_val_ml.csv")
test_final = pd.read_csv("data/model_data/test_final_ml.csv")
y_train = pd.read_csv("data/model_data/y_train_ml.csv")
y_val = pd.read_csv("data/model_data/y_val_ml.csv")

random_state = 42
seed = 42
###################################################################### Feature Selection ######################################################################
# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=200, random_state=random_state)
rf_model.fit(X_train, y_train.values.ravel())

# Get feature importance
feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf_model.feature_importances_
})

conn = duckdb.connect()
# Define the threshold for cumulative importance
threshold = 0.95

# Calculate cumulative importance and filter features
# Keep enough features such that their cumulative importance adds up to 95% of the total importance.
query = f"""
WITH sorted_importance AS (
    SELECT
        feature,
        importance
    FROM feature_importance
    ORDER BY importance DESC
),
cumulative_importance AS (
    SELECT
        feature,
        importance,
        SUM(importance) OVER (ORDER BY importance DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_importance,
        ROW_NUMBER() OVER (ORDER BY importance DESC) AS row_number
    FROM sorted_importance
),
threshold_index AS (
    SELECT
        MIN(row_number) AS threshold_row
    FROM cumulative_importance
    WHERE cumulative_importance >= {threshold}
)
SELECT
    feature
FROM cumulative_importance
INNER JOIN threshold_index
ON cumulative_importance.row_number <= threshold_index.threshold_row;

"""

# Execute the query to get selected features
selected_features_basic_rf = conn.execute(query).fetch_df()["feature"].to_list()
conn.close()


selected_numeric_features = [
    "LotArea", "MasVnrArea", "TotalBsmtSF", "1stFlrSF", 
    "GrLivArea", "BsmtFullBath", "FullBath", "HalfBath", "BedroomAbvGr", 
    "KitchenAbvGr", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", 
    "OpenPorchSF", "EnclosedPorch", "Age_House", "TotRmsAbvGrd"
]


# Combine the lists and remove duplicates using a set
combined_features_tree = list(set(selected_features_basic_rf + selected_numeric_features))
combined_features_tree.sort()

X_train_tree = X_train[combined_features_tree]
X_val_tree = X_val[combined_features_tree]

############################################## Decision Tree Regressor Model ############################################################
cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
dt = DecisionTreeRegressor(random_state=random_state, criterion="squared_error")

param_grid = {
    "max_depth": [10, 20, 30, 40, None],  # Maximum depth of the tree
    "min_samples_split": [2, 5, 10, 20],  # Minimum number of samples required to split a node
    "min_samples_leaf": [1, 2, 5, 10],  # Minimum number of samples required at a leaf node
    "min_weight_fraction_leaf": [0.0, 0.01, 0.05],  # Minimum weighted fraction of the sum of weights at a leaf node
}

gs_dt = GridSearchCV(estimator=dt,
                     param_grid=param_grid,
                     scoring="neg_root_mean_squared_error",  # Using RMSE as the evaluation metric
                     cv=cv,
                     n_jobs=-1,
                     refit=True)

gs_dt.fit(X_train_tree, y_train.values.ravel())

print("10-Fold CV RMSE:", -gs_dt.best_score_)  # RMSE is the negative value from GridSearchCV
print("Optimal Parameters:", gs_dt.best_params_)
print("Optimal Estimator:", gs_dt.best_estimator_)

final_model_dt = gs_dt.best_estimator_

selected_features_dt = X_train_tree.columns[final_model_dt.feature_importances_ > 0]
print("Selected features for Decision Tree:")
print(selected_features_dt)


############################################## Random Forest Tree Regressor Model ############################################################
cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
rf = RandomForestRegressor(random_state=random_state, bootstrap=True)

param_grid = {
    "n_estimators": [50, 100, 200],  # Fewer trees for faster training
    "max_depth": [3, 5, 10],  # Restrict depth to limit complexity
    "min_samples_split": [2, 5],  # Reduce the range for faster grid search
    "min_samples_leaf": [1, 2],  # Keep leaf size small for better granularity
    "max_features": ["sqrt", "log2"],  # Subset of features considered at each split
}

gs_rf = GridSearchCV(estimator=rf,
                     param_grid=param_grid,
                     scoring="neg_root_mean_squared_error",  # Using RMSE as the evaluation metric
                     cv=cv,
                     n_jobs=-1,
                     refit=True)

gs_rf.fit(X_train_tree, y_train.values.ravel())

print("10-Fold CV RMSE:", -gs_rf.best_score_)  # RMSE is the negative value from GridSearchCV
print("Optimal Parameters:", gs_rf.best_params_)
print("Optimal Estimator:", gs_rf.best_estimator_)

final_model_rf = gs_rf.best_estimator_

selected_features_rf = X_train_tree.columns[final_model_rf.feature_importances_ > 0]
print("Selected features for Random Forest:")
print(selected_features_rf)

# Save the trained model for future use (stacking)
with open("final_model_rf.pkl", "wb") as f:
    pickle.dump(final_model_rf, f)
print("random forest model saved to final_model_rf.pkl")

X_train_tree.to_csv("data/model_data/X_train_rf.csv", index=False)
y_train.to_csv("data/model_data/y_train_rf.csv", index=False)
X_val_tree.to_csv("data/model_data/X_val_rf.csv", index=False)

############################################## XGBoost Regressor Model ############################################################
# Feature Selection based on a basic XGBoost model
basic_xgb = xgb.XGBRegressor(random_state=random_state, objective="reg:squarederror", n_estimators=200)
basic_xgb.fit(X_train, y_train.values.ravel())

feature_importances = basic_xgb.feature_importances_
selected_features_xgb = X_train.columns[np.array(feature_importances) > 0].to_list()  # Select non-zero importance features

combined_features_xgb = list(set(selected_features_xgb + selected_numeric_features + selected_features_basic_rf))
combined_features_xgb.sort()

X_train_xgb = X_train[combined_features_xgb]
X_val_xgb = X_val[combined_features_xgb]
print("Selected Features:", combined_features_xgb)

cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
xgb_model = xgb.XGBRegressor(random_state=random_state, objective="reg:squarederror")

param_grid = {
    "n_estimators": [100, 200],  
    "learning_rate": [0.05, 0.1],        # Tune learning_rate to balance overfitting
    "max_depth": [3, 5],                     # Typical values for tree depth
    "min_child_weight": [1, 5],              # Vary min_child_weight to control overfitting
    "subsample": [0.8, 1.0],               # Subsample to prevent overfitting
    "colsample_bytree": [0.8, 1.0],        # Column subsampling to control model complexity
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

selected_features_xgb_final = X_train_xgb.columns[np.array(final_model_xgb.feature_importances_) > 0]
print("Selected features for XGBoost:")
print(selected_features_xgb_final)

# Save the trained model for future use (stacking)
with open("final_model_xgb.pkl", "wb") as f:
    pickle.dump(final_model_xgb, f)
print("xgboost model saved to final_model_xgb.pkl")

X_train_xgb.to_csv("data/model_data/X_train_xgb.csv", index=False)
y_train.to_csv("data/model_data/y_train_xgb.csv", index=False)
X_val_xgb.to_csv("data/model_data/X_val_xgb.csv", index=False)

############################################## XGBoost Regressor Model with gblinear ############################################################
sig_features = ["Age_House", "BsmtCond", "BsmtFinType1", "BsmtFullBath", "BsmtQual", 
                "Exterior1st_Exterior2nd_BrkComm_Brk Cmn", "Exterior1st_Exterior2nd_BrkFace", 
                "Exterior1st_Exterior2nd_BrkFace_Plywood", "Exterior1st_Exterior2nd_BrkFace_Wd Sdng", 
                "Exterior1st_Exterior2nd_CemntBd_CmentBd", "Exterior1st_Exterior2nd_Wd Sdng_ImStucc", 
                "Fireplaces", "Foundation_Stone", "Foundation_Wood", "FullBath", "Functional_Maj2", 
                "Functional_Min1", "Functional_Min2", "Functional_Mod", "Functional_Sev", "GarageArea", 
                "HalfBath", "Heating_HeatingQC_GasW_Fa", "KitchenAbvGr", "KitchenQual", 
                "LotConfig_LandSlope_CulDSac_Gtl", "LotShape_LandContour_IR1_HLS", 
                "LotShape_LandContour_IR2_Bnk", "Neighborhood_Condition_BrkSide_Norm", 
                "Neighborhood_Condition_BrkSide_PosN_Norm", "Neighborhood_Condition_Edwards_Artery_Norm", 
                "Neighborhood_Condition_Edwards_PosN", "Neighborhood_Condition_IDOTRR_Artery_Norm", 
                "Neighborhood_Condition_MeadowV_Norm", "Neighborhood_Condition_NAmes_Artery_Norm", 
                "Neighborhood_Condition_NAmes_PosA_Norm", "Neighborhood_Condition_NWAmes_Norm", 
                "Neighborhood_Condition_NWAmes_RRAn_Norm", "Neighborhood_Condition_NoRidge_Norm", 
                "Neighborhood_Condition_NridgHt_Norm", "Neighborhood_Condition_OldTown_Artery_Norm", 
                "Neighborhood_Condition_OldTown_Feedr_Norm", "Neighborhood_Condition_Sawyer_RRAe_Norm", 
                "Neighborhood_Condition_StoneBr_Norm", "OverallCond", "OverallQual", 
                "RoofStyle_RoofMatl_Gable_WdShngl", "RoofStyle_RoofMatl_Hip_ClyTile", 
                "SaleCondition_AdjLand", "SaleCondition_Alloca", "SaleCondition_Normal", 
                "SaleType_Oth", "SaleType_WD", "ScreenPorch", "log_GrLivArea", "log_LotArea", 
                "log_Yrs_Since_Remodel", "sqrt_TotalBsmtSF", "sqrt_WoodDeckSF"]

X_train_linear = pd.read_csv("data/model_data/X_train.csv")
X_val_linear = pd.read_csv("data/model_data/X_val.csv")
y_train_linear = pd.read_csv("data/model_data/y_train.csv")
y_val_linear = pd.read_csv("data/model_data/y_val.csv")

X_train_xgb_linear = X_train_linear[sig_features]
X_val_xgb_linear = X_val_linear[sig_features]

cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
xgb_model_linear = xgb.XGBRegressor(
    random_state=random_state, 
    objective="reg:squarederror", 
    booster="gblinear"
)

param_grid = {
    "n_estimators": [50, 100, 200],  
    "learning_rate": [0.01, 0.05, 0.1],  # Learning rate for convergence
    "lambda": [0.1, 1, 10],             # L2 regularization
    "alpha": [0, 0.1, 0.5, 1]           # L1 regularization
}

gs_xgb_linear = GridSearchCV(
    estimator=xgb_model_linear,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    refit=True
)

gs_xgb_linear.fit(X_train_xgb_linear, y_train_linear.values.ravel())

print("10-Fold CV RMSE:", -gs_xgb_linear.best_score_)
print("Optimal Parameters:", gs_xgb_linear.best_params_)
print("Optimal Estimator:", gs_xgb_linear.best_estimator_)

final_model_xgb_linear = gs_xgb_linear.best_estimator_


############################################## LightGBM Regressor Model ############################################################
# Reduce the feature set using xgb's selected features
X_train_lgbm = X_train[combined_features_xgb]
X_val_lgbm = X_val[combined_features_xgb]

cv = KFold(n_splits=10, shuffle=True, random_state=random_state)

lgbm = lgb.LGBMRegressor(random_state=random_state, objective="regression")

param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],          # Tune learning_rate to balance overfitting
    "max_depth": [-1, 3, 5],                 # LightGBM allows -1 for unlimited depth
    "num_leaves": [31, 64],               # Number of leaves, impacts model complexity
    "min_child_samples": [10, 20],        # Min number of data in a leaf
    "subsample": [0.8, 1.0],              # Subsample to prevent overfitting
    "colsample_bytree": [0.8, 1.0]       # Column subsampling to control model complexity
}

gs_lgbm = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    refit=True)

gs_lgbm.fit(X_train_lgbm, y_train.values.ravel())

print("10-Fold CV RMSE:", -gs_lgbm.best_score_) 
print("Optimal Parameters:", gs_lgbm.best_params_)
print("Optimal Estimator:", gs_lgbm.best_estimator_)

final_model_lgbm = gs_lgbm.best_estimator_

selected_features_lgbm = X_train_lgbm.columns[np.array(final_model_lgbm.feature_importances_) > 0]
print("Selected features for LightGBM:")
print(selected_features_lgbm)




############################################## LGBM Models with Bayesian Optimization ############################################################
# Learn about this (Bayesian Optimization)
# https://medium.com/analytics-vidhya/hyperparameters-optimization-for-lightgbm-catboost-and-xgboost-regressors-using-bayesian-6e7c495947a9
def bayesian_opt_lgbm(X, y, init_iter=10, n_iters=25, random_state=random_state, seed=seed):
    # Prepare LightGBM dataset
    dtrain = lgb.Dataset(data=X, label=y)

    # Custom RMSE evaluation function
    def lgb_rmse_score(preds, dtrain):
        labels = dtrain.get_label()
        rmse = root_mean_squared_error(labels, preds)
        return "rmse", rmse, False  # False indicates lower is better

    # Objective Function for Bayesian Optimization
    def hyp_lgbm(num_boost_round, learning_rate, max_depth, num_leaves, min_child_samples, min_sum_hessian_in_leaf, feature_fraction_bynode, reg_alpha, reg_lambda, min_split_gain, feature_fraction, bagging_fraction, bagging_freq):
        params = {
            "objective": "regression",
            "metric": "rmse",  # Use RMSE for evaluation
            "verbosity": -1,   # Suppress LightGBM logs
            "feature_pre_filter": False,  # Prevent pre-filtering of features when adjusting min_data_in_leaf
            "seed": seed,
            "n_jobs": -1,
            "boosting_type": "gbdt", 
        }
        params["num_boost_round"] = int(round(num_boost_round))
        params["learning_rate"] = learning_rate
        params["max_depth"] = int(round(max_depth))
        params["num_leaves"] = int(round(num_leaves))
        params["min_child_samples"] = int(round(min_child_samples))
        params["min_sum_hessian_in_leaf"] = min_sum_hessian_in_leaf 
        params["feature_fraction_bynode"] = feature_fraction_bynode
        params["reg_alpha"] = max(reg_alpha, 0)
        params["reg_lambda"] = max(reg_lambda, 0)
        params["min_split_gain"] = max(min_split_gain, 0)
        params["feature_fraction"] = max(min(feature_fraction,1),0)
        params["bagging_fraction"] = max(min(bagging_fraction, 1), 0)
        params["bagging_freq"] = int(round(bagging_freq))

        # Perform cross-validation using RMSE
        cv_results = lgb.cv(
            params,
            dtrain,
            nfold=10,
            seed=seed,
            stratified=False,
            feval=lgb_rmse_score,
        )
        return -np.min(cv_results["valid rmse-mean"])  # Return negative RMSE for maximization

    # Define hyperparameter search space
    pds = {
        "num_boost_round": (50, 300),  # Increased range
        "learning_rate": (0.005, 0.15),  # Expanded learning rate range
        "max_depth": (-1, 15),  # Increased max depth
        "num_leaves": (15, 256),  # Increased num leaves
        "min_child_samples": (5, 100),  # Expanded min child samples
        "min_sum_hessian_in_leaf": (1e-3, 10),
        "feature_fraction_bynode": (0.1, 1.0),  # Fraction of features per tree node
        "reg_alpha": (0, 2),  # Expanded regularization
        "reg_lambda": (0, 2),  # Expanded regularization
        "min_split_gain": (0, 2),  # Expanded min split gain
        "feature_fraction": (0.5, 1.0), #added feature fraction
        "bagging_fraction": (0.5, 1.0), #added bagging fraction
        "bagging_freq": (1, 10), #added bagging frequency
    }

    # Initialize Bayesian Optimization
    optimizer = BayesianOptimization(hyp_lgbm, pds, random_state=random_state)

    # Perform optimization
    optimizer.maximize(init_points=init_iter, n_iter=n_iters)

    return optimizer

results = bayesian_opt_lgbm(X_train_lgbm, y_train)

# Print the best parameters and best score
print("Best Parameters:", results.max["params"])
print("Best RMSE Score:", -results.max["target"])  # Convert back to positive RMSE


best_params = results.max["params"]
best_params["num_boost_round"] = int(round(best_params["num_boost_round"]))
best_params["learning_rate"] = best_params["learning_rate"]
best_params["max_depth"] = int(round(best_params["max_depth"]))
best_params["num_leaves"] = int(round(best_params["num_leaves"]))
best_params["min_child_samples"] = int(round(best_params["min_child_samples"]))
best_params["min_sum_hessian_in_leaf"] = best_params["min_sum_hessian_in_leaf"]
best_params["feature_fraction_bynode"] = best_params["feature_fraction_bynode"]
best_params["reg_alpha"] = max(best_params["reg_alpha"], 0)
best_params["reg_lambda"] = max(best_params["reg_lambda"], 0)
best_params["min_split_gain"] = max(best_params["min_split_gain"], 0)
best_params["feature_fraction"] = max(min(best_params["feature_fraction"], 1), 0)
best_params["bagging_fraction"] = max(min(best_params["bagging_fraction"], 1), 0)
best_params["bagging_freq"] = int(round(best_params["bagging_freq"]))

best_params["seed"] = seed
best_params["n_jobs"] = -1
best_params["objective"] = "regression"
best_params["metric"] = "rmse"
best_params["verbosity"] = -1
best_params["feature_pre_filter"] = False
best_params["boosting_type"] = "gbdt"

lgbm_bayes_model = lgb.LGBMRegressor(**best_params)
lgbm_bayes_model.fit(X_train_lgbm, y_train)

############################################## XGB Models with Bayesian Optimization ############################################################
def bayesian_opt_xgb(X, y, init_iter=20, n_iters=50, random_state=random_state, seed=seed):
    # Objective Function for Bayesian Optimization
    def hyp_xgb(n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "seed": seed,
            "n_jobs": -1,
        }
        params["n_estimators"] = int(round(n_estimators))
        params["learning_rate"] = learning_rate
        params["max_depth"] = int(round(max_depth))
        params["min_child_weight"] = min_child_weight
        params["subsample"] = max(min(subsample, 1), 0)
        params["colsample_bytree"] = max(min(colsample_bytree, 1), 0)
        params["reg_alpha"] = max(reg_alpha, 0)
        params["reg_lambda"] = max(reg_lambda, 0)

        # Perform cross-validation using RMSE
        dtrain = xgb.DMatrix(data=X, label=y)
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=int(round(n_estimators)),
            nfold=10,
            seed=seed,
            stratified=False,
            early_stopping_rounds=10,
            metrics="rmse"
        )
        return -cv_results["test-rmse-mean"].min()  # Return negative RMSE for maximization

    # Define hyperparameter search space
    pds = {
        "n_estimators": (50, 200),  # Increased range
        "learning_rate": (0.005, 0.15),  # Expanded learning rate range
        "max_depth": (3, 15),  # Increased max depth
        "min_child_weight": (1, 5),  # Expanded min child weight
        "subsample": (0.5, 1.0),  # Fraction of samples per tree
        "colsample_bytree": (0.5, 1.0),  # Fraction of features per tree
        "reg_alpha": (0, 2),  # L1 regularization
        "reg_lambda": (0, 2),  # L2 regularization
    }

    # Initialize Bayesian Optimization
    optimizer = BayesianOptimization(hyp_xgb, pds, random_state=random_state)

    # Perform optimization
    optimizer.maximize(init_points=init_iter, n_iter=n_iters)

    return optimizer

# Run Bayesian Optimization
results = bayesian_opt_xgb(X_train_xgb, y_train)
# Print the best parameters and best score
print("Best Parameters:", results.max["params"])
print("Best RMSE Score:", -results.max["target"])  # Convert back to positive RMSE


best_params = results.max["params"]
best_params["n_estimators"] = int(round(best_params["n_estimators"]))
best_params["learning_rate"] = best_params["learning_rate"]
best_params["max_depth"] = int(round(best_params["max_depth"]))
best_params["min_child_weight"] = best_params["min_child_weight"]
best_params["subsample"] = max(min(best_params["subsample"], 1), 0)
best_params["colsample_bytree"] = max(min(best_params["colsample_bytree"], 1), 0)
best_params["reg_alpha"] = max(best_params["reg_alpha"], 0)
best_params["reg_lambda"] = max(best_params["reg_lambda"], 0)

best_params["objective"] = "reg:squarederror"
best_params["eval_metric"] = "rmse"
best_params["seed"] = seed
best_params["n_jobs"] = -1
best_params["random_state"] = random_state

xgb_bayes_model = xgb.XGBRegressor(**best_params)
xgb_bayes_model.fit(X_train_xgb, y_train)

############################################## Models Generalization Performance ##############################################
def evaluate_tree_model(model, X, y, name):
    predictions = model.predict(X)
    rmse = root_mean_squared_error(y, predictions)
    print(f"{name} Performance:")
    print(f"Root Mean Squared Error: {rmse:.4f}")


def evaluate_linear_model(model, X, y, name):
    predictions = model.predict(X)
    predictions = np.expm1(predictions)
    y_actual = np.expm1(y)
    rmse = root_mean_squared_error(y_actual, predictions)
    print(f"{name} Performance:")
    print(f"Root Mean Squared Error: {rmse:.4f}")

print("############################################## 10-Fold CV Hyperparameter-Tuned ##############################################")
evaluate_tree_model(final_model_dt, X_val_tree, y_val, "Decision Tree Regressor")
evaluate_tree_model(final_model_rf, X_val_tree, y_val, "Random Forest Regressor")
evaluate_tree_model(final_model_xgb, X_val_xgb, y_val, "XGBoost Regressor")
evaluate_linear_model(final_model_xgb_linear, X_val_xgb_linear, y_val_linear, "XGBoost (gblinear) Regressor")
evaluate_tree_model(final_model_lgbm, X_val_lgbm, y_val, "LightGBM Regressor")

print("############################################## 10-Fold CV Hyperparameter-Tuned with Bayesian Optimization ##############################################")
evaluate_tree_model(xgb_bayes_model, X_val_xgb, y_val, "XGBoost Regressor (Bayesian Optimizied)")
evaluate_tree_model(lgbm_bayes_model, X_val_lgbm, y_val, "LightGBM Regressor (Bayesian Optimizied)")





################################################# Stacking Models #######################################################
# Load datasets for each base model
X_train_xgb = pd.read_csv("data/model_data/X_train_xgb.csv").values
y_train_xgb = pd.read_csv("data/model_data/y_train_xgb.csv").values.flatten()
X_train_rf = pd.read_csv("data/model_data/X_train_rf.csv").values
y_train_rf = pd.read_csv("data/model_data/y_train_rf.csv").values.flatten()
X_train_svm = pd.read_csv("data/model_data/X_train_svm.csv").values
y_train_svm = pd.read_csv("data/model_data/y_train_svm.csv").values.flatten()

# Load pre-trained base models
with open("final_model_xgb.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("final_model_rf.pkl", "rb") as f:
    rf_model = pickle.load(f)   
with open("final_model_svm.pkl", "rb") as f:
    svr_model = pickle.load(f)

base_models = [
    ("xgb", xgb_model, X_train_xgb, y_train_xgb),
    ("rf", rf_model, X_train_rf, y_train_rf),
    ("svr", svr_model, X_train_svm, y_train_svm)
]

# Create an empty array to store OOF predictions
oof_preds = np.zeros((X_train_xgb.shape[0], len(base_models)))

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Generate OOF predictions for each base model
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_xgb)):
    for i, (name, model, X_train, y_train) in enumerate(base_models):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        model.fit(X_fold_train, y_fold_train)  # Train on fold-specific data

        if name == "svr":
            oof_preds[val_idx, i] = np.expm1(model.predict(X_fold_val))
        else:
            oof_preds[val_idx, i] = model.predict(X_fold_val)  # Regular predictions for other models

oof_df = pd.DataFrame(oof_preds, columns=[name for name, _, _, _ in base_models])
oof_df["Target"] = y_train_xgb
print("OOF Predictions:")
print(oof_df)

X_agg = sm.add_constant(oof_df.drop(columns=["Target"]))
y_agg = oof_df["Target"]
# Train meta-learner (Linear Regression)
glm_lr = models.sm_glm_gaussian(X_agg, y_agg)
meta_learner = models.constrained_sm_glm_gaussian(X_agg, y_agg, glm_lr, 0.05)



# Simulate predictions on val data
X_val_xgb = pd.read_csv("data/model_data/X_val_xgb.csv")
X_val_rf = pd.read_csv("data/model_data/X_val_rf.csv")
X_val_svm = pd.read_csv("data/model_data/X_val_svm.csv")

y_val = pd.read_csv("data/model_data/y_val_ml.csv")

test_preds = np.zeros((X_val_xgb.shape[0], len(base_models)))

for i, (name, model, _, _) in enumerate(base_models):
    if name == "xgb":
        test_preds[:, i] = model.predict(X_val_xgb)
    elif name == "rf":
        test_preds[:, i] = model.predict(X_val_rf)
    elif name == "svr":
        test_preds[:, i] = np.expm1(model.predict(X_val_svm))

# Final predictions using meta-learner
test_preds = sm.add_constant(test_preds)
final_preds = meta_learner.predict(test_preds)
rmse = root_mean_squared_error(y_val, final_preds)
print(f"Stacking Performance:")
print(f"Root Mean Squared Error: {rmse:.4f}")