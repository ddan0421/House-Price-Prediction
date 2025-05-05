# Tree-based models
# use non-scaled data
import pandas as pd
import numpy as np
import duckdb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from bayes_opt import BayesianOptimization
import pickle


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


X_train = pd.read_csv("data/model_data/X_train_ml.csv")
X_val = pd.read_csv("data/model_data/X_val_ml.csv")
test_final = pd.read_csv("data/model_data/test_final_ml.csv")
y_train = pd.read_csv("data/model_data/y_train_ml.csv")
y_val = pd.read_csv("data/model_data/y_val_ml.csv")

random_state = 42
seed = 42
###################################################################### Numerical Interaction ######################################################################
# def create_interactions(df):
#     df["Living_Rooms"] = df["GrLivArea"] * df["TotRmsAbvGrd"]
#     df["Garage_Space"] = df["GarageArea"] * df["GarageCars"]
#     df["Garage_AgeCars"] = df["Age_Garage"] * df["GarageCars"]
#     df["Porch_Age"] = df["EnclosedPorch"] * df["Age_House"]
#     df["Ratio_Bedroom_Rooms"] = df["BedroomAbvGr"] / (df["TotRmsAbvGrd"])
#     df["Ratio_2ndFlr_Living"] = df["2ndFlrSF"] / (df["GrLivArea"])

#     return df
# X_train = create_interactions(X_train)
# X_val = create_interactions(X_val)
# test_final = create_interactions(test_final)
# test_final.to_csv("data/model_data/test_final_ml.csv", index=False)
###################################################################### Feature Selection ######################################################################
# # Train a Random Forest Regressor
# rf_model = RandomForestRegressor(n_estimators=200, random_state=random_state)
# rf_model.fit(X_train, y_train.values.ravel())

# # Get feature importance
# feature_importance = pd.DataFrame({
#     "feature": X_train.columns,
#     "importance": rf_model.feature_importances_
# })

# conn = duckdb.connect()
# # Define the threshold for cumulative importance
# threshold = 0.95

# # Calculate cumulative importance and filter features
# # Keep enough features such that their cumulative importance adds up to 95% of the total importance.
# query = f"""
# WITH sorted_importance AS (
#     SELECT
#         feature,
#         importance
#     FROM feature_importance
#     ORDER BY importance DESC
# ),
# cumulative_importance AS (
#     SELECT
#         feature,
#         importance,
#         SUM(importance) OVER (ORDER BY importance DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_importance,
#         ROW_NUMBER() OVER (ORDER BY importance DESC) AS row_number
#     FROM sorted_importance
# ),
# threshold_index AS (
#     SELECT
#         MIN(row_number) AS threshold_row
#     FROM cumulative_importance
#     WHERE cumulative_importance >= {threshold}
# )
# SELECT
#     feature
# FROM cumulative_importance
# INNER JOIN threshold_index
# ON cumulative_importance.row_number <= threshold_index.threshold_row;

# """

# # Execute the query to get selected features
# selected_features_basic_rf = conn.execute(query).fetch_df()["feature"].to_list()
# conn.close()



# xgb_features = ['1stFlrSF',
#  '2ndFlrSF',
#  '3SsnPorch',
#  'Age_Garage',
#  'Age_House',
#  'Alley_Pave',
#  'BedroomAbvGr',
#  'BsmtCond_encoded',
#  'BsmtExposure_encoded',
#  'BsmtFinSF1',
#  'BsmtFinSF2',
#  'BsmtFinType1_encoded',
#  'BsmtFullBath',
#  'BsmtHalfBath',
#  'BsmtQual_encoded',
#  'BsmtUnfSF',
#  'CentralAir_Y',
#  'Condition1_Norm',
#  'Condition1_PosN',
#  'Condition1_RRAe',
#  'Condition2_Norm',
#  'Electrical_SBrkr',
#  'EnclosedPorch',
#  'ExterCond_encoded',
#  'ExterQual_encoded',
#  'Exterior1st_BrkFace',
#  'Exterior1st_MetalSd',
#  'Exterior1st_Plywood',
#  'Exterior1st_VinylSd',
#  'Exterior1st_Wd Sdng',
#  'Exterior2nd_Plywood',
#  'Exterior2nd_Stucco',
#  'Exterior2nd_Wd Sdng',
#  'Exterior2nd_Wd Shng',
#  'Fence_GdWo',
#  'Fence_MnPrv',
#  'Fence_no_fence',
#  'FireplaceQu_encoded',
#  'Fireplaces',
#  'Foundation_PConc',
#  'Foundation_Wood',
#  'FullBath',
#  'Functional_Maj2',
#  'Functional_Typ',
#  'GarageArea',
#  'GarageCars',
#  'GarageCond_encoded',
#  'GarageFinish_encoded',
#  'GarageQual_encoded',
#  'GarageType_Attchd',
#  'GarageType_Basment',
#  'GarageType_CarPort',
#  'GarageType_Detchd',
#  'Age_Garage',
#  'GarageArea',
#  'GarageCars',
#  'GrLivArea',
#  'HalfBath',
#  'HeatingQC_encoded',
#  'Heating_GasA',
#  'HouseStyle_1Story',
#  'HouseStyle_2Story',
#  'HouseStyle_SLvl',
#  'KitchenAbvGr',
#  'KitchenQual_encoded',
#  'LandContour_HLS',
#  'LandContour_Lvl',
#  'GrLivArea',
#  'TotRmsAbvGrd',
#  'LotArea',
#  'LotConfig_CulDSac',
#  'LotConfig_FR2',
#  'LotConfig_Inside',
#  'LotFrontage',
#  'LotShape_encoded',
#  'MSSubClass_190',
#  'MSSubClass_30',
#  'MSSubClass_50',
#  'MSSubClass_70',
#  'MSSubClass_80',
#  'MSZoning_FV',
#  'MSZoning_RH',
#  'MSZoning_RL',
#  'MSZoning_RM',
#  'MasVnrArea',
#  'MasVnrType_Stone',
#  'Neighborhood_BrkSide',
#  'Neighborhood_ClearCr',
#  'Neighborhood_Crawfor',
#  'Neighborhood_Edwards',
#  'Neighborhood_MeadowV',
#  'Neighborhood_Mitchel',
#  'Neighborhood_NAmes',
#  'Neighborhood_NWAmes',
#  'Neighborhood_OldTown',
#  'Neighborhood_SWISU',
#  'Neighborhood_Sawyer',
#  'Neighborhood_SawyerW',
#  'Neighborhood_Somerst',
#  'Neighborhood_StoneBr',
#  'OpenPorchSF',
#  'OverallCond',
#  'OverallQual',
#  'PavedDrive_P',
#  'PoolArea',
#  'EnclosedPorch',
#  'Age_House',
#  '2ndFlrSF',
#  'RoofMatl_CompShg',
#  'RoofStyle_Gable',
#  'RoofStyle_Hip',
#  'SaleCondition_Family',
#  'SaleCondition_Normal',
#  'SaleType_New',
#  'SaleType_WD',
#  'ScreenPorch',
#  'Season_Sold_Spring',
#  'Season_Sold_Summer',
#  'Season_Sold_Winter',
#  'TotalBsmtSF',
#  'WoodDeckSF',
#  'BedroomAbvGr',
#  'Yrs_Since_Remodel']




# selected_numeric_features = [
#     "LotArea", "MasVnrArea", "TotalBsmtSF", "1stFlrSF", 
#     "GrLivArea", "BsmtFullBath", "FullBath", "HalfBath", "BedroomAbvGr", 
#     "KitchenAbvGr", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", 
#     "OpenPorchSF", "EnclosedPorch", "Age_House", "TotRmsAbvGrd",
#     "Yrs_Since_Remodel", "2ndFlrSF"
# ]


# # Combine the lists and remove duplicates using a set
# combined_features_tree = list(set(xgb_features + selected_numeric_features))
# combined_features_tree.sort()

X_train_tree = X_train.copy()
X_val_tree = X_val.copy()

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

# Save the trained model for future use (stacking)
with open("final_model_dt.pkl", "wb") as f:
    pickle.dump(final_model_dt, f)
print("decision tree model saved to final_model_dt.pkl")

X_train_tree.to_csv("data/model_data/X_train_dt.csv", index=False)
y_train.to_csv("data/model_data/y_train_dt.csv", index=False)
X_val_tree.to_csv("data/model_data/X_val_dt.csv", index=False)


############################################## Decision Tree Regressor Model ############################################################
cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
sdt = DecisionTreeRegressor(random_state=random_state)

param_grid = {
    "max_depth": [2, 3],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

gs_sdt = GridSearchCV(estimator=sdt, 
                      param_grid=param_grid, 
                      scoring="neg_root_mean_squared_error", 
                      cv=cv, n_jobs=-1, 
                      refit=True)

gs_sdt.fit(X_train_tree, y_train.values.ravel())

print("10-Fold CV RMSE:", -gs_sdt.best_score_)  # RMSE is the negative value from GridSearchCV
print("Optimal Parameters:", gs_sdt.best_params_)
print("Optimal Estimator:", gs_sdt.best_estimator_)

final_model_sdt = gs_sdt.best_estimator_

with open("final_model_sdt.pkl", "wb") as f:
    pickle.dump(final_model_sdt, f)

X_train_tree.to_csv("data/model_data/X_train_sdt.csv", index=False)
y_train.to_csv("data/model_data/y_train_sdt.csv", index=False)
X_val_tree.to_csv("data/model_data/X_val_sdt.csv", index=False)

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


############################################## ExtraTreesRegressor Model ############################################################
cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
et = ExtraTreesRegressor(random_state=random_state, criterion='squared_error')

param_grid = {
    "n_estimators": [50, 100, 200],  # Number of trees in the forest
    "max_depth": [3, 5, 10],  # Maximum depth of the trees to avoid overfitting
    "min_samples_split": [2, 5],  # Minimum number of samples required to split a node
    "min_samples_leaf": [1, 2],  # Minimum number of samples required at a leaf node
    "max_features": ["sqrt", "log2", None],  # The number of features to consider at each split
    "bootstrap": [True, False]  # Whether bootstrap samples are used when building trees
}

gs_et = GridSearchCV(estimator=et, 
                     param_grid=param_grid, 
                     scoring="neg_root_mean_squared_error", 
                     cv=cv, 
                     n_jobs=-1, 
                     refit=True)

gs_et.fit(X_train_tree, y_train.values.ravel())

print("10-Fold CV RMSE:", -gs_et.best_score_)  # RMSE is the negative value from GridSearchCV
print("Optimal Parameters:", gs_et.best_params_)
print("Optimal Estimator:", gs_et.best_estimator_)

final_model_et = gs_et.best_estimator_


with open("final_model_et.pkl", "wb") as f:
    pickle.dump(final_model_et, f)

X_train_tree.to_csv("data/model_data/X_train_et.csv", index=False)
y_train.to_csv("data/model_data/y_train_et.csv", index=False)
X_val_tree.to_csv("data/model_data/X_val_et.csv", index=False)

############################################## XGBoost Regressor Model ############################################################
X_train_xgb = X_train.copy()
X_val_xgb = X_val.copy()

cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
xgb_model = xgb.XGBRegressor(random_state=random_state, objective="reg:squarederror")

param_grid = {
    "n_estimators": [180, 200],  # Best: ~180.3
    "learning_rate": [0.07, 0.10],  # Best: ~0.074
    "max_depth": [2, 3, 4],  # Best: ~3.3
    "min_child_weight": [2, 3],  # Best: ~2.8
    "subsample": [0.78, 0.8,],  # Best: ~0.7995
    "colsample_bytree": [0.728, 0.75],  # Best: ~0.728
    "reg_alpha": [0, 0.5],  # Best: ~0.5046
    "reg_lambda": [0.281, 1]  # Best: ~0.281
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


############################################## LightGBM Regressor Model ############################################################
# Reduce the feature set using xgb's selected features
X_train_lgbm = X_train.copy()
X_val_lgbm = X_val.copy()

cv = KFold(n_splits=10, shuffle=True, random_state=random_state)

lgbm = lgb.LGBMRegressor(random_state=random_state, objective="regression", verbose=-1)

param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],          
    "max_depth": [-1, 3, 5],            
    "num_leaves": [31, 64],            
    "min_child_samples": [10, 20],      
    "subsample": [0.8, 1.0],           
    "colsample_bytree": [0.8, 1.0],    
    "reg_alpha": [0.0, 0.1],                     
    "reg_lambda": [1.0, 2.0],                      
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

# Save the trained model for future use (stacking)
with open("final_model_lgbm.pkl", "wb") as f:
    pickle.dump(final_model_lgbm, f)
print("lgbm model saved to final_model_lgbm.pkl")

X_train_lgbm.to_csv("data/model_data/X_train_lgbm.csv", index=False)
y_train.to_csv("data/model_data/y_train_lgbm.csv", index=False)
X_val_lgbm.to_csv("data/model_data/X_val_lgbm.csv", index=False)


############################################## LGBM Models with Bayesian Optimization ############################################################
# Learn about this (Bayesian Optimization)
# https://medium.com/analytics-vidhya/hyperparameters-optimization-for-lightgbm-catboost-and-xgboost-regressors-using-bayesian-6e7c495947a9
def bayesian_opt_lgbm(X, y, init_iter=40, n_iters=50, random_state=random_state, seed=seed):
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
        "num_boost_round": (150, 250),  # Focused around optimal results
        "learning_rate": (0.08, 0.15),  # Centered on optimal values
        "max_depth": (3, 10),  # Tightened range based on grid and Bayesian results
        "num_leaves": (15, 128),  # Balanced between grid search and Bayesian ranges
        "min_child_samples": (10, 25),  # Focused around optimal values
        "min_sum_hessian_in_leaf": (1e-3, 7),  # Narrowed based on Bayesian result
        "feature_fraction_bynode": (0.3, 0.7),  # Focused around Bayesian results
        "reg_alpha": (0, 1),  # Balanced between grid and Bayesian ranges
        "reg_lambda": (1, 2),  # Focused on higher values suggested by Bayesian
        "min_split_gain": (0.0, 0.1),  # Tightened range based on Bayesian results
        "feature_fraction": (0.6, 0.8),  # Focused on the Bayesian result
        "bagging_fraction": (0.7, 0.9),  # Centered around Bayesian optimal
        "bagging_freq": (3, 7),  # Tightened range based on Bayesian results
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

# Save the trained model for future use (stacking)
with open("final_model_LGBM_bayes.pkl", "wb") as f:
    pickle.dump(lgbm_bayes_model, f)
print("LGBM Bayes model saved to final_model_LGBM_bayes.pkl")

X_train_lgbm.to_csv("data/model_data/X_train_lgbm_bayes.csv", index=False)
y_train.to_csv("data/model_data/y_train_lgbm_bayes.csv", index=False)
X_val_lgbm.to_csv("data/model_data/X_val_lgbm_bayes.csv", index=False)

############################################## XGB Models with Bayesian Optimization ############################################################
def bayesian_opt_xgb(X, y, init_iter=40, n_iters=50, random_state=random_state, seed=seed):
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
        "n_estimators": (150, 200),  # Increased range
        "learning_rate": (0.07, 0.12),  # Expanded learning rate range
        "max_depth": (3, 5),  # Increased max depth
        "min_child_weight": (1, 3),  # Expanded min child weight
        "subsample": (0.7, 0.85),  # Fraction of samples per tree
        "colsample_bytree": (0.6, 0.8),  # Fraction of features per tree
        "reg_alpha": (0.5, 1.0),  # L1 regularization
        "reg_lambda": (0.2, 1.0),  # L2 regularization
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

# Save the trained model for future use (stacking)
with open("final_model_xgb_bayes.pkl", "wb") as f:
    pickle.dump(xgb_bayes_model, f)
print("XGB Bayes model saved to final_model_xgb_bayes.pkl")

X_train_xgb.to_csv("data/model_data/X_train_xgb_bayes.csv", index=False)
y_train.to_csv("data/model_data/y_train_xgb_bayes.csv", index=False)
X_val_xgb.to_csv("data/model_data/X_val_xgb_bayes.csv", index=False)

############################################## Models Generalization Performance ##############################################
def evaluate_tree_model(model, X, y, name):
    predictions = model.predict(X)
    rmse = root_mean_squared_error(y, predictions)
    print(f"{name} Performance:")
    print(f"Root Mean Squared Error: {rmse:.4f}")

print("############################################## 10-Fold CV Hyperparameter-Tuned ##############################################")
evaluate_tree_model(final_model_dt, X_val_tree, y_val, "Decision Tree Regressor")
evaluate_tree_model(final_model_sdt, X_val_tree, y_val, "Shallow Decision Tree Regressor")
evaluate_tree_model(final_model_rf, X_val_tree, y_val, "Random Forest Regressor")
evaluate_tree_model(final_model_et, X_val_tree, y_val, "Extra Trees Regressor")
evaluate_tree_model(final_model_xgb, X_val_xgb, y_val, "XGBoost Regressor")
evaluate_tree_model(final_model_lgbm, X_val_lgbm, y_val, "LightGBM Regressor")

print("############################################## 10-Fold CV Hyperparameter-Tuned with Bayesian Optimization ##############################################")
evaluate_tree_model(xgb_bayes_model, X_val_xgb, y_val, "XGBoost Regressor (Bayesian Optimizied)")
evaluate_tree_model(lgbm_bayes_model, X_val_lgbm, y_val, "LightGBM Regressor (Bayesian Optimizied)")

