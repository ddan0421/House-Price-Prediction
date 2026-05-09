import os
import pickle
import numpy as np
import duckdb
from sklearn.model_selection import GridSearchCV, KFold
import lightgbm as lgb
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

X_train_cat = load_df(conn, "X_train_cat")
X_val_cat = load_df(conn, "X_val_cat")
test_cat = load_df(conn, "test_cat")
y_train = load_df(conn, "y_train")
y_val = load_df(conn, "y_val")


lgbm_features = [
    "OverallQual", "log_GrLivArea", "FireplaceQu", "sqrt_TotalBsmtSF", "ExterQual", 
    "GarageCars", "log_LotArea", "Living_Rooms", "KitchenQual", "Age_House", 
    "BsmtQual", "sqrt_BsmtFinSF1", "GarageType", "log_Yrs_Since_Remodel", "OverallCond", 
    "Garage_Space", "log_LotFrontage", "Fireplaces", "log_1stFlrSF", "GarageFinish", 
    "BsmtFullBath", "GarageArea", "CentralAir_Electrical", "SaleCondition", "GarageQual", 
    "cbrt_OpenPorchSF", "TotRmsAbvGrd", "PavedDrive", "MSSubClass_MSZoning", "BsmtExposure", 
    "sqrt_WoodDeckSF", "HalfBath", "PoolQC", "Functional", "Neighborhood_Condition", 
    "log_2ndFlrSF", "FullBath", "Season_Sold", "Foundation", "GarageCond"
]

nominal_cat = ["MSSubClass_MSZoning", "LotConfig_LandSlope", "Neighborhood_Condition", "BldgType_HouseStyle",
               "Exterior1st_Exterior2nd", "CentralAir_Electrical", "LotShape_LandContour", "RoofStyle_RoofMatl",
               "Heating_HeatingQC", "Alley", "MasVnrType", "Foundation", "GarageType", "PavedDrive", "Fence", 
               "MiscFeature", "SaleType", "SaleCondition", "Season_Sold"]

ordinal_cat = ["Utilities", "Functional", "OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtQual", 
               "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "KitchenQual", "FireplaceQu", 
               "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Street"]


all_cat_columns = nominal_cat + ordinal_cat

cat_columns = [f for f in lgbm_features if f in all_cat_columns]

X_train_cat[cat_columns] = X_train_cat[cat_columns].astype("category")
X_val_cat[cat_columns] = X_val_cat[cat_columns].astype("category")
test_cat[cat_columns] = test_cat[cat_columns].astype("category")


X_train_lgbm = X_train_cat[lgbm_features]
X_val_lgbm = X_val_cat[lgbm_features]
test_lgbm = test_cat[lgbm_features]


############################################## LightGBM Regressor Model ############################################################
lgbm = lgb.LGBMRegressor(random_state=random_state, 
                         objective="regression", 
                         verbose=-1,
                         n_jobs=1)

param_grid = {
    "n_estimators": [150, 200],
    "learning_rate": [0.08, 0.11],          
    "max_depth": [3, 5],            
    "num_leaves": [25, 35],            
    "min_child_samples": [15, 20],      
    "subsample": [0.75, 0.85],           
    "colsample_bytree": [0.75, 0.85],    
    "reg_alpha": [0.0, 0.01],                     
    "reg_lambda": [0.9, 1.1],     
    "min_split_gain": [0.0, 0.008],                 
}

gs_lgbm = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True)

gs_lgbm.fit(X_train_lgbm, y_train.values.ravel(), categorical_feature=cat_columns)

print("10-Fold CV RMSE:", -gs_lgbm.best_score_) 
print("Optimal Parameters:", gs_lgbm.best_params_)
print("Optimal Estimator:", gs_lgbm.best_estimator_)

final_model_lgbm = gs_lgbm.best_estimator_

selected_features_lgbm = X_train_lgbm.columns[np.array(final_model_lgbm.feature_importances_) > 0]

with open("models/selected_features_lgbm.txt", "w") as f:
    for feat in selected_features_lgbm:
        f.write(f"{feat}\n")


# Save the trained model for future use (stacking)
with open("models/final_model_lgbm.pkl", "wb") as f:
    pickle.dump(final_model_lgbm, f)
print("lgbm model saved to models/final_model_lgbm.pkl")




############################################## LGBM Models with Bayesian Optimization ############################################################
# Best boosting round (from lgb.cv early stopping) recorded per BO call so we can refit
best_rounds_per_call = {}
MAX_BOOST_ROUNDS = 5000


def bayesian_opt_lgbm(X, y, cat_features, init_iter=30, n_iters=120,
                      random_state=random_state, seed=seed):
    dtrain = lgb.Dataset(data=X, label=y, categorical_feature=cat_features, free_raw_data=False)

    def hyp_lgbm(learning_rate, max_depth, num_leaves, min_child_samples,
                 min_sum_hessian_in_leaf, reg_alpha, reg_lambda, min_split_gain,
                 colsample_bytree, subsample, subsample_freq, feature_fraction_bynode):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "feature_pre_filter": False,
            "seed": seed,
            "n_jobs": -1,
            "boosting_type": "gbdt",
            "learning_rate": learning_rate,
            "max_depth": int(round(max_depth)),
            "num_leaves": int(round(num_leaves)),
            "min_child_samples": int(round(min_child_samples)),
            "min_sum_hessian_in_leaf": min_sum_hessian_in_leaf,
            "feature_fraction_bynode": max(min(feature_fraction_bynode, 1), 0),
            "reg_alpha": max(reg_alpha, 0),
            "reg_lambda": max(reg_lambda, 0),
            "min_split_gain": max(min_split_gain, 0),
            "colsample_bytree": max(min(colsample_bytree, 1), 0),
            "subsample": max(min(subsample, 1), 0),
            "subsample_freq": int(round(subsample_freq)),
        }

        cv_results = lgb.cv(
            params,
            dtrain,
            num_boost_round=MAX_BOOST_ROUNDS,
            nfold=10,
            seed=seed,
            shuffle=True,
            stratified=False,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                       lgb.log_evaluation(0)],
        )
        rmse_curve = np.array(cv_results["valid rmse-mean"])
        best_rmse = rmse_curve.min()
        best_round = int(rmse_curve.argmin()) + 1
        key = (
            round(learning_rate, 6), int(round(max_depth)), int(round(num_leaves)),
            int(round(min_child_samples)), round(min_sum_hessian_in_leaf, 6),
            round(reg_alpha, 6), round(reg_lambda, 6), round(min_split_gain, 6),
            round(colsample_bytree, 6), round(subsample, 6),
            int(round(subsample_freq)), round(feature_fraction_bynode, 6),
        )
        best_rounds_per_call[key] = best_round
        return -best_rmse

    pds = {
        "learning_rate": (0.005, 0.3),
        "max_depth": (3, 12),
        "num_leaves": (15, 200),
        "min_child_samples": (5, 50),
        "min_sum_hessian_in_leaf": (1e-3, 5),
        "reg_alpha": (0, 5),
        "reg_lambda": (0, 5),
        "min_split_gain": (0, 1),
        "colsample_bytree": (0.4, 1.0),
        "subsample": (0.5, 1.0),
        "subsample_freq": (0, 7),
        "feature_fraction_bynode": (0.5, 1.0),
    }

    optimizer = BayesianOptimization(f=hyp_lgbm, pbounds=pds, random_state=random_state)
    optimizer.maximize(init_points=init_iter, n_iter=n_iters)
    return optimizer


results = bayesian_opt_lgbm(X_train_lgbm, y_train.values.ravel(), cat_features=cat_columns)
print("Best Parameters:", results.max["params"])
print("Best RMSE Score:", -results.max["target"])

best_params = results.max["params"].copy()
best_params["max_depth"] = int(round(best_params["max_depth"]))
best_params["num_leaves"] = int(round(best_params["num_leaves"]))
best_params["min_child_samples"] = int(round(best_params["min_child_samples"]))
best_params["subsample_freq"] = int(round(best_params["subsample_freq"]))

# Look up the best boosting round captured during the optimal BO call
best_key = (
    round(best_params["learning_rate"], 6), best_params["max_depth"],
    best_params["num_leaves"], best_params["min_child_samples"],
    round(best_params["min_sum_hessian_in_leaf"], 6),
    round(best_params["reg_alpha"], 6), round(best_params["reg_lambda"], 6),
    round(best_params["min_split_gain"], 6),
    round(best_params["colsample_bytree"], 6), round(best_params["subsample"], 6),
    best_params["subsample_freq"], round(best_params["feature_fraction_bynode"], 6),
)
best_n_estimators = best_rounds_per_call.get(best_key, MAX_BOOST_ROUNDS)
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
