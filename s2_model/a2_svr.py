import os
import pickle

import duckdb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import LinearSVR, SVR

from s1_data.db_utils import load_df, save_df
from s3_validation.model_evaluation import evaluate_model

base_folder = "data"
database = "AmesHousePrice.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=False)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

############################# RBF SVR #############################
X_train_svr_raw = load_df(conn, "X_train_svr")
X_val_svr_raw = load_df(conn, "X_val_svr")
test_svr_raw = load_df(conn, "test_svr")
y_train_svr = load_df(conn, "y_train")
y_val_svr = load_df(conn, "y_val")

# sorted() so the fitted column order matches what load_df returns in a8 and s4.
rbf_svr_features = sorted([
    "log_LotFrontage", "log_LotArea", "OverallQual", "OverallCond", "cbrt_MasVnrArea",
    "sqrt_BsmtFinSF1",
    "BsmtFinSF2", "sqrt_BsmtUnfSF", "sqrt_TotalBsmtSF", "log_1stFlrSF", "log_2ndFlrSF", "log_GrLivArea",
    "BsmtFullBath", "FullBath", "HalfBath", "TotRmsAbvGrd", "Fireplaces", "GarageCars",
    "GarageArea", "sqrt_WoodDeckSF", "cbrt_OpenPorchSF", "EnclosedPorch", "ScreenPorch", "Age_House",
    "log_Yrs_Since_Remodel", "log_Age_Garage", "ExterQual_encoded", "BsmtQual_encoded",
    "BsmtCond_encoded", "BsmtExposure_encoded", "BsmtFinType1_encoded", "KitchenQual_encoded",
    "FireplaceQu_encoded", "GarageFinish_encoded", "GarageQual_encoded", "GarageCond_encoded",
    "PoolQC_encoded", "Functional_encoded", "FinishedAreaPct", "Living_Rooms", "Garage_Space",
    "Garage_AgeCars", "Ratio_Bedroom_Rooms", "Ratio_2ndFlr_Living", "Area_vs_Nbhd",
    "MSSubClass_MSZoning_20_RL",
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
])


X_train_rbf_svr = X_train_svr_raw[rbf_svr_features]
X_val_rbf_svr = X_val_svr_raw[rbf_svr_features]
test_rbf_svr = test_svr_raw[rbf_svr_features]

# gamma is searched as multiples of sklearn's 'scale' heuristic, which anchors the grid to the
# data's variance; multiplier 1.0 reproduces 'scale'. tol is a convergence tolerance, not a
# hyperparameter, so it is pinned rather than searched.
gamma_scale = 1.0 / (X_train_rbf_svr.shape[1] * X_train_rbf_svr.to_numpy(dtype=float).var())
print(f"RBF SVR gamma='scale' resolves to {gamma_scale:.5f}")

rbf_svr_estimator = SVR(kernel="rbf", tol=1e-5)
rbf_svr_param_grid = {
    "C": [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10],
    "epsilon": [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12],
    "gamma": [m * gamma_scale for m in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)],
}

rbf_svr_grid = GridSearchCV(
    estimator=rbf_svr_estimator,
    param_grid=rbf_svr_param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    refit=True,
)
rbf_svr_grid.fit(X_train_rbf_svr, y_train_svr.values.ravel())

print("RBF SVR 10-Fold CV RMSE:", -rbf_svr_grid.best_score_)
print("RBF SVR Optimal Parameter:", rbf_svr_grid.best_params_)
print("RBF SVR Optimal Estimator:", rbf_svr_grid.best_estimator_)

best_rbf_svr_model = rbf_svr_grid.best_estimator_
with open("models/final_model_svr_rbf.pkl", "wb") as f:
    pickle.dump(best_rbf_svr_model, f)
print("RBF SVR model saved to models/final_model_svr_rbf.pkl")

save_df(conn, X_train_rbf_svr, "X_train_svr_rbf")
save_df(conn, X_val_rbf_svr, "X_val_svr_rbf")
save_df(conn, test_rbf_svr, "test_svr_rbf")

############################# LinearSVR #############################
X_train_linear_raw = load_df(conn, "X_train_reg")
X_val_linear_raw = load_df(conn, "X_val_reg")
test_linear_raw = load_df(conn, "test_reg")
y_train_linear = load_df(conn, "y_train")
y_val_linear = load_df(conn, "y_val")

linear_svr_features = sorted([
    "Age_House", "Area_vs_Nbhd", "BedroomAbvGr", "BsmtExposure_encoded", "BsmtFullBath", "CentralAir_Electrical_N_SBrkr",
    "EnclosedPorch", "ExterQual_encoded", "Exterior1st_Exterior2nd_BrkFace", "Exterior1st_Exterior2nd_BrkFace_Wd Sdng", "FinishedAreaPct",
    "Fireplaces", "Foundation_PConc", "FullBath", "Functional_encoded", "GarageArea",
    "GarageCars", "Garage_AgeCars", "Garage_Space", "HalfBath", "KitchenAbvGr",
    "KitchenQual_encoded", "Living_Rooms", "Neighborhood_Condition_BrkSide_Norm", "Neighborhood_Condition_Crawfor_Norm", "Neighborhood_Condition_Edwards_PosN",
    "Neighborhood_Condition_NoRidge_Norm", "Neighborhood_Condition_NridgHt_Norm", "Neighborhood_Condition_Somerst_Norm", "Neighborhood_Condition_StoneBr_Norm", "OverallCond",
    "OverallQual", "Porch_Age", "Ratio_2ndFlr_Living", "Ratio_Bedroom_Rooms", "RoofStyle_RoofMatl_Hip_ClyTile",
    "SaleCondition_Normal", "SaleType_New", "ScreenPorch", "TotRmsAbvGrd", "cbrt_MasVnrArea",
    "cbrt_OpenPorchSF", "log_1stFlrSF", "log_2ndFlrSF", "log_GrLivArea", "log_LotArea",
    "log_Yrs_Since_Remodel", "sqrt_BsmtFinSF1", "sqrt_TotalBsmtSF", "sqrt_WoodDeckSF",
])
X_train_linear_svr = X_train_linear_raw[linear_svr_features]
X_val_linear_svr = X_val_linear_raw[linear_svr_features]
test_linear_svr = test_linear_raw[linear_svr_features]

linear_svr_estimator = LinearSVR(random_state=42, max_iter=50000)
linear_svr_param_grid = {
    "loss": ["squared_epsilon_insensitive"],
    "dual": [False],
    "C": [0.1, 1, 10, 50],
    "epsilon": [0.01, 0.05, 0.1],
    "tol": [1e-4, 1e-3],
}
linear_svr_grid = GridSearchCV(
    estimator=linear_svr_estimator,
    param_grid=linear_svr_param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    refit=True,
)
linear_svr_grid.fit(X_train_linear_svr, y_train_linear.values.ravel())

best_linear_svr_model = linear_svr_grid.best_estimator_
print("LinearSVR 10-Fold CV RMSE:", -linear_svr_grid.best_score_)
print("LinearSVR Optimal Parameter:", linear_svr_grid.best_params_)
print("LinearSVR Optimal Estimator:", best_linear_svr_model)

with open("models/final_model_linear_svr.pkl", "wb") as f:
    pickle.dump(best_linear_svr_model, f)
print("LinearSVR model saved to models/final_model_linear_svr.pkl")

save_df(conn, X_train_linear_svr, "X_train_linear_svr")
save_df(conn, X_val_linear_svr, "X_val_linear_svr")
save_df(conn, test_linear_svr, "test_linear_svr")

evaluate_model(best_rbf_svr_model, X_val_rbf_svr, y_val_svr, "RBF SVR Model")
evaluate_model(best_linear_svr_model, X_val_linear_svr, y_val_linear, "LinearSVR Model")

conn.close()