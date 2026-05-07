import os
import pickle

import duckdb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import LinearSVR, SVR

from s1_data.db_utils import load_df, save_df
from s3_validation.model_evaluation import evaluate_linear_model

base_folder = "data"
database = "AmesHousePrice.duckdb"
database_path = os.path.join(base_folder, database)
os.makedirs("models", exist_ok=True)

conn = duckdb.connect(database=database_path, read_only=False)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

############################# RBF SVR #############################
X_train_svr_raw = load_df(conn, "X_train_svr")
X_val_svr_raw = load_df(conn, "X_val_svr")
test_svr_raw = load_df(conn, "test_svr")
y_train_svr = load_df(conn, "y_train")
y_val_svr = load_df(conn, "y_val")

rbf_svr_features = [
    "3SsnPorch", "Age_House", "Alley_Pave", "Alley_no_alley", "BedroomAbvGr",
    "BldgType_HouseStyle_2fmCon_2.5Fin", "BldgType_HouseStyle_2fmCon_SLvl", "BsmtCond_encoded", "BsmtExposure_encoded", "BsmtFinSF2",
    "BsmtFinType1_encoded", "BsmtFinType2_encoded", "BsmtFullBath", "BsmtHalfBath", "BsmtQual_encoded",
    "CentralAir_Electrical_N_FuseF", "CentralAir_Electrical_N_FuseP", "CentralAir_Electrical_N_SBrkr", "CentralAir_Electrical_Y_FuseA", "CentralAir_Electrical_Y_FuseF",
    "CentralAir_Electrical_Y_SBrkr", "EnclosedPorch", "ExterCond_encoded", "ExterQual_encoded", "Exterior1st_Exterior2nd_AsbShng_Plywood",
    "Exterior1st_Exterior2nd_AsbShng_Stucco", "Exterior1st_Exterior2nd_AsphShn", "Exterior1st_Exterior2nd_BrkComm_Brk Cmn", "Exterior1st_Exterior2nd_BrkFace", "Exterior1st_Exterior2nd_BrkFace_AsbShng",
    "Exterior1st_Exterior2nd_BrkFace_HdBoard", "Exterior1st_Exterior2nd_BrkFace_Plywood", "Exterior1st_Exterior2nd_BrkFace_Stone", "Exterior1st_Exterior2nd_BrkFace_Stucco", "Exterior1st_Exterior2nd_BrkFace_Wd Sdng",
    "Exterior1st_Exterior2nd_BrkFace_Wd Shng", "Exterior1st_Exterior2nd_CBlock", "Exterior1st_Exterior2nd_CemntBd_CmentBd", "Exterior1st_Exterior2nd_CemntBd_Wd Sdng", "Exterior1st_Exterior2nd_CemntBd_Wd Shng",
    "Exterior1st_Exterior2nd_HdBoard_AsphShn", "Exterior1st_Exterior2nd_HdBoard_ImStucc", "Exterior1st_Exterior2nd_HdBoard_Plywood", "Exterior1st_Exterior2nd_HdBoard_Wd Sdng", "Exterior1st_Exterior2nd_HdBoard_Wd Shng",
    "Exterior1st_Exterior2nd_ImStucc", "Exterior1st_Exterior2nd_MetalSd_AsphShn", "Exterior1st_Exterior2nd_MetalSd_HdBoard", "Exterior1st_Exterior2nd_MetalSd_Stucco", "Exterior1st_Exterior2nd_MetalSd_Wd Sdng",
    "Exterior1st_Exterior2nd_MetalSd_Wd Shng", "Exterior1st_Exterior2nd_Plywood_Brk Cmn", "Exterior1st_Exterior2nd_Plywood_HdBoard", "Exterior1st_Exterior2nd_Plywood_ImStucc", "Exterior1st_Exterior2nd_Plywood_Wd Sdng",
    "Exterior1st_Exterior2nd_Stone", "Exterior1st_Exterior2nd_Stucco", "Exterior1st_Exterior2nd_Stucco_CmentBd", "Exterior1st_Exterior2nd_Stucco_Wd Shng", "Exterior1st_Exterior2nd_VinylSd_AsbShng",
    "Exterior1st_Exterior2nd_VinylSd_HdBoard", "Exterior1st_Exterior2nd_VinylSd_ImStucc", "Exterior1st_Exterior2nd_VinylSd_Other", "Exterior1st_Exterior2nd_VinylSd_Plywood", "Exterior1st_Exterior2nd_VinylSd_Stucco",
    "Exterior1st_Exterior2nd_VinylSd_Wd Shng", "Exterior1st_Exterior2nd_Wd Sdng_AsbShng", "Exterior1st_Exterior2nd_Wd Sdng_HdBoard", "Exterior1st_Exterior2nd_Wd Sdng_ImStucc", "Exterior1st_Exterior2nd_Wd Sdng_Plywood",
    "Exterior1st_Exterior2nd_Wd Sdng_VinylSd", "Exterior1st_Exterior2nd_Wd Sdng_Wd Shng", "Exterior1st_Exterior2nd_WdShing_Plywood", "Exterior1st_Exterior2nd_WdShing_Stucco", "Exterior1st_Exterior2nd_WdShing_Wd Sdng",
    "Exterior1st_Exterior2nd_WdShing_Wd Shng", "Fence_GdWo", "Fence_MnPrv", "Fence_MnWw", "Fence_no_fence",
    "FireplaceQu_encoded", "Fireplaces", "Foundation_PConc", "Foundation_Slab", "Foundation_Stone",
    "Foundation_Wood", "FullBath", "Functional_encoded", "GarageArea", "GarageCars",
    "GarageCond_encoded", "GarageFinish_encoded", "GarageQual_encoded", "GarageType_Attchd", "GarageType_Basment",
    "GarageType_CarPort", "GarageType_Detchd", "Garage_AgeCars", "Garage_Space", "HalfBath",
    "Heating_HeatingQC_GasW_Ex", "Heating_HeatingQC_GasW_Fa", "Heating_HeatingQC_GasW_Gd", "Heating_HeatingQC_Grav_Fa", "Heating_HeatingQC_OthW_Fa",
    "Heating_HeatingQC_Wall_Fa", "Heating_HeatingQC_Wall_TA", "KitchenAbvGr", "KitchenQual_encoded", "Living_Rooms",
    "LotConfig_LandSlope_Corner_Mod", "LotConfig_LandSlope_Corner_Sev", "LotConfig_LandSlope_CulDSac_Gtl", "LotConfig_LandSlope_CulDSac_Mod", "LotConfig_LandSlope_CulDSac_Sev",
    "LotConfig_LandSlope_FR2_Gtl", "LotConfig_LandSlope_FR3_Gtl", "LotConfig_LandSlope_Inside_Gtl", "LotConfig_LandSlope_Inside_Mod", "LotConfig_LandSlope_Inside_Sev",
    "LotShape_LandContour_IR1_HLS", "LotShape_LandContour_IR1_Low", "LotShape_LandContour_IR2_Bnk", "LotShape_LandContour_IR2_HLS", "LotShape_LandContour_IR2_Low",
    "LotShape_LandContour_IR2_Lvl", "LotShape_LandContour_IR3_Bnk", "LotShape_LandContour_IR3_HLS", "LotShape_LandContour_IR3_Low", "LotShape_LandContour_IR3_Lvl",
    "LotShape_LandContour_Reg_Bnk", "LotShape_LandContour_Reg_HLS", "LotShape_LandContour_Reg_Low", "MSSubClass_MSZoning_120_RH", "MSSubClass_MSZoning_160_RM",
    "MSSubClass_MSZoning_190_RH", "MSSubClass_MSZoning_30_C (all)", "MSSubClass_MSZoning_50_RH", "MSSubClass_MSZoning_50_RM", "Neighborhood_Condition_Blueste_Norm",
    "Neighborhood_Condition_BrDale_Norm", "Neighborhood_Condition_BrkSide_Feedr_Norm", "Neighborhood_Condition_BrkSide_Norm", "Neighborhood_Condition_BrkSide_PosN_Norm", "Neighborhood_Condition_BrkSide_RRAn_Feedr",
    "Neighborhood_Condition_BrkSide_RRAn_Norm", "Neighborhood_Condition_BrkSide_RRNn_Feedr", "Neighborhood_Condition_ClearCr_Feedr_Norm", "Neighborhood_Condition_ClearCr_Norm", "Neighborhood_Condition_CollgCr_PosN_Norm",
    "Neighborhood_Condition_Crawfor_Norm", "Neighborhood_Condition_Crawfor_PosA_Norm", "Neighborhood_Condition_Crawfor_PosN_Norm", "Neighborhood_Condition_Edwards_Artery_Norm", "Neighborhood_Condition_Edwards_Feedr_Norm",
    "Neighborhood_Condition_Edwards_PosN", "Neighborhood_Condition_Gilbert_Feedr_Norm", "Neighborhood_Condition_Gilbert_RRAn_Norm", "Neighborhood_Condition_Gilbert_RRNn_Norm", "Neighborhood_Condition_IDOTRR_Artery_Norm",
    "Neighborhood_Condition_IDOTRR_Feedr", "Neighborhood_Condition_IDOTRR_Norm", "Neighborhood_Condition_IDOTRR_RRNn_Norm", "Neighborhood_Condition_MeadowV_Norm", "Neighborhood_Condition_Mitchel_Feedr_Norm",
    "Neighborhood_Condition_Mitchel_Norm", "Neighborhood_Condition_NAmes_Artery_Norm", "Neighborhood_Condition_NAmes_Feedr_Norm", "Neighborhood_Condition_NAmes_PosA_Norm", "Neighborhood_Condition_NAmes_PosN_Norm",
    "Neighborhood_Condition_NPkVill_Norm", "Neighborhood_Condition_NWAmes_Feedr_Norm", "Neighborhood_Condition_NWAmes_Feedr_RRAn", "Neighborhood_Condition_NWAmes_PosA_Norm", "Neighborhood_Condition_NWAmes_PosN_Norm",
    "Neighborhood_Condition_NWAmes_RRAn_Norm", "Neighborhood_Condition_NoRidge_Norm", "Neighborhood_Condition_NridgHt_Norm", "Neighborhood_Condition_NridgHt_PosN", "Neighborhood_Condition_OldTown_Artery",
    "Neighborhood_Condition_OldTown_Artery_Norm", "Neighborhood_Condition_OldTown_Artery_PosA", "Neighborhood_Condition_OldTown_Feedr_Norm", "Neighborhood_Condition_OldTown_Feedr_RRNn", "Neighborhood_Condition_SWISU_Feedr_Norm",
    "Neighborhood_Condition_SWISU_Norm", "Neighborhood_Condition_SawyerW_Feedr_Norm", "Neighborhood_Condition_SawyerW_Norm", "Neighborhood_Condition_SawyerW_RRAe_Norm", "Neighborhood_Condition_SawyerW_RRNe_Norm",
    "Neighborhood_Condition_Sawyer_Feedr_Norm", "Neighborhood_Condition_Sawyer_Norm", "Neighborhood_Condition_Sawyer_PosN_Norm", "Neighborhood_Condition_Sawyer_RRAe_Norm", "Neighborhood_Condition_Somerst_Feedr_Norm",
    "Neighborhood_Condition_Somerst_Norm", "Neighborhood_Condition_Somerst_RRAn_Norm", "Neighborhood_Condition_Somerst_RRNn_Norm", "Neighborhood_Condition_StoneBr_Norm", "Neighborhood_Condition_Timber_Norm",
    "Neighborhood_Condition_Veenker_Feedr_Norm", "Neighborhood_Condition_Veenker_Norm", "OverallCond", "OverallQual", "PavedDrive_P",
    "PavedDrive_Y", "Porch_Age", "Ratio_2ndFlr_Living", "Ratio_Bedroom_Rooms", "RoofStyle_RoofMatl_Flat_Metal",
    "RoofStyle_RoofMatl_Gable_Roll", "RoofStyle_RoofMatl_Gable_WdShngl", "RoofStyle_RoofMatl_Hip_ClyTile", "RoofStyle_RoofMatl_Hip_WdShake", "RoofStyle_RoofMatl_Hip_WdShngl",
    "RoofStyle_RoofMatl_Mansard_CompShg", "RoofStyle_RoofMatl_Mansard_WdShake", "RoofStyle_RoofMatl_Shed_WdShake", "SaleCondition_AdjLand", "SaleCondition_Alloca",
    "SaleCondition_Family", "SaleCondition_Normal", "SaleType_CWD", "SaleType_Con", "SaleType_ConLD",
    "SaleType_ConLI", "SaleType_ConLw", "SaleType_New", "SaleType_Oth", "SaleType_WD",
    "ScreenPorch", "Season_Sold_Spring", "Season_Sold_Summer", "Season_Sold_Winter", "Street_encoded",
    "TotRmsAbvGrd", "Utilities_encoded", "cbrt_MasVnrArea", "cbrt_OpenPorchSF", "log_1stFlrSF",
    "log_2ndFlrSF", "log_Age_Garage", "log_GrLivArea", "log_LotArea", "log_LotFrontage",
    "log_LowQualFinSF", "log_Yrs_Since_Remodel", "sqrt_BsmtFinSF1", "sqrt_BsmtUnfSF", "sqrt_TotalBsmtSF",
    "sqrt_WoodDeckSF",
]


X_train_rbf_svr = X_train_svr_raw[rbf_svr_features]
X_val_rbf_svr = X_val_svr_raw[rbf_svr_features]
test_rbf_svr = test_svr_raw[rbf_svr_features]

rbf_svr_estimator = SVR(kernel="rbf")
rbf_svr_param_grid = {
    "C": [0.01, 0.1, 1, 10, 50, 100, 200, 500],
    "epsilon": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
    "tol": [1e-5, 1e-4, 1e-3],
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

linear_svr_features = [
    "Age_House", "BedroomAbvGr", "BsmtExposure_encoded", "BsmtFullBath", "CentralAir_Electrical_N_SBrkr",
    "EnclosedPorch", "ExterQual_encoded", "Exterior1st_Exterior2nd_BrkFace", "Exterior1st_Exterior2nd_BrkFace_Wd Sdng", "FinishedAreaPct",
    "Fireplaces", "Foundation_PConc", "FullBath", "Functional_encoded", "GarageArea",
    "GarageCars", "Garage_AgeCars", "Garage_Space", "HalfBath", "KitchenAbvGr",
    "KitchenQual_encoded", "Living_Rooms", "Neighborhood_Condition_BrkSide_Norm", "Neighborhood_Condition_Crawfor_Norm", "Neighborhood_Condition_Edwards_PosN",
    "Neighborhood_Condition_NoRidge_Norm", "Neighborhood_Condition_NridgHt_Norm", "Neighborhood_Condition_Somerst_Norm", "Neighborhood_Condition_StoneBr_Norm", "OverallCond",
    "OverallQual", "Porch_Age", "Ratio_2ndFlr_Living", "Ratio_Bedroom_Rooms", "RoofStyle_RoofMatl_Hip_ClyTile",
    "SaleCondition_Normal", "SaleType_New", "ScreenPorch", "TotRmsAbvGrd", "cbrt_MasVnrArea",
    "cbrt_OpenPorchSF", "log_1stFlrSF", "log_2ndFlrSF", "log_GrLivArea", "log_LotArea",
    "log_Yrs_Since_Remodel", "sqrt_BsmtFinSF1", "sqrt_TotalBsmtSF", "sqrt_WoodDeckSF",
]
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

evaluate_linear_model(best_rbf_svr_model, X_val_rbf_svr, y_val_svr, "RBF SVR Model")
evaluate_linear_model(best_linear_svr_model, X_val_linear_svr, y_val_linear, "LinearSVR Model")

conn.close()