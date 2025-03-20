# Tree-based models
# use non-scaled data
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import root_mean_squared_error


X_train = pd.read_csv("data/model_data/X_train_ml.csv")
X_val = pd.read_csv("data/model_data/X_val_ml.csv")
test_final = pd.read_csv("data/model_data/test_final_ml.csv")
y_train = pd.read_csv("data/model_data/y_train_ml.csv")
y_val = pd.read_csv("data/model_data/y_val_ml.csv")

selected_features = ['3SsnPorch',
 'Age_House',
 'Alley_Pave',
 'Alley_no_alley',
 'BedroomAbvGr',
 'BldgType_HouseStyle_2fmCon_2.5Fin',
 'BldgType_HouseStyle_2fmCon_SLvl',
 'BsmtCond',
 'BsmtExposure',
 'BsmtFinType1',
 'BsmtFinType2',
 'BsmtFullBath',
 'BsmtHalfBath',
 'BsmtQual',
 'CentralAir_Electrical_N_FuseF',
 'CentralAir_Electrical_N_FuseP',
 'CentralAir_Electrical_N_SBrkr',
 'CentralAir_Electrical_Y_FuseA',
 'CentralAir_Electrical_Y_FuseF',
 'CentralAir_Electrical_Y_SBrkr',
 'EnclosedPorch',
 'ExterCond',
 'ExterQual',
 'Exterior1st_Exterior2nd_AsbShng_Plywood',
 'Exterior1st_Exterior2nd_AsbShng_Stucco',
 'Exterior1st_Exterior2nd_AsphShn',
 'Exterior1st_Exterior2nd_BrkComm_Brk Cmn',
 'Exterior1st_Exterior2nd_BrkFace',
 'Exterior1st_Exterior2nd_BrkFace_AsbShng',
 'Exterior1st_Exterior2nd_BrkFace_HdBoard',
 'Exterior1st_Exterior2nd_BrkFace_Plywood',
 'Exterior1st_Exterior2nd_BrkFace_Stone',
 'Exterior1st_Exterior2nd_BrkFace_Stucco',
 'Exterior1st_Exterior2nd_BrkFace_Wd Sdng',
 'Exterior1st_Exterior2nd_BrkFace_Wd Shng',
 'Exterior1st_Exterior2nd_CBlock',
 'Exterior1st_Exterior2nd_CemntBd_CmentBd',
 'Exterior1st_Exterior2nd_CemntBd_Wd Sdng',
 'Exterior1st_Exterior2nd_CemntBd_Wd Shng',
 'Exterior1st_Exterior2nd_HdBoard_AsphShn',
 'Exterior1st_Exterior2nd_HdBoard_ImStucc',
 'Exterior1st_Exterior2nd_HdBoard_Plywood',
 'Exterior1st_Exterior2nd_HdBoard_Wd Sdng',
 'Exterior1st_Exterior2nd_HdBoard_Wd Shng',
 'Exterior1st_Exterior2nd_ImStucc',
 'Exterior1st_Exterior2nd_MetalSd_AsphShn',
 'Exterior1st_Exterior2nd_MetalSd_HdBoard',
 'Exterior1st_Exterior2nd_MetalSd_Stucco',
 'Exterior1st_Exterior2nd_MetalSd_Wd Sdng',
 'Exterior1st_Exterior2nd_MetalSd_Wd Shng',
 'Exterior1st_Exterior2nd_Plywood_Brk Cmn',
 'Exterior1st_Exterior2nd_Plywood_HdBoard',
 'Exterior1st_Exterior2nd_Plywood_ImStucc',
 'Exterior1st_Exterior2nd_Plywood_Wd Sdng',
 'Exterior1st_Exterior2nd_Stone',
 'Exterior1st_Exterior2nd_Stucco',
 'Exterior1st_Exterior2nd_Stucco_CmentBd',
 'Exterior1st_Exterior2nd_Stucco_Wd Shng',
 'Exterior1st_Exterior2nd_VinylSd_AsbShng',
 'Exterior1st_Exterior2nd_VinylSd_HdBoard',
 'Exterior1st_Exterior2nd_VinylSd_ImStucc',
 'Exterior1st_Exterior2nd_VinylSd_Other',
 'Exterior1st_Exterior2nd_VinylSd_Plywood',
 'Exterior1st_Exterior2nd_VinylSd_Wd Shng',
 'Exterior1st_Exterior2nd_Wd Sdng_AsbShng',
 'Exterior1st_Exterior2nd_Wd Sdng_HdBoard',
 'Exterior1st_Exterior2nd_Wd Sdng_ImStucc',
 'Exterior1st_Exterior2nd_Wd Sdng_Plywood',
 'Exterior1st_Exterior2nd_Wd Sdng_VinylSd',
 'Exterior1st_Exterior2nd_Wd Sdng_Wd Shng',
 'Exterior1st_Exterior2nd_WdShing_Plywood',
 'Exterior1st_Exterior2nd_WdShing_Stucco',
 'Exterior1st_Exterior2nd_WdShing_Wd Sdng',
 'Exterior1st_Exterior2nd_WdShing_Wd Shng',
 'Fence_GdWo',
 'Fence_MnPrv',
 'Fence_MnWw',
 'Fence_no_fence',
 'FireplaceQu',
 'Fireplaces',
 'Foundation_CBlock',
 'Foundation_Slab',
 'Foundation_Stone',
 'Foundation_Wood',
 'FullBath',
 'Functional_Maj2',
 'Functional_Min1',
 'Functional_Min2',
 'Functional_Mod',
 'Functional_Sev',
 'GarageArea',
 'GarageCars',
 'GarageCond',
 'GarageFinish_RFn',
 'GarageFinish_Unf',
 'GarageQual',
 'GarageType_Attchd',
 'GarageType_Basment',
 'GarageType_CarPort',
 'GarageType_Detchd',
 'HalfBath',
 'Heating_HeatingQC_GasW_Ex',
 'Heating_HeatingQC_GasW_Fa',
 'Heating_HeatingQC_GasW_Gd',
 'Heating_HeatingQC_OthW_Fa',
 'Heating_HeatingQC_Wall_Fa',
 'Heating_HeatingQC_Wall_TA',
 'KitchenAbvGr',
 'KitchenQual',
 'LotConfig_LandSlope_Corner_Mod',
 'LotConfig_LandSlope_Corner_Sev',
 'LotConfig_LandSlope_CulDSac_Gtl',
 'LotConfig_LandSlope_CulDSac_Mod',
 'LotConfig_LandSlope_CulDSac_Sev',
 'LotConfig_LandSlope_FR2_Gtl',
 'LotConfig_LandSlope_FR3_Gtl',
 'LotConfig_LandSlope_Inside_Gtl',
 'LotConfig_LandSlope_Inside_Mod',
 'LotConfig_LandSlope_Inside_Sev',
 'LotShape_LandContour_IR1_HLS',
 'LotShape_LandContour_IR1_Low',
 'LotShape_LandContour_IR2_Bnk',
 'LotShape_LandContour_IR2_HLS',
 'LotShape_LandContour_IR2_Low',
 'LotShape_LandContour_IR2_Lvl',
 'LotShape_LandContour_IR3_Bnk',
 'LotShape_LandContour_IR3_HLS',
 'LotShape_LandContour_IR3_Low',
 'LotShape_LandContour_IR3_Lvl',
 'LotShape_LandContour_Reg_Bnk',
 'LotShape_LandContour_Reg_HLS',
 'LotShape_LandContour_Reg_Low',
 'MSSubClass_MSZoning_120_RH',
 'Neighborhood_Condition_Blueste_Norm',
 'Neighborhood_Condition_BrDale_Norm',
 'Neighborhood_Condition_BrkSide_Feedr_Norm',
 'Neighborhood_Condition_BrkSide_Norm',
 'Neighborhood_Condition_BrkSide_PosN_Norm',
 'Neighborhood_Condition_BrkSide_RRAn_Feedr',
 'Neighborhood_Condition_BrkSide_RRAn_Norm',
 'Neighborhood_Condition_BrkSide_RRNn_Feedr',
 'Neighborhood_Condition_ClearCr_Feedr_Norm',
 'Neighborhood_Condition_ClearCr_Norm',
 'Neighborhood_Condition_CollgCr_PosN_Norm',
 'Neighborhood_Condition_Crawfor_PosA_Norm',
 'Neighborhood_Condition_Crawfor_PosN_Norm',
 'Neighborhood_Condition_Edwards_Artery_Norm',
 'Neighborhood_Condition_Edwards_Feedr_Norm',
 'Neighborhood_Condition_Edwards_PosN',
 'Neighborhood_Condition_Gilbert_Feedr_Norm',
 'Neighborhood_Condition_Gilbert_RRAn_Norm',
 'Neighborhood_Condition_Gilbert_RRNn_Norm',
 'Neighborhood_Condition_IDOTRR_Artery_Norm',
 'Neighborhood_Condition_IDOTRR_Feedr',
 'Neighborhood_Condition_IDOTRR_Norm',
 'Neighborhood_Condition_IDOTRR_RRNn_Norm',
 'Neighborhood_Condition_MeadowV_Norm',
 'Neighborhood_Condition_Mitchel_Feedr_Norm',
 'Neighborhood_Condition_Mitchel_Norm',
 'Neighborhood_Condition_NAmes_Artery_Norm',
 'Neighborhood_Condition_NAmes_Feedr_Norm',
 'Neighborhood_Condition_NAmes_PosA_Norm',
 'Neighborhood_Condition_NAmes_PosN_Norm',
 'Neighborhood_Condition_NPkVill_Norm',
 'Neighborhood_Condition_NWAmes_Feedr_Norm',
 'Neighborhood_Condition_NWAmes_Feedr_RRAn',
 'Neighborhood_Condition_NWAmes_Norm',
 'Neighborhood_Condition_NWAmes_PosA_Norm',
 'Neighborhood_Condition_NWAmes_PosN_Norm',
 'Neighborhood_Condition_NWAmes_RRAn_Norm',
 'Neighborhood_Condition_NoRidge_Norm',
 'Neighborhood_Condition_NridgHt_Norm',
 'Neighborhood_Condition_NridgHt_PosN',
 'Neighborhood_Condition_OldTown_Artery',
 'Neighborhood_Condition_OldTown_Artery_Norm',
 'Neighborhood_Condition_OldTown_Artery_PosA',
 'Neighborhood_Condition_OldTown_Feedr_Norm',
 'Neighborhood_Condition_OldTown_Feedr_RRNn',
 'Neighborhood_Condition_SWISU_Feedr_Norm',
 'Neighborhood_Condition_SWISU_Norm',
 'Neighborhood_Condition_SawyerW_Feedr_Norm',
 'Neighborhood_Condition_SawyerW_Norm',
 'Neighborhood_Condition_SawyerW_RRAe_Norm',
 'Neighborhood_Condition_SawyerW_RRNe_Norm',
 'Neighborhood_Condition_Sawyer_Feedr_Norm',
 'Neighborhood_Condition_Sawyer_Norm',
 'Neighborhood_Condition_Sawyer_PosN_Norm',
 'Neighborhood_Condition_Sawyer_RRAe_Norm',
 'Neighborhood_Condition_Somerst_Feedr_Norm',
 'Neighborhood_Condition_Somerst_RRAn_Norm',
 'Neighborhood_Condition_Somerst_RRNn_Norm',
 'Neighborhood_Condition_StoneBr_Norm',
 'Neighborhood_Condition_Timber_Norm',
 'Neighborhood_Condition_Veenker_Feedr_Norm',
 'Neighborhood_Condition_Veenker_Norm',
 'OverallCond',
 'OverallQual',
 'PavedDrive_P',
 'PavedDrive_Y',
 'RoofStyle_RoofMatl_Flat_Metal',
 'RoofStyle_RoofMatl_Gable_Roll',
 'RoofStyle_RoofMatl_Gable_WdShngl',
 'RoofStyle_RoofMatl_Hip_ClyTile',
 'RoofStyle_RoofMatl_Hip_WdShake',
 'RoofStyle_RoofMatl_Hip_WdShngl',
 'RoofStyle_RoofMatl_Mansard_CompShg',
 'RoofStyle_RoofMatl_Mansard_WdShake',
 'RoofStyle_RoofMatl_Shed_WdShake',
 'SaleCondition_AdjLand',
 'SaleCondition_Alloca',
 'SaleCondition_Family',
 'SaleCondition_Normal',
 'SaleType_CWD',
 'SaleType_Con',
 'SaleType_ConLD',
 'SaleType_ConLI',
 'SaleType_ConLw',
 'SaleType_Oth',
 'SaleType_WD',
 'ScreenPorch',
 'Season_Sold_Spring',
 'Season_Sold_Summer',
 'Season_Sold_Winter',
 'Street_Pave',
 'TotRmsAbvGrd',
 'Utilities_NoSeWa',
 'MasVnrArea',
 'OpenPorchSF',
 '1stFlrSF',
 '2ndFlrSF',
 'Age_Garage',
 'GrLivArea',
 'LotArea',
 'LotFrontage',
 'LowQualFinSF',
 'Yrs_Since_Remodel',
 'TotalBsmtSF',
 'WoodDeckSF']

X_train = X_train[selected_features]
X_val = X_val[selected_features]
############################################## Decision Tree Regressor Model ############################################################
random_state = 42

inner_cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
outer_cv = KFold(n_splits=10, shuffle=True, random_state=random_state)

dt = DecisionTreeRegressor(random_state=random_state, criterion="squared_error")

# Define a grid of hyperparameters to search
param_grid = {
    "max_depth": [10, 20, 30, 40, None],  # Maximum depth of the tree
    "min_samples_split": [2, 5, 10, 20],  # Minimum number of samples required to split a node
    "min_samples_leaf": [1, 2, 5, 10],  # Minimum number of samples required at a leaf node
    "min_weight_fraction_leaf": [0.0, 0.01, 0.05],  # Minimum weighted fraction of the sum of weights at a leaf node
}

gs_dt = GridSearchCV(estimator=dt,
                     param_grid=param_grid,
                     scoring="neg_root_mean_squared_error",  # Using RMSE as the evaluation metric
                     cv=inner_cv,
                     n_jobs=-1)

gs_dt.fit(X_train, y_train.values.ravel())

# Output the results
print("Non-nested CV RMSE:", -gs_dt.best_score_)  # RMSE is the negative value from GridSearchCV
print("Optimal Parameters:", gs_dt.best_params_)
print("Optimal Estimator:", gs_dt.best_estimator_)

# Nested cross-validation to estimate the performance
nested_score_gs_dt_rmse = cross_val_score(gs_dt, X=X_train, y=y_train.values.ravel(), scoring="neg_root_mean_squared_error", cv=outer_cv)
print("Nested CV RMSE:", -nested_score_gs_dt_rmse.mean(), " +/- ", nested_score_gs_dt_rmse.std())  # Multiply by -1 to get positive RMSE

# Fit the final model on the entire training data
final_model_dt = gs_dt.best_estimator_.fit(X_train, y_train.values.ravel())

# Extract the selected features based on the fitted decision tree
# For Decision Tree, we can get the feature importances instead of coefficients like in Ridge regression
selected_features_dt = X_train.columns[final_model_dt.feature_importances_ > 0]

print("Selected features for Decision Tree:")
print(selected_features_dt)



############################################## Random Forest Tree Regressor Model ############################################################
random_state = 42

inner_cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
outer_cv = KFold(n_splits=10, shuffle=True, random_state=random_state)

rf = RandomForestRegressor(random_state=random_state, bootstrap=True)

# Define a grid of hyperparameters to search
param_grid = {
    "n_estimators": [50, 100, 200],  # Fewer trees for faster training
    "max_depth": [3, 5, 10],  # Restrict depth to limit complexity
    "min_samples_split": [2, 5],  # Reduce the range for faster grid search
    "min_samples_leaf": [1, 2],  # Keep leaf size small for better granularity
    "max_features": ["sqrt", "log2"],  # Subset of features considered at each split
}

# Hyperparameter tuning for Random Forest Regressor
gs_rf = GridSearchCV(estimator=rf,
                     param_grid=param_grid,
                     scoring="neg_root_mean_squared_error",  # Using RMSE as the evaluation metric
                     cv=inner_cv,
                     n_jobs=-1)

# Fit the grid search on the training data
gs_rf.fit(X_train, y_train.values.ravel())

# Output the results
print("Non-nested CV RMSE:", -gs_rf.best_score_)  # RMSE is the negative value from GridSearchCV
print("Optimal Parameters:", gs_rf.best_params_)
print("Optimal Estimator:", gs_rf.best_estimator_)

# Nested cross-validation to estimate the performance
nested_score_gs_rf_rmse = cross_val_score(gs_rf, X=X_train, y=y_train.values.ravel(), scoring="neg_root_mean_squared_error", cv=outer_cv)
print("Nested CV RMSE:", -nested_score_gs_rf_rmse.mean(), " +/- ", nested_score_gs_rf_rmse.std())  # Multiply by -1 to get positive RMSE

# Fit the final model on the entire training data
final_model_rf = gs_rf.best_estimator_.fit(X_train, y_train.values.ravel())

# Extract the feature importances from the fitted random forest
selected_features_rf = X_train.columns[final_model_rf.feature_importances_ > 0]

print("Selected features for Random Forest:")
print(selected_features_rf)


############################################## XGBoost Regressor Model ############################################################








############################################## Models Generalization Performance ##############################################


def evaluate_tree_model(model, X, y, name):
    predictions = model.predict(X)
    rmse = root_mean_squared_error(y, predictions)
    print(f"{name} Performance:")
    print(f"Root Mean Squared Error: {rmse:.4f}")

evaluate_tree_model(final_model_dt, X_val, y_val, "Decision Tree Model")
evaluate_tree_model(final_model_rf, X_val, y_val, "GLM Gaussian Model")

