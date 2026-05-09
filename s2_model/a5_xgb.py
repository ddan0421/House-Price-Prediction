import os
import pickle
import numpy as np
import duckdb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import root_mean_squared_error
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


save_df(conn, X_train_xgb, "X_train_xgb")
save_df(conn, X_val_xgb, "X_val_xgb")
save_df(conn, test_xgb, "test_xgb")


evaluate_model(final_model_xgb, X_val_xgb, y_val, "XGBoost (GridSearch)")

