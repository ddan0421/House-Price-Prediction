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

cv = KFold(n_splits=10, shuffle=True, random_state=random_state)



############################################## LightGBM Regressor Model ############################################################
lgbm = lgb.LGBMRegressor(random_state=random_state, objective="regression", verbose=-1)

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


save_df(conn, X_train_lgbm, "X_train_lgbm")
save_df(conn, X_val_lgbm, "X_val_lgbm")
save_df(conn, test_lgbm, "test_lgbm")


evaluate_model(final_model_lgbm, X_val_lgbm, y_val, "LGBM (GridSearch)")


