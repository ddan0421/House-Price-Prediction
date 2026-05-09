import os
import pickle

import duckdb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor

from s1_data.db_utils import load_df, save_df
from s3_validation.model_evaluation import evaluate_model


base_folder = "data"
database = "AmesHousePrice.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=False)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

############################# KNN #############################
X_train_knn_raw = load_df(conn, "X_train_knn")
X_val_knn_raw = load_df(conn, "X_val_knn")
test_knn_raw = load_df(conn, "test_knn")
y_train = load_df(conn, "y_train")
y_val = load_df(conn, "y_val")


knn_features = [
    "log_LotFrontage", "log_LotArea", "OverallQual", "OverallCond", "cbrt_MasVnrArea",
    "sqrt_BsmtFinSF1",
    "BsmtFinSF2", "sqrt_BsmtUnfSF", "sqrt_TotalBsmtSF", "log_1stFlrSF", "log_2ndFlrSF", "log_GrLivArea",
    "BsmtFullBath", "FullBath", "HalfBath", "TotRmsAbvGrd", "Fireplaces", "GarageCars",
    "GarageArea", "sqrt_WoodDeckSF", "cbrt_OpenPorchSF", "EnclosedPorch", "ScreenPorch", "Age_House",
    "log_Yrs_Since_Remodel", "log_Age_Garage", "ExterQual_encoded", "BsmtQual_encoded",
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

X_train_knn = X_train_knn_raw[knn_features]
X_val_knn = X_val_knn_raw[knn_features]
test_knn = test_knn_raw[knn_features]

knn = KNeighborsRegressor()

param_grid = {
    "n_neighbors": range(3, 51),
    "weights": ["uniform", "distance"],
    "p": [1, 2, 3]
}
gs_knn = GridSearchCV(estimator=knn,
                      param_grid=param_grid,
                      scoring="neg_root_mean_squared_error",
                      cv=cv,
                      n_jobs=-1,
                      refit=True)

gs_knn.fit(X_train_knn, y_train.values.ravel())

print("KNN 10-Fold CV RMSE:", -gs_knn.best_score_)
print("KNN Optimal Parameter:", gs_knn.best_params_)
print("KNN Optimal Estimator:", gs_knn.best_estimator_)

best_knn_model = gs_knn.best_estimator_
with open("models/final_model_knn.pkl", "wb") as f:
    pickle.dump(best_knn_model, f)
print("KNN model saved to models/final_model_knn.pkl")

save_df(conn, X_train_knn, "X_train_knn_final")
save_df(conn, X_val_knn, "X_val_knn_final")
save_df(conn, test_knn, "test_knn_final")

evaluate_model(best_knn_model, X_val_knn, y_val, "KNN Model")

conn.close()
