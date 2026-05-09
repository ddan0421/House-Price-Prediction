import duckdb
import pickle
import os
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from s1_data.db_utils import load_df, save_df
from s2_model.models import sm_ols
from s3_validation.model_evaluation import evaluate_model

base_folder = "data"
database = "AmesHousePrice.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=False)

X_train = load_df(conn, "X_train_reg")
X_val = load_df(conn, "X_val_reg")
test_final = load_df(conn, "test_reg")
y_train = load_df(conn, "y_train")
y_val = load_df(conn, "y_val")

random_state = 42

lr_features = [
    "Age_House", "BedroomAbvGr", "BsmtExposure_encoded", "BsmtFullBath",
    "CentralAir_Electrical_N_SBrkr", "EnclosedPorch", "ExterQual_encoded",
    "Exterior1st_Exterior2nd_BrkFace", "Exterior1st_Exterior2nd_BrkFace_Wd Sdng",
    "FinishedAreaPct", "Fireplaces", "Foundation_PConc", "FullBath",
    "Functional_encoded", "GarageArea", "GarageCars", "Garage_AgeCars",
    "Garage_Space", "HalfBath", "KitchenAbvGr", "KitchenQual_encoded",
    "Living_Rooms", "Neighborhood_Condition_BrkSide_Norm",
    "Neighborhood_Condition_Crawfor_Norm", "Neighborhood_Condition_Edwards_PosN",
    "Neighborhood_Condition_NoRidge_Norm", "Neighborhood_Condition_NridgHt_Norm",
    "Neighborhood_Condition_Somerst_Norm", "Neighborhood_Condition_StoneBr_Norm",
    "OverallCond", "OverallQual", "Porch_Age", "Ratio_2ndFlr_Living",
    "Ratio_Bedroom_Rooms", "RoofStyle_RoofMatl_Hip_ClyTile",
    "SaleCondition_Normal", "SaleType_New", "ScreenPorch", "TotRmsAbvGrd",
    "cbrt_MasVnrArea", "cbrt_OpenPorchSF",
    "log_1stFlrSF", "log_2ndFlrSF", "log_GrLivArea", "log_LotArea",
    "log_Yrs_Since_Remodel", "sqrt_BsmtFinSF1", "sqrt_TotalBsmtSF",
    "sqrt_WoodDeckSF"
]

X_train_reg = X_train[lr_features]
X_val_reg = X_val[lr_features]

############################# Linear Regression #############################
ols_lr = sm_ols(sm.add_constant(X_train_reg), y_train)

############################# Ridge Regression #############################
cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
ridge = Ridge()

param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0] 
}

gs_ridge = GridSearchCV(estimator=ridge,
                        param_grid=param_grid,
                        scoring="neg_root_mean_squared_error", 
                        cv=cv,
                        n_jobs=-1,
                        refit=True)

gs_ridge.fit(X_train_reg, y_train.values.ravel())

print("10-Fold CV RMSE (log-transformed scale):", -gs_ridge.best_score_) 
print("Optimal Parameter:", gs_ridge.best_params_)
print("Optimal Estimator:", gs_ridge.best_estimator_)

final_model_ridge = gs_ridge.best_estimator_

# Save the trained model for future use (stacking)
with open("models/final_model_ridge.pkl", "wb") as f:
    pickle.dump(final_model_ridge, f)
print("Ridge model saved to models/final_model_ridge.pkl")


############################# Lasso Regression #############################
cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
lasso = Lasso()

param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
}

gs_lasso = GridSearchCV(estimator=lasso,
                        param_grid=param_grid,
                        scoring="neg_root_mean_squared_error", 
                        cv=cv,
                        n_jobs=-1,
                        refit=True)

gs_lasso.fit(X_train_reg, y_train.values.ravel())

print("10-Fold CV RMSE:", -gs_lasso.best_score_) 
print("Optimal Parameter:", gs_lasso.best_params_)
print("Optimal Estimator:", gs_lasso.best_estimator_)

final_model_lasso = gs_lasso.best_estimator_

# Extract the selected features based on non-zero coefficients from Lasso regression
selected_features_lasso = X_train_reg.columns[final_model_lasso.coef_.flatten() != 0]
print("Selected features for Lasso:")
print(selected_features_lasso)

# Save the trained model for future use (stacking)
with open("models/final_model_lasso.pkl", "wb") as f:
    pickle.dump(final_model_lasso, f)
print("Lasso model saved to models/final_model_lasso.pkl")



############################# ElasticNet #############################
cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
enet = ElasticNet(random_state=random_state)

param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], 
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
}

gs_enet = GridSearchCV(estimator=enet, 
                       param_grid=param_grid, 
                       scoring="neg_root_mean_squared_error", 
                       cv=cv, 
                       n_jobs=-1, 
                       refit=True)

gs_enet.fit(X_train_reg, y_train.values.ravel())

print("10-Fold CV RMSE:", -gs_enet.best_score_) 
print("Optimal Parameter:", gs_enet.best_params_)
print("Optimal Estimator:", gs_enet.best_estimator_)

final_model_enet = gs_enet.best_estimator_

with open("models/final_model_enet.pkl", "wb") as f:
    pickle.dump(final_model_enet, f)
print("ElasticNet model saved to models/final_model_enet.pkl")



# Save data for future use (stacking)
save_df(conn, X_train_reg, "X_train_reg_lr")
save_df(conn, X_val_reg, "X_val_reg_lr")

# Evaluate performance on X_val
evaluate_model(ols_lr, sm.add_constant(X_val_reg), y_val, "OLS Model")
evaluate_model(final_model_ridge, X_val_reg, y_val, "Ridge Model")
evaluate_model(final_model_lasso, X_val_reg, y_val, "Lasso Model")
evaluate_model(final_model_enet, X_val_reg, y_val, "ElasticNet Model")
