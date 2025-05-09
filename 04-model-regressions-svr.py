import pandas as pd
import numpy as np
import statsmodels.api as sm
import models
import duckdb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR
import pickle


X_train = pd.read_csv("data/model_data/X_train_reg.csv")
X_val = pd.read_csv("data/model_data/X_val_reg.csv")
test_final = pd.read_csv("data/model_data/test_final_reg.csv")
y_train = pd.read_csv("data/model_data/y_train_reg.csv")
y_val = pd.read_csv("data/model_data/y_val_reg.csv")

random_state = 42

############################# Feature Selection: Stepwise Selection for OLS Regression #############################
# Stepwise Selection for OLS Regression
selected_features_stepwise = models.ols_stepwise_selection(X_train, y_train, threshold_in=0.01, threshold_out=0.05)

############################# Feature Selection: Selected Numeric Variables based on Correlation Matrix and Domain Knowledge #############################
selected_numeric_features = [
    "log_LotArea", "cbrt_MasVnrArea", "sqrt_TotalBsmtSF", "log_1stFlrSF", 
    "log_GrLivArea", "BsmtFullBath", "FullBath", "HalfBath", "BedroomAbvGr", 
    "KitchenAbvGr", "Fireplaces", "GarageCars", "GarageArea", "sqrt_WoodDeckSF", 
    "cbrt_OpenPorchSF", "EnclosedPorch", "Age_House", "TotRmsAbvGrd",
    "Living_Rooms", "Garage_Space", "Garage_AgeCars", "Porch_Age", "log_Yrs_Since_Remodel", "Ratio_Bedroom_Rooms", "Ratio_2ndFlr_Living", "log_2ndFlrSF"
]

################################# Result from Lasso Regression (input data: selected_features_stepwise) #############################
lasso = ['OverallQual', 'log_GrLivArea', 'Age_House', 'log_LotArea',
       'OverallCond', 'sqrt_BsmtFinSF1', 'RoofStyle_RoofMatl_Hip_ClyTile',
       'Neighborhood_Condition_Edwards_PosN', 'log_Age_Garage',
       'sqrt_TotalBsmtSF', 'KitchenQual_encoded',
       'Neighborhood_Condition_Crawfor_Norm', 'Garage_Space', 'Fireplaces',
       'Foundation_PConc', 'Exterior1st_Exterior2nd_BrkFace_Wd Sdng',
       'Neighborhood_Condition_StoneBr_Norm', 'Functional_Typ',
       'BsmtExposure_encoded', 'KitchenAbvGr', 'SaleType_New',
       'SaleCondition_Normal', 'BsmtFullBath',
       'Neighborhood_Condition_NridgHt_Norm', 'PoolQC_encoded',
       'CentralAir_Electrical_N_SBrkr', 'Neighborhood_Condition_NoRidge_Norm',
       'Neighborhood_Condition_Somerst_Norm',
       'Exterior1st_Exterior2nd_BrkFace',
       'Neighborhood_Condition_BrkSide_Norm', 'MSSubClass_MSZoning_50_RM',
       'MSSubClass_MSZoning_160_RM', 'Neighborhood_Condition_ClearCr_Norm',
       'ScreenPorch', 'sqrt_WoodDeckSF']

# Combine the four lists and remove duplicates using a set
combined_features_lr = list(set(selected_numeric_features + lasso))
combined_features_lr.sort()

############################# Linear Regression #############################
# Use only the selected features from stepwise selection method for the linear regression model
X_train_regress = sm.add_constant(X_train[combined_features_lr])
X_val_regress = sm.add_constant(X_val[combined_features_lr])

ols_lr = models.sm_ols(X_train_regress, y_train)

############################# Regularized Regression Models #############################
X_train_reg_lr = X_train[combined_features_lr]
X_val_reg_lr  = X_val[combined_features_lr]

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

gs_ridge.fit(X_train_reg_lr, y_train)

print("10-Fold CV RMSE (log-transformed scale):", -gs_ridge.best_score_) 
print("Optimal Parameter:", gs_ridge.best_params_)
print("Optimal Estimator:", gs_ridge.best_estimator_)

final_model_ridge = gs_ridge.best_estimator_

# Save the trained model for future use (stacking)
with open("final_model_ridge.pkl", "wb") as f:
    pickle.dump(final_model_ridge, f)
print("Ridge model saved to final_model_ridge.pkl")

X_train_reg_lr.to_csv("data/model_data/X_train_ridge.csv", index=False)
y_train.to_csv("data/model_data/y_train_ridge.csv", index=False)
X_val_reg_lr.to_csv("data/model_data/X_val_ridge.csv", index=False)

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

gs_lasso.fit(X_train_reg_lr, y_train)

print("10-Fold CV RMSE:", -gs_lasso.best_score_) 
print("Optimal Parameter:", gs_lasso.best_params_)
print("Optimal Estimator:", gs_lasso.best_estimator_)

final_model_lasso = gs_lasso.best_estimator_

# Extract the selected features based on non-zero coefficients from Lasso regression
selected_features_lasso = X_train_reg_lr.columns[final_model_lasso.coef_.flatten() != 0]
print("Selected features for Lasso:")
print(selected_features_lasso)

# Save the trained model for future use (stacking)
with open("final_model_lasso.pkl", "wb") as f:
    pickle.dump(final_model_lasso, f)
print("Lasso model saved to final_model_lasso.pkl")

X_train_reg_lr.to_csv("data/model_data/X_train_lasso.csv", index=False)
y_train.to_csv("data/model_data/y_train_lasso.csv", index=False)
X_val_reg_lr.to_csv("data/model_data/X_val_lasso.csv", index=False)

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

gs_enet.fit(X_train_reg_lr, y_train.values.ravel())

print("10-Fold CV RMSE:", -gs_enet.best_score_) 
print("Optimal Parameter:", gs_enet.best_params_)
print("Optimal Estimator:", gs_enet.best_estimator_)

final_model_enet = gs_enet.best_estimator_

with open("final_model_enet.pkl", "wb") as f:
    pickle.dump(final_model_enet, f)

X_train_reg_lr.to_csv("data/model_data/X_train_enet.csv", index=False)
y_train.to_csv("data/model_data/y_train_enet.csv", index=False)
X_val_reg_lr.to_csv("data/model_data/X_val_enet.csv", index=False)

############################# SVM #############################
X_train = pd.read_csv("data/model_data/X_train_reg.csv")
X_val = pd.read_csv("data/model_data/X_val_reg.csv")
y_train = pd.read_csv("data/model_data/y_train_reg.csv")
y_val = pd.read_csv("data/model_data/y_val_reg.csv")
##### Feature Selection 5: VIF and Random Forest #####
X_train = sm.add_constant(X_train)
X_val = sm.add_constant(X_val)

########## Separate Feature Selection for SVR ##########
############################## Feature Selection 1: VIF ##############################
vif_data = pd.DataFrame()
vif_data["feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif_data = vif_data[vif_data["feature"] != "const"]
vif_data = vif_data.dropna()
vif_data = vif_data[vif_data["VIF"] != np.inf]
selected_features_vif = list(vif_data[vif_data["VIF"] < 10]["feature"])

############################# Feature Selection 2: Random Forest Feature Importance #############################
X_train = pd.read_csv("data/model_data/X_train_reg.csv")
X_val = pd.read_csv("data/model_data/X_val_reg.csv")
y_train = pd.read_csv("data/model_data/y_train_reg.csv")
y_val = pd.read_csv("data/model_data/y_val_reg.csv")

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
selected_features_rf = conn.execute(query).fetch_df()["feature"].to_list()
conn.close()
#################################### combine all features ####################################
# Combine the selected features from VIF, Random Forest, numeric features, and stepwise selection
combined_features_svm = list(set(selected_features_vif + selected_features_rf + selected_numeric_features + selected_features_stepwise))
combined_features_svm.sort()

X_train_svm = X_train[combined_features_svm]
X_val_svm = X_val[combined_features_svm]

cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
svm = SVR(kernel="rbf")

param_grid = {
    "C": [0.1, 1.0, 10.0, 100.0], 
    "epsilon": [0.01, 0.1, 1.0],
    "gamma": ["scale", "auto", 0.01, 0.1, 1.0], 
    "tol": [1e-4, 1e-3] 
}


gs_svm = GridSearchCV(estimator=svm,
                      param_grid=param_grid,
                      scoring="neg_root_mean_squared_error", 
                      cv=cv,
                      n_jobs=-1,
                      refit=True)

gs_svm.fit(X_train_svm, y_train.values.ravel())

print("10-Fold CV RMSE:", -gs_svm.best_score_) 
print("Optimal Parameter:", gs_svm.best_params_)
print("Optimal Estimator:", gs_svm.best_estimator_)

final_model_svm = gs_svm.best_estimator_

# Save the trained model for future use (stacking)
with open("final_model_svm.pkl", "wb") as f:
    pickle.dump(final_model_svm, f)
print("SVM model saved to final_model_svm.pkl")

X_train_svm.to_csv("data/model_data/X_train_svm.csv", index=False)
y_train.to_csv("data/model_data/y_train_svm.csv", index=False)
X_val_svm.to_csv("data/model_data/X_val_svm.csv", index=False)



############################# KNN #############################
X_train_knn = X_train[combined_features_svm]
X_val_knn = X_val[combined_features_svm]

cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
knn = KNeighborsRegressor()

param_grid = {
    "n_neighbors": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  
    "weights": ["uniform", "distance"],
    "p": [1, 2], 
    "algorithm": ["auto", "ball_tree", "kd_tree"],  
    "leaf_size": [20, 30, 40]  
}
gs_knn = GridSearchCV(estimator=knn,
                      param_grid=param_grid,
                      scoring="neg_root_mean_squared_error",
                      cv=cv,
                      n_jobs=-1,
                      refit=True)

gs_knn.fit(X_train_knn, y_train.values.ravel())

print("10-Fold CV RMSE:", -gs_knn.best_score_)
print("Optimal Parameter:", gs_knn.best_params_)
print("Optimal Estimator:", gs_knn.best_estimator_)

final_model_knn = gs_knn.best_estimator_

with open("final_model_knn.pkl", "wb") as f:
    pickle.dump(final_model_knn, f)
print("KNN model saved to final_model_knn.pkl")

X_train_svm.to_csv("data/model_data/X_train_knn.csv", index=False)
y_train.to_csv("data/model_data/y_train_knn.csv", index=False)
X_val_knn.to_csv("data/model_data/X_val_knn.csv", index=False)

############################################## Models Generalization Performance ##############################################
def evaluate_linear_model(model, X, y, name):
    predictions = model.predict(X)
    rmse = root_mean_squared_error(y, predictions)
    print(f"{name} Performance:")
    print(f"Root Mean Squared Error: {rmse:.4f}")

evaluate_linear_model(ols_lr, X_val_regress, y_val, "OLS Model")
evaluate_linear_model(final_model_ridge, X_val_reg_lr, y_val, "Ridge Model")
evaluate_linear_model(final_model_lasso, X_val_reg_lr, y_val, "Lasso Model")
evaluate_linear_model(final_model_enet, X_val_reg_lr, y_val, "ElasticNet Model")
evaluate_linear_model(final_model_svm, X_val_svm, y_val, "SVM Model")
evaluate_linear_model(final_model_knn, X_val_knn, y_val, "KNN Model")



