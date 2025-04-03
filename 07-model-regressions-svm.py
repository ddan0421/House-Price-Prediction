import pandas as pd
import numpy as np
import statsmodels.api as sm
import models
import duckdb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.svm import SVR
import pickle


X_train = pd.read_csv("data/model_data/X_train.csv")
X_val = pd.read_csv("data/model_data/X_val.csv")
test_final = pd.read_csv("data/model_data/test_final.csv")
y_train = pd.read_csv("data/model_data/y_train.csv")
y_val = pd.read_csv("data/model_data/y_val.csv")

random_state = 42

X_train = sm.add_constant(X_train)
X_val = sm.add_constant(X_val)

############################## Feature Selection 1: VIF ##############################
vif_data = pd.DataFrame()
vif_data["feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif_data = vif_data[vif_data["feature"] != "const"]
vif_data = vif_data.dropna()
vif_data = vif_data[vif_data["VIF"] != np.inf]
selected_features_vif = list(vif_data[vif_data["VIF"] < 10]["feature"])

############################# Feature Selection 2: Random Forest Feature Importance #############################
X_train = pd.read_csv("data/model_data/X_train.csv")
X_val = pd.read_csv("data/model_data/X_val.csv")
y_train = pd.read_csv("data/model_data/y_train.csv")
y_val = pd.read_csv("data/model_data/y_val.csv")

# Train a Random Forest Regressor
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


############################# Feature Selection 3: Selected Numeric Variables based on Correlation Matrix and Domain Knowledge #############################
selected_numeric_features = [
    "log_LotArea", "cbrt_MasVnrArea", "sqrt_TotalBsmtSF", "log_1stFlrSF", 
    "log_GrLivArea", "BsmtFullBath", "FullBath", "HalfBath", "BedroomAbvGr", 
    "KitchenAbvGr", "Fireplaces", "GarageCars", "GarageArea", "sqrt_WoodDeckSF", 
    "cbrt_OpenPorchSF", "EnclosedPorch", "Age_House", "TotRmsAbvGrd"
]


# Combine the three lists and remove duplicates using a set
combined_features = list(set(selected_features_vif + selected_features_rf + selected_numeric_features))
combined_features.sort()



############################# Linear Regression #############################
X_train_regress = sm.add_constant(X_train[combined_features])
X_val_regress = sm.add_constant(X_val[combined_features])

ols_lr = models.sm_ols(X_train_regress, y_train)
glm_lr = models.sm_glm_gaussian(X_train_regress, y_train)
glm_lr_constrained = models.constrained_sm_glm_gaussian(X_train_regress, y_train, glm_lr, 0.05)

############################# Regularized Regression Models #############################
ml_features = glm_lr_constrained.params.index[(glm_lr_constrained.params != 0) & (glm_lr_constrained.params.index != "const")].to_list()
X_train_ml = X_train[ml_features]
X_val_ml = X_val[ml_features]

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

gs_ridge.fit(X_train_ml, y_train)

print("10-Fold CV RMSE (log-transformed scale):", -gs_ridge.best_score_) 
print("Optimal Parameter:", gs_ridge.best_params_)
print("Optimal Estimator:", gs_ridge.best_estimator_)

final_model_ridge = gs_ridge.best_estimator_

############################# Lasso Regression #############################
cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
lasso = Lasso()

param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
}

# Hyperparameter tuning for Lasso Regression
gs_lasso = GridSearchCV(estimator=lasso,
                        param_grid=param_grid,
                        scoring="neg_root_mean_squared_error", 
                        cv=cv,
                        n_jobs=-1,
                        refit=True)

gs_lasso.fit(X_train_ml, y_train)

print("10-Fold CV RMSE:", -gs_lasso.best_score_) 
print("Optimal Parameter:", gs_lasso.best_params_)
print("Optimal Estimator:", gs_lasso.best_estimator_)

final_model_lasso = gs_lasso.best_estimator_

# Extract the selected features based on non-zero coefficients from Lasso regression
selected_features_lasso = X_train_ml.columns[final_model_lasso.coef_.flatten() != 0]
print("Selected features for Lasso:")
print(selected_features_lasso)

############################# SVM #############################
cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
svm = SVR()

param_grid = {
    "C": [0.1, 1.0, 10.0, 100.0],  # Regularization parameter
    "epsilon": [0.001, 0.01, 0.1, 0.5, 1.0],  # Epsilon parameter (for margin of tolerance)
    "gamma": ["scale", "auto", 0.01, 0.1, 1.0]
}

gs_svm = GridSearchCV(estimator=svm,
                      param_grid=param_grid,
                      scoring="neg_root_mean_squared_error", 
                      cv=cv,
                      n_jobs=-1,
                      refit=True)

gs_svm.fit(X_train_ml, y_train.values.ravel())

print("10-Fold CV RMSE:", -gs_svm.best_score_) 
print("Optimal Parameter:", gs_svm.best_params_)
print("Optimal Estimator:", gs_svm.best_estimator_)

final_model_svm = gs_svm.best_estimator_

# Save the trained model for future use (stacking)
with open("final_model_svm.pkl", "wb") as f:
    pickle.dump(final_model_svm, f)
print("SVM model saved to final_model_svm.pkl")

X_train_ml.to_csv("data/model_data/X_train_svm.csv", index=False)
y_train.to_csv("data/model_data/y_train_svm.csv", index=False)
X_val_ml.to_csv("data/model_data/X_val_svm.csv", index=False)

############################################## Models Generalization Performance ##############################################
def evaluate_linear_model(model, X, y, name):
    predictions = model.predict(X)
    rmse = root_mean_squared_error(y, predictions)
    print(f"{name} Performance:")
    print(f"Root Mean Squared Error: {rmse:.4f}")

evaluate_linear_model(ols_lr, X_val_regress, y_val, "OLS Model")
evaluate_linear_model(glm_lr, X_val_regress, y_val, "GLM Gaussian Model")
evaluate_linear_model(glm_lr_constrained, X_val_regress, y_val, "Constrained GLM Gaussian Model")
evaluate_linear_model(final_model_ridge, X_val_ml, y_val, "Ridge Model")
evaluate_linear_model(final_model_lasso, X_val_ml, y_val, "Lasso Model")
evaluate_linear_model(final_model_svm, X_val_ml, y_val, "SVM Model")


