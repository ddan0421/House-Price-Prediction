# Tree-based models
# use non-scaled data
import pandas as pd
import numpy as np
import duckdb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
import lightgbm as lgb

X_train = pd.read_csv("data/model_data/X_train_ml.csv")
X_val = pd.read_csv("data/model_data/X_val_ml.csv")
test_final = pd.read_csv("data/model_data/test_final_ml.csv")
y_train = pd.read_csv("data/model_data/y_train_ml.csv")
y_val = pd.read_csv("data/model_data/y_val_ml.csv")


###################################################################### Feature Selection ######################################################################
# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
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
selected_features_basic_rf = conn.execute(query).fetch_df()["feature"].to_list()
conn.close()


selected_numeric_features = [
    "LotArea", "MasVnrArea", "TotalBsmtSF", "1stFlrSF", 
    "GrLivArea", "BsmtFullBath", "FullBath", "HalfBath", "BedroomAbvGr", 
    "KitchenAbvGr", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", 
    "OpenPorchSF", "EnclosedPorch", "Age_House", "TotRmsAbvGrd"
]


# Combine the lists and remove duplicates using a set
combined_features_tree = list(set(selected_features_basic_rf + selected_numeric_features))
combined_features_tree.sort()

X_train_tree = X_train[combined_features_tree]
X_val_tree = X_val[combined_features_tree]

############################################## Decision Tree Regressor Model ############################################################
random_state = 42

cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
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
                     cv=cv,
                     n_jobs=-1,
                     refit=True)

gs_dt.fit(X_train_tree, y_train.values.ravel())

# Output the results
print("10-Fold CV RMSE:", -gs_dt.best_score_)  # RMSE is the negative value from GridSearchCV
print("Optimal Parameters:", gs_dt.best_params_)
print("Optimal Estimator:", gs_dt.best_estimator_)

final_model_dt = gs_dt.best_estimator_

# Extract the selected features based on the fitted decision tree
selected_features_dt = X_train_tree.columns[final_model_dt.feature_importances_ > 0]

print("Selected features for Decision Tree:")
print(selected_features_dt)



############################################## Random Forest Tree Regressor Model ############################################################
random_state = 42

cv = KFold(n_splits=10, shuffle=True, random_state=random_state)

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
                     cv=cv,
                     n_jobs=-1,
                     refit=True)

# Fit the grid search on the training data
gs_rf.fit(X_train_tree, y_train.values.ravel())

# Output the results
print("10-Fold CV RMSE:", -gs_rf.best_score_)  # RMSE is the negative value from GridSearchCV
print("Optimal Parameters:", gs_rf.best_params_)
print("Optimal Estimator:", gs_rf.best_estimator_)

final_model_rf = gs_rf.best_estimator_

# Extract the feature importances from the fitted random forest
selected_features_rf = X_train_tree.columns[final_model_rf.feature_importances_ > 0]

print("Selected features for Random Forest:")
print(selected_features_rf)


############################################## XGBoost Regressor Model ############################################################
# Feature Selection based on a basic XGBoost model
# Initialize and fit the basic XGBoost model
random_state = 42
basic_xgb = XGBRegressor(random_state=random_state, objective="reg:squarederror", n_estimators=200)
basic_xgb.fit(X_train, y_train.values.ravel())

# Extract feature importances
feature_importances = basic_xgb.feature_importances_
selected_features_xgb = X_train.columns[np.array(feature_importances) > 0].to_list()  # Select non-zero importance features

combined_features_xgb = list(set(selected_features_xgb + selected_numeric_features + selected_features_basic_rf))
combined_features_xgb.sort()

# Reduce the feature set
X_train_xgb = X_train[combined_features_xgb]
X_val_xgb = X_val[combined_features_xgb]
print("Selected Features:", combined_features_xgb)


# Perform GridSearchCV on the reduced feature set
cv = KFold(n_splits=10, shuffle=True, random_state=random_state)

xgb = XGBRegressor(random_state=random_state, objective="reg:squarederror", n_estimators=200)

# Parameter grid for hyperparameter tuning
param_grid = {
    "learning_rate": [0.05, 0.1],        # Tune learning_rate to balance overfitting
    "max_depth": [3, 5],                     # Typical values for tree depth
    "min_child_weight": [1, 5],              # Vary min_child_weight to control overfitting
    "subsample": [0.8, 1.0],               # Subsample to prevent overfitting
    "colsample_bytree": [0.8, 1.0],        # Column subsampling to control model complexity
}

# Use GridSearchCV for hyperparameter tuning
gs_xgb = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    refit=True)

# Fit the model
gs_xgb.fit(X_train_xgb, y_train.values.ravel())

# Display results from the best model
print("10-Fold CV RMSE:", -gs_xgb.best_score_)  # Convert negative RMSE back to positive
print("Optimal Parameters:", gs_xgb.best_params_)
print("Optimal Estimator:", gs_xgb.best_estimator_)

final_model_xgb = gs_xgb.best_estimator_

# Extract feature importances from the final model
selected_features_xgb = X_train_xgb.columns[np.array(final_model_xgb.feature_importances_) > 0]

print("Selected features for XGBoost:")
print(selected_features_xgb)




############################################## LightGBM Regressor Model ############################################################

# random_state = 42

# cv = KFold(n_splits=10, shuffle=True, random_state=random_state)

# # Initialize LightGBM Regressor
# lgbm = lgb.LGBMRegressor(random_state=random_state, objective="regression")

# # Parameter grid for hyperparameter tuning
# param_grid = {
#     "learning_rate": [0.01, 0.1],              # Typical learning rates
#     "n_estimators": [50, 100, 200],           # Fewer boosting rounds
#     "max_depth": [3, 6],                      # Commonly used tree depths
#     "min_child_weight": [1, 5],               # Reasonable child weight values
#     "gamma": [0, 0.2],                        # Focus on fewer gamma values
#     "subsample": [0.8, 1.0],                  # Key subsample options
#     "colsample_bytree": [0.8, 1.0],           # Impactful colsample options
#     "reg_alpha": [0, 1],                      # Reasonable L1 range
#     "reg_lambda": [1, 10],                    # Reasonable L2 range
# }


# # Use GridSearchCV for hyperparameter tuning
# gs_lgbm = GridSearchCV(
#     estimator=lgbm,
#     param_grid=param_grid,
#     scoring="neg_root_mean_squared_error",
#     cv=cv,
#     n_jobs=-1,
#     refit=True)

# # Fit the model
# gs_lgbm.fit(X_train, y_train.values.ravel())

# # Display results from the best model
# print("10-Fold CV RMSE:", -gs_lgbm.best_score_)  # Convert negative RMSE back to positive
# print("Optimal Parameters:", gs_lgbm.best_params_)
# print("Optimal Estimator:", gs_lgbm.best_estimator_)

# final_model_lgbm = gs_lgbm.best_estimator_

# # Extract feature importances from the final model
# selected_features_lgbm = X_train.columns[np.array(final_model_lgbm.feature_importances_) > 0]

# print("Selected features for LightGBM:")
# print(selected_features_lgbm)





############################################## Models Generalization Performance ##############################################
def evaluate_tree_model(model, X, y, name):
    predictions = model.predict(X)
    rmse = root_mean_squared_error(y, predictions)
    print(f"{name} Performance:")
    print(f"Root Mean Squared Error: {rmse:.4f}")

evaluate_tree_model(final_model_dt, X_val_tree, y_val, "Decision Tree Regressor Model")
evaluate_tree_model(final_model_rf, X_val_tree, y_val, "Random Forest Regressor Model")
evaluate_tree_model(final_model_xgb, X_val_xgb, y_val, "XGBoost Regressor Model")

