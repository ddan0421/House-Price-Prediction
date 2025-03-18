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





def evaluate_model(model, X, y, name):
    predictions = model.predict(X)
    predictions = np.expm1(predictions)
    y_actual = np.expm1(y)
    rmse = root_mean_squared_error(y_actual, predictions)
    print(f"{name} Performance:")
    print(f"Root Mean Squared Error: {rmse:.4f}")

evaluate_model(final_model_dt, X_val, y_val, "Decision Tree Model")
evaluate_model(final_model_rf, X_val, y_val, "GLM Gaussian Model")

