import os
import pickle
import numpy as np
import duckdb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
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


############################################## Decision Tree Regressor Model ############################################################
X_train_tree_raw = load_df(conn, "X_train_ml")
X_val_tree_raw = load_df(conn, "X_val_ml")
test_tree_raw = load_df(conn, "test_ml")
y_train = load_df(conn, "y_train")
y_val = load_df(conn, "y_val")


dt = DecisionTreeRegressor(random_state=random_state, criterion="squared_error")

param_grid = {
    "max_depth": [10, 20, 30, 40, None],  
    "min_samples_split": [2, 5, 10, 20],  
    "min_samples_leaf": [1, 2, 5, 10],  
    "min_weight_fraction_leaf": [0.0, 0.01, 0.05],  
}

gs_dt = GridSearchCV(estimator=dt,
                     param_grid=param_grid,
                     scoring="neg_root_mean_squared_error", 
                     cv=cv,
                     n_jobs=-1,
                     refit=True)

gs_dt.fit(X_train_tree_raw, y_train.values.ravel())

print("10-Fold CV RMSE:", -gs_dt.best_score_)  
print("Optimal Parameters:", gs_dt.best_params_)
print("Optimal Estimator:", gs_dt.best_estimator_)

final_model_dt = gs_dt.best_estimator_

selected_features_dt = X_train_tree_raw.columns[np.array(final_model_dt.feature_importances_) > 0]
print("Selected features for Decision Tree:")
print(selected_features_dt)

# Save the trained model for future use (stacking)
with open("models/final_model_dt.pkl", "wb") as f:
    pickle.dump(final_model_dt, f)
print("decision tree model saved to models/final_model_dt.pkl")

evaluate_model(final_model_dt, X_val_tree_raw, y_val, "Decision Tree")


############################################## Random Forest Tree Regressor Model ############################################################
rf = RandomForestRegressor(random_state=random_state, bootstrap=True)

param_grid = {
    "n_estimators": [50, 100, 200], 
    "max_depth": [3, 5, 10], 
    "min_samples_split": [2, 5],  
    "min_samples_leaf": [1, 2], 
    "max_features": ["sqrt", "log2"],  
}

gs_rf = GridSearchCV(estimator=rf,
                     param_grid=param_grid,
                     scoring="neg_root_mean_squared_error", 
                     cv=cv,
                     n_jobs=-1,
                     refit=True)

gs_rf.fit(X_train_tree_raw, y_train.values.ravel())

print("10-Fold CV RMSE:", -gs_rf.best_score_) 
print("Optimal Parameters:", gs_rf.best_params_)
print("Optimal Estimator:", gs_rf.best_estimator_)

final_model_rf = gs_rf.best_estimator_

selected_features_rf = X_train_tree_raw.columns[np.array(final_model_rf.feature_importances_) > 0]
print("Selected features for Random Forest:")
print(selected_features_rf)

# Save the trained model for future use (stacking)
with open("models/final_model_rf.pkl", "wb") as f:
    pickle.dump(final_model_rf, f)
print("random forest model saved to models/final_model_rf.pkl")

evaluate_model(final_model_rf, X_val_tree_raw, y_val, "Random Forst Tree")


############################################## ExtraTreesRegressor Model ############################################################
et = ExtraTreesRegressor(random_state=random_state, criterion='squared_error')

param_grid = {
    "n_estimators": [50, 100, 200],  
    "max_depth": [3, 5, 10],  
    "min_samples_split": [2, 5],  
    "min_samples_leaf": [1, 2], 
    "max_features": ["sqrt", "log2", None], 
    "bootstrap": [True, False]  
}

gs_et = GridSearchCV(estimator=et, 
                     param_grid=param_grid, 
                     scoring="neg_root_mean_squared_error", 
                     cv=cv, 
                     n_jobs=-1, 
                     refit=True)

gs_et.fit(X_train_tree_raw, y_train.values.ravel())

print("10-Fold CV RMSE:", -gs_et.best_score_)  
print("Optimal Parameters:", gs_et.best_params_)
print("Optimal Estimator:", gs_et.best_estimator_)

final_model_et = gs_et.best_estimator_

# Save the trained model for future use (stacking)
with open("models/final_model_et.pkl", "wb") as f:
    pickle.dump(final_model_et, f)
print("ExtraTreesRegressor model saved to models/final_model_et.pkl")

evaluate_model(final_model_et, X_val_tree_raw, y_val, "Extra Trees Regressor")


