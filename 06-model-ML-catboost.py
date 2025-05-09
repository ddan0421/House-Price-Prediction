import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import root_mean_squared_error
import pickle
import optuna
import catboost as cb

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

random_state = 42
seed = 42

X_train_cat = pd.read_csv("data/model_data/X_train_cat.csv")
X_val_cat = pd.read_csv("data/model_data/X_val_cat.csv")
y_train_cat = pd.read_csv("data/model_data/y_train_cat.csv")
y_val_cat = pd.read_csv("data/model_data/y_val_cat.csv")

cat_columns = X_train_cat.select_dtypes(include="object").columns.tolist()
cat_columns.append("MSSubClass")

train_pool = cb.Pool(data=X_train_cat, label=y_train_cat, cat_features=cat_columns)
val_pool = cb.Pool(data=X_val_cat, label=y_val_cat, cat_features=cat_columns)

############################################## CatBoost Model with Optuna Optimization ############################################################
# Define objective function for Optuna
def objective(trial):
    params = {
        "iterations": 1000,  
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),  
        "depth": trial.suggest_int("depth", 4, 8),  
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),  
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),  
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 10), 
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 5.0),  
        "random_strength": trial.suggest_float("random_strength", 0.5, 1.5), 
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["MVS", "Bernoulli"]),
        "loss_function": "RMSE",
        "early_stopping_rounds": 50,
        "random_seed": 42,  
    }

    if params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

    cv_results = cb.cv(
        params=params,
        pool=train_pool,
        partition_random_seed=seed,
        fold_count=10, 
        verbose=False,
        early_stopping_rounds=50,
        stratified=False,
    )

    best_rmse = cv_results["test-RMSE-mean"].min()
    return best_rmse

# Run Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, n_jobs=-1)

print("Best trial RMSE:", study.best_value)
print("Optimal Parameters:", study.best_params)

best_params = study.best_params
final_model_cat_optuna = cb.CatBoostRegressor(**best_params, random_seed=42)
final_model_cat_optuna.fit(train_pool, eval_set=val_pool, verbose=True)

# Save the final model
with open("final_model_catboost_optuna.pkl", "wb") as f:
    pickle.dump(final_model_cat_optuna, f)
print("CatBoost model saved to final_model_catboost_optuna.pkl")

final_model_cat_optuna.save_model("final_model_catboost_optuna.cbm", format="cbm")
print("CatBoost model saved to final_model_catboost_optuna.cbm")



############################################## CatBoost Model with Grid Search CV ############################################################
param_grid = {
    "iterations": [400, 600, 800, 1000, 1200],  
    "depth": [3, 5, 7, 9] 
}

final_model_cat_gridsearch = cb.CatBoostRegressor(loss_function="RMSE", random_seed=seed, cat_features=cat_columns)
cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
grid_search_result = final_model_cat_gridsearch.grid_search(param_grid=param_grid, 
                                       X=X_train_cat,
                                       y=y_train_cat,
                                       partition_random_seed=seed, 
                                       cv=cv,  
                                       verbose=True,  
                                       plot=False,
                                       refit=True
                                       )

print("Best RMSE:", min(grid_search_result["cv_results"]["test-RMSE-mean"]))
print("Best Parameters:", grid_search_result["params"])

# Save the final model
with open("final_model_catboost_gridsearch.pkl", "wb") as f:
    pickle.dump(final_model_cat_gridsearch, f)
print("CatBoost model saved to final_model_catboost_gridsearch.pkl")

final_model_cat_gridsearch.save_model("final_model_catboost_gridsearch.cbm", format="cbm")
print("CatBoost model saved to final_model_catboost_gridsearch.cbm")

############################################## Basic CatBoost Model ############################################################
final_model_cat_basic = cb.CatBoostRegressor(loss_function="RMSE", random_seed=42)
final_model_cat_basic.fit(train_pool, eval_set=val_pool, verbose=True)

# Save the final model
with open("final_model_catboost_basic.pkl", "wb") as f:
    pickle.dump(final_model_cat_basic, f)
print("CatBoost model saved to final_model_catboost_basic.pkl")

# Save model in CatBoost's own format
final_model_cat_basic.save_model("final_model_catboost_basic.cbm", format="cbm")
print("CatBoost model saved to final_model_catboost_basic.cbm")


############################################## Models Generalization Performance ##############################################
def evaluate_tree_model(model, X, y, name):
    predictions = model.predict(X)
    rmse = root_mean_squared_error(y, predictions)
    print(f"{name} Performance:")
    print(f"Root Mean Squared Error: {rmse:.4f}")

print("############################################## 10-Fold CatBoost CV Optuna-Tuned ##############################################")
evaluate_tree_model(final_model_cat_optuna, val_pool, y_val_cat, "CatBoost Regressor (Optuna-Tuned)")
print("############################################## 10-Fold CatBoost CV Grid Search ##############################################")
evaluate_tree_model(final_model_cat_gridsearch, val_pool, y_val_cat, "CatBoost Regressor (Grid Search)")
print("############################################## Basic CatBoost ##############################################")
evaluate_tree_model(final_model_cat_basic, val_pool, y_val_cat, "CatBoost Regressor (Grid Search)")