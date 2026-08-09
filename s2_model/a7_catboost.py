import os
import json
import catboost as cb
import duckdb
import pandas as pd
from bayes_opt import BayesianOptimization, acquisition

from s1_data.db_utils import load_df
from s3_validation.model_evaluation import evaluate_model

base_folder = "data"
database = "AmesHousePrice.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=False)

random_state = 42
seed = 42

X_train_cat = load_df(conn, "X_train_cat")
X_val_cat = load_df(conn, "X_val_cat")
y_train = load_df(conn, "y_train")
y_val = load_df(conn, "y_val")


nominal_cat = ["MSSubClass_MSZoning", "LotConfig_LandSlope", "Neighborhood_Condition", "BldgType_HouseStyle",
               "Exterior1st_Exterior2nd", "CentralAir_Electrical", "LotShape_LandContour", "RoofStyle_RoofMatl",
               "Heating_HeatingQC", "Alley", "MasVnrType", "Foundation", "GarageType", "PavedDrive", "Fence", 
               "MiscFeature", "SaleType", "SaleCondition", "Season_Sold"]

ordinal_cat = ["Utilities", "Functional", "OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtQual", 
               "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "KitchenQual", "FireplaceQu", 
               "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Street"]


cat_columns = nominal_cat + ordinal_cat

train_pool = cb.Pool(data=X_train_cat, label=y_train, cat_features=cat_columns)
val_pool = cb.Pool(data=X_val_cat, label=y_val, cat_features=cat_columns)

############################# CatBoost Hyperparameter Search #############################
max_iterations = 2000
search_folds = 5


def cat_cv(params, fold_count):
    # boost_from_average is set explicitly because cb.cv, unlike CatBoostRegressor.fit,
    # does not enable it for RMSE -- every fold would start at 0 and burn rounds
    # climbing to mean log price.
    cv_results = cb.cv(
        pool=train_pool,
        params={**params, "iterations": max_iterations, "allow_writing_files": False,
                "boost_from_average": True},
        fold_count=fold_count,
        partition_random_seed=seed,
        shuffle=True,
        early_stopping_rounds=50,
        logging_level="Silent",
    )
    curve = cv_results["test-RMSE-mean"]
    return curve.min(), int(curve.idxmin()) + 1


def bayesian_opt_cat(init_iter=10, n_iters=20):
    def hyp_cat(learning_rate, depth, l2_leaf_reg, random_strength, min_data_in_leaf):
        params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": random_state,
            "learning_rate": learning_rate,
            "depth": int(round(depth)),
            "l2_leaf_reg": l2_leaf_reg,
            "random_strength": random_strength,
            "min_data_in_leaf": int(round(min_data_in_leaf)),
        }
        best_rmse, _ = cat_cv(params, search_folds)
        return -best_rmse

    pds = {
        "learning_rate": (0.03, 0.2),
        "depth": (4, 6),
        "l2_leaf_reg": (1, 20),
        "random_strength": (0, 5),
        "min_data_in_leaf": (1, 30),
    }

    acq = acquisition.UpperConfidenceBound(
        kappa=1.0,
        exploration_decay=0.95,
        exploration_decay_delay=init_iter,
        random_state=random_state,
    )

    optimizer = BayesianOptimization(f=hyp_cat, pbounds=pds,
                                     acquisition_function=acq,
                                     random_state=random_state)
    optimizer.maximize(init_points=init_iter, n_iter=n_iters)
    return optimizer


results = bayesian_opt_cat()
print("Best Parameters:", results.max["params"])
print(f"Best {search_folds}-Fold CV RMSE:", -results.max["target"])

best = results.max["params"]
best_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": random_state,
    "boost_from_average": True,
    "learning_rate": best["learning_rate"],
    "depth": int(round(best["depth"])),
    "l2_leaf_reg": best["l2_leaf_reg"],
    "random_strength": best["random_strength"],
    "min_data_in_leaf": int(round(best["min_data_in_leaf"])),
}

cv_rmse, best_iterations = cat_cv(best_params, 10)
print("10-Fold CV RMSE:", cv_rmse)
print("Best boosting round from CV:", best_iterations)

best_params["iterations"] = best_iterations

# a8_stacking and s4_prediction rebuild CatBoost from scratch for their fold and
# refit models, so they read these params back rather than hardcoding defaults.
with open("models/catboost_best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
print("CatBoost params saved to models/catboost_best_params.json")


############################# Final CatBoost Model #############################
final_model_cat_basic = cb.CatBoostRegressor(**best_params, train_dir="models/catboost_basic")
final_model_cat_basic.fit(train_pool, verbose=200)

final_model_cat_basic.save_model("models/final_model_catboost_basic.cbm", format="cbm")
print("CatBoost model saved to models/final_model_catboost_basic.cbm")

evaluate_model(final_model_cat_basic, val_pool, y_val, "CatBoost Regressor")

feature_importance = pd.Series(
    final_model_cat_basic.get_feature_importance(type="PredictionValuesChange"),
    index=X_train_cat.columns
).sort_values(ascending=False)

with open("models/catboost_feature_importance.txt", "w") as f:
    f.write(feature_importance.to_string())

conn.close()
