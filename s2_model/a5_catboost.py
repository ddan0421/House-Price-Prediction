import os
import catboost as cb
import duckdb

from s1_data.db_utils import load_df
from s3_validation.model_evaluation import evaluate_model

base_folder = "data"
database = "AmesHousePrice.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=False)

X_train_cat = load_df(conn, "X_train_cat")
X_val_cat = load_df(conn, "X_val_cat")
test_cat = load_df(conn, "test_cat")
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

############################# Basic CatBoost Model #############################
final_model_cat_basic = cb.CatBoostRegressor(loss_function="RMSE", random_seed=42, train_dir="models/catboost_basic")
final_model_cat_basic.fit(train_pool, eval_set=val_pool, verbose=True)

final_model_cat_basic.save_model("final_model_catboost_basic.cbm", format="cbm")
print("CatBoost model saved to final_model_catboost_basic.cbm")

evaluate_model(final_model_cat_basic, val_pool, y_val, "CatBoost Regressor")