import pandas as pd
import duckdb
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from s1_data.db_utils import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


"""
KNN Model Data Preparation Workflow:
Step 1: Split the train dataset into train and validation set
Step 2: Create categorical interaction terms and time variables
Step 3: Encode nominal categorical variables separately for train, validation, and test sets to prevent data leakage, ensuring consistent feature alignment using one-hot encoding 
Step 4: Encode ordinal categorical variables and binary nominal categorical variable using label encoding
Step 5: Transform numerical terms and create interaction terms for numerical variables
Step 6: Standardization


"""

base_folder = "data"
database = "AmesHousePrice.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=False)

train = load_df(conn, "train")
test = load_df(conn, "test")

# Step 1: Split the train dataset into train and validation set
X = train.drop(columns=["SalePrice"], axis=1)
y = train["SalePrice"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




# Step 2: Creating categorical interaction terms and create time variables
"""
categorical:
- combine MSSubClass and MSZoning
- combine LotConfig and LandSlope
- combine Neighborhood and Condition1 and Condition2
- combine BldgType and HouseStyle
- combine Exterior1st and Exterior2nd
- combine CentralAir and Electrical
- combine LotShape and LandContour
- combine RoofStyle and RoofMatl
- combine Heating and HeatingQC  
- capture the seasonality of the sale based on MoSold

Feature engineering with year and month variables
- Age_House
- Yrs_Since_Remodel
- Age_Garage

"""


def feature_engineering(conn, df):
    conn.register("input_df", df)
    query = """
    WITH cte AS (
    SELECT 
        *,
        CAST(MSSubClass AS TEXT) || '_' || MSZoning AS MSSubClass_MSZoning,
        LotConfig || '_' || LandSlope AS LotConfig_LandSlope,
        Neighborhood || '_' || 
            CASE 
                WHEN Condition1 = Condition2 THEN Condition1 
                ELSE Condition1 || '_' || Condition2 
            END AS Neighborhood_Condition,
        BldgType || '_' || HouseStyle AS BldgType_HouseStyle,
        CASE 
            WHEN Exterior1st = Exterior2nd THEN Exterior1st 
            ELSE Exterior1st || '_' || Exterior2nd 
        END AS Exterior1st_Exterior2nd,
        CentralAir || '_' || Electrical AS CentralAir_Electrical,
        LotShape || '_' || LandContour AS LotShape_LandContour,
        RoofStyle || '_' || RoofMatl AS RoofStyle_RoofMatl,
        Heating || '_' || HeatingQC AS Heating_HeatingQC,
        CASE
           WHEN MoSold IN (12, 1, 2) THEN 'Winter'
           WHEN MoSold IN (3, 4, 5) THEN 'Spring'
           WHEN MoSold IN (6, 7, 8) THEN 'Summer'
           ELSE 'Fall'
        END AS Season_Sold,
        IF((YrSold - YearBuilt) < 0 OR YrSold IS NULL OR YearBuilt IS NULL, 0, (YrSold - YearBuilt)) AS Age_House,
        IF((YrSold - YearRemodAdd) < 0 OR YrSold IS NULL OR YearRemodAdd IS NULL, 0, (YrSold - YearRemodAdd)) AS Yrs_Since_Remodel,
        IF((YrSold - GarageYrBlt) < 0 OR GarageYrBlt IS NULL OR GarageType = 'no_garage', 0, (YrSold - GarageYrBlt)) AS Age_Garage
    FROM input_df)
    
    SELECT * EXCLUDE ("MSSubClass", "MSZoning", "LotConfig", "LandSlope", 
        "Condition1", "Condition2", "Neighborhood", 
        "BldgType", "HouseStyle", 
        "Exterior1st", "Exterior2nd", 
        "CentralAir", "Electrical", 
        "LotShape", "LandContour", 
        "RoofStyle", "RoofMatl", 
        "Heating", "HeatingQC", "MoSold", "YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold")
    FROM cte;
    """
    result = conn.execute(query).fetchdf()
    conn.unregister("input_df")
    return result

X_train_engineered = feature_engineering(conn, X_train)
X_val_engineered = feature_engineering(conn, X_val)
test_engineered = feature_engineering(conn, test)

# Step 3: Encode nominal categorical variables separately for train, validation, and test sets to prevent data leakage, ensuring consistent feature alignment using one-hot encoding
nominal_cat = ["MSSubClass_MSZoning", "LotConfig_LandSlope", "Neighborhood_Condition", "BldgType_HouseStyle",
               "Exterior1st_Exterior2nd", "CentralAir_Electrical", "LotShape_LandContour", "RoofStyle_RoofMatl",
               "Heating_HeatingQC", "Alley", "MasVnrType", "Foundation", 
               "GarageType", "PavedDrive", 
               "Fence", "MiscFeature", "SaleType", "SaleCondition", "Season_Sold"]

ordinal_cat = ["Utilities", "Functional", "OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", 
               "BsmtFinType1", "BsmtFinType2", "KitchenQual", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", 
               "PoolQC"]

binary_nominal = ["Street"]

# One-hot encode nominal categorical variables
X_train_encoded = pd.get_dummies(X_train_engineered, columns=nominal_cat, drop_first=False)
X_val_encoded = pd.get_dummies(X_val_engineered, columns=nominal_cat, drop_first=False)
test_encoded = pd.get_dummies(test_engineered, columns=nominal_cat, drop_first=False)

X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
test_encoded = test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Step 4: Encode ordinal categorical variables and binary nominal categorical variable using label encoding
def ordinal_encoding(conn, df):
    conn.register("input_df", df)
    query = """
    WITH cte AS (
    SELECT 
        *,
        CASE
            WHEN Utilities = 'AllPub' THEN 4
            WHEN Utilities = 'NoSewr' THEN 3
            WHEN Utilities = 'NoSeWa' THEN 2
            WHEN Utilities = 'ELO' THEN 1
            ELSE 0
        END AS Utilities_encoded,
        CASE
            WHEN Functional = 'Typ' THEN 8
            WHEN Functional = 'Min1' THEN 7
            WHEN Functional = 'Min2' THEN 6
            WHEN Functional = 'Mod' THEN 5
            WHEN Functional = 'Maj1' THEN 4
            WHEN Functional = 'Maj2' THEN 3
            WHEN Functional = 'Sev' THEN 2
            WHEN Functional = 'Sal' THEN 1
            ELSE 0
        END AS Functional_encoded,
        -- OverallQual is already in numeric format, so no need to encode it
        -- OverallCond is already in numeric format, so no need to encode it
        CASE 
            WHEN ExterQual = 'Ex' THEN 5
            WHEN ExterQual = 'Gd' THEN 4
            WHEN ExterQual = 'TA' THEN 3
            WHEN ExterQual = 'Fa' THEN 2
            WHEN ExterQual = 'Po' THEN 1
            ELSE 0
        END AS ExterQual_encoded,
        CASE
            WHEN ExterCond = 'Ex' THEN 5
            WHEN ExterCond = 'Gd' THEN 4
            WHEN ExterCond = 'TA' THEN 3
            WHEN ExterCond = 'Fa' THEN 2
            WHEN ExterCond = 'Po' THEN 1
            ELSE 0
        END AS ExterCond_encoded,
        CASE
            WHEN BsmtQual = 'Ex' THEN 5
            WHEN BsmtQual = 'Gd' THEN 4
            WHEN BsmtQual = 'TA' THEN 3
            WHEN BsmtQual = 'Fa' THEN 2
            WHEN BsmtQual = 'Po' THEN 1
            ELSE 0
        END AS BsmtQual_encoded,
        CASE
            WHEN BsmtCond = 'Ex' THEN 5
            WHEN BsmtCond = 'Gd' THEN 4
            WHEN BsmtCond = 'TA' THEN 3
            WHEN BsmtCond = 'Fa' THEN 2
            WHEN BsmtCond = 'Po' THEN 1
            ELSE 0
        END AS BsmtCond_encoded,
        CASE
            WHEN BsmtExposure = 'Gd' THEN 4
            WHEN BsmtExposure = 'Av' THEN 3
            WHEN BsmtExposure = 'Mn' THEN 2
            WHEN BsmtExposure = 'No' THEN 1
            ELSE 0
        END AS BsmtExposure_encoded,
        CASE
            WHEN BsmtFinType1 = 'GLQ' THEN 6
            WHEN BsmtFinType1 = 'ALQ' THEN 5
            WHEN BsmtFinType1 = 'BLQ' THEN 4
            WHEN BsmtFinType1 = 'Rec' THEN 3
            WHEN BsmtFinType1 = 'LwQ' THEN 2
            WHEN BsmtFinType1 = 'Unf' THEN 1
            ELSE 0
        END AS BsmtFinType1_encoded,
        CASE
            WHEN BsmtFinType2 = 'GLQ' THEN 6
            WHEN BsmtFinType2 = 'ALQ' THEN 5
            WHEN BsmtFinType2 = 'BLQ' THEN 4
            WHEN BsmtFinType2 = 'Rec' THEN 3
            WHEN BsmtFinType2 = 'LwQ' THEN 2
            WHEN BsmtFinType2 = 'Unf' THEN 1
            ELSE 0
        END AS BsmtFinType2_encoded,
        CASE
            WHEN KitchenQual = 'Ex' THEN 5
            WHEN KitchenQual = 'Gd' THEN 4
            WHEN KitchenQual = 'TA' THEN 3
            WHEN KitchenQual = 'Fa' THEN 2
            WHEN KitchenQual = 'Po' THEN 1
            ELSE 0
        END AS KitchenQual_encoded,
        CASE
            WHEN FireplaceQu = 'Ex' THEN 5
            WHEN FireplaceQu = 'Gd' THEN 4
            WHEN FireplaceQu = 'TA' THEN 3
            WHEN FireplaceQu = 'Fa' THEN 2
            WHEN FireplaceQu = 'Po' THEN 1
            ELSE 0
        END AS FireplaceQu_encoded,
        CASE
            WHEN GarageFinish = 'Fin' THEN 3
            WHEN GarageFinish = 'RFn' THEN 2
            WHEN GarageFinish = 'Unf' THEN 1
            ELSE 0
        END AS GarageFinish_encoded,
        CASE
            WHEN GarageQual = 'Ex' THEN 5
            WHEN GarageQual = 'Gd' THEN 4
            WHEN GarageQual = 'TA' THEN 3
            WHEN GarageQual = 'Fa' THEN 2
            WHEN GarageQual = 'Po' THEN 1
            ELSE 0
        END AS GarageQual_encoded,
        CASE
            WHEN GarageCond = 'Ex' THEN 5
            WHEN GarageCond = 'Gd' THEN 4
            WHEN GarageCond = 'TA' THEN 3
            WHEN GarageCond = 'Fa' THEN 2
            WHEN GarageCond = 'Po' THEN 1
            ELSE 0
        END AS GarageCond_encoded,
        CASE
            WHEN PoolQC = 'Ex' THEN 4
            WHEN PoolQC = 'Gd' THEN 3
            WHEN PoolQC = 'TA' THEN 2
            WHEN PoolQC = 'Fa' THEN 1
            ELSE 0
        END AS PoolQC_encoded,
        CASE
            WHEN Street = 'Grvl' THEN 0
            WHEN Street = 'Pave' THEN 1
            ELSE 0
        END AS Street_encoded
    FROM input_df)
    
    SELECT * EXCLUDE (
        "Utilities", "Functional", "Street",
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", 
        "BsmtFinType1", "BsmtFinType2", "KitchenQual", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", 
        "PoolQC")
    FROM cte;
    """
    result = conn.execute(query).fetchdf()
    conn.unregister("input_df")
    return result

X_train_encoded = ordinal_encoding(conn, X_train_encoded)
X_val_encoded = ordinal_encoding(conn, X_val_encoded)
test_encoded = ordinal_encoding(conn, test_encoded)


bool_columns_train = X_train_encoded.select_dtypes(include="bool").columns
bool_columns_val = X_val_encoded.select_dtypes(include="bool").columns
bool_columns_test = test_encoded.select_dtypes(include="bool").columns

X_train_encoded[bool_columns_train] = X_train_encoded[bool_columns_train].astype("int8")
X_val_encoded[bool_columns_val] = X_val_encoded[bool_columns_val].astype("int8")
test_encoded[bool_columns_test] = test_encoded[bool_columns_test].astype("int8")






# Step 5: Transform numerical terms and create interaction terms for numerical variables
def log_transform(conn, data):
    conn.register("input_df", data)
    query = """
    WITH cte AS (
        SELECT
            *,
            -- Log transformations
            LOG(1 + "LotFrontage") AS log_LotFrontage,
            LOG(1 + "LotArea") AS log_LotArea,
            LOG(1 + "1stFlrSF") AS log_1stFlrSF,
            LOG(1 + "2ndFlrSF") AS log_2ndFlrSF,
            LOG(1 + "LowQualFinSF") AS log_LowQualFinSF,
            LOG(1 + "GrLivArea") AS log_GrLivArea,
            LOG(1 + "Yrs_Since_Remodel") AS log_Yrs_Since_Remodel,
            LOG(1 + "Age_Garage") AS log_Age_Garage,

            -- Square root transformations
            SQRT("TotalBsmtSF") AS sqrt_TotalBsmtSF,
            SQRT("WoodDeckSF") AS sqrt_WoodDeckSF,
            SQRT("BsmtUnfSF") AS sqrt_BsmtUnfSF,
            SQRT("BsmtFinSF1") AS sqrt_BsmtFinSF1,

            -- Cube root transformations
            CBRT("MasVnrArea") AS cbrt_MasVnrArea,
            CBRT("OpenPorchSF") AS cbrt_OpenPorchSF,

            -- Interaction terms
            "GrLivArea" / ("TotalBsmtSF" + "1stFlrSF" + "2ndFlrSF") AS FinishedAreaPct,
            LOG(1+ "GrLivArea" * "TotRmsAbvGrd") AS Living_Rooms,
            LOG(1+ "GarageArea" * "GarageCars") AS Garage_Space,
            LOG(1+ "Age_Garage" * "GarageCars") AS Garage_AgeCars,
            LOG(1 + CBRT("EnclosedPorch") * "Age_House") AS Porch_Age,
            "BedroomAbvGr" / "TotRmsAbvGrd" AS Ratio_Bedroom_Rooms,
            "2ndFlrSF" / "GrLivArea" AS Ratio_2ndFlr_Living

        
        FROM input_df
    )
    SELECT * EXCLUDE (
        "LotFrontage", "LotArea", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea",
        "Yrs_Since_Remodel", "Age_Garage",
        "TotalBsmtSF", "WoodDeckSF", "BsmtUnfSF", "BsmtFinSF1",
        "MasVnrArea", "OpenPorchSF",
        "HPI", "HPA", "pmms", "pmms_chg", "ue", "ue_chg", "nonfarm", "nonfarm_yoy"
    )
    FROM cte;
    """
    result = conn.query(query).fetchdf()
    conn.unregister("input_df")
    return result

X_train_transformed = log_transform(conn, X_train_encoded)
X_val_transformed = log_transform(conn, X_val_encoded)
test_transformed = log_transform(conn, test_encoded)


# Step 6: Standardization
numerical_variables = [
    "log_LotFrontage", "log_LotArea", "log_1stFlrSF", "log_2ndFlrSF", "log_LowQualFinSF",
    "log_GrLivArea", "log_Yrs_Since_Remodel", "log_Age_Garage",
    "sqrt_TotalBsmtSF", "sqrt_WoodDeckSF",
    "cbrt_MasVnrArea", "cbrt_OpenPorchSF",
    "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
    "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "EnclosedPorch", "3SsnPorch",
    "ScreenPorch", "PoolArea", "MiscVal", "Age_House",
    "FinishedAreaPct", "Living_Rooms", "Garage_Space", "Garage_AgeCars", "Porch_Age", "Ratio_Bedroom_Rooms", "Ratio_2ndFlr_Living",
    "sqrt_BsmtUnfSF", "sqrt_BsmtFinSF1", "BsmtFinSF2",
]


scaler = StandardScaler()
X_train_transformed[numerical_variables] = scaler.fit_transform(X_train_transformed[numerical_variables])
X_val_transformed[numerical_variables] = scaler.transform(X_val_transformed[numerical_variables])
test_transformed[numerical_variables] = scaler.transform(test_transformed[numerical_variables])


# Register pandas DataFrames as DuckDB tables
tables = {
    "X_train_knn": X_train_transformed,
    "X_val_knn": X_val_transformed,
    "test_knn": test_transformed,
}

for table_name, df in tables.items():
    save_df(conn, df, table_name)

print(conn.execute("SHOW TABLES").fetchall())
conn.close()
