import duckdb
from sklearn.model_selection import train_test_split
import os
from s1_data.db_utils import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


"""
Catboost Model Data Preparation Workflow:
Step 1: Split the train dataset into train and validation set
Step 2: Create categorical interaction terms and time variables
Step 3: Transform numerical terms and create interaction terms for numerical variables

*Note: Catboost internally handles categorical encoding using target statistics/ordered boosting techniques

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



# Step 3: Transform numerical terms and create interaction terms for numerical variables
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

X_train_transformed = log_transform(conn, X_train_engineered)
X_val_transformed = log_transform(conn, X_val_engineered)
test_transformed = log_transform(conn, test_engineered)


# Register pandas DataFrames as DuckDB tables
tables = {
    "X_train_cat": X_train_transformed,
    "X_val_cat": X_val_transformed,
    "test_cat": test_transformed,
}

for table_name, df in tables.items():
    save_df(conn, df, table_name)

print(conn.execute("SHOW TABLES").fetchall())
conn.close()



