import pandas as pd
import duckdb
import numpy as np
# Importing IterativeImputer with the experimental module
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


"""
Advanced Imputation Workflow:
Step 1: Split the train dataset into train and validation set
Step 2: Impute categorical missing data in train, validation, and test sets with the mode from training set to prevent data leakage
Step 3: Creating categorical interaction terms and create time variables
Step 4: Encode nominal categorical variables separately for train, validation, and test sets to prevent data leakage, ensuring consistent feature alignment using one-hot encoding
Step 5: Encode ordinal categorical variables and binary nominal categorical variable using label encoding
Step 6: Impute missing continuous numerical data in the training set using IterativeImputer with BayesianRidge estimator
Step 7: Impute missing continuous numerical data in the validation set using the trained imputer
Step 8: Impute missing continuous numerical data in the test set using the trained imputer


Impute Categorical Data (train, validation, test):
- Imputing missing values in the train, validation, and test set using the mode calculated exclusively from the original train set ensures that no information from the test or validation set influences the training process.
- The validation set remains independent of the training process and can be used for model selection.

Impute Missing Continuous Numerical Data (LotFrontage) Separately (train, validation, test):
- Train the regression imputation model using only the training subset.
- Use this trained imputation model to impute missing values in the train, validation, and test subsets.
- By training the imputer only on the training data, I ensure that the imputation process does not allow any information from the validation or test sets to influence the training process.
"""

base_folder = "data"
database = "AmesHousePrice.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=False)

train = conn.execute("""select * from train_contextual_imputed order by Id""").fetch_df()
test = conn.execute("""select * from test_contextual_imputed order by Id""").fetch_df()


# Step 1: Split the train dataset into train and validation set
X = train.drop(columns=["SalePrice"], axis=1)
y = train["SalePrice"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 2: Impute categorical missing data in train, validation, and test sets with the mode from training set to prevent data leakage
X_train["Electrical"].fillna(X_train["Electrical"].mode()[0], inplace=True)
# validation set has no missing values except LotFrontage
test["MSZoning"].fillna(X_train["MSZoning"].mode()[0], inplace=True)
test["Utilities"].fillna(X_train["Utilities"].mode()[0], inplace=True)
test["KitchenQual"].fillna(X_train["KitchenQual"].mode()[0], inplace=True)
test["Functional"].fillna(X_train["Functional"].mode()[0], inplace=True)


# Step 3: Creating categorical interaction terms and create time variables
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
- YearBuilt
- YearRemodAdd
- GarageYrBlt
- YrSold

"""


def feature_engineering(df):
    conn = duckdb.connect()
    conn.register("original_df", df)
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
    FROM original_df)
    
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
    result = conn.query(query).fetchdf()
    conn.close()
    return result

X_train_engineered = feature_engineering(X_train)
X_val_engineered = feature_engineering(X_val)
test_engineered = feature_engineering(test)

# Step 4: Encode nominal categorical variables separately for train, validation, and test sets to prevent data leakage, ensuring consistent feature alignment using one-hot encoding
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
X_train_encoded = pd.get_dummies(X_train_engineered, columns=nominal_cat, drop_first=True)
X_val_encoded = pd.get_dummies(X_val_engineered, columns=nominal_cat, drop_first=True)
test_encoded = pd.get_dummies(test_engineered, columns=nominal_cat, drop_first=True)

X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
test_encoded = test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Step 5: Encode ordinal categorical variables and binary nominal categorical variable using label encoding
def ordinal_encoding(df):
    conn = duckdb.connect()
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
    result = conn.query(query).fetchdf()
    conn.close()
    return result

X_train_encoded = ordinal_encoding(X_train_encoded)
X_val_encoded = ordinal_encoding(X_val_encoded)
test_encoded = ordinal_encoding(test_encoded)


bool_columns_train = X_train_encoded.select_dtypes(include="bool").columns
bool_columns_val = X_val_encoded.select_dtypes(include="bool").columns
bool_columns_test = test_encoded.select_dtypes(include="bool").columns

X_train_encoded[bool_columns_train] = X_train_encoded[bool_columns_train].astype("int8")
X_val_encoded[bool_columns_val] = X_val_encoded[bool_columns_val].astype("int8")
test_encoded[bool_columns_test] = test_encoded[bool_columns_test].astype("int8")

############################### Train Data LotFrontage Missing Data Imputation ########################################
# Step 6: Impute missing continuous numerical data in the training set using IterativeImputer with BayesianRidge estimator
# Columns to be used as predictors for imputing "LotFrontage"
columns_for_imputation = ["LotArea", "1stFlrSF", "Street_encoded"] \
  + [col for col in X_train_encoded.columns if col.startswith("Neighborhood_Condition_")] \
  + [col for col in X_train_encoded.columns if col.startswith("LotShape_LandContour_")] \
  + [col for col in X_train_encoded.columns if col.startswith("BldgType_HouseStyle_")]

iterative_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)

X_train_imputed = X_train_encoded.copy()  
X_train_imputed["LotFrontage"] = iterative_imputer.fit_transform(X_train_encoded[columns_for_imputation + ["LotFrontage"]])[ :, -1]

# Step 7: Impute missing continuous numerical data in the validation set using the trained imputer
X_val_imputed = X_val_encoded.copy()  
X_val_imputed["LotFrontage"] = iterative_imputer.transform(X_val_encoded[columns_for_imputation + ["LotFrontage"]])[:, -1]

# Step 8: Impute missing continuous numerical data in the test set using the trained imputer
test_imputed = test_encoded.copy()  
test_imputed["LotFrontage"] = iterative_imputer.transform(test_encoded[columns_for_imputation + ["LotFrontage"]])[:, -1]
test["LotFrontage"] = test_imputed["LotFrontage"].values

# Combine imputed LotFrontage with the original train and validation sets and export train and test to duckdb
X_combined = pd.concat([X_train_imputed, X_val_imputed], axis=0)

X_combined = (
    X_combined
    .sort_values(by="Id") 
    .reset_index(drop=True)
)

train["LotFrontage"] = X_combined["LotFrontage"]

for source in ["train", "test"]:
    df = globals()[source]
    conn.execute(f"drop table if exists {source}")
    query = f"""
        create or replace table {source} as
            select * from df
            order by Id;
    """
    conn.execute(query)
print(conn.execute("SHOW TABLES").fetchall())
conn.close()

# Export for EDA
train.to_csv(os.path.join(base_folder, "train_after_imputation_EDA.csv"), index=False)

