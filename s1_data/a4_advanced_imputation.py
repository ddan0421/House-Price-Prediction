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
Step 5: Impute missing continuous numerical data in the training set using IterativeImputer with BayesianRidge estimator
Step 6: Impute missing continuous numerical data in the validation set using the trained imputer
Step 7: Impute missing continuous numerical data in the test set using the trained imputer


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

train = conn.execute("""select * from train_contextual_imputed""").fetch_df()
test = conn.execute("""select * from test_contextual_imputed""").fetch_df()


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
    conn = duckdb.connect(database=":memory:")
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
               "Heating_HeatingQC", "Street", "Alley", "Utilities", "MasVnrType", "Foundation", 
               "Functional", "GarageType", "PavedDrive", 
               "Fence", "MiscFeature", "SaleType", "SaleCondition", "Season_Sold"]


# One-hot encode nominal categorical variables
X_train_encoded = pd.get_dummies(X_train_engineered, columns=nominal_cat, drop_first=True)
X_val_encoded = pd.get_dummies(X_val_engineered, columns=nominal_cat, drop_first=True)
test_encoded = pd.get_dummies(test_engineered, columns=nominal_cat, drop_first=True)

X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
test_encoded = test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)


############################### Train Data LotFrontage Missing Data Imputation ########################################
# Step 5: Impute missing continuous numerical data in the training set using IterativeImputer with BayesianRidge estimator
# Columns to be used as predictors for imputing "LotFrontage"
columns_for_imputation = [
    "LotArea", "1stFlrSF", "Street_Pave",
    "LotShape_LandContour_IR1_HLS", "LotShape_LandContour_IR1_Low", "LotShape_LandContour_IR1_Lvl",
    "LotShape_LandContour_IR2_Bnk", "LotShape_LandContour_IR2_HLS", "LotShape_LandContour_IR2_Low",
    "LotShape_LandContour_IR2_Lvl", "LotShape_LandContour_IR3_Bnk", "LotShape_LandContour_IR3_HLS",
    "LotShape_LandContour_IR3_Low", "LotShape_LandContour_IR3_Lvl", "LotShape_LandContour_Reg_Bnk",
    "LotShape_LandContour_Reg_HLS", "LotShape_LandContour_Reg_Low", "LotShape_LandContour_Reg_Lvl",
    
    "Neighborhood_Condition_Blueste_Norm", "Neighborhood_Condition_BrDale_Norm", "Neighborhood_Condition_BrkSide_Artery",
    "Neighborhood_Condition_BrkSide_Feedr_Norm", "Neighborhood_Condition_BrkSide_Norm", "Neighborhood_Condition_BrkSide_PosN_Norm",
    "Neighborhood_Condition_BrkSide_RRAn_Feedr", "Neighborhood_Condition_BrkSide_RRAn_Norm", "Neighborhood_Condition_BrkSide_RRNn_Feedr",
    "Neighborhood_Condition_ClearCr_Feedr_Norm", "Neighborhood_Condition_ClearCr_Norm", "Neighborhood_Condition_CollgCr_Norm",
    
    "BldgType_HouseStyle_1Fam_1.5Unf", "BldgType_HouseStyle_1Fam_1Story", "BldgType_HouseStyle_1Fam_2.5Fin",
    "BldgType_HouseStyle_1Fam_2.5Unf", "BldgType_HouseStyle_1Fam_2Story", "BldgType_HouseStyle_1Fam_SFoyer",
    "BldgType_HouseStyle_1Fam_SLvl", "BldgType_HouseStyle_2fmCon_1.5Fin", "BldgType_HouseStyle_2fmCon_1.5Unf",
    "BldgType_HouseStyle_2fmCon_1Story", "BldgType_HouseStyle_2fmCon_2.5Fin", "BldgType_HouseStyle_2fmCon_2.5Unf",
    "BldgType_HouseStyle_2fmCon_2Story", "BldgType_HouseStyle_2fmCon_SLvl", "BldgType_HouseStyle_Duplex_1.5Fin",
    
    "BldgType_HouseStyle_Duplex_1Story", "BldgType_HouseStyle_Duplex_2Story", "BldgType_HouseStyle_Duplex_SFoyer",
    "BldgType_HouseStyle_Duplex_SLvl", "BldgType_HouseStyle_TwnhsE_1Story", "BldgType_HouseStyle_TwnhsE_2Story",
    "BldgType_HouseStyle_TwnhsE_SFoyer", "BldgType_HouseStyle_TwnhsE_SLvl", "BldgType_HouseStyle_Twnhs_1Story",
    "BldgType_HouseStyle_Twnhs_2Story", "BldgType_HouseStyle_Twnhs_SFoyer", "BldgType_HouseStyle_Twnhs_SLvl"
]

iterative_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)

X_train_imputed = X_train_encoded.copy()  
X_train_imputed["LotFrontage"] = iterative_imputer.fit_transform(X_train_encoded[columns_for_imputation + ["LotFrontage"]])[ :, -1]
X_train["LotFrontage"] = X_train_imputed["LotFrontage"].values

# Step 6: Impute missing continuous numerical data in the validation set using the trained imputer
X_val_imputed = X_val_encoded.copy()  
X_val_imputed["LotFrontage"] = iterative_imputer.transform(X_val_encoded[columns_for_imputation + ["LotFrontage"]])[:, -1]
X_val["LotFrontage"] = X_val_imputed["LotFrontage"].values

# Step 7: Impute missing continuous numerical data in the test set using the trained imputer
test_imputed = test_encoded.copy()  
test_imputed["LotFrontage"] = iterative_imputer.transform(test_encoded[columns_for_imputation + ["LotFrontage"]])[:, -1]
test["LotFrontage"] = test_imputed["LotFrontage"].values

# Save imputed data into the DuckDB database
y_train = y_train.to_frame(name="SalePrice")
y_val = y_val.to_frame(name="SalePrice")

for source in ["X_train", "X_val", "y_train", "y_val", "test"]:
    query = f"""
        create or replace table {source} as
            select * from {source};
    """
    conn.execute(query)
print(conn.execute("SHOW TABLES").fetchall())
conn.close()

# Export for EDA
X_combined = pd.concat([X_train_imputed.sort_values(by="Id", ascending=True), X_val_imputed.sort_values(by="Id", ascending=True)], axis=0, ignore_index=True)
train["LotFrontage"] = X_combined["LotFrontage"]
train["Age_House"] = X_combined["Age_House"]
train["Yrs_Since_Remodel"] = X_combined["Yrs_Since_Remodel"]
train["Age_Garage"] = X_combined["Age_Garage"]
train.to_csv(os.path.join(base_folder, "train_after_imputation_EDA.csv"), index=False)

