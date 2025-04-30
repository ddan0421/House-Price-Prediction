import pandas as pd
import duckdb
import numpy as np
# Importing IterativeImputer with the experimental module
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

train = pd.read_csv("data/train_clean_01.csv")
test = pd.read_csv("data/test_clean_01.csv")


"""
Workflow:
Step 1: Impute categorical missing data in train set and test set with the mode from training set
Step 2: Creating categorical interaction terms and create time variables
Step 3: Encode categorical variables separately for train and test sets to prevent data leakage, ensuring consistent feature alignment using one-hot encoding and applying label encoding mappings from the train set to the test set.
Step 4: Split the train dataset into train and validation set
Step 5: Impute missing continuous numerical data in the training set using IterativeImputer with BayesianRidge estimator
Step 6: Impute missing continuous numerical data in the validation set using the trained imputer
Step 7: Impute missing continuous numerical data in the test set using the trained imputer
Step 8: Correlation analysis and transform numerical terms
Step 9: Creating interaction terms for numerical variables

Impute Missing Values Separately:
- Train the regression imputation model using only the training subset.
- Use this trained imputation model to impute missing values in the train, validation, and test subsets.
- By training the imputer only on the training data, I ensure that the imputation process does not allow any information from the validation or test sets to influence the training process.

Encode Categorical Data Separetely:
- Train and Validation Split: When I encode the train and validation sets together, I'm still within the same training data, meaning I'm not introducing new or unseen data. 
  The validation set is just a portion of the training data, so encoding both together ensures consistency in feature mapping without causing leakage.
- Train and Test Split: The test set is entirely separate from the training process and should not influence the encoding of the training data in any way. 
  If I encode the train and test sets together, I'm potentially introducing data leakage because the encoding process might learn patterns from the test data that shouldn't be available during model training.

Final Summary:
- Encoding is about creating consistent feature mappings and does not inherently involve predicting or learning from missing data or future values, which is why encoding can be safely done on both the train and validation sets together.
- Imputation, however, relies on training a model to estimate missing values, which should only be done on the training set to avoid using any information from the test/validation sets during training. This is why imputation must be done separately.
"""

# Step 1: Impute categorical missing data in train set and test set with the mode from training set
train["Electrical"].fillna(train["Electrical"].mode()[0], inplace=True)
test_imputed = test.copy()  # Make a copy of the test set
test_imputed["MSZoning"].fillna(train["MSZoning"].mode()[0], inplace=True)
test_imputed["Utilities"].fillna(train["Utilities"].mode()[0], inplace=True)
test_imputed["KitchenQual"].fillna(train["KitchenQual"].mode()[0], inplace=True)
test_imputed["Functional"].fillna(train["Functional"].mode()[0], inplace=True)

train.to_csv("data/train_before_imputation_EDA.csv", index=False)

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
- YearBuilt
- YearRemodAdd
- GarageYrBlt
- YrSold

"""


def feature_engineering(df):
    conn = duckdb.connect()
    conn.register("original_df", df)
    query = """
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
        IF((YrSold - YearBuilt) < 0, 0, (YrSold - YearBuilt)) AS Age_House,
        IF((YrSold - YearRemodAdd) < 0, 0, (YrSold - YearRemodAdd)) AS Yrs_Since_Remodel,
        IF((YrSold - GarageYrBlt) < 0, 0, (YrSold - GarageYrBlt)) AS Age_Garage
    FROM original_df;
    """
    result = conn.execute(query).fetch_df()
    columns_to_drop = [
        "MSSubClass", "MSZoning", "LotConfig", "LandSlope", 
        "Condition1", "Condition2", "Neighborhood", 
        "BldgType", "HouseStyle", 
        "Exterior1st", "Exterior2nd", 
        "CentralAir", "Electrical", 
        "LotShape", "LandContour", 
        "RoofStyle", "RoofMatl", 
        "Heating", "HeatingQC", "MoSold", "YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold"
    ]
    result = result.drop(columns=columns_to_drop)
    conn.close()
    return result


train_new = feature_engineering(train)
test_new = feature_engineering(test_imputed)

X = train_new.drop(columns=["SalePrice"], axis=1) 
y = train_new["SalePrice"]

############################### Encode train and test data ########################################
# Step 3: Encode categorical variables separately for train and test sets
nominal_cat = ["MSSubClass_MSZoning", "LotConfig_LandSlope", "Neighborhood_Condition", "BldgType_HouseStyle",
               "Exterior1st_Exterior2nd", "CentralAir_Electrical", "LotShape_LandContour", "RoofStyle_RoofMatl",
               "Heating_HeatingQC", "Street", "Alley", "Utilities", "MasVnrType", "Foundation", 
               "Functional", "GarageType", "PavedDrive", 
               "Fence", "MiscFeature", "SaleType", "SaleCondition", "Season_Sold"]


ordinal_cat = ["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", 
               "BsmtFinType1", "BsmtFinType2", "KitchenQual", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", 
               "PoolQC"]

# One-hot encode nominal categorical variables
train_encoded = pd.get_dummies(X, columns=nominal_cat, drop_first=True)
test_encoded = pd.get_dummies(test_new, columns=nominal_cat, drop_first=True)

test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

# Label encode ordinal categorical variables

def ordinal_encoding(df):
    conn = duckdb.connect()
    conn.register("input_df", df)
    query = """
    SELECT 
        *,
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
    FROM input_df;
    """
    result = conn.execute(query).fetch_df()
    columns_to_drop = [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", 
        "BsmtFinType1", "BsmtFinType2", "KitchenQual", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", 
        "PoolQC"]
    result = result.drop(columns=columns_to_drop)
    conn.close()
    return result

train_encoded = ordinal_encoding(train_encoded)
test_encoded = ordinal_encoding(test_encoded)


bool_columns_train = train_encoded.select_dtypes(include="bool").columns
bool_columns_test = test_encoded.select_dtypes(include="bool").columns

train_encoded[bool_columns_train] = train_encoded[bool_columns_train].astype("int8")
test_encoded[bool_columns_test] = test_encoded[bool_columns_test].astype("int8")


# Step 4: Split the train dataset into train and validation set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_encoded, y, test_size=0.2, random_state=42)



############################### Train Data LotFrontage Missing Data Imputation ########################################
# Step 5: Impute missing continuous numerical data in the training set using IterativeImputer with BayesianRidge estimator
# Columns to be used as predictors for imputing "LotFrontage"
columns_for_imputation = [
    "LotArea", "1stFlrSF","Street_Pave",
    "LotShape_LandContour_IR1_HLS", "LotShape_LandContour_IR1_Low", 
    "LotShape_LandContour_IR1_Lvl", "LotShape_LandContour_IR2_Bnk", 
    "LotShape_LandContour_IR2_HLS", "LotShape_LandContour_IR2_Low", 
    "LotShape_LandContour_IR2_Lvl", "LotShape_LandContour_IR3_Bnk", 
    "LotShape_LandContour_IR3_HLS", "LotShape_LandContour_IR3_Low", 
    "LotShape_LandContour_IR3_Lvl", "LotShape_LandContour_Reg_Bnk", 
    "LotShape_LandContour_Reg_HLS", "LotShape_LandContour_Reg_Low", 
    "LotShape_LandContour_Reg_Lvl", "Neighborhood_Condition_Blueste_Norm", 
    "Neighborhood_Condition_BrDale_Norm", "Neighborhood_Condition_BrkSide_Artery", 
    "Neighborhood_Condition_BrkSide_Feedr_Norm", "Neighborhood_Condition_BrkSide_Feedr_RRNn", 
    "Neighborhood_Condition_BrkSide_Norm", "Neighborhood_Condition_BrkSide_PosN_Norm", 
    "Neighborhood_Condition_BrkSide_RRAn_Feedr", "Neighborhood_Condition_BrkSide_RRAn_Norm", 
    "Neighborhood_Condition_BrkSide_RRNn_Feedr", "Neighborhood_Condition_ClearCr_Feedr_Norm", 
    "Neighborhood_Condition_ClearCr_Norm", "Neighborhood_Condition_CollgCr_Norm", 
    "Neighborhood_Condition_CollgCr_PosN_Norm", "Neighborhood_Condition_Crawfor_Feedr_Norm", 
    "Neighborhood_Condition_Crawfor_Norm", "Neighborhood_Condition_Crawfor_PosA_Norm", 
    "Neighborhood_Condition_Crawfor_PosN_Norm", "Neighborhood_Condition_Edwards_Artery_Norm", 
    "Neighborhood_Condition_Edwards_Feedr_Norm", "Neighborhood_Condition_Edwards_Norm", 
    "Neighborhood_Condition_Edwards_PosN", "Neighborhood_Condition_Gilbert_Feedr_Norm", 
    "Neighborhood_Condition_Gilbert_Norm", "Neighborhood_Condition_Gilbert_RRAn_Norm", 
    "Neighborhood_Condition_Gilbert_RRNn_Norm", "Neighborhood_Condition_IDOTRR_Artery_Norm", 
    "Neighborhood_Condition_IDOTRR_Feedr", "Neighborhood_Condition_IDOTRR_Feedr_Norm", 
    "Neighborhood_Condition_IDOTRR_Norm", "Neighborhood_Condition_IDOTRR_RRAe_Norm", 
    "Neighborhood_Condition_IDOTRR_RRNn_Norm", "Neighborhood_Condition_MeadowV_Norm", 
    "Neighborhood_Condition_Mitchel_Artery_Norm", "Neighborhood_Condition_Mitchel_Feedr_Norm", 
    "Neighborhood_Condition_Mitchel_Norm", "Neighborhood_Condition_NAmes_Artery_Norm", 
    "Neighborhood_Condition_NAmes_Feedr_Norm", "Neighborhood_Condition_NAmes_Norm", 
    "Neighborhood_Condition_NAmes_PosA_Norm", "Neighborhood_Condition_NAmes_PosN_Norm", 
    "Neighborhood_Condition_NPkVill_Norm", "Neighborhood_Condition_NWAmes_Feedr_Norm", 
    "Neighborhood_Condition_NWAmes_Feedr_RRAn", "Neighborhood_Condition_NWAmes_Norm", 
    "Neighborhood_Condition_NWAmes_PosA_Norm", "Neighborhood_Condition_NWAmes_PosN_Norm", 
    "Neighborhood_Condition_NWAmes_RRAn_Norm", "Neighborhood_Condition_NoRidge_Norm", 
    "Neighborhood_Condition_NridgHt_Norm", "Neighborhood_Condition_NridgHt_PosN", 
    "Neighborhood_Condition_OldTown_Artery", "Neighborhood_Condition_OldTown_Artery_Norm", 
    "Neighborhood_Condition_OldTown_Artery_PosA", "Neighborhood_Condition_OldTown_Feedr_Norm", 
    "Neighborhood_Condition_OldTown_Feedr_RRNn", "Neighborhood_Condition_OldTown_Norm", 
    "Neighborhood_Condition_OldTown_RRAn_Feedr", "Neighborhood_Condition_SWISU_Feedr_Norm", 
    "Neighborhood_Condition_SWISU_Norm", "Neighborhood_Condition_SawyerW_Feedr_Norm", 
    "Neighborhood_Condition_SawyerW_Norm", "Neighborhood_Condition_SawyerW_RRAe_Norm", 
    "Neighborhood_Condition_SawyerW_RRNe_Norm", "Neighborhood_Condition_Sawyer_Feedr_Norm", 
    "Neighborhood_Condition_Sawyer_Feedr_RRAe", "Neighborhood_Condition_Sawyer_Norm", 
    "Neighborhood_Condition_Sawyer_PosN_Norm", "Neighborhood_Condition_Sawyer_RRAe_Norm", 
    "Neighborhood_Condition_Somerst_Feedr_Norm", "Neighborhood_Condition_Somerst_Norm", 
    "Neighborhood_Condition_Somerst_RRAn_Norm", "Neighborhood_Condition_Somerst_RRNn_Norm", 
    "Neighborhood_Condition_StoneBr_Norm", "Neighborhood_Condition_Timber_Norm", 
    "Neighborhood_Condition_Veenker_Feedr_Norm", "Neighborhood_Condition_Veenker_Norm", 
    "BldgType_HouseStyle_1Fam_1.5Unf", "BldgType_HouseStyle_1Fam_1Story", 
    "BldgType_HouseStyle_1Fam_2.5Fin", "BldgType_HouseStyle_1Fam_2.5Unf", 
    "BldgType_HouseStyle_1Fam_2Story", "BldgType_HouseStyle_1Fam_SFoyer", 
    "BldgType_HouseStyle_1Fam_SLvl", "BldgType_HouseStyle_2fmCon_1.5Fin", 
    "BldgType_HouseStyle_2fmCon_1.5Unf", "BldgType_HouseStyle_2fmCon_1Story", 
    "BldgType_HouseStyle_2fmCon_2.5Fin", "BldgType_HouseStyle_2fmCon_2.5Unf", 
    "BldgType_HouseStyle_2fmCon_2Story", "BldgType_HouseStyle_2fmCon_SLvl", 
    "BldgType_HouseStyle_Duplex_1.5Fin", "BldgType_HouseStyle_Duplex_1Story", 
    "BldgType_HouseStyle_Duplex_2Story", "BldgType_HouseStyle_Duplex_SFoyer", 
    "BldgType_HouseStyle_Duplex_SLvl", "BldgType_HouseStyle_TwnhsE_1Story", 
    "BldgType_HouseStyle_TwnhsE_2Story", "BldgType_HouseStyle_TwnhsE_SFoyer", 
    "BldgType_HouseStyle_TwnhsE_SLvl", "BldgType_HouseStyle_Twnhs_1Story", 
    "BldgType_HouseStyle_Twnhs_2Story", "BldgType_HouseStyle_Twnhs_SFoyer", 
    "BldgType_HouseStyle_Twnhs_SLvl"
]


iterative_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)

X_train_imputed = X_train.copy()  
X_train_imputed["LotFrontage"] = iterative_imputer.fit_transform(X_train[columns_for_imputation + ["LotFrontage"]])[ :, -1]


# check = X_train_imputed[X_train_imputed["Id"].isin(X_train[X_train["LotFrontage"].isna()]["Id"])]
# check["LotFrontage"].describe() 

# Step 6: Impute missing continuous numerical data in the validation set using the trained imputer
X_val_imputed = X_val.copy()  
X_val_imputed["LotFrontage"] = iterative_imputer.transform(X_val[columns_for_imputation + ["LotFrontage"]])[:, -1]

# check = X_val_imputed[X_val_imputed["Id"].isin(X_val[X_val["LotFrontage"].isna()]["Id"])]
# check["LotFrontage"].describe() 

X_combined = pd.concat([X_train_imputed, X_val_imputed], axis=0, ignore_index=False)
train["LotFrontage"] = X_combined["LotFrontage"]
train["Age_House"] = X_combined["Age_House"]
train["Yrs_Since_Remodel"] = X_combined["Yrs_Since_Remodel"]
train["Age_Garage"] = X_combined["Age_Garage"]
train.to_csv("data/train_after_imputation_EDA.csv", index=False)


# Step 7: Impute missing continuous numerical data in the test set using the trained imputer
test_imputed = test_encoded.copy()  
test_imputed["LotFrontage"] = iterative_imputer.transform(test_encoded[columns_for_imputation + ["LotFrontage"]])[:, -1]

###################################################################################################################################

# Step 8: Correlation analysis and transform numerical terms
"""
numerical_cols = ["LotFrontage", "LotArea", "MasVnrArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
                  "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
                  "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
                  "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF",
                  "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea",
                  "MiscVal", "Age_House", "Yrs_Since_Remodel", "Age_Garage", "SalePrice"]


log transformation
- LotFrontage
- LotArea
- 1stFlrSF
- 2ndFlrSF
- LowQualFinSF
- GrLivArea
- Yrs_Since_Remodel
- Age_Garage
- SalePrice


square root transformation
- TotalBsmtSF
- WoodDeckSF
- BsmtUnfSF
- BsmtFinSF1

cube root transformation
- MasVnrArea
- OpenPorchSF


no need to transform
- BsmtFullBath
- BsmtHalfBath
- FullBath
- HalfBath
- BedroomAbvGr
- KitchenAbvGr
- TotRmsAbvGrd
- Fireplaces
- GarageCars
- GarageArea (already pretty normal)
- EnclosedPorch
- 3SsnPorch
- ScreenPorch
- PoolArea
- MiscVal
- Age_House
- BsmtFinSF2
"""


# Transformation
def transform_and_drop(data, log_cols, sqrt_cols, cube_root_cols):
    # Log transformation
    for col in log_cols:
        data[f"log_{col}"] = np.log1p(data[col])  # Using log1p to handle zeros
        data.drop(columns=[col], inplace=True)
    
    # Square root transformation
    for col in sqrt_cols:
        data[f"sqrt_{col}"] = np.sqrt(data[col])
        data.drop(columns=[col], inplace=True)
    
    # Cube root transformation
    for col in cube_root_cols:
        data[f"cbrt_{col}"] = np.cbrt(data[col])
        data.drop(columns=[col], inplace=True)
    
    return data

# Columns to transform
log_cols = ["LotFrontage", "LotArea", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "Yrs_Since_Remodel", "Age_Garage"]
sqrt_cols = ["TotalBsmtSF", "WoodDeckSF", "BsmtUnfSF", "BsmtFinSF1"]
cube_root_cols = ["MasVnrArea", "OpenPorchSF"]

# Apply transformations to datasets
X_train_final = transform_and_drop(X_train_imputed.copy(), log_cols, sqrt_cols, cube_root_cols)
X_val_final = transform_and_drop(X_val_imputed.copy(), log_cols, sqrt_cols, cube_root_cols)
test_final = transform_and_drop(test_imputed.copy(), log_cols, sqrt_cols, cube_root_cols)


# Step 9: Creating interaction terms for numerical variables
"""
log_GrLivArea × TotRmsAbvGrd: Interaction between total rooms and living area, which could reflect spaciousness.
GarageArea × GarageCars: Correlates garage area with its car capacity, showing efficiency of garage space utilization.
log_Age_Garage × GarageCars: Reflects how the age of the garage relates to its functionality or relevance to car capacity.
BedroomAbvGr / TotRmsAbvGrd: Measures the proportion of bedrooms relative to the total rooms above grade.
2ndFlrSF / GrLivArea: captures the proportion of second-floor square footage to the total above-grade living area.

"""


def create_interactions(df):
    df["Living_Rooms"] = df["log_GrLivArea"] * df["TotRmsAbvGrd"]
    df["Garage_Space"] = df["GarageArea"] * df["GarageCars"]
    df["Garage_AgeCars"] = df["log_Age_Garage"] * df["GarageCars"]
    df["Porch_Age"] = df["EnclosedPorch"] * df["Age_House"]
    df["Ratio_Bedroom_Rooms"] = df["BedroomAbvGr"] / (df["TotRmsAbvGrd"])
    df["Ratio_2ndFlr_Living"] = df["log_2ndFlrSF"] / (df["log_GrLivArea"])

    return df

X_train_final = create_interactions(X_train_final)
X_val_final = create_interactions(X_val_final)
test_final = create_interactions(test_final)




if not os.path.exists("data/model_data"):
    os.makedirs("data/model_data")


# Transform SalePrice
y_train_final = np.log(y_train)
y_val_final = np.log(y_val)

X_train_final.to_csv("data/X_train_reg.csv", index=False)
X_val_final.to_csv("data/X_val_reg.csv", index=False)
y_train_final.to_csv("data/y_train_reg.csv", index=False)
y_val_final.to_csv("data/y_val_reg.csv", index=False)
test_final.to_csv("data/test_final_reg.csv", index=False)


















#################################################################### ML data preparation ####################################################################
train = pd.read_csv("data/train_clean_01.csv")
test = pd.read_csv("data/test_clean_01.csv")

# Step 1: Impute categorical missing data in train set and test set with the mode from training set
train["Electrical"].fillna(train["Electrical"].mode()[0], inplace=True)
test_imputed = test.copy()  # Make a copy of the test set
test_imputed["MSZoning"].fillna(train["MSZoning"].mode()[0], inplace=True)
test_imputed["Utilities"].fillna(train["Utilities"].mode()[0], inplace=True)
test_imputed["KitchenQual"].fillna(train["KitchenQual"].mode()[0], inplace=True)
test_imputed["Functional"].fillna(train["Functional"].mode()[0], inplace=True)



def feature_engineering(df):
    conn = duckdb.connect()
    conn.register("original_df", df)
    query = """
    SELECT 
        *,
        CASE
           WHEN MoSold IN (12, 1, 2) THEN 'Winter'
           WHEN MoSold IN (3, 4, 5) THEN 'Spring'
           WHEN MoSold IN (6, 7, 8) THEN 'Summer'
           ELSE 'Fall'
        END AS Season_Sold,
        IF((YrSold - YearBuilt) < 0, 0, (YrSold - YearBuilt)) AS Age_House,
        IF((YrSold - YearRemodAdd) < 0, 0, (YrSold - YearRemodAdd)) AS Yrs_Since_Remodel,
        IF((YrSold - GarageYrBlt) < 0, 0, (YrSold - GarageYrBlt)) AS Age_Garage
    FROM original_df;
    """
    result = conn.execute(query).fetch_df()
    columns_to_drop = [
        "MoSold", "YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold"
    ]
    result = result.drop(columns=columns_to_drop)
    conn.close()
    return result


train_new = feature_engineering(train)
test_new = feature_engineering(test_imputed)

X = train_new.drop(columns=["SalePrice"], axis=1) 
y = train_new["SalePrice"]

############################### Encode train and test data ########################################
# Step 3: Encode categorical variables separately for train and test sets
nominal_cat = ["MSSubClass", "MSZoning", "LotConfig","Condition1", "Condition2", "Neighborhood", "BldgType", "HouseStyle", 
                "Exterior1st", "Exterior2nd", "CentralAir", "Electrical", "LandContour", "RoofStyle", "RoofMatl", "Heating",
                "Street", "Alley", "Utilities", "MasVnrType", "Foundation", 
               "Functional", "GarageType", "PavedDrive", 
               "Fence", "MiscFeature", "SaleType", "SaleCondition", "Season_Sold"]


ordinal_cat = ["OverallQual", "OverallCond", 
               "LandSlope","LotShape", "HeatingQC", 
               "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", 
               "BsmtFinType1", "BsmtFinType2", "KitchenQual", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", 
               "PoolQC"]

# One-hot encode nominal categorical variables
train_encoded = pd.get_dummies(X, columns=nominal_cat, drop_first=True)
test_encoded = pd.get_dummies(test_new, columns=nominal_cat, drop_first=True)

test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

# Label encode ordinal categorical variables

def ordinal_encoding(df):
    conn = duckdb.connect()
    conn.register("input_df", df)
    query = """
    SELECT 
        *,
        -- OverallQual is already in numeric format, so no need to encode it
        -- OverallCond is already in numeric format, so no need to encode it
        CASE 
            WHEN LandSlope = 'Gtl' THEN 1
            WHEN LandSlope = 'Mod' THEN 2
            WHEN LandSlope = 'Sev' THEN 3
            ELSE 0 
        END AS LandSlope_encoded,
        CASE 
            WHEN LotShape = 'Reg' THEN 1
            WHEN LotShape = 'IR1' THEN 2
            WHEN LotShape = 'IR2' THEN 3
            WHEN LotShape = 'IR3' THEN 4
            ELSE 0 
        END AS LotShape_encoded,
        CASE 
            WHEN HeatingQC = 'Ex' THEN 5
            WHEN HeatingQC = 'Gd' THEN 4
            WHEN HeatingQC = 'TA' THEN 3
            WHEN HeatingQC = 'Fa' THEN 2
            WHEN HeatingQC = 'Po' THEN 1
            ELSE 0
        END AS HeatingQC_encoded,  
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
    FROM input_df;
    """
    result = conn.execute(query).fetch_df()
    columns_to_drop = [
        "LandSlope","LotShape", "HeatingQC", 
               "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", 
               "BsmtFinType1", "BsmtFinType2", "KitchenQual", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", 
               "PoolQC"]
    result = result.drop(columns=columns_to_drop)
    conn.close()
    return result

train_encoded = ordinal_encoding(train_encoded)
test_encoded = ordinal_encoding(test_encoded)


bool_columns_train = train_encoded.select_dtypes(include="bool").columns
bool_columns_test = test_encoded.select_dtypes(include="bool").columns

train_encoded[bool_columns_train] = train_encoded[bool_columns_train].astype("int8")
test_encoded[bool_columns_test] = test_encoded[bool_columns_test].astype("int8")


# Step 4: Split the train dataset into train and validation set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_encoded, y, test_size=0.2, random_state=42)

X_train_ml = X_train.copy()  
X_train_ml["LotFrontage"] = X_train_imputed["LotFrontage"]


# Step 6: Impute missing continuous numerical data in the validation set using the trained imputer
X_val_ml = X_val.copy()  
X_val_ml["LotFrontage"] = X_val_imputed["LotFrontage"]




# Step 7: Impute missing continuous numerical data in the test set using the trained imputer
test_final_ml = test_encoded.copy()  
test_final_ml["LotFrontage"] = test_imputed["LotFrontage"]

############################### Export non-transformed and non-scaled data for non-linear modeling ###############################
if not os.path.exists("data/model_data"):
    os.makedirs("data/model_data")

# Transform SalePrice
y_train_ml = np.log(y_train)
y_val_ml = np.log(y_val)

X_train_ml.drop("Id", axis=1).to_csv("data/model_data/X_train_ml.csv", index=False)
X_val_ml.drop("Id", axis=1).to_csv("data/model_data/X_val_ml.csv", index=False)
test_final_ml.to_csv("data/model_data/test_final_ml.csv", index=False)
y_train_ml.to_csv("data/model_data/y_train_ml.csv", index=False)
y_val_ml.to_csv("data/model_data/y_val_ml.csv", index=False)
