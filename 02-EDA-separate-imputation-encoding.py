import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Importing IterativeImputer with the experimental module
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

train = pd.read_csv("data/train_clean_01.csv")
test = pd.read_csv("data/test_clean_01.csv")

def missing_col(df):
    missing_col = df.isna().sum()
    missing_col_df = pd.DataFrame(missing_col[missing_col > 0])
#     print(missing_col_df.index.tolist())
    return missing_col_df

train_missing = missing_col(train)
test_missing = missing_col(test)


"""
Workflow:
Step 1: Impute categorical missing data in train set and test set with the mode from training set
Step 2: Combine the training and test datasets
Step 3: Encode the combined dataset
Step 4: Split the combined dataset back into train and test
Step 5: Split the train dataset into train and validation set
Step 6: Impute missing continuous numerical data in the training set using IterativeImputer with BayesianRidge estimator
Step 7: Impute missing continuous numerical data in the validation set using the trained imputer
Step 8: Impute missing continuous numerical data in the test set using the trained imputer


Impute Missing Values Separately:
- Train the regression imputation model using only the training subset.
- Use this trained imputation model to impute missing values in the train, validation, and test subsets.

"""

# Step 1: Impute categorical missing data in train set and test set with the mode from training set
train["Electrical"].fillna(train["Electrical"].mode()[0], inplace=True)
test_imputed = test.copy()  # Make a copy of the test set
test_imputed["MSZoning"].fillna(train["MSZoning"].mode()[0], inplace=True)
test_imputed["Utilities"].fillna(train["Utilities"].mode()[0], inplace=True)
test_imputed["KitchenQual"].fillna(train["KitchenQual"].mode()[0], inplace=True)
test_imputed["Functional"].fillna(train["Functional"].mode()[0], inplace=True)



############################### Encode combined train and test data ########################################
# Step 2: Combine the training and test datasets
train_features = train.drop("SalePrice", axis=1)
feature_df = pd.concat([train_features, test_imputed], axis=0, ignore_index=True)


# Step 3: Encode the combined dataset
nominal_cat = ["MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", 
               "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", 
               "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating", 
               "CentralAir", "Electrical", "Functional", "GarageType", "GarageFinish", "PavedDrive", 
               "Fence", "MiscFeature", "SaleType", "SaleCondition"]


ordinal_cat = ["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", 
               "BsmtFinType1", "BsmtFinType2","HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", 
               "PoolQC", "LandSlope"]


feature_df_encoded = pd.get_dummies(feature_df, columns=nominal_cat, drop_first=True)

label_encoder = LabelEncoder()
for col in ordinal_cat:
    feature_df_encoded[col] = label_encoder.fit_transform(feature_df_encoded[col])

bool_columns = feature_df_encoded.select_dtypes(include="bool").columns
for col in bool_columns:
    feature_df_encoded[col] = feature_df_encoded[col].astype(int)

# Step 4: Split the combined dataset back into train and test
train_encoded = pd.concat([feature_df_encoded.iloc[:1460, :], train["SalePrice"]], axis=1)
test_encoded = feature_df_encoded.iloc[1460:, :]

# Step 5: Split the train dataset into train and validation set
from sklearn.model_selection import train_test_split
X = train_encoded.drop(columns=["SalePrice"], axis=1) 
y = train_encoded["SalePrice"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



############################### Train Data LotFrontage Missing Data Imputation ########################################
# Step 6: Impute missing continuous numerical data in the training set using IterativeImputer with BayesianRidge estimator
# Columns to be used as predictors for imputing "LotFrontage"
columns_for_imputation = ["LotArea", "Street_Pave",
                          "LotShape_IR2", "LotShape_IR3", "LotShape_Reg",
                          "LandContour_HLS", "LandContour_Low", "LandContour_Lvl", 
                          "LotConfig_CulDSac", "LotConfig_FR2", "LotConfig_FR3", "LotConfig_Inside",
                          "Neighborhood_Blueste", "Neighborhood_BrDale",
                          "Neighborhood_BrkSide", "Neighborhood_ClearCr", "Neighborhood_CollgCr", "Neighborhood_Crawfor",
                            "Neighborhood_Edwards", "Neighborhood_Gilbert", "Neighborhood_IDOTRR", "Neighborhood_MeadowV",
                            "Neighborhood_Mitchel", "Neighborhood_NAmes", "Neighborhood_NPkVill", "Neighborhood_NWAmes",
                            "Neighborhood_NoRidge", "Neighborhood_NridgHt", "Neighborhood_OldTown", "Neighborhood_SWISU",
                            "Neighborhood_Sawyer", "Neighborhood_SawyerW", "Neighborhood_Somerst", "Neighborhood_StoneBr",
                            "Neighborhood_Timber", "Neighborhood_Veenker", 
                            "BldgType_2fmCon", "BldgType_Duplex", "BldgType_Twnhs", "BldgType_TwnhsE"]

iterative_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)

X_train_imputed = X_train.copy()  
X_train_imputed["LotFrontage"] = iterative_imputer.fit_transform(X_train[columns_for_imputation + ["LotFrontage"]])[ :, -1]


# check = X_train_imputed[X_train_imputed["Id"].isin(X_train[X_train["LotFrontage"].isna()]["Id"])]
# check["LotFrontage"].describe() 

# Step 7: Impute missing continuous numerical data in the validation set using the trained imputer
X_val_imputed = X_val.copy()  
X_val_imputed["LotFrontage"] = iterative_imputer.transform(X_val[columns_for_imputation + ["LotFrontage"]])[:, -1]

# check = X_val_imputed[X_val_imputed["Id"].isin(X_val[X_val["LotFrontage"].isna()]["Id"])]
# check["LotFrontage"].describe() 

X_train_imputed.to_csv("data/X_train.csv", index=False)
X_val_imputed.to_csv("data/X_val.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_val.to_csv("data/y_val.csv", index=False)


############################### Test Data LotFrontage Missing Data Imputation ###############################
# Step 8: Impute missing continuous numerical data in the test set using the trained imputer
test_final = test_encoded.copy()  
test_final["LotFrontage"] = iterative_imputer.transform(test_encoded[columns_for_imputation + ["LotFrontage"]])[:, -1]

test_final.to_csv("data/test_final.csv", index=False)