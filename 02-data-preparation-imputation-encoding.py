import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
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
Step 2: Encode categorical variables separately for train and test sets to prevent data leakage, ensuring consistent feature alignment using one-hot encoding and applying label encoding mappings from the train set to the test set.
Step 3: Split the train dataset into train and validation set
Step 4: Impute missing continuous numerical data in the training set using IterativeImputer with BayesianRidge estimator
Step 5: Impute missing continuous numerical data in the validation set using the trained imputer
Step 6: Impute missing continuous numerical data in the test set using the trained imputer


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

############################### Encode train and test data ########################################
# Step 2: Encode categorical variables separately for train and test sets
nominal_cat = ["MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", 
               "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", 
               "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating", 
               "CentralAir", "Electrical", "Functional", "GarageType", "GarageFinish", "PavedDrive", 
               "Fence", "MiscFeature", "SaleType", "SaleCondition"]


ordinal_cat = ["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", 
               "BsmtFinType1", "BsmtFinType2","HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", 
               "PoolQC", "LandSlope"]

# One-hot encode nominal categorical variables
train_encoded = pd.get_dummies(train, columns=nominal_cat, drop_first=True)
test_encoded = pd.get_dummies(test_imputed, columns=nominal_cat, drop_first=True)

test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

# Label encode ordinal categorical variables
label_encoder = LabelEncoder()
for col in ordinal_cat:
    train_encoded[col] = label_encoder.fit_transform(train_encoded[col])
    test_encoded[col] = label_encoder.transform(test_encoded[col])

bool_columns_train = train_encoded.select_dtypes(include="bool").columns
bool_columns_test = test_encoded.select_dtypes(include="bool").columns

train_encoded[bool_columns_train] = train_encoded[bool_columns_train].astype(int)
test_encoded[bool_columns_test] = test_encoded[bool_columns_test].astype(int)


# Step 3: Split the train dataset into train and validation set
from sklearn.model_selection import train_test_split
X = train_encoded.drop(columns=["SalePrice"], axis=1) 
y = train_encoded["SalePrice"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



############################### Train Data LotFrontage Missing Data Imputation ########################################
# Step 4: Impute missing continuous numerical data in the training set using IterativeImputer with BayesianRidge estimator
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

# Step 5: Impute missing continuous numerical data in the validation set using the trained imputer
X_val_imputed = X_val.copy()  
X_val_imputed["LotFrontage"] = iterative_imputer.transform(X_val[columns_for_imputation + ["LotFrontage"]])[:, -1]

# check = X_val_imputed[X_val_imputed["Id"].isin(X_val[X_val["LotFrontage"].isna()]["Id"])]
# check["LotFrontage"].describe() 

X_train_imputed.to_csv("data/X_train.csv", index=False)
X_val_imputed.to_csv("data/X_val.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_val.to_csv("data/y_val.csv", index=False)

X_combined = pd.concat([X_train_imputed, X_val_imputed], axis=0, ignore_index=False)
train["LotFrontage"] = X_combined["LotFrontage"]
train.to_csv("data/train_after_imputation_EDA.csv", index=False)


############################### Test Data LotFrontage Missing Data Imputation ###############################
# Step 6: Impute missing continuous numerical data in the test set using the trained imputer
test_final = test_encoded.copy()  
test_final["LotFrontage"] = iterative_imputer.transform(test_encoded[columns_for_imputation + ["LotFrontage"]])[:, -1]

test_final.to_csv("data/test_final.csv", index=False)