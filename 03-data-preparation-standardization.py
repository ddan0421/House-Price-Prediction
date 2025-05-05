# Standardize data for models that are sensitive to the scale of features (Regression, Neural Networks)
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


X_train = pd.read_csv("data/X_train_reg.csv")
X_val = pd.read_csv("data/X_val_reg.csv")
test_final = pd.read_csv("data/test_final_reg.csv")
y_train = pd.read_csv("data/y_train_reg.csv")
y_val = pd.read_csv("data/y_val_reg.csv")



numerical_variables = [
    "log_LotFrontage", "log_LotArea", "log_1stFlrSF", "log_2ndFlrSF", "log_LowQualFinSF",
    "log_GrLivArea", "log_Yrs_Since_Remodel", "log_Age_Garage",
    "sqrt_TotalBsmtSF", "sqrt_WoodDeckSF",
    "cbrt_MasVnrArea", "cbrt_OpenPorchSF",
    "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
    "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "EnclosedPorch", "3SsnPorch",
    "ScreenPorch", "PoolArea", "MiscVal", "Age_House",
    "Living_Rooms", "Garage_Space", "Garage_AgeCars", "Porch_Age", "Ratio_Bedroom_Rooms", "Ratio_2ndFlr_Living",
    "sqrt_BsmtUnfSF", "sqrt_BsmtFinSF1", "BsmtFinSF2"
]

scaler = StandardScaler()
X_train[numerical_variables] = scaler.fit_transform(X_train[numerical_variables])
X_val[numerical_variables] = scaler.transform(X_val[numerical_variables])
test_final[numerical_variables] = scaler.transform(test_final[numerical_variables])

X_train.drop("Id", axis=1, inplace=True)
X_val.drop("Id", axis=1, inplace=True)

if not os.path.exists("data/model_data"):
    os.makedirs("data/model_data")

X_train.to_csv("data/model_data/X_train_reg.csv", index=False)
X_val.to_csv("data/model_data/X_val_reg.csv", index=False)
test_final.to_csv("data/model_data/test_final_reg.csv", index=False)
y_train.to_csv("data/model_data/y_train_reg.csv", index=False)
y_val.to_csv("data/model_data/y_val_reg.csv", index=False)


