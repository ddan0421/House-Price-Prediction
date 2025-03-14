import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

train_before = pd.read_csv("data/train_before_imputation_EDA.csv")
train_after = pd.read_csv("data/train_after_imputation_EDA.csv")

categorical_cols = ['MSSubClass','MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 
       'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition']

numerical_cols = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold', 'SalePrice']


if not os.path.exists("plots"):
    os.makedirs("plots")


# Bar chart for categorical columns
for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, data=train_after)
    plt.title(f"Bar Chart of {col}")
    plt.xticks(rotation=90)
    plt.savefig(f"plots/bar_plot_{col}.png")
    plt.close()


# Histogram + Density chart and QQ plot for numerical columns
for col in numerical_cols:
    plt.figure(figsize=(10, 5))
    sns.histplot(train_after[col], kde=True)
    plt.title(f"Histogram and Density Plot of {col}")
    plt.savefig(f"plots/histogram_density_plot_{col}.png")
    plt.close()

for col in numerical_cols:
    plt.figure(figsize=(10, 5))
    stats.probplot(train_after[col], dist="norm", plot=plt)
    plt.title(f"QQ Plot of {col}")
    plt.savefig(f"plots/QQ_plot_{col}.png")
    plt.close()