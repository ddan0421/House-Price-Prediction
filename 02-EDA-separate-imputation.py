import pandas as pd
import numpy as np


train = pd.read_csv("data/train_clean_01.csv")
test = pd.read_csv("data/test_clean_01.csv")

def missing_col(df):
    missing_col = df.isna().sum()
    missing_col_df = pd.DataFrame(missing_col[missing_col > 0])
#     print(missing_col_df.index.tolist())
    return missing_col_df

train_missing = missing_col(train)
test_missing = missing_col(test)






############################### train data ###############################
# missing data in train data
# LotFrontage and Electrical have missing data

# Electrical has only one missing data, we can simply drop it
train[train["Electrical"].isna()]
train.dropna(subset=["Electrical"], inplace=True)


# LotFrontage has 259 missing data, we can build a regression model to predict it 
"""
Split the Training and Validation Sets First:
- Divide the dataset into training and validation subsets before performing any data preprocessing, including imputation.

Impute Missing Values Separately:
- Train the regression imputation model using only the training subset.
- Use this trained imputation model to impute missing values in both the training and validation subsets.


For LotFrontage, we will use sklearn's IterativeImputer class with the BayesianRidge estimator to impute missing values.
"""
