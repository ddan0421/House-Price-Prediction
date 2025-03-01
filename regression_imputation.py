import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# create dummy dataset with missing values
data = pd.DataFrame({'var1': [1, 2, np.nan, 4, 5],
                     'var2': [2, np.nan, 4, 5, 6],
                     'var3': [np.nan, 3, 4, 5, np.nan],
                     'var4': [3, 4, 5, 6, 7],
                     'var5': [4, 5, 6, np.nan, 8],
                     'var6': [5, 6, 7, 8, 9]})

# identify variables with missing data
missing_vars = ['var1', 'var2', 'var3']

# identify variables to use as predictors
predictor_vars = ['var4', 'var5', 'var6']

# fit regression model using Bayesian Ridge
imputer = IterativeImputer(estimator=BayesianRidge())

# impute missing values
imputed_data = imputer.fit_transform(data[predictor_vars + missing_vars])

# substitute imputed values for missing values
data[missing_vars] = imputed_data[:, -len(missing_vars):]

print(data)