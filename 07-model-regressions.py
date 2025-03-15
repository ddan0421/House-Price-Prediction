import pandas as pd
import numpy as np
import statsmodels.api as sm
import models
from statsmodels.stats.outliers_influence import variance_inflation_factor


X_train = pd.read_csv("data/model_data/X_train.csv")
X_val = pd.read_csv("data/model_data/X_val.csv")
test_final = pd.read_csv("data/model_data/test_final.csv")
y_train = pd.read_csv("data/model_data/y_train.csv")
y_val = pd.read_csv("data/model_data/y_val.csv")

X_train = sm.add_constant(X_train)
X_val = sm.add_constant(X_val)




############################## VIF ##############################
vif_data = pd.DataFrame()
vif_data["feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif_data = vif_data[vif_data["feature"] != "const"]
vif_data = vif_data.dropna()
vif_data = vif_data[vif_data["VIF"] != np.inf]
vif_data.sort_values(by="VIF", ascending=True).to_csv("vif_data.csv", index=False)
selected_features_vif = list(vif_data[vif_data["VIF"] < 10]["feature"])

############################# Random Forest Feature Importance #############################
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data
X_train = pd.read_csv("data/model_data/X_train.csv")
X_val = pd.read_csv("data/model_data/X_val.csv")
y_train = pd.read_csv("data/model_data/y_train.csv")
y_val = pd.read_csv("data/model_data/y_val.csv")

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train.values.ravel())

# Get feature importance
feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf_model.feature_importances_
})

# Sort feature importance in descending order
sorted_indices = np.argsort(feature_importance['importance'])[::-1]
importances = feature_importance['importance'].values[sorted_indices]

# Calculate cumulative importance
cumulative_importance = np.cumsum(importances)

# Define the threshold for cumulative importance (e.g., 90% or 95%)
threshold = 0.95

# Find the index where the cumulative importance reaches or exceeds the threshold
threshold_index = np.where(cumulative_importance >= threshold)[0][0]

# Select features that explain up to the threshold cumulative importance
selected_features_rf = feature_importance['feature'].values[sorted_indices][:threshold_index + 1]

# Print selected features
print(f"Features selected based on cumulative importance ({threshold*100}%):")
print(selected_features_rf)


############################# Selected Numeric #############################
selected_numeric_features = [
    "log_LotArea", "cbrt_MasVnrArea", "sqrt_TotalBsmtSF", "log_1stFlrSF", 
    "log_GrLivArea", "BsmtFullBath", "FullBath", "HalfBath", "BedroomAbvGr", 
    "KitchenAbvGr", "Fireplaces", "GarageCars", "GarageArea", "sqrt_WoodDeckSF", 
    "cbrt_OpenPorchSF", "EnclosedPorch", "Age_House", "TotRmsAbvGrd"
]


# Combine the three lists and remove duplicates using a set
combined_features = list(set(selected_features_vif + selected_features_rf.tolist() + selected_numeric_features))
combined_features.sort()


X_train_new = sm.add_constant(X_train[combined_features])
ols_lr = models.sm_ols(X_train_new, y_train)
glm_lr = models.sm_glm_gaussian(X_train_new, y_train)
