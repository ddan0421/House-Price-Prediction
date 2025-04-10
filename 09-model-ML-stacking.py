# Tree-based models
# use non-scaled data
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import pickle
import models
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

################################################# Stacking Models #######################################################
# Load datasets for each base model
X_train_xgb = pd.read_csv("data/model_data/X_train_xgb.csv").values
y_train_xgb = pd.read_csv("data/model_data/y_train_xgb.csv").values.flatten()
X_train_rf = pd.read_csv("data/model_data/X_train_rf.csv").values
y_train_rf = pd.read_csv("data/model_data/y_train_rf.csv").values.flatten()
X_train_svm = pd.read_csv("data/model_data/X_train_svm.csv").values
y_train_svm = pd.read_csv("data/model_data/y_train_svm.csv").values.flatten()

# Load pre-trained base models
with open("final_model_xgb.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("final_model_rf.pkl", "rb") as f:
    rf_model = pickle.load(f)   
with open("final_model_svm.pkl", "rb") as f:
    svr_model = pickle.load(f)

base_models = [
    ("xgb", xgb_model, X_train_xgb, y_train_xgb),
    ("rf", rf_model, X_train_rf, y_train_rf),
    ("svr", svr_model, X_train_svm, y_train_svm)
]

# Create an empty array to store OOF predictions
oof_preds = np.zeros((X_train_xgb.shape[0], len(base_models)))

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Generate OOF predictions for each base model
"""
k-fold cross validation overview:

train_idx: 
- In each fold, the train_idx consists of all the data indices that are not part of the val_idx for that fold.

val_idx: 
- The val_idx is unique for each fold. No data point will appear in the validation set for more than one fold.
- the sum of all val_idx across all folds will equal the total number of data points in the dataset.

After the loop completes, the oof_preds array will have predictions for all data points, 
and every prediction will be made by a model that was not trained on the corresponding data point.
This ensures the OOF predictions are unbiased estimates of the model's performance.

"""
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_xgb)):
    for i, (name, model, X_train, y_train) in enumerate(base_models):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        model.fit(X_fold_train, y_fold_train)  # Train on fold-specific data

        if name == "svr":
            oof_preds[val_idx, i] = np.expm1(model.predict(X_fold_val))
        else:
            oof_preds[val_idx, i] = model.predict(X_fold_val)  # Regular predictions for other models

oof_df = pd.DataFrame(oof_preds, columns=[name for name, _, _, _ in base_models])
oof_df["Target"] = y_train_xgb
print("OOF Predictions:")
print(oof_df)

X_agg = sm.add_constant(oof_df.drop(columns=["Target"]))
y_agg = oof_df["Target"]
# Train meta-learner (Linear Regression)
meta_learner_ols = models.sm_ols(X_agg, y_agg)

# Simulate predictions on val data
X_val_xgb = pd.read_csv("data/model_data/X_val_xgb.csv")
X_val_rf = pd.read_csv("data/model_data/X_val_rf.csv")
X_val_svm = pd.read_csv("data/model_data/X_val_svm.csv")

y_val = pd.read_csv("data/model_data/y_val_ml.csv")

test_preds = np.zeros((X_val_xgb.shape[0], len(base_models)))

for i, (name, model, _, _) in enumerate(base_models):
    if name == "xgb":
        test_preds[:, i] = model.predict(X_val_xgb)
    elif name == "rf":
        test_preds[:, i] = model.predict(X_val_rf)
    elif name == "svr":
        test_preds[:, i] = np.expm1(model.predict(X_val_svm))

# Final predictions using meta-learner
test_preds = sm.add_constant(test_preds)
final_preds = meta_learner_ols.predict(test_preds)
rmse = root_mean_squared_error(y_val, final_preds)
print(f"Stacking Performance:")
print(f"Root Mean Squared Error: {rmse:.4f}")