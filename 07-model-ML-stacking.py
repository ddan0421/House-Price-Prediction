import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import pickle
import models
import statsmodels.api as sm
import models 
import catboost as cb

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

"""
XGBoost Regressor (Hyperparameter Tuned)
Ridge Linear Regression Model
SVM Model 
Lasso Model
LightGBM Regressor (Bayesian Optimized)
XGBBoost Regressor (Bayesian Optimized)
LightGBM Regressor (Hyperparameter Tuned)
random forest regressor (Hyperparameter Tuned)
KNN Regressor (Hyperparameter Tuned)
Decision Tree Regressor (Hyperparameter Tuned)
Shallow decision tree regressor (Hyperparameter Tuned)
Elastic Net Regressor (Hyperparameter Tuned)
Extra Trees Regressor (Hyperparameter Tuned)
CatBoost Regressor (Hyperparameter Tuned)
CatBoost Regressor (Optuna Optimized)
Basic CatBoost Regressor
"""

################################################# Stacking Models #######################################################
# Load datasets for each base model
X_train_xgb = pd.read_csv("data/model_data/X_train_xgb.csv").values
y_train_xgb = pd.read_csv("data/model_data/y_train_xgb.csv").values.flatten()
X_train_ridge = pd.read_csv("data/model_data/X_train_ridge.csv").values
y_train_ridge = pd.read_csv("data/model_data/y_train_ridge.csv").values.flatten()
X_train_svm = pd.read_csv("data/model_data/X_train_svm.csv").values
y_train_svm = pd.read_csv("data/model_data/y_train_svm.csv").values.flatten()
X_train_lasso = pd.read_csv("data/model_data/X_train_lasso.csv").values
y_train_lasso = pd.read_csv("data/model_data/y_train_lasso.csv").values.flatten()
X_train_lgbm_bayes = pd.read_csv("data/model_data/X_train_lgbm_bayes.csv")
y_train_lgbm_bayes = pd.read_csv("data/model_data/y_train_lgbm_bayes.csv").values.flatten()
X_train_xgb_bayes = pd.read_csv("data/model_data/X_train_xgb_bayes.csv").values
y_train_xgb_bayes = pd.read_csv("data/model_data/y_train_xgb_bayes.csv").values.flatten()
X_train_lgbm = pd.read_csv("data/model_data/X_train_lgbm.csv")
y_train_lgbm = pd.read_csv("data/model_data/y_train_lgbm.csv").values.flatten()
X_train_rf = pd.read_csv("data/model_data/X_train_rf.csv").values
y_train_rf = pd.read_csv("data/model_data/y_train_rf.csv").values.flatten()
X_train_knn = pd.read_csv("data/model_data/X_train_knn.csv").values
y_train_knn = pd.read_csv("data/model_data/y_train_knn.csv").values.flatten()
X_train_dt = pd.read_csv("data/model_data/X_train_dt.csv").values
y_train_dt = pd.read_csv("data/model_data/y_train_dt.csv").values.flatten()
X_train_sdt = pd.read_csv("data/model_data/X_train_sdt.csv").values
y_train_sdt = pd.read_csv("data/model_data/y_train_sdt.csv").values.flatten()
X_train_enet = pd.read_csv("data/model_data/X_train_enet.csv").values
y_train_enet = pd.read_csv("data/model_data/y_train_enet.csv").values.flatten()
X_train_et = pd.read_csv("data/model_data/X_train_et.csv").values
y_train_et = pd.read_csv("data/model_data/y_train_et.csv").values.flatten()
X_train_cat = pd.read_csv("data/model_data/X_train_cat.csv")
y_train_cat = pd.read_csv("data/model_data/y_train_cat.csv").values.flatten()

cat_columns = X_train_cat.select_dtypes(include="object").columns.tolist()
cat_columns.append("MSSubClass")

# convert categorical columns to category type
X_train_lgbm[cat_columns] = X_train_lgbm[cat_columns].astype("category")
X_train_lgbm_bayes[cat_columns] = X_train_lgbm_bayes[cat_columns].astype("category")


# Load pre-trained base models
with open("final_model_xgb.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("final_model_ridge.pkl", "rb") as f:
    ridge_model = pickle.load(f)   
with open("final_model_svm.pkl", "rb") as f:
    svr_model = pickle.load(f)
with open("final_model_lasso.pkl", "rb") as f:
    lasso_model = pickle.load(f)
with open("final_model_LGBM_bayes.pkl", "rb") as f:
    lgbm_bayes_model = pickle.load(f)
with open("final_model_xgb_bayes.pkl", "rb") as f:
    xgb_bayes_model = pickle.load(f)
with open("final_model_lgbm.pkl", "rb") as f:
    lgbm_model = pickle.load(f)
with open("final_model_rf.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("final_model_knn.pkl", "rb") as f:
    knn_model = pickle.load(f)
with open("final_model_dt.pkl", "rb") as f:
    dt_model = pickle.load(f)
with open("final_model_sdt.pkl", "rb") as f:
    sdt_model = pickle.load(f)
with open("final_model_enet.pkl", "rb") as f:
    enet_model = pickle.load(f)
with open("final_model_et.pkl", "rb") as f:
    et_model = pickle.load(f)

final_model_cat_optuna = cb.CatBoostRegressor(cat_features=cat_columns)
final_model_cat_optuna.load_model("final_model_catboost_optuna.cbm")

final_model_cat_gridsearch = cb.CatBoostRegressor(cat_features=cat_columns)
final_model_cat_gridsearch.load_model("final_model_catboost_gridsearch.cbm")

final_model_cat_basic = cb.CatBoostRegressor(cat_features=cat_columns)
final_model_cat_basic.load_model("final_model_catboost_basic.cbm")


base_models = [
    ("xgb", xgb_model, X_train_xgb, y_train_xgb),
    ("ridge", ridge_model, X_train_ridge, y_train_ridge),
    ("svr", svr_model, X_train_svm, y_train_svm),
    ("lasso", lasso_model, X_train_lasso, y_train_lasso),
    ("lgbm_bayes", lgbm_bayes_model, X_train_lgbm_bayes, y_train_lgbm_bayes),
    ("xgb_bayes", xgb_bayes_model, X_train_xgb_bayes, y_train_xgb_bayes),
    ("lgbm", lgbm_model, X_train_lgbm, y_train_lgbm),
    ("rf", rf_model, X_train_rf, y_train_rf),
    ("knn", knn_model, X_train_knn, y_train_knn),
    ("dt", dt_model, X_train_dt, y_train_dt),
    # ("sdt", sdt_model, X_train_sdt, y_train_sdt),
    ("enet", enet_model, X_train_enet, y_train_enet),
    ("et", et_model, X_train_et, y_train_et),
    # ("cat_optuna", final_model_cat_optuna, X_train_cat, y_train_cat),
    # ("cat_gridsearch", final_model_cat_gridsearch, X_train_cat, y_train_cat),
    ("cat_basic", final_model_cat_basic, X_train_cat, y_train_cat)
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
    print(f"Processing Fold {fold + 1}...")
    for i, (name, model, X_train, y_train) in enumerate(base_models):
        if "cat" in name:
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train[train_idx]

            train_pool = cb.Pool(data=X_fold_train, label=y_fold_train, cat_features=cat_columns)

            model.fit(train_pool, verbose=False)
        elif "lgbm" in name:
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train[train_idx]      

            model.fit(X_fold_train, y_fold_train, categorical_feature=cat_columns)      
            
        else:
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            model.fit(X_fold_train, y_fold_train)

        oof_preds[val_idx, i] = model.predict(X_fold_val)  

oof_df = pd.DataFrame(oof_preds, columns=[name for name, _, _, _ in base_models])
oof_df["Target"] = y_train_xgb
print("OOF Predictions:")
print(oof_df)

X_agg = sm.add_constant(oof_df.drop(columns=["Target"]))
y_agg = oof_df["Target"]
# Train meta-learner (Linear Regression)
meta_learner_ols = models.sm_ols(X_agg, y_agg)

# Save the trained model for future use (stacking)
with open("meta_learner_ols.pkl", "wb") as f:
    pickle.dump(meta_learner_ols, f)
print("Meta-learner model saved to meta_learner_ols.pkl")

# Simulate predictions on val data
"""
During the prediction phase, we will first use the base models (trained on the entire training data) to generate predictions on the validation data.
Then, we will use these predictions as features for the trained meta-learner from above.
"""
X_val_xgb = pd.read_csv("data/model_data/X_val_xgb.csv")
X_val_ridge = pd.read_csv("data/model_data/X_val_ridge.csv")
X_val_svm = pd.read_csv("data/model_data/X_val_svm.csv")
X_val_lasso = pd.read_csv("data/model_data/X_val_lasso.csv")
X_val_lgbm_bayes = pd.read_csv("data/model_data/X_val_lgbm_bayes.csv")
X_val_xgb_bayes = pd.read_csv("data/model_data/X_val_xgb_bayes.csv")
X_val_lgbm = pd.read_csv("data/model_data/X_val_lgbm.csv")
X_val_rf = pd.read_csv("data/model_data/X_val_rf.csv")
X_val_knn = pd.read_csv("data/model_data/X_val_knn.csv")
X_val_dt = pd.read_csv("data/model_data/X_val_dt.csv")
X_val_sdt = pd.read_csv("data/model_data/X_val_sdt.csv")
X_val_enet = pd.read_csv("data/model_data/X_val_enet.csv")
X_val_et = pd.read_csv("data/model_data/X_val_et.csv")
X_val_cat = pd.read_csv("data/model_data/X_val_cat.csv")

y_val = pd.read_csv("data/model_data/y_val_ml.csv")


# Convert categorical columns to category type
X_val_lgbm[cat_columns] = X_val_lgbm[cat_columns].astype("category")
X_val_lgbm_bayes[cat_columns] = X_val_lgbm_bayes[cat_columns].astype("category")

test_preds = np.zeros((X_val_xgb.shape[0], len(base_models)))

for i, (name, model, _, _) in enumerate(base_models):
    if name == "xgb":
        test_preds[:, i] = model.predict(X_val_xgb)
    elif name == "ridge":
        test_preds[:, i] = model.predict(X_val_ridge)
    elif name == "svr":
        test_preds[:, i] = model.predict(X_val_svm)
    elif name == "lasso":
        test_preds[:, i] = model.predict(X_val_lasso)
    elif name == "lgbm_bayes":
        test_preds[:, i] = model.predict(X_val_lgbm_bayes)
    elif name == "xgb_bayes":
        test_preds[:, i] = model.predict(X_val_xgb_bayes)
    elif name == "lgbm":
        test_preds[:, i] = model.predict(X_val_lgbm)
    elif name == "rf":
        test_preds[:, i] = model.predict(X_val_rf)
    elif name == "knn":
        test_preds[:, i] = model.predict(X_val_knn)
    elif name == "dt":
        test_preds[:, i] = model.predict(X_val_dt)
    elif name == "sdt":
        test_preds[:, i] = model.predict(X_val_sdt)
    elif name == "enet":
        test_preds[:, i] = model.predict(X_val_enet)
    elif name == "et":
        test_preds[:, i] = model.predict(X_val_et)
    elif name == "cat_optuna":
        test_preds[:, i] = model.predict(X_val_cat)
    elif name == "cat_gridsearch":
        test_preds[:, i] = model.predict(X_val_cat)
    elif name == "cat_basic":
        test_preds[:, i] = model.predict(X_val_cat)


# Final predictions using meta-learner
test_preds = sm.add_constant(test_preds)
final_preds = meta_learner_ols.predict(test_preds)
rmse = root_mean_squared_error(y_val, final_preds)
print(f"Stacking Performance:")
print(f"Root Mean Squared Error: {rmse:.4f}")