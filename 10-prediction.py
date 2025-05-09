import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import catboost as cb



# Load train and val datasets for each base model
X_train_xgb = pd.read_csv("data/model_data/X_train_xgb.csv").values
y_train_xgb = pd.read_csv("data/model_data/y_train_xgb.csv").values.flatten()
X_train_ridge = pd.read_csv("data/model_data/X_train_ridge.csv").values
y_train_ridge = pd.read_csv("data/model_data/y_train_ridge.csv").values.flatten()
X_train_svm = pd.read_csv("data/model_data/X_train_svm.csv").values
y_train_svm = pd.read_csv("data/model_data/y_train_svm.csv").values.flatten()
X_train_lasso = pd.read_csv("data/model_data/X_train_lasso.csv").values
y_train_lasso = pd.read_csv("data/model_data/y_train_lasso.csv").values.flatten()
X_train_lgbm_bayes = pd.read_csv("data/model_data/X_train_lgbm_bayes.csv").values
y_train_lgbm_bayes = pd.read_csv("data/model_data/y_train_lgbm_bayes.csv").values.flatten()
X_train_xgb_bayes = pd.read_csv("data/model_data/X_train_xgb_bayes.csv").values
y_train_xgb_bayes = pd.read_csv("data/model_data/y_train_xgb_bayes.csv").values.flatten()
X_train_lgbm = pd.read_csv("data/model_data/X_train_lgbm.csv").values
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

X_val_xgb = pd.read_csv("data/model_data/X_val_xgb.csv").values
X_val_ridge = pd.read_csv("data/model_data/X_val_ridge.csv").values
X_val_svm = pd.read_csv("data/model_data/X_val_svm.csv").values
X_val_lasso = pd.read_csv("data/model_data/X_val_lasso.csv").values
X_val_lgbm_bayes = pd.read_csv("data/model_data/X_val_lgbm_bayes.csv").values
X_val_xgb_bayes = pd.read_csv("data/model_data/X_val_xgb_bayes.csv").values
X_val_lgbm = pd.read_csv("data/model_data/X_val_lgbm.csv").values
X_val_rf = pd.read_csv("data/model_data/X_val_rf.csv").values
X_val_knn = pd.read_csv("data/model_data/X_val_knn.csv").values
X_val_dt = pd.read_csv("data/model_data/X_val_dt.csv").values
X_val_sdt = pd.read_csv("data/model_data/X_val_sdt.csv").values
X_val_enet = pd.read_csv("data/model_data/X_val_enet.csv").values
X_val_et = pd.read_csv("data/model_data/X_val_et.csv").values
X_val_cat = pd.read_csv("data/model_data/X_val_cat.csv")

y_val = pd.read_csv("data/model_data/y_val_ml.csv").values.flatten()


# Combine train and validation sets
X_xgb = np.vstack([X_train_xgb, X_val_xgb])
y_xgb = np.concatenate([y_train_xgb, y_val])

X_ridge = np.vstack([X_train_ridge, X_val_ridge])
y_ridge = np.concatenate([y_train_ridge, y_val])

X_svm = np.vstack([X_train_svm, X_val_svm])
y_svm = np.concatenate([y_train_svm, y_val])

X_lasso = np.vstack([X_train_lasso, X_val_lasso])
y_lasso = np.concatenate([y_train_lasso, y_val])

X_lgbm_bayes = np.vstack([X_train_lgbm_bayes, X_val_lgbm_bayes])
y_lgbm_bayes = np.concatenate([y_train_lgbm_bayes, y_val])

X_xgb_bayes = np.vstack([X_train_xgb_bayes, X_val_xgb_bayes])
y_xgb_bayes = np.concatenate([y_train_xgb_bayes, y_val])

X_lgbm = np.vstack([X_train_lgbm, X_val_lgbm])
y_lgbm = np.concatenate([y_train_lgbm, y_val])

X_rf = np.vstack([X_train_rf, X_val_rf])
y_rf = np.concatenate([y_train_rf, y_val])

X_knn = np.vstack([X_train_knn, X_val_knn])
y_knn = np.concatenate([y_train_knn, y_val])

X_dt = np.vstack([X_train_dt, X_val_dt])
y_dt = np.concatenate([y_train_dt, y_val])

X_sdt = np.vstack([X_train_sdt, X_val_sdt])
y_sdt = np.concatenate([y_train_sdt, y_val])

X_enet = np.vstack([X_train_enet, X_val_enet])
y_enet = np.concatenate([y_train_enet, y_val])

X_et = np.vstack([X_train_et, X_val_et])
y_et = np.concatenate([y_train_et, y_val])

X_cat = pd.concat([X_train_cat, X_val_cat], axis=0, ignore_index=True)
y_cat = np.concatenate([y_train_cat, y_val])

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
    ("xgb", xgb_model, X_xgb, y_xgb),
    ("ridge", ridge_model, X_ridge, y_ridge),
    ("svr", svr_model, X_svm, y_svm),
    ("lasso", lasso_model, X_lasso, y_lasso),
    ("lgbm_bayes", lgbm_bayes_model, X_lgbm_bayes, y_lgbm_bayes),
    ("xgb_bayes", xgb_bayes_model, X_xgb_bayes, y_xgb_bayes),
    ("lgbm", lgbm_model, X_lgbm, y_lgbm), 
    ("rf", rf_model, X_rf, y_rf),
    ("knn", knn_model, X_knn, y_knn),
    ("dt", dt_model, X_dt, y_dt),
    # ("sdt", sdt_model, X_sdt, y_sdt),
    ("enet", enet_model, X_enet, y_enet),
    ("et", et_model, X_et, y_et),
    # ("cat_optuna", final_model_cat_optuna, X_cat, y_cat),
    # ("cat_gridsearch", final_model_cat_gridsearch, X_cat, y_cat),
    ("cat_basic", final_model_cat_basic, X_cat, y_cat)
]


trained_base_models = {}

# Train base models with the entire training set
for name, model, X, y in base_models:
    if "cat" in name:
        train_pool = cb.Pool(data=X, label=y, cat_features=cat_columns)
        model.fit(train_pool, verbose=False)
    else:
        model.fit(X, y)
    trained_base_models[name] = model


# Predict on test data
test_final_ml = pd.read_csv("data/model_data/test_final_ml.csv")
test_final_regress = pd.read_csv("data/model_data/test_final_reg.csv")
test_final_regress = sm.add_constant(test_final_regress)
test_final_cat = pd.read_csv("data/model_data/test_final_cat.csv")

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

y_val = pd.read_csv("data/model_data/y_val_ml.csv")


X_test_xgb = test_final_ml[X_val_xgb.columns].values
X_test_ridge = test_final_regress[X_val_ridge.columns].values
X_test_svm = test_final_regress[X_val_svm.columns].values
X_test_lasso = test_final_regress[X_val_lasso.columns].values
X_test_lgbm_bayes = test_final_ml[X_val_lgbm_bayes.columns].values
X_test_xgb_bayes = test_final_ml[X_val_xgb_bayes.columns].values
X_test_lgbm = test_final_ml[X_val_lgbm.columns].values
X_test_rf = test_final_ml[X_val_rf.columns].values
X_test_knn = test_final_regress[X_val_knn.columns].values
X_test_dt = test_final_ml[X_val_dt.columns].values
X_test_sdt = test_final_ml[X_val_sdt.columns].values
X_test_enet = test_final_regress[X_val_enet.columns].values
X_test_et = test_final_ml[X_val_et.columns].values
X_test_cat = test_final_cat.drop(columns=["Id"], axis=1)




test_preds = np.zeros((X_test_xgb.shape[0], len(trained_base_models)))

for i, (name, model,) in enumerate(trained_base_models.items()):
    if name == "xgb":
        test_preds[:, i] = model.predict(X_test_xgb)
    elif name == "ridge":
        test_preds[:, i] = model.predict(X_test_ridge)
    elif name == "svr":
        test_preds[:, i] = model.predict(X_test_svm)
    elif name == "lasso":
        test_preds[:, i] = model.predict(X_test_lasso)
    elif name == "lgbm_bayes":
        test_preds[:, i] = model.predict(X_test_lgbm_bayes)
    elif name == "xgb_bayes":
        test_preds[:, i] = model.predict(X_test_xgb_bayes)
    elif name == "lgbm":
        test_preds[:, i] = model.predict(X_test_lgbm)
    elif name == "rf":
        test_preds[:, i] = model.predict(X_test_rf)
    elif name == "knn":
        test_preds[:, i] = model.predict(X_test_knn)
    elif name == "dt":
        test_preds[:, i] = model.predict(X_test_dt)
    elif name == "sdt":
        test_preds[:, i] = model.predict(X_test_sdt)
    elif name == "enet":
        test_preds[:, i] = model.predict(X_test_enet)
    elif name == "et":
        test_preds[:, i] = model.predict(X_test_et)
    elif name == "cat_optuna":
        test_pool = cb.Pool(data=X_test_cat, cat_features=cat_columns)
        test_preds[:, i] = model.predict(test_pool)
    elif name == "cat_gridsearch":
        test_pool = cb.Pool(data=X_test_cat, cat_features=cat_columns)
        test_preds[:, i] = model.predict(test_pool)
    elif name == "cat_basic":
        test_pool = cb.Pool(data=X_test_cat, cat_features=cat_columns)
        test_preds[:, i] = model.predict(test_pool)
    

# Load meta-learner model
with open("meta_learner_ols.pkl", "rb") as f:
    meta_learner_ols = pickle.load(f)

# Final predictions using meta-learner
test_preds = sm.add_constant(test_preds)
final_preds = meta_learner_ols.predict(test_preds)

# Create a DataFrame with Id and SalePrice
final_preds_df = pd.DataFrame({
    "Id": test_final_ml["Id"],  
    "SalePrice": np.exp(final_preds)
})

final_preds_df.to_csv("data/submission.csv", index=False)

