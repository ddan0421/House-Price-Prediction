import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import models



# Load train and val datasets for each base model
X_train_xgb = pd.read_csv("data/model_data/X_train_xgb.csv").values
y_train_xgb = pd.read_csv("data/model_data/y_train_xgb.csv").values.flatten()
X_train_lr = pd.read_csv("data/model_data/X_train_lr.csv").values
y_train_lr = pd.read_csv("data/model_data/y_train_lr.csv").values.flatten()
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


X_val_xgb = pd.read_csv("data/model_data/X_val_xgb.csv").values
X_val_lr = pd.read_csv("data/model_data/X_val_lr.csv").values
X_val_svm = pd.read_csv("data/model_data/X_val_svm.csv").values
X_val_lasso = pd.read_csv("data/model_data/X_val_lasso.csv").values
X_val_lgbm_bayes = pd.read_csv("data/model_data/X_val_lgbm_bayes.csv").values
X_val_xgb_bayes = pd.read_csv("data/model_data/X_val_xgb_bayes.csv").values
X_val_lgbm = pd.read_csv("data/model_data/X_val_lgbm.csv").values

y_val = pd.read_csv("data/model_data/y_val_ml.csv").values.flatten()


# Combine train and validation sets
X_xgb = np.vstack([X_train_xgb, X_val_xgb])
y_xgb = np.concatenate([y_train_xgb, y_val])

X_lr = np.vstack([X_train_lr, X_val_lr])
y_lr = np.concatenate([y_train_lr, y_val])

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


# Load pre-trained base models
with open("final_model_xgb.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("final_model_lr_constraiend.pkl", "rb") as f:
    lr_model = pickle.load(f)   
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


base_models = [
    ("xgb", xgb_model, X_xgb, y_xgb),
    ("lr", lr_model, X_lr, y_lr),
    ("svr", svr_model, X_svm, y_svm),
    # ("lasso", lasso_model, X_lasso, y_lasso),
    ("lgbm_bayes", lgbm_bayes_model, X_lgbm_bayes, y_lgbm_bayes),
    ("xgb_bayes", xgb_bayes_model, X_xgb_bayes, y_xgb_bayes),
    ("lgbm", lgbm_model, X_lgbm, y_lgbm),
]


trained_base_models = {}

# Train base models with the entire training set
for name, model, X, y in base_models:
    if name == "lr":
        # Convert features to DataFrame for SM GLM model
        X_train_df = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(X.shape[1])])
        # Train the unconstrained GLM model
        model_lr = models.sm_glm_gaussian(X_train_df, y)
        # Train the constrained GLM model
        trained_model = models.constrained_sm_glm_gaussian(X_train_df, y, model_lr, 0.05)
    else:
        trained_model = model.fit(X, y)

    trained_base_models[name] = trained_model



# Predict on test data
test_final_ml = pd.read_csv("data/model_data/test_final_ml.csv")
test_final_regress = pd.read_csv("data/model_data/test_final.csv")
test_final_regress = sm.add_constant(test_final_regress)

X_val_xgb = pd.read_csv("data/model_data/X_val_xgb.csv")
X_val_lr = pd.read_csv("data/model_data/X_val_lr.csv")
X_val_svm = pd.read_csv("data/model_data/X_val_svm.csv")
X_val_lasso = pd.read_csv("data/model_data/X_val_lasso.csv")
X_val_lgbm_bayes = pd.read_csv("data/model_data/X_val_lgbm_bayes.csv")
X_val_xgb_bayes = pd.read_csv("data/model_data/X_val_xgb_bayes.csv")
X_val_lgbm = pd.read_csv("data/model_data/X_val_lgbm.csv")

y_val = pd.read_csv("data/model_data/y_val_ml.csv")


X_test_xgb = test_final_ml[X_val_xgb.columns].values
X_test_lr = test_final_regress[X_val_lr.columns].values
X_test_svm = test_final_regress[X_val_svm.columns].values
X_test_lasso = test_final_regress[X_val_lasso.columns].values
X_test_lgbm_bayes = test_final_ml[X_val_lgbm_bayes.columns].values
X_test_xgb_bayes = test_final_ml[X_val_xgb_bayes.columns].values
X_test_lgbm = test_final_ml[X_val_lgbm.columns].values




test_preds = np.zeros((X_test_xgb.shape[0], len(trained_base_models)))

for i, (name, model,) in enumerate(trained_base_models.items()):
    if name == "xgb":
        test_preds[:, i] = model.predict(X_test_xgb)
    elif name == "lr":
        test_preds[:, i] = model.predict(X_test_lr)
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

