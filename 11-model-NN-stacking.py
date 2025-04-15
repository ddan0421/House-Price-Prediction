import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import pickle
import models
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

################################################# Stacking Models #######################################################
# Load datasets for each base model
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
X_train_nn = pd.read_csv("data/model_data/X_train.csv").values
y_train_nn = pd.read_csv("data/model_data/y_train.csv").values.flatten()

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
nn_model = tf.keras.models.load_model("final_model_NN.keras")

base_models = [
    ("xgb", xgb_model, X_train_xgb, y_train_xgb),
    ("lr", lr_model, X_train_lr, y_train_lr),
    ("svr", svr_model, X_train_svm, y_train_svm),
    # ("lasso", lasso_model, X_train_lasso, y_train_lasso),
    # ("lgbm_bayes", lgbm_bayes_model, X_train_lgbm_bayes, y_train_lgbm_bayes),
    # ("xgb_bayes", xgb_bayes_model, X_train_xgb_bayes, y_train_xgb_bayes),
    # ("lgbm", lgbm_model, X_train_lgbm, y_train_lgbm),
    ("nn", nn_model, X_train_nn, y_train_nn)
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

# Save initial weights of the neural network
initial_weights = nn_model.get_weights()

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_xgb)):
    for i, (name, model, X_train, y_train) in enumerate(base_models):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        if name == "nn":
            # Reset weights before each fold
            model.set_weights(initial_weights)
            early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

            # Train the neural network
            model.fit(
                X_fold_train, y_fold_train,
                validation_data=(X_fold_val, y_fold_val),
                epochs=50,
                batch_size=32,
                verbose=1,
                callbacks=[early_stopping],
                shuffle=True
            )
        elif name == "lr":
            X_fold_train_df = pd.DataFrame(X_fold_train, columns=[f"feature_{j}" for j in range(X_fold_train.shape[1])])
            model_lr = models.sm_glm_gaussian(X_fold_train_df, y_fold_train)
            model = models.constrained_sm_glm_gaussian(X_fold_train_df, y_fold_train, model_lr, 0.05)
        else:
            # Train traditional models
            model.fit(X_fold_train, y_fold_train)

        oof_preds[val_idx, i] = model.predict(X_fold_val).flatten()  

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
X_val_lr = pd.read_csv("data/model_data/X_val_lr.csv")
X_val_svm = pd.read_csv("data/model_data/X_val_svm.csv")
X_val_lasso = pd.read_csv("data/model_data/X_val_lasso.csv")
X_val_lgbm_bayes = pd.read_csv("data/model_data/X_val_lgbm_bayes.csv")
X_val_xgb_bayes = pd.read_csv("data/model_data/X_val_xgb_bayes.csv")
X_val_lgbm = pd.read_csv("data/model_data/X_val_lgbm.csv")
X_val_nn = pd.read_csv("data/model_data/X_val.csv")

y_val = pd.read_csv("data/model_data/y_val_ml.csv")

test_preds = np.zeros((X_val_xgb.shape[0], len(base_models)))

for i, (name, model, _, _) in enumerate(base_models):
    if name == "xgb":
        test_preds[:, i] = model.predict(X_val_xgb)
    elif name == "lr":
        test_preds[:, i] = model.predict(X_val_lr)
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
    elif name == "nn":
        test_preds[:, i] = model.predict(X_val_nn, batch_size=X_val_nn.shape[0]).flatten()

# Final predictions using meta-learner
test_preds = sm.add_constant(test_preds)
final_preds = meta_learner_ols.predict(test_preds)
rmse = root_mean_squared_error(y_val, final_preds)
print(f"Stacking Performance:")
print(f"Root Mean Squared Error: {rmse:.4f}")