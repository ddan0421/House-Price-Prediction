import pandas as pd
import numpy as np
import statsmodels.api as sm
import models
import duckdb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.svm import SVR


X_train = pd.read_csv("data/model_data/X_train.csv")
X_val = pd.read_csv("data/model_data/X_val.csv")
test_final = pd.read_csv("data/model_data/test_final.csv")
y_train = pd.read_csv("data/model_data/y_train.csv")
y_val = pd.read_csv("data/model_data/y_val.csv")

random_state = 42

X_train = sm.add_constant(X_train)
X_val = sm.add_constant(X_val)

############################## Feature Selection 1: VIF ##############################
vif_data = pd.DataFrame()
vif_data["feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif_data = vif_data[vif_data["feature"] != "const"]
vif_data = vif_data.dropna()
vif_data = vif_data[vif_data["VIF"] != np.inf]
selected_features_vif = list(vif_data[vif_data["VIF"] < 10]["feature"])

############################# Feature Selection 2: Random Forest Feature Importance #############################
X_train = pd.read_csv("data/model_data/X_train.csv")
X_val = pd.read_csv("data/model_data/X_val.csv")
y_train = pd.read_csv("data/model_data/y_train.csv")
y_val = pd.read_csv("data/model_data/y_val.csv")

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=200, random_state=random_state)
rf_model.fit(X_train, y_train.values.ravel())

# Get feature importance
feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf_model.feature_importances_
})

conn = duckdb.connect()
# Define the threshold for cumulative importance
threshold = 0.95

# Calculate cumulative importance and filter features
# Keep enough features such that their cumulative importance adds up to 95% of the total importance.
query = f"""
WITH sorted_importance AS (
    SELECT
        feature,
        importance
    FROM feature_importance
    ORDER BY importance DESC
),
cumulative_importance AS (
    SELECT
        feature,
        importance,
        SUM(importance) OVER (ORDER BY importance DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_importance,
        ROW_NUMBER() OVER (ORDER BY importance DESC) AS row_number
    FROM sorted_importance
),
threshold_index AS (
    SELECT
        MIN(row_number) AS threshold_row
    FROM cumulative_importance
    WHERE cumulative_importance >= {threshold}
)
SELECT
    feature
FROM cumulative_importance
INNER JOIN threshold_index
ON cumulative_importance.row_number <= threshold_index.threshold_row;

"""

# Execute the query to get selected features
selected_features_rf = conn.execute(query).fetch_df()["feature"].to_list()
conn.close()


############################# Feature Selection 3: Selected Numeric Variables based on Correlation Matrix and Domain Knowledge #############################
selected_numeric_features = [
    "log_LotArea", "cbrt_MasVnrArea", "sqrt_TotalBsmtSF", "log_1stFlrSF", 
    "log_GrLivArea", "BsmtFullBath", "FullBath", "HalfBath", "BedroomAbvGr", 
    "KitchenAbvGr", "Fireplaces", "GarageCars", "GarageArea", "sqrt_WoodDeckSF", 
    "cbrt_OpenPorchSF", "EnclosedPorch", "Age_House", "TotRmsAbvGrd"
]


# Combine the three lists and remove duplicates using a set
combined_features = list(set(selected_features_vif + selected_features_rf + selected_numeric_features))
combined_features.sort()



############################# Linear Regression #############################
X_train_NN= X_train[combined_features]
X_val_NN = X_val[combined_features]



############################# Feed-Forward Neural Network #############################
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner.tuners import RandomSearch
import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import random

# Set the random seed for reproducibility
random.seed(42)  # Python random seed
np.random.seed(42)  # NumPy random seed
tf.random.set_seed(42)  # TensorFlow random seed

# Assuming X_train, X_val, y_train, y_val are already defined and preprocessed

X_train = np.array(X_train_NN)
X_val = np.array(X_val_NN)
y_train = np.array(y_train)
y_val = np.array(y_val)

# Data is already scaled and preprocessed

# Define the model-building function for hyperparameter tuning
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))

    # Add hidden layers with hyperparameters
    for i in range(hp.Int("num_layers", 1, 3)):  # Tune the number of layers (1 to 3)
        model.add(
            Dense(
                units=hp.Int(f"units_{i}", min_value=8, max_value=128, step=8), 
                activation=hp.Choice("activation", ["relu", "tanh"]), 
                kernel_regularizer=tf.keras.regularizers.l2(
                    hp.Float("l2", min_value=0.0, max_value=0.1, step=0.01)
                )
            )
        )
        if hp.Boolean("dropout"):
            model.add(Dropout(rate=hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)))

    # Output layer for regression
    model.add(Dense(units=1, activation=None))

    # Compile the model with hyperparameters
    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
        ),
        loss="mean_squared_error",
        metrics=["mae"]
    )
    return model

# Set up the Keras Tuner
tuner = RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=20,  # Number of hyperparameter combinations to try
    executions_per_trial=1,  # Number of executions per trial
    directory="hyperparameter_tuning",
    project_name="price_nn_tuning"
)

# Early stopping to prevent overfitting
stop_early = EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=1)

# Run the search for the best hyperparameters
tuner.search(
    X_train, y_train, 
    validation_data=(X_val, y_val),
    epochs=100, 
    batch_size=32,
    callbacks=[stop_early],
    verbose=1
)

# Retrieve the best hyperparameters and build the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best Hyperparameters:")
for param, value in best_hps.values.items():
    print(f"{param}: {value}")

# Train the model with the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)

history = best_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[stop_early],
    verbose=1
)

# Evaluate the model
def evaluate_model(model, X, y, name):
    predictions = model.predict(X, batch_size=X.shape[0])
    predictions = np.expm1(predictions)
    y_actual = np.expm1(y)
    rmse = root_mean_squared_error(y_actual, predictions)
    print(f"{name} Performance:")
    print(f"Root Mean Squared Error: {rmse:.4f}")

evaluate_model(best_model, X_val, y_val, "Tuned Neural Network Model")








# from tensorflow.keras import Sequential, Input
# from tensorflow.keras.layers import Dense
# from tensorflow.keras import optimizers
# from tensorflow.keras.callbacks import EarlyStopping
# import numpy as np
# from sklearn.metrics import root_mean_squared_error
# import tensorflow as tf
# import random

# # Set the random seed for reproducibility
# random.seed(42)  # Python random seed
# np.random.seed(42)  # NumPy random seed
# tf.random.set_seed(42)  # TensorFlow random seed


# X_train = np.array(X_train_NN)
# X_val = np.array(X_val_NN)
# y_train = np.array(y_train)
# y_val = np.array(y_val)

# # Data is already scaled and preprocessed

# NEpochs = 10000
# BatchSize = 32
# Optimizer = optimizers.RMSprop(learning_rate=0.001)

# # Building the regression neural network
# priceNN = Sequential()

# priceNN.add(Input(shape=(X_train.shape[1],)))
# priceNN.add(Dense(units=16, activation="relu", use_bias=True))
# priceNN.add(Dense(units=8, activation="relu", use_bias=True))
# priceNN.add(Dense(units=1, activation=None, use_bias=True))  # Output layer for regression

# # Compile the model with regression loss
# priceNN.compile(loss="mean_squared_error", optimizer=Optimizer, metrics=["mae"])  # Metrics will calculate MAE

# # Early stopping to prevent overfitting
# StopRule = EarlyStopping(monitor="val_loss", mode="min", verbose=0, patience=10, min_delta=0.0)

# # Train the model
# priceNN.fit(
#     X_train, y_train, 
#     validation_data=(X_val, y_val), 
#     epochs=NEpochs, verbose=0, 
#     batch_size=BatchSize, callbacks=[StopRule]
# )


# def evaluate_model(model, X, y, name):
#     predictions = model.predict(X, batch_size=X.shape[0])
#     predictions = np.expm1(predictions)
#     y_actual = np.expm1(y)
#     rmse = root_mean_squared_error(y_actual, predictions)
#     print(f"{name} Performance:")
#     print(f"Root Mean Squared Error: {rmse:.4f}")
    
# evaluate_model(priceNN, X_val, y_val, "Neural Network Model")



























