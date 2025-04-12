import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from keras_tuner.tuners import RandomSearch, BayesianOptimization
import tensorflow as tf
import random

X_train = pd.read_csv("data/model_data/X_train.csv")
X_val = pd.read_csv("data/model_data/X_val.csv")
test_final = pd.read_csv("data/model_data/test_final.csv")
y_train = pd.read_csv("data/model_data/y_train.csv")
y_val = pd.read_csv("data/model_data/y_val.csv")


X_train = np.array(X_train)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_val = np.array(y_val)

############################# Feed-Forward Neural Network #############################
# Set the random seed for reproducibility
random.seed(42)  # Python random seed
np.random.seed(42)  # NumPy random seed
tf.random.set_seed(42)  # TensorFlow random seed


def rmse_transformed(y_true, y_pred):
    # Compute RMSE for transformed values
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


# Define the model-building function for hyperparameter tuning
def build_model(hp):
    model = Sequential()

    # Input layer for regression
    model.add(Input(shape=(X_train.shape[1],)))

    # Add hidden layers with hyperparameters
    for i in range(hp.Int("num_layers", min_value=1, max_value=5, step=1)):  # Tune the number of layers 
        model.add(
            Dense(
                units=hp.Int(f"units_{i}", min_value=8, max_value=256, step=8), # Tune the units per layer
                activation=hp.Choice("activation", values=["relu", "tanh", "elu"]), # Tune activation functions
                kernel_regularizer=tf.keras.regularizers.l2(
                    hp.Float("l2", min_value=0.0, max_value=0.2, step=0.01) # Tune L2 regularization
                )
            )
        )
        if hp.Boolean("dropout"):
            model.add(Dropout(rate=hp.Float("dropout_rate", min_value=0.0, max_value=0.8, step=0.1))) # Tune Dropout Rates

    # Output layer for regression
    model.add(Dense(units=1, activation=None))

    # Compile the model with hyperparameters
    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log") # Tune Learning Rate. According to Andrew Ng lol
        ),
        loss="mean_squared_error",
        metrics=[rmse_transformed]
    )
    return model

# Set up the Keras Tuner
tuner = BayesianOptimization(
    build_model,
    objective="val_loss",
    max_trials=50,  # Number of hyperparameter combinations to try
    executions_per_trial=1,  # Number of executions per trial
    directory="hyperparameter_tuning",
    project_name="price_nn_tuning",
    overwrite=True,
    distribution_strategy=tf.distribute.MirroredStrategy()
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
    rmse = root_mean_squared_error(y, predictions)
    print(f"{name} Performance:")
    print(f"Root Mean Squared Error: {rmse:.4f}")

evaluate_model(best_model, X_val, y_val, "Tuned Neural Network Model")

best_model.save("final_model_NN.h5")
# new_model = tf.keras.models.load_model("final_model_NN.h5")






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



























