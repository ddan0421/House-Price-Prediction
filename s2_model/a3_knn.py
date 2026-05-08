import os
import pickle

import duckdb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor

from s1_data.db_utils import load_df, save_df
from s3_validation.model_evaluation import evaluate_linear_model


base_folder = "data"
database = "AmesHousePrice.duckdb"
database_path = os.path.join(base_folder, database)
os.makedirs("models", exist_ok=True)

conn = duckdb.connect(database=database_path, read_only=False)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

############################# KNN #############################
X_train_knn_raw = load_df(conn, "X_train_knn")
X_val_knn_raw = load_df(conn, "X_val_knn")
test_knn_raw = load_df(conn, "test_knn")
y_train_knn = load_df(conn, "y_train")
y_val_knn = load_df(conn, "y_val")

knn = KNeighborsRegressor()

param_grid = {
    "n_neighbors": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
    "algorithm": ["auto", "ball_tree", "kd_tree"],
    "leaf_size": [20, 30, 40]
}
gs_knn = GridSearchCV(estimator=knn,
                      param_grid=param_grid,
                      scoring="neg_root_mean_squared_error",
                      cv=cv,
                      n_jobs=-1,
                      refit=True)

gs_knn.fit(X_train_knn_raw, y_train_knn.values.ravel())

print("KNN 10-Fold CV RMSE:", -gs_knn.best_score_)
print("KNN Optimal Parameter:", gs_knn.best_params_)
print("KNN Optimal Estimator:", gs_knn.best_estimator_)

best_knn_model = gs_knn.best_estimator_
with open("models/final_model_knn.pkl", "wb") as f:
    pickle.dump(best_knn_model, f)
print("KNN model saved to models/final_model_knn.pkl")

save_df(conn, X_train_knn_raw, "X_train_knn_final")
save_df(conn, X_val_knn_raw, "X_val_knn_final")
save_df(conn, test_knn_raw, "test_knn_final")

evaluate_linear_model(best_knn_model, X_val_knn_raw, y_val_knn, "KNN Model")

conn.close()
