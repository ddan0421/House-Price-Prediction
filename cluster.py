import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Sample dataset with missing values
data = pd.DataFrame({
    'Feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
    'Feature2': [10.0, 12.0, 13.0, np.nan, 15.0],
    'Feature3': [100.0, np.nan, 130.0, 140.0, 150.0]
})

# Step 1: Preprocessing
# Impute initial missing values (e.g., using the mean) for clustering
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Standardize the data (important for K-Means)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Step 2: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Step 3: Impute Missing Values Based on Clusters
for col in data.columns[:-1]:  # Exclude 'Cluster' column
    for cluster in data['Cluster'].unique():
        # Calculate the mean of the column within the cluster
        cluster_mean = data[data['Cluster'] == cluster][col].mean()
        # Impute missing values in the column for rows in the cluster
        data.loc[(data['Cluster'] == cluster) & (data[col].isnull()), col] = cluster_mean

print(data)
