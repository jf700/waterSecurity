import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# filepath: c:\Users\ethan\Documents\josh_code\kmeans_clustering.py

# Load the data
file_path = "imerg_basin_monthly_timeseries_anomalies.csv"  # Update with the correct path if needed
data = pd.read_csv(file_path)

# Drop the 'time' column and keep only numerical data
X = data.drop(columns=['time'])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA for dimensionality reduction (optional, but useful for visualization)
pca = PCA(n_components=2)  # Reduce to 2 dimensions for clustering
X_pca = pca.fit_transform(X_scaled)

# Try different values of k and select the optimal one using silhouette score
silhouette_scores = []
k_values = range(2, 11)  # Test k from 2 to 10

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    silhouette_scores.append(score)
    print(f"Silhouette score for k={k}: {score:.4f}")

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different K in KMeans')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Choose the best k
best_k = k_values[np.argmax(silhouette_scores)]
print(f"Best number of clusters by silhouette score: {best_k}")

# Perform final clustering with the best k
kmeans_final = KMeans(n_clusters=best_k, random_state=42)
cluster_labels = kmeans_final.fit_predict(X_pca)

# Add cluster labels to the original data
data['Cluster'] = cluster_labels

# Save the clustered data to a new CSV file
output_file = "clustered_timeseries_anomalies.csv"
data.to_csv(output_file, index=False)
print(f"Clustered data saved to {output_file}")

# Visualize the clusters
plt.figure(figsize=(10, 7))
for cluster in range(best_k):
    plt.scatter(X_pca[cluster_labels == cluster, 0], X_pca[cluster_labels == cluster, 1], label=f'Cluster {cluster}')
plt.title('KMeans Clustering Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()