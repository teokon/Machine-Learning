import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


file_paths = [
    "../Desktop/harth/S006.csv",
    "../Desktop/harth/S008.csv",
     "../Desktop/harth/S009.csv",
    "../Desktop/harth/S010.csv",
    "../Desktop/harth/S012.csv", 
    "../Desktop/harth/S013.csv",
    "../Desktop/harth/S014.csv", 
    "../Desktop/harth/S015.csv", 
    "../Desktop/harth/S016.csv",
    "../Desktop/harth/S017.csv", 
    "../Desktop/harth/S018.csv", 
    "../Desktop/harth/S019.csv",
    "../Desktop/harth/S020.csv", 
    "../Desktop/harth/S021.csv", 
    "../Desktop/harth/S022.csv",
    "../Desktop/harth/S023.csv", 
    "../Desktop/harth/S024.csv", 
    "../Desktop/harth/S025.csv",
    "../Desktop/harth/S026.csv", 
    "../Desktop/harth/S027.csv", 
    "../Desktop/harth/S028.csv",
    "../Desktop/harth/S029.csv"
]


all_data = pd.concat([pd.read_csv(file) for file in file_paths])
all_data['timestamp'] = pd.to_datetime(all_data['timestamp'])
data_last_two_hours = all_data[all_data['timestamp'] >= (all_data['timestamp'].max() - pd.Timedelta(hours=2))]
data_last_two_hours = data_last_two_hours.drop(columns=['index', 'Unnamed: 0', 'timestamp'])
data_last_two_hours = data_last_two_hours.sample(frac=0.02)
data_last_two_hours_without_labels = data_last_two_hours.drop(columns=['label'])

# mean and standard deviation 
mean = data_last_two_hours_without_labels.mean()
std = data_last_two_hours_without_labels.std()
# remove outliers using the Z-score method
def remove_outliers_zscore(data, mean, std, threshold=3):
    z_scores = (data - mean) / std
    return data[(np.abs(z_scores) < threshold).all(axis=1)]

# Remove outliers
data_no_outliers = remove_outliers_zscore(data_last_two_hours_without_labels, mean, std)

#silhouette 
k_values = range(2, 15)
silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
    kmeans.fit(data_no_outliers)
    silhouette_avg = silhouette_score(data_no_outliers, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

# Plot the silhouette scores
plt.figure(figsize=(8, 4))
plt.plot(k_values, silhouette_scores)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method For Optimal k')
plt.show()

# optimal k 
optimal_k = k_values[np.argmax(silhouette_scores)]
print("Optimal number of clusters based on silhouette score:", optimal_k)

# K-means with optimal k
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=0)
kmeans.fit(data_no_outliers)

# Get the cluster labels
final_labels = kmeans.labels_
# Compute and print the silhouette score g
silhouette_avg = silhouette_score(data_no_outliers, final_labels)
print("Silhouette Score for k =", optimal_k, "is:", silhouette_avg)
final_calinski_harabasz_score = calinski_harabasz_score(data_no_outliers, final_labels)
print("Final Calinski-Harabasz Index:", final_calinski_harabasz_score)

# Reduce dimensionality to 2D using PCA to plot the clusters
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data_no_outliers)
plt.figure(figsize=(8, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=final_labels, cmap='viridis', alpha=0.5)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Clusters in 2D Space')
plt.colorbar(label='Cluster Label')
plt.show()

