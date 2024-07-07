import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
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
    "../Desktop/harth/S029.csv",
]


all_data = pd.concat([pd.read_csv(file) for file in file_paths])
all_data['timestamp'] = pd.to_datetime(all_data['timestamp'])

#data for the last two hours
data_last_two_hours = all_data[all_data['timestamp'] >= (all_data['timestamp'].max() - pd.Timedelta(hours=2))]
data_last_two_hours = data_last_two_hours.drop(columns=['index', 'Unnamed: 0', 'timestamp'])
data_last_two_hours = data_last_two_hours.sample(frac=0.005)
labels = data_last_two_hours['label']

# Drop the 'label' column for clustering
data_last_two_hours_without_labels = data_last_two_hours.drop(columns=['label'])
 # Compute mean and standard deviation from the scaled data
mean = data_last_two_hours_without_labels.mean()
std = data_last_two_hours_without_labels.std()
 # Define a function to remove outliers using the Z-score method
def remove_outliers_zscore(data, mean, std, threshold=4):
     z_scores = (data - mean) / std
     return data[(np.abs(z_scores) < threshold).all(axis=1)]

 # Remove outliers using z score
data_no_outliers = remove_outliers_zscore(data_last_two_hours_without_labels, mean, std)


#Scaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_no_outliers)

#Silhouette method
def find_best_k(data, k_range):
    silhouette_scores = []
    
    for k in k_range:
        clustering = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
        labels = clustering.fit_predict(data)
        
        silhouette_avg = silhouette_score(data, labels)
        silhouette_scores.append(silhouette_avg)
        print(f"Silhouette Score for {k} clusters: {silhouette_avg}")
    
    return k_range, silhouette_scores

#range of k values to try
k_range = range(2, 15)

# Find the best k
k_values, silhouette_scores = find_best_k(data_no_outliers, k_range)

# Plot the silhouette scores 
plt.figure(figsize=(8, 4))
plt.plot(k_values, silhouette_scores)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method For Optimal k')
plt.show()

# Determine the optimal number of clusters using the silhouette method
optimal_k = k_values[np.argmax(silhouette_scores)]
print("Optimal number of clusters based on silhouette score:", optimal_k)

#clustering with the optimal k
final_clustering = AgglomerativeClustering(n_clusters=optimal_k, metric='cosine', linkage='average')
final_labels = final_clustering.fit_predict(data_no_outliers)

# Calculate final Silhouette Score and Calinski-Harabasz
final_silhouette_score = silhouette_score(data_no_outliers, final_labels)
final_calinski_harabasz_score = calinski_harabasz_score(data_no_outliers, final_labels)

print("Final Silhouette Score:", final_silhouette_score)
print("Final Calinski-Harabasz Index:", final_calinski_harabasz_score)

# Reduce dimensionality to 2D using PCA to plot the clusters
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data_no_outliers)
plt.figure(figsize=(8, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=final_labels, cmap='viridis', alpha=0.5)
plt.xlabel('X axis')
plt.ylabel('Y axis ')
plt.title('Clusters in 2D Space')
plt.colorbar(label='Cluster Label')
plt.show()

