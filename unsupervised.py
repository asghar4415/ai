#customer segmentation:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns

# Load data from CSV
df = pd.read_csv('Mall_Customers.csv')

# Display first few rows
print("Data Preview:")
print(df.head())

# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()

# Apply K-Means with optimal clusters (assuming 5 based on elbow)
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add clusters to dataframe
df['Cluster'] = clusters

# Visualize clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster', palette='viridis', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='red', marker='X', label='Centroids')
plt.title('Customer Segmentation')
plt.legend()
plt.grid()
plt.show()

# Evaluate clustering
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"\nSilhouette Score: {silhouette_avg:.2f}")

# Cluster analysis
print("\nCluster Profiles:")
cluster_summary = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Count'})
print(cluster_summary.round(2))

# Save results
df.to_csv('segmented_customers.csv', index=False)
print("\nResults saved to 'segmented_customers.csv'")




















#vehicle fleet optimization:
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Load data from CSV
df = pd.read_csv('Fleet_Data.csv')

# Display data info
print("Data Information:")
print(df.info())

# Preprocessing: Encode categorical variable
le = LabelEncoder()
df['vehicle_type_encoded'] = le.fit_transform(df['vehicle_type'])

# Select features (excluding ID and original categorical)
features = ['mileage', 'fuel_efficiency', 'maintenance_cost', 'vehicle_type_encoded']
X = df[features]

# Feature scaling (except the encoded categorical)
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[features[:-1]] = scaler.fit_transform(X[features[:-1]])

# Determine optimal clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()

# Apply K-Means (assuming 3 clusters)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualize clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=100)
plt.title('Vehicle Fleet Clustering (PCA Reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster')
plt.grid()
plt.show()

# Cluster analysis
print("\nVehicle Cluster Profiles:")
for cluster_num in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster_num]
    print(f"\nCluster {cluster_num}:")
    print(f"Number of vehicles: {len(cluster_data)}")
    print(f"Avg mileage: {cluster_data['mileage'].mean():,.0f} miles")
    print(f"Avg fuel efficiency: {cluster_data['fuel_efficiency'].mean():.1f} mpg")
    print(f"Avg maintenance cost: ${cluster_data['maintenance_cost'].mean():,.0f}")
    print("Vehicle types:", cluster_data['vehicle_type'].value_counts().to_dict())

# Save results
df.to_csv('optimized_fleet.csv', index=False)
print("\nResults saved to 'optimized_fleet.csv'")





















#studentperformance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# Load data from CSV
df = pd.read_csv('Student_Performance.csv')

# Display summary statistics
print("Data Summary:")
print(df.describe())

# Select features
features = ['GPA', 'study_hours', 'attendance_rate']
X = df[features]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal clusters
wcss = []
silhouette_scores = []
for i in range(2, 7):  # Testing 2-6 clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, clusters))

# Plot evaluation metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 7), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')

plt.subplot(1, 2, 2)
plt.plot(range(2, 7), silhouette_scores, marker='o')
plt.title('Silhouette Scores')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.tight_layout()
plt.show()

# Apply K-Means with optimal clusters (assuming 4 based on plots)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# 3D Visualization
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

colors = ['red', 'blue', 'green', 'purple']
for i in range(optimal_clusters):
    ax.scatter(df[df['Cluster'] == i]['GPA'],
               df[df['Cluster'] == i]['study_hours'],
               df[df['Cluster'] == i]['attendance_rate'],
               c=colors[i], s=60, label=f'Cluster {i+1}')

ax.set_title('Student Performance Clustering')
ax.set_xlabel('GPA')
ax.set_ylabel('Study Hours')
ax.set_zlabel('Attendance Rate (%)')
ax.legend()
plt.show()

# 2D Visualization (GPA vs Study Hours)
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='GPA', y='study_hours', hue='Cluster',
                palette='viridis', s=100)
plt.title('Student Clusters by GPA and Study Hours')
plt.grid()
plt.show()

# Cluster analysis
print("\nStudent Cluster Profiles:")
cluster_profiles = df.groupby('Cluster').agg({
    'GPA': ['mean', 'std'],
    'study_hours': ['mean', 'std'],
    'attendance_rate': ['mean', 'std'],
    'student_id': 'count'
}).round(2)
print(cluster_profiles)

# Save results
df.to_csv('grouped_students.csv', index=False)
print("\nResults saved to 'grouped_students.csv'")