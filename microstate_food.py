import numpy as np
import os
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import DBSCAN

# --- Load features and labels ---
data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20100705S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6"
curvature = np.load(os.path.join(data_dir, "ric_curvature.npy"))  # (64, n_epochs)
labels = np.load(os.path.join(data_dir, "epoch_labels.npy"))

X = curvature.T  # (epochs, 64)
y = labels

# --- UMAP projection ---
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(X)

# --- DBSCAN clustering for microstates ---
db = DBSCAN(eps=0.8, min_samples=15).fit(embedding)
clusters = db.labels_  # -1 = noise/outlier

# --- Plot microstate clusters ---
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='tab10', s=18, alpha=0.7)
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.title('Microstate/Attractor Clusters in RIC Space (Food-Tracking)')
plt.colorbar(scatter, label='Microstate/Cluster')
plt.tight_layout()
plt.show()

n_microstates = len(set(clusters)) - (1 if -1 in clusters else 0)
print(f"Identified {n_microstates} microstates (clusters) in RIC space.")
