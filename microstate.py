import numpy as np
import os
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import DBSCAN
import scipy.io as sio

# ==== Load curvature features ====
data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128\Session1"
curvature = np.load(os.path.join(data_dir, "ric_curvature.npy"))  # (128, n_epochs)
n_epochs = curvature.shape[1]
epoch_length_sec = 2
epoch_times = np.arange(n_epochs) * epoch_length_sec

# ==== Assign per-epoch labels (as before) ====
mat = sio.loadmat(os.path.join(data_dir, "Condition.mat"))
ct = mat['ConditionTime'].flatten()
labels = np.zeros(n_epochs, dtype=int)
labels[(epoch_times >= ct[0]) & (epoch_times < ct[1])] = 1  # Opened
labels[(epoch_times >= ct[2]) & (epoch_times < ct[3])] = 2  # Closed

keep = (labels > 0)
X = curvature[:, keep].T  # (epochs, 128)
y = labels[keep]

# ==== UMAP to 2D ====
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(X)

# ==== DBSCAN clustering ====
db = DBSCAN(eps=0.9, min_samples=20).fit(embedding)
clusters = db.labels_  # -1 = noise/outlier

# ==== Plot microstates ====
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='tab10', s=12, alpha=0.7)
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.title('Microstate/Attractor Clusters in RIC Space')
plt.colorbar(scatter, label='Microstate/Cluster')
plt.tight_layout()
plt.show()

# ==== Print summary ====
n_microstates = len(set(clusters)) - (1 if -1 in clusters else 0)
print(f"Identified {n_microstates} microstates (clusters) in RIC space.")
