import numpy as np
import os
import matplotlib.pyplot as plt
import umap

# --- Load data ---
data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20100705S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6"
curvature = np.load(os.path.join(data_dir, "ric_curvature.npy"))  # (64, epochs)
labels = np.load(os.path.join(data_dir, "epoch_labels.npy"))

X = curvature.T  # shape (epochs, 64)
y = labels  # 0 = not moving, 1 = moving

# --- UMAP projection ---
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(X)

# --- Plot UMAP, colored by movement label ---
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embedding[:, 0], embedding[:, 1],
    c=y, cmap='coolwarm', s=18, alpha=0.75,
    label=None
)
cbar = plt.colorbar(scatter, ticks=[0, 1])
cbar.set_ticklabels(['Not Moving', 'Moving'])
cbar.set_label('Behavioral State')
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.title('UMAP Embedding of ECoG Epochs by RIC Curvature (Food-Tracking Task)')
plt.tight_layout()
plt.show()
