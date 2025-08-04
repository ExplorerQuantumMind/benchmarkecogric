import numpy as np
import matplotlib.pyplot as plt
import umap

data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128\Session1"
curvature = np.load(f"{data_dir}/ric_curvature.npy")

# Shape: (channels, epochs) -> (epochs, channels) for UMAP
X = curvature.T

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(X)  # shape (985, 2)

plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], s=7, alpha=0.7, c=np.mean(X, axis=1), cmap='coolwarm')
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.title('UMAP Embedding of ECoG Epochs by RIC Curvature')
plt.colorbar(label='Mean Curvature')
plt.tight_layout()
plt.show()
