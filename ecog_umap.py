import numpy as np
import matplotlib.pyplot as plt
import umap

# Directory containing your RIC curvature matrix
data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128\Session1"
curvature = np.load(f"{data_dir}/ric_curvature.npy")  # shape (128, 985)

# UMAP expects (samples, features), so we transpose to (epochs, channels)
X = curvature.T  # shape (985, 128)

# Run UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(X)  # (985, 2)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], s=8, alpha=0.7, c=np.mean(X, axis=1), cmap='coolwarm')
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.title('UMAP Embedding of ECoG Epochs by RIC Curvature')
plt.colorbar(label='Mean Curvature per Epoch')
plt.tight_layout()
plt.show()
