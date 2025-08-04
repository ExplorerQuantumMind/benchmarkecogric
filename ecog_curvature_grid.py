import numpy as np
import matplotlib.pyplot as plt

# Set your data directory
data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128\Session1"

# Load mean curvature per channel (already computed)
curvature = np.load(f"{data_dir}/ric_curvature.npy")  # shape (128, n_epochs)
mean_curv = np.mean(curvature, axis=1)  # shape (128,)

# Create a synthetic 8x16 grid
n_rows, n_cols = 8, 16
xx, yy = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
coords = np.c_[xx.ravel(), yy.ravel()]  # shape (128, 2)

plt.figure(figsize=(13, 7))
sc = plt.scatter(coords[:, 0], coords[:, 1], c=mean_curv, cmap='coolwarm', s=250, edgecolor='k')
plt.colorbar(sc, label="Mean RIC Curvature")
plt.title("ECoG Channel-wise Mean RIC Curvature (Synthetic 8×16 Grid)", fontsize=15)
plt.gca().invert_yaxis()
plt.axis('off')
plt.tight_layout()
plt.show()
