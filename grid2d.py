import numpy as np
import os
import matplotlib.pyplot as plt

# --- Load mean curvature per channel ---
data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20100705S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6"
curvature = np.load(os.path.join(data_dir, "ric_curvature.npy"))  # (64, epochs)
mean_curv = np.mean(curvature, axis=1)

# --- Reshape to 8x8 grid (synthetic) ---
grid = mean_curv.reshape(8, 8)

plt.figure(figsize=(10, 6))
sc = None
for row in range(8):
    for col in range(8):
        sc = plt.scatter(col, row, c=grid[row, col], cmap='coolwarm', s=320,
                         vmin=mean_curv.min(), vmax=mean_curv.max(), edgecolor='k')

plt.gca().invert_yaxis()
plt.xticks(range(8))
plt.yticks(range(8))
plt.title("ECoG Channel-wise Mean RIC Curvature (Food-Tracking, Synthetic 8×8 Grid)")
cbar = plt.colorbar(sc, label='Mean RIC Curvature')  # <--- THIS FIXES IT
plt.axis('off')
plt.tight_layout()
plt.show()
