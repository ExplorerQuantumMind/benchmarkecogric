import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, surface

# Load data
data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128\Session1"
curvature = np.load(f"{data_dir}/ric_curvature.npy")
mean_curv = np.mean(curvature, axis=1)

# Load a standard surface mesh (human fsaverage for demo)
fsaverage = datasets.fetch_surf_fsaverage()
coords, faces = surface.load_surf_mesh(fsaverage['pial_left'])

# Select 128 random surface vertices as "electrodes" for demo
rng = np.random.default_rng(42)
electrode_indices = rng.choice(coords.shape[0], size=128, replace=False)
electrode_coords = coords[electrode_indices]

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot surface mesh
ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2], triangles=faces, color='lightgray', alpha=0.45, linewidth=0.1, antialiased=True)

# Overlay electrode points colored by mean_curv
p = ax.scatter(electrode_coords[:, 0], electrode_coords[:, 1], electrode_coords[:, 2], 
               c=mean_curv, cmap='coolwarm', s=80, edgecolor='k', lw=0.6)

# Set view angle similar to neuroimaging figures
ax.view_init(elev=105, azim=110)
ax.set_axis_off()
fig.colorbar(p, shrink=0.7, label='Mean RIC Curvature')
plt.title('3D Brain Projection: Mean RIC Curvature', fontsize=15)
plt.tight_layout()
plt.show()
