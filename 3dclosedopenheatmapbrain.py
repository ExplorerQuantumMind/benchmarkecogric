import numpy as np
from nilearn import datasets, surface, plotting
from scipy.spatial import cKDTree

# --- Load RIC features and labels ---
data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128\Session1"
curvature = np.load(f"{data_dir}/ric_curvature.npy")   # shape (128, n_epochs)

# ---- Assign per-epoch labels for "opened" (1) and "closed" (2) ---
# Use your previous logic or load from saved labels
epoch_length_sec = 2
n_epochs = curvature.shape[1]
epoch_times = np.arange(n_epochs) * epoch_length_sec

import scipy.io as sio
mat = sio.loadmat(f"{data_dir}/Condition.mat")
ct = mat['ConditionTime'].flatten()
labels = np.zeros(n_epochs, dtype=int)
labels[(epoch_times >= ct[0]) & (epoch_times < ct[1])] = 1  # Opened
labels[(epoch_times >= ct[2]) & (epoch_times < ct[3])] = 2  # Closed

# --- Compute mean RIC curvature for each state ---
mean_opened = np.mean(curvature[:, labels == 1], axis=1)
mean_closed = np.mean(curvature[:, labels == 2], axis=1)

# --- Load standard surface mesh ---
fsaverage = datasets.fetch_surf_fsaverage()
surf_coords, faces = surface.load_surf_mesh(fsaverage['pial_left'])

# --- Synthetic 128 "electrodes" on mesh (consistent layout) ---
rng = np.random.default_rng(42)
electrode_indices = rng.choice(surf_coords.shape[0], size=128, replace=False)
electrode_coords = surf_coords[electrode_indices]
tree = cKDTree(electrode_coords)
_, nearest_idx = tree.query(surf_coords, k=1)

# --- Interpolate values to surface ---
surf_values_opened = mean_opened[nearest_idx]
surf_values_closed = mean_closed[nearest_idx]

# --- Use same color scale for both ---
vmin = min(np.min(surf_values_opened), np.min(surf_values_closed))
vmax = max(np.max(surf_values_opened), np.max(surf_values_closed))

angles = [
    ('lateral', 'lateral', None),
    ('dorsal', 'dorsal', None),
    ('medial', 'medial', None),
    ('posterior', 'posterior', None)
]

for label, view_type, _ in angles:
    # Eyes Opened
    plotting.plot_surf_stat_map(
        fsaverage['pial_left'],
        stat_map=surf_values_opened,
        hemi='left',
        view=view_type,
        bg_map=fsaverage['sulc_left'],
        cmap='coolwarm',
        colorbar=True,
        threshold=None,
        title=f"RIC Curvature — {label.capitalize()} (Eyes Opened)",
        symmetric_cbar=False,
        darkness=0.4,
        vmin=vmin,
        vmax=vmax,
        output_file=f"ric_surface_eyesopened_{label}.png"
    )
    # Eyes Closed
    plotting.plot_surf_stat_map(
        fsaverage['pial_left'],
        stat_map=surf_values_closed,
        hemi='left',
        view=view_type,
        bg_map=fsaverage['sulc_left'],
        cmap='coolwarm',
        colorbar=True,
        threshold=None,
        title=f"RIC Curvature — {label.capitalize()} (Eyes Closed)",
        symmetric_cbar=False,
        darkness=0.4,
        vmin=vmin,
        vmax=vmax,
        output_file=f"ric_surface_eyesclosed_{label}.png"
    )

print("Saved surface heatmaps for both states and all angles as ric_surface_eyesopened_[view].png and ric_surface_eyesclosed_[view].png")
