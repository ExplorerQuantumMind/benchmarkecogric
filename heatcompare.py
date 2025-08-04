import numpy as np
from nilearn import datasets, surface, plotting
from scipy.spatial import cKDTree

# --- Load data ---
data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20100705S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6"
curvature = np.load(f"{data_dir}/ric_curvature.npy")
labels = np.load(f"{data_dir}/epoch_labels.npy")

# --- Compute mean RIC for each state ---
mean_notmoving = np.mean(curvature[:, labels == 0], axis=1)
mean_moving = np.mean(curvature[:, labels == 1], axis=1)

fsaverage = datasets.fetch_surf_fsaverage()
surf_coords, faces = surface.load_surf_mesh(fsaverage['pial_left'])

# --- Assign 64 surface vertices as "electrodes" (demo) ---
rng = np.random.default_rng(42)
electrode_indices = rng.choice(surf_coords.shape[0], size=64, replace=False)
electrode_coords = surf_coords[electrode_indices]
tree = cKDTree(electrode_coords)
_, nearest_idx = tree.query(surf_coords, k=1)

# --- Interpolate RIC values to surface ---
surf_values_rest = mean_notmoving[nearest_idx]
surf_values_move = mean_moving[nearest_idx]

# --- Use same color scale for both maps ---
vmin = min(np.min(surf_values_rest), np.min(surf_values_move))
vmax = max(np.max(surf_values_rest), np.max(surf_values_move))

angles = [
    ('lateral', 'lateral', None),
    ('dorsal', 'dorsal', None),
    ('medial', 'medial', None),
    ('posterior', 'posterior', None)
]

for label, view_type, _ in angles:
    # Not Moving
    plotting.plot_surf_stat_map(
        fsaverage['pial_left'],
        stat_map=surf_values_rest,
        hemi='left',
        view=view_type,
        bg_map=fsaverage['sulc_left'],
        cmap='coolwarm',
        colorbar=True,
        threshold=None,
        title=f"RIC Curvature — {label.capitalize()} (Not Moving)",
        symmetric_cbar=False,
        darkness=0.4,
        vmin=vmin,
        vmax=vmax,
        output_file=f"ric_surface_notmoving_{label}.png"
    )
    # Moving
    plotting.plot_surf_stat_map(
        fsaverage['pial_left'],
        stat_map=surf_values_move,
        hemi='left',
        view=view_type,
        bg_map=fsaverage['sulc_left'],
        cmap='coolwarm',
        colorbar=True,
        threshold=None,
        title=f"RIC Curvature — {label.capitalize()} (Moving)",
        symmetric_cbar=False,
        darkness=0.4,
        vmin=vmin,
        vmax=vmax,
        output_file=f"ric_surface_moving_{label}.png"
    )

print("Saved surface heatmaps for both states and all angles as ric_surface_[state]_[view].png")
