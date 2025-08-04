import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
import os

# ==== User variables ====
data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20100705S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6"
epoch_length_sec = 2  # seconds per epoch
fs = 1000  # sampling rate
n_bins = 6  # number of quantile bins for symbolic coding
alpha, beta = 1.0, 1.0

# ==== Load data ====
ecog_data = np.load(os.path.join(data_dir, "ecog_preprocessed.npy"))
n_channels, n_timepoints = ecog_data.shape

labels = np.load(os.path.join(data_dir, "epoch_labels.npy"))
min_epochs = len(labels)
samples_per_epoch = epoch_length_sec * fs

# ==== Helper: symbolic discretization ====
def symbolic_discretize(x, n_bins):
    return np.digitize(x, np.quantile(x, np.linspace(0, 1, n_bins+1)[1:-1]))

# ==== Preallocate outputs ====
curvature_mat = np.zeros((n_channels, min_epochs))
entropy_mat = np.zeros((n_channels, min_epochs))
rec_gain_mat = np.zeros((n_channels, min_epochs))

for ch in range(n_channels):
    print(f"Processing channel {ch+1}/{n_channels}")
    x = ecog_data[ch]
    for ep in range(min_epochs):
        segment = x[ep * samples_per_epoch : (ep + 1) * samples_per_epoch]
        symbols = symbolic_discretize(segment, n_bins)
        counts = np.bincount(symbols, minlength=n_bins)
        prob = counts / counts.sum()
        H = entropy(prob, base=np.e)
        rec_gain = mutual_info_score(symbols[:-1], symbols[1:])
        if ep == 0:
            dH = 0
        else:
            dH = H - entropy_mat[ch, ep - 1]
        K = alpha * rec_gain - beta * dH
        curvature_mat[ch, ep] = K
        entropy_mat[ch, ep] = H
        rec_gain_mat[ch, ep] = rec_gain

# ==== Save outputs ====
np.save(os.path.join(data_dir, "ric_curvature.npy"), curvature_mat)
np.save(os.path.join(data_dir, "ric_entropy.npy"), entropy_mat)
np.save(os.path.join(data_dir, "ric_rec_gain.npy"), rec_gain_mat)
print("RIC feature extraction complete!")
print(f"curvature_mat shape: {curvature_mat.shape}")
