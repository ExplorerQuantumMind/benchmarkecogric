import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
import os

# ==== User variables ====
data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128\Session1"
epoch_length_sec = 2  # seconds per epoch
fs = 1000  # sampling rate
n_bins = 6  # number of quantile bins for symbolic coding
alpha, beta = 1.0, 1.0  # coupling params, change if you want

# ==== Load data ====
ecog_data = np.load(os.path.join(data_dir, "ecog_preprocessed.npy"))
n_channels, n_timepoints = ecog_data.shape
samples_per_epoch = epoch_length_sec * fs
n_epochs = n_timepoints // samples_per_epoch

# ==== Helper: symbolic discretization (quantile binning) ====
def symbolic_discretize(x, n_bins):
    return np.digitize(x, np.quantile(x, np.linspace(0, 1, n_bins+1)[1:-1]))

# ==== Preallocate outputs ====
curvature_mat = np.zeros((n_channels, n_epochs))
entropy_mat = np.zeros((n_channels, n_epochs))
rec_gain_mat = np.zeros((n_channels, n_epochs))

for ch in range(n_channels):
    print(f"Processing channel {ch+1}/{n_channels}")
    x = ecog_data[ch]
    for ep in range(n_epochs):
        segment = x[ep * samples_per_epoch : (ep + 1) * samples_per_epoch]
        # Symbolic discretization
        symbols = symbolic_discretize(segment, n_bins)
        # Symbolic entropy (Shannon entropy)
        counts = np.bincount(symbols, minlength=n_bins)
        prob = counts / counts.sum()
        H = entropy(prob, base=np.e)
        # Recursive gain (lag-1 mutual info or autocorr)
        rec_gain = mutual_info_score(symbols[:-1], symbols[1:])
        # Or, for speed: np.corrcoef(symbols[:-1], symbols[1:])[0,1]
        # dH/dt (simple: entropy diff between this and previous epoch)
        if ep == 0:
            dH = 0
        else:
            dH = H - entropy_mat[ch, ep - 1]
        # RIC curvature
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
