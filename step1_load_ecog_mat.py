import numpy as np
import scipy.signal as signal
import scipy.io as sio
import os

# ==== User variables ====
data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128\Session1"
n_channels = 128
fs = 1000  # Hz (update if needed!)

all_channels = []

for ch in range(1, n_channels + 1):
    file_path = os.path.join(data_dir, f"ECoG_ch{ch}.mat")
    print(f"Loading {file_path} ...")
    mat = sio.loadmat(file_path)
    # Print all keys in the .mat file (to help you identify the signal variable)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    print(f"Available variables in {file_path}: {keys}")
    # CHANGE THIS if your variable name is not 'data'!
    if 'data' in keys:
        data = mat['data'].squeeze()
    else:
        # Try the first non-metadata variable
        data = mat[keys[0]].squeeze()
    # Bandpass filter (1–100 Hz)
    sos = signal.butter(4, [1, 100], btype='bandpass', fs=fs, output='sos')
    filtered = signal.sosfiltfilt(sos, data)
    # Z-score normalization
    normed = (filtered - np.mean(filtered)) / np.std(filtered)
    all_channels.append(normed)

ecog_data = np.vstack(all_channels)
np.save(os.path.join(data_dir, "ecog_preprocessed.npy"), ecog_data)
print(f"Loaded and preprocessed ECoG data: {ecog_data.shape}")
