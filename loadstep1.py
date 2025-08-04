import numpy as np
import scipy.io as sio
import scipy.signal as signal
import os

data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20100705S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6"
n_channels = 64
fs = 1000

all_channels = []
for ch in range(1, n_channels + 1):
    file_path = os.path.join(data_dir, f"ECoG_ch{ch}.mat")
    mat = sio.loadmat(file_path)
    # Use the key 'ECoGData_chN'
    arr_keys = [k for k in mat if not k.startswith("__")]
    # Guess the main variable name
    data_key = arr_keys[0]
    data = mat[data_key].squeeze()
    # Bandpass filter 1–100 Hz
    sos = signal.butter(4, [1, 100], btype='bandpass', fs=fs, output='sos')
    filtered = signal.sosfiltfilt(sos, data)
    # Z-score normalization
    normed = (filtered - np.mean(filtered)) / np.std(filtered)
    all_channels.append(normed)

ecog_data = np.vstack(all_channels)
np.save(os.path.join(data_dir, "ecog_preprocessed.npy"), ecog_data)
print(f"Loaded and preprocessed ECoG data: {ecog_data.shape}")
