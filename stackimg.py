import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# --- Load data ---
data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20100705S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6"
curvature = np.load(os.path.join(data_dir, "ric_curvature.npy"))  # (64, epochs)
labels = np.load(os.path.join(data_dir, "epoch_labels.npy"))
ecog_data = np.load(os.path.join(data_dir, "ecog_preprocessed.npy"))  # (64, timepoints)

n_channels, n_epochs = curvature.shape
epoch_length_sec = 2
fs = 1000
samples_per_epoch = epoch_length_sec * fs

# --- Compute mean amplitude per channel/epoch ---
amp_feats = np.zeros_like(curvature)
for ch in range(n_channels):
    for ep in range(n_epochs):
        seg = ecog_data[ch, ep * samples_per_epoch : (ep + 1) * samples_per_epoch]
        amp_feats[ch, ep] = np.mean(np.abs(seg))

# --- Feature stacking: [RIC | amplitude] ---
X_stack = np.concatenate([curvature, amp_feats], axis=0).T  # (epochs, 128)
y = labels

clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
accuracy = cross_val_score(clf, X_stack, y, cv=10).mean()
print(f"\nStacked RIC+Amplitude RandomForest accuracy: {accuracy:.2f} ({accuracy*100:.1f}%)")
