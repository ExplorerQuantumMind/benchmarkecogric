from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import os

data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20100705S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6"
curvature = np.load(os.path.join(data_dir, "ric_curvature.npy"))
labels = np.load(os.path.join(data_dir, "epoch_labels.npy"))

# Remove epochs with ambiguous or missing labels (shouldn't be any here)
keep = np.ones_like(labels, dtype=bool)
X = curvature[:, keep].T  # shape (epochs, channels)
y = labels[keep]

clf = LogisticRegression(max_iter=1000)
accuracy = cross_val_score(clf, X, y, cv=10).mean()
print(f"\nMultichannel RIC decoding accuracy (Moving vs Not Moving): {accuracy:.2f} ({accuracy*100:.1f}%)")
