import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import scipy.io as sio

# Load features
data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128\Session1"
curvature = np.load(os.path.join(data_dir, "ric_curvature.npy"))  # (128, n_epochs)
n_epochs = curvature.shape[1]
epoch_length_sec = 2
epoch_times = np.arange(n_epochs) * epoch_length_sec

# Load and assign labels
mat = sio.loadmat(os.path.join(data_dir, "Condition.mat"))
ct = mat['ConditionTime'].flatten()
labels = np.zeros(n_epochs, dtype=int)
labels[(epoch_times >= ct[0]) & (epoch_times < ct[1])] = 1  # AwakeEyesOpened
labels[(epoch_times >= ct[2]) & (epoch_times < ct[3])] = 2  # AwakeEyesClosed

# Only keep labeled epochs
keep = (labels > 0)
X = curvature[:, keep].T  # (n_epochs, 128)
y = labels[keep]

print(f"Shape of X: {X.shape}, y: {y.shape}")

clf = LogisticRegression(max_iter=1000)
accuracy = cross_val_score(clf, X, y, cv=10).mean()
print(f"\nMultichannel RIC decoding accuracy (Opened vs Closed): {accuracy:.2f} ({accuracy*100:.1f}%)")
