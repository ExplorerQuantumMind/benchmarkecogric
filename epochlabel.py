import numpy as np
import scipy.io as sio
import os

data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128\Session1"
curvature = np.load(os.path.join(data_dir, "ric_curvature.npy"))  # (128, n_epochs)
n_epochs = curvature.shape[1]
epoch_length_sec = 2
epoch_times = np.arange(n_epochs) * epoch_length_sec  # epoch start times in seconds

# Load condition data
mat = sio.loadmat(os.path.join(data_dir, "Condition.mat"))
ct = mat['ConditionTime'].flatten()
clabels = ['AwakeEyesOpened', 'AwakeEyesClosed']  # Use these two as states

# Define intervals for "Opened" and "Closed"
# 1. Opened: ct[0] <= t < ct[1]
# 2. Closed: ct[2] <= t < ct[3]
labels = np.zeros(n_epochs, dtype=int)  # 0=other/unlabeled, 1=opened, 2=closed

# Label "Opened"
opened_mask = (epoch_times >= ct[0]) & (epoch_times < ct[1])
labels[opened_mask] = 1

# Label "Closed"
closed_mask = (epoch_times >= ct[2]) & (epoch_times < ct[3])
labels[closed_mask] = 2

print(f"Epochs labeled 'Opened': {np.sum(labels==1)}, 'Closed': {np.sum(labels==2)}, Unlabeled: {np.sum(labels==0)}")

# Remove unlabeled epochs for classification
keep = (labels > 0)
X = np.mean(curvature[:, keep], axis=0).reshape(-1, 1)
y = labels[keep]

# Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

clf = LogisticRegression(max_iter=1000)
accuracy = cross_val_score(clf, X, y, cv=10).mean()
print(f"\nRIC-based detection accuracy (Opened vs Closed): {accuracy:.2f} ({accuracy*100:.1f}%)")
