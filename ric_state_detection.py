import numpy as np
import scipy.io as sio
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import os

# ==== Load RIC features ====
data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128\Session1"
curvature = np.load(os.path.join(data_dir, "ric_curvature.npy"))  # (128, epochs)

# ==== Load Condition.mat and auto-detect label variable ====
matfile = os.path.join(data_dir, "Condition.mat")
mat = sio.loadmat(matfile)

# Print available variables (for transparency)
print("Variables in Condition.mat:", [k for k in mat if not k.startswith("__")])

# Try to find the variable that contains the label vector (assume 1D or (N,1) or (1,N))
label_var = None
for k in mat:
    if not k.startswith("__"):
        arr = mat[k]
        if isinstance(arr, np.ndarray) and arr.size > 1 and (arr.ndim == 1 or arr.ndim == 2):
            label_var = k
            print(f"Found label variable: {k}, shape: {arr.shape}")
            break

if label_var is None:
    raise ValueError("No suitable label variable found in Condition.mat!")

# Flatten labels to 1D array, ensure length matches number of epochs
y = np.array(mat[label_var]).squeeze()
if y.ndim > 1:
    y = y.flatten()
n_epochs = curvature.shape[1]
if len(y) > n_epochs:
    y = y[:n_epochs]
elif len(y) < n_epochs:
    raise ValueError("Label vector is shorter than number of epochs!")

print(f"Label vector shape: {y.shape}, unique labels: {np.unique(y)}")

# ==== Feature extraction: mean RIC curvature across channels for each epoch ====
X = np.mean(curvature, axis=0).reshape(-1, 1)  # shape (epochs, 1)

# ==== Classification using Logistic Regression ====
clf = LogisticRegression(max_iter=1000)
accuracy = cross_val_score(clf, X, y, cv=10).mean()
print(f"\nRIC-based state detection accuracy: {accuracy:.2f} ({accuracy*100:.1f}%)\n")

# ==== (Optional) Compare to classic feature: mean signal amplitude ====
# Uncomment if you want to add classic comparison
# ecog_data = np.load(os.path.join(data_dir, "ecog_preprocessed.npy"))  # shape (128, n_timepoints)
# amp_per_epoch = np.mean(np.abs(ecog_data), axis=0).reshape(-1, 1)
# classic_accuracy = cross_val_score(clf, amp_per_epoch, y, cv=10).mean()
# print(f"Classic amplitude-based accuracy: {classic_accuracy:.2f} ({classic_accuracy*100:.1f}%)")
