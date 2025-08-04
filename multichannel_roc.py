import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import scipy.io as sio

# Load features and labels (same code as before)
data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128\Session1"
curvature = np.load(os.path.join(data_dir, "ric_curvature.npy"))
n_epochs = curvature.shape[1]
epoch_length_sec = 2
epoch_times = np.arange(n_epochs) * epoch_length_sec

mat = sio.loadmat(os.path.join(data_dir, "Condition.mat"))
ct = mat['ConditionTime'].flatten()
labels = np.zeros(n_epochs, dtype=int)
labels[(epoch_times >= ct[0]) & (epoch_times < ct[1])] = 1  # AwakeEyesOpened
labels[(epoch_times >= ct[2]) & (epoch_times < ct[3])] = 2  # AwakeEyesClosed

keep = (labels > 0)
X = curvature[:, keep].T
y = labels[keep]

# For ROC, convert to binary (0 and 1)
y_bin = (y == 2).astype(int)  # "Closed" = 1, "Opened" = 0

clf = LogisticRegression(max_iter=1000)

# Cross-validated prediction
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_true = []
y_score = []

for train, test in cv.split(X, y_bin):
    clf.fit(X[train], y_bin[train])
    y_true.extend(y_bin[test])
    y_score.extend(clf.predict_proba(X[test])[:, 1])  # Probability of "Closed"

fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkred', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Multichannel RIC Decoding (Closed vs Opened)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
