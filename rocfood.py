import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

# --- Load data ---
data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20100705S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6"
curvature = np.load(os.path.join(data_dir, "ric_curvature.npy"))  # (64, epochs)
labels = np.load(os.path.join(data_dir, "epoch_labels.npy"))
ecog_data = np.load(os.path.join(data_dir, "ecog_preprocessed.npy"))  # (64, timepoints)

n_channels, n_epochs = curvature.shape
epoch_length_sec = 2
fs = 1000
samples_per_epoch = epoch_length_sec * fs

# --- Amplitude features ---
amp_feats = np.zeros_like(curvature)
for ch in range(n_channels):
    for ep in range(n_epochs):
        seg = ecog_data[ch, ep * samples_per_epoch : (ep + 1) * samples_per_epoch]
        amp_feats[ch, ep] = np.mean(np.abs(seg))

# --- Stack features [RIC | amplitude] ---
X_stack = np.concatenate([curvature, amp_feats], axis=0).T
y = labels

# --- Cross-validated ROC ---
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_true, y_score = [], []

for train, test in cv.split(X_stack, y):
    clf.fit(X_stack[train], y[train])
    y_true.extend(y[test])
    y_score.extend(clf.predict_proba(X_stack[test])[:, 1])  # Prob for class "moving"

fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkred', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Stacked RIC+Amplitude RandomForest (Moving vs Not Moving)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
