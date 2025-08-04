from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import os

data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20100705S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6"
curvature = np.load(os.path.join(data_dir, "ric_curvature.npy"))
labels = np.load(os.path.join(data_dir, "epoch_labels.npy"))

X = curvature.T  # shape (epochs, 64)
y = labels

clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
accuracy = cross_val_score(clf, X, y, cv=10).mean()
print(f"\nRandomForest RIC decoding accuracy (Moving vs Not Moving): {accuracy:.2f} ({accuracy*100:.1f}%)")
