import scipy.io as sio
import os
import numpy as np

data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20100705S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6"
fs_motion = 120
epoch_length_sec = 2

mat = sio.loadmat(os.path.join(data_dir, "Motion.mat"))
MotionData = mat['MotionData']

# Use Marker 4 (index 4) for movement
wrist_marker_index = 4
lwri_xyz = MotionData[wrist_marker_index, 0].astype(float)
speed = np.linalg.norm(np.diff(lwri_xyz, axis=0), axis=1) * fs_motion
speed = np.concatenate(([0], speed))

samples_per_epoch_motion = int(epoch_length_sec * fs_motion)
n_epochs_motion = len(speed) // samples_per_epoch_motion

labels = np.zeros(n_epochs_motion, dtype=int)
threshold = np.percentile(speed, 70)  # "moving" = top 30% speeds; adjust as needed

for ep in range(n_epochs_motion):
    ep_speed = speed[ep * samples_per_epoch_motion : (ep + 1) * samples_per_epoch_motion]
    labels[ep] = int(np.median(ep_speed) > threshold)

np.save(os.path.join(data_dir, "epoch_labels.npy"), labels)
print(f"Labeled {np.sum(labels==1)} as moving, {np.sum(labels==0)} as not moving, total {len(labels)} epochs")
