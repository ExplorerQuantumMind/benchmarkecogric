import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20100705S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6"
mat = sio.loadmat(os.path.join(data_dir, "Motion.mat"))
MotionData = mat['MotionData']

# Let's plot all six to see which is most active (likely the wrist or hand)
for i in range(6):
    marker_xyz = MotionData[i, 0]  # shape (124230, 3)
    speed = np.linalg.norm(np.diff(marker_xyz.astype(float), axis=0), axis=1)
    plt.plot(speed, label=f'Marker {i}')
plt.legend()
plt.title("Marker Speeds (First Derivative of Each Marker)")
plt.show()
