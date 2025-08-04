import scipy.io as sio
import os

data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20100705S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6"
mat = sio.loadmat(os.path.join(data_dir, "Motion.mat"))

MotionData = mat['MotionData']
print("MotionData shape:", MotionData.shape)
for i in range(MotionData.shape[0]):
    entry = MotionData[i, 0]
    print(f"\nEntry {i}: type={type(entry)}")
    if hasattr(entry, 'dtype') and entry.dtype.names:
        print("Fields:", entry.dtype.names)
        for field in entry.dtype.names:
            print(f"  Field '{field}': shape={getattr(entry[field][0,0], 'shape', None)}")
