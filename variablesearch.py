import scipy.io as sio
import os

data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20100705S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6"
mat = sio.loadmat(os.path.join(data_dir, "Motion.mat"))

print("Variables in Motion.mat:")
for k in mat:
    if not k.startswith("__"):
        print(f"{k}: type={type(mat[k])}, shape={getattr(mat[k], 'shape', None)}")
        if isinstance(mat[k], np.ndarray) and mat[k].dtype.names:
            print(f"   (dtype.names: {mat[k].dtype.names})")
