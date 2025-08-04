import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load data ---
data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20100705S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6"
curvature = np.load(os.path.join(data_dir, "ric_curvature.npy"))  # shape (64, epochs)

plt.figure(figsize=(18, 6))
sns.violinplot(data=curvature.T, inner='box', color='0.3')
plt.xlabel("Channel")
plt.ylabel("RIC Curvature")
plt.title("Distribution of RIC Curvature Across Channels (Food-Tracking Task)")
plt.xticks(
    ticks=np.arange(0, curvature.shape[0], 4),
    labels=[str(i) for i in range(0, curvature.shape[0], 4)],
    rotation=90, fontsize=8
)
plt.tight_layout()
plt.show()
