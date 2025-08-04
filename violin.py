import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128\Session1"
curvature = np.load(f"{data_dir}/ric_curvature.npy")

plt.figure(figsize=(20, 7))
sns.violinplot(data=curvature.T, inner="box", color="#3b8ec2")

# Clean up x-axis: show fewer ticks
plt.xlabel('Channel')
plt.ylabel('RIC Curvature')
plt.title('Distribution of RIC Curvature Across Channels')

ax = plt.gca()
n_channels = curvature.shape[0]
step = 8  # Show every 8th channel label
ax.set_xticks(np.arange(0, n_channels, step))
ax.set_xticklabels(np.arange(1, n_channels+1, step), rotation=0, fontsize=11)

plt.tight_layout()
plt.show()
