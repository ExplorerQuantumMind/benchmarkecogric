import matplotlib.pyplot as plt
from PIL import Image
import os

img_dir = r'C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128\Session1'

img_files = {
    'Lateral': {
        'Eyes Closed': 'ric_surface_eyesclosed_lateral.png',
        'Eyes Opened': 'ric_surface_eyesopened_lateral.png'
    },
    'Posterior': {
        'Eyes Closed': 'ric_surface_eyesclosed_posterior.png',
        'Eyes Opened': 'ric_surface_eyesopened_posterior.png'
    },
    'Medial': {
        'Eyes Closed': 'ric_surface_eyesclosed_medial.png',
        'Eyes Opened': 'ric_surface_eyesopened_medial.png'
    },
    'Dorsal': {
        'Eyes Closed': 'ric_surface_eyesclosed_dorsal.png',
        'Eyes Opened': 'ric_surface_eyesopened_dorsal.png'
    }
}

col_titles = ['Eyes Closed', 'Eyes Opened']
row_titles = ['Lateral', 'Posterior', 'Medial', 'Dorsal']

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(6, 8))
plt.subplots_adjust(wspace=0, hspace=0.03, top=0.94, bottom=0.03)

for row_idx, region in enumerate(row_titles):
    for col_idx, condition in enumerate(col_titles):
        img_path = os.path.join(img_dir, img_files[region][condition])
        img = Image.open(img_path)
        axs[row_idx, col_idx].imshow(img)
        axs[row_idx, col_idx].axis('off')
        if col_idx == 0:
            axs[row_idx, col_idx].text(-0.12, 0.5, region, fontsize=11, va='center', ha='right', transform=axs[row_idx, col_idx].transAxes, fontweight='bold')
        if row_idx == 0:
            axs[row_idx, col_idx].set_title(condition, fontsize=12, pad=4)

fig.suptitle('RIC Curvature Maps: Eyes Closed vs Eyes Opened', fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0.01, 1, 0.95], pad=0.1)
save_path = os.path.join(img_dir, 'ric_surface_composite_eyes_supercompact.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Composite image saved to: {save_path}")
