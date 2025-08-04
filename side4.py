import matplotlib.pyplot as plt
from PIL import Image
import os

img_dir = r'C:\Users\mahsa\Downloads\ricsimulationironclad\step1'

# List pairs [(ROC food, ROC eye), (microstate food, microstate eye)]
figure_pairs = [
    ('Figure_4food.png', 'Figure_4eye.png'),   # ROC curves
    ('Figure_5food.png', 'Figure_5eye.png')    # Microstate/attractor clusters
]

row_titles = ['ROC Curve', 'Microstate/Attractor Clusters']
col_titles = ['Food-Tracking Dataset', 'Eyes Open/Closed Dataset']

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
plt.subplots_adjust(wspace=0.02, hspace=0.03, top=0.93, bottom=0.04)

for row in range(2):
    for col in range(2):
        img_file = figure_pairs[row][col]
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path)
        axs[row, col].imshow(img)
        axs[row, col].axis('off')

        if col == 0:
            axs[row, col].text(-0.10, 0.5, row_titles[row], fontsize=13, va='center', ha='right', transform=axs[row, col].transAxes, fontweight='bold')
        if row == 0:
            axs[row, col].set_title(col_titles[col], fontsize=13, pad=8)

fig.suptitle('ROC and Microstate Clustering: Food-Tracking vs Eyes Open/Closed', fontsize=16, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
save_path = os.path.join(img_dir, 'ric_roc_microstate_sidebyside_corrected.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Corrected composite figure saved to: {save_path}")
