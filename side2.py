import matplotlib.pyplot as plt
from PIL import Image
import os

img_dir = r'C:\Users\mahsa\Downloads\ricsimulationironclad\step1'

# List your pairs [(food, eye)]
figure_pairs = [
    ('Figure_1food.png', 'Figure_1eye.png'),  # Violin plots
    ('Figure_2food.png', 'Figure_2eye.png')   # UMAPs
]

row_titles = ['RIC Curvature Distribution', 'UMAP Embedding']
col_titles = ['Food-Tracking Dataset', 'Eyes Open/Closed Dataset']

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(11, 10))
plt.subplots_adjust(wspace=0.02, hspace=0.03, top=0.93, bottom=0.04)

for row in range(2):
    for col in range(2):
        # Determine which image to load
        img_file = figure_pairs[row][col]
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path)
        axs[row, col].imshow(img)
        axs[row, col].axis('off')

        # Add row labels on the left
        if col == 0:
            axs[row, col].text(-0.10, 0.5, row_titles[row], fontsize=13, va='center', ha='right', transform=axs[row, col].transAxes, fontweight='bold')

        # Add column titles on the top
        if row == 0:
            axs[row, col].set_title(col_titles[col], fontsize=13, pad=8)

fig.suptitle('RIC Curvature Benchmark: Food-Tracking vs Eyes Open/Closed', fontsize=16, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
save_path = os.path.join(img_dir, 'ric_benchmark_sidebyside.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Composite figure saved to: {save_path}")
