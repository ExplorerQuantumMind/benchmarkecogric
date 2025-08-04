import matplotlib.pyplot as plt
from PIL import Image
import os

img_dir = r'C:\Users\mahsa\Downloads\ricsimulationironclad\step1'

# File names for the new grid comparison
img_files = [
    ('Figure_3food.png', 'Figure_3eye.png')
]

col_titles = ['Food-Tracking Dataset', 'Eyes Open/Closed Dataset']

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
plt.subplots_adjust(wspace=0.02, top=0.93, bottom=0.05)

for col in range(2):
    img_path = os.path.join(img_dir, img_files[0][col])
    img = Image.open(img_path)
    axs[col].imshow(img)
    axs[col].axis('off')
    axs[col].set_title(col_titles[col], fontsize=14, pad=8)

fig.suptitle('Channel-wise Mean RIC Curvature: Food-Tracking vs Eyes Open/Closed', fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0.01, 1, 0.95])
save_path = os.path.join(img_dir, 'ric_channelgrid_sidebyside.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Composite figure saved to: {save_path}")
