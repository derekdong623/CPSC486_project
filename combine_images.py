import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Define the number of rows and columns
rows = 1
cols = 2

figure_num = 2
num_subfigures = rows * cols

image_folder = f'figures/'

# List of image file names in order
image_files = [
    f"fig{figure_num}{chr(ord('a')+i)}.png" for i in range(num_subfigures)
]

# Open the images and find the maximum width and height of the images
images = [Image.open(image_folder + image) for image in image_files]
widths, heights = zip(*(i.size for i in images))

# Determine the size of the combined image
total_width = max(widths) * cols
total_height = max(heights) * rows

# Create a new image with a white background
new_image = Image.new('RGB', (total_width, total_height), 'white')

# Paste each image into the new image
for index, image in enumerate(images):
    x_offset = (index % cols) * max(widths)
    y_offset = (index // cols) * max(heights)
    new_image.paste(image, (x_offset, y_offset))

# Save the combined image
new_image.save(image_folder + f'fig{figure_num}.png')

# # Set up the figure and axes
# fig, axs = plt.subplots(rows, cols, figsize=(6, 8))  # Adjust figsize to scale the final image size

# for ax, img in zip(axs.ravel(), images):
#     ax.imshow(np.array(img))
#     ax.axis('off')  # Hide the axes

# # Add title at the bottom
# fig.suptitle('Figure 1. Effect of childbirth event', fontsize=8, y=0.05)  # Adjust fontsize and y to move and resize title

# # Adjust layout to prevent overlapping
# plt.tight_layout()

# # Save the resulting image
# plt.savefig(image_folder + 'fig1.png')
# plt.show()