from PIL import Image, ImageOps
import numpy as np
from skimage.feature import match_template
import os

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
icons_dir = os.path.join(os.path.dirname(script_dir), "icons")
screen_img_path = os.path.join(script_dir, "Screenshot_2025-10-07-21-19-07-449_com.ludia.dragons.jpg")
template_path = os.path.join(icons_dir, "X.png ")

# Load images
img = Image.open(screen_img_path).convert("RGB")
template = Image.open(template_path).convert("RGBA")

# Find bounding box of opaque pixels
alpha = np.array(template)[:, :, 3]
ys, xs = np.where(alpha > 0)
if ys.size == 0 or xs.size == 0:
    raise ValueError("Template has no opaque pixels!")
bbox = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
template_cropped = template.crop(bbox)

# Convert to grayscale numpy arrays
gray_img = np.array(ImageOps.grayscale(img), dtype=np.float32)
gray_template = np.array(ImageOps.grayscale(template_cropped), dtype=np.float32)

# Fast template matching
result = match_template(gray_img, gray_template)
ij = np.unravel_index(np.argmax(result), result.shape)
y, x = ij
max_val = result[y, x]

print(f"Best match at (x={x}, y={y}) with confidence {max_val:.3f}")

# Visualization (optional)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(1)
ax.imshow(img)
h, w = gray_template.shape
rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.title(f"Best match at ({x}, {y}), confidence={max_val:.3f}")
plt.show()