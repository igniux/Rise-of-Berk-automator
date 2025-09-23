import cv2
import numpy as np
import os

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
icons_dir = os.path.join(parent_dir, "icons")
screen_img_path = os.path.join(script_dir, "Screenshot_20250921_141606_com.ludia.dragons.png")
bag_template_path = os.path.join(icons_dir, "Bag.png")

# Load images
image_a = cv2.imread(screen_img_path)
bag_b = cv2.imread(bag_template_path, cv2.IMREAD_UNCHANGED)

if image_a is None:
    print("[ERROR] Could not load screenshot.")
    exit(1)
if bag_b is None:
    print("[ERROR] Could not load Bag.png template.")
    exit(1)

# If Bag.png has alpha channel, create mask
mask = None
if bag_b.shape[2] == 4:
    alpha_channel = bag_b[:, :, 3]
    mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)[1]
    bag_b = cv2.cvtColor(bag_b, cv2.COLOR_BGRA2BGR)

# Convert to grayscale for template matching
gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(bag_b, cv2.COLOR_BGR2GRAY)

# Template matching
result = cv2.matchTemplate(gray_a, gray_b, cv2.TM_CCOEFF_NORMED, mask=mask)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print("Maximum similarity:", max_val)

threshold = 0.8
if max_val >= threshold:
    h, w = bag_b.shape[:2]
    top_left = max_loc
    center_x = top_left[0] + w // 2
    center_y = top_left[1] + h // 2
    print(f"Bag found! Center coordinates: ({center_x}, {center_y})")
else:
    print("Bag not found.")
    exit(1)

# Function to get patch mean color
def get_patch_mean(img, x, y, patch_size=10):
    patch = img[y-patch_size:y+patch_size+1, x-patch_size:x+patch_size+1]
    return patch.mean(axis=(0,1))

# List of screenshots to process
screenshots = [
    "Screenshot_20250920_131828_com.ludia.dragons.jpg",
    "Screenshot_20250920_131902_com.ludia.dragons.jpg",
    "Screenshot_20250921_141435_com.ludia.dragons.jpg",
    "Screenshot_20250921_141503_com.ludia.dragons.jpg"
]

print("\nBag patch mean colors in screenshots:")
for fname in screenshots:
    img_path = os.path.join(script_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not load {fname}")
        continue
    mean_color = get_patch_mean(img, center_x, center_y, patch_size=10)
    print(f"{fname}: {mean_color}")