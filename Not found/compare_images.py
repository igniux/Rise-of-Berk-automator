import cv2
import numpy as np
import os

# Get the folder where the script is running
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (main project folder)
parent_dir = os.path.dirname(script_dir)

# Build paths relative to the script folder
image_a_path = os.path.join(script_dir, "not_found_Terrible_Terror_Search_Selection_2025-10-09_01-07-28.png")
logo_b_path = os.path.join(parent_dir, "icons", "Terrible_Terror_Search_Selection.png")

# Load images
image_a = cv2.imread(image_a_path)
logo_b = cv2.imread(logo_b_path, cv2.IMREAD_UNCHANGED)
print(logo_b.shape)  # If it prints (h, w, 4), there’s an alpha channel
# If logo has alpha channel (transparency), create a mask
if logo_b.shape[2] == 4:
    alpha_channel = logo_b[:, :, 3]
    mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)[1]
    logo_b = cv2.cvtColor(logo_b, cv2.COLOR_BGRA2BGR)
else:
    mask = None

# Convert to grayscale for template matching
gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(logo_b, cv2.COLOR_BGR2GRAY)

# Template matching
result = cv2.matchTemplate(gray_a, gray_b, cv2.TM_CCOEFF_NORMED, mask=mask)

# Get the maximum similarity
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print("Maximum similarity:", max_val)

# Decide a threshold for "existence"
threshold = 0.8
if max_val >= threshold:
    print("Logo exists in image!")
    # Optional: draw rectangle where it was found
    h, w = logo_b.shape[:2]

    top_left = max_loc
    result = cv2.matchTemplate(gray_a, gray_b, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(image_a, top_left, bottom_right, (0, 255, 0), 2)
    right_middle = (top_left[0] + w, top_left[1] + h // 2)
    print(f"Logo found! Right-middle point coordinates: {right_middle}")
    cv2.imshow("Detected Logo", image_a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Logo not found.")
