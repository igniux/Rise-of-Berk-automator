import cv2
import numpy as np
import os

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
image_a_path = os.path.join(script_dir, "Screenshot_20250920_131535_com.ludia.dragons.jpg")
logo_b_path = os.path.join(script_dir, "X.png")

# --- Load images ---
image_a = cv2.imread(image_a_path)
logo_b = cv2.imread(logo_b_path, cv2.IMREAD_UNCHANGED)

# Handle alpha channel
if logo_b.shape[2] == 4:
    alpha_channel = logo_b[:, :, 3]
    mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)[1]
    logo_b = cv2.cvtColor(logo_b, cv2.COLOR_BGRA2BGR)
    cv2.imshow("Logo with Alpha", logo_b)
else:
    mask = None

gray_b = cv2.cvtColor(logo_b, cv2.COLOR_BGR2GRAY)

# --- Define crop area (top-right, half the size of original) ---
h_img, w_img = image_a.shape[:2]

# Crop coordinates
crop_w = w_img // 2
crop_h = h_img // 2
crop_x = w_img - crop_w  # top-right
crop_y = 0               # top

# Crop image
crop_area = image_a[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
gray_crop = cv2.cvtColor(crop_area, cv2.COLOR_BGR2GRAY)

# --- Template matching in the cropped area ---
result = cv2.matchTemplate(gray_crop, gray_b, cv2.TM_CCOEFF_NORMED, mask=mask)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

threshold = 0.8
if max_val >= threshold:
    # Coordinates relative to original image
    top_left = (max_loc[0] + crop_x, max_loc[1] + crop_y)
    h, w = logo_b.shape[:2]
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Right-middle point
    right_middle = (top_left[0] + w, top_left[1] + h // 2)

    print(f"Logo found! Right-middle coordinates in original image: {right_middle}")

    # Draw rectangle for visualization
    cv2.rectangle(image_a, top_left, bottom_right, (0, 255, 0), 2)
    cv2.imshow("Detected Logo", image_a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Logo not found.")


# Below is locate and press function from compare_with_orb.py with cropping logic integrated
'''def locate_and_press(img, device, template_name, action_desc, threshold=0.8):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, template_name) 
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)

    if template is None:
        print(f"Template image {template_name} not found.")
        return False

    # Handle alpha channel (for transparent background templates)
    if template.shape[2] == 4:
        alpha_channel = template[:, :, 3]
        mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)[1]
        template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)
    else:
        mask = None

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Use template matching with mask
    result = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED, mask=mask)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        h, w = template.shape[:2]
        top_left = max_loc
        center_x = top_left[0] + w // 2
        center_y = top_left[1] + h // 2
        device.input_tap(center_x, center_y)
        print(f"{action_desc} - Pressed at ({center_x}, {center_y}) with confidence {max_val:.2f}")
        time.sleep(2)  # Wait for action to complete
        return True
    else:
        print(f"{action_desc} - Not found (max confidence {max_val:.2f})")
        return False'''