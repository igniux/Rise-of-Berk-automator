import os
import cv2
import numpy as np
import time

template_name = "X.png"
threshold = 0.8  # Confidence threshold

def locate_and_press(img, device, template_name, action_desc):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, template_name)
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)

    if template is None:
        print(f"Template image {template_name} not found.")
        return False

    # Remove alpha channel if present
    if template.shape[2] == 4:
        template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)

    # Convert both to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

    # Create mask of the logo (non-background pixels)
    lower = np.array([0,0,0])
    upper = np.array([255,255,255])
    logo_mask = cv2.inRange(hsv_template, lower, upper)

    # Use template matching with mask
    result = cv2.matchTemplate(hsv_img, hsv_template, cv2.TM_CCOEFF_NORMED, mask=logo_mask)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        h, w = template.shape[:2]
        top_left = max_loc
        center_x = top_left[0] + w // 3
        center_y = top_left[1] + h // 3
        device.input_tap(center_x, center_y)
        print(f"{action_desc} - Pressed at ({center_x}, {center_y}) with confidence {max_val:.2f}")
        time.sleep(2)  # Wait for action to complete
        return True
    else:
        print(f"{action_desc} - Not found (max confidence {max_val:.2f})")
        return False
    
if __name__ == "__main__":
    # For testing purposes
    test_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Screenshot_20250920_131535_com.ludia.dragons.jpg")
    img = cv2.imread(test_image_path)
    if img is not None:
        locate_and_press(img, None, template_name, "Test Action")
    else:
        print("Test image not found.")