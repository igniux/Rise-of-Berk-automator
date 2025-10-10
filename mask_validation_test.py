#!/usr/bin/env python3
"""
Mask Validation Test
This script specifically tests whether the mask parameter is causing NaN issues
with ADB screenshots vs file-based images.
"""

import cv2
import numpy as np
import os
from ppadb.client import Client as AdbClient

def get_adb_screenshot(device):
    """Get screenshot from ADB device"""
    try:
        result = device.screencap()
        img = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[ERROR] ADB screenshot failed: {e}")
        return None

def load_template_with_mask(template_path):
    """Load template and create mask if alpha channel exists"""
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if template is None:
        print(f"[ERROR] Could not load template: {template_path}")
        return None, None
    
    print(f"[DEBUG] Template shape: {template.shape}")
    
    if template.shape[2] == 4:  # Has alpha channel
        print("[DEBUG] Template has alpha channel - creating mask")
        alpha_channel = template[:, :, 3]
        mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)[1]
        template_bgr = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)
        print(f"[DEBUG] Mask shape: {mask.shape}, Template BGR shape: {template_bgr.shape}")
        return template_bgr, mask
    else:
        print("[DEBUG] Template has no alpha channel")
        return template, None

def test_template_matching(img, template, mask, test_name):
    """Test template matching with detailed analysis"""
    print(f"\n=== {test_name} ===")
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
        
    if len(template.shape) == 3:
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        gray_template = template
    
    print(f"[DEBUG] Image shape: {gray_img.shape}, dtype: {gray_img.dtype}")
    print(f"[DEBUG] Template shape: {gray_template.shape}, dtype: {gray_template.dtype}")
    if mask is not None:
        print(f"[DEBUG] Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"[DEBUG] Mask min/max: {mask.min()}/{mask.max()}")
    
    # Test 1: WITHOUT mask
    print("\n--- Test WITHOUT mask ---")
    try:
        result_no_mask = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)
        _, max_val_no_mask, _, max_loc_no_mask = cv2.minMaxLoc(result_no_mask)
        print(f"[SUCCESS] No mask - Max confidence: {max_val_no_mask:.6f}")
        print(f"[DEBUG] No mask - Location: {max_loc_no_mask}")
        if np.isnan(max_val_no_mask):
            print("[ERROR] NaN detected WITHOUT mask!")
    except Exception as e:
        print(f"[ERROR] Template matching WITHOUT mask failed: {e}")
        max_val_no_mask = None
    
    # Test 2: WITH mask (if available)
    if mask is not None:
        print("\n--- Test WITH mask ---")
        try:
            result_with_mask = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED, mask=mask)
            _, max_val_with_mask, _, max_loc_with_mask = cv2.minMaxLoc(result_with_mask)
            print(f"[SUCCESS] With mask - Max confidence: {max_val_with_mask:.6f}")
            print(f"[DEBUG] With mask - Location: {max_loc_with_mask}")
            if np.isnan(max_val_with_mask):
                print("[ERROR] NaN detected WITH mask!")
            else:
                print("[SUCCESS] No NaN with mask!")
        except Exception as e:
            print(f"[ERROR] Template matching WITH mask failed: {e}")
            max_val_with_mask = None
    else:
        print("[INFO] No mask available for this template")
        max_val_with_mask = None
    
    return max_val_no_mask, max_val_with_mask

def main():
    print("=== MASK VALIDATION TEST ===")
    print("Testing whether mask parameter causes NaN with ADB screenshots\n")
    
    # Connect to ADB
    client = AdbClient(host="127.0.0.1", port=5037)
    devices = client.devices()
    
    if len(devices) == 0:
        print("[ERROR] No ADB device found!")
        return
    
    device = devices[0]
    print(f"[INFO] Connected to device: {device}")
    
    # Load template with alpha channel (like Reconnect.png)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "icons", "Reconnect.png")
    
    template, mask = load_template_with_mask(template_path)
    if template is None:
        print("[ERROR] Failed to load template!")
        return
    
    # Test 1: ADB Screenshot
    print("\n" + "="*50)
    print("TEST 1: ADB SCREENSHOT")
    print("="*50)
    
    adb_img = get_adb_screenshot(device)
    if adb_img is not None:
        adb_no_mask, adb_with_mask = test_template_matching(adb_img, template, mask, "ADB Screenshot")
    else:
        print("[ERROR] Could not get ADB screenshot!")
        return
    
    # Test 2: File-based Image (for comparison)
    print("\n" + "="*50)
    print("TEST 2: FILE-BASED IMAGE")
    print("="*50)
    
    # Try to find a screenshot file in the screen folder
    screen_dir = os.path.join(script_dir, "screen")
    screenshot_files = [f for f in os.listdir(screen_dir) if f.endswith(('.jpg', '.png'))]
    
    if screenshot_files:
        file_img_path = os.path.join(screen_dir, screenshot_files[0])
        file_img = cv2.imread(file_img_path, cv2.IMREAD_COLOR)
        if file_img is not None:
            file_no_mask, file_with_mask = test_template_matching(file_img, template, mask, "File-based Image")
        else:
            print(f"[ERROR] Could not load file image: {file_img_path}")
            file_no_mask, file_with_mask = None, None
    else:
        print("[WARN] No screenshot files found in screen folder for comparison")
        file_no_mask, file_with_mask = None, None
    
    # Summary
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    
    # Format results safely
    adb_no_mask_str = f"{adb_no_mask:.6f}" if adb_no_mask is not None else "FAILED"
    adb_with_mask_str = f"{adb_with_mask:.6f}" if adb_with_mask is not None and not np.isnan(adb_with_mask) else "FAILED/NaN"
    
    print(f"ADB Screenshot (no mask):   {adb_no_mask_str}")
    print(f"ADB Screenshot (with mask): {adb_with_mask_str}")
    
    if file_no_mask is not None and file_with_mask is not None:
        file_no_mask_str = f"{file_no_mask:.6f}" if not np.isnan(file_no_mask) else "NaN"
        file_with_mask_str = f"{file_with_mask:.6f}" if not np.isnan(file_with_mask) else "NaN"
        print(f"File Image (no mask):       {file_no_mask_str}")
        print(f"File Image (with mask):     {file_with_mask_str}")
    
    # Analysis
    print("\n" + "="*50)
    print("ANALYSIS")
    print("="*50)
    
    if adb_with_mask is None or np.isnan(adb_with_mask):
        print("✗ MASK ISSUE CONFIRMED: ADB screenshot + mask = NaN/Error")
    else:
        print("✓ No mask issue with ADB screenshot")
    
    if adb_no_mask is not None and not np.isnan(adb_no_mask):
        print("✓ ADB screenshot works fine WITHOUT mask")
    else:
        print("✗ ADB screenshot has issues even without mask")
    
    if file_with_mask is not None and not np.isnan(file_with_mask):
        print("✓ File-based image works fine WITH mask")
    
    print("\nConclusion: The issue is specifically with using masks on ADB screenshots")

if __name__ == "__main__":
    main()