#!/usr/bin/env python3
"""
Bot-specific Mask Test
This replicates the exact conditions and template matching logic from bot_run.py
"""

import cv2
import numpy as np
import os
from ppadb.client import Client as AdbClient
import time

def get_screen_capture(device):
    """Exact copy of bot_run.py get_screen_capture"""
    try:
        result = device.screencap()
        img = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            print(f"[DEBUG] Screenshot successfull: {img.shape}")
        else:
            print(f"[ERROR] Screenshot decode failed")
        return img
    except Exception as e:
        print(f"[ERROR] Screenshot capture failed: {e}")
        return None

def test_bot_template_matching(device, template_name):
    """Exact copy of the template matching logic from locate_and_press"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icons_dir = os.path.join(script_dir, "icons")
    template_path = os.path.join(icons_dir, template_name)
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)

    if template is None:
        print(f"Template image {template_name} not found in icons folder.")
        return False

    print(f"[DEBUG] Template {template_name} loaded successfully")
    print(f"[DEBUG] Original template shape: {template.shape}")
    
    # Check for valid template data
    if template.size == 0:
        print(f"[ERROR] Template {template_name} is empty!")
        return False

    # Handle alpha channel (exactly like bot_run.py)
    if template.shape[2] == 4:
        print(f"[DEBUG] Template has alpha channel, creating mask")
        alpha_channel = template[:, :, 3]
        mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)[1]
        template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)
        print(f"[DEBUG] Mask created: shape={mask.shape}, dtype={mask.dtype}")
        print(f"[DEBUG] Template converted to BGR: shape={template.shape}")
    else:
        print(f"[DEBUG] Template has no alpha channel")
        mask = None

    # Get screenshot exactly like bot
    for attempt in range(3):
        print(f"\n--- Attempt {attempt + 1} ---")
        img = get_screen_capture(device)
        
        if img is None:
            print(f"[ERROR] Failed to capture screenshot on attempt {attempt + 1}")
            continue
            
        print(f"[DEBUG] Screenshot shape: {img.shape}, dtype: {img.dtype}")
            
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        print(f"[DEBUG] Gray image shape: {gray_img.shape}, dtype: {gray_img.dtype}")
        print(f"[DEBUG] Gray template shape: {gray_template.shape}, dtype: {gray_template.dtype}")
        if mask is not None:
            print(f"[DEBUG] Mask shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"[DEBUG] Mask values - min: {mask.min()}, max: {mask.max()}")
        
        # Template matching (exactly like bot_run.py)
        print(f"[DEBUG] Calling cv2.matchTemplate with mask={'None' if mask is None else 'provided'}")
        try:
            result = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED, mask=mask)
            print(f"[DEBUG] Template matching successful, result shape: {result.shape}")
        except Exception as e:
            print(f"[ERROR] cv2.matchTemplate failed: {e}")
            continue
        
        # Get the maximum similarity (exactly like bot_run.py)
        try:
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            print(f"[DEBUG] Template matching confidence: {max_val:.6f}")
            
            # Check for nan values
            if np.isnan(max_val):
                print(f"[ERROR] Template matching returned NaN!")
                print(f"[DEBUG] Result array stats - min: {np.nanmin(result)}, max: {np.nanmax(result)}")
                print(f"[DEBUG] Result contains {np.count_nonzero(np.isnan(result))} NaN values")
                return False
            else:
                print(f"[SUCCESS] Template matching returned valid value: {max_val:.6f}")
                return True
                
        except Exception as e:
            print(f"[ERROR] cv2.minMaxLoc failed: {e}")
            continue
    
    return False

def main():
    print("=== BOT-SPECIFIC MASK TEST ===")
    print("Replicating exact bot_run.py conditions\n")
    
    # Connect to ADB
    client = AdbClient(host="127.0.0.1", port=5037)
    devices = client.devices()
    
    if len(devices) == 0:
        print("[ERROR] No ADB device found!")
        return
    
    device = devices[0]
    print(f"[INFO] Connected to device: {device}")
    
    # Test templates that caused issues in bot
    test_templates = [
        "Reconnect.png",
        "X.png", 
        "Head_toothless_left_up.png"
    ]
    
    for template_name in test_templates:
        print(f"\n{'='*60}")
        print(f"TESTING: {template_name}")
        print('='*60)
        
        if not os.path.exists(os.path.join("icons", template_name)):
            print(f"[SKIP] Template {template_name} not found")
            continue
            
        success = test_bot_template_matching(device, template_name)
        if success:
            print(f"[RESULT] ✅ {template_name} - SUCCESS")
        else:
            print(f"[RESULT] ❌ {template_name} - FAILED (likely NaN)")

if __name__ == "__main__":
    main()