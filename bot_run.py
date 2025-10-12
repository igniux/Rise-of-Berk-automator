from ppadb.client import Client as AdbClient
import cv2
import numpy as np
import time
import configuration as c
import threading
import os
import datetime  # Add this import at the top if not present
import json

# The name of the app, to check if it's running
TARGET_APP_PKG = "com.ludia.dragons"
TARGET_APP_ACTIVITY = "com.ludia.dragons/com.ludia.engine.application"

# If you run the script directly on mobile, set this to True to disable
# incompatible functions, like real-time image view, and configure for this
RUN_ON_MOBILE = False

# Import Termux:GUI to diplay overlay if script is running on Android
if RUN_ON_MOBILE:
    import termuxgui as tg

# Load or create tap_info.json
tap_file = "tap_info.json"

if not os.path.exists(tap_file):
    print("[INFO] tap_info.json not found — creating a new one.")
    with open(tap_file, "w") as f:
        json.dump({}, f, indent=2)

with open(tap_file, "r") as f:
    try:
        tap_info = json.load(f)
    except json.JSONDecodeError:
        print("[WARN] tap_info.json was invalid, resetting.")
        tap_info = {}
        with open(tap_file, "w") as fw:
            json.dump(tap_info, fw, indent=2)


last_collected = None  # or last_collected = ""

# A function that generates a button with specified text, layout, and optional width.
def create_overlay_button(activity, text, layout, width=40):
    button = tg.Button(activity, text, layout)
    button.settextsize(12)
    button.setlinearlayoutparams(1)
    if width:
        button.setwidth(width)
        button.setheight(width)

    return button

# Create an overlay with buttons to control bot state
def display_overlay_on_android(height, connection):
    activity = tg.Activity(connection, tid=110, overlay=True)
    activity.setposition(9999, int(height * 0.1))  # Position at 10% from top
    activity.keepscreenon(True)
    activity.sendoverlayevents(False)
    
    rootLinear = tg.LinearLayout(activity, vertical=False)
    
    play_pause_btn = create_overlay_button(activity, "⏸️ Pause", rootLinear) 
    exit_btn = create_overlay_button(activity, "❌", rootLinear) 
    
    time.sleep(1)
            
    return play_pause_btn, exit_btn

# Set flags for next action when button press
pause_flag, exit_flag = [False] * 2
def action_on_overlay_button_press(connection, play_pause_btn, exit_btn):
    global pause_flag, exit_flag
    for event in connection.events():
        if event and event.type == tg.Event.click and event.value["id"] == play_pause_btn.id:
            pause_flag = not pause_flag
            if pause_flag:
                play_pause_btn.settext("▶️ Continue")
                connection.toast("Bot is pausing")
            else:
                play_pause_btn.settext("⏸️ Pause")
                connection.toast("Bot starting")
        if event and event.type == tg.Event.click and event.value["id"] == exit_btn.id:
            exit_flag = True
            connection.toast("Closing bot")

# Check the state of the button flags
def do_button_flags(device):
    global pause_flag, exit_flag
    print(f"-")
    if not check_app_in_foreground(device, TARGET_APP_PKG):
        print("[ERROR] Rise of Berk app is not running anymore")
        exit()
    while pause_flag:
        print("[INFO] Bot is paused. Waiting for resume or exit...")
        time.sleep(0.5)
        print(f"[DEBUG] Inside pause loop: pause_flag={pause_flag}, exit_flag={exit_flag}")
        if exit_flag:
            print("[INFO] Exit flag detected during pause. Exiting bot.")
            exit()
    if exit_flag:
        print("[INFO] Exit flag detected. Exiting bot.")
        exit()  

# Get and image from ADB and transform it to opencv image
def get_screen_capture(device):
    try:
        result = device.screencap()
        img = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[ERROR] Screenshot decode failed")
        return img
    except Exception as e:
        print(f"[ERROR] Screenshot capture failed: {e}")
        return None

# Checks if the current app running on the device is Rise of Berk and the screen is on.
def check_app_in_foreground(device, target):
    result = device.shell("dumpsys window | grep -E 'mCurrentFocus'")
    if target in result or 'com.termux.gui' in result:
        return True
    return False

# Global debug counter to prevent infinite recursion
debug_call_count = 0

def debugger(device, template_name=None, type="check_color_and_tap"):
    global debug_call_count
    debug_call_count += 1
    
    if debug_call_count > 5:
        print(f"\033[91m{'='*60}")
        print(f"💀 FATAL: DEBUGGER CALLED {debug_call_count} TIMES")
        print(f"💀 PREVENTING INFINITE LOOP - TAKING SCREENSHOT")
        print(f"{'='*60}\033[0m")
        img = get_screen_capture(device)
        fatal_error("Too many debugger calls - possible infinite loop", img, device)
        debug_call_count = 0
        return False
    
    print(f"\033[91m{'='*40}")
    print(f"🚨 DEBUGGER CALLED - ATTEMPT {debug_call_count}")
    print(f"{'='*40}\033[0m")
    
    # Single call to handle all debugger scenarios
    found_template = locate_and_press(device, [
        ("X.png", True),                      # Verify if found
        ("Reconnect.png", True),             # Verify if found
        ("Head_toothless_left_up.png", True) # Verify if found
    ], "Handle debugger actions", timeout=5, no_debugger=True)
    
    # Handle different logic based on what was found
    
        
    if found_template == "Reconnect.png":
        print("[INFO] Reconnect button detected, handling reconnection")
        locate_and_press(device, "Reconnect.png", "Reconnect button found", no_debugger=True, timeout=5)
        print("Waiting for reconnect button to clear")
        debug_call_count = 0  # Reset on success
        last_seen = time.time()
        while True:
            if check_color_and_tap(device, "Reconnect", timeout=1.0, no_debugger=True):
                print("[INFO] Reconnect detected and tapped. Resetting timer.")
                last_seen = time.time()
            else:
                if time.time() - last_seen >= 10:
                    print("[INFO] No Reconnect detected for 10 seconds. Proceeding.")
                    return True
            time.sleep(0.5)
    elif found_template == "X.png":
        print("[INFO] X button detected, closing popup")
        locate_and_press(device, "X.png", "Close popup", no_debugger=True, timeout=5)
        debug_call_count = 0  # Reset on success
        return True
    elif found_template == "Head_toothless_left_up.png":
        print("[INFO] Toothless detected, trying to navigate back to main screen")
        locate_and_press(device, [("Head_toothless_left_up.png", False), ("Resend.png", True)], "Press on toothless to get to resend state", timeout=10, no_debugger=True)
        # debug_call_count = 0  # Reset on success
        return True
        
    else:
        # No standard recovery options found, try last activity if provided
        if template_name:
            print(f"[INFO] No standard recovery found, trying last activity: {template_name}")
            if type == "locate_and_press":
                if locate_and_press(device, template_name, f"Try {template_name}", no_debugger=True, timeout=5):
                    print(f"{template_name} found and pressed.")
                    debug_call_count = 0  # Reset on success
                    return True
            elif type == "check_color_and_tap":
                if check_color_and_tap(device, template_name, no_debugger=True, timeout=5):
                    print(f"{template_name} found and pressed.")
                    debug_call_count = 0  # Reset on success
                    return True
        
        print("[ERROR] Debugger could not resolve the issue. This is a critical failure!")
        # Take a screenshot of the current state
        current_img = get_screen_capture(device)
        if current_img is not None:
            fatal_error("Debugger failed to find any recovery options - bot cannot continue", current_img, device)
        debug_call_count = 0  # Reset to prevent accumulation
        return False
            
def fatal_error(msg, img, device=None):
    # Create error folder if it doesn't exist
    error_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "error")
    os.makedirs(error_dir, exist_ok=True)
    # Generate filename with current time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"error_{timestamp}.png"
    filepath = os.path.join(error_dir, filename)
    # Save image
    cv2.imwrite(filepath, img)
    
    # HIGHLY VISIBLE FATAL ERROR MESSAGE
    print(f"\033[41;97m{'='*80}")
    print(f"💀💀💀 FATAL ERROR - BOT CANNOT CONTINUE 💀💀💀")
    print(f"{'='*80}")
    print(f"🔥 ERROR: {msg}")
    print(f"📸 Screenshot saved: {filepath}")
    print(f"⚠️  Rise of Berk app will be terminated")
    print(f"{'='*80}\033[0m")
    
    # Terminate Rise of Berk app if device is not None
    if device is not None:
        device.shell(f"am force-stop {TARGET_APP_PKG}")
        print(f"\033[91m🛑 Rise of Berk app terminated\033[0m")
    
    # Do not exit, just return to continue Python code
    return

def Start_Rise_app(device):
    if check_app_in_foreground(device, TARGET_APP_PKG):
        print("Rise of Berk app is running")
        return True
    else:
        print("Please open Rise of Berk. Waiting 25 seconds.")
        print("Attempting to start Rise of Berk app")
        
        # Try multiple launch methods
        try:
            # Method 1: Using monkey command with package name only
            print(f"[DEBUG] Launching with monkey: {TARGET_APP_PKG}")
            result = device.shell(f"monkey -p {TARGET_APP_PKG} -c android.intent.category.LAUNCHER 1")
            print(f"[DEBUG] Monkey result: {result}")
        except Exception as e:
            print(f"[ERROR] Monkey launch failed: {e}")
            
        time.sleep(3)
        
        # Method 2: Try with am start if monkey failed
        if not check_app_in_foreground(device, TARGET_APP_PKG):
            try:
                print(f"[DEBUG] Trying am start with activity: {TARGET_APP_ACTIVITY}")
                result = device.shell(f"am start -n {TARGET_APP_ACTIVITY}")
                print(f"[DEBUG] Am start result: {result}")
            except Exception as e:
                print(f"[ERROR] Am start failed: {e}")
        
        if not check_app_in_foreground(device, TARGET_APP_PKG):
            fatal_error("Failed to start Rise of Berk app. Exiting script.", get_screen_capture(device))
            return False 
        return True

def locate_and_press(device, template_configs, action_desc, threshold=0.95, timeout=4.0, last_activity_name=None, no_debugger=False, patch_size=25):
    do_button_flags(device)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icons_dir = os.path.join(script_dir, "icons")
    
    # Handle different input formats
    if isinstance(template_configs, str):
        # Single template name (backward compatibility)
        template_configs = [(template_configs, False)]  # False = press
    elif isinstance(template_configs, list) and all(isinstance(item, str) for item in template_configs):
        # List of template names (backward compatibility)
        template_configs = [(name, False) for name in template_configs]
    # If already list of tuples, use as-is
    
    # Load all templates
    templates = []
    for template_name, verify_only in template_configs:
        template_path = os.path.join(icons_dir, template_name)
        template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        
        if template is None:
            print(f"Template image {template_name} not found in icons folder.")
            continue
            
        # Handle alpha channel
        if template.shape[2] == 4:
            alpha_channel = template[:, :, 3]
            mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)[1]
            template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)
        else:
            mask = None
            
        templates.append((template_name, template, mask, verify_only))
        print(f"[DEBUG] Template {template_name} loaded successfully ({'verify only' if verify_only else 'press'})")

    if not templates:
        print(f"No valid templates found from {template_configs}")
        return False

    start_time = time.time()
    attempt_count = 0
    found = False
    
    while time.time() - start_time < timeout:
        attempt_count += 1
        
        # Time image acquisition (only once per attempt)
        img_start_time = time.time()
        img = get_screen_capture(device)
        img_end_time = time.time()
        img_acquisition_time = (img_end_time - img_start_time) * 1000
        print(f"[TIMING] Image acquisition took: {img_acquisition_time:.2f}ms")

        # Convert to grayscale once
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try all templates on the same image
        best_match = None
        best_confidence = 0
        
        processing_start_time = time.time()
        
        for template_name, template, mask, verify_only in templates:
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # Try masked template matching first (if mask available)
            if mask is not None:
                result = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED, mask=mask)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                print(f"[DEBUG] {template_name} masked confidence: {max_val:.3f}")
                
                # If masked confidence is below threshold, try unmasked
                if max_val < threshold:
                    result = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    print(f"[DEBUG] {template_name} unmasked confidence: {max_val:.3f}")
            else:
                result = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                print(f"[DEBUG] {template_name} confidence: {max_val:.3f}")
            
            # Keep track of best match
            if max_val >= threshold and max_val > best_confidence and np.isfinite(max_val):
                best_confidence = max_val
                best_match = (template_name, template, max_loc, max_val, verify_only)
        
        processing_end_time = time.time()
        total_processing_time = (processing_end_time - processing_start_time) * 1000
        total_time_from_acquisition = (processing_end_time - img_start_time) * 1000
        
        print(f"[TIMING] Template matching took: {total_processing_time:.2f}ms")
        print(f"[TIMING] Total time (acquisition + processing): {total_time_from_acquisition:.2f}ms")
        
        print(f"[DEBUG] {action_desc} - Attempt {attempt_count}: Best confidence = {best_confidence:.3f}, Threshold = {threshold}")

        if best_match:
            found = True
            template_name, template, max_loc, max_val, verify_only = best_match
            h, w = template.shape[:2]
            top_left = max_loc
            center_x = top_left[0] + w // 2
            center_y = top_left[1] + h // 2

            # Remove .png extension for tap_info key
            key_name = template_name.replace('.png', '')

            # Save location and color info
            patch = img[
                center_y - patch_size : center_y + patch_size + 1,
                center_x - patch_size : center_x + patch_size + 1
            ]
            color_code = patch.mean(axis=(0,1))
            print(f"[DEBUG] Patch color: {color_code}")
            
            if not verify_only:
                tap_info[key_name] = [center_x, center_y, color_code.tolist()]
                with open("tap_info.json", "w") as f:
                    json.dump(tap_info, f, indent=2)
                print(f"[DEBUG] Saved to tap_info: {key_name} -> [{center_x}, {center_y}, color]")
                device.input_tap(center_x, center_y)
                time.sleep(0.3)
                device.input_tap(center_x, center_y)
                print(f"\033[92m✅ {action_desc} - PRESSED {template_name} at ({center_x}, {center_y}) with confidence {max_val:.2f}\033[0m")
            else:
                tap_info[key_name] = [center_x, center_y, color_code.tolist()]
                with open("tap_info.json", "w") as f:
                    json.dump(tap_info, f, indent=2)
                print(f"[DEBUG] Saved to tap_info: {key_name} -> [{center_x}, {center_y}, color]")
                print(f"\033[92m✅ {action_desc} - VERIFIED {template_name} at ({center_x}, {center_y}) with confidence {max_val:.2f}\033[0m")
                return template_name
        
        do_button_flags(device)
        time.sleep(0.5)
    
    print(f"{action_desc} - {'Found' if found else 'Not found'} after {timeout} seconds. Total attempts: {attempt_count}")
    if found and best_match:
        return best_match[0]  # Return the actual template name
    if not found:
        if no_debugger:
            return False
        elif debugger(device, template_configs[0][0] if template_configs else None, type="locate_and_press"):
            return True
        return False

def swipe_up(device, img, duration_ms=500):
    """
    Swipe from bottom to top at the horizontal center of the screen.
    duration_ms: swipe duration in milliseconds
    """
    height, width = img.shape[:2]
    x = width // 2
    y_start = int(height * 0.55)  # Start near bottom
    y_end = int(height * 0.3)    # End near top
    device.shell(f"input swipe {x} {y_start} {x} {y_end} {duration_ms}")
    print(f"Swiped from ({x}, {y_start}) to ({x}, {y_end})")
    time.sleep(0.1)  # Wait for swipe action to complete
    
def Initiate_bot_resend_sequence(device):
    print("Initiating bot sequence")
    img = get_screen_capture(device)
    
    locate_and_press(device, [("X.png", False), ("Head_toothless_left_up.png", False), ("Night_Fury.png", True)], "Close any Limited Offers or verify toothless is up", timeout=30)
    locate_and_press(device, [("Search_Button.png", False), ("X.png", True)], "Locate and press Search button until X", timeout=10)
    max_swipes = 5
    for attempt in range(max_swipes):
        if locate_and_press(device, [("Terrible_Terror_Search.png", False), ("Terrible_Terror_Verify.png", True)], "Locate and press Terrible terror in the list", timeout=4, no_debugger=True):
            break
        swipe_up(device, img)
        attempt += 1
        time.sleep(0.3)  # Wait a moment for screen to stabilize
    else:
        print("[ERROR] Terrible Terror not found after swiping. Aborting or handling error.")
        # Optionally call debugger or handle as needed
    locate_and_press(device, [("1_new.png", False), ("Start_Explore.png", True)], "select 1 bag search option", timeout=10)
    locate_and_press(device, [("Start_Explore.png", False), ("Head_toothless_left_up.png", True)], "Locate and press Start Explore button", timeout=10, last_activity_name="Start_Explore.png")
    locate_and_press(device, [("Speed_up.png", False), ("Speedup_Verification.png", True)], "Speed up the exploration free", timeout=10, last_activity_name="Speed_up.png")
    locate_and_press(device, [("Head_toothless_left_up.png", False), ("Bag.png", True)], "Press on toothless, which is back", timeout=10, last_activity_name="Head_toothless_left_up.png")
    locate_and_press(device, [("Bag.png", False), ("Collect.png", True)], "Open Bag", timeout=10, last_activity_name="Bag.png")
    locate_and_press(device, "Collect.png", "Collect toothless rewards", timeout=10, last_activity_name="Collect.png")
    
    # Single call to handle all post-collection scenarios
    found_template = locate_and_press(device, [
        ("No_thanks.png", True),           # Verify if found
        ("Release.png", True),             # Verify if found
        ("Head_toothless_left_up.png", True) # Verify if found
    ], "Handle post-collection actions", timeout=10, last_activity_name="Collect.png")
    
    # Handle different logic based on what was found
    if found_template == "No_thanks.png":
        print("[INFO] Buy egg popup closed, sequence complete")
        return True
        
    elif found_template == "Release.png":
        print("[INFO] Egg detected, starting release sequence")
        locate_and_press(device, [("Release.png", False), ("Yes.png", True)], "Press Release egg", timeout=10, last_activity_name="Release.png")
        locate_and_press(device, [("Yes.png", False), ("Yes_2.png", False), ("Head_toothless_left_up.png", True)], "Confirm Release egg", timeout=10, last_activity_name="Yes.png")
        locate_and_press(device, [("X.png", False), ("Head_toothless_left_up.png", False)], "Close final popup", no_debugger=True, timeout=10)

    elif found_template == "Head_toothless_left_up.png":
        print("[INFO] Toothless detected, starting resend sequence")
        locate_and_press(device, [("Head_toothless_left_up.png", False), ("Resend.png", True)], "Press on toothless until resend appears", timeout=10, last_activity_name="Head_toothless_left_up.png")
        return True
        
    else:
        locate_and_press(device, [("Head_toothless_left_up.png", False), ("Resend.png", True)], "Press on toothless until resend appears", timeout=10, last_activity_name="Head_toothless_left_up.png")
        print("[INFO] No specific post-collection action needed")
        return True
    
def check_color_and_tap_multi(device, patch_configs, action_desc, tolerance=4, patch_size=25, timeout=7.0, tap_count=2, last_activity_name=None, no_debugger=False):
    do_button_flags(device)
    
    # Handle different input formats
    if isinstance(patch_configs, str):
        # Single patch name (backward compatibility)
        patch_configs = [(patch_configs, False)]  # False = press
    elif isinstance(patch_configs, list) and all(isinstance(item, str) for item in patch_configs):
        # List of patch names (backward compatibility)
        patch_configs = [(name, False) for name in patch_configs]
    # If already list of tuples, use as-is
    
    # Validate all patches exist in tap_info, create missing ones with locate_and_press
    valid_patches = []
    for patch_name, verify_only in patch_configs:
        if patch_name not in tap_info:
            print(f"[INFO] No saved location for {patch_name} in tap_info. Attempting to find with template matching...")
            
            # Try to find the template and create the patch
            template_name = f"{patch_name}.png"
            result = locate_and_press(device, [(template_name, False)], f"Create patch for {patch_name}", 
                                    timeout=5, no_debugger=True)  # Changed to use tuple format with False = press
    
            if result:
                print(f"[SUCCESS] Created patch for {patch_name} in tap_info AND pressed it")
                return result  # Return immediately since we already pressed it
            else:
                print(f"[ERROR] Could not find template {template_name} to create patch for {patch_name}")
                continue
                
        # Double-check that patch now exists in tap_info
        if patch_name in tap_info:
            valid_patches.append((patch_name, verify_only))
            print(f"[DEBUG] Patch {patch_name} loaded successfully ({'verify only' if verify_only else 'press'})")
        else:
            print(f"[ERROR] Patch {patch_name} still not available after template search")

    if not valid_patches:
        print(f"No valid patches found from {patch_configs}")
        return False

    start_time = time.time()
    attempt_count = 0
    found = False
    
    while time.time() - start_time < timeout:
        attempt_count += 1
        
        # Time image acquisition (only once per attempt)
        img_start_time = time.time()
        img = get_screen_capture(device)
        img_end_time = time.time()
        img_acquisition_time = (img_end_time - img_start_time) * 1000
        print(f"[TIMING] Image acquisition took: {img_acquisition_time:.2f}ms")

        # Try all patches on the same image
        best_match = None
        best_confidence = float('inf')  # For color matching, lower distance is better
        
        processing_start_time = time.time()
        
        for patch_name, verify_only in valid_patches:
            center_x, center_y, saved_color = tap_info[patch_name]
            saved_color = np.array(saved_color)
            
            # Extract patch from current image
            patch = img[center_y-patch_size:center_y+patch_size+1, center_x-patch_size:center_x+patch_size+1]
            mean_color = patch.mean(axis=(0,1))
            dist = np.linalg.norm(mean_color - saved_color)
            
            print(f"[DEBUG] {patch_name} color distance: {dist:.3f}")
            
            # Keep track of best match (lowest distance that meets tolerance)
            if dist < tolerance and dist < best_confidence and np.isfinite(dist):
                best_confidence = dist
                best_match = (patch_name, center_x, center_y, dist, verify_only)
        
        processing_end_time = time.time()
        total_processing_time = (processing_end_time - processing_start_time) * 1000
        total_time_from_acquisition = (processing_end_time - img_start_time) * 1000
        
        print(f"[TIMING] Color matching took: {total_processing_time:.2f}ms")
        print(f"[TIMING] Total time (acquisition + processing): {total_time_from_acquisition:.2f}ms")
        
        print(f"[DEBUG] {action_desc} - Attempt {attempt_count}: Best distance = {best_confidence:.3f}, Tolerance = {tolerance}")

        if best_match:
            found = True
            patch_name, center_x, center_y, dist, verify_only = best_match
            
            if not verify_only:
                for _ in range(tap_count):
                    device.input_tap(center_x, center_y)
                    time.sleep(0.3)
                print(f"\033[92m✅ {action_desc} - PRESSED {patch_name} at ({center_x}, {center_y}) with distance {dist:.2f}\033[0m")
                # DO NOT RETURN HERE - continue the loop to keep pressing until timeout
            else:
                print(f"\033[92m✅ {action_desc} - VERIFIED {patch_name} at ({center_x}, {center_y}) with distance {dist:.2f}\033[0m")
                return patch_name  # Only return immediately for verification
        
        do_button_flags(device)
        time.sleep(0.1)
    
    print(f"{action_desc} - {'Found' if found else 'Not found'} after {timeout} seconds. Total attempts: {attempt_count}")
    
    # Return the last found patch name if any was pressed during the timeout
    if found and best_match:
        return best_match[0]  # Return the patch name that was found
    
    if not found:
        if no_debugger:
            return False
        elif debugger(device, patch_configs[0][0] if patch_configs else None, type="check_color_and_tap"):
            return True
        return False
    return found


def check_color_and_tap(
    device, tap_target, tolerance=4, patch_size=25, timeout=7.0, tap_count=2, last_activity_name=None, verify_instead_of_press=False, no_debugger=False
):
    start_time = time.time()
    while time.time() - start_time < timeout:
        img = get_screen_capture(device)
        if tap_target not in tap_info:
            print(f"[ERROR] No saved location for {tap_target} button in tap_info.")
            continue
        center_x, center_y, saved_color = tap_info[tap_target]
        saved_color = np.array(saved_color)
        patch = img[center_y-patch_size:center_y+patch_size+1, center_x-patch_size:center_x+patch_size+1]
        mean_color = patch.mean(axis=(0,1))
        dist = np.linalg.norm(mean_color - saved_color)
        print(f"[DEBUG] {tap_target} button color distance: {dist:.2f}")
        if dist < tolerance:
            print(f"[INFO] {tap_target} button detected by color match.")
            if not verify_instead_of_press:
                for _ in range(tap_count):
                    device.input_tap(center_x, center_y)
                    time.sleep(0.3)
            return True
        time.sleep(0.1)
    print(f"[INFO] {tap_target} button not detected within {timeout} seconds.")
    # Try reconnect and last activity
    if no_debugger:
        do_button_flags(device)
        return False
    elif debugger(device, last_activity_name):
        return True  # Recovery succeeded, treat as success

# Example reference colors (BGR)
COLLECT_TYPES = {
    "egg": np.array([  63.91156463, 173.5260771 , 125.96371882]),
    "fish": np.array([150.46485261, 123.73696145,  66.27210884]),
    "wood": np.array([ 87.94557823, 114.2585034 , 160.27437642]),
    "rune": np.array([104.42176871, 136.29478458,  34.46258503])
}

def classify_bag_patch(mean_color, tolerance=10):
    for name, ref_color in COLLECT_TYPES.items():
        if np.linalg.norm(mean_color - ref_color) < tolerance:
            return name
    return "unknown"

def collect_and_classify_bag(device, tolerance=4, patch_size=10, timeout=7.0):
    global last_collected
    
    # First, collect the rewards
    if not check_color_and_tap_multi(device, "Collect", "Collect toothless rewards", tolerance=tolerance, patch_size=patch_size, timeout=timeout):
        print("[ERROR] Could not collect rewards")
        return False
    
    # Get current image for classification
    img = get_screen_capture(device)
    
    # Classify what was collected using bag patch
    if "Bag" in tap_info:
        bag_x, bag_y, _ = tap_info["Bag"]
        bag_patch = img[bag_y-patch_size:bag_y+patch_size+1, bag_x-patch_size:bag_x+patch_size+1]
        bag_mean = bag_patch.mean(axis=(0,1))
        collected_type = classify_bag_patch(bag_mean)
        last_collected = collected_type
        print(f"[INFO] Detected collected type: {last_collected}")
    else:
        print("[WARN] Bag coordinates not found in tap_info. Cannot classify collection type.")
        last_collected = "unknown"
    
    # Single call to handle all post-collection scenarios
    found_template = check_color_and_tap_multi(device, [
        ("No_thanks", True),                    # Verify if found
        ("Release", True),                      # Verify if found  
        ("Head_toothless_left_up", True)        # Verify if found
    ], "Handle post-collection actions", tolerance=tolerance, timeout=10, no_debugger=True)
    
    # Handle different logic based on what was found
    if found_template == "No_thanks":
        print("[INFO] Buy egg popup detected - no egg collected")
        check_color_and_tap_multi(device, "No_thanks", "Close buy egg popup", tolerance=tolerance)
        # Navigate back to resend state
        check_color_and_tap_multi(device, [("Head_toothless_left_up", False), ("Resend", True)], "Navigate to resend state", tolerance=tolerance, timeout=10)
        return True
        
    elif found_template == "Release":
        print(f"[INFO] Egg detected (collected type: {last_collected}), starting release sequence")
        check_color_and_tap_multi(device, [("Release", False), ("Yes", True)], "Press Release egg", tolerance=tolerance, timeout=10)
        check_color_and_tap_multi(device, [("Yes", False), ("Yes_2", True)], "Confirm Release egg", tolerance=tolerance, timeout=10)
        check_color_and_tap_multi(device, [("Yes_2", False), ("Head_toothless_left_up", True)], "Close really popup", tolerance=tolerance, timeout=10)
        # Navigate back to resend state
        wait_for_patch_match(device, "Resend")
        return True

    elif found_template == "Head_toothless_left_up":
        print(f"[INFO] Toothless detected (collected type: {last_collected}), navigating to resend state")
        wait_for_patch_match(device, "Resend")
        return True
        
    else:
        print(f"[DEBUG] No specific post-collection UI detected (collected type: {last_collected}) - calling debugger")
        # This is an unexpected state - call debugger to handle recovery
        if debugger(device, "Collect", type="check_color_and_tap"):
            print("[INFO] Debugger resolved the issue, continuing")
            return True
        else:
            print("[ERROR] Debugger could not resolve post-collection state")
            return False

def wait_for_patch_match(device, target_name, tolerance=4, patch_size=10, timeout=1.5):
    global screen_center_x, screen_center_y  # Add this line
    """Tap center until the patch at target_name matches its saved color."""
    start_time = time.time()
    target_x, target_y, saved_color = tap_info[target_name]
    saved_color = np.array(saved_color)
    while time.time() - start_time < timeout:
        img = get_screen_capture(device)
        patch = img[target_y-patch_size:target_y+patch_size+1, target_x-patch_size:target_x+patch_size+1]
        mean_color = patch.mean(axis=(0,1))
        dist = np.linalg.norm(mean_color - saved_color)
        print(f"[DEBUG] {target_name} patch color distance: {dist:.2f}")
        if dist < tolerance:
            print(f"[INFO] {target_name} patch matched by color.")
            return True
        device.input_tap(screen_center_x, screen_center_y)  # Tap center of screen
        time.sleep(0.3)
    print(f"[INFO] {target_name} patch not matched within {timeout} seconds.")
    return False

def main():
    global screen_height, screen_width, screen_center_x, screen_center_y  # Add this line
    print("Make sure you are connected to the ADB, check `adb devices`!\n")
    # Starting adb daemon server
    os.system("adb start-server")
    time.sleep(1)

    # Connect to ADB
    client = AdbClient(host="127.0.0.1", port=5037)
    devices = client.devices()
    
    # Create an Termux:GUI connection
    if RUN_ON_MOBILE:
        connection = tg.Connection()

    # If no device is detected, open the developer options
    if len(devices) == 0:
        if RUN_ON_MOBILE:
            connection.toast("Please connect to ADB Wi-Fi IP from developer options", long = True)
            os.system("am start -a com.android.settings.APPLICATION_DEVELOPMENT_SETTINGS")
        print("No device found. Please connect to device using ADB!")
        return

    device = devices[0]

    print("Checking if the Rise of Berk app is running")

    # Check if Rise of Berk app is running
    Start_Rise_app(device)


    screen_height, screen_width, _ = get_screen_capture(device).shape
    screen_center_x = screen_width // 2
    screen_center_y = screen_height // 2
    # Display control overlay on mobile and create an thread to verify input
    if RUN_ON_MOBILE:
        play_pause_btn, exit_btn = display_overlay_on_android(screen_height, connection)
        watcher = threading.Thread(
            target=action_on_overlay_button_press, 
            args=(connection, play_pause_btn, exit_btn), 
            daemon=True
        )
        watcher.start()

    Initiate_bot_resend_sequence(device)

    print("\033[38;5;208m" + "="*50)
    print("🚀 STARTING MAIN BOT LOOP 🚀")
    print("="*50 + "\033[0m")
    while True:
        do_button_flags(device)
        check_color_and_tap_multi(device, [("Resend", False), ("1_new", True)], "Resend till 1_new bag", timeout=10)
        check_color_and_tap_multi(device, [("1_new", False), ("Start_Explore", True)], "1_new bag select", timeout=10)
        check_color_and_tap_multi(device, [("Start_Explore", False), ("Head_toothless_left_up", True)], "Start_Explore")
        check_color_and_tap_multi(device, [("Speed_up", False), ("Speedup_Verification", True)], "Speed_up the exploration free", timeout=5)
        check_color_and_tap_multi(device, [("Head_toothless_left_up", False), ("Bag", True)], "Press on toothless, which is back", timeout=10)
        wait_for_patch_match(device, "Bag")
        check_color_and_tap_multi(device, [("Bag", False), ("Collect", True)], "Bag")
        collect_and_classify_bag(device)


if __name__ == "__main__":
    main()