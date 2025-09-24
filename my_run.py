from ppadb.client import Client as AdbClient
import cv2
import numpy as np
import time
import configuration as c
import threading
import os
import datetime  # Add this import at the top if not present
import json

# Import Termux:GUI to diplay overlay if script is running on Android
if c.RUN_ON_MOBILE:
    import termuxgui as tg

# Load tap info
with open("tap_info.json", "r") as f:
    tap_info = json.load(f)

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
    activity.setposition(9999, (c.DEL_TOP / 4.5) * height)
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
def do_button_flags(img, device):
    global pause_flag, exit_flag
    print(f"[DEBUG] Checking flags: pause_flag={pause_flag}, exit_flag={exit_flag}")
    if not check_app_in_foreground(device, c.TARGET_APP_PKG):
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
    result = device.screencap()
    img = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
    return img

# Checks if the current app running on the device is Rise of Berk and the screen is on.
def check_app_in_foreground(device, target):
    result = device.shell("dumpsys window | grep -E 'mCurrentFocus'")
    if target in result or 'com.termux.gui' in result:
        return True

    return False

def debugger(device, last_activity_name=None, type="check_color_and_tap"):
    if locate_and_press(device, "X.png", "Close popup"):
        print("X button found and clicked.")
        return True
    if check_color_and_tap(device, "Reconnect"):
        print("Reconnect button found and clicked. Waiting for reconnect button to clear")
        last_seen = time.time()
        while True:
            if check_color_and_tap(device, "Reconnect", timeout=1.0):
                print("[INFO] Reconnect detected and tapped. Resetting timer.")
                last_seen = time.time()
            else:
                if time.time() - last_seen >= 10:
                    print("[INFO] No Reconnect detected for 10 seconds. Proceeding.")
                    return True
            time.sleep(0.5)
    if last_activity_name:
        print(f"Trying last activity: {last_activity_name}")
        if type == "locate_and_press":
            if locate_and_press(device, last_activity_name, f"Try {last_activity_name}"):
                print(f"{last_activity_name} found and pressed.")
                return True
        elif type == "check_color_and_tap":
            if check_color_and_tap(device, last_activity_name):
                print(f"{last_activity_name} found and pressed.")
                return True
    fatal_error("Unidentified problem", get_screen_capture(device), device)
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
    print(f"Error screen saved to {filepath}")
    print(msg)
    # Terminate Rise of Berk app if device is not None
    if device is not None:
        device.shell(f"am force-stop {c.TARGET_APP_PKG}")
    # Do not exit, just return to continue Python code
    return

def Start_Rise_app(device):
    if check_app_in_foreground(device, c.TARGET_APP_PKG):
        print("Rise of Berk app is running")
        return True
    else:
        print("Please open Rise of Berk. Waiting 25 seconds.")
        print("Attempting to start Rise of Berk app")
        device.shell(f"monkey -p {c.TARGET_APP_PKG} -c android.intent.category.LAUNCHER 1")
        time.sleep(15)
        if not check_app_in_foreground(device, c.TARGET_APP_PKG):
            fatal_error("Failed to start Rise of Berk app. Exiting script.", get_screen_capture(device))
            return False 
        return True

def locate_and_press(device, template_name, action_desc, threshold=0.8, verify_instead_of_press=False, timeout=2.0, last_activity_name=None, no_debugger=False):
    do_button_flags(device)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icons_dir = os.path.join(script_dir, "icons")
    template_path = os.path.join(icons_dir, template_name)
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)

    if template is None:
        print(f"Template image {template_name} not found in icons folder.")
        return False

    # Handle alpha channel (for transparent background templates)
    if template.shape[2] == 4:
        alpha_channel = template[:, :, 3]
        mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)[1]
        template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)
    else:
        mask = None

    start_time = time.time()
    while time.time() - start_time < timeout:
        img = get_screen_capture(device)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED, mask=mask)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            h, w = template.shape[:2]
            top_left = max_loc
            center_x = top_left[0] + w // 2
            center_y = top_left[1] + h // 2

            # Remove .png extension for tap_info key
            key_name = template_name.replace('.png', '')

            # To save/update a location:
            patch = img[center_y-10:center_y+10+1, center_x-10:center_x+10+1]
            color_code = patch.mean(axis=(0,1))
            tap_info[key_name] = [center_x, center_y, color_code.tolist()]
            with open("tap_info.json", "w") as f:
                json.dump(tap_info, f, indent=2)
            if not verify_instead_of_press:
                device.input_tap(center_x, center_y)
                print(f"{action_desc} - Pressed at ({center_x}, {center_y}) with confidence {max_val:.2f}")
            return True
        do_button_flags(device)
        time.sleep(0.1)
    print(f"{action_desc} - Not found after {timeout} seconds.")
    # Try reconnect and last activity
    if no_debugger:
        return False
    if debugger(device, last_activity_name, type="locate_and_press"):
        return True  # Recovery succeeded
    return False    # Only if debugger could not recover

def swipe_up(device, img, duration_ms=500):
    """
    Swipe from bottom to top at the horizontal center of the screen.
    duration_ms: swipe duration in milliseconds
    """
    height, width = img.shape[:2]
    x = width // 2
    y_start = int(height * 0.85)  # Start near bottom
    y_end = int(height * 0.3)    # End near top
    device.shell(f"input swipe {x} {y_start} {x} {y_end} {duration_ms}")
    print(f"Swiped from ({x}, {y_start}) to ({x}, {y_end})")
    time.sleep(0.1)  # Wait for swipe action to complete
    
def Initiate_bot_resend_sequence(device):
    print("Initiating bot sequence")
    img = get_screen_capture(device)
    
    locate_and_press(device, "X.png", "Close any Limited Offers", timeout=15, last_activity_name="X.png")
    locate_and_press(device, "Head_toothless_left_up.png", "Locate and press Head toothless left up", timeout=2, last_activity_name="Head_toothless_left_up.png")
    locate_and_press(device, "Night_Fury.png", "Verify that Night Fury is selected", verify_instead_of_press=True, timeout=2, last_activity_name="Head_toothless_left_up.png")
    locate_and_press(device, "Search_button.png", "Locate and press Search button", timeout=2, last_activity_name="Search_button.png")
    max_swipes = 3
    for attempt in range(max_swipes):
        if locate_and_press(device, "Terrible_Terror_Search_Selection.png", "Locate and press Terrible terror in the list", timeout=2, last_activity_name="Terrible_Terror_Search_Selection.png"):
            break
        swipe_up(device, img)
    else:
        print("[ERROR] Terrible Terror not found after swiping. Aborting or handling error.")
        # Optionally call debugger or handle as needed
    locate_and_press(device, "1_bag_search.png", "select 1 bag search option", timeout=2, last_activity_name="1_bag_search.png")
    locate_and_press(device, "Start_Explore.png", "Locate and press Start Explore button", timeout=2, last_activity_name="Start_Explore.png")
    locate_and_press(device, "Speed_up.png", "Speed up the exploration free", timeout=2, last_activity_name="Speed_up.png")

    with open("tap_info.json", "w") as f:
        json.dump(tap_info, f, indent=2)  
    device.input_tap(screen_center_x, screen_center_y)
    locate_and_press(device, "Bag.png", "Open Bag", timeout=5, last_activity_name="Bag.png")
    locate_and_press(device, "Collect.png", "Collect toothless rewards", timeout=5, last_activity_name="Collect.png")
    if locate_and_press(device, "No_thanks.png", "Close Buy egg popup", no_debugger=True, timeout=2):
        return True
    if locate_and_press(device, "Release.png", "Release egg", no_debugger=True, timeout=2):
        locate_and_press(device, "Yes.png", "Confirm Release egg", timeout=2, last_activity_name="Yes.png")
        locate_and_press(device, "Yes_2.png", "Close really popup", timeout=2, last_activity_name="Yes_2.png")
        locate_and_press(device, "X.png", "Popup", no_debugger=True, timeout=2)
    if locate_and_press(device, "Head_toothless_left_up.png", "Locate Head toothless left up", verify_instead_of_press=True, timeout=2):
        device.input_tap(screen_center_x, screen_center_y)
        while not locate_and_press(device, "Resend.png", "Resend toothless button exists", verify_instead_of_press=True, timeout=0.5, last_activity_name="Head_toothless_left_up.png"):
            device.input_tap(screen_center_x, screen_center_y)
            time.sleep(0.3)  # Adjust delay as needed
        return True

def check_color_and_tap(
    device, tap_target, tolerance=4, patch_size=10, timeout=2.0, tap_count=2, last_activity_name=None, verify_instead_of_press=False
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
    if debugger(device, last_activity_name):
        return True  # Recovery succeeded, treat as success
    do_button_flags(device)
    return False    # Only if debugger could not recover

# Example reference colors (BGR)
COLLECT_TYPES = {
    "egg": np.array([ 63.91156463, 173.5260771 , 125.96371882]),
    "fish": np.array([150.46485261, 123.73696145,  66.27210884]),
    "wood": np.array([ 87.94557823, 114.2585034 , 160.27437642]),
    "rune": np.array([104.42176871, 136.29478458,  34.46258503])
}

def classify_bag_patch(mean_color, tolerance=10):
    for name, ref_color in COLLECT_TYPES.items():
        if np.linalg.norm(mean_color - ref_color) < tolerance:
            return name
    return "unknown"

def collect_and_classify_bag(device, tolerance=4, patch_size=10, timeout=2.0, tap_count=2):
    global last_collected
    start_time = time.time()
    found = False
    img = None
    # 1. Try to match and tap Collect, and use that img for classification
    while time.time() - start_time < timeout:
        img = get_screen_capture(device)
        if "Collect" not in tap_info:
            print(f"[ERROR] No saved location for Collect button in tap_info.")
            break
        center_x, center_y, saved_color = tap_info["Collect"]
        saved_color = np.array(saved_color)
        patch = img[center_y-patch_size:center_y+patch_size+1, center_x-patch_size:center_x+patch_size+1]
        mean_color = patch.mean(axis=(0,1))
        dist = np.linalg.norm(mean_color - saved_color)
        print(f"[DEBUG] Collect button color distance: {dist:.2f}")
        if dist < tolerance:
            print(f"[INFO] Collect button detected by color match.")
            for _ in range(tap_count):
                device.input_tap(center_x, center_y)
                time.sleep(0.3)
            found = True
            break
        time.sleep(0.1)
    if not found:
        print(f"[INFO] Collect button not detected within {timeout} seconds.")
        do_button_flags(device)
        debugger(device, last_activity_name="Collect")

    # 2. Classify bag patch using the same img
    if "Bag" in tap_info:
        bag_x, bag_y, _ = tap_info["Bag"]
        bag_patch = img[bag_y-patch_size:bag_y+patch_size+1, bag_x-patch_size:bag_x+patch_size+1]
        bag_mean = bag_patch.mean(axis=(0,1))
        collected_type = classify_bag_patch(bag_mean)
        last_collected = collected_type
        print(f"[INFO] Detected collected type: {last_collected}")
    else:
        print("[WARN] Bag coordinates not found in tap_info.")

    if last_collected == "egg":
        check_color_and_tap(device, "Release", tolerance, patch_size, timeout, tap_count)
        check_color_and_tap(device, "Yes", tolerance, patch_size, timeout, tap_count)
        check_color_and_tap(device, "Yes_2", tolerance, patch_size, timeout, tap_count)
        check_color_and_tap(device, "Head_toothless_left_up", tolerance, patch_size, timeout, tap_count, verify_instead_of_press=True)
        max_attempts = 10
        attempts = 0
        while not check_color_and_tap(device, "Resend", tolerance, patch_size, timeout, tap_count, verify_instead_of_press=True):
            device.input_tap(center_x, center_y)
            time.sleep(0.3)
            attempts += 1
            if attempts >= max_attempts:
                print("Resend not detected after multiple attempts. Exiting loop.")
                break
    else:
        max_attempts = 10
        attempts = 0
        while not check_color_and_tap(device, "Head_to_toothless_left_up", tolerance, patch_size, timeout, tap_count, verify_instead_of_press=True):
            check_color_and_tap(device, "No_thanks", tolerance, patch_size, timeout=1.0, tap_count=2)
            attempts += 1
            if attempts >= max_attempts:
                print("Head_to_toothless_left_up not detected after multiple attempts. Exiting loop.")
                debugger(device, last_activity_name="Head_to_toothless_left_up")
                break

def wait_for_patch_match(device, target_name, tolerance=4, patch_size=10, timeout=1.5):
    """Tap center until the patch at target_name matches its saved color."""
    start_time = time.time()
    if target_name not in tap_info:
        print(f"[ERROR] No saved location for {target_name} in tap_info.")
        return False
    center_x = screen_center_x
    center_y = screen_center_y
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
        device.input_tap(center_x, center_y)
        time.sleep(0.3)
    print(f"[INFO] {target_name} patch not matched within {timeout} seconds.")
    return False

screen_height = None
screen_width = None
screen_center_x = None
screen_center_y = None

def main():
    print("Make sure you are connected to the ADB, check `adb devices`!\n")
    # Starting adb daemon server
    os.system("adb start-server")
    time.sleep(1)

    # Connect to ADB
    client = AdbClient(host="127.0.0.1", port=5037)
    devices = client.devices()
    
    # Create an Termux:GUI connection
    if c.RUN_ON_MOBILE:
        connection = tg.Connection()

    # If no device is detected, open the developer options
    if len(devices) == 0:
        if c.RUN_ON_MOBILE:
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
    if c.RUN_ON_MOBILE:
        play_pause_btn, exit_btn = display_overlay_on_android(screen_height, connection)
        watcher = threading.Thread(
            target=action_on_overlay_button_press, 
            args=(connection, play_pause_btn, exit_btn), 
            daemon=True
        )
        watcher.start()

    Initiate_bot_resend_sequence(device)

    while True:
        do_button_flags(device)
        check_color_and_tap(device, "Resend")
        check_color_and_tap(device, "1_bag_search")
        check_color_and_tap(device, "Start_Explore")
        check_color_and_tap(device, "Speed_up")
        wait_for_patch_match(device, "Bag")
        check_color_and_tap(device, "Bag")
        collect_and_classify_bag(device)


if __name__ == "__main__":
    main()
