from ppadb.client import Client as AdbClient
import cv2
import numpy as np
import time
import configuration as c
import threading
import os

# Import Termux:GUI to diplay overlay if script is running on Android
if c.RUN_ON_MOBILE:
    import termuxgui as tg

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
    
    play_pause_btn = create_overlay_button(activity, "⏯️", rootLinear) 
    farm_btn = create_overlay_button(activity, "🧑‍🌾", rootLinear) 
    exit_btn = create_overlay_button(activity, "❌", rootLinear) 
    
    time.sleep(1)
            
    return play_pause_btn, farm_btn, exit_btn

# Set flags for next action when button press
pause_flag, farm_flag, exit_flag = [False] * 3
def action_on_overlay_button_press(connection, play_pause_id, farm_id, exit_id):
    global pause_flag, farm_flag, exit_flag
    for event in connection.events():
        if event and event.type == tg.Event.click and event.value["id"] == play_pause_id:
            pause_flag = not pause_flag
            if pause_flag:
                connection.toast("Bot is pausing")
            else:
                connection.toast("Bot starting")
        if event and event.type == tg.Event.click and event.value["id"] == farm_id:
            farm_flag = True and pause_flag
            if farm_flag:
                connection.toast("Starting energy farm")
            else:
                connection.toast("Please pause before farm")
        if event and event.type == tg.Event.click and event.value["id"] == exit_id:
            exit_flag = True
            connection.toast("Closing bot")

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

# Waits for the Aliexpress app to be opened on the device.
def wait_for_Rise_app(device):
    if check_app_in_foreground(device, c.TARGET_APP_PKG):
        print("Rise of Berk app is running")
        return True
    else:
        print("Please open Rise of Berk. Waiting 15 seconds.")
        time.sleep(15)
        if not check_app_in_foreground(device, c.TARGET_APP_PKG):
            print("Rise of Berk is not in focus. Exiting script")
            exit()
            

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
    wait_for_Rise_app(device)

    img = get_screen_capture(device)

    # Only try to merge objects with a similarity above this threshold
    height, width, _ = img.shape
    
    # Display control overlay on mobile and create an thread to verify input
    if c.RUN_ON_MOBILE:
        play_pause_id, farm_id, exit_id = display_overlay_on_android(height, connection)
        watcher = threading.Thread(
                            target=action_on_overlay_button_press, 
                            args=(connection, play_pause_id, farm_id, exit_id), 
                            daemon=True
                  )
        watcher.start()

    # Define the region of interest for duplicate findings
    # Top, bottom, left, right padding
    roi = int(c.ROI_TOP * height), int(c.ROI_BOTTOM * height), int(width * c.ROI_PADDING)

    # Generate ROI grid contours
    grid_contours = generate_grid_contours(img, roi, c.GRID_PADDING)

    # Remember the energy farm status
    farm_the_energy = c.AUTO_FARM_ENERGY

    while True:
        do_button_flags(img, device)
            
        img = get_screen_capture(device)

        extracted_imgs, count_blanks = extract_imgs_from_contours(img, grid_contours)

        grouped_items = group_similar_imgs(extracted_imgs, c.SIMILARITY_THRESHOLD)

        check_if_should_exit(device)
            
        swipe_elements(device, grid_contours, grouped_items, roi)
        
        check_if_space_left(count_blanks, grouped_items)

        # Check the energy left and matches
        if c.CHECK_ENERGY_LEVEL and len(grouped_items) <= c.MAX_GENERATOR_GROUP_NUMBERS:
            if (
                generate_objects(device, grid_contours, img) == False
                and len(grouped_items) == 0
            ):
                print("No group found.")
                if farm_the_energy:
                    print("Starting to farm energy.")
                    farm_energy(img, device)
                    print("Finish farming.")
                    farm_the_energy = False
                else:
                    print("No energy to farm. Exit.")
                    break

        debug_display_img(img, grid_contours, grouped_items, roi, extracted_imgs)

        check_if_should_exit(device)

        if c.AUTOMATIC_DELIVERY:
            try_to_delivery(device, img.shape)


if __name__ == "__main__":
    main()
