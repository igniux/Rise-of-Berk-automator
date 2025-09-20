import os
import json 

# The name of the app, to check if it's running
TARGET_APP_PKG = "com.ludia.dragons/com.ludia.engine.application"

# If you run the script directly on mobile, set this to True to disable
# incompatible functions, like real-time image view, and configure for this
RUN_ON_MOBILE = False