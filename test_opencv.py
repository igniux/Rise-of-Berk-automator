#!/usr/bin/env python3
"""
Simple OpenCV Test Script
Tests both native OpenCV and compatibility layer
"""

import sys
import numpy as np

def test_opencv_native():
    """Test native OpenCV installation"""
    print("🧪 Testing Native OpenCV...")
    try:
        import cv2
        print(f"✅ Native OpenCV {cv2.__version__} imported successfully")
        
        # Test basic functions
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test imdecode
        buffer = test_img.tobytes()
        decoded = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
        if decoded is not None:
            print("✅ cv2.imdecode works")
        else:
            print("❌ cv2.imdecode failed")
            return False
        
        # Test template matching
        template = test_img[10:50, 10:50]
        result = cv2.matchTemplate(test_img, template, cv2.TM_CCOEFF_NORMED)
        print(f"✅ Template matching works: max confidence = {np.max(result):.3f}")
        
        # Test color conversion
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        print(f"✅ Color conversion works: {gray.shape}")
        
        print("🎉 Native OpenCV is fully functional!")
        return True
        
    except ImportError:
        print("❌ Native OpenCV not available")
        return False
    except Exception as e:
        print(f"❌ Native OpenCV test failed: {e}")
        return False

def test_opencv_compat():
    """Test OpenCV compatibility layer"""
    print("\n🧪 Testing OpenCV Compatibility Layer...")
    try:
        # Import the compatibility layer
        sys.path.insert(0, '.')
        from cv2_compat import cv2
        
        print("✅ cv2_compat imported successfully")
        
        # Test basic functions
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test imdecode
        buffer = test_img.tobytes()
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decoded is not None:
            print(f"✅ cv2_compat.imdecode works: {decoded.shape}")
        else:
            print("❌ cv2_compat.imdecode failed")
            return False
        
        # Test template matching
        template = test_img[10:50, 10:50]
        result = cv2.matchTemplate(test_img, template, cv2.TM_CCOEFF_NORMED)
        print(f"✅ cv2_compat template matching works: {result.shape}")
        
        # Test color conversion
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        print(f"✅ cv2_compat color conversion works: {gray.shape}")
        
        print("🎉 OpenCV compatibility layer is fully functional!")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility layer test failed: {e}")
        return False

def test_bot_requirements():
    """Test specific requirements for the bot"""
    print("\n🤖 Testing Bot Requirements...")
    
    # Test numpy
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} available")
    except ImportError:
        print("❌ NumPy not available - REQUIRED for bot")
        return False
    
    # Test PIL (optional but helpful)
    try:
        from PIL import Image
        print(f"✅ PIL available")
    except ImportError:
        print("⚠️  PIL not available - will impact image loading")
    
    # Test ppadb
    try:
        from ppadb.client import Client as AdbClient
        print("✅ ppadb available for ADB communication")
    except ImportError:
        print("❌ ppadb not available - REQUIRED for bot")
        return False
    
    # Test JSON
    try:
        import json
        print("✅ JSON available")
    except ImportError:
        print("❌ JSON not available - REQUIRED for bot")
        return False
    
    print("✅ All bot requirements satisfied!")
    return True

def test_adb_screenshot():
    """Test ADB screenshot capability"""
    print("\n📱 Testing ADB Screenshot...")
    
    try:
        from ppadb.client import Client as AdbClient
        
        # Connect to ADB
        client = AdbClient(host="127.0.0.1", port=5037)
        devices = client.devices()
        
        if len(devices) == 0:
            print("❌ No ADB devices connected")
            print("💡 Connect your device with: adb connect <ip>:5555")
            return False
        
        device = devices[0]
        print(f"✅ Connected to device: {device}")
        
        # Test screenshot
        print("📷 Taking test screenshot...")
        result = device.screencap()
        
        if result and len(result) > 1000:
            print(f"✅ Screenshot captured: {len(result)} bytes")
            
            # Test with compatibility layer
            try:
                from cv2_compat import cv2
                img = cv2.imdecode(result, cv2.IMREAD_COLOR)
                if img is not None:
                    print(f"✅ Screenshot decoded successfully: {img.shape}")
                    return True
                else:
                    print("❌ Screenshot decode failed")
                    return False
            except Exception as e:
                print(f"❌ Screenshot decode error: {e}")
                return False
        else:
            print("❌ Screenshot capture failed")
            return False
            
    except Exception as e:
        print(f"❌ ADB test failed: {e}")
        return False

def main():
    print("🔬 OpenCV and Bot Functionality Test")
    print("=" * 50)
    
    # Test native OpenCV
    native_works = test_opencv_native()
    
    # Test compatibility layer
    compat_works = test_opencv_compat()
    
    # Test bot requirements
    requirements_ok = test_bot_requirements()
    
    # Test ADB if possible
    adb_works = test_adb_screenshot()
    
    # Summary
    print("\n📋 TEST SUMMARY")
    print("=" * 30)
    print(f"Native OpenCV:     {'✅ PASS' if native_works else '❌ FAIL'}")
    print(f"Compatibility:     {'✅ PASS' if compat_works else '❌ FAIL'}")
    print(f"Bot Requirements:  {'✅ PASS' if requirements_ok else '❌ FAIL'}")
    print(f"ADB Screenshot:    {'✅ PASS' if adb_works else '❌ FAIL (device not connected)'}")
    
    print("\n🎯 RECOMMENDATION:")
    if native_works:
        print("🚀 Use native OpenCV - best performance!")
    elif compat_works and requirements_ok:
        print("🚀 Use compatibility layer - should work fine!")
        print("Your bot will automatically fall back to cv2_compat.py")
    else:
        print("❌ Bot may not work properly. Check the setup guide.")
    
    if not adb_works:
        print("\n📱 ADB Setup Needed:")
        print("1. Enable Developer Options on your phone")
        print("2. Enable USB Debugging")
        print("3. Connect with: adb connect <phone_ip>:5555")
    
    return native_works or compat_works

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)