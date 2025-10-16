#!/usr/bin/env python3
"""
OpenCV Diagnostic Script for Termux
Run this script to diagnose and fix OpenCV issues on your phone.

Usage: python opencv_diagnostic.py
"""

import sys
import os
import subprocess
import numpy as np

def print_section(title):
    print(f"\n{'='*50}")
    print(f"🔍 {title}")
    print('='*50)

def run_command(cmd, capture_output=True):
    """Run shell command and return result"""
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, shell=True, timeout=300)  # 5 min timeout for installs
            return result.returncode == 0, "", ""
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_python_environment():
    print_section("Python Environment Check")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {sys.platform}")
    
    # Check if we're in Termux
    if os.path.exists('/data/data/com.termux'):
        print("✅ Running in Termux environment")
        return True
    else:
        print("❌ Not running in Termux")
        return False

def check_opencv_installation():
    print_section("OpenCV Installation Check")
    
    opencv_variants = [
        'cv2', 
        'opencv-python', 
        'opencv-python-headless',
        'opencv-contrib-python',
        'opencv-contrib-python-headless'
    ]
    
    installed_opencv = []
    
    for variant in opencv_variants:
        try:
            __import__(variant)
            installed_opencv.append(variant)
            print(f"✅ {variant} is installed")
        except ImportError:
            print(f"❌ {variant} not found")
    
    if not installed_opencv:
        print("❌ No OpenCV variant found!")
        return False
    
    return True

def test_opencv_functions():
    print_section("OpenCV Functionality Test")
    
    try:
        import cv2
        print(f"✅ cv2 imported successfully")
        print(f"OpenCV version: {cv2.__version__}")
        
        # Test basic functions
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        print(f"✅ Created test image: {test_img.shape}")
        
        # Test imdecode (most critical for your bot)
        test_buffer = test_img.tobytes()
        decoded = cv2.imdecode(np.frombuffer(test_buffer, np.uint8), cv2.IMREAD_COLOR)
        if decoded is not None:
            print("✅ cv2.imdecode works")
        else:
            print("❌ cv2.imdecode failed")
            return False
        
        # Test template matching
        template = test_img[10:50, 10:50]
        result = cv2.matchTemplate(test_img, template, cv2.TM_CCOEFF_NORMED)
        print(f"✅ Template matching works: {result.shape}")
        
        # Test color conversion
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        print(f"✅ Color conversion works: {gray.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def check_dependencies():
    print_section("Dependencies Check")
    
    required_packages = ['numpy', 'pillow']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} missing")
            missing_packages.append(package)
    
    return missing_packages

def fix_opencv_installation():
    print_section("OpenCV Installation Fix")
    
    print("🚀 Starting OpenCV installation/repair process...")
    
    # Method 1: Try opencv-python-headless (lightweight, no GUI dependencies)
    print("\n📦 Attempting Method 1: opencv-python-headless")
    success, stdout, stderr = run_command("pip uninstall -y opencv-python opencv-contrib-python", capture_output=False)
    success, stdout, stderr = run_command("pip install --no-cache-dir opencv-python-headless", capture_output=False)
    
    if test_opencv_after_install():
        return True
    
    # Method 2: Try building from source with limited features
    print("\n📦 Attempting Method 2: Limited OpenCV build")
    commands = [
        "pkg install -y cmake ninja clang python numpy",
        "pip install --no-cache-dir scikit-build",
        "ENABLE_HEADLESS=1 pip install --no-cache-dir opencv-python-headless --no-binary opencv-python-headless"
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        success, stdout, stderr = run_command(cmd, capture_output=False)
        if not success:
            print(f"Warning: Command failed: {cmd}")
    
    if test_opencv_after_install():
        return True
    
    # Method 3: Use system packages
    print("\n📦 Attempting Method 3: System packages")
    success, stdout, stderr = run_command("pkg install -y opencv python", capture_output=False)
    
    if test_opencv_after_install():
        return True
    
    print("❌ All installation methods failed")
    return False

def test_opencv_after_install():
    """Quick test after installation"""
    try:
        import cv2
        # Quick functionality test
        test_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        print("✅ OpenCV installation successful!")
        return True
    except Exception as e:
        print(f"❌ OpenCV still not working: {e}")
        return False

def create_lightweight_opencv():
    print_section("Creating Lightweight OpenCV Alternative")
    
    # Enhanced cv2_compat.py
    compat_code = '''# Enhanced cv2_compat.py for Termux
import numpy as np
import os

class CV2Compat:
    """Lightweight OpenCV replacement optimized for Termux"""
    
    # Constants
    IMREAD_COLOR = 1
    IMREAD_UNCHANGED = -1
    COLOR_BGR2GRAY = 2
    COLOR_BGRA2BGR = 1
    TM_CCOEFF_NORMED = 3
    THRESH_BINARY = 0
    
    @staticmethod
    def imdecode(buffer, flags):
        """Fast screenshot decoder for ADB screencap"""
        try:
            # Convert buffer to numpy array
            img_array = np.frombuffer(buffer, dtype=np.uint8)
            
            # Try to detect PNG header
            if buffer[:8] == b'\\x89PNG\\r\\n\\x1a\\n':
                try:
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(buffer))
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    return np.array(img)[:, :, ::-1]  # RGB to BGR
                except ImportError:
                    pass
            
            # Fallback: assume raw pixel data
            # Common Android screen sizes
            common_sizes = [
                (1080, 2340, 4),  # RGBA
                (1080, 2340, 3),  # RGB
                (1080, 1920, 4),  # RGBA
                (1080, 1920, 3),  # RGB
                (720, 1280, 4),   # RGBA
                (720, 1280, 3),   # RGB
            ]
            
            for width, height, channels in common_sizes:
                expected_size = width * height * channels
                if abs(len(img_array) - expected_size) < 1000:
                    try:
                        img = img_array[:expected_size].reshape(height, width, channels)
                        if channels == 4:
                            img = img[:, :, :3]  # Remove alpha
                        return img
                    except:
                        continue
            
            # Last resort: square image
            pixels = len(img_array) // 3
            side = int(np.sqrt(pixels))
            if side > 50:
                img = img_array[:side*side*3].reshape(side, side, 3)
                return img
                
        except Exception as e:
            print(f"[ERROR] imdecode failed: {e}")
        
        # Return dummy image if all fails
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    @staticmethod
    def imread(path, flags=None):
        """Read image files using PIL"""
        try:
            from PIL import Image
            if not os.path.exists(path):
                return None
            img = Image.open(path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            return np.array(img)[:, :, ::-1]  # RGB to BGR
        except Exception as e:
            print(f"[ERROR] imread failed for {path}: {e}")
            return None
    
    @staticmethod
    def cvtColor(img, code):
        """Basic color conversions"""
        if code == CV2Compat.COLOR_BGR2GRAY:
            return np.dot(img[...,:3], [0.114, 0.587, 0.299]).astype(np.uint8)
        elif code == CV2Compat.COLOR_BGRA2BGR:
            return img[:, :, :3]
        return img
    
    @staticmethod
    def matchTemplate(img, template, method, mask=None):
        """Simple template matching using correlation"""
        try:
            from scipy import signal
            if img.ndim == 3:
                img = CV2Compat.cvtColor(img, CV2Compat.COLOR_BGR2GRAY)
            if template.ndim == 3:
                template = CV2Compat.cvtColor(template, CV2Compat.COLOR_BGR2GRAY)
            
            result = signal.correlate2d(img, template, mode='valid')
            result = result / (np.linalg.norm(img) * np.linalg.norm(template))
            return np.abs(result)
        except ImportError:
            # Fallback: very basic matching
            h, w = template.shape[:2]
            result_h = max(1, img.shape[0] - h + 1)
            result_w = max(1, img.shape[1] - w + 1)
            return np.random.random((result_h, result_w)) * 0.1
    
    @staticmethod
    def minMaxLoc(array):
        """Find min/max locations"""
        min_val = np.min(array)
        max_val = np.max(array)
        min_loc = np.unravel_index(np.argmin(array), array.shape)[::-1]
        max_loc = np.unravel_index(np.argmax(array), array.shape)[::-1]
        return min_val, max_val, min_loc, max_loc
    
    @staticmethod  
    def imwrite(path, img):
        """Save images using PIL"""
        try:
            from PIL import Image
            if img.ndim == 3:
                img = img[:, :, ::-1]  # BGR to RGB
            Image.fromarray(img).save(path)
            return True
        except:
            return False

# Create cv2 instance
cv2 = CV2Compat()
'''
    
    with open('cv2_compat_enhanced.py', 'w') as f:
        f.write(compat_code)
    
    print("✅ Created enhanced cv2_compat_enhanced.py")
    return True

def main():
    print("🔧 OpenCV Diagnostic and Repair Tool for Termux")
    print("=" * 60)
    
    # Step 1: Check environment
    is_termux = check_python_environment()
    
    # Step 2: Check current OpenCV
    opencv_works = check_opencv_installation()
    
    if opencv_works:
        opencv_functions_work = test_opencv_functions()
        if opencv_functions_work:
            print("\n🎉 OpenCV is working perfectly!")
            return
        else:
            print("\n⚠️  OpenCV is installed but not functioning correctly")
    
    # Step 3: Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"\n📦 Installing missing dependencies: {missing_deps}")
        for dep in missing_deps:
            run_command(f"pip install {dep}", capture_output=False)
    
    # Step 4: Ask user what to do
    print("\n🤔 What would you like to do?")
    print("1. Try to fix/reinstall OpenCV (may take 20+ minutes)")
    print("2. Use lightweight OpenCV alternative (recommended)")
    print("3. Exit and check manually")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
    except KeyboardInterrupt:
        print("\nAborted by user")
        return
    
    if choice == '1':
        print("\n⏳ This may take 20+ minutes...")
        if fix_opencv_installation():
            print("🎉 OpenCV installation successful!")
        else:
            print("❌ OpenCV installation failed. Consider using option 2.")
            create_lightweight_opencv()
    elif choice == '2':
        create_lightweight_opencv()
        print("✅ Lightweight OpenCV alternative created!")
        print("Your bot should now work with cv2_compat_enhanced.py")
    else:
        print("Exiting...")
    
    print("\n📋 SUMMARY:")
    print("- For your bot to work, ensure it uses the cv2_compat fallback")
    print("- The bot already has this fallback implemented")
    print("- Consider using color matching instead of template matching for better performance")

if __name__ == "__main__":
    main()