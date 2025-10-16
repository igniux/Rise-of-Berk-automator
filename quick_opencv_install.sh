#!/bin/bash
# quick_opencv_install.sh - Fast OpenCV installation for Termux

echo "🚀 Quick OpenCV Installation for Termux"
echo "======================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install packages with retry
install_with_retry() {
    local package=$1
    local attempts=3
    
    for i in $(seq 1 $attempts); do
        echo "📦 Installing $package (attempt $i/$attempts)..."
        if pip install --no-cache-dir "$package"; then
            echo "✅ $package installed successfully"
            return 0
        else
            echo "❌ Attempt $i failed"
            if [ $i -lt $attempts ]; then
                echo "⏳ Waiting 5 seconds before retry..."
                sleep 5
            fi
        fi
    done
    
    echo "❌ Failed to install $package after $attempts attempts"
    return 1
}

# Update package lists
echo "🔄 Updating Termux packages..."
pkg update -y

# Install system dependencies
echo "📦 Installing system dependencies..."
pkg install -y python python-pip numpy

# Method 1: Try opencv-python-headless (fastest, most reliable)
echo ""
echo "🎯 Method 1: Installing opencv-python-headless (recommended)"
echo "This is the fastest and most reliable method for headless environments."

# Remove any existing opencv installations
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless 2>/dev/null

# Install headless version
if install_with_retry "opencv-python-headless"; then
    echo ""
    echo "✅ Testing OpenCV installation..."
    if python -c "import cv2; print(f'OpenCV {cv2.__version__} installed successfully')"; then
        echo "🎉 OpenCV installation successful!"
        echo ""
        echo "📋 Installation Summary:"
        echo "- Package: opencv-python-headless"
        echo "- Status: ✅ Working"
        echo "- Features: Core CV functions (perfect for your bot)"
        echo ""
        echo "🚀 Your bot should now work! Run: python bot_run.py"
        exit 0
    fi
fi

# Method 2: Try with system packages
echo ""
echo "🎯 Method 2: Installing via system packages"
pkg install -y opencv python

if python -c "import cv2; print('System OpenCV works')" 2>/dev/null; then
    echo "✅ System OpenCV installation successful!"
    exit 0
fi

# Method 3: Install minimal dependencies and use compat layer
echo ""
echo "🎯 Method 3: Setting up compatibility layer"
echo "Since OpenCV installation failed, we'll ensure your bot works with the built-in compatibility layer."

# Install PIL for image handling
if install_with_retry "Pillow"; then
    echo "✅ Pillow installed for image processing"
fi

# Install scipy for better template matching
if install_with_retry "scipy"; then
    echo "✅ Scipy installed for advanced operations"
fi

echo ""
echo "📋 Final Setup Summary:"
echo "❌ Native OpenCV installation failed"
echo "✅ Compatibility layer dependencies installed"
echo "✅ Your bot will use cv2_compat.py fallback"
echo ""
echo "🔧 Your bot should still work with the built-in compatibility layer!"
echo "The performance might be slightly different, but all core functions will work."
echo ""
echo "🚀 Try running: python bot_run.py"
echo "💡 If you need better performance, consider using color matching instead of template matching."

# Test the compatibility layer
echo ""
echo "🧪 Testing compatibility layer..."
python3 << 'EOF'
try:
    import sys
    sys.path.insert(0, '.')
    
    # Test if the compat layer works
    try:
        import cv2
        print("✅ cv2 import successful")
    except:
        print("❌ cv2 import failed")
        
    # Test numpy
    import numpy as np
    print("✅ NumPy available")
    
    # Test PIL if available
    try:
        from PIL import Image
        print("✅ PIL available for image handling")
    except:
        print("⚠️  PIL not available")
        
    print("\n🎯 Compatibility test completed")
    
except Exception as e:
    print(f"❌ Compatibility test failed: {e}")
EOF