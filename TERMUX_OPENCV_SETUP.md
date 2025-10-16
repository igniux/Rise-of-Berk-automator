# OpenCV Setup Guide for Termux 📱

This guide will help you get OpenCV working on your Termux environment for the Rise of Berk bot.

## 🚀 Quick Setup (Recommended)

### Method 1: Automated Setup

```bash
# Make script executable and run
chmod +x quick_opencv_install.sh
./quick_opencv_install.sh
```

### Method 2: Manual Setup

```bash
# Update Termux
pkg update && pkg upgrade -y

# Install Python and basic dependencies
pkg install -y python python-pip numpy

# Try opencv-python-headless (fastest)
pip install --no-cache-dir opencv-python-headless
```

## 🧪 Test Your Installation

### Run Diagnostic Script

```bash
python opencv_diagnostic.py
```

### Quick Test

```python
# Test OpenCV
python -c "import cv2; print(f'OpenCV {cv2.__version__} works!')"

# Test your bot
python bot_run.py
```

## ❌ Common Issues & Solutions

### Issue 1: "ImportError: No module named cv2"

**Solution:**

```bash
pip install opencv-python-headless
```

If that fails:

```bash
pkg install opencv python
```

### Issue 2: "AttributeError: 'module' has no attribute 'imdecode'"

**Solution:** Your bot will automatically use the compatibility layer (`cv2_compat.py`). This is normal and expected!

### Issue 3: Installation takes 20+ minutes

**Why:** OpenCV has many dependencies and needs to compile native code.

**Solutions:**

1. **Use headless version** (recommended):

   ```bash
   pip install opencv-python-headless
   ```

2. **Use precompiled binaries**:

   ```bash
   pkg install opencv python
   ```

3. **Install during off-peak hours** when your internet is faster

4. **Use the compatibility layer** - your bot works without native OpenCV!

### Issue 4: "Building wheel for opencv-python failed"

**Solution:**

```bash
# Install build dependencies
pkg install -y clang cmake ninja

# Or use system packages instead
pkg install opencv python
```

### Issue 5: Memory/storage issues during install

**Solution:**

```bash
# Free up space
pkg clean

# Install with no cache
pip install --no-cache-dir opencv-python-headless
```

## 🔧 Performance Tips

### 1. Use the Right OpenCV Version

- **Best**: `opencv-python-headless` (no GUI dependencies)
- **Good**: System packages (`pkg install opencv`)
- **Fallback**: `cv2_compat.py` (always works)

### 2. Optimize Your Bot

Your bot automatically uses:

- Color matching (faster than template matching)
- Compatibility layer fallback
- Optimized screenshot handling

### 3. Network Optimization

If using wireless ADB:

```bash
# Use compression for ADB
adb shell settings put global stay_on_while_plugged_in 0
```

## 🛠️ Advanced Troubleshooting

### Check What's Installed

```bash
# List installed packages
pip list | grep opencv

# Check OpenCV version and features
python -c "import cv2; print(cv2.getBuildInformation())"
```

### Force Reinstall

```bash
# Remove all OpenCV packages
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless

# Reinstall fresh
pip install --no-cache-dir opencv-python-headless
```

### Build from Source (Last Resort)

```bash
# Install all build dependencies
pkg install -y clang cmake ninja python-dev numpy

# Set environment variables
export ENABLE_HEADLESS=1
export CV_DISABLE_OPTIMIZATION=1

# Build minimal OpenCV
pip install --no-cache-dir opencv-python-headless --no-binary opencv-python-headless
```

## 🎯 What Your Bot Actually Needs

Your bot only needs these OpenCV functions:

- `cv2.imdecode()` - decode screenshots
- `cv2.matchTemplate()` - find UI elements
- `cv2.cvtColor()` - color conversions
- `cv2.imread()` - load template images

**Good news:** The `cv2_compat.py` provides all of these! So your bot works even if OpenCV installation fails.

## 📊 Performance Comparison

| Method              | Speed      | Reliability | Setup Time  |
| ------------------- | ---------- | ----------- | ----------- |
| Native OpenCV       | ⭐⭐⭐⭐⭐ | ⭐⭐⭐      | 20+ minutes |
| opencv-headless     | ⭐⭐⭐⭐   | ⭐⭐⭐⭐    | 5 minutes   |
| System packages     | ⭐⭐⭐     | ⭐⭐⭐⭐⭐  | 2 minutes   |
| Compatibility layer | ⭐⭐⭐     | ⭐⭐⭐⭐⭐  | 0 minutes   |

## 🎉 Success Indicators

You'll know it's working when:

```
[INFO] Using system OpenCV
✅ OpenCV 4.x.x installed successfully
🎉 Your bot should now work!
```

Or if using compatibility layer:

```
[INFO] Using OpenCV compatibility layer
[SUCCESS] cv2_compat basic functionality test passed
```

## 🆘 Still Having Issues?

1. **Try the compatibility layer** - it's designed to work without OpenCV
2. **Use color matching** instead of template matching (your bot already does this)
3. **Check the diagnostic output** from `opencv_diagnostic.py`
4. **Ensure you have enough storage** (at least 1GB free)

## 📱 Termux-Specific Notes

- **Storage**: OpenCV needs ~500MB space
- **Memory**: Compilation needs ~1GB RAM
- **Network**: Download is ~200MB
- **Time**: Allow 20-30 minutes for full installation

Your bot is designed to work with or without native OpenCV, so don't worry if installation fails - the compatibility layer has you covered! 🚀
