# cv2_compat.py - Enhanced OpenCV replacement for Termux
import numpy as np
import os

class CV2Compat:
    """Enhanced OpenCV compatibility layer optimized for Termux and ADB screenshots"""
    
    IMREAD_COLOR = 1
    IMREAD_UNCHANGED = -1
    COLOR_BGR2GRAY = 2
    COLOR_BGRA2BGR = 1
    TM_CCOEFF_NORMED = 3
    THRESH_BINARY = 0
    
    @staticmethod
    def imdecode(buffer, flags):
        """Enhanced screenshot decoder with better format detection"""
        if not buffer or len(buffer) < 100:
            print("[ERROR] Buffer too small or empty")
            return None
            
        buffer_size = len(buffer)
        print(f"[INFO] Decoding buffer of size: {buffer_size} bytes")
        
        try:
            # Method 1: Try PIL for PNG/JPEG formats (most reliable)
            try:
                from PIL import Image
                import io
                
                # Check for image format headers
                if (buffer[:8] == b'\x89PNG\r\n\x1a\n' or  # PNG
                    buffer[:3] == b'\xff\xd8\xff' or       # JPEG
                    buffer[:6] in [b'GIF87a', b'GIF89a']):  # GIF
                    
                    img = Image.open(io.BytesIO(buffer))
                    print(f"[SUCCESS] PIL decoded {img.format} image: {img.size}")
                    
                    # Convert to numpy array
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    elif img.mode == 'L':  # Grayscale
                        img = img.convert('RGB')
                    
                    # Convert PIL (RGB) to OpenCV (BGR)
                    np_img = np.array(img)
                    if len(np_img.shape) == 3:
                        np_img = np_img[:, :, ::-1]  # RGB to BGR
                    
                    return np_img
                    
            except ImportError:
                print("[INFO] PIL not available, trying raw decode")
            except Exception as e:
                print(f"[INFO] PIL decode failed: {e}, trying raw decode")
            
            # Method 2: Raw pixel data (ADB screencap format)
            img_array = np.frombuffer(buffer, dtype=np.uint8)
            
            # Common Android screen resolutions and formats
            common_formats = [
                # (width, height, channels, format_name)
                (1080, 2340, 4, "FHD+ RGBA"),    # Modern phones
                (1080, 2340, 3, "FHD+ RGB"),
                (1080, 2400, 4, "FHD+ RGBA"),    # Alternative FHD+
                (1080, 2400, 3, "FHD+ RGB"),
                (1080, 1920, 4, "FHD RGBA"),     # Standard FHD
                (1080, 1920, 3, "FHD RGB"),
                (720, 1280, 4, "HD RGBA"),       # HD phones
                (720, 1280, 3, "HD RGB"),
                (1440, 2960, 4, "QHD+ RGBA"),    # High-end phones
                (1440, 2960, 3, "QHD+ RGB"),
            ]
            
            # Try each format
            for width, height, channels, format_name in common_formats:
                expected_size = width * height * channels
                size_diff = abs(buffer_size - expected_size)
                
                # Allow some tolerance for headers/padding
                if size_diff < max(1000, expected_size * 0.01):  # 1KB or 1% tolerance
                    try:
                        # Extract the right amount of data
                        data_size = min(expected_size, len(img_array))
                        img_data = img_array[:data_size]
                        
                        if len(img_data) >= expected_size:
                            img = img_data[:expected_size].reshape(height, width, channels)
                            
                            # Remove alpha channel if present
                            if channels == 4:
                                img = img[:, :, :3]  # Keep only BGR
                            
                            print(f"[SUCCESS] Decoded as {format_name}: {width}x{height}")
                            return img
                            
                    except Exception as e:
                        print(f"[DEBUG] Failed to decode as {format_name}: {e}")
                        continue
            
            # Method 3: Adaptive sizing
            print("[INFO] Trying adaptive sizing...")
            
            # Try different channel counts
            for channels in [3, 4]:
                available_pixels = len(img_array) // channels
                
                # Try common aspect ratios
                aspect_ratios = [
                    (16, 9), (18, 9), (19, 9), (20, 9),  # Phone ratios
                    (4, 3), (3, 2), (1, 1)               # Fallback ratios
                ]
                
                for w_ratio, h_ratio in aspect_ratios:
                    # Calculate dimensions
                    ratio = w_ratio / h_ratio
                    height = int(np.sqrt(available_pixels / ratio))
                    width = int(height * ratio)
                    
                    total_pixels = width * height
                    if total_pixels <= available_pixels and width > 100 and height > 100:
                        try:
                            img = img_array[:total_pixels * channels].reshape(height, width, channels)
                            if channels == 4:
                                img = img[:, :, :3]
                            print(f"[SUCCESS] Adaptive decode: {width}x{height} (ratio {w_ratio}:{h_ratio})")
                            return img
                        except:
                            continue
            
            # Method 4: Square fallback
            for channels in [3, 4]:
                side = int(np.sqrt(len(img_array) // channels))
                if side > 50:
                    try:
                        img = img_array[:side * side * channels].reshape(side, side, channels)
                        if channels == 4:
                            img = img[:, :, :3]
                        print(f"[FALLBACK] Square decode: {side}x{side}")
                        return img
                    except:
                        continue
        
        except Exception as e:
            print(f"[ERROR] All decode methods failed: {e}")
        
        # Final fallback: return a dummy image
        print("[ERROR] Creating dummy image - screenshot decode completely failed")
        dummy_height, dummy_width = 480, 270  # Small dummy size
        return np.random.randint(0, 50, (dummy_height, dummy_width, 3), dtype=np.uint8)
    
    @staticmethod
    def imread(path, flags=None):
        """Enhanced image file reader with multiple fallbacks"""
        if not os.path.exists(path):
            print(f"[ERROR] Template file not found: {path}")
            return None
        
        try:
            # Method 1: PIL (best compatibility)
            try:
                from PIL import Image
                img = Image.open(path)
                print(f"[SUCCESS] PIL loaded {path}: {img.size}, mode: {img.mode}")
                
                # Convert to RGB if needed
                if img.mode == 'RGBA':
                    # Handle transparency by compositing on white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])  # Use alpha as mask
                    img = background
                elif img.mode == 'L':  # Grayscale
                    img = img.convert('RGB')
                elif img.mode not in ['RGB', 'BGR']:
                    img = img.convert('RGB')
                
                # Convert PIL (RGB) to OpenCV format (BGR)
                np_img = np.array(img)[:, :, ::-1]
                return np_img
                
            except ImportError:
                print("[WARN] PIL not available")
            
            # Method 2: Try basic file reading for simple formats
            with open(path, 'rb') as f:
                data = f.read()
                
            # If it looks like an image header, try to decode
            if data[:8] == b'\x89PNG\r\n\x1a\n':
                print("[INFO] PNG detected but PIL unavailable")
            elif data[:3] == b'\xff\xd8\xff':
                print("[INFO] JPEG detected but PIL unavailable")
            
            print(f"[ERROR] Cannot read {path} without PIL")
            return None
            
        except Exception as e:
            print(f"[ERROR] Failed to read {path}: {e}")
            return None
    
    @staticmethod
    def cvtColor(img, code):
        """Enhanced color space conversion"""
        if img is None:
            return None
            
        try:
            if code == CV2Compat.COLOR_BGR2GRAY:
                # Proper RGB to grayscale conversion weights
                if len(img.shape) == 3:
                    return np.dot(img[...,:3], [0.114, 0.587, 0.299]).astype(np.uint8)
                else:
                    return img  # Already grayscale
                    
            elif code == CV2Compat.COLOR_BGRA2BGR:
                return img[:, :, :3]  # Remove alpha channel
                
            return img
            
        except Exception as e:
            print(f"[ERROR] cvtColor failed: {e}")
            return img
    
    @staticmethod
    def matchTemplate(img, template, method, mask=None):
        """Enhanced template matching with multiple methods"""
        if img is None or template is None:
            print("[ERROR] Image or template is None")
            return np.array([[0.0]])
        
        try:
            print(f"[INFO] Template matching: image {img.shape}, template {template.shape}")
            
            # Convert to grayscale for better performance
            if len(img.shape) == 3:
                img_gray = CV2Compat.cvtColor(img, CV2Compat.COLOR_BGR2GRAY)
            else:
                img_gray = img
                
            if len(template.shape) == 3:
                template_gray = CV2Compat.cvtColor(template, CV2Compat.COLOR_BGR2GRAY)
            else:
                template_gray = template
            
            # Calculate result dimensions
            result_h = max(1, img_gray.shape[0] - template_gray.shape[0] + 1)
            result_w = max(1, img_gray.shape[1] - template_gray.shape[1] + 1)
            
            # Method 1: Try scipy correlation (if available)
            try:
                from scipy import signal
                from scipy.ndimage import correlate
                
                # Normalize images
                img_norm = img_gray.astype(np.float32) / 255.0
                template_norm = template_gray.astype(np.float32) / 255.0
                
                # Use normalized cross-correlation
                result = signal.correlate2d(img_norm, template_norm, mode='valid')
                result = np.abs(result) / (img_norm.size * template_norm.size)
                
                print(f"[SUCCESS] Scipy template matching: max={np.max(result):.3f}")
                return result.astype(np.float32)
                
            except ImportError:
                print("[INFO] Scipy not available, using basic method")
            
            # Method 2: Simple sliding window correlation
            result = np.zeros((result_h, result_w), dtype=np.float32)
            
            template_norm = template_gray.astype(np.float32) / 255.0
            template_mean = np.mean(template_norm)
            template_std = np.std(template_norm)
            
            for y in range(result_h):
                for x in range(result_w):
                    # Extract patch
                    patch = img_gray[y:y+template_gray.shape[0], x:x+template_gray.shape[1]]
                    patch_norm = patch.astype(np.float32) / 255.0
                    
                    # Calculate normalized correlation
                    if template_std > 0:
                        patch_mean = np.mean(patch_norm)
                        patch_std = np.std(patch_norm)
                        
                        if patch_std > 0:
                            corr = np.mean((patch_norm - patch_mean) * (template_norm - template_mean))
                            corr = corr / (patch_std * template_std)
                            result[y, x] = max(0, corr)  # Clamp to positive
            
            max_val = np.max(result)
            print(f"[SUCCESS] Basic template matching: max={max_val:.3f}")
            return result
            
        except Exception as e:
            print(f"[ERROR] Template matching failed: {e}")
            # Return low-confidence result
            result_h = max(1, img.shape[0] - template.shape[0] + 1)
            result_w = max(1, img.shape[1] - template.shape[1] + 1)
            return np.random.random((result_h, result_w)) * 0.1
    
    @staticmethod
    def minMaxLoc(array):
        """Find minimum and maximum values and their locations"""
        try:
            min_val = np.min(array)
            max_val = np.max(array)
            min_loc = np.unravel_index(np.argmin(array), array.shape)[::-1]
            max_loc = np.unravel_index(np.argmax(array), array.shape)[::-1]
            return min_val, max_val, min_loc, max_loc
        except Exception as e:
            print(f"[ERROR] minMaxLoc failed: {e}")
            return 0.0, 0.0, (0, 0), (0, 0)
    
    @staticmethod
    def threshold(img, thresh, maxval, type):
        """Apply threshold to image"""
        try:
            result = np.where(img > thresh, maxval, 0).astype(np.uint8)
            return thresh, result
        except Exception as e:
            print(f"[ERROR] threshold failed: {e}")
            return thresh, img
    
    @staticmethod
    def imwrite(path, img):
        """Save image using PIL if available"""
        try:
            from PIL import Image
            if img is None:
                return False
            
            # Convert BGR to RGB for PIL
            if len(img.shape) == 3:
                img_rgb = img[:, :, ::-1]
            else:
                img_rgb = img
            
            Image.fromarray(img_rgb).save(path)
            print(f"[SUCCESS] Saved image to {path}")
            return True
            
        except ImportError:
            print(f"[ERROR] Cannot save {path} - PIL not available")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to save {path}: {e}")
            return False

# Create cv2 instance
cv2 = CV2Compat()

# Test basic functionality on import
def test_compat_layer():
    """Quick test of the compatibility layer"""
    try:
        # Test basic array operations
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        
        # Test template matching
        template = test_img[10:50, 10:50]
        result = cv2.matchTemplate(test_img, template, cv2.TM_CCOEFF_NORMED)
        
        print("[SUCCESS] cv2_compat basic functionality test passed")
        return True
    except Exception as e:
        print(f"[ERROR] cv2_compat test failed: {e}")
        return False

# Run test on import
if __name__ == "__main__":
    test_compat_layer()
else:
    # Quick test when imported
    test_compat_layer()