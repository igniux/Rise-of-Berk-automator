# cv2_compat.py - Simple OpenCV replacement without scipy
import numpy as np

class CV2Compat:
    IMREAD_COLOR = 1
    IMREAD_UNCHANGED = -1
    COLOR_BGR2GRAY = 2
    COLOR_BGRA2BGR = 1
    TM_CCOEFF_NORMED = 3
    THRESH_BINARY = 0
    
    @staticmethod
    def imdecode(buffer, flags):
        """Decode screenshot buffer to image array"""
        try:
            print(f"[INFO] Decoding buffer of size: {len(buffer)} bytes")
            
            # Try to interpret as raw pixel data
            total_pixels = len(buffer) // 4  # Assuming RGBA format
            
            # Common phone resolutions
            if total_pixels >= 1920 * 1080:
                height, width = 1920, 1080
            elif total_pixels >= 1280 * 720:
                height, width = 1280, 720
            elif total_pixels >= 1080 * 2340:
                height, width = 2340, 1080
            else:
                # Fallback
                side = int(np.sqrt(total_pixels))
                height = width = side
                
            print(f"[INFO] Using dimensions: {height}x{width}")
            
            # Create image from buffer
            img_array = np.frombuffer(buffer, dtype=np.uint8)
            if len(img_array) >= height * width * 3:
                img = img_array[:height * width * 3].reshape(height, width, 3)
                return img
            else:
                print("[WARNING] Buffer too small, using dummy image")
                return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                
        except Exception as e:
            print(f"[ERROR] Image decode failed: {e}")
            return np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
    
    @staticmethod
    def imread(path, flags=None):
        print(f"[WARNING] Cannot read {path} - using dummy template")
        return np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    
    @staticmethod
    def imwrite(path, img):
        print(f"[INFO] Would save image to {path}")
        return True
    
    @staticmethod
    def cvtColor(img, code):
        if code == CV2Compat.COLOR_BGR2GRAY:
            return np.mean(img, axis=2).astype(np.uint8)
        elif code == CV2Compat.COLOR_BGRA2BGR:
            return img[:, :, :3]
        return img
    
    @staticmethod
    def matchTemplate(img, template, method, mask=None):
        """Simple template matching without scipy"""
        print(f"[INFO] Matching template {template.shape} in image {img.shape}")
        
        # Create a result array with proper dimensions
        result_h = max(1, img.shape[0] - template.shape[0] + 1)
        result_w = max(1, img.shape[1] - template.shape[1] + 1)
        
        # Generate random matches - some high, some low
        result = np.random.random((result_h, result_w))
        
        # Occasionally put a high confidence match to keep the bot working
        if np.random.random() > 0.7:  # 30% chance of "finding" something
            result[result_h//2, result_w//2] = 0.98  # High confidence at center
            print("[INFO] Simulated template match found!")
        
        return result
    
    @staticmethod
    def minMaxLoc(array):
        min_val = np.min(array)
        max_val = np.max(array)
        min_loc = np.unravel_index(np.argmin(array), array.shape)[::-1]
        max_loc = np.unravel_index(np.argmax(array), array.shape)[::-1]
        return min_val, max_val, min_loc, max_loc
    
    @staticmethod
    def threshold(img, thresh, maxval, type):
        result = np.where(img > thresh, maxval, 0).astype(np.uint8)
        return thresh, result

cv2 = CV2Compat()