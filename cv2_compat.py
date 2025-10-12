# cv2_compat.py - Simple OpenCV replacement
import numpy as np
from PIL import Image
import io

class CV2Compat:
    IMREAD_COLOR = 1
    IMREAD_UNCHANGED = -1
    COLOR_BGR2GRAY = 2
    TM_CCOEFF_NORMED = 3
    
    @staticmethod
    def imdecode(buffer, flags):
        img = Image.open(io.BytesIO(buffer))
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        return np.array(img)[:, :, ::-1]  # RGB to BGR
    
    @staticmethod
    def imread(path, flags=None):
        img = Image.open(path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        return np.array(img)[:, :, ::-1]
    
    @staticmethod
    def imwrite(path, img):
        if len(img.shape) == 3:
            img = img[:, :, ::-1]  # BGR to RGB
        Image.fromarray(img).save(path)
        return True
    
    @staticmethod
    def cvtColor(img, code):
        if code == CV2Compat.COLOR_BGR2GRAY:
            return np.dot(img[...,:3], [0.114, 0.587, 0.299]).astype(np.uint8)
        return img
    
    @staticmethod
    def matchTemplate(img, template, method):
        # Simple correlation - good enough for your bot
        from scipy.ndimage import correlate
        if len(img.shape) == 3:
            img = CV2Compat.cvtColor(img, CV2Compat.COLOR_BGR2GRAY)
        if len(template.shape) == 3:
            template = CV2Compat.cvtColor(template, CV2Compat.COLOR_BGR2GRAY)
        return correlate(img.astype(float), template.astype(float), mode='constant')
    
    @staticmethod
    def minMaxLoc(array):
        min_val = np.min(array)
        max_val = np.max(array)
        min_loc = np.unravel_index(np.argmin(array), array.shape)[::-1]
        max_loc = np.unravel_index(np.argmax(array), array.shape)[::-1]
        return min_val, max_val, min_loc, max_loc

cv2 = CV2Compat()