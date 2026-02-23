import cv2
import numpy as np
from skimage.measure import shannon_entropy
from skimage.morphology import disk
from skimage.filters.rank import entropy

class CEAV:

    @staticmethod
    def get_luminance_from_rgb(r: int, g: int, b: int):
        return 0.299 * r + 0.587 * g + 0.144 * b
    
    @staticmethod
    def luminance_mean_from_img(img):

        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l = img_lab[:, :, 0]
        luminance_mean = np.mean(l)
        return luminance_mean
    
    @staticmethod
    def analyse(img):
        luminance = CEAV.luminance_mean_from_img(img)
        entropy = CEAV.get_global_entropy(img)
        local_entropy = CEAV.get_local_entropy(img)

        return luminance, entropy, local_entropy
    
    @staticmethod
    def get_global_entropy(img):
        
        # grey scale
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        global_entropy = shannon_entropy(img_gray)
        return global_entropy
    
    @staticmethod
    def get_local_entropy(img):

        # grey scale
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        if img_gray.dtype != np.uint8:
            img_gray = img_gray.astype(np.uint8)

        window = disk(5)
        entropy_map = entropy(img_gray, window)
        return entropy_map


        