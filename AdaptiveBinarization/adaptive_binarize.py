import os
import numpy as np
import cv2
import skimage.color
import skimage.filters._gaussian as gaussian
from skimage.filters import thresholding
from typing import Optional

class AdaptiveBinarization:
    def __init__(self, src_img: np.ndarray, scale: str = "255"):
        assert (len(src_img.shape) == 2 or len(src_img.shape) == 3)
        self.src_img = src_img
        self.binarized_img = None
        self.scale = scale
    
    def process(self, image: Optional[np.ndarray] = None, threshold_method = "Sauvola", blur = True, sigma = 1.0, **kwargs):
        if (image is None):
            image = self.src_img
        
        if (len(image.shape) == 3 and image.shape[2] == 3):
            image = skimage.color.rgb2gray(image)
        
        if (blur):
            image = gaussian.gaussian(image, sigma=sigma)

        binarized = None
        if (threshold_method == "Sauvola"):
            threshold = thresholding.threshold_sauvola(image, **kwargs)
        elif (threshold_method == "Niblack"):
            threshold = thresholding.threshold_niblack(image, **kwargs)
        elif (threshold_method == "Otsu"):
            threshold = thresholding.threshold_otsu(image, **kwargs)
        elif (threshold_method == "Yen"):
            threshold = thresholding.threshold_yen(image, **kwargs)
        else:
            raise NotImplementedError("Thresholding methods supported are Sauvola, Niblack, Otsu, and Yen")

        binarized = image > threshold
        
        if (self.scale == "255"):
            self.binarized_img = binarized.astype(int) * 255
            
    
    def save_output(self, output_filename, tgt_image_path):
        if (self.binarized_img is None):
            raise AttributeError("No non-null attribute 'binarized_img', call process() first?")

        binarized_img = self.binarized_img
        isfolder = os.path.exists(tgt_image_path)
        if not isfolder:
            os.makedirs(tgt_image_path)
        
        if (len(binarized_img.shape) != 3):
            binarized_img = skimage.color.gray2rgb(binarized_img)
            
        output_fp = os.path.join(tgt_image_path, f"{output_filename}_corrected.jpg")
        
        cv2.imwrite(output_fp, binarized_img)
        
        


