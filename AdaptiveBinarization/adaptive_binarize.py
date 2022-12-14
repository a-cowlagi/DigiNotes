import os
import numpy as np
import cv2
import skimage.color
import skimage.filters._gaussian as gaussian
from skimage.filters import thresholding
from typing import Optional
from .utils import enhance

class AdaptiveBinarization:
    def __init__(self, src_img: np.ndarray, scale: str = "uint8"):
        assert (len(src_img.shape) == 2 or len(src_img.shape) == 3)
        self.src_img = src_img
        self.binarized_img = None
        self.scale = scale
    
    def process(self, image: Optional[np.ndarray] = None, enhance_first = True, threshold_method = "Sauvola", **kwargs):
        if (image is None):
            image = self.src_img
        
        if (enhance_first):
            self.src_img = enhance(image)

        if (len(image.shape) == 3 and image.shape[2] == 3):
            image = skimage.color.rgb2gray(image)

        self.binarized_img = image
        if (threshold_method is not None):
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
            
            if (self.scale == "uint8"):
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
        
    def filter_equations(self, equations):
        if (self.binarized_img is None):
            raise AttributeError("No non-null attribute 'binarized_img', call process() first?")
        
        for i, equation in enumerate(equations):
            bbox = equation["bbox"]
            xmin, ymin, xmax, ymax = bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]   
            self.binarized_img[ymin:ymax, xmin:xmax] = (255 if self.scale == "uint8" else 1)
        
        if (len(self.binarized_img.shape) != 3):
            self.binarized_img = skimage.color.gray2rgb(self.binarized_img)
        
        


