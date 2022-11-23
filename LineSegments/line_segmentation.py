import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
import os

class LineSegmentation:
    def __init__(self, image):
        if (image is not None):
            self.src_image = image
        else:
            raise ValueError("Image can't be NoneType")
        
        self.out_images: List = None
   
    def process(self, image: Optional[np.ndarray] = None, th = 2, min_lw = 5):
        if (image is None):
            image = self.src_image
        # Ensure gray scale
        if (len(image.shape) == 3):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else: 
            gray = image
        
        # Run thresholding algorithm and horizontal projection profile
        _, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        hpp = cv2.reduce(threshed,1, cv2.REDUCE_AVG).reshape(-1)

        H,W = image.shape[:2]
        uppers = [y for y in range(H-1) if hpp[y]<=th and hpp[y+1]>th]
        lowers = [y for y in range(H-1) if hpp[y]>th and hpp[y+1]<=th]

        uppers = np.array(uppers)
        lowers = np.array(lowers)
        lowers = lowers[lowers > uppers[0]]
        uppers = uppers[uppers < lowers[-1]]

        threshed_inv = ~threshed
        split_by_line_images = []

        for (upper, lower) in zip(uppers, lowers):
            if (lower - upper >= min_lw):
                split_by_line_images.append(threshed_inv[upper:lower, :])

        self.out_images = split_by_line_images
    
    def save_output(self, output_path: str, rgb = True):
        isfolder = os.path.exists(output_path)
        if not isfolder:
            os.makedirs(output_path)

        if self.out_images is None:
            raise ValueError("Call process() first!")
        
        for i, line_img in enumerate(self.out_images):
            if (rgb):
                line_img = cv2.cvtColor(line_img, cv2.COLOR_GRAY2BGR)
            output_fp = os.path.join(output_path, f"line_{i}.jpg")
            cv2.imwrite(output_fp, line_img)
            

    
