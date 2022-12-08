import os
import sys
sys.path.append("./")
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.spatial import ConvexHull, distance
from PerspectiveCorrection.perspective_correction import PerspectiveCorrection

# Relative to repository root
src_image_path = "Tests/TestImages/perspective_correction_test_whiteboard.jpg"
tgt_image_path = "Tests/Outputs/PerspectiveCorrectionOutput"

characteristic_length = 1000


if __name__ == "__main__":
    src_img = np.array(Image.open(src_image_path).convert('RGB'))
    perspective_corrector = PerspectiveCorrection(src_img)
    perspective_corrector.process(characteristic_length = characteristic_length)
    output_filename = src_image_path.split("/")[-1].split(".")[0]
    perspective_corrector.save_output(output_filename=output_filename, tgt_image_path= tgt_image_path)
    
