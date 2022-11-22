import sys
sys.path.append("./")
import numpy as np
from PIL import Image
from AdaptiveBinarization.adaptive_binarize import AdaptiveBinarization

# Relative to repository root
src_image_path = "Tests/TestImages/adaptive_binarization_test.jpg"
tgt_image_path = "Tests/Outputs/AdaptiveBinarizationOutput"

params = {"threshold_method": "Sauvola", "blur": True, "sigma": 1.0, "kwargs": {"window_size": 25}}

if __name__ == "__main__":
    src_img = np.array(Image.open(src_image_path).convert('RGB'))
    perspective_corrector = AdaptiveBinarization(src_img)
    perspective_corrector.process(threshold_method= params["threshold_method"], 
                                blur = params["blur"], sigma = params["sigma"], **params["kwargs"])
    output_filename = src_image_path.split("/")[-1].split(".")[0]
    perspective_corrector.save_output(output_filename=output_filename, tgt_image_path= tgt_image_path)
    
