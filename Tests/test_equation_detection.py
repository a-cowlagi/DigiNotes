import sys
import os
import numpy as np
from PIL import Image
sys.path.append("./")
from EquationDetection.equation_detection import EquationDetection

output_dir = "Tests/Outputs/EquationDetection/"
src_image_path = "Tests/TestImages/test_equations.jpg"

# get my api key and id from environment variables
api_key = os.environ.get("MATHPIX_API_KEY")
api_id = os.environ.get("MATHPIX_API_ID")

if __name__ == "__main__":
    src_img = np.array(Image.open(src_image_path).convert('RGB'))
    
    eqn_detector = EquationDetection(src_img, api_key, api_id)
    eqn_detector.process()
    eqn_detector.save_output(output_dir)
