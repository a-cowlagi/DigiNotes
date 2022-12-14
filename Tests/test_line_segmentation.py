import sys
import os
import matplotlib.pyplot as plt
sys.path.append("./")
import argparse
from LineSegments.line_segmentation import LineSegmentation

# Relative to repository root
src_image_path = "Tests/TestImages/test_main_lines.jpg"
tgt_images_path = "Tests/Outputs/LineSegmentationOutput/Main"

def generate_lines(src_path, tgt_path):
    image = plt.imread(src_path)
    line_segmenter = LineSegmentation(image)
    line_segmenter.process()
    line_segmenter.save_output(tgt_path)

if __name__ == "__main__":
    generate_lines(src_image_path, tgt_images_path)   
    