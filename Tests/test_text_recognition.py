import sys
import os
import matplotlib.pyplot as plt
sys.path.append("./")
import argparse
from LineSegments.line_segmentation import LineSegmentation
from TextRecognition.text_recognize import TextRecognition

# Relative to repository root
src_image_path = "Tests/TestImages/ocr_test.jpg"
output_text_filedir = "Tests/Outputs/TextRecognitionOutput/"
line_segmentation_output_dir = "Tests/Outputs/LineSegmentationOutput/OCRTest/"
output_text_filename = "ocr_test.txt"

    
if __name__ == "__main__":
    image = plt.imread(src_image_path)
    line_segmenter = LineSegmentation(image)
    line_segmenter.process()
    line_segmenter.save_output(line_segmentation_output_dir)
    text_recognizer = TextRecognition(line_segmenter.line_images, line_segmenter.line_coords)
    text_recognizer.process()
    text_recognizer.save_output(output_text_filedir, output_text_filename)
    