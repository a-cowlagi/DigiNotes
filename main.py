import matplotlib.pyplot as plt
from AdaptiveBinarization import utils
from AdaptiveBinarization.adaptive_binarize import AdaptiveBinarization
from EquationDetection.equation_detection import EquationDetection
from LineSegments.line_segmentation import LineSegmentation
from TextRecognition.text_recognize import TextRecognition
from Synthesize.synthesize import Synthesizer
import os


src_image_path = "Sample Notes.jpg"
output_dir = "Main/"


def main(src_image_path):
    image = plt.imread(src_image_path)
    enhanced_img = utils.enhance(image)
    print("-------Finished Enhancing Image-------")
    
    # Equation Detection
    eqn_detector = EquationDetection(enhanced_img)
    eqn_detector.process()
    print("-------Finished Equation Detection-------")
    
    # Adaptive Binarization 
    
    adaptive_binarizer = AdaptiveBinarization(image)
    adaptive_binarizer.process()
    # Filter out equations from adaptive binarization
    adaptive_binarizer.filter_equations(eqn_detector.equations)
    print("-------Finished Adaptive Binarization-------")

    # Line Segmentation
    line_segmenter = LineSegmentation(adaptive_binarizer.binarized_img)
    line_segmenter.process()
    print("-------Finished Line Segmentation-------")

    # Text Recognition
    text_recognizer = TextRecognition(line_segmenter.line_images, line_segmenter.line_coords)
    text_recognizer.process()
    print("-------Finished Text Recognition-------")

    # Synthesize Output
    synthesizer = Synthesizer(eqn_detector.equations, text_recognizer.text_dict)
    synthesizer.process()
    synthesizer.save_output(output_dir, output_file = f"{src_image_path.split('.')[0]}.pdf", title = src_image_path.split('.')[0])

    text_recognizer.save_output(f"{output_dir}TextRecognition/", f"{src_image_path.split('.')[0]}.txt")
    print("-------Finished Saving Output-------")



if __name__ == "__main__":
    main(src_image_path)