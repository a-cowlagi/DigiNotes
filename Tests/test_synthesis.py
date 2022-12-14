import sys
import os
import matplotlib.pyplot as plt
sys.path.append("./")
from Synthesize.synthesize import Synthesizer


def main():
    equation1 = {"bbox": [(0, 0), (0, 1), (1, 1), (1, 0)], "latex": "x + y = 2", "digitized_image": None}

    # make equation 2 shifted down by 10 pixels
    equation2 = {"bbox": [(0, 10), (0, 11), (1, 11), (1, 10)], "latex": "x + y = 2", "digitized_image": None}

    text_dict = {(2, 5): "''hello world!", (7, 8): "hi there", (13, 14): "this is a test", (15, 16): "of the synthesis system"}

    synthesizer = Synthesizer([equation1, equation2], text_dict) 

    synthesizer.process()
    synthesizer.save_output("Tests/Outputs/SynthesisOutput/", "synthesis_test.pdf", "Title")

if __name__ == "__main__":
    main()