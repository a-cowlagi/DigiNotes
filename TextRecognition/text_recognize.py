import numpy as np
from typing import List
from PIL import Image
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
import os
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class TextRecognition:

    def __init__(self, line_images: List[np.ndarray], line_coords: List[tuple]):
        self.line_images = line_images
        self.line_coords = line_coords
        self.text_dict = {}

    def process(self):
        self.trocr_images = []
        for line_image in self.line_images:
            # Convert to RGB PIL image
            line_image = Image.fromarray(line_image).convert('RGB')

            self.trocr_images.append(line_image)

        self.text_dict = self.trocr(self.trocr_images)
    
        
    def trocr(self, images: List[Image.Image]):
        # Load model and processor
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

        # Tokenize images
        batch = processor(images, return_tensors="pt").pixel_values

        # Generate text
        labels = model.generate(batch, max_new_tokens=150, num_beams=1, do_sample= False)

        # Convert to text
        texts = processor.batch_decode(labels, skip_special_tokens=True)

        # For every coordinate, use coordinate as key and text as value
        text_dict = {}
        for i, text in enumerate(texts):
            text_dict[self.line_coords[i]] = text
        
        
        return text_dict

    def save_output(self, output_dir: str, output_file: str):
        isfolder = os.path.exists(output_dir)
        if not isfolder:
            os.makedirs(output_dir)

        output_fp = os.path.join(output_dir, output_file)

        # Write text dict to a text file saving the pixel coords and text
        with open(output_fp, "w") as f:
            for key, value in self.text_dict.items():
                print(key, value)
                f.write(f"{key}: {value} \n")
    