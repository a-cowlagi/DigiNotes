from typing import Optional
import numpy as np
import skimage.io
import io
from PIL import ImageOps, Image, ImageChops
from scipy.ndimage import morphology, label
import os
from mathpix.mathpix import MathPix
from sympy import preview
from sympy.abc import x, y, z
from sympy.parsing.latex import parse_latex
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

class EquationDetection:

    def __init__(self, image: Optional[np.ndarray], api_key: Optional[str] = None, api_id: Optional[str] = None):

        self.src_img = image
        self.equations = None
        self.api_key = api_key
        self.api_id = api_id

    
    def process(self, image = None):
        if image is not None:
            self.src_img = image


        self.src_img, boxes = self.boxes(self.src_img)
        self.equations = self.compile_equations(self.src_img, boxes)

        return self.equations

    
    def boxes(self, orig: np.ndarray):
        # Convert orig to a PIL image
        orig = Image.fromarray(orig)
        img = ImageOps.grayscale(orig)

        im = np.array(img)

        # Inner morphological gradient.
        im = morphology.grey_dilation(im, (3, 3)) - im

        # Binarize.
        mean, std = im.mean(), im.std()
        t = mean + std
        im[im < t] = 0
        im[im >= t] = 1

        # Connected components.
        lbl, numcc = label(im)
        # Size threshold.
        min_size = 200 # pixels
        box = []
        for i in range(1, numcc + 1):
            py, px = np.nonzero(lbl == i)
            if len(py) < min_size:
                im[lbl == i] = 0
                continue

            xmin, xmax, ymin, ymax = px.min(), px.max(), py.min(), py.max()
            # Four corners and centroid.
            box.append([
                [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)],
                (np.mean(px), np.mean(py))])

        # Check for boxes that are completely contained within other boxes.
        box_to_remove = set()
        for i, b1 in enumerate(box):
            for j, b2 in enumerate(box):
                if i != j and b1[0][0][0] >= b2[0][0][0] and b1[0][0][1] >= b2[0][0][1] and b1[0][2][0] <= b2[0][2][0] and b1[0][2][1] <= b2[0][2][1]:
                    box_to_remove.add(i)
        box = [b for i, b in enumerate(box) if i not in box_to_remove]

        # Convert PIL img to numpy array.
        img = np.array(img)

        return img, box


    def compile_equations(self, image, boxes):
        # Create a list to hold the dictionaries.
        equations = []

        # Iterate over the boxes and crop the image.
        for i, box in enumerate(boxes):
            # Extract the coordinates of the box.
            xmin, ymin, xmax, ymax = box[0][0][0], box[0][0][1], box[0][2][0], box[0][2][1]
            
            eqn_img = image[ymin:ymax, xmin:xmax]
            
            ocr = self._eqn_img_to_latex(eqn_img)
            try: 
                print(ocr.latex)
                digitized_equation = self.tex_to_image(ocr.latex)
            except ValueError as e:
                digitized_equation = None       
            finally:
                equation_dict = {
                    "orig_image": eqn_img,
                    "digitized_image": digitized_equation,
                    "latex": ocr.latex,
                    "centroid": box[1],
                    "bbox": box[0]
                }

                equations.append(equation_dict)
                

        # Sort the equations from left to right, top to bottom.
        equations.sort(key=lambda x: (x["centroid"][1], x["centroid"][0]))

        return equations

    def _eqn_img_to_latex(self, image):
        # Temporarily save the equation image to a file.
        skimage.io.imsave("equation.jpg", image)
        
        mathpix = MathPix(app_key=self.api_key, app_id=self.api_id)
        ocr = mathpix.process_image(image_path = "equation.jpg")
        # Delete the saved temporary image.
        os.remove("equation.jpg")
        return ocr

    def tex_to_image(self, tex_str: str):
        white = (255, 255, 255, 255)
        buf = io.BytesIO()
        
        plt.clf()
        plt.rc('text', usetex=False)
        plt.axis('off')
        plt.text(0.05, 0.5, f"${tex_str}$", size=40)
        plt.savefig(buf, format='png', bbox_inches = "tight")
        plt.close()

        im = Image.open(buf)
        bg = Image.new(im.mode, im.size, white)
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        
        digitized = im.crop(bbox)

        # Find the bounding box of the digitized image
        digitized = np.array(digitized)
        
        # Find the bounding box of the binary image.
        rows = np.any(~digitized, axis=1)
        cols = np.any(~digitized, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        # Crop the original image using the bounding box coordinates.
        foreground = digitized[ymin:ymax, xmin:xmax]
        foreground = foreground[:,:, 0]

        # Make foreground an RGB image.
        foreground = np.stack((foreground, foreground, foreground), axis=2)

        return foreground


    def save_output(self, output_dir):
        if (self.equations is None):
            raise AttributeError("No non-null attribute 'equations', call process() first?")

        isfolder = os.path.exists(output_dir)
        if not isfolder:
            os.makedirs(output_dir)
        
        for i, equation in enumerate(self.equations):
            output_orig_fp = os.path.join(output_dir, f"equation_orig_{i}.jpg")
            output_digitized_fp = os.path.join(output_dir, f"equation_digitized_{i}.jpg")
            
            if (equation['digitized_image'] is None):
                continue
            
            skimage.io.imsave(output_digitized_fp, equation['digitized_image'])
            skimage.io.imsave(output_orig_fp, equation['orig_image'])
            
            
            
            





        