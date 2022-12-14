import os
import fpdf
import skimage.io
import re

class Synthesizer:
    def __init__(self, equations, text_dict):
        self.equations = equations
        self.text_lines = list(zip(text_dict.keys(), text_dict.values()))

        self.output = None
        self.output_dir = None
        self.output_file = None

    def process(self):
        self.pdf_stream = []

        line_index = 0 
        for i, equation in enumerate(self.equations):
            bbox = equation["bbox"]
            xmin, ymin, xmax, ymax = bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]

            while (line_index < len(self.text_lines) and self._get_lower_bound(line_index) < ymin):
                self.pdf_stream.append((self.text_lines[line_index][1], False))
                line_index += 1
            
            self.pdf_stream.append((equation, True))
            print(f"Equation: {str(equation['latex'])}")

        while (line_index < len(self.text_lines)):
            self.pdf_stream.append((self.text_lines[line_index][1], False))
            line_index += 1
        
        print(self.pdf_stream)

    def save_output(self, output_dir, output_file, title):
        self.output_dir = output_dir
        self.output_file = output_file

        if (self.pdf_stream is None):
            raise AttributeError("No non-null attribute 'output', call process() first?")
        
        isfolder = os.path.exists(self.output_dir)
        if not isfolder:
            os.makedirs(self.output_dir)
        
        pdf = self.generate_pdf(title)
        pdf.output(os.path.join(self.output_dir, self.output_file))

    def generate_pdf(self, title):
        pdf = fpdf.FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'BI', size=35)
        pdf.cell(200, 10, txt=title, ln=1, align="C")

        pdf.set_font("Arial", size=25)

        curr_eqn = 0
        for i, (item, is_equation) in enumerate(self.pdf_stream):
            if (is_equation):
                curr_eqn += 1
                if (item["digitized_image"] is None):
                    pdf.cell(200, 10, txt=item["latex"], ln=1, align="C")
                else:
                    eqn_img = item["digitized_image"]
                    print(item["latex"])
                    # temporarily save the equation image to a file
                    skimage.io.imsave(f"equation_{curr_eqn}.jpg", eqn_img)
                    pdf.image(f"equation_{curr_eqn}.jpg", h = 10, x = 25)
                    
            else:
                text = item.replace("...", '')
                text = text.replace("'", '')
                text = text.replace('"', '')
                pdf.multi_cell(0, 20, txt=text, align="L")
        
        # remove all the temporary equation images
        for i in range(1, curr_eqn + 1):
            os.remove(f"equation_{i}.jpg")

        return pdf
        

    # static method get_lower_bound
    def _get_lower_bound(self, line_index):
        return self.text_lines[line_index][0][1]
