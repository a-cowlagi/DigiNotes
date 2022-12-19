# DigiNotes: A Comprehensive Lecture Note Digitization Tool

Whether it’s the professors’ lecture notes on the chalkboard, or your friend’s handwritten notes on paper, the common method for copying down other people’s notes is taking a quick picture on your phone. One expects the content to be replete with some text, numerous equations, and a few diagrams. Our project explores the idea of converting handwritten words, equations, and diagrams into an easily digestible, digital format. Our project aims to recognize and process these inputs, by constructing a rich digital representation of the image for later use.

### Installation Requirements

The project can readily be run / tested in a `conda` environment. Simply run the following to set up the conda environment:

```conda create --name <env> --file requirements.txt```

In addition to this, a MathPix API ID and API Key are required (requires purchasing developer access to MathPix) and must be set in the conda environment variables as ``MATHPIX_API_KEY`` and ``MATHPIX_API_ID`` respectively. If unable to acquire a key / ID, contact acowlagi@seas.upenn.edu for temporary access. 

### Usage Details

Using the tool presently requires running scripts from the command line and minor modifications to the scripts themselves. Every stage of the processing pipeline can be tested by running the relevant scripts in the "Tests" directory from the root of the repository. For each stage of the pipeline, simply update the relative path from the repository root to the source image file, and the relative path from the repository root to the output image file(s).

`main.py` integrates all stages of the pipeline after perspective correction, and can presently handle single-board images with text and equations (diagram support has not yet been integrated). Running it on `Sample Notes.jpg` will produce a PDF output saved in the `Main/` directory.
