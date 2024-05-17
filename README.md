# Coin Detection Project
## Overview
This project, developed by Jinhua Fan, focuses on detecting coins in images using various computer vision techniques. The program utilizes a sequence of image processing tasks including color separation, grayscale conversion, normalization, edge detection, blurring, thresholding, and morphological operations (dilation and erosion) to isolate and identify coins.

## Installation
### Prerequisites
- Python 3.x 
- Matplotlib (for image processing and visualization)
- imageIO (for handling PNG images)

## Setup
To set up the project, clone this repository and install the required Python packages:

``` bash
git clone https://github.com/jfan200/Coin_Detection.git
cd your-repository
pip install matplotlib
pip install imageio
```

## Usage
To run the coin detection, execute the main script with the path to the input image and the path where the output should be saved:

```bash
python coin_detection.py
```

## Features
The project performs the following tasks:

1. **Color Separation:** Splits the input RGB image into separate red, green, and blue components. 
2. **Grayscale Conversion:** Converts the RGB image to grayscale using a weighted sum approach. 
3. **Normalization:** Applies percentile-based normalization to enhance the image contrast. 
4. **Edge Detection:** Uses the Scharr operator in both x and y directions to detect edges. 
5. **Image Blurring:** Applies a mean filter to smooth the image, reducing high-frequency noise. 
6. **Thresholding:** Segments the image into binary format using a custom dual-threshold technique. 
7. **Morphological Operations:** Performs dilation and erosion to refine the shapes of detected coins. 
8. **Connected Component Analysis:** Identifies connected regions in the binary image, marking potential coins. 
9. **Bounding Box Detection:** Draws bounding boxes around detected coins, visualizing the results.


## Output
The program outputs images at various stages of processing, including the final result with bounding boxes around detected coins. These images are saved in the specified output directory.

