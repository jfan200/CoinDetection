# Built in packages
import math
import sys
import cv2

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import numpy as np

# import our basic, light-weight png reader library
import imageIO.png

# Define constant and global variables
TEST_MODE = False  # Please, DO NOT change this variable!


def readRGBImageToSeparatePixelArrays(input_filename):
    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue=0):
    new_pixel_array = []
    for _ in range(image_height):
        new_row = []
        for _ in range(image_width):
            new_row.append(initValue)
        new_pixel_array.append(new_row)

    return new_pixel_array


###########################################
### You can add your own functions here ###
###########################################

# Task 1 Convert to greyscale and normalize
# Conversion from RGB to Greyscale
def convert_RGB_to_Grayscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for row in range(image_height):
        for col in range(image_width):
            r = pixel_array_r[row][col]
            g = pixel_array_g[row][col]
            b = pixel_array_b[row][col]

            # RGB channel ratio 0.3 x red, 0.6 x green, 0.1 x blue
            greyscale_pixel_array[row][col] = round(r * 0.3 + g * 0.6 + b * 0.1)

    return greyscale_pixel_array

# Using 5-95 percentile strategy
def get_percentiles(pixel_array):
    # Convert pixel_array into numpy array
    pixel_array = np.array(pixel_array)

    # Count percentiles
    percentiles = {
        '5th': np.percentile(pixel_array, 5),
        '95th': np.percentile(pixel_array, 95)
    }

    return percentiles

def percentile_based_mapping(pixel_array, image_width, image_height):
    contrast_stretched_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    percentiles = get_percentiles(pixel_array)

    for row in range(image_height):
        for col in range(image_width):
            f = pixel_array[row][col]
            g = ((f - percentiles['5th']) * 255) / (percentiles['95th'] - percentiles['5th'])
            contrast_stretched_pixel_array[row][col] = g
    return contrast_stretched_pixel_array


# Task 2 Edge detection
def scharr_filter(pixel_array, image_width, image_height, direction):
    scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    scharr_y = np.array([[-3, 10, -3], [0, 0, 0], [3, 10, 3]])
    if direction == 'x':
        scharr_filter = scharr_x
        return cv2.Scharr(pixel_array, cv2.CV_64F, 1, 0)
    elif direction == 'y':
        scharr_filter = scharr_y





# This is our code skeleton that performs the coin detection.
def main(input_path, output_path):
    # This is the default input image, you may change the 'image_name' variable to test other images.
    image_name = 'easy_case_1'
    input_filename = f'./Images/easy/{image_name}.png'
    if TEST_MODE:
        input_filename = input_path

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    ###################################
    ### STUDENT IMPLEMENTATION Here ###
    ###################################
    original_image = np.stack((px_array_r, px_array_g, px_array_b), axis=-1)

    # Task 1 Convert to greyscale and normalize
    gray_scale_image = convert_RGB_to_Grayscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    stretched_image = percentile_based_mapping(gray_scale_image, image_width, image_height)

    fig, axs = pyplot.subplots(1, 2)
    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(stretched_image, cmap='gray')
    axs[1].set_title('Stretched Image')
    fig.suptitle('Convert to greyscale and normalize')

    # Task 2 Edge detection


    ############################################
    ### Bounding box coordinates information ###
    ### bounding_box[0] = min x
    ### bounding_box[1] = min y
    ### bounding_box[2] = max x
    ### bounding_box[3] = max y
    ############################################

    bounding_box_list = [
        [150, 140, 200, 190]]  # This is a dummy bounding box list, please comment it out when testing your own code.
    px_array = px_array_r

    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(px_array, aspect='equal')

    # Loop through all bounding boxes
    for bounding_box in bounding_box_list:
        bbox_min_x = bounding_box[0]
        bbox_min_y = bounding_box[1]
        bbox_max_x = bounding_box[2]
        bbox_max_y = bounding_box[3]

        bbox_xy = (bbox_min_x, bbox_min_y)
        bbox_width = bbox_max_x - bbox_min_x
        bbox_height = bbox_max_y - bbox_min_y
        rect = Rectangle(bbox_xy, bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')
        axs.add_patch(rect)

    pyplot.axis('off')
    pyplot.tight_layout()
    default_output_path = f'./output_images/{image_name}_with_bbox.png'
    if not TEST_MODE:
        # Saving output image to the above directory
        pyplot.savefig(default_output_path, bbox_inches='tight', pad_inches=0)

        # Show image with bounding box on the screen
        pyplot.imshow(px_array, cmap='gray', aspect='equal')
        pyplot.show()
    else:
        # Please, DO NOT change this code block!
        pyplot.savefig(output_path, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    num_of_args = len(sys.argv) - 1

    input_path = None
    output_path = None
    if num_of_args > 0:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        TEST_MODE = True

    main(input_path, output_path)
