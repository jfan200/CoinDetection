# Built in packages
import sys
import cv2
import CS373_coin_detection_extension as cs373
from matplotlib import pyplot as plt

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


def plot_task_1(original_image, stretched_image):
    fig, axs = pyplot.subplots(1, 2)
    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(stretched_image, cmap='gray')
    axs[1].set_title('Stretched Image')
    axs[1].axis('off')
    fig.suptitle('Convert to greyscale and normalize')

def plot_task_2(scharr_x, scharr_y, abs_diff):
    # Plot scharr_x and scharr_y mapping
    fig, axs = pyplot.subplots(1, 3)
    axs[0].imshow(scharr_x, cmap='gray')
    axs[0].set_title('scharr x-direction')
    axs[0].axis('off')
    axs[1].imshow(scharr_y, cmap='gray')
    axs[1].set_title('scharr y-direction')
    axs[1].axis('off')

    # Plot absolute different
    axs[2].imshow(abs_diff, cmap='gray')
    axs[2].set_title('abs different')
    axs[2].axis('off')
    fig.suptitle('Edge detection')

def plot_task_3(blurred_image_1, blurred_image_2):
    blurred_image_1 = np.array(blurred_image_1)
    blurred_image_2 = np.array(blurred_image_2)
    # apply gaussian fill
    fig, axs = pyplot.subplots(2, 2)
    axs[0][0].imshow(blurred_image_1, cmap='gray')
    axs[0][0].set_title('One time')
    axs[0][0].axis('off')
    axs[0][1].hist(blurred_image_1.ravel(), 256, [0, 255])
    axs[0][1].set_title('Histogram')

    axs[1][0].imshow(blurred_image_2, cmap='gray')
    axs[1][0].set_title('Two times')
    axs[1][0].axis('off')
    axs[1][1].hist(blurred_image_2.ravel(), 256, [0, 255])

    fig.suptitle('Applying Gaussian filter')

def plot_task_4(blurred_image_2, binary_image):
    fig, axs = pyplot.subplots(1, 2)
    axs[0].imshow(blurred_image_2, cmap='gray')
    axs[0].set_title('blurred_image_2')
    axs[0].axis('off')
    axs[1].imshow(binary_image, cmap='gray')
    axs[1].set_title('binary_image')
    axs[1].axis('off')
    fig.suptitle('Threshold the Image')

def polt_task_5(dilation_2, dilation_3, dilation_4):
    fig, axs = pyplot.subplots(1, 3)
    axs[0].imshow(dilation_2, cmap='gray')
    axs[0].set_title('dilation_2')
    axs[0].axis('off')
    axs[1].imshow(dilation_3, cmap='gray')
    axs[1].set_title('dilation_3')
    axs[1].axis('off')
    axs[2].imshow(dilation_4, cmap='gray')
    axs[2].set_title('dilation_4')
    axs[2].axis('off')
    fig.suptitle('Erosion and Dilation')


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
    gray_scale_image = cs373.convert_RGB_to_Grayscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    stretched_image = cs373.percentile_based_mapping(gray_scale_image, image_width, image_height)
    plot_task_1(original_image, stretched_image)

    # Task 2 Edge detection
    # Find both scharr_x and scharr_y mapping
    scharr_x = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
    scharr_y = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]]

    scharr_x = cs373.apply_3x3_filter(stretched_image, image_width, image_height,  scharr_x)
    scharr_y = cs373.apply_3x3_filter(stretched_image, image_width, image_height,  scharr_y)
    abs_diff = cs373.abs_diff(scharr_x, scharr_y)
    plot_task_2(scharr_x, scharr_y, abs_diff)

    # Task 3 Image blurring
    mean_filter_5x5 = [[1/5 for i in range(5)] for j in range(5)]
    print(mean_filter_5x5)
    blurred_image_1 = cs373.apply_5x5_filter(abs_diff, image_width, image_height, mean_filter_5x5)
    blurred_image_2 = cs373.apply_5x5_filter(blurred_image_1, image_width, image_height, mean_filter_5x5)


    plot_task_3(blurred_image_1, blurred_image_2)

    # Task 4 Threshold the Image
    threshold_value = 250  # Based on the histogram
    binary_image = cs373.custom_threshold(blurred_image_2, image_width, image_height, 50, 200)
    plot_task_4(blurred_image_2, binary_image)

    # Task 5 Erosion and Dilation
    circular_kernel = np.array([[0, 0, 1, 0, 0],
                                [0, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1],
                                [0, 1, 1, 1, 0],
                                [0, 0, 1, 0, 0]], dtype=np.uint8)

    # dilation_2 = cv2.dilate(binary_image, circular_kernel, iterations=2)
    # dilation_3 = cv2.dilate(binary_image, circular_kernel, iterations=3)
    # dilation_4 = cv2.dilate(binary_image, circular_kernel, iterations=4)
    # polt_task_5(dilation_2, dilation_3, dilation_4)

    # Task 6 Connected Component Analysis
    regions = cs373.computeConnectedComponentLabeling(binary_image, image_width, image_height)

    # Task 7 Draw Bounding Box
    bounding_box_list = cs373.detect_the_region(regions)

    ############################################
    ### Bounding box coordinates information ###
    ### bounding_box[0] = min x
    ### bounding_box[1] = min y
    ### bounding_box[2] = max x
    ### bounding_box[3] = max y
    ############################################
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
