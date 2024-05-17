# Built in packages
import sys
import CS373_coin_detection_extension as cs373

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



# This is our code skeleton that performs the coin detection.
def main(input_path, output_path):
    # This is the default input image, you may change the 'image_name' variable to test other images.
    image_name = 'easy_case_1'
    input_filename = f'./Images/easy/{image_name}.png'
    if TEST_MODE:
        input_filename = input_path

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    print("RGB to Grey")
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    ###################################
    ### STUDENT IMPLEMENTATION Here ###
    ###################################


    # Task 1 Convert to greyscale and normalize
    print("Task 1")
    grey_scale_image = cs373.convert_RGB_to_Grayscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    stretched_image = cs373.percentile_based_mapping(grey_scale_image, image_width, image_height)
    cs373.plot_task_1(grey_scale_image, stretched_image)

    # Task 2 Edge detection
    # Find both scharr_x and scharr_y mapping
    print("Task 2")
    scharr_x = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
    scharr_y = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]]

    scharr_x = cs373.apply_filter(stretched_image, image_width, image_height,  scharr_x)
    scharr_y = cs373.apply_filter(stretched_image, image_width, image_height,  scharr_y)
    abs_diff = cs373.abs_diff(scharr_x, scharr_y)
    cs373.plot_task_2(scharr_x, scharr_y, abs_diff)

    # Task 3 Image blurring
    print("Task 3")
    mean_filter_5x5 = [[0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04]]

    blurred_image_1 = cs373.apply_filter(abs_diff, image_width, image_height, mean_filter_5x5)
    blurred_image_2 = cs373.apply_filter(blurred_image_1, image_width, image_height, mean_filter_5x5)
    blurred_image_2 = cs373.apply_filter(blurred_image_2, image_width, image_height, mean_filter_5x5)
    cs373.plot_task_3(blurred_image_1, blurred_image_2)


    # Task 4 Threshold the Image
    print("Task 4")
    binary_image = cs373.custom_threshold(blurred_image_1, image_width, image_height, 180, 200)
    cs373.plot_task_4(blurred_image_2, binary_image)

    # Task 5 Erosion and Dilation
    print("Task 5")
    circular_kernel = [[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]]

    dilated_image = cs373.dilate_image(binary_image, image_width, image_height, circular_kernel, iterations=2)
    eroded_image = cs373.erode_image(dilated_image, image_width, image_height, circular_kernel, iterations=4)
    cs373.polt_task_5(binary_image, dilated_image, eroded_image)

    # Task 6 Connected Component Analysis
    print("Task 6")
    regions = cs373.computeConnectedComponentLabeling(eroded_image, image_width, image_height)

    # Task 7 Draw Bounding Box
    print("Task 7")
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
