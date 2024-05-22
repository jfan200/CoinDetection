# Built in packages
import math
import sys

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.text import Text

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

def plot_task_1(gray_scale_image, stretched_image):
    fig, axs = pyplot.subplots(1, 2)
    axs[0].imshow(gray_scale_image, cmap='gray')
    axs[0].set_title('Gray Scale Image')
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
    flat_list_1 = [item for sublist in blurred_image_1 for item in sublist]
    flat_list_2 = [item for sublist in blurred_image_2 for item in sublist]
    # apply gaussian fill
    fig, axs = pyplot.subplots(2, 2)
    axs[0][0].imshow(blurred_image_1, cmap='gray')
    axs[0][0].set_title('One time')
    axs[0][0].axis('off')
    axs[0][1].hist(flat_list_1, 256, range=(0, 255))
    axs[0][1].set_title('Histogram')

    axs[1][0].imshow(blurred_image_2, cmap='gray')
    axs[1][0].set_title('Three times')
    axs[1][0].axis('off')
    axs[1][1].hist(flat_list_2, 256, range=(0, 255))
    fig.suptitle('Applying mean filter')


def plot_task_4(blurred_image_2, binary_image):
    fig, axs = pyplot.subplots(1, 2)
    axs[0].imshow(blurred_image_2, cmap='gray')
    axs[0].set_title('blurred_image')
    axs[0].axis('off')
    axs[1].imshow(binary_image, cmap='gray')
    axs[1].set_title('binary_image')
    axs[1].axis('off')
    fig.suptitle('Threshold the Image')


def polt_task_5(binary_image, dilated_image, eroded_image):
    fig, axs = pyplot.subplots(1, 3)
    axs[0].imshow(binary_image, cmap='gray')
    axs[0].set_title('binary_image')
    axs[0].axis('off')
    axs[1].imshow(dilated_image, cmap='gray')
    axs[1].set_title('dilated_image')
    axs[1].axis('off')
    axs[2].imshow(eroded_image, cmap='gray')
    axs[2].set_title('eroded_image')
    axs[2].axis('off')
    fig.suptitle('Dilation and Erosion ')



# Step1 functions
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


# Step2 functions
def flatten_pixel_array(pixel_array):
    flattened_pixel_array = []
    for row in pixel_array:
        for col in row:
            flattened_pixel_array.append(col)
    return flattened_pixel_array


def percentile_based_mapping(pixel_array, image_width, image_height):
    contrast_stretched_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    size = image_height * image_width
    sorted_pixel_array = sorted(flatten_pixel_array(pixel_array))
    q_alpha = sorted_pixel_array[size * 5 // 100]
    q_beta = sorted_pixel_array[size * 95 // 100]

    for row in range(image_height):
        for col in range(image_width):
            f = pixel_array[row][col]
            g = ((f - q_alpha) * 255) / (q_beta - q_alpha)
            contrast_stretched_pixel_array[row][col] = g
    return contrast_stretched_pixel_array


# Step3 functions
def apply_filter(pixel_array, image_width, image_height, kernel, iterations=1):
    if iterations == 0:
        return pixel_array
    filtered_image = createInitializedGreyscalePixelArray(image_width, image_height)
    kernel_size = len(kernel)
    kernel_center = len(kernel) // 2

    for row in range(kernel_center, image_height - kernel_center):
        for col in range(kernel_center, image_width - kernel_center):
            filtered_value = 0
            for i in range(kernel_size):
                for j in range(kernel_size):
                    filtered_value += pixel_array[row + i - kernel_center][col + j - kernel_center] * kernel[i][j]
            filtered_image[row][col] = filtered_value
    return apply_filter(filtered_image, image_width, image_height, kernel, iterations-1)


def absolute_diff(pixel_array_1, pixel_array_2, image_width, image_height):
    abs_difference = createInitializedGreyscalePixelArray(image_width, image_height)

    for row in range(image_height):
        for col in range(image_width):
            abs_difference[row][col] = abs(pixel_array_1[row][col] + pixel_array_2[row][col])

    return abs_difference


# Step4 functions
def threshold(image, image_width, image_height, threshold):
    for row in range(image_height):
        for col in range(image_width):
            if image[row][col] < threshold:
                image[row][col] = 0
            elif image[row][col] >= threshold:
                image[row][col] = 255
    return image


# Step5 functions
def dilation(image, image_width, image_height, kernel, iterations=1):
    if iterations == 0:
        return image

    kernel_height, kernel_width = len(kernel), len(kernel)
    kernel_center = len(kernel)//2

    output = createInitializedGreyscalePixelArray(image_width, image_height)

    for row in range(kernel_center, image_height - kernel_center):
        for col in range(kernel_center, image_width - kernel_center):
            try:
                for i in range(kernel_height):
                    for j in range(kernel_width):
                        if kernel[i][j] == 1 and image[row + i - kernel_center][col + j - kernel_center] != 0:
                            raise IndexError
                output[row][col] = 0
            except IndexError:
                output[row][col] = 255

    return dilation(output, image_width, image_height, kernel, iterations - 1)


def erosion(image, image_width, image_height, kernel, iterations):
    if iterations == 0:
        return image

    kernel_height, kernel_width = len(kernel), len(kernel[0])
    kernel_center = len(kernel)//2

    output = createInitializedGreyscalePixelArray(image_width, image_height)

    for row in range(kernel_center, image_height - kernel_center):
        for col in range(kernel_center, image_width - kernel_center):
            try:
                for i in range(kernel_height):
                    for j in range(kernel_width):
                        if kernel[i][j] == 1 and image[row + i - kernel_center][col + j - kernel_center] == 0:
                            raise IndexError
                output[row][col] = 255
            except IndexError:
                output[row][col] = 0
    return erosion(output, image_width, image_height, kernel, iterations - 1)


# Step6 functions
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    num_region = 1
    detected_regions = {}

    for row in range(1, image_height):
        for col in range(1, image_width):
            if pixel_array[row][col] != 0 and pixel_array[row][col] != -999:
                q = Queue()
                q.enqueue((row, col,))
                pixel_array[row][col] = -999
                while not q.isEmpty():
                    (x_cor, y_cor) = q.dequeue()
                    if num_region in detected_regions.keys():
                        detected_regions[num_region].append((x_cor, y_cor))
                    else:
                        detected_regions[num_region] = [(x_cor, y_cor)]

                    if pixel_array[x_cor + 1][y_cor] != 0 and pixel_array[x_cor + 1][y_cor] != -999:
                        q.enqueue((x_cor + 1, y_cor,))
                        pixel_array[x_cor + 1][y_cor] = -999
                    if pixel_array[x_cor - 1][y_cor] != 0 and pixel_array[x_cor - 1][y_cor] != -999:
                        q.enqueue((x_cor - 1, y_cor,))
                        pixel_array[x_cor - 1][y_cor] = -999
                    if pixel_array[x_cor][y_cor + 1] != 0 and pixel_array[x_cor][y_cor + 1] != -999:
                        q.enqueue((x_cor, y_cor + 1,))
                        pixel_array[x_cor][y_cor + 1] = -999

                    if pixel_array[x_cor][y_cor - 1] != 0 and pixel_array[x_cor][y_cor - 1] != -999:
                        q.enqueue((x_cor, y_cor - 1,))
                        pixel_array[x_cor][y_cor - 1] = -999
                num_region += 1
    return detected_regions


# Step7 functions
def get_bounding_boxes(regions):
    bounding_box_list = []

    for i in regions.keys():
        y_cor = [j[0] for j in regions[i]]
        x_cor = [j[1] for j in regions[i]]

        min_x_cor = min(x_cor)
        min_y_cor = min(y_cor)
        max_x_cor = max(x_cor)
        max_y_cor = max(y_cor)
        if 200000 > (max_x_cor - min_x_cor) * (max_y_cor - min_y_cor) > 30000:
            bounding_box_list.append((min_x_cor, min_y_cor, max_x_cor, max_y_cor))

    return bounding_box_list


# This is our code skeleton that performs the coin detection.
def main(input_path, output_path):
    # This is the default input image, you may change the 'image_name' variable to test other images.
    image_name = 'easy_case_4'
    input_filename = f'./Images/easy/{image_name}.png'
    if TEST_MODE:
        input_filename = input_path

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    ###################################
    ### STUDENT IMPLEMENTATION Here ###
    ###################################

    # Step1 Convert to greyscale and normalize
    grey_scaled_image = convert_RGB_to_Grayscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    stretched_image = percentile_based_mapping(grey_scaled_image, image_width, image_height)

    # Step2 Edge detection
    laplacian_kernel_x = [[0,  1, 0],
                          [1, -4, 1],
                          [0,  1, 0]]
    laplacian_kernel_y = [[1,  1, 1],
                          [1, -8, 1],
                          [1,  1, 1]]

    scharred_x_image = apply_filter(stretched_image, image_width, image_height, laplacian_kernel_x)
    scharred_y_image = apply_filter(stretched_image, image_width, image_height, laplacian_kernel_y)
    absolute_diierences = absolute_diff(scharred_x_image, scharred_y_image, image_width, image_height)

    # Step3 Image blurring
    mean_kernel = [[0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04]]

    blurred_image = apply_filter(absolute_diierences, image_width, image_height, mean_kernel)

    # Step4 Threshold the Image
    binary_image = threshold(blurred_image, image_width, image_height, 180)

    # Step5 Erosion and Dilation
    circular_kernel = [[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]]

    dilated_image = dilation(binary_image, image_width, image_height, circular_kernel, iterations=6)
    eroded_image = erosion(dilated_image, image_width, image_height, circular_kernel, iterations=4)

    # Step6 Connected Component Analysis
    detected_regions = computeConnectedComponentLabeling(eroded_image, image_width, image_height)

    # Step7 Draw Bounding Box
    bounding_box_list = get_bounding_boxes(detected_regions)



    print(f"Detected {len(bounding_box_list)} coin(s) in {input_filename}")

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

        area = (bbox_max_x - bbox_min_x) * (bbox_max_y - bbox_min_y)
        if 63000 < area < 67000:
            axs.text(bbox_min_x, bbox_max_y + 30, "50cents", fontsize=12, color='blue')
        elif 55000 < area < 57000:
            axs.text(bbox_min_x, bbox_max_y + 30, "1dollar", fontsize=12, color='blue')

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
