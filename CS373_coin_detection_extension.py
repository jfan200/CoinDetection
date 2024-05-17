import numpy as np


def createInitializedGreyscalePixelArray(image_width, image_height, initValue=0):
    new_pixel_array = []
    for _ in range(image_height):
        new_row = []
        for _ in range(image_width):
            new_row.append(initValue)
        new_pixel_array.append(new_row)

    return new_pixel_array



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
def apply_3x3_filter(pixel_array, image_width, image_height, filter):
    filtered_image = createInitializedGreyscalePixelArray(image_width, image_height)

    for row in range(1, image_height - 1):
        for col in range(1, image_width - 1):
            filtered_value = 0
            for i in range(3):
                for j in range(3):
                    filtered_value += pixel_array[row + i - 1][col + j - 1] * filter[i][j]
            filtered_image[row][col] = filtered_value
    return filtered_image

def abs_diff(image1, image2):

    # 计算绝对差异
    image_height = len(image1)
    image_width = len(image1[0])
    abs_difference = createInitializedGreyscalePixelArray(image_width, image_height, 0)

    for row in range(image_height):
        for col in range(image_width):
            abs_difference[row][col] = abs(image1[row][col]-image2[row][col])

    return abs_difference


# Task 3 Image blurring
def apply_5x5_filter(pixel_array, image_width, image_height, filter):
    filtered_image = createInitializedGreyscalePixelArray(image_width, image_height)

    for row in range(2, image_height - 2):
        for col in range(2, image_width - 2):
            filtered_value = 0
            for i in range(5):
                for j in range(5):
                    filtered_value += pixel_array[row + i - 2][col + j - 2] * filter[i][j]
            filtered_image[row][col] = filtered_value
    return filtered_image

# Task 4 Threshold
def custom_threshold(image, image_width, image_height, threshold_1, threshold_2):
    for row in range(image_height):
        for col in range(image_width):
            if image[row][col] < threshold_1:
                image[row][col] = 0
            elif image[row][col] > threshold_2:
                image[row][col] = 255
    return image

# Task 6
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

def computeConnectedComponentLabeling(image, image_width, image_height):

    # Add a border with value of pixel 0
    image = [[0] * image_width] + image + [[0] * image_width]
    for index in range(len(image)):
        image[index] = [0] + image[index] + [0]

    num = 1
    regions = {}

    for row in range(1, image_height):
        for col in range(1, image_width):
            if image[row][col] != 0 and image[row][col] != -999:
                q = Queue()
                q.enqueue((row, col,))
                image[row][col] = -999
                while not q.isEmpty():
                    (a, b,) = q.dequeue()
                    if num not in regions.keys():
                        regions[num] = [(a, b,)]
                    else:
                        regions[num].append((a, b,))

                    if image[a - 1][b] != 0 and image[a - 1][b] != -999:
                        q.enqueue((a - 1, b,))
                        image[a - 1][b] = -999
                    if image[a + 1][b] != 0 and image[a + 1][b] != -999:
                        q.enqueue((a + 1, b,))
                        image[a + 1][b] = -999
                    if image[a][b + 1] != 0 and image[a][b + 1] != -999:
                        q.enqueue((a, b + 1,))
                        image[a][b + 1] = -999
                    if image[a][b - 1] != 0 and image[a][b - 1] != -999:
                        q.enqueue((a, b - 1,))
                        image[a][b - 1] = -999
                num += 1
    return regions

# Task 7
def detect_the_region(regions):
    print()
    print("Start detect the region from the image...")
    bounding_box_list = []

    for i in regions.keys():
        y_cor = [j[0] for j in regions[i]]
        x_cor = [j[1] for j in regions[i]]

        min_x_cor = min(x_cor)
        min_y_cor = min(y_cor)
        max_x_cor = max(x_cor)
        max_y_cor = max(y_cor)
        if (max_x_cor - min_x_cor) > 20 and (max_y_cor - min_y_cor) > 20:
            bounding_box_list.append((min_x_cor, min_y_cor, max_x_cor, max_y_cor))


    print(bounding_box_list)
    print("The region has been detected from the image! ")

    return bounding_box_list
