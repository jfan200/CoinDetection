from matplotlib import pyplot
import imageIO

# Plot
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
    axs[1][0].set_title('Two times')
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



def createInitializedGreyscalePixelArray(image_width, image_height, initValue=0):
    new_pixel_array = []
    for _ in range(image_height):
        new_row = []
        for _ in range(image_width):
            new_row.append(initValue)
        new_pixel_array.append(new_row)

    return new_pixel_array
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
def percentile(pixel_array, percentile):
    # 将数据转换为列表（如果不是）并排序
    pixel_array = [pixel_array[row][col] for row in range(len(pixel_array)) for col in range(len(pixel_array[0]))]
    size = len(pixel_array)
    sorted_data = sorted(pixel_array)
    # 计算百分位数位置
    index = size * percentile // 100
    # 根据百分位数返回相应的数据值
    return sorted_data[index]

def percentile_based_mapping(pixel_array, image_width, image_height):
    contrast_stretched_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    percentiles = {
        '5th': percentile(pixel_array, 5),
        '95th': percentile(pixel_array, 95)
    }

    for row in range(image_height):
        for col in range(image_width):
            f = pixel_array[row][col]
            g = ((f - percentiles['5th']) * 255) / (percentiles['95th'] - percentiles['5th'])
            contrast_stretched_pixel_array[row][col] = g
    return contrast_stretched_pixel_array


# Task 2 Edge detection
def apply_filter(pixel_array, image_width, image_height, kernel):
    filtered_image = createInitializedGreyscalePixelArray(image_width, image_height)
    kernel_size = len(kernel)
    kernel_center = len(kernel) // 2

    for row in range(kernel_center, image_height - kernel_center): ## kernel_size 3x3 (1, height - 1)
        for col in range(kernel_center, image_width - kernel_center):
            filtered_value = 0
            for i in range(kernel_size):
                for j in range(kernel_size):
                    filtered_value += pixel_array[row + i - kernel_center][col + j - kernel_center] * kernel[i][j]
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


# Task 4 Threshold
def custom_threshold(image, image_width, image_height, threshold_1, threshold_2):
    for row in range(image_height):
        for col in range(image_width):
            if image[row][col] < threshold_1:
                image[row][col] = 0
            elif image[row][col] > threshold_2:
                image[row][col] = 255
    return image


# Task 5
def dilate_image(image, image_width, image_height, kernel, iterations):
    if iterations == 0:
        return image

    # 获取 kernel 尺寸和中心点
    k_height, k_width = len(kernel), len(kernel)
    kernel_center = len(kernel)//2

    # 创建输出图像数组，初始为零（黑色）
    output = createInitializedGreyscalePixelArray(image_width, image_height)

    # 遍历图像的每个像素
    for row in range(kernel_center, image_height - kernel_center):
        for col in range(kernel_center, image_width - kernel_center):
            pixel_values = []
            # 遍历 kernel
            try:
                for i in range(k_height):
                    for j in range(k_width):
                        if kernel[i][j] == 1 and image[row + i - kernel_center][col + j - kernel_center] != 0:
                            raise IndexError
                output[row][col] = 0
            except IndexError:
                output[row][col] = 255

    # 递归进行下一次膨胀
    return dilate_image(output, image_width, image_height, kernel, iterations - 1)

def erode_image(image, image_width, image_height, kernel, iterations):
    if iterations == 0:
        return image

    # 获取 kernel 尺寸和中心点
    k_height, k_width = len(kernel), len(kernel[0])
    kernel_center = len(kernel)//2

    output = createInitializedGreyscalePixelArray(image_width, image_height)

    # 遍历图像的每个像素
    for row in range(kernel_center, image_height - kernel_center):
        for col in range(kernel_center, image_width - kernel_center):
            # 遍历 kernel
            try:
                for i in range(k_height):
                    for j in range(k_width):
                        if kernel[i][j] == 1 and image[row + i - kernel_center][col + j - kernel_center] == 0:
                            raise IndexError
                output[row][col] = 255
            except IndexError:
                output[row][col] = 0
    return erode_image(output, image_width, image_height, kernel, iterations - 1)


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
    num_region = 1
    regions = {}

    for row in range(1, image_height):
        for col in range(1, image_width):
            if image[row][col] != 0 and image[row][col] != -999:
                q = Queue()
                q.enqueue((row, col,))
                image[row][col] = -999
                while not q.isEmpty():
                    (x_cor, y_cor) = q.dequeue()
                    if num_region not in regions.keys():
                        regions[num_region] = [(x_cor, y_cor)]
                    else:
                        regions[num_region].append((x_cor, y_cor))

                    if image[x_cor - 1][y_cor] != 0 and image[x_cor - 1][y_cor] != -999:
                        q.enqueue((x_cor - 1, y_cor,))
                        image[x_cor - 1][y_cor] = -999
                    if image[x_cor + 1][y_cor] != 0 and image[x_cor + 1][y_cor] != -999:
                        q.enqueue((x_cor + 1, y_cor,))
                        image[x_cor + 1][y_cor] = -999
                    if image[x_cor][y_cor + 1] != 0 and image[x_cor][y_cor + 1] != -999:
                        q.enqueue((x_cor, y_cor + 1,))
                        image[x_cor][y_cor + 1] = -999
                    if image[x_cor][y_cor - 1] != 0 and image[x_cor][y_cor - 1] != -999:
                        q.enqueue((x_cor, y_cor - 1,))
                        image[x_cor][y_cor - 1] = -999
                num_region += 1
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
        # if (max_x_cor - min_x_cor) > 50 and (max_y_cor - min_y_cor) > 50 and abs((max_x_cor - min_x_cor) - (max_y_cor - min_y_cor)) < 20:
        bounding_box_list.append((min_x_cor, min_y_cor, max_x_cor, max_y_cor))


    print("The region has been detected from the image! ")

    return bounding_box_list




