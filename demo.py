def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_pixel_array = []
    for _ in range(image_height):
        new_row = []
        for _ in range(image_width):
            new_row.append(initValue)
        new_pixel_array.append(new_row)

    return new_pixel_array


def computeHistogram(pixel_array, image_width, image_height, nr_bins = 8):
    histogram_counts = [0 for i in range(nr_bins)]
    for i in range(image_height):
        for j in range(image_width):
            histogram_counts[pixel_array[i][j]] += 1

    return histogram_counts


def computeCumulativeHistogram(pixel_array, image_width, image_height, nr_bins):
    histogram_counts = computeHistogram(pixel_array, image_width, image_height, nr_bins)
    cumulative_counts = []
    for i in range(nr_bins):
        if i == 0:
            cumulative_counts.append(histogram_counts[i])
        else:
            cumulative_counts.append(histogram_counts[i] + cumulative_counts[i-1])
    return cumulative_counts



def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            r = pixel_array_r[i][j]
            g = pixel_array_g[i][j]
            b = pixel_array_b[i][j]

            greyscale_pixel_array[i][j] = round(0.299 * r + 0.587 * g + 0.114 * b)

    return greyscale_pixel_array

import numpy as np

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
            contrast_stretched_pixel_array[row][col] = round(g)
    return contrast_stretched_pixel_array


# 输入图像
image = np.array([
    [16, 32, 16, 16, 32],
    [32, 8, 16, 64, 32],
    [64, 64, 32, 16, 16],
    [32, 64, 64, 16, 64],
    [32, 32, 32, 8, 8]
])

print(image)
# 进行百分位数映射
mapped_image = percentile_based_mapping(image, 5, 5)
print(mapped_image)
