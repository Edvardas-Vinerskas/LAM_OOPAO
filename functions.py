
import numpy as np




# returns the distance to pixel value [x, y] from the centre at position [x, y]
def circle_radius(shape):
    x, y = np.indices(shape) # returns indices of a grid of shape == shape
    center = [shape[0] // 2, shape[1] // 2]
    radius = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    radius = radius.astype(int)
    return radius


# calculates the circular average
# need shape of the array and the PSF/OTF
def circular_average(shape, psf_or_otf):
    radius = circle_radius(shape)
    pixel_sum_at_r = np.bincount(radius.ravel(), psf_or_otf.ravel())  # weighted sum of pixels at the same radius
    number_of_pixels_at_r = np.bincount(radius.ravel())
    average_sum = pixel_sum_at_r / number_of_pixels_at_r
    average_sum = average_sum / np.max(average_sum)
    radial_number = np.arange(len(average_sum)) / len(average_sum)
    return radial_number, average_sum







