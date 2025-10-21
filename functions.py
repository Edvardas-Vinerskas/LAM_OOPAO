from idlelib.editor import darwin

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from numpy.fft import fft2, fftshift


import OOPAO
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Pyramid import Pyramid
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Zernike import Zernike
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.tools.displayTools import displayMap, makeSquareAxes



# returns the distance to pixel value [x, y] from the centre at position [x, y]
def circle_radius(shape):
    x, y = np.indices(shape) # returns indices of a grid of shape == shape
    center = [shape[0] // 2, shape[1] // 2]
    radius = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    radius = radius.astype(int)
    return radius


# calculates the circular average
def circular_average(shape, psf_or_otf):
    radius = circle_radius(shape)
    pixel_sum_at_r = np.bincount(radius.ravel(), psf_or_otf.ravel())  # weighted sum of pixels at the same radius
    number_of_pixels_at_r = np.bincount(radius.ravel())
    average_sum = pixel_sum_at_r / number_of_pixels_at_r
    radial_number = np.arange(len(average_sum))
    return radial_number, average_sum







