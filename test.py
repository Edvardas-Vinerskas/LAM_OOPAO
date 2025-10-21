import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from numpy.fft import fft2, fftshift
from functions import *

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


pupil = np.zeros((101, 101))
radius_array = circle_radius(pupil.shape)


#we calculate here for a circular pupil
#also maybe redo it using oopao
print(radius_array)
error

pupil[40:60, 40:60] = 1  # white square

# FFT without shift
psf = fftshift(fft2(pupil))
psf = np.abs(psf) ** 2
psf /= np.sum(psf)



# The shifted version has DC (brightest spot) in the CENTER
# The unshifted version has DC in the CORNERS

plt.figure()

plt.imshow(psf)
plt.colorbar()

otf = fftshift(fft2(fftshift(psf)))

x_axis, otf_averaged = circular_average((np.abs(otf)).shape, np.abs(otf))
print(otf.shape)
plt.figure()
plt.plot(x_axis, otf_averaged)
plt.yscale('log')
plt.xscale('log')



plt.show()




