import matplotlib.pyplot as plt
import numpy as np


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


# Create a simple image
image = np.zeros((100, 100))
image[40:60, 40:60] = 1  # white square

# FFT without shift
fft_result = np.fft.fft2(image)
magnitude = np.abs(fft_result)

# FFT with shift
fft_shifted = np.fft.fftshift(fft_result)
magnitude_shifted = np.abs(fft_shifted)

# The shifted version has DC (brightest spot) in the CENTER
# The unshifted version has DC in the CORNERS

plt.figure()

plt.subplot(121)
plt.imshow(magnitude)
plt.subplot(122)
plt.imshow(magnitude_shifted)
print(magnitude.shape)
plt.show()




