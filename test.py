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




# Create a simple image
pupil = np.zeros((100, 100))

x, y = np.indices(pupil.shape) #x is the index of each row and y is the index of each column
center = [pupil.shape[0] // 2, pupil.shape[1] // 2]
print(x)
print(y)

r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
print(r)

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
print(otf.shape)
plt.figure()
plt.plot(np.abs(otf[:, (100 - 1) // 2]))



plt.show()




