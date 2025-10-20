"""
Testing for different src and ngs (next step is extended source?)

LIST OF ARGUMENTS TO TEST:
* Calculate and plot the error terms for your AO system
* Play around with the light_ratio parameter for PWFS
* How to find out how many Zernike/KL modes you need? (connected to number of actuators/cut off frequency?)
* Write down stuff like interaction matrices and how PSF are calculated on paper
* How to optimize for resolution?
* Build a small RL model? (CNN possibly)
* In the real AO bench they had a gain sensing camera? how does that incorporate here (especially papytwin I guess)
* Calculate the spatial cutoff frequency
* WHAT DOES THIS REPRESENT/??? print(np.linalg.pinv(np.squeeze(dm.modes[tel.pupilLogical, :])).shape)
* plot running averages of SR (compare marechal approximation with the PSF estimate)
* PLOT DIFFERENCE BETWEEN ORIGINAL AND CORRECTED WAVEFRONT
* test the increasing subaperture number for shwfs (same result)


* reproject residual phase on Zernike modes
* plot PSD vs modes/frequency


SOME ANSWERS:
* actuators number = n_subap (or at least when you use it for the dm)
* for zonal you just use the identity of M2C
* What is the Z.modesFullRes vs Z.modes?
    these store the zernike polynomial values
    Z.modesFullRes just reformats the polynomial values according to the mirror resolution
* test the nsubap influence on what outputs you get for the pwfs
    resolution of the pwfs cam


REGARDING THE SUBAPERTURE THING, YOU CAN TEST FOR TEMPORAL ERROR BY REMOVING IT
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftshift, fft2 #need to shift just because of formatting

import OOPAO
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Pyramid import Pyramid
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.Zernike import Zernike


N_SUBAPERTURE  = 30
DIAMETER       = 1.52
RESOLUTION     = N_SUBAPERTURE * 6
FREQUENCY      = 1000
FOV            = 10
MODULATION     = 2
LIGHT_RATIO    = 0.1
POST_PROCESS   = "slopesMaps"
R_0            = 0.1
L_0            = 25
WIND_SPEED     = [40, 50, 100]
WIND_DIRECTION = [0, 100, 160]
FRACTIONAL_C_N2= [0.5, 0.3, 0.2]
ALTITUDE       = [0, 4500, 10000]

#CRUCIAL PIXEL SIZE CHECK#
pixel_size = DIAMETER / RESOLUTION


if (3 * pixel_size) > R_0:
    raise SystemExit("ERROR: pixel size is too big for r_0 value")


#SOURCE#

NGS = Source(optBand     = "I",
             magnitude   = 4)

SRC = Source(optBand     = "V",
             magnitude   = 8,
             coordinates = [2, 30])

#TELESCOPE#
TEL = Telescope(resolution          = RESOLUTION,
                diameter            = DIAMETER,
                samplingTime        = 1 / FREQUENCY,
                centralObstruction  = 0.1,
                fov                 = FOV)

NGS*TEL

#ATMOSPHERE#
ATMOSPHERE = Atmosphere(telescope    = TEL,
                        r0           = R_0,
                        L0           = L_0,
                        windSpeed    = WIND_SPEED,
                        windDirection= WIND_DIRECTION,
                        fractionalR0 = FRACTIONAL_C_N2,
                        altitude     = ALTITUDE
                        )

ATMOSPHERE.initializeAtmosphere(telescope = TEL)
TEL + ATMOSPHERE

#DEFORMABLE_MIRROR#

DM = DeformableMirror(telescope    = TEL,
                      nSubap       = N_SUBAPERTURE,
                      mechCoupling = 0.45)

#PWFS#

PWFS = Pyramid(nSubap         = N_SUBAPERTURE,
               telescope      = TEL,
               modulation     = MODULATION,
               lightRatio     = LIGHT_RATIO,
               postProcessing = POST_PROCESS)

#ZONAL/MODAL FUNCTIONS#
ZERNIKE = Zernike(telObject = TEL,
                  J         = 300)

ZERNIKE.computeZernike(telObject2 = TEL)
print(ZERNIKE.modes)
print(ZERNIKE.modes.shape)
print(ZERNIKE.modesFullRes.shape) #[TEL.resolution, TEL.resolution, J]
error

M2C = np.linalg.pinv(np.squeeze(DM.modes[TEL.pupilLogical, :])) @ ZERNIKE.modes
#INTERACTION MATRIX#

CALIBRATION_MATRIX = InteractionMatrix(ngs            = NGS,
                                       tel            = TEL,
                                       dm             = DM,
                                       wfs            = PWFS,
                                       M2C            = M2C,
                                       atm            = ATMOSPHERE,
                                       nMeasurements  = 1)






