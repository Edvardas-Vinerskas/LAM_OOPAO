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

ngs = Source(magnitude = 5,
             optBand   ='I')


sensing_wavelength = ngs.wavelength      # sensing wavelength of the WFS, read from the ngs object
n_subaperture      = 20                  # number of subaperture accross the diameter
diameter           = 8                   # diameter of the support of the phase screens in [m]
resolution         = n_subaperture*8     # resolution of the phase screens in pixels
pixel_size         = diameter/resolution # size of the pixels in [m]
obs_ratio          = 0.1                 # central obstruction in fraction of the telescope diameter
sampling_time      = 1/1000              # sampling time of the AO loop in [s]

# initialize the telescope object
tel = Telescope(diameter          = diameter,
               resolution         = resolution,
               centralObstruction = obs_ratio,
               samplingTime       = sampling_time,
                fov= 0)




nModes= 200

Zer = Zernike(tel, nModes)
Zer.computeZernike(tel)

plt.figure()
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(Zer.modesFullRes[:, :, i])
#displayMap(Zer.modesFullRes)
#plt.show()

plt.figure()
plt.imshow(Zer.modes.T @ Zer.modes / tel.pixelArea)
plt.title("Cross product matrix")


ngs * tel


# compute the pseudo inverse of the zernike polynomials (least square minimization)
Z_inv = np.linalg.pinv(Zer.modes)

# create the Atmosphere object
atm=Atmosphere(telescope     = tel,\
               r0            = 0.15,\
               L0            = 25,\
               windSpeed     = [10],\
               fractionalR0  = [1],\
               windDirection = [10],\
               altitude      = [0])



# initialize atmosphere
atm.initializeAtmosphere(tel)

plt.figure()
plt.imshow(atm.OPD)
tel+atm
tel.computePSF(zeroPaddingFactor=8)
plt.figure()
plt.imshow((np.abs(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.xlabel('Angular Resolution [arcsec]')
plt.ylabel('Angular Resolution [arcsec]')



# compute the pseudo inverse of the zernike polynomials (least square minimization)
Z_inv = np.linalg.pinv(Zer.modes)

# get the modal coefficients corresponding to a given phase screen of the atmosphere in [m]

    # 1)reshape OPD of the atmosphere in 1D and truncated to the pupil area
OPD_atm =atm.OPD[np.where(tel.pupil==1)]

    # 2) multiply using Z_inv
coef_atm = Z_inv@OPD_atm


# reconstruct the atm.OPD using the coef_atm
OPD_atm_rec =  tel.OPD = np.squeeze(Zer.modesFullRes@coef_atm)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(atm.OPD)
plt.title('Atmosphere phase screen [m]')
plt.subplot(1,3,2)
plt.imshow(OPD_atm_rec)
plt.title('Fitted by modal basis [m]')

plt.subplot(1,3,3)
plt.plot(1e9*coef_atm)
plt.xlabel('Zernike Modes')
plt.ylabel('[nm]')
plt.title('Modal decomposition of the phase screen')
makeSquareAxes(plt.gca())

plt.show()







