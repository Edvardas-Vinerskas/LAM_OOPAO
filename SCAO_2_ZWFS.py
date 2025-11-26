
"""
* need to also implement different loop frequencies
    * I think you can just run the zernike loop multiple times so that it improves the same
    * for now the sampling is implemented manually


"""


import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Circle
import numpy as np
from numpy.fft import fftshift, fft, fft2, fftfreq, rfft, rfftfreq #need to shift just because of formatting

import OOPAO
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Pyramid import Pyramid
from OOPAO.ZWFS import ZWFS
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.Zernike import Zernike
from OOPAO.Detector import Detector

from functions import *


#---------------------------------------------------GLOBALS---------------------------------------------------#
#define all OOPAO variables

N_SUBAPERTURE_pyr   = 20
N_SUBAPERTURE_zer   = 9
DIAMETER            = 1.52
CENTRAL_OBSTRUCTION = 0 #0.15
RESOLUTION          = N_SUBAPERTURE_pyr * 8
FREQUENCY           = 1500
SAMPLING_TIME       = 1/FREQUENCY
FOV                 = 10
MECHANICAL_COUPLING = 0.35
MODULATION          = 3
LIGHT_RATIO         = 0.1
POST_PROCESS        = "slopesMaps"
r_0                 = 0.15
L_0                 = 25
WIND_SPEED          = [10, 20, 60] #[10, 20, 60]
WIND_DIRECTION      = [0, 100, 160] #[0, 100, 160]
FRACTIONAL_C_N2     = [0.6, 0.3, 0.1] #[0.5, 0.3, 0.2]
ALTITUDE            = [0, 4500, 10000] #[0, 4500, 10000]
Z_coefs             = 200 #200 #above 200 does not work great per Benoit



zeroPaddingFactor = 4
rad2arcsec        = 180 * 60 * 60 / np.pi


use_pwfs          = True


use_zonal         = False
use_zernike       = False
use_KL            = True



#pixel_size check for sufficient r_0 sampling
pixel_size = DIAMETER / RESOLUTION
if (3 * pixel_size) > r_0:
    raise SystemExit("ERROR: pixel size is too big for r_0 value")


#---------------------------------------------------SOURCE---------------------------------------------------#

NGS = Source(optBand     = "I",
             magnitude   =  2)

SRC = Source(optBand     = "I",
             magnitude   =  2)

wvl = 500e-9 #r_0 specified above is for this wavelength
r_0_src = r_0 * (SRC.wavelength / wvl) ** (6/5)
r_0_ngs = r_0 * (NGS.wavelength / wvl) ** (6/5)

#---------------------------------------------------TELESCOPE---------------------------------------------------#
TEL = Telescope(resolution          = RESOLUTION,
                diameter            = DIAMETER,
                samplingTime        = SAMPLING_TIME,
                centralObstruction  = CENTRAL_OBSTRUCTION)
                #fov                 = FOV)

#MUST couple source object to telescope
SRC * TEL

TEL.computePSF(zeroPaddingFactor)
#diffraction limited OTF calculation
OTF_dl = fftshift(fft2(fftshift(TEL.PSF / np.sum(TEL.PSF))))
x_axis, OTF_dl_averaged = circular_average((np.abs(OTF_dl)).shape, np.abs(OTF_dl))

#---------------------------------------------------ATMOSPHERE---------------------------------------------------#
ATM = Atmosphere(telescope    = TEL,
                        r0           = r_0,
                        L0           = L_0,
                        windSpeed    = WIND_SPEED,
                        windDirection= WIND_DIRECTION,
                        fractionalR0 = FRACTIONAL_C_N2,
                        altitude     = ALTITUDE
                        )

ATM.initializeAtmosphere(telescope = TEL)
def temp_vib(k, f, t, shape):
    k = k
    f = f
    x = np.linspace(-np.pi, np.pi, shape[0])
    vibration = np.cos(2 * np.pi * (k * x - f * t)) * np.ones(shape)
    return vibration



#---------------------------------------------------DEFORMABLE_MIRROR---------------------------------------------------#

DM_pyr = DeformableMirror(telescope    = TEL,
                      nSubap       = N_SUBAPERTURE_pyr,
                      mechCoupling = MECHANICAL_COUPLING)

DM_zer = DeformableMirror(telescope    = TEL,
                      nSubap       = N_SUBAPERTURE_zer,
                      mechCoupling = MECHANICAL_COUPLING)


#control radius calculation for dm
control_radius_1 = ((N_SUBAPERTURE_pyr + 1) * SRC.wavelength) /(2 * TEL.D) * rad2arcsec
control_radius_2 = ((N_SUBAPERTURE_zer + 1) * SRC.wavelength) /(2 * TEL.D) * rad2arcsec
corr_zone_1 = Circle([0,0], control_radius_1, fc='none', ec='w', ls=':')
corr_zone_2 = Circle([0,0], control_radius_2, fc='none', ec='w', ls=':')

#pitch check
if (2 * DM_pyr.pitch) > r_0_src:
    raise SystemExit(f"ERROR: DM actuator density is insufficient for r_0 {r_0_src}, dm pitch {DM_pyr.pitch} ")

#---------------------------------------------------WFS---------------------------------------------------#
PWFS = Pyramid(nSubap         = N_SUBAPERTURE_pyr,
                   telescope      = TEL,
                   modulation     = MODULATION,
                   lightRatio     = LIGHT_RATIO,
                   postProcessing = POST_PROCESS)


zernike_WFS = ZWFS(tel = TEL, zpf = 8)
zernike_WFS.cam = Detector(round(N_SUBAPERTURE_zer*zeroPaddingFactor))



#---------------------------------------------------MODAL_BASIS---------------------------------------------------#
M2C_pyr   = None
modes_pyr = None

M2C_zer   = None
modes_zer = None
if use_zonal:
    M2C_zonal_pyr = np.identity(DM_pyr.nValidAct)
    M2C_pyr       = M2C_zonal_pyr
    modes_pyr     = DM_pyr.modes

    M2C_zonal_zer = np.identity(DM_zer.nValidAct)
    M2C_zer = M2C_zonal_zer
    modes_zer = DM_zer.modes


if use_zernike:

    zernike = Zernike(telObject = TEL,
                       J = Z_coefs)
    zernike.computeZernike(telObject2 = TEL)

    M2C_Z_pyr = np.linalg.pinv(np.squeeze(DM_pyr.modes[TEL.pupilLogical, :])) @ zernike.modes
    M2C_pyr   = M2C_Z_pyr
    modes_pyr = zernike.modes

    M2C_Z_zer = np.linalg.pinv(np.squeeze(DM_zer.modes[TEL.pupilLogical, :])) @ zernike.modes
    M2C_zer = M2C_Z_zer
    modes_zer = zernike.modes

if use_KL:

    M2C_KL_pyr = compute_KL_basis(tel = TEL, atm = ATM, dm = DM_pyr)
    M2C_pyr    = M2C_KL_pyr[:, :200]
    modes_pyr  = DM_pyr.modes @ M2C_pyr

    M2C_KL_zer = compute_KL_basis(tel=TEL, atm=ATM, dm=DM_zer)
    M2C_zer = M2C_KL_zer[:, :200]
    modes_zer = DM_zer.modes @ M2C_zer

#---------------------------------------------------INTERACTION MATRIX---------------------------------------------------#



stroke = SRC.wavelength / 16
CALIB_pyr = InteractionMatrix(ngs        = SRC,
                          tel            = TEL,
                          dm             = DM_pyr,
                          wfs            = PWFS,
                          M2C            = M2C_pyr,
                          atm            = ATM,
                          nMeasurements  = 5,
                          stroke         = stroke,
                          noise          = "off")


CALIB_zer = InteractionMatrix(ngs        = SRC,
                          tel            = TEL,
                          dm             = DM_zer,
                          wfs            = zernike_WFS,
                          M2C            = M2C_zer,
                          atm            = ATM,
                          nMeasurements  = 1,
                          stroke         = stroke,
                          noise          = "off")


#---------------------------------------------------FITTING ERROR CALC---------------------------------------------------#

#for zernike and zonal only (later extract the KL modes from the source code?)
modes_inv = None
#the rest of the code is in the for loop

#takes in phase and outputs modes
if use_zonal:
    modes_inv = np.linalg.pinv(np.squeeze(modes_pyr[TEL.pupilLogical, :]))

if use_zernike:
    modes_inv = np.linalg.pinv(np.squeeze(modes_pyr))

if use_KL:
    modes_inv = np.linalg.pinv(np.squeeze(modes_pyr[TEL.pupilLogical, :]))


#the rest of the code is in the for loop

#---------------------------------------------------CAMERA---------------------------------------------------#

CAM = Detector(integrationTime = 100 * TEL.samplingTime,  # integration time of the detector
                photonNoise=False,  # enable photon noise
                readoutNoise=0,  # readout of the detector in [e-/pixel]
                QE=1,  # quantum efficiency
                psf_sampling=2,  # sampling for the PSF computation 2 = Shannon sampling
                binning=1)  # Binning factor of the PSF


#---------------------------------------------------SIMULATION---------------------------------------------------#

ATM.generateNewPhaseScreen(seed = 2)
#reset everything just in case
TEL.resetOPD()
DM_pyr.coefs = 0
DM_zer.coefs = 0
TEL + ATM
SRC * TEL * DM_pyr * PWFS * DM_zer * zernike_WFS
TEL.print_optical_path()

#delay implementation
frame_delay         = 2
delay               = frame_delay - 1 #frame delay of 1 is already built-in
if frame_delay >= 2:
    wfssignal_buffer = [np.zeros(PWFS.nSignal) for i in range(delay)]
else:
    wfssignal_buffer = []

#variables and performance metric initialisation
nLoop                        = 1000
sr                           = np.zeros(nLoop)
sr_running                   = np.zeros(nLoop)
total_error                  = np.zeros(nLoop)
residual_error               = np.zeros(nLoop)

#variables for PSD calculation
residual_OPD_list            = []
final_residual_OPD           = 0
final_atmosphere_OPD         = 0
final_dm_OPD                 = 0




#for temp_err_delay = 3, the list has current, previous and previous_previous
temp_err_delay               = 4
atm_OPD_list                 = [np.zeros(ATM.OPD.shape) for i in range((temp_err_delay + 1))] #(temp_err_delay + 1) tells you how many current + previous frames you want to keep in the buffer
sim_temp_error_2_frame_delay = []
sim_fit_error_list           = []
sim_fit_error_list_2D        = []


#vibration implementation




tel_psf_list = []


CL_gain_pyr = 0.4
CL_gain_zer = 0.4

reconstructor_pyr = M2C_pyr @ CALIB_pyr.M
reconstructor_zer = M2C_zer @ CALIB_zer.M

for i in range(nLoop):
    #update phase screen
    ATM.update()

    total_error[i] = np.std(TEL.OPD[np.where(TEL.pupil > 0)]) * 1e9

    #propagate through AO with the dm commands applied
    SRC * TEL * DM_pyr * PWFS * DM_zer * zernike_WFS
    #propagate to the source with the dm commands applied (old dm commands)
    #the point of this line is that for the wfs propagation you would be using NGS
    SRC * TEL


    #frame delay implementation
    wfssignal_buffer.append(PWFS.signal)
    pwfs_delayed_signal = wfssignal_buffer[0]
    wfssignal_buffer.pop(0)


    #update the dm commands
    if i % 3 == 0:
        DM_pyr.coefs = DM_pyr.coefs - CL_gain_pyr * np.matmul(reconstructor_pyr, PWFS.signal)
    DM_zer.coefs = DM_zer.coefs - CL_gain_zer * np.matmul(reconstructor_zer, zernike_WFS.signal)



    #performance metrics
    sr[i] = np.exp(-np.var(TEL.src.phase[np.where(TEL.pupil == 1)]))
    residual_error[i] = np.std(TEL.OPD[np.where(TEL.pupil > 0)]) * 1e9
    print("Loop" + str(i) + "/" + str(nLoop) + " " + "AO residual: " + str(residual_error[i]) + "nm" + "total err: " + str(total_error[i]) + "nm")
    print(f"strehl {sr[i]}")


    #if sr[i] > 0.5:
    residual_OPD_list.append(TEL.OPD)
    print("residual_OPD_append")


    TEL.computePSF(zeroPaddingFactor)
    tel_psf_list.append(TEL.PSF)





#---------------------------------------------------Error decomposition---------------------------------------------------#
time = np.arange(0, nLoop * SAMPLING_TIME, SAMPLING_TIME)
plt.figure()
plt.plot(time, residual_error, label = "residual")
#plt.plot(time, sim_temp_error_2_frame_delay, label = "simulational temporal error 2 frame delay")
plt.title("error decomposition (nm)")
plt.xlabel("time s")
plt.yscale("log")
plt.legend()







#---------------------------------------------------Strehl---------------------------------------------------#
sr_mean = np.mean(sr)
kernel = np.ones(30) / 30

# pad sr
pad_left = len(kernel) // 2
pad_right = len(kernel) - pad_left - 1
sr_padded = np.pad(sr, (pad_left, pad_right), mode='constant', constant_values=sr_mean)
sr_running = np.convolve(sr_padded, kernel, mode='valid')


plt.figure()
plt.plot(time, sr, label = "strehl")
plt.plot(time, sr_running, label = "running_strehl")
plt.title("Strehl ratio")
plt.xlabel("time s")
plt.ylim(bottom = (sr_mean - 0.3))
plt.legend()


#---------------------------------------------------AO PSF---------------------------------------------------#

plt.figure()
TEL.computePSF(zeroPaddingFactor)

arcsec_per_pixel = 206265 * (TEL.src.wavelength/TEL.D) / zeroPaddingFactor
N = 50
AO_PSF = TEL.PSF[N: -N, N: -N]
fov    = AO_PSF.shape[0] * arcsec_per_pixel
AO_PSF = AO_PSF/np.sum(AO_PSF)
plt.imshow(AO_PSF, norm = SymLogNorm(1e-6), extent = [-fov/2, fov/2, -fov/2, fov/2])
plt.title("AO corrected PSF")
plt.gca().add_artist(corr_zone_1)
plt.xlabel("arcsec")
plt.ylabel("arcsec")
plt.colorbar()

tel_psf_list = np.array(tel_psf_list)
AO_PSF_LE = np.mean(tel_psf_list, axis = 0)
AO_PSF_LE = AO_PSF_LE[N: -N, N: -N]
AO_PSF_LE = AO_PSF_LE / np.sum(AO_PSF_LE)
plt.figure()
plt.imshow(AO_PSF_LE, norm = SymLogNorm(1e-6), extent = [-fov/2, fov/2, -fov/2, fov/2])
plt.title("AO corrected PSF LE")
plt.gca().add_artist(corr_zone_2)
plt.xlabel("arcsec")
plt.ylabel("arcsec")
plt.colorbar()

plt.show()




























































































