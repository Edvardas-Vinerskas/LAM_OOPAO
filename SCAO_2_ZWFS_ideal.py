"""

"""


import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Circle
import numpy as np
from scipy import signal
from numpy.fft import fftshift, fft, fft2, fftfreq, rfft, rfftfreq #need to shift just because of formatting

import OOPAO
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Pyramid import Pyramid
from OOPAO.ZWFS import ZWFS
from OOPAO.ZWFS2 import ZWFS2
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.Zernike import Zernike
from OOPAO.Detector import Detector

from functions import *

#NOW DOING WITHOUT ZERNIKE FOR COMPARISON
#---------------------------------------------------GLOBALS---------------------------------------------------#
#define all OOPAO variables

N_SUBAPERTURE_pyr   = 20
N_SUBAPERTURE_zer   = 9 #3.2 spatial cut off frequency
DIAMETER            = 1.52
CENTRAL_OBSTRUCTION = 0 #0.15
RESOLUTION          = N_SUBAPERTURE_pyr * 8
FREQUENCY           = 1500 #pyramid is running at 500
directory_name      = 'vZWFS_metrics_ideal_no_wind'
SAMPLING_TIME       = 1/FREQUENCY
FOV                 = 10
MECH_COUPLING       = 0.35
MODULATION          = 3
LIGHT_RATIO         = 0.1
POST_PROCESS        = "slopesMaps"
r_0                 = 0.15
L_0                 = 25
WIND_SPEED          = [0.1, 0.1] #[10, 20, 60]
WIND_DIRECTION      = [0, 100] #[0, 100, 160]
FRACTIONAL_C_N2     = [0.6, 0.4] #[0.5, 0.3, 0.2]
ALTITUDE            = [0, 4500] #[0, 4500, 10000]
Z_coefs             = 250 #200 #above 200 does not work great per Benoit



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
ATM = Atmosphere(telescope           = TEL,
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
                          mechCoupling = MECH_COUPLING)

DM_zer = DeformableMirror(telescope    = TEL,
                          nSubap       = N_SUBAPERTURE_zer,
                          mechCoupling = MECH_COUPLING)


#this is our DM mask that should be used inside the models
#mask = np.reshape(DM_zer.validAct, (10, 10)) 2D boolean mask of valid actuators
#woofer_tweeter correction
N_1 = np.squeeze(DM_pyr.modes[TEL.pupilLogical, :])
N_2_inv = np.linalg.pinv(np.squeeze(DM_zer.modes[TEL.pupilLogical, :]))
N2N1 = np.matmul(N_2_inv, N_1)


#control radius calculation for dm
control_radius_1 = ((N_SUBAPERTURE_pyr + 1) * SRC.wavelength) /(2 * TEL.D) * rad2arcsec
control_radius_2 = ((N_SUBAPERTURE_zer + 1) * SRC.wavelength) /(2 * TEL.D) * rad2arcsec
corr_zone_1 = Circle([0,0], control_radius_1, fc='none', ec='w', ls=':')
corr_zone_2 = Circle([0,0], control_radius_2, fc='none', ec='r', ls=':')
corr_zone_1_LE = Circle([0,0], control_radius_1, fc='none', ec='w', ls=':')
corr_zone_2_LE = Circle([0,0], control_radius_2, fc='none', ec='r', ls=':')

#pitch check
if (2 * DM_pyr.pitch) > r_0_src:
    raise SystemExit(f"ERROR: DM actuator density is insufficient for r_0 {r_0_src}, dm pitch {DM_pyr.pitch} ")

#---------------------------------------------------WFS---------------------------------------------------#
PWFS = Pyramid(nSubap         = N_SUBAPERTURE_pyr,
                   telescope      = TEL,
                   modulation     = MODULATION,
                   lightRatio     = LIGHT_RATIO,
                   postProcessing = POST_PROCESS)


#zernike_WFS = ZWFS(tel = TEL, zpf = 8) # zpf is the diameter/resolution of the zernike mask in the fourier plane
"""vZWFS = ZWFS2(tel         = TEL,
              zpf         = 30,
              diameter    = 2.14,
              phase_shift = [-0.75 * np.pi,0.3 * np.pi])
"""
vZWFS = ZWFS2(tel         = TEL,
              zpf         = 8,
              diameter    = 1.06,
              phase_shift = [-np.pi/2,np.pi/2])

#TODO later recheck the vZWFS sensor according to what Matthieu said
#zpf use 30 (size of the mask in pixels)
#diameter of 2.14
#phase shift 0.3 pi - -0.75 pi
#THE ABOVE ARE THE SYSTEM REPRESENTATIVE PARAMETER
#answer the question how is the dynamic range and sensitivity impacted by the
    #diameter
    #phase shift change
#look at wfs_measure function (are you supposed to use this one in interaction matrix?)

#there is arctan and arcsin reconstructor inside the updated zwfs classes
#and remember that the problem with these reconstructors that you did need iterative correction


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
    M2C_pyr    = M2C_KL_pyr[:, :250]
    modes_pyr  = DM_pyr.modes @ M2C_pyr

    M2C_KL_zer = compute_KL_basis(tel=TEL, atm=ATM, dm = DM_zer)
    M2C_zer = M2C_KL_zer[:, :250]
    modes_zer = DM_zer.modes @ M2C_zer

#---------------------------------------------------INTERACTION MATRIX---------------------------------------------------#



stroke = SRC.wavelength / 16
CALIB_pyr = InteractionMatrix(ngs        = SRC,
                          tel            = TEL,
                          dm             = DM_pyr,
                          wfs            = PWFS,
                          M2C            = M2C_pyr,
                          atm            = ATM,
                          nMeasurements  = 1,
                          stroke         = stroke,
                          noise          = "off")


#calibration for zernike (there are better methods out there for calibration under real conditions but this will suffice for now)
CALIB_zer = np.zeros((vZWFS.signal.shape[0], M2C_zer.shape[1]))


#zernike_wfs calibration with DM_zer
for i in range(M2C_zer.shape[1]):
    v_plus = M2C_zer[:, i] * stroke
    v_minus = -M2C_zer[:, i] * stroke


    SRC.reset()
    DM_zer.coefs = v_plus
    SRC ** TEL * DM_zer * vZWFS
    w_plus = vZWFS.signal


    SRC.reset()
    DM_zer.coefs = v_minus
    SRC ** TEL * DM_zer * vZWFS
    w_minus = vZWFS.signal


    CALIB_zer[:, i] = (w_plus - w_minus) / (2 * stroke)


# takes in modes and outputs wfs signal
#FOR NOW YOU ARE NOT TRUNCATING ANY EIGENVALUES? CHANGE THIS IN THE FUTURE
CALIB_zer_obj = CalibrationVault(CALIB_zer)


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
    modes_inv_pyr = np.linalg.pinv(np.squeeze(modes_pyr[TEL.pupilLogical, :]))
    modes_inv_zer = np.linalg.pinv(np.squeeze(modes_zer[TEL.pupilLogical, :]))



#---------------------------------------------------SIMULATION---------------------------------------------------#

ATM.generateNewPhaseScreen(seed = 10)
#reset everything just in case
SRC.reset()
DM_pyr.coefs = 0
DM_zer.coefs = 0
DM_zer_copy = 0
SRC ** ATM * TEL * DM_pyr * PWFS * DM_zer * vZWFS
TEL.print_optical_path()

#variables and performance metric initialisation
nLoop                        = 2000
sr                           = np.zeros(nLoop)
sr_1st                       = []
sr_running                   = np.zeros(nLoop)
total_error                  = np.zeros(nLoop)
residual_error               = np.zeros(nLoop)

#variables for PSD calculation
residual_OPD_list            = []
residual_OPD_list_pwfs       = []
#vibration implementation
atm_OPD_temp_list = []
atm_OPD_list = [np.zeros(ATM.OPD.shape) for i in range(3)]


tel_psf_list = []


CL_gain_pyr = 0.4
CL_gain_zer = 0.4

reconstructor_pyr = M2C_pyr @ CALIB_pyr.M #takes in slopes and outputs modes, then takes in modes and outputs controle shape(357, 664)
reconstructor_zer = M2C_zer @ CALIB_zer_obj.M #takes in wfs signal and outputs control
ATM.generateNewPhaseScreen(seed = 10)

PWFS_signal_avg = 0
PWFS_signal_avg_norm = 0
for i in range(nLoop):

    ATM.update()
    atm_opd = ATM.OPD.copy()
    atm_OPD_temp_list.append(atm_opd)
    atm_OPD_list.append(atm_opd)
    atm_OPD_list.pop(0)

    total_error[i] = np.std(TEL.OPD[np.where(TEL.pupil > 0)]) * 1e9

    #update the dm commands
    if (i + 1) % 3 == 0:
        pyr_atm_OPD = np.mean([atm_OPD_list[-3][np.where(TEL.pupil > 0)], atm_OPD_list[-2][np.where(TEL.pupil > 0)],
                               atm_OPD_list[-1][np.where(TEL.pupil > 0)]], axis=0)

        mode_coefs_pyr = modes_inv_pyr @ pyr_atm_OPD
        DM_pyr.coefs = -M2C_pyr @ mode_coefs_pyr
        DM_pyr_copy = DM_pyr.coefs.copy()



    mode_coefs_zer = modes_inv_zer @ atm_OPD_list[-1][np.where(TEL.pupil > 0)]
    DM_zer_coefs = -M2C_zer @ mode_coefs_zer
    DM_zer_copy = DM_zer_coefs.copy()


    #woofer_tweeter corr
    DM_zer.coefs = DM_zer_copy - np.matmul(N2N1, DM_pyr.coefs)

    # propagate through AO with the dm commands applied
    SRC * TEL * DM_pyr * PWFS
    if (i + 1) % 3 == 0:
        residual_OPD_list_pwfs.append(TEL.OPD)
        sr_1st.append(np.exp(-np.var(TEL.src.phase[np.where(TEL.pupil == 1)])))
    TEL * DM_zer * vZWFS


    #performance metrics
    sr[i] = np.exp(-np.var(TEL.src.phase[np.where(TEL.pupil == 1)]))
    residual_error[i] = np.std(TEL.OPD[np.where(TEL.pupil > 0)]) * 1e9
    print("Loop" + str(i) + "/" + str(nLoop) + " " + "AO residual: " + str(residual_error[i]) + "nm" + "total err: " + str(total_error[i]) + "nm")
    print(f"strehl {sr[i]}")


    residual_OPD_list.append(TEL.OPD)

    #when plotting high strehl PSF, just
    TEL.computePSF(zeroPaddingFactor)
    tel_psf_list.append(TEL.PSF)




residual_error_array = np.asarray(residual_error)
np.save(f"temp_save_dir/{directory_name}/residual_error", residual_error_array)

strehl_array_1st = np.asarray(sr_1st)
strehl_array_1st = np.repeat(strehl_array_1st, 3)
strehl_array_1st = np.insert(strehl_array_1st, 0, [0, 0])
np.save(f"temp_save_dir/{directory_name}/strehl_array_1st", strehl_array_1st)

strehl_array_2nd = np.asarray(sr)
np.save(f"temp_save_dir/{directory_name}/strehl_array_2nd", strehl_array_2nd)

tel_psf_array = np.asarray(tel_psf_list)
np.save(f"temp_save_dir/{directory_name}/tel_psf_array", tel_psf_array)

residual_OPD_array = np.asarray(residual_OPD_list) #for use in spatial PSD/KL modes var and correlation
#you can later do the spatial PSD when you save the required atm parameter
np.save(f"temp_save_dir/{directory_name}/residual_OPD_array", residual_OPD_array)

atm_OPD_array = np.asarray(atm_OPD_temp_list) #not sure where I would use it
np.save(f"temp_save_dir/{directory_name}/atm_OPD_array", atm_OPD_array)

total_err_array = np.asarray(total_error) #not useful for now
np.save(f"temp_save_dir/{directory_name}/total_err_array", total_err_array)

#YOU SHOULD ALSO CHECK IF YOU ONLY HAVE AN EPISODE OF DATA IN ALL OF THESE BECAUSE I DON'T REMEMBER
time_plot = np.arange(0, nLoop * SAMPLING_TIME, SAMPLING_TIME)
np.save(f"temp_save_dir/{directory_name}/time_array", time_plot)
np.save(f"temp_save_dir/{directory_name}/frequency", FREQUENCY)



print('data saved')






































