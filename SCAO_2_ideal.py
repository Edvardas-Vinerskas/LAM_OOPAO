"""
This is the ideal case
    * no delays
    * pure fitting error
    * goal - set the upper limit of performance
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
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.Zernike import Zernike
from OOPAO.Detector import Detector

from functions import *


#---------------------------------------------------GLOBALS---------------------------------------------------#
#define all OOPAO variables

N_SUBAPERTURE       = 20
DIAMETER            = 1.5
CENTRAL_OBSTRUCTION = 0 #0.15
RESOLUTION          = N_SUBAPERTURE * 8
FREQUENCY           = 500
SAMPLING_TIME       = 1/FREQUENCY
FOV                 = 10
MECH_COUPLING       = 0.35
MODULATION          = 3
LIGHT_RATIO         = 0.1
POST_PROCESS        = "slopesMaps"
r_0                 = 0.15
L_0                 = 25
WIND_SPEED          = [10, 20] #[10, 20, 60]
WIND_DIRECTION      = [0, 100] #[0, 100, 160]
FRACTIONAL_C_N2     = [0.6, 0.4] #[0.5, 0.3, 0.2]
ALTITUDE            = [0, 4500] #[0, 4500, 10000]
Z_coefs             = 250 #200 #above 200 does not work great per Benoit
KL_no               = 250



zeroPaddingFactor = 6
rad2arcsec        = 180 * 60 * 60 / np.pi


use_pwfs          = True
use_shwfs         = False

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


#---------------------------------------------------DEFORMABLE_MIRROR---------------------------------------------------#

DM = DeformableMirror(telescope    = TEL,
                      nSubap       = N_SUBAPERTURE,
                      mechCoupling = MECH_COUPLING)

#control radius calculation for dm
control_radius = ((N_SUBAPERTURE + 1) * SRC.wavelength) /(2 * TEL.D) * rad2arcsec
corr_zone_1 = Circle([0,0], control_radius, fc='none', ec='w', ls=':')
corr_zone_2 = Circle([0,0], control_radius, fc='none', ec='w', ls=':')

#pitch check
if (2 * DM.pitch) > r_0_src:
    raise SystemExit(f"ERROR: DM actuator density is insufficient for r_0 {r_0_src}, dm pitch {DM.pitch} ")

#---------------------------------------------------WFS---------------------------------------------------#
WFS = None
if use_pwfs:

    PWFS = Pyramid(nSubap         = N_SUBAPERTURE,
                   telescope      = TEL,
                   modulation     = MODULATION,
                   lightRatio     = LIGHT_RATIO,
                   postProcessing = POST_PROCESS)
    WFS = PWFS

if use_shwfs:
    SHWFS = ShackHartmann(nSubap       = N_SUBAPERTURE,
                          telescope    = TEL,
                          lightRatio   = LIGHT_RATIO,
                          is_geometric = False)
    WFS = SHWFS

#---------------------------------------------------MODAL_BASIS---------------------------------------------------#
M2C   = None
modes = None
if use_zonal:
    M2C_zonal = np.identity(DM.nValidAct)
    M2C       = M2C_zonal
    modes     = DM.modes


if use_zernike:

    zernike = Zernike(telObject = TEL,
                       J = Z_coefs)
    zernike.computeZernike(telObject2 = TEL)

    M2C_Z = np.linalg.pinv(np.squeeze(DM.modes[TEL.pupilLogical, :])) @ zernike.modes
    M2C   = M2C_Z
    modes = zernike.modes

ZZ = Zernike(telObject = TEL,
                       J = 5)
ZZ.computeZernike(telObject2 = TEL)

if use_KL:

    M2C_KL = compute_KL_basis(tel = TEL, atm = ATM, dm = DM)
    M2C    = M2C_KL[:, :KL_no]
    modes  = DM.modes @ M2C

#---------------------------------------------------INTERACTION MATRIX---------------------------------------------------#



stroke = SRC.wavelength / 16
CALIB = InteractionMatrix(ngs            = SRC,
                          tel            = TEL,
                          dm             = DM,
                          wfs            = WFS,
                          M2C            = M2C,
                          atm            = ATM,
                          nMeasurements  = 5,
                          stroke         = stroke,
                          noise          = "off")


#---------------------------------------------------FITTING ERROR CALC---------------------------------------------------#

#for zernike and zonal only (later extract the KL modes from the source code?)
modes_inv = None
#the rest of the code is in the for loop

#takes in phase and outputs modes
if use_zonal:
    modes_inv = np.linalg.pinv(np.squeeze(modes[TEL.pupilLogical, :]))

if use_zernike:
    modes_inv = np.linalg.pinv(np.squeeze(modes))

if use_KL:
    modes_inv = np.linalg.pinv(np.squeeze(modes[TEL.pupilLogical, :]))


#the rest of the code is in the for loop
#---------------------------------------------------SIMULATION---------------------------------------------------#
#for now we are only using SRC

ATM.generateNewPhaseScreen(seed = 10)
#reset everything just in case
SRC.reset()
DM.coefs = 0
SRC ** ATM * TEL * DM * WFS
TEL.print_optical_path()

#variables and performance metric initialisation
nLoop                        = 3000
sr                           = np.zeros(nLoop)
sr_running                   = np.zeros(nLoop)
total_error                  = np.zeros(nLoop)
residual_error               = np.zeros(nLoop)

#variables for PSD calculation
residual_OPD_list            = []
final_residual_OPD           = 0
final_atmosphere_OPD         = 0
final_dm_OPD                 = 0
modes_atm = []
modes_1st_stage = []

tel_psf_list = []
atm_OPD_temp_list = []


#what is inside the loop is basically what should be in the step function of the RL OOPAO environment
ATM.generateNewPhaseScreen(seed = 10)
for i in range(nLoop):
    #update phase screen
    ATM.update()
    atm_opd = ATM.OPD

    total_error[i] = np.std(ATM.OPD[np.where(TEL.pupil > 0)]) * 1e9

    # update the dm commands
    mode_coefs = modes_inv @ atm_opd[np.where(TEL.pupil > 0)]
    DM.coefs = - M2C @ mode_coefs
    dm_coefs_copy = DM.coefs

    #propagate through AO with the dm commands applied
    SRC ** ATM * TEL
    modes_atm.append(modes_inv @ SRC.OPD.flatten()[np.where(TEL.pupil == 1)])

    SRC ** ATM * TEL * DM * WFS
    modes_1st_stage.append(modes_inv @ SRC.OPD.flatten()[np.where(TEL.pupil == 1)])


    #performance metrics
    sr[i] = np.exp(-np.var(TEL.src.phase[np.where(TEL.pupil == 1)]))
    residual_error[i] = np.std(TEL.OPD[np.where(TEL.pupil > 0)]) * 1e9
    print("Loop" + str(i) + "/" + str(nLoop) + " " + "AO residual: " + str(residual_error[i]) + "nm" + "total err: " + str(total_error[i]) + "nm")
    print(f"strehl {sr[i]}")





#TODO clean up later
save_files = True
directory_name = 'PWFS_metrics_500_ideal'
if save_files == True:
    residual_error_array = np.asarray(residual_error)
    np.save(f"temp_save_dir/{directory_name}/residual_error", residual_error_array)

    strehl_array_1st = np.asarray(sr)
    np.save(f"temp_save_dir/{directory_name}/strehl_array_1st", strehl_array_1st)

    modes_1st_stage = np.asarray(modes_1st_stage)
    np.save(f"temp_save_dir/{directory_name}/modes_1st_stage", modes_1st_stage)

    modes_atm = np.asarray(modes_atm)
    np.save(f"temp_save_dir/{directory_name}/modes_atm", modes_atm)

    total_err_array = np.asarray(total_error) #not useful for now
    np.save(f"temp_save_dir/{directory_name}/total_err_array", total_err_array)

    #YOU SHOULD ALSO CHECK IF YOU ONLY HAVE AN EPISODE OF DATA IN ALL OF THESE BECAUSE I DON'T REMEMBER
    time_plot = np.arange(0, nLoop * SAMPLING_TIME, SAMPLING_TIME)
    np.save(f"temp_save_dir/{directory_name}/time_array", time_plot)
    np.save(f"temp_save_dir/{directory_name}/frequency", FREQUENCY)



    print('data saved')


























































