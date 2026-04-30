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
N_SUBAPERTURE_zer   = 9
DIAMETER            = 1.5
CENTRAL_OBSTRUCTION = 0 #0.15
RESOLUTION          = N_SUBAPERTURE_pyr * 8
FREQUENCY           = 1500 #pyramid is running at 500
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
KL_coefs_pyr        = 250
KL_coefs_zer        = 81



zeroPaddingFactor = 6
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
DM = DeformableMirror(telescope        = TEL,  # Telescope
                      nSubap           = N_SUBAPERTURE_pyr,
                      mechCoupling     = MECH_COUPLING)


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


#---------------------------------------------------MODAL_BASIS---------------------------------------------------#
# use of a single modal basis to prevent modal coupling between DM of stage 1 and stage 2
full_M2C = compute_KL_basis(TEL,
                            ATM,
                            DM,
                            lim=0)  # inversion stability criterion

# compute KL modal projector truncating the modes by the pupil used
KL_basis_dm = (DM.modes @ full_M2C) * np.tile(TEL.pupil.flatten()[:, None], full_M2C.shape[1])
projector_kl = np.linalg.pinv(KL_basis_dm)


M2C_KL_pyr = compute_KL_basis(tel = TEL, atm = ATM, dm = DM_pyr)
M2C_pyr    = M2C_KL_pyr[:, :KL_coefs_pyr]
modes_pyr  = (DM_pyr.modes * np.tile(TEL.pupil.flatten()[:, None], DM_pyr.modes.shape[1])) @ M2C_pyr

M2C_KL_zer = compute_KL_basis(tel=TEL, atm=ATM, dm = DM_zer)
M2C_zer = M2C_KL_zer[:, :KL_coefs_zer]
modes_zer = (DM_zer.modes * np.tile(TEL.pupil.flatten()[:, None], DM_zer.modes.shape[1])) @ M2C_zer
#---------------------------------------------------FITTING ERROR CALC---------------------------------------------------#
modes_inv_pyr = np.linalg.pinv(modes_pyr)
modes_inv_zer = np.linalg.pinv(modes_pyr)

#---------------------------------------------------SIMULATION---------------------------------------------------#

ATM.generateNewPhaseScreen(seed = 10)
#reset everything just in case
SRC.reset()
DM_pyr.coefs = 0
DM_zer.coefs = 0
DM_zer_copy = 0
SRC ** ATM * TEL * DM_pyr
SRC ** ATM * TEL * DM_pyr * DM_zer
TEL.print_optical_path()

#variables and performance metric initialisation
nLoop                        = 12000
sr                           = np.zeros(nLoop)
sr_1st                       = []
sr_running                   = np.zeros(nLoop)
total_error                  = np.zeros(nLoop)
residual_error               = np.zeros(nLoop)

#variables for PSD calculation
#vibration implementation
atm_OPD_list = [np.zeros(ATM.OPD.shape) for i in range(3)]

modes_atm                    = []
modes_1st_stage              = []
modes_2nd_stage              = []


CL_gain_pyr = 0.4
CL_gain_zer = 0.4
ATM.generateNewPhaseScreen(seed = 10)

PWFS_signal_avg = 0
PWFS_signal_avg_norm = 0
for i in range(nLoop):

    ATM.update()
    atm_opd = ATM.OPD.copy()
    atm_OPD_list.append(atm_opd)
    atm_OPD_list.pop(0)

    total_error[i] = np.std(ATM.OPD[np.where(TEL.pupil > 0)]) * 1e9

    #update the dm commands
    if (i + 1) % 3 == 0:
        pyr_atm_OPD = np.mean([atm_OPD_list[-3], atm_OPD_list[-2],
                               atm_OPD_list[-1]], axis=0)

        mode_coefs_pyr = modes_inv_pyr @ pyr_atm_OPD
        DM_pyr.coefs = -M2C_pyr @ mode_coefs_pyr
        DM_pyr_copy = DM_pyr.coefs.copy()



    mode_coefs_zer = modes_inv_zer @ atm_OPD_list[-1]
    DM_zer_coefs = -M2C_zer @ mode_coefs_zer
    DM_zer_copy = DM_zer_coefs.copy()


    #woofer_tweeter corr
    DM_zer.coefs = DM_zer_copy - np.matmul(N2N1, DM_pyr.coefs)

    SRC ** ATM * TEL
    modes_atm.append(projector_kl @ SRC.OPD.flatten())

    # propagate through AO with the dm commands applied
    SRC ** ATM * TEL * DM_pyr
    modes_1st_stage.append(projector_kl @ SRC.OPD.flatten())
    if (i + 1) % 3 == 0:
        sr_1st.append(np.exp(-np.var(TEL.src.phase[np.where(TEL.pupil == 1)])))
    SRC ** ATM * TEL * DM_pyr * DM_zer


    #performance metrics
    sr[i] = np.exp(-np.var(TEL.src.phase[np.where(TEL.pupil == 1)]))
    modes_2nd_stage.append(projector_kl @ SRC.OPD.flatten())
    residual_error[i] = np.std(TEL.OPD[np.where(TEL.pupil > 0)]) * 1e9
    print("Loop" + str(i) + "/" + str(nLoop) + " " + "AO residual: " + str(residual_error[i]) + "nm" + "total err: " + str(total_error[i]) + "nm")
    print(f"strehl {sr[i]}")







save_files = True
directory_name = 'vZWFS_metrics_ideal'
if save_files == True:
    residual_error_array = np.asarray(residual_error)
    np.save(f"temp_save_dir/{directory_name}/residual_error", residual_error_array)

    strehl_array_1st = np.asarray(sr_1st)
    strehl_array_1st = np.repeat(strehl_array_1st, 3)
    strehl_array_1st = np.insert(strehl_array_1st, 0, [0, 0])
    np.save(f"temp_save_dir/{directory_name}/strehl_array_1st", strehl_array_1st)

    strehl_array_2nd = np.asarray(sr)
    np.save(f"temp_save_dir/{directory_name}/strehl_array_2nd", strehl_array_2nd)


    modes_1st_stage = np.asarray(modes_1st_stage)
    np.save(f"temp_save_dir/{directory_name}/modes_1st_stage", modes_1st_stage)

    modes_2nd_stage = np.asarray(modes_2nd_stage)
    np.save(f"temp_save_dir/{directory_name}/modes_2nd_stage", modes_2nd_stage)


    modes_atm = np.asarray(modes_atm)
    np.save(f"temp_save_dir/{directory_name}/modes_atm", modes_atm)


    total_err_array = np.asarray(total_error) #not useful for now
    np.save(f"temp_save_dir/{directory_name}/total_err_array", total_err_array)

    #YOU SHOULD ALSO CHECK IF YOU ONLY HAVE AN EPISODE OF DATA IN ALL OF THESE BECAUSE I DON'T REMEMBER
    time_plot = np.arange(0, nLoop * SAMPLING_TIME, SAMPLING_TIME)
    np.save(f"temp_save_dir/{directory_name}/time_array", time_plot)
    np.save(f"temp_save_dir/{directory_name}/frequency", FREQUENCY)



    print('data saved')






































