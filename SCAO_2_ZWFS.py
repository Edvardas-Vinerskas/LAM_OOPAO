
"""
TODO figure out how Zernike object works, specificaly what does ZWFS.signal represent
    i.e. read some papers
TODO I have removed the 2 frame delay

* the std limit of the phase rms getting into zernike is about 75 nm for a wvl of 790
    * so when training the RL zwfs on phase or whatever you can
        * either input the phase with the appropriate rms
        * or just do it from the pyramid (you should do this because it is more representative)


* going straight from pixel is thus another potential advantage
* you might need to .copy() arrays if you don't want their values to magically change

* does Jalo use some kind of validation for the dynamics model? seems like not
    * but this does not matter that much since you will be updating constantly
    * from theory I remember that the dynamics models overfit or similar where (RL dude) and that is why we need multiple dynamics model to average out the result


* you have fixed the 500 Hz problem via woofer tweeter but
    *how is the atmosphere signal aliased due to the 2 stage AO running at different speeds?

* insert type checking into your code
* and unit tests
"""


import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Circle
import numpy as np
import os
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
TEL - ATM
CALIB_zer = np.zeros((vZWFS.signal.shape[0], M2C_zer.shape[1]))


#zernike_wfs calibration with DM_zer
for i in range(M2C_zer.shape[1]):
    v_plus = M2C_zer[:, i] * stroke
    v_minus = -M2C_zer[:, i] * stroke


    TEL.resetOPD()
    DM_zer.coefs = v_plus
    SRC * TEL * DM_zer * vZWFS
    w_plus = vZWFS.signal


    TEL.resetOPD()
    DM_zer.coefs = v_minus
    SRC * TEL * DM_zer * vZWFS
    w_minus = vZWFS.signal


    CALIB_zer[:, i] = (w_plus - w_minus) / (2 * stroke)



#zernike_wfs phase reconstruction via sin function (the b value that is needed currently escapes me
"""for i in range(M2C_zer.shape[1]):
    # the self.zwfs1 is phase shifted -np.pi/2 LEFT HANDED
    # the self.zwfs2 is phase shifted np.pi/2 RIGHT HANDED
    I_delta = vZWFS.zwfs2.img_ZWFS - vZWFS.zwfs1.img_ZWFS

    phi = np.arcsin(I_delta / (2 * b_0))"""



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

ATM.generateNewPhaseScreen(seed = 10)
#reset everything just in case
TEL.resetOPD()
DM_pyr.coefs = 0
DM_zer.coefs = 0
DM_zer_copy = 0
DM_zer_copy_woof_tweet = 0
TEL + ATM
SRC * TEL * DM_pyr * PWFS * DM_zer * vZWFS
TEL.print_optical_path()

#delay implementation
pwfs_frame_delay         = 2
pwfs_delay               = pwfs_frame_delay - 1 #frame delay of 1 is already built-in
if pwfs_frame_delay >= 2:
    pwfssignal_buffer = [np.zeros(PWFS.nSignal) for i in range(pwfs_delay)]
    vzwfssignal_buffer= [np.zeros(vZWFS.nSignal) for i in range(pwfs_delay)]
else:
    pwfssignal_buffer = []
    vzwfssignal_buffer =[]

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
final_residual_OPD           = 0
final_atmosphere_OPD         = 0
final_dm_OPD                 = 0




#for temp_err_delay = 3, the list has current, previous and previous_previous
temp_err_delay               = 4
atm_OPD_list                 = [np.zeros(ATM.OPD.shape) for i in range((temp_err_delay + 1))] #(temp_err_delay + 1) tells you how many current + previous frames you want to keep in the buffer
sim_temp_error_2_frame_delay = []
sim_fit_error_list           = []
sim_fit_error_list_2D        = []



tel_psf_list = []


CL_gain_pyr = 0.4
CL_gain_zer = 0.4

reconstructor_pyr = M2C_pyr @ CALIB_pyr.M #takes in slopes and outputs modes, then takes in modes and outputs controle shape(357, 664)
reconstructor_zer = M2C_zer @ CALIB_zer_obj.M #takes in wfs signal and outputs control
ATM.generateNewPhaseScreen(seed = 10)
"""ZZ = Zernike(telObject = TEL,
                       J = 5)
ZZ.computeZernike(telObject2 = TEL)
tip     = ZZ.modesFullRes[:,:,0]
def sine(f, t):
    result = np.sin(2 * np.pi * (f * t))
    return result
"""

PWFS_signal_avg = 0
PWFS_signal_avg_norm = 0
for i in range(nLoop):
    #tip_vibr = 1e-7 * sine(40, i * SAMPLING_TIME) * tip
    #update phase screen
    ATM.update()
    atm_opd = ATM.OPD.copy()
    atm_OPD_temp_list.append(atm_opd)

    total_error[i] = np.std(TEL.OPD[np.where(TEL.pupil > 0)]) * 1e9

    # PWFS slope averaging over 3 frames
    PWFS_signal_avg += (PWFS.signal_2D + PWFS.referenceSignal_2D) * PWFS.norma
    PWFS_signal_avg_norm += PWFS.norma

    #update the dm commands
    if (i + 1) % 3 == 0:
        #PWFS slope averaging over 3 frames (should be implemented correctly, but if something goes wrong you can turn this off and check)
        PWFS_signal_avg = PWFS_signal_avg / PWFS_signal_avg_norm - PWFS.referenceSignal_2D
        PWFS_signal_avg = PWFS_signal_avg[np.where(PWFS.validSignal == 1)]

        #frame delay implementation (pyramid)
        pwfssignal_buffer.append(PWFS_signal_avg)
        pwfs_delayed_signal = pwfssignal_buffer[0]
        pwfssignal_buffer.pop(0)

        DM_pyr.coefs = DM_pyr.coefs - CL_gain_pyr * np.matmul(reconstructor_pyr, pwfs_delayed_signal)
        DM_pyr_copy = DM_pyr.coefs.copy()


        PWFS_signal_avg = 0
        PWFS_signal_avg_norm = 0

    # frame delay implementation (zernike)
    vzwfssignal_buffer.append(vZWFS.signal)
    vzwfs_delayed_signal = vzwfssignal_buffer[0]
    vzwfssignal_buffer.pop(0)


    DM_zer_coefs = DM_zer_copy - CL_gain_zer * np.matmul(reconstructor_zer, vzwfs_delayed_signal)
    DM_zer_copy = DM_zer_coefs.copy()
    #woofer_tweeter corr
    DM_zer.coefs = DM_zer_copy - np.matmul(N2N1, DM_pyr.coefs)


    #propagate through AO with the dm commands applied
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



#TODO clean up later
#TODO for some plots you need to check the sr > 0.5 condition
#TODO change the 2000 timeery condition
#TODO what other plots are missing?
save_files = True
directory_name = 'test_1st_1500'
savedir = f'temp_save_dir/{directory_name}/'

if not os.path.exists(savedir):
    os.makedirs(savedir)

if save_files == True:
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



#---------------------------------------------------Error decomposition---------------------------------------------------#
time_plot = np.arange(0, nLoop * SAMPLING_TIME, SAMPLING_TIME)
plt.figure()
plt.plot(time_plot, residual_error, label = "residual")
#plt.plot(time, sim_temp_error_2_frame_delay, label = "simulational temporal error 2 frame delay")
plt.title("error decomposition (nm)")
plt.xlabel("time s")
plt.yscale("log")
plt.grid(True)
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
plt.plot(time_plot, sr, label = "strehl")
plt.plot(time_plot, sr_running, label = "running_strehl")
plt.plot(time_plot, sr_1st, label = "1st stage strehl")
plt.title("Strehl ratio")
plt.xlabel("time s")
plt.ylim(bottom = (sr_mean - 0.3))
plt.grid(True)
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
plt.gca().add_artist(corr_zone_2)
plt.xlabel("arcsec")
plt.ylabel("arcsec")
plt.grid(True)
plt.colorbar()

tel_psf_list = np.array(tel_psf_list)
AO_PSF_LE = np.mean(tel_psf_list, axis = 0)
AO_PSF_LE = AO_PSF_LE[N: -N, N: -N]
AO_PSF_LE = AO_PSF_LE / np.sum(AO_PSF_LE)
plt.figure()
plt.imshow(AO_PSF_LE, norm = SymLogNorm(1e-6), extent = [-fov/2, fov/2, -fov/2, fov/2])
plt.title("AO corrected PSF LE")
plt.gca().add_artist(corr_zone_1_LE)
plt.gca().add_artist(corr_zone_2_LE)
plt.xlabel("arcsec")
plt.ylabel("arcsec")
plt.grid(True)
plt.colorbar()



#---------------------------------------------------Spatial PSD---------------------------------------------------#
#PSD is calculated using a square inside the tel.pupil
x_square = int(np.sqrt(TEL.resolution ** 2 / 2))
x_square_index = int((TEL.resolution - x_square) / 2 + 1)
N_length = TEL.resolution - 2 * x_square_index


#PSD normalisation factors
delta_x = TEL.D / TEL.resolution
delta_f = 1 / (delta_x * N_length)
f_DM = 1/ (2 * DM_pyr.pitch)



PSD_residual_circavg_list = []
PSD_residual_circavg_freq = 0
PSD_residual_freq_x       = 0
#PSD calculation for residual OPD
for j in range(len(residual_OPD_list)):
    residual_OPD = residual_OPD_list[j][x_square_index:-x_square_index, x_square_index:-x_square_index]
    residual_OPD = (residual_OPD - np.mean(residual_OPD)) * 1e9
    PSD_residual = np.abs(fftshift(fft2(residual_OPD))) ** 2 * delta_x ** 2 / (x_square ** 2)
    PSD_residual_freq_x = np.fft.fftfreq(PSD_residual.shape[0], d=(TEL.D / TEL.resolution))
    PSD_residual_freq_x = np.max(PSD_residual_freq_x)
    PSD_residual_freq_max = np.max(np.sqrt(2 * (PSD_residual_freq_x) ** 2))
    PSD_residual_freq, PSD_residual_circavg = circular_sum_PSD(np.abs(PSD_residual).shape, np.abs(PSD_residual), PSD_residual_freq_max)
    PSD_residual_circavg = PSD_residual_circavg * delta_f
    print("\n")
    print(f"TEL std {residual_error[ - len(residual_OPD_list) + j]}, TEL std from PSD {np.sqrt(np.sum(PSD_residual) * delta_f ** 2)}, TEL std from PSD {np.sqrt(np.sum(PSD_residual_circavg) * delta_f)}")



    if j == 0:
        PSD_residual_circavg_freq = PSD_residual_freq

    PSD_residual_circavg_list.append(PSD_residual_circavg)


'''PSD_fitting_circavg_list = []
PSD_fitting_circavg_freq = 0
#PSD calculation for fitting error
for j in range(len(sim_fit_error_list_2D)):
    fitting_OPD = sim_fit_error_list_2D[j][x_square_index:-x_square_index, x_square_index:-x_square_index]
    fitting_OPD = (fitting_OPD - np.mean(fitting_OPD)) * 1e9
    PSD_fitting = np.abs(fftshift(fft2(fitting_OPD))) ** 2 * delta_x ** 2 / (x_square ** 2)
    PSD_fitting_freq_x = np.fft.fftfreq(PSD_fitting.shape[0], d=(TEL.D / TEL.resolution))
    PSD_fitting_freq_x = np.max(PSD_fitting_freq_x)
    PSD_fitting_freq_max = np.max(np.sqrt(2 * (PSD_fitting_freq_x) ** 2))
    PSD_fitting_freq, PSD_fitting_circavg = circular_sum_PSD(np.abs(PSD_fitting).shape, np.abs(PSD_fitting), PSD_fitting_freq_max)
    PSD_fitting_circavg = PSD_fitting_circavg * delta_f

    if j == 0:
        PSD_fitting_circavg_freq = PSD_fitting_freq

    PSD_fitting_circavg_list.append(PSD_fitting_circavg)'''


PSD_atmosphere_circavg_list = []
PSD_atmosphere_circavg_freq = 0
ATM_OPD_generated_list = []
#PSD calculation for the atmosphere
for i in range(20):
    ATM.generateNewPhaseScreen(seed = i)
    for j in range(int(nLoop / 20)):
        ATM.update()
        atmosphere_OPD = ATM.OPD[x_square_index:-x_square_index, x_square_index:-x_square_index]
        atmosphere_OPD = (atmosphere_OPD - np.mean(atmosphere_OPD)) * 1e9
        PSD_atmosphere = np.abs(fftshift(fft2(atmosphere_OPD))) ** 2 * delta_x ** 2 / (x_square ** 2)
        PSD_atmosphere_freq_x = np.fft.fftfreq(PSD_atmosphere.shape[0], d = (TEL.D / TEL.resolution))
        PSD_atmosphere_freq_x = np.max(PSD_atmosphere_freq_x)
        PSD_atmosphere_freq_max = np.max(np.sqrt(2 * (PSD_atmosphere_freq_x) ** 2))
        PSD_atmosphere_freq, PSD_atmosphere_circavg = circular_sum_PSD(np.abs(PSD_atmosphere).shape, np.abs(PSD_atmosphere), PSD_atmosphere_freq_max)
        PSD_atmosphere_circavg = PSD_atmosphere_circavg * delta_f
        ATM_OPD_generated_list.append(ATM.OPD)
        print("\n")
        print(f"ATM std {np.std(atmosphere_OPD)}, ATM std from PSD {np.sqrt(np.sum(PSD_atmosphere) * delta_f ** 2)}, ATM std from PSD {np.sqrt(np.sum(PSD_atmosphere_circavg) * delta_f)}")



        PSD_atmosphere_circavg_list.append(PSD_atmosphere_circavg)
        if i == 0:
            PSD_atmosphere_circavg_freq = PSD_atmosphere_freq

PSD_residual_circavg_list  = np.array(PSD_residual_circavg_list)
PSD_atmosphere_circavg_list = np.array(PSD_atmosphere_circavg_list)


#analytical kolmogorov spectrum
def PSD_von_karman(f):
    result = 2 * np.pi * f * 0.023 * (1/r_0_src) ** (5/3) * (f ** 2 + L_0 ** (-2)) ** (-11/6) * (SRC.wavelength * 1e9 / (2 * np.pi)) ** 2
    return result

PSD_kolmogorov = PSD_von_karman(PSD_residual_circavg_freq)



PSD_residual_statavg = np.mean(PSD_residual_circavg_list, axis = 0)
PSD_atmosphere_statavg = np.mean(PSD_atmosphere_circavg_list, axis = 0)
#PSD_fitting_statavg = np.mean(PSD_fitting_circavg_list, axis = 0)
print(f"PSD_residual average error {np.sqrt(np.sum(PSD_residual_statavg) * delta_f)}")
print(f"PSD_atmosphere average error {np.sqrt(np.sum(PSD_atmosphere_statavg) * delta_f)}")
print(f"PSD_atmosphere analytical error {np.sqrt(np.sum(PSD_kolmogorov) * delta_f)}")


#fitting error from the PSDs
print(f"f_DM {f_DM}")
print(f"PSD_residual fitting {np.sqrt(np.sum(PSD_residual_statavg[PSD_residual_circavg_freq>= f_DM]) * delta_f)}")
print(f"PSD_atmosphere fitting {np.sqrt(np.sum(PSD_atmosphere_statavg[PSD_atmosphere_circavg_freq >= f_DM]) * delta_f)}")
#print(f"PSD_fitting fitting {np.sqrt(np.sum(PSD_fitting_statavg[PSD_atmosphere_circavg_freq >= f_DM]) * delta_f)}")
print(f"PSD_atmosphere analytical fitting error {np.sqrt(np.sum(PSD_kolmogorov[PSD_atmosphere_circavg_freq >= f_DM]) * delta_f)}")
print("\n")

#44 and 9
expon_ = np.log(PSD_residual_statavg[9]/PSD_residual_statavg[44])/np.log(PSD_residual_circavg_freq[9]/PSD_residual_circavg_freq[44])
print(f'residual exponential {expon_}')




plt.figure()
plt.plot(PSD_residual_circavg_freq, PSD_residual_statavg, label = "PSD_residual_pwfs")
plt.plot(PSD_atmosphere_circavg_freq, PSD_atmosphere_statavg, label = "atmosphere_PSD")
#plt.plot(PSD_fitting_circavg_freq, PSD_fitting_statavg, label = "fitting_err_PSD")
plt.plot(PSD_residual_circavg_freq, PSD_kolmogorov, label = "atmosphere_PSD_karman")
plt.axvline(x=f_DM, color='red', linestyle='-', linewidth=1.5)
plt.title("PSD residual vs atmosphere comparison")
plt.ylabel("PSD")
plt.xlabel("spatial frequency m^-1")
plt.xscale("log")
plt.yscale("log")
plt.xlim(right = PSD_residual_freq_x)
plt.ylim(bottom = 1e-2)
plt.grid(True)
plt.legend()



#---------------------------------------------------Zernike/KL decomposition---------------------------------------------------#
label_str = 0
if use_zernike:
    label_str = "Zernike"
elif use_KL:
    label_str = "KL"

ATM_OPD_generated_array = np.array(ATM_OPD_generated_list)

coefficient_matrix_atmosphere_list = []
coefficient_matrix_res_list        = []

if use_zernike or use_KL:
    for i in range(ATM_OPD_generated_array.shape[0]):
        atmosphere_phase = 2 * np.pi * ATM_OPD_generated_list[i] / SRC.wavelength
        coefficient_matrix_atmosphere = modes_inv @ atmosphere_phase[np.where(TEL.pupil > 0)]
        coefficient_matrix_atmosphere_list.append(coefficient_matrix_atmosphere)

    for i in range(len(residual_OPD_list)):
        final_residual_phase = 2 * np.pi * residual_OPD_list[i] / SRC.wavelength
        coefficient_matrix_res = modes_inv @ final_residual_phase[np.where(TEL.pupil > 0)]
        coefficient_matrix_res_list.append(coefficient_matrix_res)

    zernike_names = []
    for i in range(len(coefficient_matrix_res_list[0])):
        zernike_names.append(f"{i+1}")


    coefficient_matrix_atmosphere_var = np.var(np.array(coefficient_matrix_atmosphere_list), axis = 0)
    coefficient_matrix_res_var        = np.var(np.array(coefficient_matrix_res_list), axis = 0)


    plt.figure()
    plt.bar(zernike_names, coefficient_matrix_res_var, color = "red", label = f"{label_str} coeffs for residual phase")
    plt.title(f"{label_str} coefficients for corrected vs atmospher phase")
    plt.bar(zernike_names, coefficient_matrix_atmosphere_var, color = "black", alpha = 0.4, label = f"{label_str} coeffs for atmospheric phase")
    plt.yscale("log")
    plt.tight_layout()
    plt.legend()



#---------------------------------------------------Temporal PSD---------------------------------------------------#
#temporal PSD calculation from the std
delta_t = SAMPLING_TIME
f_samp = 1 / delta_t


def welch_method_scipy(data, fs=f_samp, nperseg=256):
    frequencies, psd = signal.welch(
        data,
        fs=fs,
        window='hann',  # Hanning window
        nperseg=nperseg,
        scaling='density'
    )
    return frequencies, psd





coefficient_matrix_res_list = np.array(coefficient_matrix_res_list)
residual_tip_curve = coefficient_matrix_res_list[:, 0]
residual_tilt_curve = coefficient_matrix_res_list[:, 1]
residual_defocus_curve = coefficient_matrix_res_list[:, 2]
residual_100_curve = coefficient_matrix_res_list[:, 100]
residual_200_curve = coefficient_matrix_res_list[:, 200]


temp_atm_coef_list = []
for i in range(len(atm_OPD_temp_list)):
    atmosphere_phase = 2 * np.pi * atm_OPD_temp_list[i] / SRC.wavelength
    coefficient_matrix_atmosphere = modes_inv @ atmosphere_phase[np.where(TEL.pupil > 0)]
    temp_atm_coef_list.append(coefficient_matrix_atmosphere)


temp_atm_coef_array = np.array(temp_atm_coef_list)
atm_tip_curve = temp_atm_coef_array[:, 0]
atm_tilt_curve = temp_atm_coef_array[:, 1]
atm_defocus_curve = temp_atm_coef_array[:, 2]
atm_100_curve = temp_atm_coef_array[:, 100]
atm_200_curve = temp_atm_coef_array[:, 200]


#I am not sure about the normalisation (it just works)
#tip
PSD_residual_tip_freq_t, PSD_residual_tip = welch_method_scipy(residual_tip_curve)
PSD_atm_tip_freq_t, PSD_atm_tip = welch_method_scipy(atm_tip_curve)


#tilt
PSD_residual_tilt_freq_t, PSD_residual_tilt = welch_method_scipy(residual_tilt_curve)
PSD_atm_tilt_freq_t, PSD_atm_tilt = welch_method_scipy(atm_tilt_curve)


#defocus
PSD_residual_defocus_freq_t, PSD_residual_defocus = welch_method_scipy(residual_defocus_curve)
PSD_atm_defocus_freq_t, PSD_atm_defocus = welch_method_scipy(atm_defocus_curve)


#modes 100 and 200
PSD_residual_100_freq_t, PSD_residual_100 = welch_method_scipy(residual_100_curve)
PSD_residual_200_freq_t, PSD_residual_200 = welch_method_scipy(residual_200_curve)

PSD_atm_100_freq_t, PSD_atm_100 = welch_method_scipy(atm_100_curve)
PSD_atm_200_freq_t, PSD_atm_200 = welch_method_scipy(atm_200_curve)



plt.figure()
plt.plot(PSD_residual_tip_freq_t, PSD_residual_tip, label = "residual_PSD_tip")
plt.plot(PSD_atm_tip_freq_t, PSD_atm_tip, label = "atm_PSD_tip")
plt.title(f"residual PSD tip, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True)
plt.legend()
plt.ylabel("PSD")


plt.figure()
plt.plot(time_plot, residual_tip_curve, label = "residual_tip_curve")
plt.plot(time_plot, atm_tip_curve, label = "atm_tip_curve")
plt.title(f"residual/atm timeseries tip, gain {CL_gain_pyr}")
plt.xlabel("time (s)")
plt.grid(True)
plt.legend()
plt.ylabel("residual tip")


plt.figure()
plt.plot(PSD_residual_tilt_freq_t, PSD_residual_tilt, label = "residual_PSD_tilt")
plt.plot(PSD_atm_tilt_freq_t, PSD_atm_tilt, label = "atm_PSD_tilt")
plt.title(f"residual PSD tilt, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True)
plt.legend()
plt.ylabel("PSD")


plt.figure()
plt.plot(PSD_residual_defocus_freq_t, PSD_residual_defocus, label = "residual_PSD_defocus")
plt.plot(PSD_atm_defocus_freq_t, PSD_atm_defocus, label = "atm_PSD_defocus")
plt.title(f"residual PSD defocus, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True)
plt.legend()
plt.ylabel("PSD")


plt.figure()
plt.plot(PSD_residual_100_freq_t, PSD_residual_100, label = "PSD_residual_100")
plt.plot(PSD_atm_100_freq_t, PSD_atm_100, label = "atm_PSD_100")
plt.title(f"residual PSD 100, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True)
plt.legend()
plt.ylabel("PSD")



plt.figure()
plt.plot(PSD_residual_200_freq_t, PSD_residual_200, label = "PSD_residual_200")
plt.plot(PSD_atm_200_freq_t, PSD_atm_200, label = "atm_PSD_200")
plt.title(f"residual PSD 200, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True)
plt.legend()
plt.ylabel("PSD")





#---------------------------------------------------temporal Error transfer function---------------------------------------------------#
tETF_tip = PSD_residual_tip/PSD_atm_tip
tETF_tilt = PSD_residual_tilt/PSD_atm_tilt
tETF_defocus = PSD_residual_defocus/PSD_atm_defocus
tETF_100 = PSD_residual_100/PSD_atm_100
tETF_200 = PSD_residual_200/PSD_atm_200


plt.figure()
plt.plot(PSD_residual_tip_freq_t, tETF_tip, label = "ETF tip")
plt.plot(PSD_residual_tilt_freq_t, tETF_tilt, label = "ETF tilt")
plt.plot(PSD_residual_defocus_freq_t, tETF_defocus, label = "ETF defocus")
plt.plot(PSD_residual_100_freq_t, tETF_100, label = "ETF 100")
plt.plot(PSD_residual_200_freq_t, tETF_200, label = "ETF 200")
plt.title("temporal error transfer functions")
plt.ylabel("ETF")
plt.xlabel("frequency Hz")
plt.xscale("log")
plt.yscale("log")
plt.xlim(right = np.max(PSD_residual_tip_freq_t))
plt.grid(True)
plt.legend()

#---------------------------------------------------correlation calculation---------------------------------------------------#

#here it is the covariance divided by standard deviation
def correlation_f(phase_1, phase_2):
    phase_1 = phase_1[np.where(TEL.pupil == 1)]
    phase_2 = phase_2[np.where(TEL.pupil == 1)]

    phase_1_centered = phase_1 - np.mean(phase_1)
    phase_2_centered = phase_2 - np.mean(phase_2)
    correlation = np.corrcoef(phase_1_centered, phase_2_centered)[0, 1]

    return correlation


atm_corr_coefs = []
residual_pwfs_corr_coefs = []
residual_zwfs_corr_coefs = []
for i in range(len(atm_OPD_temp_list[:200])):
    a1 = correlation_f(atm_OPD_temp_list[0], atm_OPD_temp_list[i])
    atm_corr_coefs.append(a1)
for i in range(len(residual_OPD_list_pwfs[300:500])):
    r1 = correlation_f(residual_OPD_list_pwfs[300], residual_OPD_list_pwfs[300 + i])
    residual_pwfs_corr_coefs.append(r1)

    r2 = correlation_f(residual_OPD_list[300], residual_OPD_list[300 + i])
    residual_zwfs_corr_coefs.append(r2)

plt.figure()
plt.title("atmosphere vs residual correlation")
plt.plot(atm_corr_coefs, label = "atm corr")
plt.plot(residual_pwfs_corr_coefs, label = "res corr pwfs")
plt.plot(residual_zwfs_corr_coefs, label = "res corr zwfs")
plt.grid(True)
plt.legend()

plt.show()




























































































