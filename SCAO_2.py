"""
Testing for different src and ngs (next step is extended source?)

LIST OF ARGUMENTS TO TEST:
* In the real AO bench they had a gain sensing camera? how does that incorporate here (especially papytwin I guess)


SOME ANSWERS:
* for zonal you just use the identity of M2C
for the PSD problem:
    * can do apodizing (welsh filter or smooth filter going to 0 over 2 pixels)
    * do maths about how the square filter affects our FFT
        * just like mentioned (and I checked) square filter causes the transfer of power from lower to higher frequencies
        * this happens due to the sidelobs and is somewhat assymetric (judging from the graphical function representation)
        * this problem is called spectral leakage (very appropriate)
"""


"""    
* try doing a zero delay loop to see if the residual completely gets rid of the temporal error

* ETF x input should give you an understanding of how different frequencies are compensated for vs amplified

* add a sine function to the KL or zernike modes and appropriately select the frequency I guess
    * and atm.update() with this new OPD

* implement cameras for the SE PSF and LE PSF

* LONG TERM TASK - ASSUME SOME ZERNIKE OUTPUT PHASE AND CONTROL A 9x9 DM (I left some papers for the weekend reading)

* fix the OTF calculation (surely I need to average the PSFs no?)

* for the non-corrected tilt frequency just try out different gains
    * plot integrator controller analytical control curve (i.e. watch the control lectures that you have been neglecting)
    * read the RL book
* For the KL modes the non-correction could either due to 
    * non-truncation of SVD values
    * or more likely gain, so once again try playing with it
I am not sure why the LE PSF does not have the clear correction zone
    * you should do a comparison with corrected and non-corrected and see what is the difference


For now, tel.resetOPD() is still functional but it will display a warning. To reset the light propagation, the reset() method from the Source object can be triggered using the ** operator: 

src**tel : will correspond to a free-space propagation applying the pupil and Telescope effects
src**atm : will propagate the light through the atmosphere layers (No pupil applied)
src**atm*tel : will propagate the light through the atmosphere and then Telescope (applying the pupil and Telescope effects) 
"""


import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Circle
import numpy as np
from scipy import signal
import os
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
#TODO save all of these variables somewhere
N_SUBAPERTURE       = 20
DIAMETER            = 1.52
CENTRAL_OBSTRUCTION = 0 #0.15
RESOLUTION          = N_SUBAPERTURE * 8
FREQUENCY           = 1500
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
ALTITUDE            = [0, 45000] #[0, 45000, 4500, 10000]
Z_coefs             = 250 #200 #above 200 does not work great per Benoit
KL_no               = 250

photNoise           = False
readNoise           = 0
quanteff            = 1


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

WFS.cam.photonNoise = photNoise
WFS.cam.readoutNoise = readNoise
#WFS.cam.darkCurrent = 100 #for some reason the integration time is not automatically retrieved from the telescope object
WFS.cam.QE = quanteff

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

#---------------------------------------------------CAMERA---------------------------------------------------#

CAM = Detector(integrationTime = 100 * TEL.samplingTime,  # integration time of the detector
                photonNoise=False,  # enable photon noise
                readoutNoise=0,  # readout of the detector in [e-/pixel]
                QE=1,  # quantum efficiency
                psf_sampling=2,  # sampling for the PSF computation 2 = Shannon sampling
                binning=1)  # Binning factor of the PSF







#---------------------------------------------------SIMULATION---------------------------------------------------#
#for now we are only using SRC

ATM.generateNewPhaseScreen(seed = 10)
#reset everything just in case
SRC.reset()
DM.coefs = 0
SRC ** ATM * TEL * DM * WFS
TEL.print_optical_path()

#delay implementation
frame_delay         = 2
delay               = frame_delay - 1 #frame delay of 1 is already built-in
if frame_delay >= 2:
    wfssignal_buffer = [np.zeros(WFS.nSignal) for i in range(delay)]
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
atm_OPD_temp_list = []
tip     = ZZ.modesFullRes[:,:,0]
tilt    = ZZ.modesFullRes[:,:,1]
defocus = ZZ.modesFullRes[:,:,2]


aa = np.random.uniform(-1, 1)
bb = np.random.uniform(-1, 1)
cc = np.random.uniform(-1, 1)
phi_aa = np.random.uniform(-np.pi, np.pi)
phi_bb = np.random.uniform(-np.pi, np.pi)
phi_cc = np.random.uniform(-np.pi, np.pi)


def sine(f, t, phi):
    result = np.sin(2 * np.pi * (f * t) + phi)
    return result




tel_psf_list = []


CL_gain = 0.4
reconstructor = M2C @ CALIB.M

tip_f     = 10
tilt_f    = 50
defocus_f = 100


WFS.cam.photonNoise = photNoise
WFS.cam.readoutNoise = readNoise
#WFS.cam.darkCurrent = 100 #for some reason the integration time is not automatically retrieved from the telescope object
WFS.cam.QE = quanteff
#what is inside the loop is basically what should be in the step function of the RL OOPAO environment
ATM.generateNewPhaseScreen(seed = 10)
for i in range(nLoop):
    #temporal vibration injection
    tip_vibr  = sine(tip_f, i * SAMPLING_TIME, phi_aa) * tip
    tilt_vibr = sine(tilt_f, i * SAMPLING_TIME, phi_bb) * tilt
    defocus_vibr = sine(defocus_f, i * SAMPLING_TIME, phi_bb) * defocus
    vibr = 1e-7 * (aa * tip_vibr + bb * tilt_vibr)# + cc * defocus_vibr)

    #update phase screen
    ATM.update()
    atm_opd = ATM.OPD
    atm_OPD_temp_list.append(atm_opd)
    #ATM.update(atm_opd + vibr)


    total_error[i] = np.std(ATM.OPD[np.where(TEL.pupil > 0)]) * 1e9



    #frame delay implementation
    wfssignal_buffer.append(WFS.signal)
    delayed_signal = wfssignal_buffer[0]
    wfssignal_buffer.pop(0)


    #update the dm commands
    DM.coefs = DM.coefs - CL_gain * np.matmul(reconstructor, delayed_signal)
    dm_coefs_copy = DM.coefs

    # propagate through AO with the dm commands applied
    SRC ** ATM * TEL * DM * WFS

    #fitting error calculation for every frame (written for 2 frame delay)
    mode_coefs = modes_inv @ atm_OPD_list[-3][np.where(TEL.pupil > 0)]
    DM.coefs = M2C @ mode_coefs
    fitting_error = (atm_OPD_list[-3] - DM.OPD)
    simulational_fitting_error = np.std(fitting_error[np.where(TEL.pupil > 0)]) * 1e9
    sim_fit_error_list.append(simulational_fitting_error)

    #TODO you might need to move this above fitting error calculation?
    #temporal error calculation for every frame for 1, 2, 3 frame delay
    atm_OPD_list.append(ATM.OPD) #this includes the error from vibration too (and maybe it shouldn't)
    atm_OPD_list.pop(0)

    atm_coefs_t = modes_inv @ atm_OPD_list[-1][np.where(TEL.pupil > 0)]
    DM.coefs = M2C @ atm_coefs_t
    atm_t_OPD = DM.OPD
    atm_coefs_t2 = modes_inv @ atm_OPD_list[-3][np.where(TEL.pupil > 0)]
    DM.coefs = M2C @ atm_coefs_t2
    atm_t2_OPD = DM.OPD


    atm_OPD_2_frame = (atm_t_OPD - atm_t2_OPD)[np.where(TEL.pupil > 0)]
    sim_temp_error_2_frame_delay.append(np.std(atm_OPD_2_frame) * 1e9)


    #convert back the DM.coefs to their actual CL values
    DM.coefs = dm_coefs_copy



    #performance metrics
    sr[i] = np.exp(-np.var(TEL.src.phase[np.where(TEL.pupil == 1)]))
    residual_error[i] = np.std(TEL.OPD[np.where(TEL.pupil > 0)]) * 1e9
    print("Loop" + str(i) + "/" + str(nLoop) + " " + "AO residual: " + str(residual_error[i]) + "nm" + "total err: " + str(total_error[i]) + "nm")
    print(f"strehl {sr[i]}")


    #if sr[i] > 0.5:
    residual_OPD_list.append(TEL.OPD)
    sim_fit_error_list_2D.append(fitting_error)


    TEL.computePSF(zeroPaddingFactor)
    tel_psf_list.append(TEL.PSF)


#TODO clean up later
#TODO for some plots you need to check the sr > 0.5 condition
#TODO change the 2000 timeery condition
#TODO what other plots are missing?

save_files = True
directory_name = 'PWFS_500_integrator_wind_10_100deg_test-v2'
savedir = f'temp_save_dir/{directory_name}/'


if save_files == True:
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    residual_error_array = np.asarray(residual_error)
    np.save(f"temp_save_dir/{directory_name}/residual_error", residual_error_array)

    strehl_array_1st = np.asarray(sr)
    np.save(f"temp_save_dir/{directory_name}/strehl_array_1st", strehl_array_1st)

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

err
#---------------------------------------------------PLOTTING---------------------------------------------------#
sim_fit_error_list = np.array(sim_fit_error_list)
sim_temp_error_2_frame_delay = np.array(sim_temp_error_2_frame_delay)
sim_fit_temp_sum = np.sqrt(sim_temp_error_2_frame_delay ** 2 + sim_fit_error_list ** 2)



time_plot = np.arange(0, nLoop * SAMPLING_TIME, SAMPLING_TIME)



fitting_error_analytical = (0.23) ** (1/2) * (DM.pitch/ r_0_src) ** (5 / 6) * SRC.wavelength / (2 * np.pi) * 1e9
fitting_error_analytical = np.ones(len(time_plot)) * fitting_error_analytical

fitting_error_sim_avg = np.mean(sim_fit_error_list)
fitting_error_sim_avg = np.ones(len(time_plot)) * fitting_error_sim_avg

residual_sim_difference = residual_error - sim_fit_temp_sum


#---------------------------------------------------Error decomposition---------------------------------------------------#
plt.figure()
plt.plot(time_plot, residual_error, label = "residual")
#plt.plot(time, sim_temp_error_2_frame_delay, label = "simulational temporal error 2 frame delay")
plt.plot(time_plot, fitting_error_analytical, label = "analyical fitting error")
plt.plot(time_plot, fitting_error_sim_avg, label = "simulational average fitting error")
plt.plot(time_plot, sim_fit_error_list, label = "simulational fitting error")
plt.plot(time_plot, sim_fit_temp_sum, label = "fitting + temporal 2 frame delay")
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
plt.plot(time_plot, sr, label = "strehl")
plt.plot(time_plot, sr_running, label = "running_strehl")
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

#---------------------------------------------------AO OTF---------------------------------------------------#


OTF_AO = fftshift(fft2(fftshift(TEL.PSF / np.sum(TEL.PSF))))
x_axis___, OTF_AO_averaged = circular_average(np.abs(OTF_AO).shape, np.abs(OTF_AO))



plt.figure()
plt.plot(x_axis, OTF_dl_averaged, label = "diffraction limited")
plt.plot(x_axis___, OTF_AO_averaged, label = "AO corrected pwfs")
plt.title("OTF magnitude diffraction limited")
plt.xlabel("Frequency domain")
plt.ylabel("MTF")
plt.xscale("log")
plt.yscale("log")
plt.ylim(bottom = 1e-3)
plt.legend()



#---------------------------------------------------Spatial PSD---------------------------------------------------#
#PSD is calculated using a square inside the tel.pupil
x_square = int(np.sqrt(TEL.resolution ** 2 / 2))
x_square_index = int((TEL.resolution - x_square) / 2 + 1)
N_length = TEL.resolution - 2 * x_square_index


#PSD normalisation factors
delta_x = TEL.D / TEL.resolution
delta_f = 1 / (delta_x * N_length)
f_DM = 1/ (2 * DM.pitch)



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


PSD_fitting_circavg_list = []
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

    PSD_fitting_circavg_list.append(PSD_fitting_circavg)


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
PSD_fitting_statavg = np.mean(PSD_fitting_circavg_list, axis = 0)
print(f"PSD_residual average error {np.sqrt(np.sum(PSD_residual_statavg) * delta_f)}")
print(f"PSD_atmosphere average error {np.sqrt(np.sum(PSD_atmosphere_statavg) * delta_f)}")
print(f"PSD_atmosphere analytical error {np.sqrt(np.sum(PSD_kolmogorov) * delta_f)}")


#fitting error from the PSDs
print(f"f_DM {f_DM}")
print(f"PSD_residual fitting {np.sqrt(np.sum(PSD_residual_statavg[PSD_residual_circavg_freq>= f_DM]) * delta_f)}")
print(f"PSD_atmosphere fitting {np.sqrt(np.sum(PSD_atmosphere_statavg[PSD_atmosphere_circavg_freq >= f_DM]) * delta_f)}")
print(f"PSD_fitting fitting {np.sqrt(np.sum(PSD_fitting_statavg[PSD_atmosphere_circavg_freq >= f_DM]) * delta_f)}")
print(f"PSD_atmosphere analytical fitting error {np.sqrt(np.sum(PSD_kolmogorov[PSD_atmosphere_circavg_freq >= f_DM]) * delta_f)}")
print("\n")

#44 and 9
expon_ = np.log(PSD_residual_statavg[9]/PSD_residual_statavg[44])/np.log(PSD_residual_circavg_freq[9]/PSD_residual_circavg_freq[44])
print(f'residual exponential {expon_}')




plt.figure()
plt.plot(PSD_residual_circavg_freq, PSD_residual_statavg, label = "PSD_residual_pwfs")
plt.plot(PSD_atmosphere_circavg_freq, PSD_atmosphere_statavg, label = "atmosphere_PSD")
plt.plot(PSD_fitting_circavg_freq, PSD_fitting_statavg, label = "fitting_err_PSD")
plt.plot(PSD_residual_circavg_freq, PSD_kolmogorov, label = "atmosphere_PSD_karman")
plt.axvline(x=f_DM, color='red', linestyle='-', linewidth=1.5)
plt.title("PSD residual vs atmosphere comparison")
plt.ylabel("PSD")
plt.xlabel("spatial frequency m^-1")
plt.xscale("log")
plt.yscale("log")
plt.xlim(right = PSD_residual_freq_x)
plt.ylim(bottom = 1e-2)
plt.legend()



#---------------------------------------------------Zernike/KL decomposition---------------------------------------------------#
#I mean this must be temporal
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


    coefficient_matrix_atmosphere_var = np.var(np.asarray(coefficient_matrix_atmosphere_list), axis = 0)
    coefficient_matrix_res_var        = np.var(np.asarray(coefficient_matrix_res_list), axis = 0)


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


temp_atm_coef_list = []
for i in range(len(atm_OPD_temp_list)):
    atmosphere_phase = 2 * np.pi * atm_OPD_temp_list[i] / SRC.wavelength
    coefficient_matrix_atmosphere = modes_inv @ atmosphere_phase[np.where(TEL.pupil > 0)]
    temp_atm_coef_list.append(coefficient_matrix_atmosphere)


temp_atm_coef_array = np.array(temp_atm_coef_list)
atm_tip_curve = temp_atm_coef_array[:, 0]
atm_tilt_curve = temp_atm_coef_array[:, 1]
atm_defocus_curve = temp_atm_coef_array[:, 2]


#I am not sure about the normalisation
#tip
PSD_residual_tip_freq_t, PSD_residual_tip = welch_method_scipy(residual_tip_curve)
#PSD_residual_tip = np.abs(np.fft.rfft(residual_tip_curve)) ** 2 * delta_t * 2/ (len(residual_tip_curve))
#PSD_residual_tip_freq_t = np.fft.rfftfreq(len(residual_tip_curve), d=delta_t)

PSD_atm_tip_freq_t, PSD_atm_tip = welch_method_scipy(atm_tip_curve)
#PSD_atm_tip = np.abs(np.fft.rfft(atm_tip_curve)) ** 2 * delta_t * 2/ (len(atm_tip_curve))
#PSD_atm_tip_freq_t = np.fft.rfftfreq(len(atm_tip_curve), d=delta_t)

#tilt
PSD_residual_tilt_freq_t, PSD_residual_tilt = welch_method_scipy(residual_tilt_curve)
#PSD_residual_tilt = np.abs(np.fft.rfft(residual_tilt_curve)) ** 2 * delta_t * 2/ (len(residual_tilt_curve))
#PSD_residual_tilt_freq_t = np.fft.rfftfreq(len(residual_tilt_curve), d=delta_t)

PSD_atm_tilt_freq_t, PSD_atm_tilt = welch_method_scipy(atm_tilt_curve)
#PSD_atm_tilt = np.abs(np.fft.rfft(atm_tilt_curve)) ** 2 * delta_t * 2/ (len(atm_tilt_curve))
#PSD_atm_tilt_freq_t = np.fft.rfftfreq(len(atm_tilt_curve), d=delta_t)

#defocus
PSD_residual_defocus_freq_t, PSD_residual_defocus = welch_method_scipy(residual_defocus_curve)
#PSD_residual_defocus = np.abs(np.fft.rfft(residual_defocus_curve)) ** 2 * delta_t * 2/ (len(residual_defocus_curve))
#PSD_residual_defocus_freq_t = np.fft.rfftfreq(len(residual_defocus_curve), d=delta_t)

PSD_atm_defocus_freq_t, PSD_atm_defocus = welch_method_scipy(atm_defocus_curve)
#PSD_atm_defocus = np.abs(np.fft.rfft(atm_defocus_curve)) ** 2 * delta_t * 2/ (len(atm_defocus_curve))
#PSD_atm_defocus_freq_t = np.fft.rfftfreq(len(atm_defocus_curve), d=delta_t)


plt.figure()
plt.plot(PSD_residual_tip_freq_t, PSD_residual_tip, label = "residual_PSD_tip")
plt.plot(PSD_atm_tip_freq_t, PSD_atm_tip, label = "atm_PSD_tip")
plt.title(f"residual PSD tip, gain {CL_gain}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.ylabel("PSD")


plt.figure()
plt.plot(time_plot, residual_tip_curve, label = "residual_PSD_tip")
plt.plot(time_plot, atm_tip_curve, label = "atm_PSD_tip")
plt.title(f"timeseries tip residual vs atmosphere, gain {CL_gain}")
plt.xlabel("time (s)")
plt.legend()
plt.ylabel("residual")


plt.figure()
plt.plot(PSD_residual_tilt_freq_t, PSD_residual_tilt, label = "residual_PSD_tilt")
plt.plot(PSD_atm_tilt_freq_t, PSD_atm_tilt, label = "atm_PSD_tilt")
plt.title(f"residual PSD tilt, gain {CL_gain}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.ylabel("PSD")


plt.figure()
plt.plot(time_plot, residual_tilt_curve, label = "residual_PSD_tilt")
plt.plot(time_plot, atm_tilt_curve, label = "atm_PSD_tilt")
plt.title(f"timeseries tilt residual vs atmosphere, gain {CL_gain}")
plt.xlabel("time (s)")
plt.legend()
plt.ylabel("residual")




plt.figure()
plt.plot(PSD_residual_defocus_freq_t, PSD_residual_defocus, label = "residual_PSD_defocus")
plt.plot(PSD_atm_defocus_freq_t, PSD_atm_defocus, label = "atm_PSD_defocus")
plt.title(f"residual PSD defocus, gain {CL_gain}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.ylabel("PSD")


plt.figure()
plt.plot(time_plot, residual_defocus_curve, label = "residual_PSD_defocus")
plt.plot(time_plot, atm_defocus_curve, label = "atm_PSD_defocus")
plt.title(f"timeseries defocus residual vs atmosphere, gain {CL_gain}")
plt.xlabel("time (s)")
plt.legend()
plt.ylabel("residual")


#---------------------------------------------------spatial Error transfer function---------------------------------------------------#
sETF = PSD_residual_statavg/PSD_kolmogorov

plt.figure()
plt.plot(PSD_residual_circavg_freq, sETF)
plt.axvline(x=f_DM, color='red', linestyle='-', linewidth=1.5)
plt.title("spatial error transfer function")
plt.ylabel("ETF")
plt.xlabel("spatial frequency m^-1")
plt.xscale("log")
plt.yscale("log")
plt.xlim(right = PSD_residual_freq_x)
plt.ylim(bottom = 1e-2)




#---------------------------------------------------temporal Error transfer function---------------------------------------------------#
tETF_tip = PSD_residual_tip/PSD_atm_tip
tETF_tilt = PSD_residual_tilt/PSD_atm_tilt
tETF_defocus = PSD_residual_defocus/PSD_atm_defocus


plt.figure()
plt.plot(PSD_residual_tip_freq_t, tETF_tip, label = "ETF tip")
plt.plot(PSD_residual_tilt_freq_t, tETF_tilt, label = "ETF tilt")
plt.plot(PSD_residual_defocus_freq_t, tETF_defocus, label = "ETF defocus")
plt.title("temporal error transfer functions")
plt.ylabel("ETF")
plt.xlabel("frequency Hz")
plt.xscale("log")
plt.yscale("log")
plt.xlim(right = np.max(PSD_residual_tip_freq_t))
plt.legend()






plt.show()

























































