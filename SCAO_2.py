"""
Testing for different src and ngs (next step is extended source?)

LIST OF ARGUMENTS TO TEST:
* Play around with the light_ratio parameter for PWFS
* In the real AO bench they had a gain sensing camera? how does that incorporate here (especially papytwin I guess)


SOME ANSWERS:
* for zonal you just use the identity of M2C
"""
from torch import dtype

"""
* the PSD should have units of nm^2 m^2 (check thoroughly on paper)
    
* try doing a zero delay loop to see if the residual completely gets rid of the temporal error
* inject temporal frequencies into ATM.OPD (one above DM freq and one below)

for the PSD problem:
    * can do apodizing (welsh filter or smooth filter going to 0 over 2 pixels)
    * do maths about how the square filter affects our FFT
        * just like mentioned (and I checked) square filter causes the transfer of power from lower to higher frequencies
        * this happens due to the sidelobs and is somewhat assymetric (judging from the graphical function representation)
        * this problem is called spectral leakage (very appropriate)
    * plot theoretical atmosphere PSD (THIS IS THE ONLY SOLUTION REALLY)
    * just do modes
    * analyse the fitting error peaks? (do spatial and temporal PSDs?)
    * project the temporal error on the DM for more accurate representation
    * just check the code to see if there is a mistake in how you get your PSDs
* cut the PSD plots at the appropriate frequencies
* Fix the spatial error propagation

* Compute temporal ETF
* ETF x input should give you an understanding of how

* Add temporally varying tip-tilt

* implement cameras for the SE PSF and LE PSF

* LONG TERM TASK - ASSUME SOME ZERNIKE OUTPUT PHASE AND CONTROL A 9x9 DM (I left some papers for the weekend reading)
"""


import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Circle
import numpy as np
from numpy.fft import fftshift, fft2, fftfreq #need to shift just because of formatting

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
DIAMETER            = 1.52
CENTRAL_OBSTRUCTION = 0 #0.15
RESOLUTION          = N_SUBAPERTURE * 8
FREQUENCY           = 1000
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
Z_coefs             = 300 #200 #above 200 does not work great per Benoit



zeroPaddingFactor = 2
rad2arcsec        = 180 * 60 * 60 / np.pi


use_pwfs          = True
use_shwfs         = False

use_zernike       = False
use_zonal         = True
use_KL            = False #not yet implemented fully



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
                      mechCoupling = MECHANICAL_COUPLING)


#control radius calculation for dm
control_radius = ((N_SUBAPERTURE + 1) * SRC.wavelength) /(2 * TEL.D) * rad2arcsec
corr_zone_1 = Circle([0,0], control_radius, fc='none', ec='w', ls=':')
corr_zone_2 = Circle([0,0], control_radius, fc='none', ec='w', ls=':')

#pitch check
if (2 * DM.pitch) > r_0_src:
    raise SystemExit(f"ERROR: DM actuator density is insufficient for r_0 {r_0_src} ")

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

if use_KL:

    M2C_KL = compute_KL_basis(tel = TEL, atm = ATM, dm = DM)
    M2C = M2C_KL[:, :300]


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

ATM.generateNewPhaseScreen(seed = 2)
#reset everything just in case
TEL.resetOPD()
DM.coefs = 0
TEL + ATM
SRC * TEL * DM * WFS
TEL.print_optical_path()

#delay implementation
frame_delay         = 2
delay               = frame_delay - 1 #frame delay of 1 is already built-in
if frame_delay >= 2:
    wfssignal_buffer = [np.zeros(WFS.nSignal) for i in range(delay)]
else:
    wfssignal_buffer = []

#variables and performance metric initialisation
nLoop                        = 50
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


tel_psf_list = []


CL_gain = 0.4
reconstructor = M2C @ CALIB.M

#what is inside the loop is basically what should be in the step function of the RL OOPAO environment
for i in range(nLoop):
    #update phase screen
    ATM.update()

    total_error[i] = np.std(TEL.OPD[np.where(TEL.pupil > 0)]) * 1e9


    #propagate through AO with the dm commands applied
    SRC * TEL * DM * WFS
    #propagate to the source with the dm commands applied (old dm commands)
    #the point of this line is that for the wfs propagation you would be using NGS
    SRC * TEL


    #frame delay implementation
    wfssignal_buffer.append(WFS.signal)
    delayed_signal = wfssignal_buffer[0]
    wfssignal_buffer.pop(0)


    #update the dm commands
    DM.coefs = DM.coefs - CL_gain * np.matmul(reconstructor, delayed_signal)
    dm_coefs_copy = DM.coefs



    #fitting error calculation for every frame (written for 2 frame delay)
    mode_coefs = modes_inv @ atm_OPD_list[-3][np.where(TEL.pupil > 0)]
    DM.coefs = M2C @ mode_coefs
    fitting_error = (atm_OPD_list[-3] - DM.OPD)
    simulational_fitting_error = np.std(fitting_error[np.where(TEL.pupil > 0)]) * 1e9
    sim_fit_error_list.append(simulational_fitting_error)


    # temporal error calculation for every frame for 1, 2, 3 frame delay
    atm_OPD_list.append(ATM.OPD)
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


    if sr[i] > 0.5:
        residual_OPD_list.append(TEL.OPD)
        sim_fit_error_list_2D.append(fitting_error)
        print("residual_OPD_append")

    if i == (nLoop - 1):
        final_residual_OPD   = TEL.OPD
        final_atmosphere_OPD = ATM.OPD
        final_dm_OPD         = DM.OPD


    TEL.computePSF(zeroPaddingFactor)
    tel_psf_list.append(TEL.PSF)


#---------------------------------------------------PLOTTING---------------------------------------------------#
sim_fit_error_list = np.array(sim_fit_error_list)
sim_temp_error_2_frame_delay = np.array(sim_temp_error_2_frame_delay)
sim_fit_temp_sum = np.sqrt(sim_temp_error_2_frame_delay ** 2 + sim_fit_error_list ** 2)



time = np.arange(0, nLoop * SAMPLING_TIME, SAMPLING_TIME)



fitting_error_analytical = (0.23) ** (1/2) * (DM.pitch/ r_0_src) ** (5 / 6) * SRC.wavelength / (2 * np.pi) * 1e9
fitting_error_analytical = np.ones(len(time)) * fitting_error_analytical

fitting_error_sim_avg = np.mean(sim_fit_error_list)
fitting_error_sim_avg = np.ones(len(time)) * fitting_error_sim_avg

residual_sim_difference = residual_error - sim_fit_temp_sum


#---------------------------------------------------Error decomposition---------------------------------------------------#
plt.figure()
plt.plot(time, residual_error, label = "residual")
#plt.plot(time, sim_temp_error_2_frame_delay, label = "simulational temporal error 2 frame delay")
plt.plot(time, fitting_error_analytical, label = "analyical fitting error")
plt.plot(time, fitting_error_sim_avg, label = "simulational average fitting error")
plt.plot(time, sim_fit_error_list, label = "simulational fitting error")
plt.plot(time, sim_fit_temp_sum, label = "fitting + temporal 2 frame delay")
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


#---------------------------------------------------Zernike decomposition---------------------------------------------------#
if use_zernike:

    atmosphere_phase = 2 * np.pi * final_atmosphere_OPD / SRC.wavelength
    final_residual_phase = 2 * np.pi * final_residual_OPD / SRC.wavelength
    Z_coefficient_matrix = modes_inv @ final_residual_phase[np.where(TEL.pupil > 0)]
    Z_coefficient_matrix_atmosphere = modes_inv @ atmosphere_phase[np.where(TEL.pupil > 0)]
    zernike_names = []
    for i in range(len(Z_coefficient_matrix)):
        zernike_names.append(f"z{i+1}")


    plt.figure()
    plt.bar(zernike_names[:50], Z_coefficient_matrix[:50], color = "red", label = "Zernike coeffs for residual phase")
    plt.title("Zernike coefficients for corrected phase")
    plt.bar(zernike_names[:50],Z_coefficient_matrix_atmosphere[:50], color = "black", alpha = 0.4, label = "Zernike coeffs for atmospheric phase")
    plt.title("Zernike coefficients for atmosphere phase")
    plt.tight_layout()
    plt.legend()


#---------------------------------------------------PSD---------------------------------------------------#
#PSD is calculated using a square inside the tel.pupil
x_square = np.sqrt(TEL.resolution ** 2 / 2)
x_square_index = int((TEL.resolution - x_square) / 2 + 1)


PSD_residual_circavg_list = []
PSD_residual_circavg_freq = 0
PSD_residual_freq_x       = 0
#PSD calculation for residual OPD
for j in range(len(residual_OPD_list)):
    #residual_OPD_list[j][TEL.pupil > 0] = (residual_OPD_list[j][TEL.pupil > 0] - np.mean(residual_OPD_list[j][TEL.pupil > 0])) * 1e9
    residual_OPD = residual_OPD_list[j][x_square_index:-x_square_index, x_square_index:-x_square_index]
    residual_OPD = (residual_OPD - np.mean(residual_OPD)) * 1e9
    PSD_residual = np.abs(fftshift(fft2(residual_OPD))) ** 2 / (x_square ** 4)
    PSD_residual_freq_x = np.fft.fftfreq(PSD_residual.shape[0], d=(TEL.D / TEL.resolution))
    PSD_residual_freq_x = np.max(PSD_residual_freq_x)
    PSD_residual_freq_max = np.max(np.sqrt(2 * (PSD_residual_freq_x) ** 2))
    PSD_residual_freq, PSD_residual_circavg = circular_sum_PSD(np.abs(PSD_residual).shape, np.abs(PSD_residual), PSD_residual_freq_max)
    print("\n")
    #fix the residual error, since it plots every residual error, while the residual_OPD is instead calculated for a high strehl ratio
    print(f"TEL std {residual_error[ - len(residual_OPD_list) + j]}, TEL std from PSD {np.sqrt(np.sum(PSD_residual_circavg))}")


    if j == 0:
        PSD_residual_circavg_freq = PSD_residual_freq

    PSD_residual_circavg_list.append(PSD_residual_circavg)


PSD_fitting_circavg_list = []
PSD_fitting_circavg_freq = 0
#PSD calculation for fitting error
for j in range(len(sim_fit_error_list_2D)):
    fitting_OPD = sim_fit_error_list_2D[j][x_square_index:-x_square_index, x_square_index:-x_square_index]
    fitting_OPD = (fitting_OPD - np.mean(fitting_OPD)) * 1e9
    PSD_fitting = np.abs(fftshift(fft2(fitting_OPD))) ** 2 / (x_square ** 4)
    PSD_fitting_freq_x = np.fft.fftfreq(PSD_fitting.shape[0], d=(TEL.D / TEL.resolution))
    PSD_fitting_freq_x = np.max(PSD_fitting_freq_x)
    PSD_fitting_freq_max = np.max(np.sqrt(2 * (PSD_fitting_freq_x) ** 2))
    PSD_fitting_freq, PSD_fitting_circavg = circular_sum_PSD(np.abs(PSD_fitting).shape, np.abs(PSD_fitting), PSD_fitting_freq_max)

    if j == 0:
        PSD_fitting_circavg_freq = PSD_fitting_freq

    PSD_fitting_circavg_list.append(PSD_fitting_circavg)


PSD_atmosphere_circavg_list = []
PSD_atmosphere_circavg_freq = 0
#PSD calculation for the atmosphere
for i in range(20):
    ATM.generateNewPhaseScreen(seed = i)
    for j in range(100):
        ATM.update()
        #ATM.OPD[TEL.pupil > 0] = (ATM.OPD[TEL.pupil > 0] - np.mean(ATM.OPD[TEL.pupil > 0])) * 1e9
        atmosphere_OPD = ATM.OPD[x_square_index:-x_square_index, x_square_index:-x_square_index]
        atmosphere_OPD = (atmosphere_OPD - np.mean(atmosphere_OPD)) * 1e9
        PSD_atmosphere = np.abs(fftshift(fft2(atmosphere_OPD))) ** 2 / (x_square ** 4)
        PSD_atmosphere_freq_x = np.fft.fftfreq(PSD_atmosphere.shape[0], d = (TEL.D / TEL.resolution))
        PSD_atmosphere_freq_x = np.max(PSD_atmosphere_freq_x)
        PSD_atmosphere_freq_max = np.max(np.sqrt(2 * (PSD_atmosphere_freq_x) ** 2))
        PSD_atmosphere_freq, PSD_atmosphere_circavg = circular_sum_PSD(np.abs(PSD_atmosphere).shape, np.abs(PSD_atmosphere), PSD_atmosphere_freq_max)
        #print(f"ATM std from PSD {np.sqrt(np.sum(PSD_atmosphere_circavg))}")



        PSD_atmosphere_circavg_list.append(PSD_atmosphere_circavg)
        if i == 0:
            PSD_atmosphere_circavg_freq = PSD_atmosphere_freq

PSD_residual_circavg_list  = np.array(PSD_residual_circavg_list)
PSD_atmosphere_circavg_list = np.array(PSD_atmosphere_circavg_list)


PSD_residual_statavg = np.mean(PSD_residual_circavg_list, axis = 0)
PSD_atmosphere_statavg = np.mean(PSD_atmosphere_circavg_list, axis = 0)
PSD_fitting_statavg = np.mean(PSD_fitting_circavg_list, axis = 0)
print(f"PSD_residual average error {np.sqrt(np.sum(PSD_residual_statavg))}")
print(f"PSD_atmosphere average error {np.sqrt(np.sum(PSD_atmosphere_statavg))}")
# 7 or 8 index for fitting error (no padding)

print(f"PSD_residual fitting error 7 {np.sqrt(np.sum(PSD_residual_statavg[7: ]))}")
print(f"PSD_atmosphere fitting error 7 {np.sqrt(np.sum(PSD_atmosphere_statavg[7: ]))}")
print(f"PSD_residual fitting error 8 {np.sqrt(np.sum(PSD_residual_statavg[8: ]))} for freq {PSD_residual_circavg_freq[8]}")
print(f"PSD_atmosphere fitting error 8 {np.sqrt(np.sum(PSD_atmosphere_statavg[8: ]))} for freq {PSD_atmosphere_circavg_freq[8]}")
print("\n")

print("length PSD fit", len(PSD_fitting_statavg))
print("length PSD residual", len(PSD_residual_statavg))
print("length PSD atmosphere", len(PSD_atmosphere_statavg))
plt.figure()
plt.plot(PSD_residual_circavg_freq, PSD_residual_statavg, label = "PSD_residual_pwfs")
plt.plot(PSD_atmosphere_circavg_freq, PSD_atmosphere_statavg, label = "atmosphere_PSD")
plt.plot(PSD_fitting_circavg_freq, PSD_fitting_statavg, label = "fitting_err_PSD")
plt.axvline(x=6.57, color='red', linestyle='-', linewidth=1.5)
plt.title("PSD residual vs atmosphere comparison")
plt.ylabel("PSD")
plt.xlabel("spatial frequency m^-1")
plt.xscale("log")
plt.yscale("log")
plt.xlim(right = PSD_residual_freq_x)
plt.ylim(bottom = 1e-2)
plt.legend()

#---------------------------------------------------Error transfer function---------------------------------------------------#
ETF = PSD_residual_statavg/PSD_atmosphere_statavg

plt.figure()
plt.plot(PSD_residual_circavg_freq, ETF)
plt.axvline(x=6.57, color='red', linestyle='-', linewidth=1.5)
plt.title("Error transfer function")
plt.ylabel("ETF")
plt.xlabel("spatial frequency m^-1")
plt.xscale("log")
plt.yscale("log")
#plt.ylim(bottom = 1e-2)
plt.legend()



plt.show()

























































