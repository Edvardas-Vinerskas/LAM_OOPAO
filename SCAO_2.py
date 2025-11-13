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
from torch import dtype

"""
* now the RL has been done for central_obstruction = 0
    *redo it for central_obstruction = 0.1

*fix all of the units
* Normalised the PSD to preserve variance, however the corrected PSD and atmosphere PSD still do not merge
    *a potential idea is to simulate a high resolution phase screen, input the correct  pixel spacing and sum up the error above the DM cut off frequency
    *you should get the fitting error if everything is correct
    *plot the fitting error in 2 ways
        * atm - dm -> calculate std
        * atm -> PSD -> block out the DM frequencies -> atm -> calculate std
        * this might not work that well as the second method does not take into account influence functions but maybe just the right ballpark?
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
plt.figure()
plt.imshow(ATM.OPD)
plt.show()

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
nLoop                        = 100
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
sim_temp_error_1_frame_delay = []
sim_temp_error_2_frame_delay = []
sim_temp_error_3_frame_delay = []
sim_fit_error_list           = []


CL_gain = 0.4
reconstructor = M2C @ CALIB.M

#what is inside the loop is basically what should be in the step function of the RL OOPAO environment
for i in range(nLoop):
    #update phase screen
    ATM.update()

    #temporal error calculation for every frame for 1, 2, 3 frame delay
    atm_OPD_list.append(ATM.OPD)
    atm_OPD_list.pop(0)

    atm_OPD_1_frame = (atm_OPD_list[-1] - atm_OPD_list[-2])[np.where(TEL.pupil > 0)]
    atm_OPD_2_frame = (atm_OPD_list[-1] - atm_OPD_list[-3])[np.where(TEL.pupil > 0)]
    atm_OPD_3_frame = (atm_OPD_list[-1] - atm_OPD_list[-4])[np.where(TEL.pupil > 0)]

    sim_temp_error_1_frame_delay.append(np.std(atm_OPD_1_frame) * 1e9)
    sim_temp_error_2_frame_delay.append(np.std(atm_OPD_2_frame) * 1e9)
    sim_temp_error_3_frame_delay.append(np.std(atm_OPD_3_frame) * 1e9)



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


    #convert back the DM.coefs to their actual CL values
    DM.coefs = dm_coefs_copy



    #performance metrics
    sr[i] = np.exp(-np.var(TEL.src.phase[np.where(TEL.pupil == 1)]))
    residual_error[i] = np.std(TEL.OPD[np.where(TEL.pupil > 0)]) * 1e9
    print("Loop" + str(i) + "/" + str(nLoop) + " " + "AO residual: " + str(residual_error[i]) + "nm" + "total err: " + str(total_error[i]) + "nm")
    print(f"strehl {sr[i]}")


    if sr[i] > 0.5:
        residual_OPD_list.append(TEL.OPD)
        print("residual_OPD_append")

    if i == (nLoop - 1):
        final_residual_OPD   = TEL.OPD
        final_atmosphere_OPD = ATM.OPD
        final_dm_OPD         = DM.OPD


#---------------------------------------------------PLOTTING---------------------------------------------------#
sim_fit_error_list = np.array(sim_fit_error_list)
sim_temp_error_2_frame_delay = np.array(sim_temp_error_2_frame_delay)
sim_fit_temp_sum = np.sqrt(sim_temp_error_2_frame_delay ** 2 + sim_fit_error_list ** 2)



time = np.arange(0, nLoop * SAMPLING_TIME, SAMPLING_TIME)



fitting_error_analytical = (0.23) ** (1/2) * (DM.pitch/ r_0_src) ** (5 / 6) * SRC.wavelength / (2 * np.pi) * 1e9
fitting_error_analytical = np.ones(len(time)) * fitting_error_analytical


residual_sim_difference = residual_error - sim_fit_temp_sum


#---------------------------------------------------Error decomposition---------------------------------------------------#
plt.figure()
plt.plot(time, residual_error, label = "residual")
#plt.plot(time, sim_temp_error_1_frame_delay, label = "simulational temporal error 1 frame delay")
#plt.plot(time, sim_temp_error_2_frame_delay, label = "simulational temporal error 2 frame delay")
plt.plot(time, fitting_error_analytical, label = "analyical fitting error")
plt.plot(time, sim_fit_error_list, label = "simulational fitting error")
plt.plot(time, sim_fit_temp_sum, label = "fitting + temporal 2 frame delay")
#plt.plot(time, residual_sim_difference, label = "difference between residual and simulational errors")
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
plt.imshow(AO_PSF, norm = SymLogNorm(1e-7), extent = [-fov/2, fov/2, -fov/2, fov/2])
plt.title("AO corrected PSF")
plt.gca().add_artist(corr_zone_1)
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
#if ever your PSD integral does not agree with residual error, then check the values of the OPD outside the pupil mask (they should be 0)
PSD_residual_circavg_list = []
PSD_residual_circavg_freq = 0

PSD_atmosphere_circavg_list = []
PSD_atmosphere_circavg_freq = 0
for j in range(len(residual_OPD_list)):
    residual_OPD_list[j][TEL.pupil > 0] = (residual_OPD_list[j][TEL.pupil > 0] - np.mean(residual_OPD_list[j][TEL.pupil > 0])) * 1e9
    residual_OPD = residual_OPD_list[j]
    PSD_residual = np.abs(fftshift(fft2(residual_OPD))) ** 2 / (len(TEL.pupil[TEL.pupil > 0]) ** 2)
    PSD_residual_freq_x = np.fft.fftfreq(PSD_residual.shape[0], d=(TEL.D / TEL.resolution))
    PSD_residual_freq_max = np.max(np.sqrt(2 * (PSD_residual_freq_x) ** 2))
    PSD_residual_freq, PSD_residual_circavg = circular_sum_PSD(np.abs(PSD_residual).shape, np.abs(PSD_residual), PSD_residual_freq_max)
    print("\n")
    #fix the residual error, since it plots every residual error, while the residual_OPD is instead calculated for a high strehl ratio
    print(f"TEL std {residual_error[ - len(residual_OPD_list) + j]}, TEL std from PSD {np.sqrt(np.sum(PSD_residual_circavg))}, TEL std from PSD {np.sqrt(np.sum(PSD_residual))}")

    if j == 0:
        PSD_residual_circavg_freq = PSD_residual_freq

    PSD_residual_circavg_list.append(PSD_residual_circavg)


for i in range(20):
    ATM.generateNewPhaseScreen(seed = i)
    for j in range(100):
        ATM.update()
        ATM.OPD[TEL.pupil > 0] = (ATM.OPD[TEL.pupil > 0] - np.mean(ATM.OPD[TEL.pupil > 0])) * 1e9
        atmosphere_OPD = ATM.OPD
        PSD_atmosphere = np.abs(fftshift(fft2(atmosphere_OPD))) ** 2 / (len(TEL.pupil[TEL.pupil > 0]) ** 2)
        PSD_atmosphere_freq_x = np.fft.fftfreq(PSD_atmosphere.shape[0], d = (TEL.D / TEL.resolution))
        PSD_atmosphere_freq_max = np.max(np.sqrt(2 * (PSD_atmosphere_freq_x) ** 2))
        PSD_atmosphere_freq, PSD_atmosphere_circavg = circular_sum_PSD(np.abs(PSD_atmosphere).shape, np.abs(PSD_atmosphere), PSD_atmosphere_freq_max)




        PSD_atmosphere_circavg_list.append(PSD_atmosphere_circavg)
        if i == 0:
            PSD_atmosphere_circavg_freq = PSD_atmosphere_freq

PSD_residual_circavg_list  = np.array(PSD_residual_circavg_list)
PSD_atmosphere_circavg_list = np.array(PSD_atmosphere_circavg_list)


PSD_residual_statavg = np.mean(PSD_residual_circavg_list, axis = 0)
PSD_atmosphere_statavg = np.mean(PSD_atmosphere_circavg_list, axis = 0)
print(PSD_residual_statavg.shape, PSD_residual_circavg_list.shape)
print(PSD_atmosphere_statavg.shape, PSD_atmosphere_circavg_list.shape)
print(np.sqrt(np.sum(PSD_residual_statavg)))
print(np.sqrt(np.sum(PSD_atmosphere_statavg)))
print("\n")
print(np.sqrt(np.sum(PSD_residual_statavg[11:]))) #not sure how useful this is since there is aliasing from the FFT itself when calculating the PSD and also temporal error?
print(np.sqrt(np.sum(PSD_atmosphere_statavg[11:])))




plt.figure()
plt.plot(PSD_residual_circavg_freq, PSD_residual_statavg, label = "PSD_residual_pwfs")
plt.plot(PSD_atmosphere_circavg_freq, PSD_atmosphere_statavg, label = "atmosphere_PSD")
plt.title("PSD residual vs atmosphere comparison")
plt.xlabel("Frequency domain")
plt.ylabel("PSD")
plt.xlabel("spatial frequency m^-1")
plt.xscale("log")
plt.yscale("log")
plt.ylim(bottom = 1e-2)
plt.legend()


plt.show()

























































