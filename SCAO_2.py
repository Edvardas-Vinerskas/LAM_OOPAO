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
from matplotlib.patches import Circle
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
from OOPAO.Detector import Detector

from functions import *


#---------------------------------------------------GLOBALS---------------------------------------------------#
#define all OOPAO variables

N_SUBAPERTURE       = 20
DIAMETER            = 1.52
CENTRAL_OBSTRUCTION = 0.1 #0.15
RESOLUTION          = N_SUBAPERTURE * 8
FREQUENCY           = 1000
SAMPLING_TIME       = 1/FREQUENCY
FOV                 = 10
MECHANICAL_COUPLING = 0.4
MODULATION          = 3
LIGHT_RATIO         = 0.1
POST_PROCESS        = "slopesMaps"
r_0                 = 0.15
L_0                 = 20
WIND_SPEED          = [5, 50] #[10, 20, 60]
WIND_DIRECTION      = [30, 60] #[0, 100, 160]
FRACTIONAL_C_N2     = [0.85, 0.15] #[0.5, 0.3, 0.2]
ALTITUDE            = [0, 10000] #[0, 4500, 10000]
Z_coefs             = 300 #200 #above 200 does not work great per Benoit



zeroPaddingFactor = 2
rad2arcsec        = 180 * 60 * 60 / np.pi


use_pwfs          = True
use_shwfs         = False

use_zernike       = True
use_zonal         = False
use_KL            = False



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
M2C = None
if use_zonal:
    M2C_zonal = np.identity(DM.nValidAct)
    M2C       = M2C_zonal


if use_zernike:

    zernike = Zernike(telObject = TEL,
                       J = Z_coefs)
    zernike.computeZernike(telObject2 = TEL)

    M2C_Z = np.linalg.pinv(np.squeeze(DM.modes[TEL.pupilLogical, :])) @ zernike.modes
    M2C   = M2C_Z

if use_KL:

    M2C_KL = compute_KL_basis(tel = TEL, atm = ATM, dm = DM)
    M2C = M2C_KL[:, :300]


#---------------------------------------------------INTERACTION MATRIX---------------------------------------------------#



stroke = SRC.wavelength / 16
CALIB = InteractionMatrix(ngs            = SRC,
                          tel            = TEL,
                          dm             = DM,
                          wfs            = PWFS,
                          M2C            = M2C,
                          atm            = ATM,
                          nMeasurements  = 5,
                          stroke         = stroke,
                          noise          = "off")




#---------------------------------------------------SIMULATION---------------------------------------------------#
#for now we are only using SRC


#reset everything just in case
TEL.resetOPD()
DM.coefs = 0
TEL + ATM
SRC * TEL * DM * WFS
TEL.print_optical_path()

#delay implementation
frame_delay         = 1
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

#for temp_err_delay = 3, the list has current, previous and previous_previous
atm_OPD_list                 = []
temp_err_delay = 3 #don't change
sim_temp_error_1_frame_delay = []
sim_temp_error_2_frame_delay = []
sim_temp_error_3_frame_delay = []
sim_fit_error_list = []


CL_gain = 0.4
reconstructor = M2C @ CALIB.M


for i in range(nLoop):
    #update phase screen
    ATM.update()

    #temporal error calculation for every frame for 1, 2, 3 frame delay
    atm_OPD_list.append(ATM.OPD)
    if i >= temp_err_delay:
        atm_OPD_1_frame = (atm_OPD_list[0] - atm_OPD_list[1])[np.where(TEL.pupil > 0)]
        atm_OPD_2_frame = (atm_OPD_list[0] - atm_OPD_list[2])[np.where(TEL.pupil > 0)]
        atm_OPD_3_frame = (atm_OPD_list[0] - atm_OPD_list[3])[np.where(TEL.pupil > 0)]

        sim_temp_error_1_frame_delay.append(np.std(atm_OPD_1_frame) * 1e9)
        sim_temp_error_2_frame_delay.append(np.std(atm_OPD_2_frame) * 1e9)
        sim_temp_error_3_frame_delay.append(np.std(atm_OPD_3_frame) * 1e9)

        atm_OPD_list.pop(0)


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


    frame_delay = 4 #changed the stroke btw (change it back, did not help)
    #the error is because the dm corrects for the phase
    #in SCAO_1 I had instead remapped the phase of the atmosphere onto the DM
    #here I am not remapping, so the OPD should be 1/2 for the DM and also flipped in phase
    if i >= frame_delay:
        fitting_error = atm_OPD_list[-2] + 2 * DM.OPD

        if i >= 49:
            plt.figure()

            plt.subplot(131)
            plt.imshow(atm_OPD_list[-2])
            plt.title("previous cycle atm.OPD")
            plt.subplot(133)
            plt.imshow(-2 * DM.OPD)
            plt.title("DM.OPD")
            plt.subplot(132)
            plt.imshow(atm_OPD_list[-3])
            plt.title("2 previous cycle atm.OPD")
            plt.tight_layout()
            plt.show()


        sim_fitting_error = np.std(fitting_error[np.where(TEL.pupil > 0)]) * 1e9
        print("sim_fitting_error", sim_fitting_error)
        print(np.average(atm_OPD_list[(-2)]), np.average(DM.OPD))
        sim_fit_error_list.append(sim_fitting_error)



    #performance metrics
    sr[i] = np.exp(-np.var(TEL.src.phase[np.where(TEL.pupil == 1)]))
    residual_error[i] = np.std(TEL.OPD[np.where(TEL.pupil > 0)]) * 1e9
    print("Loop" + str(i) + "/" + str(nLoop) + "AO residual: " + str(residual_error[i]) + "nm")
    print(f"strehl {sr[i]}")
    if i == (nLoop - 1):
        final_residual_phase   = TEL.src.phase
        final_atmosphere_OPD   = ATM.OPD
        final_dm_OPD           = DM.OPD


#---------------------------------------------------PLOTTING + ERRORS---------------------------------------------------#



time = np.arange(0, nLoop * SAMPLING_TIME, SAMPLING_TIME)
time_temp = np.arange(temp_err_delay * SAMPLING_TIME, nLoop * SAMPLING_TIME, SAMPLING_TIME)
time_fit  = np.arange(frame_delay * SAMPLING_TIME, nLoop * SAMPLING_TIME, SAMPLING_TIME)
plt.figure()
plt.plot(time, residual_error, label = "residual")
plt.plot(time_temp, sim_temp_error_1_frame_delay, label = "simulational temporal error 1 frame delay")
plt.plot(time_temp, sim_temp_error_2_frame_delay, label = "simulational temporal error 2 frame delay")
plt.plot(time_temp, sim_temp_error_3_frame_delay, label = "simulational temporal error 3 frame delay")
plt.plot(time_fit, sim_fit_error_list, label = "simulational fitting error")
plt.title("error decomposition (nm)")
plt.xlabel("time s")
plt.yscale("log")
plt.legend()
plt.show()
















































