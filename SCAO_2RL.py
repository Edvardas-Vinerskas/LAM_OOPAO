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
"""
* now the RL has been done for central_obstruction = 0
    *redo it for central_obstruction = 0.1
    * when redoing it, you must change how you split for zernike tip tilt modes
    * because the current version only works with no central obstruction
    * there are apparently fitting error functions in OOPAO.calibration

* ON FRIDAY YOU FOR ONCE FIGURE OUT ALL OF THE WEBSITES THAT YOU SHOULD HAVE ACCESS TO
* estimate the wfs limited resolution error
* I guess estimate all possible errors due to limited resolution
* calculate std for the errors

* something is wrong with how I calculate errors and the residual seems to be dominated be the residual error at all times
* the DM is reconstructing half the phase, and so the error is twice as small!
* I should also recalculate the fitting error based on old phase screens and NOT the current ones
    * this would effectively just shift my fitting error curve to the right
    * and thus probably better align with the temporal error
    * AND OF COURSE THE CURRENT DM IS TRYING TO FIT THE OLD PHASE SO CLEARLY YOU ARE NOT SMART
    
* integrate all of the important observation parameters into reset and step functions
"""

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Circle
import gymnasium as gym
import torch
import numpy as np
from numpy.fft import fftshift, fft2  # need to shift just because of formatting

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
from OOPAO_environment import OOPAO_environment
from RL_test import Actor

# ---------------------------------------------------GLOBALS---------------------------------------------------#
# define all OOPAO variables

N_SUBAPERTURE = 20
DIAMETER = 1.52
CENTRAL_OBSTRUCTION = 0  # 0.15
RESOLUTION = N_SUBAPERTURE * 8
FREQUENCY = 1000
SAMPLING_TIME = 1 / FREQUENCY
FOV = 10
MECHANICAL_COUPLING = 0.35
MODULATION = 3
LIGHT_RATIO = 0.1
POST_PROCESS = "slopesMaps"
r_0 = 0.15
L_0 = 25
WIND_SPEED = [10, 20, 60]  # [10, 20, 60]
WIND_DIRECTION = [0, 100, 160]  # [0, 100, 160]
FRACTIONAL_C_N2 = [0.6, 0.3, 0.1]  # [0.5, 0.3, 0.2]
ALTITUDE = [0, 4500, 10000]  # [0, 4500, 10000]
Z_coefs = 50  #above 200 does not work great per Benoit

zeroPaddingFactor = 6 #6 used in RL training
rad2arcsec = 180 * 60 * 60 / np.pi

use_pwfs = True
use_shwfs = False

use_zernike = True
use_zonal = False
use_KL = False

# pixel_size check for sufficient r_0 sampling
pixel_size = DIAMETER / RESOLUTION
if (3 * pixel_size) > r_0:
    raise SystemExit("ERROR: pixel size is too big for r_0 value")

# ---------------------------------------------------RL ACTOR---------------------------------------------------#
use_RL_actor = True

env = OOPAO_environment()

def make_env():
    def thunk():
        env = OOPAO_environment()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

envs = gym.vector.SyncVectorEnv([make_env()])


actor = Actor(envs)

#load the model from the models directory
#actor.load_state_dict(torch.load("./models/best_model_delay_run_0.pth", map_location=torch.device('cpu'))["model_state_dict"])
#actor.load_state_dict(torch.load("./models/best_model_delay_run_0_Z_30.pth", map_location=torch.device('cpu'))["model_state_dict"])
actor.load_state_dict(torch.load("./models/best_model_delay_run_0_Z_50.pth", map_location=torch.device('cpu'))["model_state_dict"])
#actor.load_state_dict(torch.load("./models/best_model_delay_run_0_Z_100.pth", map_location=torch.device('cpu'))["model_state_dict"])


# ---------------------------------------------------SOURCE---------------------------------------------------#

NGS = Source(optBand="I",
             magnitude=2)

SRC = Source(optBand="I",
             magnitude=2)

wvl = 500e-9  # r_0 specified above is for this wavelength
r_0_src = r_0 * (SRC.wavelength / wvl) ** (6 / 5)
r_0_ngs = r_0 * (NGS.wavelength / wvl) ** (6 / 5)

# ---------------------------------------------------TELESCOPE---------------------------------------------------#
TEL = Telescope(resolution=RESOLUTION,
                diameter=DIAMETER,
                samplingTime=SAMPLING_TIME,
                centralObstruction=CENTRAL_OBSTRUCTION)
# fov                 = FOV)

# MUST couple source object to telescope
SRC * TEL

TEL.computePSF(zeroPaddingFactor)
# diffraction limited OTF calculation
OTF_dl = fftshift(fft2(fftshift(TEL.PSF / np.sum(TEL.PSF))))
x_axis, OTF_dl_averaged = circular_average((np.abs(OTF_dl)).shape, np.abs(OTF_dl))

# ---------------------------------------------------ATMOSPHERE---------------------------------------------------#
ATM = Atmosphere(telescope=TEL,
                 r0=r_0,
                 L0=L_0,
                 windSpeed=WIND_SPEED,
                 windDirection=WIND_DIRECTION,
                 fractionalR0=FRACTIONAL_C_N2,
                 altitude=ALTITUDE
                 )

ATM.initializeAtmosphere(telescope=TEL)

# ---------------------------------------------------DEFORMABLE_MIRROR---------------------------------------------------#

DM = DeformableMirror(telescope=TEL,
                      nSubap=N_SUBAPERTURE,
                      mechCoupling=MECHANICAL_COUPLING)

# control radius calculation for dm
control_radius = ((N_SUBAPERTURE + 1) * SRC.wavelength) / (2 * TEL.D) * rad2arcsec
corr_zone_1 = Circle([0, 0], control_radius, fc='none', ec='w', ls=':')
corr_zone_2 = Circle([0, 0], control_radius, fc='none', ec='w', ls=':')

# pitch check
if (2 * DM.pitch) > r_0_src:
    raise SystemExit(f"ERROR: DM actuator density is insufficient for r_0 {r_0_src} ")

# ---------------------------------------------------WFS---------------------------------------------------#
WFS = None
if use_pwfs:
    PWFS = Pyramid(nSubap=N_SUBAPERTURE,
                   telescope=TEL,
                   modulation=MODULATION,
                   lightRatio=LIGHT_RATIO,
                   postProcessing=POST_PROCESS)
    WFS = PWFS
if use_shwfs:
    SHWFS = ShackHartmann(nSubap=N_SUBAPERTURE,
                          telescope=TEL,
                          lightRatio=LIGHT_RATIO,
                          is_geometric=False)
    WFS = SHWFS

# ---------------------------------------------------MODAL_BASIS---------------------------------------------------#
M2C = None
modes = None
if use_zonal:
    M2C_zonal = np.identity(DM.nValidAct)
    M2C = M2C_zonal
    modes = DM.modes

if use_zernike:
    zernike = Zernike(telObject=TEL,
                      J=Z_coefs)
    zernike.computeZernike(telObject2=TEL)

    M2C_Z = np.linalg.pinv(np.squeeze(DM.modes[TEL.pupilLogical, :])) @ zernike.modes
    M2C = M2C_Z
    modes = zernike.modes

if use_KL:
    M2C_KL = compute_KL_basis(tel=TEL, atm=ATM, dm=DM)
    M2C = M2C_KL[:, :300]

# ---------------------------------------------------INTERACTION MATRIX---------------------------------------------------#


stroke = SRC.wavelength / 16
CALIB = InteractionMatrix(ngs=SRC,
                          tel=TEL,
                          dm=DM,
                          wfs=WFS,
                          M2C=M2C,
                          atm=ATM,
                          nMeasurements=5,
                          stroke=stroke,
                          noise="off")

# ---------------------------------------------------FITTING ERROR CALC---------------------------------------------------#

# for zernike and zonal only (later extract the KL modes from the source code?)
modes_inv = None
# the rest of the code is in the for loop

# takes in phase and outputs modes
if use_zonal:
    modes_inv = np.linalg.pinv(np.squeeze(modes[TEL.pupilLogical, :]))

if use_zernike:
    modes_inv = np.linalg.pinv(np.squeeze(modes))

# the rest of the code is in the for loop

# ---------------------------------------------------SIMULATION---------------------------------------------------#
# for now we are only using SRC


# reset everything just in case
TEL.resetOPD()
DM.coefs = 0
TEL + ATM
SRC * TEL * DM * WFS
TEL.print_optical_path()

# delay implementation
frame_delay = 2
delay = frame_delay - 1  # frame delay of 1 is already built-in
if frame_delay >= 2:
    wfssignal_buffer = [np.zeros(WFS.nSignal) for i in range(delay)]
else:
    wfssignal_buffer = []

# variables and performance metric initialisation
nLoop = 1000
sr = np.zeros(nLoop)
sr_running = np.zeros(nLoop)
total_error = np.zeros(nLoop)
residual_error = np.zeros(nLoop)
final_residual_phase = 0
final_atmosphere_OPD = 0
final_dm_OPD = 0

# for temp_err_delay = 3, the list has current, previous and previous_previous
temp_err_delay = 4
atm_OPD_list = [np.zeros(ATM.OPD.shape) for i in range((
                                                                   temp_err_delay + 1))]  # (temp_err_delay + 1) tells you how many current + previous frames you want to keep in the buffer
sim_temp_error_1_frame_delay = []
sim_temp_error_2_frame_delay = []
sim_temp_error_3_frame_delay = []
sim_fit_error_list = []

CL_gain = 0.4
reconstructor = M2C @ CALIB.M

ATM.generateNewPhaseScreen(seed=0)
for i in range(nLoop):
    # update phase screen
    ATM.update()

    # temporal error calculation for every frame for 1, 2, 3 frame delay
    atm_OPD_list.append(ATM.OPD)
    atm_OPD_list.pop(0)

    atm_OPD_1_frame = (atm_OPD_list[-1] - atm_OPD_list[-2])[np.where(TEL.pupil > 0)]
    atm_OPD_2_frame = (atm_OPD_list[-1] - atm_OPD_list[-3])[np.where(TEL.pupil > 0)]
    atm_OPD_3_frame = (atm_OPD_list[-1] - atm_OPD_list[-4])[np.where(TEL.pupil > 0)]

    sim_temp_error_1_frame_delay.append(np.std(atm_OPD_1_frame) * 1e9)
    sim_temp_error_2_frame_delay.append(np.std(atm_OPD_2_frame) * 1e9)
    sim_temp_error_3_frame_delay.append(np.std(atm_OPD_3_frame) * 1e9)

    total_error[i] = np.std(TEL.OPD[np.where(TEL.pupil > 0)]) * 1e9

    # propagate through AO with the dm commands applied
    SRC * TEL * DM * WFS
    # propagate to the source with the dm commands applied (old dm commands)
    # the point of this line is that for the wfs propagation you would be using NGS
    SRC * TEL

    # frame delay implementation
    wfssignal_buffer.append(WFS.signal)
    delayed_signal = wfssignal_buffer[0]
    wfssignal_buffer.pop(0)

    # update the dm commands
    DM.coefs = DM.coefs - CL_gain * np.matmul(reconstructor, delayed_signal)
    dm_coefs_copy = DM.coefs

    # fitting error calculation for every frame (written for 2 frame delay)
    mode_coefs = modes_inv @ atm_OPD_list[-3][np.where(TEL.pupil > 0)]
    DM.coefs = M2C @ mode_coefs
    fitting_error = 0.5 * (atm_OPD_list[-3] - DM.OPD)
    simulational_fitting_error = np.std(fitting_error[np.where(TEL.pupil > 0)]) * 1e9
    sim_fit_error_list.append(simulational_fitting_error)

    # convert back the DM.coefs to their actual CL values
    DM.coefs = dm_coefs_copy

    # performance metrics
    sr[i] = np.exp(-np.var(TEL.src.phase[np.where(TEL.pupil == 1)]))
    residual_error[i] = np.std(TEL.OPD[np.where(TEL.pupil > 0)]) * 1e9
    print("Loop" + str(i) + "/" + str(nLoop) + "AO residual: " + str(residual_error[i]) + "nm")
    print(f"strehl {sr[i]}")
    if i == (nLoop - 1):
        final_residual_phase = TEL.src.phase
        final_atmosphere_OPD = ATM.OPD
        final_dm_OPD = DM.OPD

#---------------------------------------------------RL LOOP---------------------------------------------------#

# variables and performance metric initialisation
sr_RL = np.zeros(nLoop)
sr_running_RL = np.zeros(nLoop)
total_error_RL = np.zeros(nLoop)
residual_error_RL = np.zeros(nLoop)
final_residual_phase_RL = 0
final_atmosphere_OPD_RL = 0
final_dm_OPD_RL = 0

# for temp_err_delay = 3, the list has current, previous and previous_previous
temp_err_delay = 4
atm_OPD_list_RL = [np.zeros(ATM.OPD.shape) for i in range((temp_err_delay + 1))]  # (temp_err_delay + 1) tells you how many current + previous frames you want to keep in the buffer
sim_temp_error_1_frame_delay_RL = []
sim_temp_error_2_frame_delay_RL = []
sim_temp_error_3_frame_delay_RL = []
sim_fit_error_list_RL = []


#the RL only applies correction to tt
M2C_TT = M2C[:, :env.J_corr]
obs, info = env.reset(seed=0)
ATM.generateNewPhaseScreen(seed=0)
for i in range(nLoop):

    # temporal error calculation for every frame for 1, 2, 3 frame delay
    atm_OPD_list.append(env.ATM.OPD)
    atm_OPD_list.pop(0)

    atm_OPD_1_frame_RL = (atm_OPD_list[-1] - atm_OPD_list[-2])[np.where(env.TEL.pupil > 0)]
    atm_OPD_2_frame_RL = (atm_OPD_list[-1] - atm_OPD_list[-3])[np.where(env.TEL.pupil > 0)]
    atm_OPD_3_frame_RL = (atm_OPD_list[-1] - atm_OPD_list[-4])[np.where(env.TEL.pupil > 0)]

    sim_temp_error_1_frame_delay_RL.append(np.std(atm_OPD_1_frame_RL) * 1e9)
    sim_temp_error_2_frame_delay_RL.append(np.std(atm_OPD_2_frame_RL) * 1e9)
    sim_temp_error_3_frame_delay_RL.append(np.std(atm_OPD_3_frame_RL) * 1e9)

    total_error_RL[i] = np.std(env.TEL.OPD[np.where(env.TEL.pupil > 0)]) * 1e9


    actions, _, _ = actor.get_action(torch.Tensor(obs[np.newaxis, :]))
    obs, reward, terminated, truncated, info = env.step(actions[0].detach().numpy())


    # performance metrics
    sr_RL[i] = info["strehl"]
    residual_error_RL[i] = np.std(env.TEL.OPD[np.where(env.TEL.pupil > 0)]) * 1e9
    print("Loop" + str(i) + "/" + str(nLoop) + "AO residual: " + str(residual_error_RL[i]) + "nm")
    print(f"strehl {sr_RL[i]}")
    if i == (nLoop - 1):
        final_residual_phase_RL = env.TEL.src.phase
        final_atmosphere_OPD_RL = env.ATM.OPD
        final_dm_OPD_RL = env.DM.OPD

#---------------------------------------------------PLOTTING---------------------------------------------------#
sim_fit_error_list = np.array(sim_fit_error_list)
sim_temp_error_2_frame_delay = np.array(sim_temp_error_2_frame_delay)
sim_fit_temp_sum = sim_temp_error_2_frame_delay + sim_fit_error_list


time = np.arange(0, nLoop * SAMPLING_TIME, SAMPLING_TIME)



#---------------------------------------------------Error decomposition---------------------------------------------------#
plt.figure()
plt.plot(time, residual_error, label = "residual integrator")
plt.plot(time, residual_error_RL, label = "residual_RL")
plt.plot(time, total_error_RL, label = "total_error_RL")
#plt.plot(time, sim_temp_error_1_frame_delay, label = "simulational temporal error 1 frame delay")
plt.plot(time, sim_temp_error_2_frame_delay, label = "simulational temporal error 2 frame delay")
#plt.plot(time, sim_temp_error_3_frame_delay, label = "simulational temporal error 3 frame delay")
plt.plot(time, sim_fit_error_list, label = "simulational fitting error")
plt.plot(time, sim_fit_temp_sum, label = "fitting + temporal 2 frame delay")
plt.title("error decomposition (nm)")
plt.xlabel("time s")
plt.yscale("log")
plt.legend()







#---------------------------------------------------Strehl---------------------------------------------------#
sr_mean = np.mean(sr)
sr_mean_RL = np.mean(sr_RL)
kernel = np.ones(30) / 30

# pad sr
pad_left = len(kernel) // 2
pad_right = len(kernel) - pad_left - 1
sr_padded = np.pad(sr, (pad_left, pad_right), mode='constant', constant_values=sr_mean)
sr_running = np.convolve(sr_padded, kernel, mode='valid')

sr_padded_RL = np.pad(sr_RL, (pad_left, pad_right), mode='constant', constant_values=sr_mean_RL)
sr_running_RL = np.convolve(sr_padded_RL, kernel, mode='valid')


plt.figure()
plt.plot(time, sr, label = "strehl integrator", alpha = 0.5)
plt.plot(time, sr_RL, label = "strehl integrator", alpha = 0.5)
plt.plot(time, sr_running, label = "running_strehl integrator")
plt.plot(time, sr_running_RL, label = "running_strehl RL")
plt.title("Strehl ratio")
plt.xlabel("time s")
plt.ylim(bottom = (sr_mean - 0.3))
plt.legend()


#---------------------------------------------------AO PSF---------------------------------------------------#

plt.figure()
TEL.computePSF(zeroPaddingFactor)
env.TEL.computePSF(zeroPaddingFactor)

arcsec_per_pixel = 206265 * (TEL.src.wavelength/TEL.D) / zeroPaddingFactor
N = 50
AO_PSF = TEL.PSF[N: -N, N: -N]
fov    = AO_PSF.shape[0] * arcsec_per_pixel
AO_PSF = AO_PSF/np.sum(AO_PSF)

AO_PSF_RL = env.TEL.PSF[N: -N, N: -N]
AO_PSF_RL = AO_PSF_RL/np.sum(AO_PSF_RL)

plt.subplot(121)
plt.imshow(AO_PSF, norm = SymLogNorm(1e-7), extent = [-fov/2, fov/2, -fov/2, fov/2])
plt.title("AO corrected PSF ")
plt.gca().add_artist(corr_zone_1)
plt.xlabel("arcsec")
plt.ylabel("arcsec")

plt.subplot(122)
plt.imshow(AO_PSF_RL, norm = SymLogNorm(1e-7), extent = [-fov/2, fov/2, -fov/2, fov/2])
plt.title("AO corrected PSF RL")
plt.gca().add_artist(corr_zone_2)
plt.xlabel("arcsec")
plt.ylabel("arcsec")
plt.colorbar()

#---------------------------------------------------AO OTF---------------------------------------------------#


OTF_AO = fftshift(fft2(fftshift(TEL.PSF / np.sum(TEL.PSF))))
x_axis___, OTF_AO_averaged = circular_average(np.abs(OTF_AO).shape, np.abs(OTF_AO))

OTF_AO_RL = fftshift(fft2(fftshift(env.TEL.PSF / np.sum(env.TEL.PSF))))
x_axis___RL, OTF_AO_averaged_RL = circular_average(np.abs(OTF_AO_RL).shape, np.abs(OTF_AO_RL))

plt.figure()
plt.plot(x_axis, OTF_dl_averaged, label = "diffraction limited")
plt.plot(x_axis___, OTF_AO_averaged, label = "AO corrected pwfs")
plt.plot(x_axis___RL, OTF_AO_averaged_RL, label = "AO corrected pwfs RL")
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
    Z_coefficient_matrix = modes_inv @ final_residual_phase[np.where(TEL.pupil > 0)]
    Z_coefficient_matrix_atmosphere = modes_inv @ atmosphere_phase[np.where(TEL.pupil > 0)]
    zernike_names = []
    for i in range(len(Z_coefficient_matrix)):
        zernike_names.append(f"z{i+1}")


    plt.figure()
    plt.bar(zernike_names[:Z_coefs], Z_coefficient_matrix[:Z_coefs], color = "red", label = "Zernike coeffs for residual phase")
    plt.bar(zernike_names[:Z_coefs], info["tt_modes"], color="b",
            label="Zernike coeffs for residual phase RL")
    plt.title("Zernike coefficients")
    plt.bar(zernike_names[:Z_coefs],Z_coefficient_matrix_atmosphere[:Z_coefs], color = "black", alpha = 0.4, label = "Zernike coeffs for atmospheric phase")
    plt.tight_layout()
    plt.legend()


#---------------------------------------------------PSD---------------------------------------------------#

corrected_phase = final_residual_phase
PSD_corrected = np.abs(fftshift(fft2(corrected_phase))) ** 2 / ((TEL.D / 2) ** 2 * np.pi)
x_axis_PSD_residual, PSD_corrected_averaged = circular_average(np.abs(PSD_corrected).shape, np.abs(PSD_corrected))

corrected_phase_RL = final_residual_phase_RL
PSD_corrected_RL = np.abs(fftshift(fft2(corrected_phase_RL))) ** 2 / ((env.TEL.D / 2) ** 2 * np.pi)
x_axis_PSD_residual_RL, PSD_corrected_averaged_RL = circular_average(np.abs(PSD_corrected_RL).shape, np.abs(PSD_corrected_RL))

atmosphere_phase = 2*np.pi * final_atmosphere_OPD / SRC.wavelength
PSD_atmosphere = np.abs(fftshift(fft2(atmosphere_phase))) ** 2 / ((TEL.D / 2) ** 2 * np.pi)
x_axis_PSD_atmosphere_residual, atmosphere_residual_averaged = circular_average(np.abs(PSD_atmosphere).shape, np.abs(PSD_atmosphere))

plt.figure()
plt.subplot(121)
plt.imshow(final_residual_phase)
plt.title("integrator residual phase")
plt.subplot(122)
plt.imshow(final_residual_phase_RL)
plt.title("RL residual phase")


plt.figure()
plt.plot(x_axis_PSD_residual, PSD_corrected_averaged, label = "PSD_residual_pwfs")
plt.plot(x_axis_PSD_residual_RL, PSD_corrected_averaged_RL, label = "PSD_residual_pwfs_RL")
plt.plot(x_axis_PSD_atmosphere_residual, atmosphere_residual_averaged, label = "atmosphere_PSD")
plt.title("PSD residual vs atmosphere comparison")
plt.xlabel("Frequency domain")
plt.ylabel("PSD")
plt.xscale("log")
plt.yscale("log")
plt.ylim(bottom = 1e-7)
plt.legend()


plt.show()
