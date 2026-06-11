import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Circle
from matplotlib import cm
import matplotlib
from numpy.fft import fft2, fftshift
from functions import *
import time
import timeit
from skimage.transform import resize
from PAPYRIIS_2stage_CNN_RL.OOPAO_PAPYRIIS_env import OOPAO_environment_PAPYRIIS
import torch
from scipy import signal
#TODO check that the pupils are correctly applied innit
#TODO check that the pupils for vzwfs are correct size
#TODO add cred2 to vzwfs
#TODO add gain to OCAM2K and test if it works
#TODO test that the noise is applied on all detectors
#TODO add tqdm progress bars everywhere
#TODO change the old saves into the new saving format
#TODO check again what Mathieu is saving
#TODO everything important (except atmosphere basically) should get shoved into their own folder
#TODO considering the quantisation error, it does not have a case for quantisation of 14 and so you need to code it yourself (or turn it off)
#TODO should the EMCCD with a gain of 1 perform the same as the CCD? (i.e. it should not introduce any noise?)
#TODO sky pupil works no problemo
#TODO don't forget to revert back to the original source
#TODO I do everything on the calibration pupil first
#TODO I left the H source for 1st stage whatever I guess for now
#TODO I don't think you have set an int precision for CRED2
#TODO redo Mathieu's croppping in your own code because you need to crop the atmosphere according to pupils (if atm is 90x90 but pupil is 80, then you should cut off the 10 extra pixels before projecting to different stages)
#TODO maybe write a single file with the important metrics saved for each real observation?
#TODO there might be a problem with 2nd stage projector (check the modes)
#TODO redo the mode calculation for first stage using M2C instead of projector

PAPYRIIS_env = OOPAO_environment_PAPYRIIS()
#savedir_test = "PAPYRIIS_2stage_CNN_RL/~2026-06-01/PAPYRIIS_arcturus_noise"
#----------------------------------------------------Generate 2nd atmosphere#----------------------------------------------------
#TODO don't forget to change back to your favourite atmosphere
#"C:\Users\evinerskas\PycharmProjects\LAM_OOPAO\PAPYRIIS_2stage_CNN_RL\generated_atm_2nd_stage\parameter_testing"
#print(PAPYRIIS_env.atm_2nd.r0)
#print(PAPYRIIS_env.atm_2nd.V0)

def welch_method_scipy(data, fs, nperseg=512):
    frequencies, psd = signal.welch(
        data,
        fs=fs,
        window='hann',  #windowing
        nperseg=nperseg,
        scaling='density'
    )
    return frequencies, psd


from OOPAO.DeformableMirror import DeformableMirror, MisRegistration
from OOPAO.tools.interpolateGeometricalTransformation import interpolate_cube
from PAPYRIIS_2stage_CNN_RL.parameterFile_papyriis import initializeParameterFile
param = initializeParameterFile()
T152onDM_size       = 35.5 # mm
PapyrusOnDM_size    = 37.5 # mm 
ratio_sky_calib = T152onDM_size/PapyrusOnDM_size
from OOPAO.Telescope import Telescope
tel = Telescope(resolution    = int(90),
                    diameter            = param['diameter']/ratio_sky_calib,
                    samplingTime        = param['samplingTime'],
                    centralObstruction  = 0,
                    fov                 = 0)

# mis-registrations object
misReg          = MisRegistration(param)
pitch           = 2.5 #mm
DM_diag_size    = param['nActuator'] * pitch #mm
scale_T152DM = DM_diag_size / T152onDM_size
D_T152 = 1.52

x = np.linspace(-scale_T152DM * D_T152/2, scale_T152DM * D_T152/2, param['nActuator'])
[X,Y] = np.meshgrid(x,x)

DM_coordinates = np.asarray([X.reshape(17**2),Y.reshape(17**2)]).T
dist           = np.sqrt(DM_coordinates[:,0]**2 + DM_coordinates[:,1]**2)
DM_coordinates = DM_coordinates[dist <= D_T152/2 + 2.2 *pitch * D_T152 / T152onDM_size, :]
DM_pitch       = pitch * D_T152 / T152onDM_size

# hardcoded for now
alpao_unit     = 30*7591.024876

param['dm_coordinates'] = DM_coordinates
param['pitch']          = DM_pitch

dm_1st=DeformableMirror(telescope    = tel,\
                    nSubap       = 16,\
                    mechCoupling = 0.36,\
                    misReg       = misReg, \
                    coordinates  = DM_coordinates,\
                    pitch        = DM_pitch,\
                    modes        = None,
                    flip_lr      = True,
                    sign         = -1/alpao_unit)

#our atmosphere
#you can also do this from CL1OL2 I guess, in fact you can do this then from any telemetry file
#you cannot use FULL_OL for total dm shape mode extraction since no commands were actually applied to the deformable mirror
Full_OL = np.load(f'bench_sky_04_15/onsky_arcturus_1st200_2nd400_v7_20260416-011431/2026-04-16T01_28_58_telemetry_data_OL.npy', allow_pickle = True)
CL1OL2  = np.load(f'bench_sky_04_15/onsky_arcturus_1st200_2nd400_v7_20260416-011431/2026-04-16T01_19_12_telemetry_data_RLiter50.npy', allow_pickle = True)
print(Full_OL.item().keys()) #'dmCmdCube' should be the total dm commands #modeCube should be wfs measurements in mode space


atm_simulated_2nd = np.load("PAPYRIIS_2stage_CNN_RL/generated_atm_2nd_stage/atm_OPDs_2nd_r0_0.050_V0_4.000_L0_0.100_single_layer.npz")
atm_simulated_2nd_2 = np.load("PAPYRIIS_2stage_CNN_RL/generated_atm_2nd_stage/atm_OPDs_2nd_r0_0.050_V0_10.000_L0_0.100_single_layer.npz")
atm_simulated_2nd_3 = np.load("PAPYRIIS_2stage_CNN_RL/generated_atm_2nd_stage/atm_OPDs_2nd_r0_0.050_V0_20.000_L0_0.100_single_layer.npz")
atm_simulated_2nd_4 = np.load("PAPYRIIS_2stage_CNN_RL/generated_atm_2nd_stage/atm_OPDs_2nd_r0_0.050_V0_50.000_L0_0.100_single_layer.npz")
atm_simulated_1st = np.load("PAPYRIIS_2stage_CNN_RL/projected_atm_1st_stage/atm_OPDs_1st_r0_0.050_V0_4.121.npz")
print(atm_simulated_2nd.files)
print(atm_simulated_2nd['L0'])
print(atm_simulated_2nd_2['L0'])
print(atm_simulated_2nd_3['L0'])
print(atm_simulated_2nd_4['L0'])
print(atm_simulated_2nd['windSpeed'])


atm_OPDs_2nd = atm_simulated_2nd['atm_OPDs_2nd'][:5000]
atm_OPDs_2nd_2 = atm_simulated_2nd_2['atm_OPDs_2nd'][:5000]
atm_OPDs_2nd_3 = atm_simulated_2nd_3['atm_OPDs_2nd'][:5000]
atm_OPDs_2nd_4 = atm_simulated_2nd_4['atm_OPDs_2nd'][:5000]


atm_OPDs_1st = atm_simulated_1st['atm_OPDs_1st'][:5000]

results_2nd_stage_RL= np.load("PAPYRIIS_2stage_CNN_RL/~2026-06-01\PAPYRIIS_arcturus_noise/results_2nd_stage.npz")
pupil_mask_2nd      = results_2nd_stage_RL["telescope_pupil"].astype(bool)
projector_kl_2nd    = results_2nd_stage_RL["projector_kl_2nd"].reshape(-1, 90, 90)[:, pupil_mask_2nd]
atm_OPDs_2nd        = atm_OPDs_2nd[:, pupil_mask_2nd]


atm_OPDs_2nd_2        = atm_OPDs_2nd_2[:, pupil_mask_2nd]
atm_OPDs_2nd_3        = atm_OPDs_2nd_3[:, pupil_mask_2nd]
atm_OPDs_2nd_4        = atm_OPDs_2nd_4[:, pupil_mask_2nd]



results_1st_stage   = np.load("PAPYRIIS_2stage_CNN_RL/~2026-06-01/PAPYRIIS_arcturus_noise/results_1st_stage_r0_0.050_V0_4.121.npz")
pupil_mask_1st      = results_1st_stage['telescope_pupil'].astype(bool)
projector_kl_1st    = results_1st_stage['projector_kl_1st'].reshape(-1, 80, 80)[:, pupil_mask_1st]
atm_OPDs_1st        = atm_OPDs_1st[:, pupil_mask_1st]




dmCmdCube = CL1OL2.item()['dmCmdCube'].squeeze() #timeseries of dm total shape

modeCube = Full_OL.item()['modeCube'].squeeze()
M2C_1st = Full_OL.item()['m2c'].squeeze()
C2M_1st = np.linalg.pinv(M2C_1st)
dmCmdCube_modes = dmCmdCube @ C2M_1st.T
print(dmCmdCube_modes.shape)
print(np.ptp(dmCmdCube, axis = 1))
print(np.ptp(dmCmdCube, axis = 1).shape)
print(np.ptp(dmCmdCube_modes, axis = 1))
print(np.ptp(dmCmdCube_modes, axis = 1).shape)

#from OPD to dm coefficients
dm_1st_modes = dm_1st.modes.reshape(90, 90, 241)
dm_1st_modes = dm_1st_modes[pupil_mask_2nd, :]
dm_1st_modes_inv = np.linalg.pinv(dm_1st_modes) #opd to control


atm_simulated_2nd_modes = atm_OPDs_2nd @ dm_1st_modes_inv.T @ C2M_1st.T

atm_simulated_2nd_modes_2 = atm_OPDs_2nd_2 @ dm_1st_modes_inv.T @ C2M_1st.T
atm_simulated_2nd_modes_3 = atm_OPDs_2nd_3 @ dm_1st_modes_inv.T @ C2M_1st.T
atm_simulated_2nd_modes_4 = atm_OPDs_2nd_4 @ dm_1st_modes_inv.T @ C2M_1st.T


atm_simulated_1st_modes = atm_OPDs_2nd @ projector_kl_2nd.T
#atm_simulated_1st_modes = atm_OPDs_1st @ projector_kl_1st.T


dmCmdCube_modes_1_curve  = atm_simulated_2nd_modes_2[:, 10]
dmCmdCube_modes_2_curve  = atm_simulated_2nd_modes_2[:, 20]
dmCmdCube_modes_3_curve  = atm_simulated_2nd_modes_2[:, 30]
dmCmdCube_modes_4_curve  = atm_simulated_2nd_modes_2[:, 40]


modeCube_1_curve  = atm_simulated_2nd_modes_3[:, 10]
modeCube_2_curve  = atm_simulated_2nd_modes_3[:, 20]
modeCube_3_curve  = atm_simulated_2nd_modes_3[:, 30]
modeCube_4_curve  = atm_simulated_2nd_modes_3[:, 40]


atm_simulated_2nd_modes_1_curve  = atm_simulated_2nd_modes[:, 10]
atm_simulated_2nd_modes_2_curve  = atm_simulated_2nd_modes[:, 20]
atm_simulated_2nd_modes_3_curve  = atm_simulated_2nd_modes[:, 30]
atm_simulated_2nd_modes_4_curve  = atm_simulated_2nd_modes[:, 40]


atm_simulated_1st_modes_1_curve  = atm_simulated_2nd_modes_4[:, 10]
atm_simulated_1st_modes_2_curve  = atm_simulated_2nd_modes_4[:, 20]
atm_simulated_1st_modes_3_curve  = atm_simulated_2nd_modes_4[:, 30]
atm_simulated_1st_modes_4_curve  = atm_simulated_2nd_modes_4[:, 40]



dmCmdCube_modes_1_psd_t, dmCmdCube_modes_1_psd = welch_method_scipy(dmCmdCube_modes_1_curve, 400)
dmCmdCube_modes_2_psd_t, dmCmdCube_modes_2_psd = welch_method_scipy(dmCmdCube_modes_2_curve, 400)
dmCmdCube_modes_3_psd_t, dmCmdCube_modes_3_psd = welch_method_scipy(dmCmdCube_modes_3_curve, 400)
dmCmdCube_modes_4_psd_t, dmCmdCube_modes_4_psd = welch_method_scipy(dmCmdCube_modes_4_curve, 400)


modeCube_1_psd_t, modeCube_1_psd = welch_method_scipy(modeCube_1_curve, 400)
modeCube_2_psd_t, modeCube_2_psd = welch_method_scipy(modeCube_2_curve, 400)
modeCube_3_psd_t, modeCube_3_psd = welch_method_scipy(modeCube_3_curve, 400)
modeCube_4_psd_t, modeCube_4_psd = welch_method_scipy(modeCube_4_curve, 400)


atm_simulated_2nd_modes_1_psd_t, atm_simulated_2nd_modes_1_psd = welch_method_scipy(atm_simulated_2nd_modes_1_curve, 400)
atm_simulated_2nd_modes_2_psd_t, atm_simulated_2nd_modes_2_psd = welch_method_scipy(atm_simulated_2nd_modes_2_curve, 400)
atm_simulated_2nd_modes_3_psd_t, atm_simulated_2nd_modes_3_psd = welch_method_scipy(atm_simulated_2nd_modes_3_curve, 400)
atm_simulated_2nd_modes_4_psd_t, atm_simulated_2nd_modes_4_psd = welch_method_scipy(atm_simulated_2nd_modes_4_curve, 400)


atm_simulated_1st_modes_1_psd_t, atm_simulated_1st_modes_1_psd = welch_method_scipy(atm_simulated_1st_modes_1_curve, 400)
atm_simulated_1st_modes_2_psd_t, atm_simulated_1st_modes_2_psd = welch_method_scipy(atm_simulated_1st_modes_2_curve, 400)
atm_simulated_1st_modes_3_psd_t, atm_simulated_1st_modes_3_psd = welch_method_scipy(atm_simulated_1st_modes_3_curve, 400)
atm_simulated_1st_modes_4_psd_t, atm_simulated_1st_modes_4_psd = welch_method_scipy(atm_simulated_1st_modes_4_curve, 400)



fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)


ax1.plot(dmCmdCube_modes_1_psd_t, dmCmdCube_modes_1_psd, label = "V0 = 10")#dmCmdCube
ax1.plot(modeCube_1_psd_t, modeCube_1_psd, label = "V0 = 20")#modeCube
ax1.plot(atm_simulated_2nd_modes_1_psd_t, atm_simulated_2nd_modes_1_psd, label = "V0 = 4")#atm_simulated_2nd_projected
ax1.plot(atm_simulated_1st_modes_1_psd_t, atm_simulated_1st_modes_1_psd, label = "V0 = 50")#atm_simulated_2nd
ax1.set_title("mode 10")
ax1.set_xlabel("frequency (Hz)")
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.grid(True, which='both', alpha=0.5)
ax1.minorticks_on()
ax1.legend()


ax2.plot(dmCmdCube_modes_2_psd_t, dmCmdCube_modes_2_psd, label = "dmCmdCube")
ax2.plot(modeCube_2_psd_t, modeCube_2_psd, label = "modeCube")
ax2.plot(atm_simulated_2nd_modes_2_psd_t, atm_simulated_2nd_modes_2_psd, label = "atm_simulated_2nd_projected")
ax2.plot(atm_simulated_1st_modes_2_psd_t, atm_simulated_1st_modes_2_psd, label = "atm_simulated_2nd")
ax2.set_title("mode 20")
ax2.set_xlabel("frequency (Hz)")
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.grid(True, which='both', alpha=0.5)
ax2.minorticks_on()

ax3.plot(dmCmdCube_modes_3_psd_t, dmCmdCube_modes_3_psd, label = "dmCmdCube")
ax3.plot(modeCube_3_psd_t, modeCube_3_psd, label = "modeCube")
ax3.plot(atm_simulated_2nd_modes_3_psd_t, atm_simulated_2nd_modes_3_psd, label = "atm_simulated_2nd_projected")
ax3.plot(atm_simulated_1st_modes_3_psd_t, atm_simulated_1st_modes_3_psd, label = "atm_simulated_2nd")
ax3.set_title("mode 30")
ax3.set_xlabel("frequency (Hz)")
ax3.set_yscale("log")
ax3.set_xscale("log")
ax3.grid(True, which='both', alpha=0.5)
ax3.minorticks_on()


ax4.plot(dmCmdCube_modes_4_psd_t, dmCmdCube_modes_4_psd)
ax4.plot(modeCube_4_psd_t, modeCube_4_psd)
ax4.plot(atm_simulated_2nd_modes_4_psd_t, atm_simulated_2nd_modes_4_psd)
ax4.plot(atm_simulated_1st_modes_4_psd_t, atm_simulated_1st_modes_4_psd)
ax4.set_title("mode 40")
ax4.set_xlabel("frequency (Hz)")
ax4.set_yscale("log")
ax4.set_xscale("log")
ax4.grid(True, which='both', alpha=0.5)
ax4.minorticks_on()


plt.show()




errr

atm_OPDs_2nd = PAPYRIIS_env.generate_second_stage_atmosphere(nLoop=5000)


np.savez(
    f"PAPYRIIS_2stage_CNN_RL/generated_atm_2nd_stage/atm_OPDs_2nd_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_single_layer.npz",
    atm_OPDs_2nd=atm_OPDs_2nd,
    r0=PAPYRIIS_env.atm_2nd.r0,
    L0=PAPYRIIS_env.atm_2nd.L0,
    windSpeed=PAPYRIIS_env.atm_2nd.windSpeed,
    fractionalR0=PAPYRIIS_env.atm_2nd.fractionalR0,
    windDirection=PAPYRIIS_env.atm_2nd.windDirection,
    altitude=PAPYRIIS_env.atm_2nd.altitude
)
errr
#----------------------------------------------------Project atmosphere to 1st stage#----------------------------------------------------
"""
atm_OPDs_2nd = np.load(f"PAPYRIIS_2stage_CNN_RL/generated_atm_2nd_stage/atm_OPDs_2nd_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}.npz")
atm_OPDs_2nd = atm_OPDs_2nd["atm_OPDs_2nd"]
print(f"atm loaded: {PAPYRIIS_env.atm_2nd.r0:.3f}, {PAPYRIIS_env.atm_2nd.V0:.3f}")
print(atm_OPDs_2nd.shape)

atm_OPDs_1st = PAPYRIIS_env.project_atmosphere_to_first_stage(atm_OPDs_2nd[:128000, :, :])

np.savez(f"PAPYRIIS_2stage_CNN_RL/projected_atm_1st_stage/atm_OPDs_1st_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}.npz",
        atm_OPDs_1st=atm_OPDs_1st,
        r0=PAPYRIIS_env.atm_2nd.r0,
        L0=PAPYRIIS_env.atm_2nd.L0,
        windSpeed=PAPYRIIS_env.atm_2nd.windSpeed,
        fractionalR0=PAPYRIIS_env.atm_2nd.fractionalR0,
        windDirection=PAPYRIIS_env.atm_2nd.windDirection,
        altitude=PAPYRIIS_env.atm_2nd.altitude
)
"""
#----------------------------------------------------1st stage CL#----------------------------------------------------
'''
atm_OPD_1st = np.load(f"PAPYRIIS_2stage_CNN_RL/projected_atm_1st_stage/atm_OPDs_1st_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}.npz")
atm_OPD_1st = atm_OPD_1st["atm_OPDs_1st"]
print(f"atm loaded: {PAPYRIIS_env.atm_2nd.r0:.3f}, {PAPYRIIS_env.atm_2nd.V0:.3f}")

first_stage_results = PAPYRIIS_env.run_first_stage_loop(120, atm_OPD_1st)
np.savez(f"{savedir_test}/results_1st_stage_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}.npz", **{
    k: v for k, v in first_stage_results.items() 
    if k != "config"
},
    # Config fields flattened
    nLoop=first_stage_results["config"].nLoop,
    gainCL=first_stage_results["config"].gainCL,
    leak=first_stage_results["config"].leak,
    frame_delay=first_stage_results["config"].frame_delay,
    photon_noise=first_stage_results["config"].photon_noise,
)
'''
#----------------------------------------------------1st stage residual entering to 2nd#----------------------------------------------------
"""
#PAPYRIIS_2stage_CNN_RL\~2026-06-01\PAPYRIIS_arcturus_nonoise\results_2nd_stage.npz
loaddir_test = "PAPYRIIS_2stage_CNN_RL/~2026-06-01/PAPYRIIS_arcturus_nonoise"
loaddir_test = savedir_test
first_stage_results = np.load(f"{loaddir_test}/results_1st_stage_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}.npz")
residuals_opds_1rst = first_stage_results['residuals_opds_1rst']
print(first_stage_results.files)
print(f"{loaddir_test}/results_1st_stage_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}.npz")

#results_2nd_stage = np.load(f"{loaddir_test}/results_2nd_stage_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}.npz")
#print(results_2nd_stage.files)

#all_src_opd = results_2nd_stage["all_src_opd"]

dm_commands = np.zeros((30000, 97))
reconstructed_cmd = np.zeros((30000, 97))
scnd_stage_strehl = np.zeros((30000))
tel_2nd_pupil = 0
src_opd = np.zeros((30000, 90, 90))
projector_kl_2nd = np.zeros((87, 8100))


obs, _ = PAPYRIIS_env.reset(residuals_opds_1rst)

for t in range(30000):
    action = 0.3 * obs.unsqueeze(0).unsqueeze(0)
    next_obs, INFO = PAPYRIIS_env.step(action.squeeze(), residuals_opds_1rst)
    print(INFO['2nd_stage_strehl'])
    dm_commands[t] =INFO["dm_commands"]
    reconstructed_cmd[t] =INFO["reconstructed_cmd"].detach().cpu().numpy()
    scnd_stage_strehl[t] =INFO["2nd_stage_strehl"]
    src_opd[t] =INFO["src_opd"]
    tel_2nd_pupil = INFO["telescope_pupil"]
    projector_kl_2nd = INFO["projector_kl_2nd"]

    obs = next_obs


np.savez(f"{savedir_test}/results_2nd_stage_CL1OL2.npz",
    # Concatenated across iterations
    all_2nd_stage_strehl    = scnd_stage_strehl,
    all_dm_commands         = dm_commands,
    all_reconstructed_cmd   = reconstructed_cmd,
    residual_opds_2nd       = src_opd,
    telescope_pupil         = tel_2nd_pupil,
    projector_kl_2nd        = projector_kl_2nd,
    )
"""