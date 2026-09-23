"""
In this code you can compare the different PSDs from the deformable mirror and from OPD_screens (so no filtering effect)

"""



import numpy as np
from functions import *
import matplotlib.pyplot as plt
from OOPAO.DeformableMirror import DeformableMirror, MisRegistration
from OOPAO.tools.interpolateGeometricalTransformation import interpolate_cube
from PAPYRIIS_2stage_CNN_RL.parameterFile_papyriis import initializeParameterFile
param = initializeParameterFile()
T152onDM_size       = 35.5 # mm
PapyrusOnDM_size    = 37.5 # mm 
ratio_sky_calib = T152onDM_size/PapyrusOnDM_size
from OOPAO.Telescope import Telescope
# tel = Telescope(resolution    = int(90),
#                     diameter            = param['diameter']/ratio_sky_calib,
#                     samplingTime        = param['samplingTime'],
#                     centralObstruction  = 0,
#                     fov                 = 0)

# # mis-registrations object
# misReg          = MisRegistration(param)
# pitch           = 2.5 #mm
# DM_diag_size    = param['nActuator'] * pitch #mm
# scale_T152DM = DM_diag_size / T152onDM_size
# D_T152 = 1.52

# x = np.linspace(-scale_T152DM * D_T152/2, scale_T152DM * D_T152/2, param['nActuator'])
# [X,Y] = np.meshgrid(x,x)

# DM_coordinates = np.asarray([X.reshape(17**2),Y.reshape(17**2)]).T
# dist           = np.sqrt(DM_coordinates[:,0]**2 + DM_coordinates[:,1]**2)
# DM_coordinates = DM_coordinates[dist <= D_T152/2 + 2.2 *pitch * D_T152 / T152onDM_size, :]
# DM_pitch       = pitch * D_T152 / T152onDM_size

# # hardcoded for now
# alpao_unit     = 30*7591.024876

# param['dm_coordinates'] = DM_coordinates
# param['pitch']          = DM_pitch

# dm_1st=DeformableMirror(telescope    = tel,\
#                     nSubap       = 16,\
#                     mechCoupling = 0.36,\
#                     misReg       = misReg, \
#                     coordinates  = DM_coordinates,\
#                     pitch        = DM_pitch,\
#                     modes        = None,
#                     flip_lr      = True,
#                     sign         = -1/alpao_unit)



#our atmosphere
#you can also do this from CL1OL2 I guess, in fact you can do this then from any telemetry file
CL1OL2  = np.load(f'bench_sky_04_15/onsky_arcturus_1st200_2nd400_v7_20260416-011431/2026-04-16T01_19_12_telemetry_data_RLiter50.npy', allow_pickle = True)

print(CL1OL2.item().keys()) #'dmCmdCube' should be the total dm commands #modeCube should be wfs measurements in mode space


#dm_command_loader
loaddir = "PAPYRIIS_2stage_CNN_RL/~2026-06-23/PAPYRIIS_arcturus_noise_quantisation_pwfs_calibration_pupil_EMCCD"
dm_coefs_file_1 = np.load(f"{loaddir}/results_1st_stage_r0_0.050_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz")
dm_coefs_file_2 = np.load(f"{loaddir}/results_1st_stage_r0_0.050_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz")
dm_coefs_file_3 = np.load(f"{loaddir}/results_1st_stage_r0_0.050_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz")
dm_coefs_file_4 = np.load(f"{loaddir}/results_1st_stage_r0_0.050_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz")


#phase screen loader
# OPD_screen_file_1 = np.load("PAPYRIIS_2stage_CNN_RL/generated_atm_2nd_stage/atm_OPDs_2nd_r0_0.050_V0_4.121_L0_30.000_tboil_2.000_single_layer.npz")
# OPD_screen_file_2 = np.load("PAPYRIIS_2stage_CNN_RL/generated_atm_2nd_stage/atm_OPDs_2nd_r0_0.050_V0_4.049_L0_30.000_tboil_2.000_multi_layer.npz")
# OPD_screen_file_3 = np.load("PAPYRIIS_2stage_CNN_RL/generated_atm_2nd_stage/atm_OPDs_2nd_r0_0.050_V0_4.121_L0_30.000_tboil_2.000_single_layer.npz")
# OPD_screen_file_4 = np.load("PAPYRIIS_2stage_CNN_RL/generated_atm_2nd_stage/atm_OPDs_2nd_r0_0.050_V0_4.121_L0_30.000_tboil_2000.000_single_layer.npz")
# print(OPD_screen_file_1.files)
label_1 = 'atm_sky'
label_2 = 'atm_sim_tboil_5.000_multi_layer_gain04'
label_3 = 'atm_sim_tboil_5.000_multi_layer_gain04'
label_4 = 'atm_sim_tboil_5.000_multi_layer_gain04'
alpha = 0.7

# OPD_screen_1 = OPD_screen_file_1['atm_OPDs_2nd'][:5000]
# OPD_screen_2 = OPD_screen_file_2['atm_OPDs_2nd'][:5000]
# OPD_screen_3 = OPD_screen_file_3['atm_OPDs_2nd'][:5000]
# OPD_screen_4 = OPD_screen_file_4['atm_OPDs_2nd'][:5000]


results_2nd_stage_RL= np.load("PAPYRIIS_2stage_CNN_RL/~2026-06-01\PAPYRIIS_arcturus_noise/results_2nd_stage.npz")
pupil_mask      = results_2nd_stage_RL["telescope_pupil"].astype(bool)
projector_kl_2nd    = results_2nd_stage_RL["projector_kl_2nd"].reshape(-1, 90, 90)[:, pupil_mask]


results_1st_stage   = np.load("PAPYRIIS_2stage_CNN_RL/~2026-06-01/PAPYRIIS_arcturus_noise/results_1st_stage_r0_0.050_V0_4.121.npz")
pupil_mask_1st      = results_1st_stage['telescope_pupil'].astype(bool)
projector_kl_1st    = results_1st_stage['projector_kl_1st'].reshape(-1, 80, 80)[:, pupil_mask_1st]



# OPD_screen_1        = OPD_screen_1[:, pupil_mask_2nd]
# OPD_screen_2        = OPD_screen_2[:, pupil_mask_2nd]
# OPD_screen_3        = OPD_screen_3[:, pupil_mask_2nd]
# OPD_screen_4        = OPD_screen_4[:, pupil_mask_2nd]


M2C_1st = - np.load("PAPYRIIS_2stage_CNN_RL/M2C_1rst.npy")


#from OPD to dm coefficients
dm_1st_modes, pupil_mask = first_stage_dm_builder()
dm_1st_modes = dm_1st_modes.reshape(100, 100, 241)
dm_1st_modes_masked = dm_1st_modes[pupil_mask, :]


dmCmdCube_modes_1 = mode_calculator_fromDM(CL1OL2.item()['dmCmdCube'].squeeze(), M2C_1st)#(CL1OL2.item()['dmCmdCube'].squeeze(), M2C_1st)
dmCmdCube_modes_2 = mode_calculator_fromDM(dm_coefs_file_1["dm_commands"], M2C_1st)
dmCmdCube_modes_3 = mode_calculator_fromDM(dm_coefs_file_2["dm_commands"], M2C_1st)
dmCmdCube_modes_4 = mode_calculator_fromDM(dm_coefs_file_3["dm_commands"], M2C_1st)


OPD_screen_modes_1 = dmCmdCube_modes_1#dmCmdCube_modes_1#mode_calculator_fromOPD(OPD_screen_1, M2C_1st, dm_1st_modes_masked)
OPD_screen_modes_2 = dmCmdCube_modes_2#dmCmdCube_modes_2#mode_calculator_fromOPD(OPD_screen_2, M2C_1st, dm_1st_modes_masked)
OPD_screen_modes_3 = dmCmdCube_modes_3#dmCmdCube_modes_3#mode_calculator_fromOPD(OPD_screen_3, M2C_1st, dm_1st_modes_masked)
OPD_screen_modes_4 = dmCmdCube_modes_4#dmCmdCube_modes_4#mode_calculator_fromOPD(OPD_screen_4, M2C_1st, dm_1st_modes_masked)


G_diag = np.diag(mode_covariance(dm_1st_modes_masked * 1e9 @ M2C_1st))

modes_psd_f_1_10, modes_psd_1_10, var_from_PSD_1_10 = tPSD_calculator(OPD_screen_modes_1, 10, G_diag, 200)
modes_psd_f_1_20, modes_psd_1_20, var_from_PSD_1_20 = tPSD_calculator(OPD_screen_modes_1, 20, G_diag, 200)
modes_psd_f_1_30, modes_psd_1_30, var_from_PSD_1_30 = tPSD_calculator(OPD_screen_modes_1, 30, G_diag, 200)
modes_psd_f_1_40, modes_psd_1_40, var_from_PSD_1_40 = tPSD_calculator(OPD_screen_modes_1, 40, G_diag, 200)


modes_psd_f_2_10, modes_psd_2_10, var_from_PSD_2_10 = tPSD_calculator(OPD_screen_modes_2, 10, G_diag, 400)
modes_psd_f_2_20, modes_psd_2_20, var_from_PSD_2_20 = tPSD_calculator(OPD_screen_modes_2, 20, G_diag, 400)
modes_psd_f_2_30, modes_psd_2_30, var_from_PSD_2_30 = tPSD_calculator(OPD_screen_modes_2, 30, G_diag, 400)
modes_psd_f_2_40, modes_psd_2_40, var_from_PSD_2_40 = tPSD_calculator(OPD_screen_modes_2, 40, G_diag, 400)

modes_psd_f_3_10, modes_psd_3_10, var_from_PSD_3_10 = tPSD_calculator(OPD_screen_modes_3, 10, G_diag, 400)
modes_psd_f_3_20, modes_psd_3_20, var_from_PSD_3_20 = tPSD_calculator(OPD_screen_modes_3, 20, G_diag, 400)
modes_psd_f_3_30, modes_psd_3_30, var_from_PSD_3_30 = tPSD_calculator(OPD_screen_modes_3, 30, G_diag, 400)
modes_psd_f_3_40, modes_psd_3_40, var_from_PSD_3_40 = tPSD_calculator(OPD_screen_modes_3, 40, G_diag, 400)

modes_psd_f_4_10, modes_psd_4_10, var_from_PSD_4_10 = tPSD_calculator(OPD_screen_modes_4, 10, G_diag, 400)
modes_psd_f_4_20, modes_psd_4_20, var_from_PSD_4_20 = tPSD_calculator(OPD_screen_modes_4, 20, G_diag, 400)
modes_psd_f_4_30, modes_psd_4_30, var_from_PSD_4_30 = tPSD_calculator(OPD_screen_modes_4, 30, G_diag, 400)
modes_psd_f_4_40, modes_psd_4_40, var_from_PSD_4_40 = tPSD_calculator(OPD_screen_modes_4, 40, G_diag, 400)


#FITTING
modes_psd_2_10_fit_results = fit_psd(modes_psd_f_2_10, modes_psd_2_10, model="kolmogorov", freq_range=(10, 100))
modes_psd_2_20_fit_results = fit_psd(modes_psd_f_2_20, modes_psd_2_20, model="kolmogorov", freq_range=(10, 100))
modes_psd_2_30_fit_results = fit_psd(modes_psd_f_2_30, modes_psd_2_30, model="kolmogorov", freq_range=(10, 100))
modes_psd_2_40_fit_results = fit_psd(modes_psd_f_2_40, modes_psd_2_40, model="kolmogorov", freq_range=(10, 100))
modes_psd_2_10_fit = psd_kolmogorov(modes_psd_f_2_10, modes_psd_2_10_fit_results.A, modes_psd_2_10_fit_results.alpha)
modes_psd_2_20_fit = psd_kolmogorov(modes_psd_f_2_20, modes_psd_2_20_fit_results.A, modes_psd_2_20_fit_results.alpha)
modes_psd_2_30_fit = psd_kolmogorov(modes_psd_f_2_30, modes_psd_2_30_fit_results.A, modes_psd_2_30_fit_results.alpha)
modes_psd_2_40_fit = psd_kolmogorov(modes_psd_f_2_40, modes_psd_2_40_fit_results.A, modes_psd_2_40_fit_results.alpha)



modes_psd_3_10_fit_results = fit_psd(modes_psd_f_3_10, modes_psd_3_10, model="kolmogorov", freq_range=(10, 100))
modes_psd_3_20_fit_results = fit_psd(modes_psd_f_3_20, modes_psd_3_20, model="kolmogorov", freq_range=(10, 100))
modes_psd_3_30_fit_results = fit_psd(modes_psd_f_3_30, modes_psd_3_30, model="kolmogorov", freq_range=(10, 100))
modes_psd_3_40_fit_results = fit_psd(modes_psd_f_3_40, modes_psd_3_40, model="kolmogorov", freq_range=(10, 100))
modes_psd_3_10_fit = psd_kolmogorov(modes_psd_f_3_10, modes_psd_3_10_fit_results.A, modes_psd_3_10_fit_results.alpha)
modes_psd_3_20_fit = psd_kolmogorov(modes_psd_f_3_20, modes_psd_3_20_fit_results.A, modes_psd_3_20_fit_results.alpha)
modes_psd_3_30_fit = psd_kolmogorov(modes_psd_f_3_30, modes_psd_3_30_fit_results.A, modes_psd_3_30_fit_results.alpha)
modes_psd_3_40_fit = psd_kolmogorov(modes_psd_f_3_40, modes_psd_3_40_fit_results.A, modes_psd_3_40_fit_results.alpha)



modes_psd_1_10_fit_results = fit_psd(modes_psd_f_1_10, modes_psd_1_10, model="kolmogorov", freq_range=(10, 100))
modes_psd_1_20_fit_results = fit_psd(modes_psd_f_1_20, modes_psd_1_20, model="kolmogorov", freq_range=(10, 100))
modes_psd_1_30_fit_results = fit_psd(modes_psd_f_1_30, modes_psd_1_30, model="kolmogorov", freq_range=(10, 100))
modes_psd_1_40_fit_results = fit_psd(modes_psd_f_1_40, modes_psd_1_40, model="kolmogorov", freq_range=(10, 100))
modes_psd_1_10_fit = psd_kolmogorov(modes_psd_f_1_10, modes_psd_1_10_fit_results.A, modes_psd_1_10_fit_results.alpha)
modes_psd_1_20_fit = psd_kolmogorov(modes_psd_f_1_20, modes_psd_1_20_fit_results.A, modes_psd_1_20_fit_results.alpha)
modes_psd_1_30_fit = psd_kolmogorov(modes_psd_f_1_30, modes_psd_1_30_fit_results.A, modes_psd_1_30_fit_results.alpha)
modes_psd_1_40_fit = psd_kolmogorov(modes_psd_f_1_40, modes_psd_1_40_fit_results.A, modes_psd_1_40_fit_results.alpha)




modes_psd_4_10_fit_results = fit_psd(modes_psd_f_4_10, modes_psd_4_10, model="kolmogorov", freq_range=(10, 100))
modes_psd_4_20_fit_results = fit_psd(modes_psd_f_4_20, modes_psd_4_20, model="kolmogorov", freq_range=(10, 100))
modes_psd_4_30_fit_results = fit_psd(modes_psd_f_4_30, modes_psd_4_30, model="kolmogorov", freq_range=(10, 100))
modes_psd_4_40_fit_results = fit_psd(modes_psd_f_4_40, modes_psd_4_40, model="kolmogorov", freq_range=(10, 100))
modes_psd_4_10_fit = psd_kolmogorov(modes_psd_f_4_10, modes_psd_4_10_fit_results.A, modes_psd_4_10_fit_results.alpha)
modes_psd_4_20_fit = psd_kolmogorov(modes_psd_f_4_20, modes_psd_4_20_fit_results.A, modes_psd_4_20_fit_results.alpha)
modes_psd_4_30_fit = psd_kolmogorov(modes_psd_f_4_30, modes_psd_4_30_fit_results.A, modes_psd_4_30_fit_results.alpha)
modes_psd_4_40_fit = psd_kolmogorov(modes_psd_f_4_40, modes_psd_4_40_fit_results.A, modes_psd_4_40_fit_results.alpha)


import matplotlib.pyplot as plt

# Create a 2x2 grid and flatten the axes array to unpack easily
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
ax1, ax2, ax3, ax4 = axs.flatten()

# --- Mode 10 (Top Left) ---
l, = ax1.plot(modes_psd_f_1_10, modes_psd_1_10, alpha=alpha)
ax1.plot(modes_psd_f_1_10, modes_psd_1_10_fit, color=l.get_color(), ls="--", label=f"{label_1}, alpha = {modes_psd_1_10_fit_results.alpha:.2f}+-{modes_psd_1_10_fit_results.alpha_err:.2f}, {var_from_PSD_1_10:.0f}")
l, = ax1.plot(modes_psd_f_2_10, modes_psd_2_10, alpha=alpha)
ax1.plot(modes_psd_f_2_10, modes_psd_2_10_fit, color=l.get_color(), ls="--", label=f"{label_2}, alpha = {modes_psd_2_10_fit_results.alpha:.2f}+-{modes_psd_2_10_fit_results.alpha_err:.2f}, {var_from_PSD_2_10:.0f}")
l, = ax1.plot(modes_psd_f_3_10, modes_psd_3_10, alpha=alpha)
ax1.plot(modes_psd_f_3_10, modes_psd_3_10_fit, color=l.get_color(), ls="--", label=f"{label_3}, alpha = {modes_psd_3_10_fit_results.alpha:.2f}+-{modes_psd_3_10_fit_results.alpha_err:.2f}, {var_from_PSD_3_10:.0f}")
l, = ax1.plot(modes_psd_f_4_10, modes_psd_4_10, alpha=alpha)
ax1.plot(modes_psd_f_4_10, modes_psd_4_10_fit, color=l.get_color(), ls="--", label=f"{label_4}, alpha = {modes_psd_4_10_fit_results.alpha:.2f}+-{modes_psd_4_10_fit_results.alpha_err:.2f}, {var_from_PSD_4_10:.0f}")

ax1.set_title("mode 10")
ax1.set_xlabel("frequency (Hz)")
ax1.set_ylabel(r"nm$^2$/Hz")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.grid(True, which='both', alpha=0.5)
ax1.minorticks_on()
ax1.legend()

# --- Mode 20 (Top Right) ---
l, = ax2.plot(modes_psd_f_1_20, modes_psd_1_20, alpha=alpha)
ax2.plot(modes_psd_f_1_20, modes_psd_1_20_fit, color=l.get_color(), ls="--", label=f"{label_1}, alpha = {modes_psd_1_20_fit_results.alpha:.2f}+-{modes_psd_1_20_fit_results.alpha_err:.2f}, {var_from_PSD_1_20:.0f}")
l, = ax2.plot(modes_psd_f_2_20, modes_psd_2_20, alpha=alpha)
ax2.plot(modes_psd_f_2_20, modes_psd_2_20_fit, color=l.get_color(), ls="--", label=f"{label_2}, alpha = {modes_psd_2_20_fit_results.alpha:.2f}+-{modes_psd_2_20_fit_results.alpha_err:.2f}, {var_from_PSD_2_20:.0f}")
l, = ax2.plot(modes_psd_f_3_20, modes_psd_3_20, alpha=alpha)
ax2.plot(modes_psd_f_3_20, modes_psd_3_20_fit, color=l.get_color(), ls="--", label=f"{label_3}, alpha = {modes_psd_3_20_fit_results.alpha:.2f}+-{modes_psd_3_20_fit_results.alpha_err:.2f}, {var_from_PSD_3_20:.0f}")
l, = ax2.plot(modes_psd_f_4_20, modes_psd_4_20, alpha=alpha)
ax2.plot(modes_psd_f_4_20, modes_psd_4_20_fit, color=l.get_color(), ls="--", label=f"{label_4}, alpha = {modes_psd_4_20_fit_results.alpha:.2f}+-{modes_psd_4_20_fit_results.alpha_err:.2f}, {var_from_PSD_4_20:.0f}")

ax2.set_title("mode 20")
ax2.set_xlabel("frequency (Hz)")
ax2.set_ylabel(r"nm$^2$/Hz")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.grid(True, which='both', alpha=0.5)
ax2.minorticks_on()
ax2.legend()

# --- Mode 30 (Bottom Left) ---
l, = ax3.plot(modes_psd_f_1_30, modes_psd_1_30, alpha=alpha)
ax3.plot(modes_psd_f_1_30, modes_psd_1_30_fit, color=l.get_color(), ls="--", label=f"{label_1}, alpha = {modes_psd_1_30_fit_results.alpha:.2f}+-{modes_psd_1_30_fit_results.alpha_err:.2f}, {var_from_PSD_1_30:.0f}")
l, = ax3.plot(modes_psd_f_2_30, modes_psd_2_30, alpha=alpha)
ax3.plot(modes_psd_f_2_30, modes_psd_2_30_fit, color=l.get_color(), ls="--", label=f"{label_2}, alpha = {modes_psd_2_30_fit_results.alpha:.2f}+-{modes_psd_2_30_fit_results.alpha_err:.2f}, {var_from_PSD_2_30:.0f}")
l, = ax3.plot(modes_psd_f_3_30, modes_psd_3_30, alpha=alpha)
ax3.plot(modes_psd_f_3_30, modes_psd_3_30_fit, color=l.get_color(), ls="--", label=f"{label_3}, alpha = {modes_psd_3_30_fit_results.alpha:.2f}+-{modes_psd_3_30_fit_results.alpha_err:.2f}, {var_from_PSD_3_30:.0f}")
l, = ax3.plot(modes_psd_f_4_30, modes_psd_4_30, alpha=alpha)
ax3.plot(modes_psd_f_4_30, modes_psd_4_30_fit, color=l.get_color(), ls="--", label=f"{label_4}, alpha = {modes_psd_4_30_fit_results.alpha:.2f}+-{modes_psd_4_30_fit_results.alpha_err:.2f}, {var_from_PSD_4_30:.0f}")

ax3.set_title("mode 30")
ax3.set_xlabel("frequency (Hz)")
ax3.set_ylabel(r"nm$^2$/Hz")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.grid(True, which='both', alpha=0.5)
ax3.minorticks_on()
ax3.legend()

# --- Mode 40 (Bottom Right) ---
l, = ax4.plot(modes_psd_f_1_40, modes_psd_1_40, alpha=alpha)
ax4.plot(modes_psd_f_1_40, modes_psd_1_40_fit, color=l.get_color(), ls="--", label=f"{label_1}, alpha = {modes_psd_1_40_fit_results.alpha:.2f}+-{modes_psd_1_40_fit_results.alpha_err:.2f}, {var_from_PSD_1_40:.0f}")
l, = ax4.plot(modes_psd_f_2_40, modes_psd_2_40, alpha=alpha)
ax4.plot(modes_psd_f_2_40, modes_psd_2_40_fit, color=l.get_color(), ls="--", label=f"{label_2}, alpha = {modes_psd_2_40_fit_results.alpha:.2f}+-{modes_psd_2_40_fit_results.alpha_err:.2f}, {var_from_PSD_2_40:.0f}")
l, = ax4.plot(modes_psd_f_3_40, modes_psd_3_40, alpha=alpha)
ax4.plot(modes_psd_f_3_40, modes_psd_3_40_fit, color=l.get_color(), ls="--", label=f"{label_3}, alpha = {modes_psd_3_40_fit_results.alpha:.2f}+-{modes_psd_3_40_fit_results.alpha_err:.2f}, {var_from_PSD_3_40:.0f}")
l, = ax4.plot(modes_psd_f_4_40, modes_psd_4_40, alpha=alpha)
ax4.plot(modes_psd_f_4_40, modes_psd_4_40_fit, color=l.get_color(), ls="--", label=f"{label_4}, alpha = {modes_psd_4_40_fit_results.alpha:.2f}+-{modes_psd_4_40_fit_results.alpha_err:.2f}, {var_from_PSD_4_40:.0f}")

ax4.set_title("mode 40")
ax4.set_xlabel("frequency (Hz)")
ax4.set_ylabel(r"nm$^2$/Hz")
ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.grid(True, which='both', alpha=0.5)
ax4.minorticks_on()
ax4.legend()

# Prevent overlapping labels in the 2x2 layout
plt.tight_layout()
plt.show()



