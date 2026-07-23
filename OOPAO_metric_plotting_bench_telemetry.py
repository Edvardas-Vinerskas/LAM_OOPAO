"""
#a bit of philosophy
#   atm reconstruction from 1st stage DM shape like in test.py (from dm shape)
        you can either do this, or do from slopes but it should be consistent
        Benoit told me to reconstruct the atm from DM shape so that I avoid optical gains but I am not sure if that helps
        I think the main difference should be coming from the slightly different transfer functions
        I also get optical gains for the 1st stage, but the 2nd stage CNN reconstruction definitely has no optical gains
        there should be papers talking about atm reconstruction
#   1st stage residual reconstruction from 2nd stage CNN reconstruction in DM space innit (CL1OL2)
#   2nd stage residual reconstruction from CNN reconstruction in DM space innit
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from functions import *

#TODO check the timeseries length of onsky and simulation data (make them the same for aesthetics)
RL                   = True
integrator           = True
ideal                = False
atm_RL               = True
atm_int              = False
atm_ideal            = False
plot_timeseries      = False
plot_tPSD            = True
plot_onsky = True   # plot on-sky results (solid/dashed lines)
plot_sim   = True   # plot simulation results (dotted lines)
#TODO for the sim mismatch, you should analyse
# influence functions
# maybe you should apply different limits for the timeseries being analysed
CL_gain_pyr = 0.3
frequency_1st = 400 #should be the same as 2nd if you are using CL1OL2 with 2nd stage looking at the 1st stage residuals
frequency_2nd = 400
frequency_atm = 200
label_RL = "RL"
label_int = "int"
label_ideal = "ideal"
label_linear = "linear"
label_RL_sim  = "RL (sim)"
label_int_sim = "int (sim)"

KL_frames_RL    = [1, -1]   # [start, end] indices for RL timeseries window
KL_frames_int   = [1, -1]   # [start, end] indices for integrator timeseries window
KL_frames_ideal = [1, -1]   # [start, end] indices for ideal timeseries window
#modes for PSDs
psd_modes = [0, 10, 20, 30, 40]   # KL mode indices to plot PSDs for
mode_1, mode_2, mode_3, mode_4, mode_5 = psd_modes

# on sky data
loaddir = 'bench_sky_04_15/onsky_arcturus_1st200_2nd400_v10_20260416-020659'
loaddirlinear = "bench_sky_04_16/onsky_arcturus_1st200_2nd400_v9_linear_20260417-042610"
linear_telemetry_2nd    = np.load(f'{loaddirlinear}/2026-04-17T04_35_20_telemetry_2nd_data_pythint.npy', allow_pickle = True) #label integrator
RL_telemetry_2nd    = np.load(f'{loaddir}/2026-04-16T02_11_05_telemetry_2nd_data_RLiter70.npy', allow_pickle = True) #label CNN + PO4AO
RL_telemetry_1st    = np.load(f'{loaddir}/2026-04-16T02_10_40_telemetry_data_RLiter70.npy', allow_pickle = True)
# RL_telemetry_1st    = np.load(f'{loaddirlinear}/2026-04-17T04_18_08_telemetry_data_pythint.npy', allow_pickle = True)
# {loaddirlinear}/2026-04-17T04_50_14_telemetry_2nd_data_pythint
# {loaddir}/2026-04-16T02_14_21_telemetry_2nd_data_pythint
int_telemetry_2nd   = np.load(f'{loaddir}/2026-04-16T02_14_21_telemetry_2nd_data_pythint.npy', allow_pickle = True) #label CNN
CL1st_OL_2nd        = np.load(f'{loaddir}/2026-04-16T02_17_55_telemetry_2nd_data_CL1OL2.npy', allow_pickle = True)


loaddir_sim = "PAPYRIIS_2stage_CNN_RL/~2026-06-23/PAPYRIIS_arcturus_noise_quantisation_pwfs_calibration_pupil_EMCCD"
RL_telemetry_2nd_sim    = np.load(f"{loaddir_sim}/results_2nd_stage.npz")
RL_telemetry_1st_sim    = np.load(f"{loaddir_sim}/results_1st_stage_r0_0.050_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz")
int_telemetry_2nd_sim   = np.load(f"{loaddir_sim}/results_2nd_stage_int.npz")
CL1st_OL_2nd_sim        = np.load(f"{loaddir_sim}/results_2nd_stage_CL1OL2.npz")



#TODO the simulation results are not the same
#you should maybe use different influence functions for simulations and on sky?
#TODO these don't have the representative masks (masking done as per test.py or just look inside the 2 below functions)
dm_1st_inf, pupil_mask_1 = first_stage_dm_builder()
dm_2nd_inf, pupil_mask_2 = second_stage_dm_builder()


dm_1st_inf = dm_1st_inf.reshape(100, 100, 241)
dm_1st_inf_masked = dm_1st_inf[pupil_mask_1, :]

dm_2nd_inf = dm_2nd_inf.reshape(100, 100, 97)
dm_2nd_inf_masked = dm_2nd_inf[pupil_mask_2, :]

#TODO do you need a - sign here
M2C_1st = - np.load('PAPYRIIS_2stage_CNN_RL/M2C_1rst.npy')
M2C_2nd = np.load('PAPYRIIS_2stage_CNN_RL/M2C_2nd.npy')

C2M_2nd = np.linalg.pinv(M2C_2nd)
M2P_1st = dm_1st_inf_masked @ M2C_1st

GG = mode_covariance(M2P_1st)
projector_kl_2nd_to_1st = np.linalg.pinv(dm_1st_inf_masked @ M2C_1st) @ (dm_2nd_inf_masked @ M2C_2nd)


#RL
if RL:
    next_states_2nd = RL_telemetry_2nd.item()['slavedreconsCube'].squeeze()
    modes_2nd_stage_RL = next_states_2nd @ C2M_2nd.T
    modes_2nd_stage_RL = modes_2nd_stage_RL @ projector_kl_2nd_to_1st.T


    next_states_1st = CL1st_OL_2nd.item()['slavedreconsCube'].squeeze() #1st stage residual as measured by 2nd stage
    modes_1st_stage_RL = next_states_1st @ C2M_2nd.T
    
    modes_1st_stage_RL = modes_1st_stage_RL @ projector_kl_2nd_to_1st.T
    dm_atm_coefs = RL_telemetry_1st.item()['dmCmdCube'].squeeze()
    modes_atm_RL = mode_calculator_fromDM(dm_atm_coefs, M2C_1st)

    dynamics_loss = np.load(f"{loaddir}/dynamics_loss_warmup.npy") 
    policy_loss = np.load(f"{loaddir}/policy_loss_warmup.npy") 

    frequency_RL = frequency_2nd
    time_plot_RL = np.arange(0, 30000 / frequency_2nd, 1 / frequency_2nd)



#integrator
if integrator:
    next_states_2nd = int_telemetry_2nd.item()['slavedreconsCube'].squeeze()
    modes_2nd_stage_int = next_states_2nd @ C2M_2nd.T
    modes_2nd_stage_int = modes_2nd_stage_int @ projector_kl_2nd_to_1st.T

    frequency_int = frequency_2nd
    time_plot_int = np.arange(0, 30000 / frequency_2nd, 1 / frequency_2nd)

#linear
next_states_2nd_linear = linear_telemetry_2nd.item()['slavedreconsCube'].squeeze()
modes_2nd_stage_linear = next_states_2nd_linear @ C2M_2nd.T
modes_2nd_stage_linear = modes_2nd_stage_linear @ projector_kl_2nd_to_1st.T


if RL and plot_sim:
    next_states_2nd_sim = RL_telemetry_2nd_sim['all_reconstructed_cmd'].squeeze()
    modes_2nd_stage_RL_sim = next_states_2nd_sim @ C2M_2nd.T
    modes_2nd_stage_RL_sim = modes_2nd_stage_RL_sim @ projector_kl_2nd_to_1st.T

    next_states_1st_sim = CL1st_OL_2nd_sim['all_reconstructed_cmd'].squeeze()
    modes_1st_stage_RL_sim = next_states_1st_sim @ C2M_2nd.T
    modes_1st_stage_RL_sim = modes_1st_stage_RL_sim @ projector_kl_2nd_to_1st.T

    dm_atm_coefs_sim = RL_telemetry_1st_sim['dm_commands']
    modes_atm_RL_sim = mode_calculator_fromDM(dm_atm_coefs_sim, M2C_1st)

    frequency_RL_sim = frequency_2nd
    time_plot_RL_sim = np.arange(0, len(modes_2nd_stage_RL_sim) / frequency_2nd, 1 / frequency_2nd)

if integrator and plot_sim:
    next_states_2nd_int_sim = int_telemetry_2nd_sim['all_reconstructed_cmd'].squeeze()
    modes_2nd_stage_int_sim = next_states_2nd_int_sim @ C2M_2nd.T
    modes_2nd_stage_int_sim = modes_2nd_stage_int_sim @ projector_kl_2nd_to_1st.T

    frequency_int_sim = frequency_2nd
    time_plot_int_sim = np.arange(0, len(modes_2nd_stage_int_sim) / frequency_2nd, 1 / frequency_2nd)


modes_atm = modes_atm_RL
time_plot_atm = time_plot_RL
f_samp_atm = frequency_atm

if plot_sim and RL:
    modes_atm_sim = modes_atm_RL_sim
    f_samp_atm_sim = frequency_1st



# ---------------------------------------------------Loss---------------------------------------------------#
if RL:
    plt.figure()
    plt.subplot(121)
    plt.title("dynamics_loss warmup")
    plt.plot(dynamics_loss)
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.xscale('log')
    plt.yscale('log')
    plt.subplot(122)
    plt.title("policy_loss warmup")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.plot(policy_loss)
    plt.xscale('log')
    plt.yscale('log')


# ---------------------------------------------------Zernike/KL decomposition---------------------------------------------------#
#RL
def KL_variance_calculator(modes, GG_diag, frame_bounds):
    """
    Calculate variance of KL modes in nm^2.

    modes shape: (timeseries length, no_of_modes)
    GG_diag: precomputed diagonal of mode_covariance(m2p_nm), where m2p_nm = dm_inf_masked * 1e9 @ m2c
    """
    modes_masked = modes[frame_bounds[0]:frame_bounds[1], :]
    modes_var = np.var(np.asarray(modes_masked), axis=0) * GG_diag
    return modes_var, modes_masked

# M2P_1st is already computed above; scale to nm before computing covariance
GG_diag = np.diag(mode_covariance(M2P_1st * 1e9))

if RL:
    coefs_var_1st_stage_masked_RL, modes_1st_stage_RL_masked = KL_variance_calculator(modes_1st_stage_RL, GG_diag, KL_frames_RL)
    coefs_var_2nd_stage_masked_RL, modes_2nd_stage_RL_masked = KL_variance_calculator(modes_2nd_stage_RL, GG_diag, KL_frames_RL)

if integrator:
    coefs_var_2nd_stage_masked_int, modes_2nd_stage_int_masked = KL_variance_calculator(modes_2nd_stage_int, GG_diag, KL_frames_int)

coefs_var_2nd_stage_masked_linear, modes_2nd_stage_linear_masked = KL_variance_calculator(modes_2nd_stage_linear, GG_diag, KL_frames_int)

if ideal:
    coefs_var_1st_stage_masked_ideal, modes_1st_stage_ideal_masked = KL_variance_calculator(modes_1st_stage_ideal, GG_diag, KL_frames_ideal)
    coefs_var_2nd_stage_masked_ideal, modes_2nd_stage_ideal_masked = KL_variance_calculator(modes_2nd_stage_ideal, GG_diag, KL_frames_ideal)
#atmosphere
coefs_var_atm, _ = KL_variance_calculator(modes_atm, GG_diag, KL_frames_int)

if RL and plot_sim:
    coefs_var_1st_stage_masked_RL_sim, modes_1st_stage_RL_sim_masked = KL_variance_calculator(modes_1st_stage_RL_sim, GG_diag, KL_frames_RL)
    coefs_var_2nd_stage_masked_RL_sim, modes_2nd_stage_RL_sim_masked = KL_variance_calculator(modes_2nd_stage_RL_sim, GG_diag, KL_frames_RL)

if integrator and plot_sim:
    coefs_var_2nd_stage_masked_int_sim, modes_2nd_stage_int_sim_masked = KL_variance_calculator(modes_2nd_stage_int_sim, GG_diag, KL_frames_int)

if plot_sim and RL:
    coefs_var_atm_sim, _ = KL_variance_calculator(modes_atm_sim, GG_diag, KL_frames_int)


plt.figure(figsize=(12, 8))
if plot_onsky:
    plt.plot(coefs_var_atm, color="black", label="Atm")
if plot_sim and RL:
    plt.plot(coefs_var_atm_sim, ':', color="black", label="Atm (sim)")
if RL:
    if plot_onsky:
        var_1st = np.sum(coefs_var_1st_stage_masked_RL[:70])
        plt.plot(coefs_var_1st_stage_masked_RL, '--', color="cornflowerblue", lw=2.5, label=f"1st stage (integrator)")


# var_linear = np.sum(coefs_var_2nd_stage_masked_linear[:70])
# plt.plot(coefs_var_2nd_stage_masked_linear, color="blue", lw=2.5, label=f"2nd stage ({label_linear})")
if integrator:
    if plot_onsky:
        var_int = np.sum(coefs_var_2nd_stage_masked_int[:70])
        plt.plot(coefs_var_2nd_stage_masked_int, color="indianred", lw=2.5, label=f"2nd stage ({label_int})")
    if plot_sim:
        plt.plot(coefs_var_2nd_stage_masked_int_sim, ':', color="indianred", lw=2.5, label=f"2nd stage ({label_int_sim})")


if RL:
    if plot_onsky:
        var_RL  = np.sum(coefs_var_2nd_stage_masked_RL[:70])
        plt.plot(coefs_var_2nd_stage_masked_RL, color="red", lw=2.5, label=f"2nd stage ({label_RL})")
    if plot_sim:
        plt.plot(coefs_var_1st_stage_masked_RL_sim, ':', color="cornflowerblue", lw=2.5, label=f"1st stage ({label_int_sim})")
        plt.plot(coefs_var_2nd_stage_masked_RL_sim, ':', color="red", lw=2.5, label=f"2nd stage ({label_RL_sim})")

if ideal:
    var_ideal = np.sum(coefs_var_2nd_stage_masked_ideal[:70])
    plt.plot(coefs_var_2nd_stage_masked_ideal, color="green", lw=2.5, label=f"2nd stage ({label_ideal})")

# if RL and plot_onsky:
#     ratio = var_linear / var_RL
#     modes_x = np.arange(len(coefs_var_2nd_stage_masked_linear))
#     plt.fill_between(
#         modes_x[:70],
#         coefs_var_2nd_stage_masked_RL[:70],
#         coefs_var_2nd_stage_masked_linear[:70],
#         where=coefs_var_2nd_stage_masked_linear[:70] > coefs_var_2nd_stage_masked_RL[:70],
#         color='red', alpha=0.12)
#     plt.text(0.04, 0.04, f'Gain of ×{ratio:.1f}', fontsize=36, fontweight='bold',
#              color='darkred', va='bottom', ha='left',
#              transform=plt.gca().transAxes)

plt.title(f"KL mode variance", fontsize=30)
plt.yscale("log")
plt.xscale("log")
# plt.ylim(bottom=0.5)
plt.xlabel("KL mode index", fontsize=24)
plt.ylabel("Temporal residual variance (nm²)", fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend(fontsize=18)
plt.tight_layout()
# plt.savefig(f'{loaddir}/KL_mode_variance.png', dpi=300, bbox_inches='tight')



# ---------------------------------------------------Temporal PSD---------------------------------------------------#
if RL:
    residual_mode_1_curve_1st_RL_full = modes_1st_stage_RL[:, mode_1]
    residual_mode_2_curve_1st_RL_full = modes_1st_stage_RL[:, mode_2]
    residual_mode_3_curve_1st_RL_full = modes_1st_stage_RL[:, mode_3]
    residual_mode_4_curve_1st_RL_full = modes_1st_stage_RL[:, mode_4]
    residual_mode_5_curve_1st_RL_full = modes_1st_stage_RL[:, mode_5]

    residual_mode_1_curve_2nd_RL_full = modes_2nd_stage_RL[:, mode_1]
    residual_mode_2_curve_2nd_RL_full = modes_2nd_stage_RL[:, mode_2]
    residual_mode_3_curve_2nd_RL_full = modes_2nd_stage_RL[:, mode_3]
    residual_mode_4_curve_2nd_RL_full = modes_2nd_stage_RL[:, mode_4]
    residual_mode_5_curve_2nd_RL_full = modes_2nd_stage_RL[:, mode_5]

if integrator:
    residual_mode_1_curve_2nd_int_full = modes_2nd_stage_int[:, mode_1]
    residual_mode_2_curve_2nd_int_full = modes_2nd_stage_int[:, mode_2]
    residual_mode_3_curve_2nd_int_full = modes_2nd_stage_int[:, mode_3]
    residual_mode_4_curve_2nd_int_full = modes_2nd_stage_int[:, mode_4]
    residual_mode_5_curve_2nd_int_full = modes_2nd_stage_int[:, mode_5]

if ideal:
    residual_mode_1_curve_1st_ideal_full = modes_1st_stage_ideal[:, mode_1]
    residual_mode_2_curve_1st_ideal_full = modes_1st_stage_ideal[:, mode_2]
    residual_mode_3_curve_1st_ideal_full = modes_1st_stage_ideal[:, mode_3]
    residual_mode_4_curve_1st_ideal_full = modes_1st_stage_ideal[:, mode_4]
    residual_mode_5_curve_1st_ideal_full = modes_1st_stage_ideal[:, mode_5]

    residual_mode_1_curve_2nd_ideal_full = modes_2nd_stage_ideal[:, mode_1]
    residual_mode_2_curve_2nd_ideal_full = modes_2nd_stage_ideal[:, mode_2]
    residual_mode_3_curve_2nd_ideal_full = modes_2nd_stage_ideal[:, mode_3]
    residual_mode_4_curve_2nd_ideal_full = modes_2nd_stage_ideal[:, mode_4]
    residual_mode_5_curve_2nd_ideal_full = modes_2nd_stage_ideal[:, mode_5]

if RL and plot_sim:
    residual_mode_1_curve_1st_RL_sim_full = modes_1st_stage_RL_sim[:, mode_1]
    residual_mode_2_curve_1st_RL_sim_full = modes_1st_stage_RL_sim[:, mode_2]
    residual_mode_3_curve_1st_RL_sim_full = modes_1st_stage_RL_sim[:, mode_3]
    residual_mode_4_curve_1st_RL_sim_full = modes_1st_stage_RL_sim[:, mode_4]
    residual_mode_5_curve_1st_RL_sim_full = modes_1st_stage_RL_sim[:, mode_5]

    residual_mode_1_curve_2nd_RL_sim_full = modes_2nd_stage_RL_sim[:, mode_1]
    residual_mode_2_curve_2nd_RL_sim_full = modes_2nd_stage_RL_sim[:, mode_2]
    residual_mode_3_curve_2nd_RL_sim_full = modes_2nd_stage_RL_sim[:, mode_3]
    residual_mode_4_curve_2nd_RL_sim_full = modes_2nd_stage_RL_sim[:, mode_4]
    residual_mode_5_curve_2nd_RL_sim_full = modes_2nd_stage_RL_sim[:, mode_5]

if integrator and plot_sim:
    residual_mode_1_curve_2nd_int_sim_full = modes_2nd_stage_int_sim[:, mode_1]
    residual_mode_2_curve_2nd_int_sim_full = modes_2nd_stage_int_sim[:, mode_2]
    residual_mode_3_curve_2nd_int_sim_full = modes_2nd_stage_int_sim[:, mode_3]
    residual_mode_4_curve_2nd_int_sim_full = modes_2nd_stage_int_sim[:, mode_4]
    residual_mode_5_curve_2nd_int_sim_full = modes_2nd_stage_int_sim[:, mode_5]


if RL:
    PSD_residual_mode_1_freq_t_1st_RL, PSD_residual_mode_1_1st_RL, _ = tPSD_calculator(modes_1st_stage_RL_masked, mode_1, GG_diag, frequency_1st)
    PSD_residual_mode_2_freq_t_1st_RL, PSD_residual_mode_2_1st_RL, _ = tPSD_calculator(modes_1st_stage_RL_masked, mode_2, GG_diag, frequency_1st)
    PSD_residual_mode_3_freq_t_1st_RL, PSD_residual_mode_3_1st_RL, _ = tPSD_calculator(modes_1st_stage_RL_masked, mode_3, GG_diag, frequency_1st)
    PSD_residual_mode_4_freq_t_1st_RL, PSD_residual_mode_4_1st_RL, _ = tPSD_calculator(modes_1st_stage_RL_masked, mode_4, GG_diag, frequency_1st)
    PSD_residual_mode_5_freq_t_1st_RL, PSD_residual_mode_5_1st_RL, _ = tPSD_calculator(modes_1st_stage_RL_masked, mode_5, GG_diag, frequency_1st)

    PSD_residual_mode_1_freq_t_2nd_RL, PSD_residual_mode_1_2nd_RL, _ = tPSD_calculator(modes_2nd_stage_RL_masked, mode_1, GG_diag, frequency_RL)
    PSD_residual_mode_2_freq_t_2nd_RL, PSD_residual_mode_2_2nd_RL, _ = tPSD_calculator(modes_2nd_stage_RL_masked, mode_2, GG_diag, frequency_RL)
    PSD_residual_mode_3_freq_t_2nd_RL, PSD_residual_mode_3_2nd_RL, _ = tPSD_calculator(modes_2nd_stage_RL_masked, mode_3, GG_diag, frequency_RL)
    PSD_residual_mode_4_freq_t_2nd_RL, PSD_residual_mode_4_2nd_RL, _ = tPSD_calculator(modes_2nd_stage_RL_masked, mode_4, GG_diag, frequency_RL)
    PSD_residual_mode_5_freq_t_2nd_RL, PSD_residual_mode_5_2nd_RL, _ = tPSD_calculator(modes_2nd_stage_RL_masked, mode_5, GG_diag, frequency_RL)

if integrator:
    PSD_residual_mode_1_freq_t_2nd_int, PSD_residual_mode_1_2nd_int, _ = tPSD_calculator(modes_2nd_stage_int_masked, mode_1, GG_diag, frequency_int)
    PSD_residual_mode_2_freq_t_2nd_int, PSD_residual_mode_2_2nd_int, _ = tPSD_calculator(modes_2nd_stage_int_masked, mode_2, GG_diag, frequency_int)
    PSD_residual_mode_3_freq_t_2nd_int, PSD_residual_mode_3_2nd_int, _ = tPSD_calculator(modes_2nd_stage_int_masked, mode_3, GG_diag, frequency_int)
    PSD_residual_mode_4_freq_t_2nd_int, PSD_residual_mode_4_2nd_int, _ = tPSD_calculator(modes_2nd_stage_int_masked, mode_4, GG_diag, frequency_int)
    PSD_residual_mode_5_freq_t_2nd_int, PSD_residual_mode_5_2nd_int, _ = tPSD_calculator(modes_2nd_stage_int_masked, mode_5, GG_diag, frequency_int)

if ideal:
    PSD_residual_mode_1_freq_t_1st_ideal, PSD_residual_mode_1_1st_ideal, _ = tPSD_calculator(modes_1st_stage_ideal_masked, mode_1, GG_diag, frequency_1st)
    PSD_residual_mode_2_freq_t_1st_ideal, PSD_residual_mode_2_1st_ideal, _ = tPSD_calculator(modes_1st_stage_ideal_masked, mode_2, GG_diag, frequency_1st)
    PSD_residual_mode_3_freq_t_1st_ideal, PSD_residual_mode_3_1st_ideal, _ = tPSD_calculator(modes_1st_stage_ideal_masked, mode_3, GG_diag, frequency_1st)
    PSD_residual_mode_4_freq_t_1st_ideal, PSD_residual_mode_4_1st_ideal, _ = tPSD_calculator(modes_1st_stage_ideal_masked, mode_4, GG_diag, frequency_1st)
    PSD_residual_mode_5_freq_t_1st_ideal, PSD_residual_mode_5_1st_ideal, _ = tPSD_calculator(modes_1st_stage_ideal_masked, mode_5, GG_diag, frequency_1st)

    PSD_residual_mode_1_freq_t_2nd_ideal, PSD_residual_mode_1_2nd_ideal, _ = tPSD_calculator(modes_2nd_stage_ideal_masked, mode_1, GG_diag, frequency_ideal)
    PSD_residual_mode_2_freq_t_2nd_ideal, PSD_residual_mode_2_2nd_ideal, _ = tPSD_calculator(modes_2nd_stage_ideal_masked, mode_2, GG_diag, frequency_ideal)
    PSD_residual_mode_3_freq_t_2nd_ideal, PSD_residual_mode_3_2nd_ideal, _ = tPSD_calculator(modes_2nd_stage_ideal_masked, mode_3, GG_diag, frequency_ideal)
    PSD_residual_mode_4_freq_t_2nd_ideal, PSD_residual_mode_4_2nd_ideal, _ = tPSD_calculator(modes_2nd_stage_ideal_masked, mode_4, GG_diag, frequency_ideal)
    PSD_residual_mode_5_freq_t_2nd_ideal, PSD_residual_mode_5_2nd_ideal, _ = tPSD_calculator(modes_2nd_stage_ideal_masked, mode_5, GG_diag, frequency_ideal)

PSD_residual_mode_1_freq_t_linear, PSD_residual_mode_1_linear, _ = tPSD_calculator(modes_2nd_stage_linear_masked, mode_1, GG_diag, frequency_2nd)

PSD_atm_mode_1_freq_t, PSD_atm_mode_1, _ = tPSD_calculator(modes_atm, mode_1, GG_diag, f_samp_atm)
PSD_atm_mode_2_freq_t, PSD_atm_mode_2, _ = tPSD_calculator(modes_atm, mode_2, GG_diag, f_samp_atm)
PSD_atm_mode_3_freq_t, PSD_atm_mode_3, _ = tPSD_calculator(modes_atm, mode_3, GG_diag, f_samp_atm)
PSD_atm_mode_4_freq_t, PSD_atm_mode_4, _ = tPSD_calculator(modes_atm, mode_4, GG_diag, f_samp_atm)
PSD_atm_mode_5_freq_t, PSD_atm_mode_5, _ = tPSD_calculator(modes_atm, mode_5, GG_diag, f_samp_atm)

if RL and plot_sim:
    PSD_residual_mode_1_freq_t_1st_RL_sim, PSD_residual_mode_1_1st_RL_sim, _ = tPSD_calculator(modes_1st_stage_RL_sim_masked, mode_1, GG_diag, frequency_1st)
    PSD_residual_mode_2_freq_t_1st_RL_sim, PSD_residual_mode_2_1st_RL_sim, _ = tPSD_calculator(modes_1st_stage_RL_sim_masked, mode_2, GG_diag, frequency_1st)
    PSD_residual_mode_3_freq_t_1st_RL_sim, PSD_residual_mode_3_1st_RL_sim, _ = tPSD_calculator(modes_1st_stage_RL_sim_masked, mode_3, GG_diag, frequency_1st)
    PSD_residual_mode_4_freq_t_1st_RL_sim, PSD_residual_mode_4_1st_RL_sim, _ = tPSD_calculator(modes_1st_stage_RL_sim_masked, mode_4, GG_diag, frequency_1st)
    PSD_residual_mode_5_freq_t_1st_RL_sim, PSD_residual_mode_5_1st_RL_sim, _ = tPSD_calculator(modes_1st_stage_RL_sim_masked, mode_5, GG_diag, frequency_1st)

    PSD_residual_mode_1_freq_t_2nd_RL_sim, PSD_residual_mode_1_2nd_RL_sim, _ = tPSD_calculator(modes_2nd_stage_RL_sim_masked, mode_1, GG_diag, frequency_RL_sim)
    PSD_residual_mode_2_freq_t_2nd_RL_sim, PSD_residual_mode_2_2nd_RL_sim, _ = tPSD_calculator(modes_2nd_stage_RL_sim_masked, mode_2, GG_diag, frequency_RL_sim)
    PSD_residual_mode_3_freq_t_2nd_RL_sim, PSD_residual_mode_3_2nd_RL_sim, _ = tPSD_calculator(modes_2nd_stage_RL_sim_masked, mode_3, GG_diag, frequency_RL_sim)
    PSD_residual_mode_4_freq_t_2nd_RL_sim, PSD_residual_mode_4_2nd_RL_sim, _ = tPSD_calculator(modes_2nd_stage_RL_sim_masked, mode_4, GG_diag, frequency_RL_sim)
    PSD_residual_mode_5_freq_t_2nd_RL_sim, PSD_residual_mode_5_2nd_RL_sim, _ = tPSD_calculator(modes_2nd_stage_RL_sim_masked, mode_5, GG_diag, frequency_RL_sim)

if integrator and plot_sim:
    PSD_residual_mode_1_freq_t_2nd_int_sim, PSD_residual_mode_1_2nd_int_sim, _ = tPSD_calculator(modes_2nd_stage_int_sim_masked, mode_1, GG_diag, frequency_int_sim)
    PSD_residual_mode_2_freq_t_2nd_int_sim, PSD_residual_mode_2_2nd_int_sim, _ = tPSD_calculator(modes_2nd_stage_int_sim_masked, mode_2, GG_diag, frequency_int_sim)
    PSD_residual_mode_3_freq_t_2nd_int_sim, PSD_residual_mode_3_2nd_int_sim, _ = tPSD_calculator(modes_2nd_stage_int_sim_masked, mode_3, GG_diag, frequency_int_sim)
    PSD_residual_mode_4_freq_t_2nd_int_sim, PSD_residual_mode_4_2nd_int_sim, _ = tPSD_calculator(modes_2nd_stage_int_sim_masked, mode_4, GG_diag, frequency_int_sim)
    PSD_residual_mode_5_freq_t_2nd_int_sim, PSD_residual_mode_5_2nd_int_sim, _ = tPSD_calculator(modes_2nd_stage_int_sim_masked, mode_5, GG_diag, frequency_int_sim)

if plot_sim and RL:
    PSD_atm_mode_1_freq_t_sim, PSD_atm_mode_1_sim, _ = tPSD_calculator(modes_atm_sim, mode_1, GG_diag, f_samp_atm_sim)
    PSD_atm_mode_2_freq_t_sim, PSD_atm_mode_2_sim, _ = tPSD_calculator(modes_atm_sim, mode_2, GG_diag, f_samp_atm_sim)
    PSD_atm_mode_3_freq_t_sim, PSD_atm_mode_3_sim, _ = tPSD_calculator(modes_atm_sim, mode_3, GG_diag, f_samp_atm_sim)
    PSD_atm_mode_4_freq_t_sim, PSD_atm_mode_4_sim, _ = tPSD_calculator(modes_atm_sim, mode_4, GG_diag, f_samp_atm_sim)
    PSD_atm_mode_5_freq_t_sim, PSD_atm_mode_5_sim, _ = tPSD_calculator(modes_atm_sim, mode_5, GG_diag, f_samp_atm_sim)


if plot_timeseries:
    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(time_plot_RL, residual_mode_1_curve_1st_RL_full, color="indianred", label=f"mode_{mode_1}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_1_curve_2nd_RL_full, color="red", label=f"mode_{mode_1}_2nd_stage_{label_RL}")
        if plot_sim:
            plt.plot(time_plot_RL_sim, residual_mode_1_curve_1st_RL_sim_full, ':', color="indianred", label=f"mode_{mode_1}_1st_stage_{label_int_sim}")
            plt.plot(time_plot_RL_sim, residual_mode_1_curve_2nd_RL_sim_full, ':', color="red", label=f"mode_{mode_1}_2nd_stage_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(time_plot_int, residual_mode_1_curve_2nd_int_full, color="blue", label=f"mode_{mode_1}_2nd_stage_{label_int}")
        if plot_sim:
            plt.plot(time_plot_int_sim, residual_mode_1_curve_2nd_int_sim_full, ':', color="blue", label=f"mode_{mode_1}_2nd_stage_{label_int_sim}")
    if ideal:
        plt.plot(time_plot_ideal, residual_mode_1_curve_1st_ideal_full, label=f"mode_{mode_1}_1st_stage_{label_ideal}")
        plt.plot(time_plot_ideal, residual_mode_1_curve_2nd_ideal_full, label=f"mode_{mode_1}_2nd_stage_{label_ideal}")
    plt.title(f"residual/atm timeseries mode_{mode_1}, gain {CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_1}")


    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(time_plot_RL, residual_mode_2_curve_1st_RL_full, color="indianred", label=f"mode_{mode_2}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_2_curve_2nd_RL_full, color="red", label=f"mode_{mode_2}_2nd_stage_{label_RL}")
        if plot_sim:
            plt.plot(time_plot_RL_sim, residual_mode_2_curve_1st_RL_sim_full, ':', color="indianred", label=f"mode_{mode_2}_1st_stage_{label_int_sim}")
            plt.plot(time_plot_RL_sim, residual_mode_2_curve_2nd_RL_sim_full, ':', color="red", label=f"mode_{mode_2}_2nd_stage_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(time_plot_int, residual_mode_2_curve_2nd_int_full, color="blue", label=f"mode_{mode_2}_2nd_stage_{label_int}")
        if plot_sim:
            plt.plot(time_plot_int_sim, residual_mode_2_curve_2nd_int_sim_full, ':', color="blue", label=f"mode_{mode_2}_2nd_stage_{label_int_sim}")
    if ideal:
        plt.plot(time_plot_ideal, residual_mode_2_curve_1st_ideal_full, label=f"mode_{mode_2}_1st_stage_{label_ideal}")
        plt.plot(time_plot_ideal, residual_mode_2_curve_2nd_ideal_full, label=f"mode_{mode_2}_2nd_stage_{label_ideal}")
    plt.title(f"residual/atm timeseries mode_{mode_2}, gain {CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_2}")


    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(time_plot_RL, residual_mode_3_curve_1st_RL_full, color="indianred", label=f"mode_{mode_3}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_3_curve_2nd_RL_full, color="red", label=f"mode_{mode_3}_2nd_stage_{label_RL}")
        if plot_sim:
            plt.plot(time_plot_RL_sim, residual_mode_3_curve_1st_RL_sim_full, ':', color="indianred", label=f"mode_{mode_3}_1st_stage_{label_int_sim}")
            plt.plot(time_plot_RL_sim, residual_mode_3_curve_2nd_RL_sim_full, ':', color="red", label=f"mode_{mode_3}_2nd_stage_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(time_plot_int, residual_mode_3_curve_2nd_int_full, color="blue", label=f"mode_{mode_3}_2nd_stage_{label_int}")
        if plot_sim:
            plt.plot(time_plot_int_sim, residual_mode_3_curve_2nd_int_sim_full, ':', color="blue", label=f"mode_{mode_3}_2nd_stage_{label_int_sim}")
    if ideal:
        plt.plot(time_plot_ideal, residual_mode_3_curve_1st_ideal_full, label=f"mode_{mode_3}_1st_stage__{label_ideal}")
        plt.plot(time_plot_ideal, residual_mode_3_curve_2nd_ideal_full, label=f"mode_{mode_3}_2nd_stage_{label_ideal}")
    plt.title(f"residual/atm timeseries mode_{mode_3}, gain {CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_3}")


    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(time_plot_RL, residual_mode_4_curve_1st_RL_full, color="indianred", label=f"mode_{mode_4}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_4_curve_2nd_RL_full, color="red", label=f"mode__{mode_4}_2nd_stage_{label_RL}")
        if plot_sim:
            plt.plot(time_plot_RL_sim, residual_mode_4_curve_1st_RL_sim_full, ':', color="indianred", label=f"mode_{mode_4}_1st_stage_{label_int_sim}")
            plt.plot(time_plot_RL_sim, residual_mode_4_curve_2nd_RL_sim_full, ':', color="red", label=f"mode_{mode_4}_2nd_stage_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(time_plot_int, residual_mode_4_curve_2nd_int_full, color="blue", label=f"mode_{mode_4}_2nd_stage_{label_int}")
        if plot_sim:
            plt.plot(time_plot_int_sim, residual_mode_4_curve_2nd_int_sim_full, ':', color="blue", label=f"mode_{mode_4}_2nd_stage_{label_int_sim}")
    if ideal:
        plt.plot(time_plot_ideal, residual_mode_4_curve_1st_ideal_full, label=f"mode_{mode_4}_1st_stage_{label_ideal}")
        plt.plot(time_plot_ideal, residual_mode_4_curve_2nd_ideal_full, label=f"mode_{mode_4}_2nd_stage_{label_ideal}")
    plt.title(f"residual/atm timeseries mode_{mode_4}, gain {CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_4}")


    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(time_plot_RL, residual_mode_5_curve_1st_RL_full, color="indianred", label=f"mode_{mode_5}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_5_curve_2nd_RL_full, color="red", label=f"mode_{mode_5}_2nd_stage_{label_RL}")
        if plot_sim:
            plt.plot(time_plot_RL_sim, residual_mode_5_curve_1st_RL_sim_full, ':', color="indianred", label=f"mode_{mode_5}_1st_stage_{label_int_sim}")
            plt.plot(time_plot_RL_sim, residual_mode_5_curve_2nd_RL_sim_full, ':', color="red", label=f"mode_{mode_5}_2nd_stage_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(time_plot_int, residual_mode_5_curve_2nd_int_full, color="blue", label=f"mode_{mode_5}_2nd_stage_{label_int}")
        if plot_sim:
            plt.plot(time_plot_int_sim, residual_mode_5_curve_2nd_int_sim_full, ':', color="blue", label=f"mode_{mode_5}_2nd_stage_{label_int_sim}")
    if ideal:
        plt.plot(time_plot_ideal, residual_mode_5_curve_1st_ideal_full, label=f"mode_{mode_5}_1st_stage_{label_ideal}")
        plt.plot(time_plot_ideal, residual_mode_5_curve_2nd_ideal_full, label=f"mode_{mode_5}_2nd_stage_{label_ideal}")
    plt.title(f"residual/atm timeseries mode_{mode_5}, gain {CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_5}")


if plot_tPSD:
    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(PSD_residual_mode_1_freq_t_1st_RL, PSD_residual_mode_1_1st_RL, '--', color="indianred", label=f"PSD_mode_{mode_1}_1st_int")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_RL, PSD_residual_mode_1_2nd_RL, color="red", label=f"PSD_mode_{mode_1}_2nd_{label_RL}")
        if plot_sim:
            plt.plot(PSD_residual_mode_1_freq_t_1st_RL_sim, PSD_residual_mode_1_1st_RL_sim, ':', color="indianred", label=f"PSD_mode_{mode_1}_1st_int ({label_int_sim})")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_RL_sim, PSD_residual_mode_1_2nd_RL_sim, ':', color="red", label=f"PSD_mode_{mode_1}_2nd_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(PSD_residual_mode_1_freq_t_2nd_int, PSD_residual_mode_1_2nd_int, color="blue", label=f"PSD_mode_{mode_1}_2nd_{label_int}")
        if plot_sim:
            plt.plot(PSD_residual_mode_1_freq_t_2nd_int_sim, PSD_residual_mode_1_2nd_int_sim, ':', color="blue", label=f"PSD_mode_{mode_1}_2nd_{label_int_sim}")
    if ideal:
        plt.plot(PSD_residual_mode_1_freq_t_2nd_ideal, PSD_residual_mode_1_2nd_ideal, label=f"PSD_mode_{mode_1}_2nd_{label_ideal}")
    if plot_onsky:
        plt.plot(PSD_atm_mode_1_freq_t, PSD_atm_mode_1, color="black", label=f"atm_PSD_mode_{mode_1}")
    if plot_sim and RL:
        plt.plot(PSD_atm_mode_1_freq_t_sim, PSD_atm_mode_1_sim, ':', color="black", label=f"atm_PSD_mode_{mode_1} (sim)")
    plt.title(f"residual PSD mode_{mode_1}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD (nm² Hz⁻¹)")

    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(PSD_residual_mode_2_freq_t_1st_RL, PSD_residual_mode_2_1st_RL, '--', color="indianred", label=f"PSD_mode_{mode_2}_1st_int")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_RL, PSD_residual_mode_2_2nd_RL, color="red", label=f"PSD_mode_{mode_2}_2nd_{label_RL}")
        if plot_sim:
            plt.plot(PSD_residual_mode_2_freq_t_1st_RL_sim, PSD_residual_mode_2_1st_RL_sim, ':', color="indianred", label=f"PSD_mode_{mode_2}_1st_int ({label_int_sim})")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_RL_sim, PSD_residual_mode_2_2nd_RL_sim, ':', color="red", label=f"PSD_mode_{mode_2}_2nd_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(PSD_residual_mode_2_freq_t_2nd_int, PSD_residual_mode_2_2nd_int, color="blue", label=f"PSD_mode_{mode_2}_2nd_{label_int}")
        if plot_sim:
            plt.plot(PSD_residual_mode_2_freq_t_2nd_int_sim, PSD_residual_mode_2_2nd_int_sim, ':', color="blue", label=f"PSD_mode_{mode_2}_2nd_{label_int_sim}")
    if ideal:
        plt.plot(PSD_residual_mode_2_freq_t_2nd_ideal, PSD_residual_mode_2_2nd_ideal, label=f"PSD_mode_{mode_2}_2nd_{label_ideal}")
    if plot_onsky:
        plt.plot(PSD_atm_mode_2_freq_t, PSD_atm_mode_2, color="black", label=f"atm_PSD_mode_{mode_2}")
    if plot_sim and RL:
        plt.plot(PSD_atm_mode_2_freq_t_sim, PSD_atm_mode_2_sim, ':', color="black", label=f"atm_PSD_mode_{mode_2} (sim)")
    plt.title(f"residual PSD mode_{mode_2}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD (nm² Hz⁻¹)")


    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(PSD_residual_mode_3_freq_t_1st_RL, PSD_residual_mode_3_1st_RL, '--', color="indianred", label=f"PSD_mode_{mode_3}_1st_int")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_RL, PSD_residual_mode_3_2nd_RL, color="red", label=f"PSD_mode_{mode_3}_2nd_{label_RL}")
        if plot_sim:
            plt.plot(PSD_residual_mode_3_freq_t_1st_RL_sim, PSD_residual_mode_3_1st_RL_sim, ':', color="indianred", label=f"PSD_mode_{mode_3}_1st_int ({label_int_sim})")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_RL_sim, PSD_residual_mode_3_2nd_RL_sim, ':', color="red", label=f"PSD_mode_{mode_3}_2nd_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(PSD_residual_mode_3_freq_t_2nd_int, PSD_residual_mode_3_2nd_int, color="blue", label=f"PSD_mode_{mode_3}_2nd_{label_int}")
        if plot_sim:
            plt.plot(PSD_residual_mode_3_freq_t_2nd_int_sim, PSD_residual_mode_3_2nd_int_sim, ':', color="blue", label=f"PSD_mode_{mode_3}_2nd_{label_int_sim}")
    if ideal:
        plt.plot(PSD_residual_mode_3_freq_t_2nd_ideal, PSD_residual_mode_3_2nd_ideal, label=f"PSD_mode_{mode_3}_2nd_{label_ideal}")
    if plot_onsky:
        plt.plot(PSD_atm_mode_3_freq_t, PSD_atm_mode_3, color="black", label=f"atm_PSD_mode_{mode_3}")
    if plot_sim and RL:
        plt.plot(PSD_atm_mode_3_freq_t_sim, PSD_atm_mode_3_sim, ':', color="black", label=f"atm_PSD_mode_{mode_3} (sim)")
    plt.title(f"residual PSD mode_{mode_3}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD (nm² Hz⁻¹)")


    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(PSD_residual_mode_4_freq_t_1st_RL, PSD_residual_mode_4_1st_RL, '--', color="indianred", label=f"PSD_mode_{mode_4}_1st_int")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_RL, PSD_residual_mode_4_2nd_RL, color="red", label=f"PSD_mode_{mode_4}_2nd_{label_RL}")
        if plot_sim:
            plt.plot(PSD_residual_mode_4_freq_t_1st_RL_sim, PSD_residual_mode_4_1st_RL_sim, ':', color="indianred", label=f"PSD_mode_{mode_4}_1st_int ({label_int_sim})")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_RL_sim, PSD_residual_mode_4_2nd_RL_sim, ':', color="red", label=f"PSD_mode_{mode_4}_2nd_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(PSD_residual_mode_4_freq_t_2nd_int, PSD_residual_mode_4_2nd_int, color="blue", label=f"PSD_mode_{mode_4}_2nd_{label_int}")
        if plot_sim:
            plt.plot(PSD_residual_mode_4_freq_t_2nd_int_sim, PSD_residual_mode_4_2nd_int_sim, ':', color="blue", label=f"PSD_mode_{mode_4}_2nd_{label_int_sim}")
    if ideal:
        plt.plot(PSD_residual_mode_4_freq_t_2nd_ideal, PSD_residual_mode_4_2nd_ideal, label=f"PSD_mode_{mode_4}_2nd_{label_ideal}")
    if plot_onsky:
        plt.plot(PSD_atm_mode_4_freq_t, PSD_atm_mode_4, color="black", label=f"atm_PSD_mode_{mode_4}")
    if plot_sim and RL:
        plt.plot(PSD_atm_mode_4_freq_t_sim, PSD_atm_mode_4_sim, ':', color="black", label=f"atm_PSD_mode_{mode_4} (sim)")
    plt.title(f"residual PSD mode_{mode_4}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD (nm² Hz⁻¹)")


    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(PSD_residual_mode_5_freq_t_1st_RL, PSD_residual_mode_5_1st_RL, '--', color="indianred", label=f"PSD_mode_{mode_5}_1st_int")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_RL, PSD_residual_mode_5_2nd_RL, color="red", label=f"PSD_mode_{mode_5}_2nd_{label_RL}")
        if plot_sim:
            plt.plot(PSD_residual_mode_5_freq_t_1st_RL_sim, PSD_residual_mode_5_1st_RL_sim, ':', color="indianred", label=f"PSD_mode_{mode_5}_1st_int ({label_int_sim})")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_RL_sim, PSD_residual_mode_5_2nd_RL_sim, ':', color="red", label=f"PSD_mode_{mode_5}_2nd_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(PSD_residual_mode_5_freq_t_2nd_int, PSD_residual_mode_5_2nd_int, color="blue", label=f"PSD_mode_{mode_5}_2nd_{label_int}")
        if plot_sim:
            plt.plot(PSD_residual_mode_5_freq_t_2nd_int_sim, PSD_residual_mode_5_2nd_int_sim, ':', color="blue", label=f"PSD_mode_{mode_5}_2nd_{label_int_sim}")
    if ideal:
        plt.plot(PSD_residual_mode_5_freq_t_2nd_ideal, PSD_residual_mode_5_2nd_ideal, label=f"PSD_mode_{mode_5}_2nd_{label_ideal}")
    if plot_onsky:
        plt.plot(PSD_atm_mode_5_freq_t, PSD_atm_mode_5, color="black", label=f"atm_PSD_mode_{mode_5}")
    if plot_sim and RL:
        plt.plot(PSD_atm_mode_5_freq_t_sim, PSD_atm_mode_5_sim, ':', color="black", label=f"atm_PSD_mode_{mode_5} (sim)")
    plt.title(f"residual PSD mode_{mode_5}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD (nm² Hz⁻¹)")



#---------------------------------------------------Cumulative PSD---------------------------------------------------#



if plot_tPSD:
    plt.figure(figsize=(12, 8))
    if RL:
        if plot_onsky:
            plt.plot(PSD_residual_mode_1_freq_t_1st_RL, np.cumsum(PSD_residual_mode_1_1st_RL), '--', color="indianred", lw=2.5, label="1st stage integrator")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_RL, np.cumsum(PSD_residual_mode_1_2nd_RL), color="red", lw=2.5, label="2nd stage CNN + PO4AO")
        if plot_sim:
            plt.plot(PSD_residual_mode_1_freq_t_1st_RL_sim, np.cumsum(PSD_residual_mode_1_1st_RL_sim), ':', color="indianred", lw=2.5, label=f"1st stage integrator ({label_int_sim})")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_RL_sim, np.cumsum(PSD_residual_mode_1_2nd_RL_sim), ':', color="red", lw=2.5, label=f"2nd stage CNN + PO4AO ({label_RL_sim})")
    if integrator:
        if plot_onsky:
            plt.plot(PSD_residual_mode_1_freq_t_2nd_int, np.cumsum(PSD_residual_mode_1_2nd_int), color="blue", lw=2.5, label="2nd stage CNN")
        if plot_sim:
            plt.plot(PSD_residual_mode_1_freq_t_2nd_int_sim, np.cumsum(PSD_residual_mode_1_2nd_int_sim), ':', color="blue", lw=2.5, label=f"2nd stage CNN ({label_int_sim})")
    if ideal:
        plt.plot(PSD_residual_mode_1_freq_t_2nd_ideal, np.cumsum(PSD_residual_mode_1_2nd_ideal), lw=2.5, label=f"PSD_mode_{mode_1}_2nd_{label_ideal}")

    plt.title(f"Cumulative PSD tip mode", fontsize=30)
    plt.xlabel("Frequency (Hz)", fontsize=24)
    plt.ylabel("Cumulative PSD (nm²)", fontsize=24)
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(bottom=7)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend(fontsize=18)
    plt.axvspan(50, 60, alpha=0.2, color='red', zorder=0)#color='red', alpha=0.12
    plt.text(70, 150, 'Vibration\nCompensation', fontsize=24, fontweight='bold',
             color='darkred', ha='left', va='center')
    # plt.savefig(f'{loaddir}/cumsum_PSD.png', dpi=300, bbox_inches='tight')


    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(PSD_residual_mode_2_freq_t_1st_RL, np.cumsum(PSD_residual_mode_2_1st_RL), '--', color="indianred", label=f"PSD_mode_{mode_2}_1st_int")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_RL, np.cumsum(PSD_residual_mode_2_2nd_RL), color="red", label=f"PSD_mode_{mode_2}_2nd_{label_RL}")
        if plot_sim:
            plt.plot(PSD_residual_mode_2_freq_t_1st_RL_sim, np.cumsum(PSD_residual_mode_2_1st_RL_sim), ':', color="indianred", label=f"PSD_mode_{mode_2}_1st_int ({label_int_sim})")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_RL_sim, np.cumsum(PSD_residual_mode_2_2nd_RL_sim), ':', color="red", label=f"PSD_mode_{mode_2}_2nd_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(PSD_residual_mode_2_freq_t_2nd_int, np.cumsum(PSD_residual_mode_2_2nd_int), color="blue", label=f"PSD_mode_{mode_2}_2nd_{label_int}")
        if plot_sim:
            plt.plot(PSD_residual_mode_2_freq_t_2nd_int_sim, np.cumsum(PSD_residual_mode_2_2nd_int_sim), ':', color="blue", label=f"PSD_mode_{mode_2}_2nd_{label_int_sim}")
    if ideal:
        plt.plot(PSD_residual_mode_2_freq_t_2nd_ideal, np.cumsum(PSD_residual_mode_2_2nd_ideal), label=f"PSD_mode_{mode_2}_2nd_{label_ideal}")

    plt.title(f"Cumulative PSD KL mode {mode_2}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("cumulative PSD (nm²)")


    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(PSD_residual_mode_3_freq_t_1st_RL, np.cumsum(PSD_residual_mode_3_1st_RL), '--', color="indianred", label=f"PSD_mode_{mode_3}_1st_int")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_RL, np.cumsum(PSD_residual_mode_3_2nd_RL), color="red", label=f"PSD_mode_{mode_3}_2nd_{label_RL}")
        if plot_sim:
            plt.plot(PSD_residual_mode_3_freq_t_1st_RL_sim, np.cumsum(PSD_residual_mode_3_1st_RL_sim), ':', color="indianred", label=f"PSD_mode_{mode_3}_1st_int ({label_int_sim})")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_RL_sim, np.cumsum(PSD_residual_mode_3_2nd_RL_sim), ':', color="red", label=f"PSD_mode_{mode_3}_2nd_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(PSD_residual_mode_3_freq_t_2nd_int, np.cumsum(PSD_residual_mode_3_2nd_int), color="blue", label=f"PSD_mode_{mode_3}_2nd_{label_int}")
        if plot_sim:
            plt.plot(PSD_residual_mode_3_freq_t_2nd_int_sim, np.cumsum(PSD_residual_mode_3_2nd_int_sim), ':', color="blue", label=f"PSD_mode_{mode_3}_2nd_{label_int_sim}")
    if ideal:
        plt.plot(PSD_residual_mode_3_freq_t_2nd_ideal, np.cumsum(PSD_residual_mode_3_2nd_ideal), label=f"PSD_mode_{mode_3}_2nd_{label_ideal}")

    plt.title(f"Cumulative PSD KL mode {mode_3}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("cumulative PSD (nm²)")


    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(PSD_residual_mode_4_freq_t_1st_RL, np.cumsum(PSD_residual_mode_4_1st_RL), '--', color="indianred", label=f"PSD_mode_{mode_4}_1st_int")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_RL, np.cumsum(PSD_residual_mode_4_2nd_RL), color="red", label=f"PSD_mode_{mode_4}_2nd_{label_RL}")
        if plot_sim:
            plt.plot(PSD_residual_mode_4_freq_t_1st_RL_sim, np.cumsum(PSD_residual_mode_4_1st_RL_sim), ':', color="indianred", label=f"PSD_mode_{mode_4}_1st_int ({label_int_sim})")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_RL_sim, np.cumsum(PSD_residual_mode_4_2nd_RL_sim), ':', color="red", label=f"PSD_mode_{mode_4}_2nd_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(PSD_residual_mode_4_freq_t_2nd_int, np.cumsum(PSD_residual_mode_4_2nd_int), color="blue", label=f"PSD_mode_{mode_4}_2nd_{label_int}")
        if plot_sim:
            plt.plot(PSD_residual_mode_4_freq_t_2nd_int_sim, np.cumsum(PSD_residual_mode_4_2nd_int_sim), ':', color="blue", label=f"PSD_mode_{mode_4}_2nd_{label_int_sim}")
    if ideal:
        plt.plot(PSD_residual_mode_4_freq_t_2nd_ideal, np.cumsum(PSD_residual_mode_4_2nd_ideal), label=f"PSD_mode_{mode_4}_2nd_{label_ideal}")

    plt.title(f"Cumulative PSD KL mode {mode_4}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("cumulative PSD (nm²)")


    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(PSD_residual_mode_5_freq_t_1st_RL, np.cumsum(PSD_residual_mode_5_1st_RL), '--', color="indianred", label=f"PSD_mode_{mode_5}_1st_int")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_RL, np.cumsum(PSD_residual_mode_5_2nd_RL), color="red", label=f"PSD_mode_{mode_5}_2nd_{label_RL}")
        if plot_sim:
            plt.plot(PSD_residual_mode_5_freq_t_1st_RL_sim, np.cumsum(PSD_residual_mode_5_1st_RL_sim), ':', color="indianred", label=f"PSD_mode_{mode_5}_1st_int ({label_int_sim})")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_RL_sim, np.cumsum(PSD_residual_mode_5_2nd_RL_sim), ':', color="red", label=f"PSD_mode_{mode_5}_2nd_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(PSD_residual_mode_5_freq_t_2nd_int, np.cumsum(PSD_residual_mode_5_2nd_int), color="blue", label=f"PSD_mode_{mode_5}_2nd_{label_int}")
        if plot_sim:
            plt.plot(PSD_residual_mode_5_freq_t_2nd_int_sim, np.cumsum(PSD_residual_mode_5_2nd_int_sim), ':', color="blue", label=f"PSD_mode_{mode_5}_2nd_{label_int_sim}")
    if ideal:
        plt.plot(PSD_residual_mode_5_freq_t_2nd_ideal, np.cumsum(PSD_residual_mode_5_2nd_ideal), label=f"PSD_mode_{mode_5}_2nd_{label_ideal}")

    plt.title(f"Cumulative PSD KL mode {mode_5}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("cumulative PSD (nm²)")







# --------------------------------------------------- Ratio ---------------------------------------------------#

if RL:
    PSD_mode_1_ratio_RL = PSD_residual_mode_1_1st_RL/PSD_residual_mode_1_2nd_RL
    PSD_mode_2_ratio_RL = PSD_residual_mode_2_1st_RL/PSD_residual_mode_2_2nd_RL
    PSD_mode_3_ratio_RL = PSD_residual_mode_3_1st_RL/PSD_residual_mode_3_2nd_RL
    PSD_mode_4_ratio_RL = PSD_residual_mode_4_1st_RL/PSD_residual_mode_4_2nd_RL
    PSD_mode_5_ratio_RL = PSD_residual_mode_5_1st_RL/PSD_residual_mode_5_2nd_RL

if integrator:
    PSD_mode_1_ratio_int = PSD_residual_mode_1_1st_RL/PSD_residual_mode_1_2nd_int
    PSD_mode_2_ratio_int = PSD_residual_mode_2_1st_RL/PSD_residual_mode_2_2nd_int
    PSD_mode_3_ratio_int = PSD_residual_mode_3_1st_RL/PSD_residual_mode_3_2nd_int
    PSD_mode_4_ratio_int = PSD_residual_mode_4_1st_RL/PSD_residual_mode_4_2nd_int
    PSD_mode_5_ratio_int = PSD_residual_mode_5_1st_RL/PSD_residual_mode_5_2nd_int

if RL and plot_sim:
    PSD_mode_1_ratio_RL_sim = PSD_residual_mode_1_1st_RL_sim/PSD_residual_mode_1_2nd_RL_sim
    PSD_mode_2_ratio_RL_sim = PSD_residual_mode_2_1st_RL_sim/PSD_residual_mode_2_2nd_RL_sim
    PSD_mode_3_ratio_RL_sim = PSD_residual_mode_3_1st_RL_sim/PSD_residual_mode_3_2nd_RL_sim
    PSD_mode_4_ratio_RL_sim = PSD_residual_mode_4_1st_RL_sim/PSD_residual_mode_4_2nd_RL_sim
    PSD_mode_5_ratio_RL_sim = PSD_residual_mode_5_1st_RL_sim/PSD_residual_mode_5_2nd_RL_sim

if integrator and plot_sim:
    PSD_mode_1_ratio_int_sim = PSD_residual_mode_1_1st_RL_sim/PSD_residual_mode_1_2nd_int_sim
    PSD_mode_2_ratio_int_sim = PSD_residual_mode_2_1st_RL_sim/PSD_residual_mode_2_2nd_int_sim
    PSD_mode_3_ratio_int_sim = PSD_residual_mode_3_1st_RL_sim/PSD_residual_mode_3_2nd_int_sim
    PSD_mode_4_ratio_int_sim = PSD_residual_mode_4_1st_RL_sim/PSD_residual_mode_4_2nd_int_sim
    PSD_mode_5_ratio_int_sim = PSD_residual_mode_5_1st_RL_sim/PSD_residual_mode_5_2nd_int_sim

if ideal:
    PSD_mode_1_ratio_ideal = PSD_residual_mode_1_1st_ideal/PSD_residual_mode_1_2nd_ideal
    PSD_mode_2_ratio_ideal = PSD_residual_mode_2_1st_ideal/PSD_residual_mode_2_2nd_ideal
    PSD_mode_3_ratio_ideal = PSD_residual_mode_3_1st_ideal/PSD_residual_mode_3_2nd_ideal
    PSD_mode_4_ratio_ideal = PSD_residual_mode_4_1st_ideal/PSD_residual_mode_4_2nd_ideal
    PSD_mode_5_ratio_ideal = PSD_residual_mode_5_1st_ideal/PSD_residual_mode_5_2nd_ideal



plt.figure()
if RL:
    if plot_onsky:
        plt.plot(PSD_residual_mode_1_freq_t_2nd_RL, PSD_mode_1_ratio_RL, color="red", label=f"PSD_mode_{mode_1}_2nd_{label_RL}")
    if plot_sim:
        plt.plot(PSD_residual_mode_1_freq_t_2nd_RL_sim, PSD_mode_1_ratio_RL_sim, ':', color="red", label=f"PSD_mode_{mode_1}_2nd_{label_int_sim}")
if integrator:
    if plot_onsky:
        plt.plot(PSD_residual_mode_1_freq_t_2nd_int, PSD_mode_1_ratio_int, color="blue", label=f"PSD_mode_{mode_1}_2nd_{label_int}")
    if plot_sim:
        plt.plot(PSD_residual_mode_1_freq_t_2nd_int_sim, PSD_mode_1_ratio_int_sim, ':', color="blue", label=f"PSD_mode_{mode_1}_2nd_{label_int_sim}")
if ideal:
    plt.plot(PSD_residual_mode_1_freq_t_2nd_ideal, PSD_mode_1_ratio_ideal, label=f"PSD_mode_{mode_1}_2nd_{label_ideal}")
plt.title(f"Gain of 2nd stage over 1st stage {mode_1}, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("1st / 2nd stage PSD ratio")



plt.figure()
if RL:
    if plot_onsky:
        plt.plot(PSD_residual_mode_2_freq_t_2nd_RL, PSD_mode_2_ratio_RL, color="red", label=f"PSD_mode_{mode_2}_2nd_{label_RL}")
    if plot_sim:
        plt.plot(PSD_residual_mode_2_freq_t_2nd_RL_sim, PSD_mode_2_ratio_RL_sim, ':', color="red", label=f"PSD_mode_{mode_2}_2nd_{label_RL_sim}")
if integrator:
    if plot_onsky:
        plt.plot(PSD_residual_mode_2_freq_t_2nd_int, PSD_mode_2_ratio_int, color="blue", label=f"PSD_mode_{mode_2}_2nd_{label_int}")
    if plot_sim:
        plt.plot(PSD_residual_mode_2_freq_t_2nd_int_sim, PSD_mode_2_ratio_int_sim, ':', color="blue", label=f"PSD_mode_{mode_2}_2nd_{label_int_sim}")
if ideal:
    plt.plot(PSD_residual_mode_2_freq_t_2nd_ideal, PSD_mode_2_ratio_ideal, label=f"PSD_mode_{mode_2}_2nd_{label_ideal}")
plt.title(f"Gain of 2nd stage over 1st stage {mode_2}, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("1st / 2nd stage PSD ratio")



plt.figure()
if RL:
    if plot_onsky:
        plt.plot(PSD_residual_mode_3_freq_t_2nd_RL, PSD_mode_3_ratio_RL, color="red", label=f"PSD_mode_{mode_3}_2nd_{label_RL}")
    if plot_sim:
        plt.plot(PSD_residual_mode_3_freq_t_2nd_RL_sim, PSD_mode_3_ratio_RL_sim, ':', color="red", label=f"PSD_mode_{mode_3}_2nd_{label_RL_sim}")
if integrator:
    if plot_onsky:
        plt.plot(PSD_residual_mode_3_freq_t_2nd_int, PSD_mode_3_ratio_int, color="blue", label=f"PSD_mode_{mode_3}_2nd_{label_int}")
    if plot_sim:
        plt.plot(PSD_residual_mode_3_freq_t_2nd_int_sim, PSD_mode_3_ratio_int_sim, ':', color="blue", label=f"PSD_mode_{mode_3}_2nd_{label_int_sim}")
if ideal:
    plt.plot(PSD_residual_mode_3_freq_t_2nd_ideal, PSD_mode_3_ratio_ideal, label=f"PSD_mode_{mode_3}_2nd_{label_ideal}")
plt.title(f"Gain of 2nd stage over 1st stage {mode_3}, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("1st / 2nd stage PSD ratio")





plt.figure()
if RL:
    if plot_onsky:
        plt.plot(PSD_residual_mode_4_freq_t_2nd_RL, PSD_mode_4_ratio_RL, color="red", label=f"PSD_mode_{mode_4}_2nd_{label_RL}")
    if plot_sim:
        plt.plot(PSD_residual_mode_4_freq_t_2nd_RL_sim, PSD_mode_4_ratio_RL_sim, ':', color="red", label=f"PSD_mode_{mode_4}_2nd_{label_RL_sim}")
if integrator:
    if plot_onsky:
        plt.plot(PSD_residual_mode_4_freq_t_2nd_int, PSD_mode_4_ratio_int, color="blue", label=f"PSD_mode_{mode_4}_2nd_{label_int}")
    if plot_sim:
        plt.plot(PSD_residual_mode_4_freq_t_2nd_int_sim, PSD_mode_4_ratio_int_sim, ':', color="blue", label=f"PSD_mode_{mode_4}_2nd_{label_int_sim}")
if ideal:
    plt.plot(PSD_residual_mode_4_freq_t_2nd_ideal, PSD_mode_4_ratio_ideal, label=f"PSD_mode_{mode_4}_2nd_{label_ideal}")
plt.title(f"Gain of 2nd stage over 1st stage {mode_4}, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("1st / 2nd stage PSD ratio")



plt.figure()
if RL:
    if plot_onsky:
        plt.plot(PSD_residual_mode_5_freq_t_2nd_RL, PSD_mode_5_ratio_RL, color="red", label=f"PSD_mode_{mode_5}_2nd_{label_RL}")
    if plot_sim:
        plt.plot(PSD_residual_mode_5_freq_t_2nd_RL_sim, PSD_mode_5_ratio_RL_sim, ':', color="red", label=f"PSD_mode_{mode_5}_2nd_{label_RL_sim}")
if integrator:
    if plot_onsky:
        plt.plot(PSD_residual_mode_5_freq_t_2nd_int, PSD_mode_5_ratio_int, color="blue", label=f"PSD_mode_{mode_5}_2nd_{label_int}")
    if plot_sim:
        plt.plot(PSD_residual_mode_5_freq_t_2nd_int_sim, PSD_mode_5_ratio_int_sim, ':', color="blue", label=f"PSD_mode_{mode_5}_2nd_{label_int_sim}")
if ideal:
    plt.plot(PSD_residual_mode_5_freq_t_2nd_ideal, PSD_mode_5_ratio_ideal, label=f"PSD_mode_{mode_5}_2nd_{label_ideal}")
plt.title(f"Gain of 2nd stage over 1st stage {mode_5}, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("1st / 2nd stage PSD ratio")


plt.show()


# ---------------------------------------------------temporal Error transfer function---------------------------------------------------#

#to select for the common frequencies since the 1st stage OL was running at 200Hz on sky
idx_CL = np.isin(np.round(PSD_residual_mode_1_freq_t_2nd_RL, 6), np.round(PSD_atm_mode_1_freq_t, 6))
idx_atm = np.isin(np.round(PSD_atm_mode_1_freq_t, 6), np.round(PSD_residual_mode_1_freq_t_2nd_RL, 6))
f2_selected  = PSD_residual_mode_1_freq_t_2nd_RL[idx_CL]
psd2_selected = PSD_residual_mode_1_2nd_RL[idx_CL]

if RL and plot_sim:
    idx_CL_sim  = np.isin(np.round(PSD_residual_mode_1_freq_t_2nd_RL_sim, 6), np.round(PSD_atm_mode_1_freq_t_sim, 6))
    idx_atm_sim = np.isin(np.round(PSD_atm_mode_1_freq_t_sim, 6), np.round(PSD_residual_mode_1_freq_t_2nd_RL_sim, 6))

if plot_tPSD:
    if RL:
        if plot_onsky:
            tETF_mode_1_1st_RL = PSD_residual_mode_1_1st_RL[idx_CL] / PSD_atm_mode_1[idx_atm]
            tETF_mode_2_1st_RL = PSD_residual_mode_2_1st_RL[idx_CL] / PSD_atm_mode_2[idx_atm]
            tETF_mode_3_1st_RL = PSD_residual_mode_3_1st_RL[idx_CL] / PSD_atm_mode_3[idx_atm]
            tETF_mode_4_1st_RL = PSD_residual_mode_4_1st_RL[idx_CL] / PSD_atm_mode_4[idx_atm]
            tETF_mode_5_1st_RL = PSD_residual_mode_5_1st_RL[idx_CL] / PSD_atm_mode_5[idx_atm]

            tETF_mode_1_2nd_RL = PSD_residual_mode_1_2nd_RL[idx_CL] / PSD_atm_mode_1[idx_atm]
            tETF_mode_2_2nd_RL = PSD_residual_mode_2_2nd_RL[idx_CL] / PSD_atm_mode_2[idx_atm]
            tETF_mode_3_2nd_RL = PSD_residual_mode_3_2nd_RL[idx_CL] / PSD_atm_mode_3[idx_atm]
            tETF_mode_4_2nd_RL = PSD_residual_mode_4_2nd_RL[idx_CL] / PSD_atm_mode_4[idx_atm]
            tETF_mode_5_2nd_RL = PSD_residual_mode_5_2nd_RL[idx_CL] / PSD_atm_mode_5[idx_atm]

        if plot_sim:
            tETF_mode_1_1st_RL_sim = PSD_residual_mode_1_1st_RL_sim[idx_CL_sim] / PSD_atm_mode_1_sim[idx_atm_sim]
            tETF_mode_2_1st_RL_sim = PSD_residual_mode_2_1st_RL_sim[idx_CL_sim] / PSD_atm_mode_2_sim[idx_atm_sim]
            tETF_mode_3_1st_RL_sim = PSD_residual_mode_3_1st_RL_sim[idx_CL_sim] / PSD_atm_mode_3_sim[idx_atm_sim]
            tETF_mode_4_1st_RL_sim = PSD_residual_mode_4_1st_RL_sim[idx_CL_sim] / PSD_atm_mode_4_sim[idx_atm_sim]
            tETF_mode_5_1st_RL_sim = PSD_residual_mode_5_1st_RL_sim[idx_CL_sim] / PSD_atm_mode_5_sim[idx_atm_sim]

            tETF_mode_1_2nd_RL_sim = PSD_residual_mode_1_2nd_RL_sim[idx_CL_sim] / PSD_atm_mode_1_sim[idx_atm_sim]
            tETF_mode_2_2nd_RL_sim = PSD_residual_mode_2_2nd_RL_sim[idx_CL_sim] / PSD_atm_mode_2_sim[idx_atm_sim]
            tETF_mode_3_2nd_RL_sim = PSD_residual_mode_3_2nd_RL_sim[idx_CL_sim] / PSD_atm_mode_3_sim[idx_atm_sim]
            tETF_mode_4_2nd_RL_sim = PSD_residual_mode_4_2nd_RL_sim[idx_CL_sim] / PSD_atm_mode_4_sim[idx_atm_sim]
            tETF_mode_5_2nd_RL_sim = PSD_residual_mode_5_2nd_RL_sim[idx_CL_sim] / PSD_atm_mode_5_sim[idx_atm_sim]


    if integrator:
        if plot_onsky:
            tETF_mode_1_2nd_int = PSD_residual_mode_1_2nd_int[idx_CL] / PSD_atm_mode_1[idx_atm]
            tETF_mode_2_2nd_int = PSD_residual_mode_2_2nd_int[idx_CL] / PSD_atm_mode_2[idx_atm]
            tETF_mode_3_2nd_int = PSD_residual_mode_3_2nd_int[idx_CL] / PSD_atm_mode_3[idx_atm]
            tETF_mode_4_2nd_int = PSD_residual_mode_4_2nd_int[idx_CL] / PSD_atm_mode_4[idx_atm]
            tETF_mode_5_2nd_int = PSD_residual_mode_5_2nd_int[idx_CL] / PSD_atm_mode_5[idx_atm]

        if plot_sim:
            tETF_mode_1_2nd_int_sim = PSD_residual_mode_1_2nd_int_sim[idx_CL_sim] / PSD_atm_mode_1_sim[idx_atm_sim]
            tETF_mode_2_2nd_int_sim = PSD_residual_mode_2_2nd_int_sim[idx_CL_sim] / PSD_atm_mode_2_sim[idx_atm_sim]
            tETF_mode_3_2nd_int_sim = PSD_residual_mode_3_2nd_int_sim[idx_CL_sim] / PSD_atm_mode_3_sim[idx_atm_sim]
            tETF_mode_4_2nd_int_sim = PSD_residual_mode_4_2nd_int_sim[idx_CL_sim] / PSD_atm_mode_4_sim[idx_atm_sim]
            tETF_mode_5_2nd_int_sim = PSD_residual_mode_5_2nd_int_sim[idx_CL_sim] / PSD_atm_mode_5_sim[idx_atm_sim]


    if ideal:
        tETF_mode_1_1st_ideal = PSD_residual_mode_1_1st_ideal[idx_CL] / PSD_atm_mode_1[idx_atm]
        tETF_mode_2_1st_ideal = PSD_residual_mode_2_1st_ideal[idx_CL] / PSD_atm_mode_2[idx_atm]
        tETF_mode_3_1st_ideal = PSD_residual_mode_3_1st_ideal[idx_CL] / PSD_atm_mode_3[idx_atm]
        tETF_mode_4_1st_ideal = PSD_residual_mode_4_1st_ideal[idx_CL] / PSD_atm_mode_4[idx_atm]
        tETF_mode_5_1st_ideal = PSD_residual_mode_5_1st_ideal[idx_CL] / PSD_atm_mode_5[idx_atm]

        tETF_mode_1_2nd_ideal = PSD_residual_mode_1_2nd_ideal[idx_CL] / PSD_atm_mode_1[idx_atm]
        tETF_mode_2_2nd_ideal = PSD_residual_mode_2_2nd_ideal[idx_CL] / PSD_atm_mode_2[idx_atm]
        tETF_mode_3_2nd_ideal = PSD_residual_mode_3_2nd_ideal[idx_CL] / PSD_atm_mode_3[idx_atm]
        tETF_mode_4_2nd_ideal = PSD_residual_mode_4_2nd_ideal[idx_CL] / PSD_atm_mode_4[idx_atm]
        tETF_mode_5_2nd_ideal = PSD_residual_mode_5_2nd_ideal[idx_CL] / PSD_atm_mode_5[idx_atm]



    #tip
    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(PSD_residual_mode_1_freq_t_1st_RL[idx_CL], tETF_mode_1_1st_RL, '--', color="indianred", label=f"ETF mode_{mode_1}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_RL[idx_CL], tETF_mode_1_2nd_RL, color="red", label=f"ETF mode_{mode_1}_2nd_{label_RL}")
        if plot_sim:
            plt.plot(PSD_residual_mode_1_freq_t_1st_RL_sim[idx_CL_sim], tETF_mode_1_1st_RL_sim, ':', color="indianred", label=f"ETF mode_{mode_1}_1st_{label_int_sim}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_RL_sim[idx_CL_sim], tETF_mode_1_2nd_RL_sim, ':', color="red", label=f"ETF mode_{mode_1}_2nd_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(PSD_residual_mode_1_freq_t_2nd_int[idx_CL], tETF_mode_1_2nd_int, color="blue", label=f"ETF mode_{mode_1}_2nd_{label_int}")
        if plot_sim:
            plt.plot(PSD_residual_mode_1_freq_t_2nd_int_sim[idx_CL_sim], tETF_mode_1_2nd_int_sim, ':', color="blue", label=f"ETF mode_{mode_1}_2nd_{label_int_sim}")
    if ideal:
        plt.plot(PSD_residual_mode_1_freq_t_1st_ideal[idx_CL], tETF_mode_1_1st_ideal, '--', label=f"ETF mode_{mode_1}_1st_{label_ideal}")
        plt.plot(PSD_residual_mode_1_freq_t_2nd_ideal[idx_CL], tETF_mode_1_2nd_ideal, label=f"ETF mode_{mode_1}_2nd_{label_ideal}")
    plt.title("temporal error transfer functions")
    plt.ylabel("tETF")
    plt.xlabel("frequency (Hz)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()


    #tilt
    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(PSD_residual_mode_2_freq_t_1st_RL[idx_CL], tETF_mode_2_1st_RL, '--', color="indianred", label=f"ETF mode_{mode_2}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_RL[idx_CL], tETF_mode_2_2nd_RL, color="red", label=f"ETF mode_{mode_2}_2nd_{label_RL}")
        if plot_sim:
            plt.plot(PSD_residual_mode_2_freq_t_1st_RL_sim[idx_CL_sim], tETF_mode_2_1st_RL_sim, ':', color="indianred", label=f"ETF mode_{mode_2}_1st_{label_int_sim}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_RL_sim[idx_CL_sim], tETF_mode_2_2nd_RL_sim, ':', color="red", label=f"ETF mode_{mode_2}_2nd_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(PSD_residual_mode_2_freq_t_2nd_int[idx_CL], tETF_mode_2_2nd_int, color="blue", label=f"ETF mode_{mode_2}_2nd_{label_int}")
        if plot_sim:
            plt.plot(PSD_residual_mode_2_freq_t_2nd_int_sim[idx_CL_sim], tETF_mode_2_2nd_int_sim, ':', color="blue", label=f"ETF mode_{mode_2}_2nd_{label_int_sim}")
    if ideal:
        plt.plot(PSD_residual_mode_2_freq_t_1st_ideal[idx_CL], tETF_mode_2_1st_ideal, '--', label=f"ETF mode_{mode_2}_1st_{label_ideal}")
        plt.plot(PSD_residual_mode_2_freq_t_2nd_ideal[idx_CL], tETF_mode_2_2nd_ideal, label=f"ETF mode_{mode_2}_2nd_{label_ideal}")
    plt.title("temporal error transfer functions")
    plt.ylabel("tETF")
    plt.xlabel("frequency (Hz)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()

    #100
    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(PSD_residual_mode_3_freq_t_1st_RL[idx_CL], tETF_mode_3_1st_RL, '--', color="indianred", label=f"ETF mode_{mode_3}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_RL[idx_CL], tETF_mode_3_2nd_RL, color="red", label=f"ETF mode_{mode_3}_2nd_{label_RL}")
        if plot_sim:
            plt.plot(PSD_residual_mode_3_freq_t_1st_RL_sim[idx_CL_sim], tETF_mode_3_1st_RL_sim, ':', color="indianred", label=f"ETF mode_{mode_3}_1st_{label_int_sim}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_RL_sim[idx_CL_sim], tETF_mode_3_2nd_RL_sim, ':', color="red", label=f"ETF mode_{mode_3}_2nd_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(PSD_residual_mode_3_freq_t_2nd_int[idx_CL], tETF_mode_3_2nd_int, color="blue", label=f"ETF mode_{mode_3}_2nd_{label_int}")
        if plot_sim:
            plt.plot(PSD_residual_mode_3_freq_t_2nd_int_sim[idx_CL_sim], tETF_mode_3_2nd_int_sim, ':', color="blue", label=f"ETF mode_{mode_3}_2nd_{label_int_sim}")
    if ideal:
        plt.plot(PSD_residual_mode_3_freq_t_1st_ideal[idx_CL], tETF_mode_3_1st_ideal, '--', label=f"ETF mode_{mode_3}_1st_{label_ideal}")
        plt.plot(PSD_residual_mode_3_freq_t_2nd_ideal[idx_CL], tETF_mode_3_2nd_ideal, label=f"ETF mode_{mode_3}_2nd_{label_ideal}")
    plt.title("temporal error transfer functions")
    plt.ylabel("tETF")
    plt.xlabel("frequency (Hz)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()

    #200
    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(PSD_residual_mode_4_freq_t_1st_RL[idx_CL], tETF_mode_4_1st_RL, '--', color="indianred", label=f"ETF mode_{mode_4}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_RL[idx_CL], tETF_mode_4_2nd_RL, color="red", label=f"ETF mode_{mode_4}_2nd_{label_RL}")
        if plot_sim:
            plt.plot(PSD_residual_mode_4_freq_t_1st_RL_sim[idx_CL_sim], tETF_mode_4_1st_RL_sim, ':', color="indianred", label=f"ETF mode_{mode_4}_1st_{label_int_sim}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_RL_sim[idx_CL_sim], tETF_mode_4_2nd_RL_sim, ':', color="red", label=f"ETF mode_{mode_4}_2nd_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(PSD_residual_mode_4_freq_t_2nd_int[idx_CL], tETF_mode_4_2nd_int, color="blue", label=f"ETF mode_{mode_4}_2nd_{label_int}")
        if plot_sim:
            plt.plot(PSD_residual_mode_4_freq_t_2nd_int_sim[idx_CL_sim], tETF_mode_4_2nd_int_sim, ':', color="blue", label=f"ETF mode_{mode_4}_2nd_{label_int_sim}")
    if ideal:
        plt.plot(PSD_residual_mode_4_freq_t_1st_ideal[idx_CL], tETF_mode_4_1st_ideal, '--', label=f"ETF mode_{mode_4}_1st_{label_ideal}")
        plt.plot(PSD_residual_mode_4_freq_t_2nd_ideal[idx_CL], tETF_mode_4_2nd_ideal, label=f"ETF mode_{mode_4}_2nd_{label_ideal}")
    plt.title("temporal error transfer functions")
    plt.ylabel("tETF")
    plt.xlabel("frequency (Hz)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()


    #5
    plt.figure()
    if RL:
        if plot_onsky:
            plt.plot(PSD_residual_mode_5_freq_t_1st_RL[idx_CL], tETF_mode_5_1st_RL, '--', color="indianred", label=f"ETF mode_{mode_5}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_RL[idx_CL], tETF_mode_5_2nd_RL, color="red", label=f"ETF mode_{mode_5}_2nd_{label_RL}")
        if plot_sim:
            plt.plot(PSD_residual_mode_5_freq_t_1st_RL_sim[idx_CL_sim], tETF_mode_5_1st_RL_sim, ':', color="indianred", label=f"ETF mode_{mode_5}_1st_{label_int_sim}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_RL_sim[idx_CL_sim], tETF_mode_5_2nd_RL_sim, ':', color="red", label=f"ETF mode_{mode_5}_2nd_{label_RL_sim}")
    if integrator:
        if plot_onsky:
            plt.plot(PSD_residual_mode_5_freq_t_2nd_int[idx_CL], tETF_mode_5_2nd_int, color="blue", label=f"ETF mode_{mode_5}_2nd_{label_int}")
        if plot_sim:
            plt.plot(PSD_residual_mode_5_freq_t_2nd_int_sim[idx_CL_sim], tETF_mode_5_2nd_int_sim, ':', color="blue", label=f"ETF mode_{mode_5}_2nd_{label_int_sim}")
    if ideal:
        plt.plot(PSD_residual_mode_5_freq_t_1st_ideal[idx_CL], tETF_mode_5_1st_ideal, '--', label=f"ETF mode_{mode_5}_1st_{label_ideal}")
        plt.plot(PSD_residual_mode_5_freq_t_2nd_ideal[idx_CL], tETF_mode_5_2nd_ideal, label=f"ETF mode_{mode_5}_2nd_{label_ideal}")
    plt.title("temporal error transfer functions")
    plt.ylabel("tETF")
    plt.xlabel("frequency (Hz)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()




plt.show()





























