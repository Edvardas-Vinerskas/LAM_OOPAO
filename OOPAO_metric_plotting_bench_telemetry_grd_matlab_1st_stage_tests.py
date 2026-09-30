"""
1st stage tests: compares on-sky data with 4 simulations (KL mode variance + temporal PSDs only)

#   atm reconstruction from 1st stage DM shape (on sky: dmCmdCube, sim: results_1st_stage_*.npz 'dm_commands')
#   1st stage residual reconstruction from 2nd stage CNN reconstruction in DM space (CL1OL2)

Plotting style follows OOPAO_metric_plotting_bench_telemetry_grd_matlab.py:
    on sky     -> atm solid black, 1st stage dashed ('--')
    simulation -> dotted (':'), one shade of the same colour per simulation (dark -> light)

Figures are saved at save_dpi into save_dir (nothing is shown)
Run from the LAM_OOPAO folder
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: figures are saved to disk instead of shown
import matplotlib.pyplot as plt
from functions import *

save_dir = 'PAPYRIIS_2stage_CNN_RL/2026-09-23/figures_1st_stage_tests_noise_check'   # folder where all figures are saved
save_dpi = 200
os.makedirs(save_dir, exist_ok=True)

plot_onsky    = True   # plot on-sky results (solid/dashed lines)
plot_sim      = True   # plot simulation results (dotted lines)
plot_atm      = True   # atmosphere from 1st stage DM commands
plot_residual = True   # 1st stage residual from CL1OL2

frequency_1st = 400    # sampling frequency of the 1st stage residual (CL1OL2, measured by the 2nd stage) and of the sim DM commands
frequency_atm = 200    # sampling frequency of the on-sky 1st stage DM commands
CL_gain_pyr   = 0.3
KL_frames     = [1, -1]              # [start, end] indices for the timeseries window
psd_modes     = [0, 10, 20, 30, 40]  # KL mode indices to plot PSDs for

# on sky data
loaddir = 'bench_sky_04_15/arcturus_v7'
RL_telemetry_1st = np.load(f'{loaddir}/2026-04-16T01_19_12_telemetry_data_RLiter50.npy', allow_pickle = True)
CL1st_OL_2nd     = np.load(f'{loaddir}/2026-04-16T01_27_23_telemetry_2nd_data_CL1OL2_noCRED2.npy', allow_pickle = True)

# simulations to compare: label, folder, 1st stage results file, 2nd stage CL1OL2 results file (same folder)
simdir = "PAPYRIIS_2stage_CNN_RL/2026-09-23"
simulations = [
    {"label": "noise", "folder": f"{simdir}/PAPYRIIS_arcturus_v4", "file_1st": "results_1st_stage_r0_0.100_V0_3.002_L0_30.000_tboil_5.000_multi_layer.npz", "file_2nd": "results_2nd_stage_CL1OL2_r0_0.100_V0_3.002_L0_30.000_tboil_5.000.npz"},
    {"label": "no_dark_cur", "folder": f"{simdir}/PAPYRIIS_arcturus_v4_nodarkcurr_cred", "file_1st": "results_1st_stage_r0_0.100_V0_3.002_L0_30.000_tboil_5.000_multi_layer.npz", "file_2nd": "results_2nd_stage_CL1OL2_r0_0.100_V0_3.002_L0_30.000_tboil_5.000.npz"},
    {"label": "no_photon", "folder": f"{simdir}/PAPYRIIS_arcturus_v4_nophoton_cred", "file_1st": "results_1st_stage_r0_0.100_V0_3.002_L0_30.000_tboil_5.000_multi_layer.npz", "file_2nd": "results_2nd_stage_CL1OL2_r0_0.100_V0_3.002_L0_30.000_tboil_5.000.npz"},
    {"label": "no_noise_cred", "folder": f"{simdir}/PAPYRIIS_arcturus_v4_nonoisecred", "file_1st": "results_1st_stage_r0_0.100_V0_3.002_L0_30.000_tboil_5.000_multi_layer.npz", "file_2nd": "results_2nd_stage_CL1OL2_r0_0.100_V0_3.002_L0_30.000_tboil_5.000.npz"},
]

# colours as in the original file (on sky), sims get a dark -> light gradient of the same colour
color_atm           = "black"
color_1st_variance  = "cornflowerblue"
color_1st_psd       = "indianred"
def sim_shades(cmap_name, n):
    return [plt.get_cmap(cmap_name)(x) for x in np.linspace(0.9, 0.45, n)]
shades_atm          = sim_shades("Greys", len(simulations))
shades_1st_variance = sim_shades("Blues", len(simulations))
shades_1st_psd      = sim_shades("Reds",  len(simulations))


#TODO these don't have the representative masks (masking done as per PAPYRIIS_main.py or just look inside the 2 below functions)
dm_1st_inf, pupil_mask_1 = first_stage_dm_builder()
dm_2nd_inf, pupil_mask_2 = second_stage_dm_builder()

dm_1st_inf = dm_1st_inf.reshape(100, 100, 241)
dm_1st_inf_masked = dm_1st_inf[pupil_mask_1, :]

dm_2nd_inf = dm_2nd_inf.reshape(100, 100, 97)
dm_2nd_inf_masked = dm_2nd_inf[pupil_mask_2, :]

M2C_1st = - np.load('PAPYRIIS_2stage_CNN_RL/M2C_1rst.npy')
M2C_2nd = np.load('PAPYRIIS_2stage_CNN_RL/M2C_2nd.npy')

C2M_2nd = np.linalg.pinv(M2C_2nd)
M2P_1st = dm_1st_inf_masked @ M2C_1st
projector_kl_2nd_to_1st = np.linalg.pinv(dm_1st_inf_masked @ M2C_1st) @ (dm_2nd_inf_masked @ M2C_2nd)

# M2P_1st scaled to nm before computing covariance
GG_diag = np.diag(mode_covariance(M2P_1st * 1e9))


def KL_variance_calculator(modes, GG_diag, frame_bounds):
    """
    Calculate variance of KL modes in nm^2.

    modes shape: (timeseries length, no_of_modes)
    GG_diag: precomputed diagonal of mode_covariance(m2p_nm), where m2p_nm = dm_inf_masked * 1e9 @ m2c
    """
    modes_masked = modes[frame_bounds[0]:frame_bounds[1], :]
    modes_var = np.var(np.asarray(modes_masked), axis=0) * GG_diag
    return modes_var, modes_masked


# ---------------------------------------------------Load data---------------------------------------------------#
if plot_onsky:
    next_states_1st = CL1st_OL_2nd.item()['slavedreconsCube'].squeeze() #1st stage residual as measured by 2nd stage
    modes_1st_stage = next_states_1st @ C2M_2nd.T @ projector_kl_2nd_to_1st.T
    coefs_var_1st_stage, modes_1st_stage_masked = KL_variance_calculator(modes_1st_stage, GG_diag, KL_frames)

    modes_atm = mode_calculator_fromDM(RL_telemetry_1st.item()['dmCmdCube'].squeeze(), M2C_1st)
    coefs_var_atm, modes_atm_masked = KL_variance_calculator(modes_atm, GG_diag, KL_frames)

if plot_sim:
    for sim in simulations:
        print(f"loading {sim['folder']}")
        if plot_atm:
            dm_atm_coefs = np.load(f"{sim['folder']}/{sim['file_1st']}")['dm_commands']  # only this array is read from the npz
            sim["modes_atm"] = mode_calculator_fromDM(dm_atm_coefs, M2C_1st)
            sim["var_atm"], sim["modes_atm_masked"] = KL_variance_calculator(sim["modes_atm"], GG_diag, KL_frames)

        if plot_residual:
            path_2nd = f"{sim['folder']}/{sim['file_2nd']}"
            if not os.path.exists(path_2nd):  # simulation may still be running: skip its residual curves
                print(f"missing {path_2nd}, skipping 1st stage residual for {sim['label']}")
                continue
            next_states_1st_sim = np.load(path_2nd)['all_reconstructed_cmd'].squeeze()
            sim["modes_1st"] = next_states_1st_sim @ C2M_2nd.T @ projector_kl_2nd_to_1st.T
            sim["var_1st"], sim["modes_1st_masked"] = KL_variance_calculator(sim["modes_1st"], GG_diag, KL_frames)


# ---------------------------------------------------KL mode variance---------------------------------------------------#
plt.figure(figsize=(19, 8))  # wider than the original (12, 8) to fit the legend on the right
if plot_atm:
    if plot_onsky:
        plt.plot(coefs_var_atm, color=color_atm, label=f"Atm, σ² = {np.sum(coefs_var_atm):,.0f} nm²")
    if plot_sim:
        for sim, color in zip(simulations, shades_atm):
            plt.plot(sim["var_atm"], ':', color=color, label=f"Atm (sim, {sim['label']}), σ² = {np.sum(sim['var_atm']):,.0f} nm²")
if plot_residual:
    if plot_onsky:
        plt.plot(coefs_var_1st_stage, '--', color=color_1st_variance, lw=2.5, label=f"1st stage (integrator), σ² = {np.sum(coefs_var_1st_stage):,.0f} nm²")
    if plot_sim:
        for sim, color in zip(simulations, shades_1st_variance):
            if "var_1st" not in sim:
                continue
            plt.plot(sim["var_1st"], ':', color=color, lw=2.5, label=f"1st stage (sim, {sim['label']}), σ² = {np.sum(sim['var_1st']):,.0f} nm²")

plt.title(f"KL mode variance", fontsize=30)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("KL mode index", fontsize=24)
plt.ylabel("Temporal residual variance (nm²)", fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1.01, 1))  # outside the axes so it does not cover the curves
plt.tight_layout()
plt.savefig(f"{save_dir}/KL_mode_variance.png", dpi=save_dpi, bbox_inches='tight')
plt.close()


# ---------------------------------------------------Temporal PSD---------------------------------------------------#
for mode in psd_modes:
    plt.figure(figsize=(19, 8))  # wider than the original (12, 8) to fit the legend on the right
    if plot_residual:
        if plot_onsky:
            f_1st, psd_1st, _ = tPSD_calculator(modes_1st_stage_masked, mode, GG_diag, frequency_1st)
            plt.plot(f_1st, psd_1st, '--', color=color_1st_psd, lw=2.5, label="1st stage integrator")
        if plot_sim:
            for sim, color in zip(simulations, shades_1st_psd):
                if "modes_1st_masked" not in sim:
                    continue
                f_1st, psd_1st, _ = tPSD_calculator(sim["modes_1st_masked"], mode, GG_diag, frequency_1st)
                plt.plot(f_1st, psd_1st, ':', color=color, lw=2.5, label=f"1st stage integrator (sim, {sim['label']})")
    if plot_atm:
        if plot_onsky:
            f_atm, psd_atm, _ = tPSD_calculator(modes_atm_masked, mode, GG_diag, frequency_atm)
            plt.plot(f_atm, psd_atm, lw=2.5, color=color_atm, label="atm")
        if plot_sim:
            for sim, color in zip(simulations, shades_atm):
                f_atm, psd_atm, _ = tPSD_calculator(sim["modes_atm_masked"], mode, GG_diag, frequency_1st)
                plt.plot(f_atm, psd_atm, ':', color=color, lw=2.5, label=f"atm (sim, {sim['label']})")

    plt.title("Tip PSD" if mode == 0 else f"PSD KL mode {mode}", fontsize=30)
    plt.xlabel("frequency (Hz)", fontsize=24)
    plt.ylabel("PSD (nm² Hz⁻¹)", fontsize=24)
    plt.yscale("log")
    plt.xscale("log")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1.01, 1))  # outside the axes so it does not cover the curves
    plt.tight_layout()
    plt.savefig(f"{save_dir}/tPSD_mode_{mode}.png", dpi=save_dpi, bbox_inches='tight')
    plt.close()

if plot_sim and plot_atm:
    for sim in simulations:
        print(f"total atm variance ({sim['label']}): {np.sum(sim['var_atm']):,.0f} nm²")
print(f"figures saved in {save_dir}")
