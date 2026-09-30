"""
1st stage tests: compares on-sky data with 6 ways of calculating the sim 1st stage residual (KL mode variance + temporal PSDs only)
curves are selected by commenting out lines in onsky_curves, atm_sims and residual_methods

#   atm reconstruction from 1st stage DM shape (on sky: dmCmdCube, sim: results_1st_stage_*.npz 'dm_commands')
#   1st stage residual reconstruction from 2nd stage CNN reconstruction in DM space (CL1OL2)

Plotting style follows OOPAO_metric_plotting_bench_telemetry_grd_matlab.py:
    on sky     -> atm solid black, 1st stage dashed ('--')
    simulation -> dotted (':'), one colour per 1st stage residual calculation method (OPD screens: shades of green)

Figures are saved at save_dpi into save_dir (nothing is shown)
Run from the LAM_OOPAO folder


this file is a copy of 1st_stage_tests but for comparing results of OPDs and different way of calculating the 1st stage results
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: figures are saved to disk instead of shown
import matplotlib.pyplot as plt
from functions import *

save_dir = 'PAPYRIIS_2stage_CNN_RL/2026-09-23/figures_1st_stage_OPD'   # folder where all figures are saved
save_dpi = 200
os.makedirs(save_dir, exist_ok=True)

plot_onsky    = True   # plot on-sky results (solid/dashed lines)
plot_sim      = True   # plot simulation results (dotted lines)
plot_atm      = True   # atmosphere from 1st stage DM commands
plot_residual = True   # 1st stage residual from CL1OL2

frequency_1st = 400    # sampling frequency of the 1st stage residual (CL1OL2, measured by the 2nd stage) and of the sim DM commands
frequency_atm = 200    # sampling frequency of the on-sky 1st stage DM commands
CL_gain_pyr   = 0.3
KL_frames     = [0, 15000]              # [start, end] indices for the timeseries window
psd_modes     = [0, 10, 20, 30, 40]  # KL mode indices to plot PSDs for

# on sky data
loaddir = 'bench_sky_04_15/arcturus_v7'
RL_telemetry_1st = np.load(f'{loaddir}/2026-04-16T01_19_12_telemetry_data_RLiter50.npy', allow_pickle = True)
CL1st_OL_2nd     = np.load(f'{loaddir}/2026-04-16T01_27_23_telemetry_2nd_data_CL1OL2_noCRED2.npy', allow_pickle = True)



# simulation to compare: folder, 1st stage results file, 2nd stage CL1OL2 results file (same folder), atm OPD file
simdir = "PAPYRIIS_2stage_CNN_RL/2026-09-23"
sim = {"label": "arcturus_v4",
       "folder":   f"{simdir}/PAPYRIIS_arcturus_v1",
       "file_1st": "results_1st_stage_r0_0.050_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz",
       "file_2nd": "results_2nd_stage_CL1OL2.npz",
       "file_atm_1st": "PAPYRIIS_2stage_CNN_RL/projected_atm_1st_stage/atm_OPDs_1st_r0_0.050_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz",
       "file_atm_2nd": "PAPYRIIS_2stage_CNN_RL/generated_atm_2nd_stage/atm_OPDs_2nd_r0_0.050_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz"}


# curves to plot: comment out a line to leave that curve out (only the selected curves are computed)

# on sky: key, kind (atm/residual), label, line style, colour in the variance plot, colour in the PSD plots, sampling frequency
onsky_curves = [
    {"key": "atm",        "kind": "atm",      "label": "Atm",                          "ls": "-",  "color": "black",          "color_psd": "black",        "freq": frequency_atm},
    {"key": "integrator", "kind": "residual", "label": "1st stage (integrator)",       "ls": "--", "color": "cornflowerblue", "color_psd": "indianred",    "freq": frequency_1st},
    {"key": "pwfs",       "kind": "residual", "label": "1st stage (PWFS measurement)", "ls": "--", "color": "mediumpurple",   "color_psd": "mediumpurple", "freq": frequency_atm},  # modeCube at the on sky 1st stage rate
]

# sim atmospheres: key, label, colour, sampling frequency
atm_sims = [
    {"key": "dm",     "label": "DM",        "color": "dimgray",     "freq": frequency_1st},  # from the 1st stage DM commands
    {"key": "opd_80", "label": "OPD 80x80", "color": "saddlebrown", "freq": frequency_1st},  # atm OPD screens (real, unfiltered atmosphere)
    {"key": "opd_90", "label": "OPD 90x90", "color": "peru",        "freq": frequency_1st},  # same atm on the 2nd stage 90x90 grid
]

# different ways of calculating the sim 1st stage residual: key, label, colour, sampling frequency
# OPD screens share the same colour (Greens) with different intensity
greens = plt.get_cmap("Greens")
residual_methods = [
    {"key": "CNN",          "label": "CNN reconstruction (no OG)",   "color": "tab:blue",    "freq": frequency_1st},
    {"key": "opd_2nd_90",   "label": "residual OPD 2nd stage (90x90)",   "color": greens(0.95),  "freq": frequency_1st},
    {"key": "opd_1st_90",   "label": "residual OPD 1st stage (90x90)",   "color": greens(0.70),  "freq": frequency_1st},
    {"key": "opd_1st_80",   "label": "residual OPD 1st stage (80x80)",   "color": greens(0.45),  "freq": frequency_1st},
    {"key": "pwfs",         "label": "PWFS measurement (OG)",     "color": "tab:purple",  "freq": frequency_1st / 2},  # PWFS measured every 2nd frame (7500 samples)
    {"key": "atm_minus_dm", "label": "φ_atm - φ_dm (no OG)",         "color": "tab:orange",  "freq": frequency_1st},
]


#TODO these don't have the representative masks (masking done as per PAPYRIIS_main.py or just look inside the 2 below functions)
dm_1st_inf, pupil_mask_1 = first_stage_dm_builder()
dm_2nd_inf, pupil_mask_2 = second_stage_dm_builder()

dm_1st_inf = dm_1st_inf.reshape(100, 100, 241)
dm_1st_inf_masked = dm_1st_inf[pupil_mask_1, :]

dm_2nd_inf = dm_2nd_inf.reshape(100, 100, 97)
dm_2nd_inf_masked = dm_2nd_inf[pupil_mask_2, :]

M2C_1st = - np.load('PAPYRIIS_2stage_CNN_RL/M2C_1rst.npy')
#M2C_2nd = np.load('PAPYRIIS_2stage_CNN_RL/M2C_2nd.npy')  # on sky 2nd stage basis (66 modes)
M2C_KL  = np.load('PAPYRIIS_2stage_CNN_RL/M2C_KL.npy')   # sim 2nd stage basis (87 modes), used in OOPAO_PAPYRIIS_env.py (projector_kl_2nd, CNN output)

C2M_1st = np.linalg.pinv(M2C_1st)
#C2M_2nd = np.linalg.pinv(M2C_2nd)
C2M_KL  = np.linalg.pinv(M2C_KL)
M2P_1st = dm_1st_inf_masked @ M2C_1st
#projector_kl_2nd_to_1st     = np.linalg.pinv(M2P_1st) @ (dm_2nd_inf_masked @ M2C_2nd)  # on sky
projector_kl_2nd_to_1st = np.linalg.pinv(M2P_1st) @ (dm_2nd_inf_masked @ M2C_KL)   # used for both on sky and sim
#stage_2nd_filter = projector_kl_2nd_to_1st @ np.linalg.pinv(projector_kl_2nd_to_1st)  #passes 1st stage measuremrents through 2nd stage basis to apply the effects of the 2nd stage


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
def curve_selected(kind):
    return (kind == "atm" and plot_atm) or (kind == "residual" and plot_residual)

if plot_onsky:
    # every source gives 1st stage KL modes, shape (timeseries length, no_of_modes); only the selected ones are computed
    onsky_sources = {
        "atm":        lambda: mode_calculator_fromDM(RL_telemetry_1st.item()['dmCmdCube'].squeeze(), M2C_1st),
        "integrator": lambda: CL1st_OL_2nd.item()['slavedreconsCube'].squeeze() @ C2M_KL.T @ projector_kl_2nd_to_1st.T,  #1st stage residual as measured by 2nd stage
        "pwfs":       lambda: RL_telemetry_1st.item()['modeCube'].squeeze(),
    }
    onsky = {"var": {}, "modes_masked": {}}
    for curve in onsky_curves:
        if curve_selected(curve["kind"]):
            key = curve["key"]
            onsky["var"][key], onsky["modes_masked"][key] = KL_variance_calculator(onsky_sources[key](), GG_diag, KL_frames)

if plot_sim:
    print(f"loading {sim['folder']}")
    path_1st = np.load(f"{sim['folder']}/{sim['file_1st']}")  # npz arrays are only read when accessed
    path_2nd = np.load(f"{sim['folder']}/{sim['file_2nd']}")

    opd_mask_1st = path_1st['telescope_pupil'] > 0   # 80x80
    opd_mask_2nd = path_2nd['telescope_pupil'] > 0   # 90x90
    # sim KL bases in OPD space (m) on the pupil pixels: pinv(projector) = dm.modes @ M2C (M2C_1st for 80x80, M2C_KL for 90x90)
    kl_basis_1st = np.linalg.pinv(path_1st['projector_kl_1st'].reshape(-1, 80, 80)[:, opd_mask_1st])
    kl_basis_2nd = np.linalg.pinv(path_2nd['projector_kl_2nd'].reshape(-1, 90, 90)[:, opd_mask_2nd])

    # OPD cube -> 1st stage KL modes, piston removed before projection (mode_calculator_fromOPD); M2C already in the basis -> identity
    def opd_80_to_kl_1st(opds):
        return mode_calculator_fromOPD(opds[:, opd_mask_1st], np.eye(kl_basis_1st.shape[1]), kl_basis_1st)
    def opd_90_to_kl_1st(opds):
        return mode_calculator_fromOPD(opds[:, opd_mask_2nd], np.eye(kl_basis_2nd.shape[1]), kl_basis_2nd) @ projector_kl_2nd_to_1st.T

    # every source gives 1st stage KL modes, shape (timeseries length, no_of_modes); only the selected ones are computed
    atm_sources = {
        "dm":     lambda: mode_calculator_fromDM(path_1st['dm_commands'], M2C_1st),
        "opd_80": lambda: opd_80_to_kl_1st(np.load(sim["file_atm_1st"])['atm_OPDs_1st']),
        "opd_90": lambda: opd_90_to_kl_1st(np.load(sim["file_atm_2nd"])['atm_OPDs_2nd']),  # 2nd stage KL modes, then projected onto the 1st stage KL modes
    }
    sim["modes_atm"] = {}
    def sim_atm_modes(key):  # computed once, also used by φ_atm - φ_dm
        if key not in sim["modes_atm"]:
            sim["modes_atm"][key] = atm_sources[key]()
        return sim["modes_atm"][key]

    sim["var_atm"], sim["modes_atm_masked"] = {}, {}
    if plot_atm:
        for atm in atm_sims:
            key = atm["key"]
            sim["var_atm"][key], sim["modes_atm_masked"][key] = KL_variance_calculator(sim_atm_modes(key), GG_diag, KL_frames)
            print(f"  atm {atm['label']:31s} σ² = {np.sum(sim['var_atm'][key]):,.0f} nm²")

    sim["var_1st"], sim["modes_1st_masked"] = {}, {}
    if plot_residual:
        residual_sources = {
            "CNN":          lambda: path_2nd['all_reconstructed_cmd'].squeeze() @ C2M_KL.T @ projector_kl_2nd_to_1st.T,  # sim CNN cmds = M2C_KL[:, :50] @ cnn_output
            "opd_2nd_90":   lambda: opd_90_to_kl_1st(path_2nd['residual_opds_2nd']),
            "opd_1st_90":   lambda: opd_90_to_kl_1st(path_1st['residuals_opds_1rst']),
            "opd_1st_80":   lambda: opd_80_to_kl_1st(path_1st['src_opds']),
            "pwfs":         lambda: path_1st['reconstructed_cmd'] @ C2M_1st.T,
            # M2C_1st is negated, so DM modes = -φ_dm in KL space -> φ_atm - φ_dm = atm + DM modes
            "atm_minus_dm": lambda: sim_atm_modes("opd_80") + sim_atm_modes("dm"),
        }
        for method in residual_methods:
            key = method["key"]
            sim["var_1st"][key], sim["modes_1st_masked"][key] = KL_variance_calculator(residual_sources[key](), GG_diag, KL_frames)
            print(f"  {method['label']:35s} σ² = {np.sum(sim['var_1st'][key]):,.0f} nm²")


# ---------------------------------------------------KL mode variance---------------------------------------------------#
plt.figure(figsize=(19, 8))  # wider than the original (12, 8) to fit the legend on the right
def plot_onsky_variance(kind):
    for curve in onsky_curves:
        if curve["kind"] == kind:
            var = onsky["var"][curve["key"]]
            plt.plot(var, curve["ls"], color=curve["color"], lw=1.5 if kind == "atm" else 2.5, label=f"{curve['label']}, σ² = {np.sum(var):,.0f} nm²")

if plot_atm:
    if plot_onsky:
        plot_onsky_variance("atm")
    if plot_sim:
        for atm in atm_sims:
            var = sim["var_atm"][atm["key"]]
            plt.plot(var, ':', color=atm["color"], label=f"Atm (sim, {atm['label']}), σ² = {np.sum(var):,.0f} nm²")
if plot_residual:
    if plot_onsky:
        plot_onsky_variance("residual")
    if plot_sim:
        for method in residual_methods:
            var = sim["var_1st"][method["key"]]
            plt.plot(var, ':', color=method["color"], lw=2.5, label=f"1st stage (sim, {method['label']}), σ² = {np.sum(var):,.0f} nm²")

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
def plot_onsky_psd(kind, mode):
    for curve in onsky_curves:
        if curve["kind"] == kind:
            f, psd, _ = tPSD_calculator(onsky["modes_masked"][curve["key"]], mode, GG_diag, curve["freq"])
            plt.plot(f, psd, curve["ls"], color=curve["color_psd"], lw=2.5, label=curve["label"])

for mode in psd_modes:
    plt.figure(figsize=(19, 8))  # wider than the original (12, 8) to fit the legend on the right
    if plot_residual:
        if plot_onsky:
            plot_onsky_psd("residual", mode)
        if plot_sim:
            for method in residual_methods:
                f_1st, psd_1st, _ = tPSD_calculator(sim["modes_1st_masked"][method["key"]], mode, GG_diag, method["freq"])
                plt.plot(f_1st, psd_1st, ':', color=method["color"], lw=2.5, label=f"1st stage (sim, {method['label']})")
    if plot_atm:
        if plot_onsky:
            plot_onsky_psd("atm", mode)
        if plot_sim:
            for atm in atm_sims:
                f_atm, psd_atm, _ = tPSD_calculator(sim["modes_atm_masked"][atm["key"]], mode, GG_diag, atm["freq"])
                plt.plot(f_atm, psd_atm, ':', color=atm["color"], lw=2.5, label=f"Atm (sim, {atm['label']})")

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

print(f"figures saved in {save_dir}")
