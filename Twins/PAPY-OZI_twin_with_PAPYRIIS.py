# -*- coding: utf-8 -*-
"""
PAPY-OZI_twin_with_PAPYRIIS.py

Script version of the PAPY--OZI two-stage simulation using the PAPYRIIS driver
class explicitly, without calling sim.run_nominal_workflow().

This file intentionally keeps the original notebook/cell structure. The goal is
not to hide the workflow inside one method, but to make every step explicit while
letting PAPYRIIS store all relevant intermediate products in `sim`.

Required local files, typically in the working directory or with absolute paths:
- PAPYRIIS.py
- M2C_1rst.npy
- useful_pix.npy
- intMat_klOOPAO_synthetic_bin=1_F=500_rMod=5_20250604_0307.mat
- the existing Papyrus/OZIRIIS/parallel_utils_twin modules
"""

import numpy as np
import matplotlib.pyplot as plt

from PAPYRIIS import PAPYRIIS

# Optional plotting helpers used by the original analysis cells.
try:
    from plot_functions import (
        plot_psd_aa,
        plot_sr_aa,
        plot_psf_aa,
        plot_frame_count_aa,
        plot_phase_map_aa,
        plot_etf_fit_aa,
        plot_cumulative_psd_aa,
        plot_phase_comparison_aa,
        plot_n_psd_aa,
        plot_curves_aa,
    )
except Exception:
    plot_psd_aa = None
    plot_n_psd_aa = None
    plot_curves_aa = None
    plot_etf_fit_aa = None

try:
    from utils import compute_etf
except Exception:
    compute_etf = None

import tqdm
# =============================================================================
# User paths and main parameters
# =============================================================================

# Stop hard-coding this like a savage in the middle of the code. Put your paths
# here once, then the rest of the script is reusable.
M2C_1RST_PATH = "M2C_1rst.npy"
VALID_PIXEL_PATH = "useful_pix.npy"
INTERACTION_MATRIX_PATH = (
    "C:/Users/mmotte/oopao/OOPAO/tutorials/PAPYRUS/"
    "intMat_klOOPAO_synthetic_bin=1_F=500_rMod=5_20250604_0307.mat"
)
MAT_KEY = "matrix_inf"

MASTER_FOLDER = "C:/Users/mmotte/OZIRIIS/data_calibration/"
master_folder = ''
N_LOOP = 10000
FIRST_STAGE_END_MODE = 195
SECOND_STAGE_NMODES = 35
ATM_SEED = 15

FIRST_GAIN = 0.5
FIRST_LEAK = 0.995
FIRST_FRAME_DELAY = 2
FIRST_SKY_OFFSET = (2, 2)

SECOND_GAIN = 0.0
SECOND_LEAK = 0.98
SECOND_FRAME_DELAY = 2

RECONSTRUCT_ATAN = True
ATAN_PARALLEL = True
ATAN_NJOBS = 6
ATAN_ITERATION = 20
ATAN_DAMPING = 0.5

#%%
# =============================================================================
# 1. Build the simulation driver and initialise both stages
# =============================================================================

sim = PAPYRIIS(
    auto_init_first_stage=True,
    auto_init_second_stage=True,
    first_stage_calibration_pupil=True,
    first_stage_sky_offset=FIRST_SKY_OFFSET,
    second_stage_is_onsky=True,
    second_stage_controlled_modes=SECOND_STAGE_NMODES,
)

# Keep the same short names as the old script for interactive inspection.
Papytwin = sim.Papytwin
tel = sim.tel
ngs = sim.ngs
dm = sim.dm
wfs = sim.wfs
atm = sim.atm
slow_tt = sim.slow_tt
param = sim.param

OZItwin = sim.OZItwin

#%%
# =============================================================================
# 2. First-stage pupil selection and one PWFS propagation
# =============================================================================

# Original script behavior:
#   Papytwin.set_pupil(calibration=False, sky_offset=[2, 2])
#   Papytwin.set_pupil(calibration=True)
#   ngs * tel * wfs

sim.set_first_stage_pupil(calibration=True)
sim.initialize_first_stage_propagation()

#%%
# =============================================================================
# 3. Load PAPYRUS bench inputs
# =============================================================================

sim.load_first_stage_inputs(
    M2C_path=M2C_1RST_PATH,
    valid_pixel_path=VALID_PIXEL_PATH,
    interaction_matrix_path=INTERACTION_MATRIX_PATH,
    mat_key=MAT_KEY,
    bin_bench_data=True,
)

M2C = sim.M2C_1rst
valid_pixel = sim.valid_pixel
im = sim.int_mat_1rst
valid_pixel_binned = sim.valid_pixel_binned
int_mat_binned = sim.int_mat_binned

# Index of the KL modes included in the original interaction-matrix comparison.
ind = [1, 5, 10, 20, 30, 50, 80, 100, 150]
int_mat_extract = im[:, ind]

#%%
# =============================================================================
# 4. Optional PAPYRUS/PAPYTWIN PWFS pupil comparison/correction
# =============================================================================

var_im = np.var(im, axis=1).reshape(240, 240)
var_im /= var_im.max()
var_im = var_im > 0.005

# Set to False if you do not want the pupil shift correction step.
DO_CHECK_PWFS_PUPILS = True
if DO_CHECK_PWFS_PUPILS:
    sim.check_first_stage_pwfs_pupils(
        valid_pixel_map=var_im,
        correct=True,
        n_it=6,
    )

#%%
# =============================================================================
# 5. Optional first-stage synthetic IM diagnostics
# =============================================================================

# The original script computes two InteractionMatrix objects for comparison.
# This is not needed to run the closed loop. Keep these disabled unless you
# explicitly want the diagnostic plots/computation.
DO_FIRST_STAGE_IM_DIAGNOSTICS = True
if DO_FIRST_STAGE_IM_DIAGNOSTICS:
    from OOPAO.calibration.InteractionMatrix import InteractionMatrix

    wfs.modulation = 5
    stroke = 1e-4

    M2C_extract = M2C[:, ind]
    calib_extract = InteractionMatrix(
        ngs=ngs,
        atm=atm,
        tel=tel,
        dm=dm,
        wfs=wfs,
        M2C=M2C_extract,
        stroke=stroke,
        phaseOffset=0,
        nMeasurements=1,
        noise="off",
        print_time=False,
        display=True,
    )

    calib_full = InteractionMatrix(
        ngs=ngs,
        atm=atm,
        tel=tel,
        dm=dm,
        wfs=wfs,
        M2C=M2C,
        stroke=stroke,
        phaseOffset=0,
        nMeasurements=1,
        noise="off",
        print_time=False,
        display=True,
    )
#%%

# =============================================================================
# 6. First-stage closed-loop calibration
# =============================================================================

tel.resetOPD()

sim.calibrate_first_stage(
    end_mode=FIRST_STAGE_END_MODE,
    compute_synthetic_IM=True,
    use_binned_interaction_matrix=False,
    switch_to_sky=True,
    sky_offset=FIRST_SKY_OFFSET,
)

M2C_CL = sim.M2C_CL
calib_CL = sim.calib_1rst
reconstructor = sim.reconstructor_1rst

#%%
# =============================================================================
# 7. Second-stage calibration
# =============================================================================

sim.calibrate_second_stage(
    nmodes=SECOND_STAGE_NMODES,
    stroke_nm=12,
    use_zwfs=1,
)

IM_z1 = sim.IM_z1
IM_z2 = sim.IM_z2
calib_CL_2nd = sim.calib_2nd_M
reconstructor_2nd = sim.reconstructor_2nd

#%%
# =============================================================================
# 8. Generate second-stage atmosphere and project it to the first stage
# =============================================================================

atm_OPDs_2nd = sim.generate_second_stage_atmosphere(
    nLoop=N_LOOP,
    seed=ATM_SEED,
    use_no_pupil=True,
    progress=True,
)

atm_OPDs_1rst = sim.project_atmosphere_to_first_stage()
#%%

# =============================================================================
# 9. First-stage loop
# =============================================================================
nloop = 500
first_ol = sim.run_first_stage_loop(
    nLoop=nloop,
    gainCL=0.,
    leak=FIRST_LEAK,
    frame_delay=FIRST_FRAME_DELAY,
    photon_noise=False,
    progress=True,
)

first_cl = sim.run_first_stage_loop(
    nLoop=nloop,
    gainCL=0.7,
    leak=0.995,
    frame_delay=FIRST_FRAME_DELAY,
    photon_noise=False,
    progress=True,
)
#%%
pupil = tel.pupil.ravel().astype(bool)

# projecteur 35 modes, pas le projecteur global
tel.resetOPD()
dm.coefs = 0
dm.coefs = sim.M2C_1rst[:,:]#np.identity(dm.nValidAct) #OZItwin.M2C[:, :nmodes]
tel * dm

modes = tel.OPD.copy().reshape(tel.resolution**2, -1)
modes_pupil = modes
modes_pupil /= modes_pupil.std(axis=0, keepdims=True)

# proj_1rst = np.linalg.pinv(modes_pupil)
cov_modes = modes_pupil.T @ modes_pupil
diag = np.diag(cov_modes)
diag = np.where(np.abs(diag) < 1e-30, 1.0, diag)

proj_1rst = np.linalg.pinv(modes_pupil)#(np.diag(1.0 / diag) @ modes_pupil.T).astype(np.float32)
#%%
projected_if_ol = proj_1rst@sim.atm_OPDs_1rst[:nloop].reshape(nloop,-1).T
projected_if_cl = proj_1rst@(first_cl['dm_opds'][:nloop].reshape(nloop,-1)+sim.atm_OPDs_1rst[:nloop].reshape(nloop,-1)).T
projected_if_cl = proj_1rst@(first_cl['src_opds'][:nloop].reshape(nloop,-1)).T

#%%
projected_if_ol = sim.atm_OPDs_2nd[:nloop, sim.tel_2nd.pupil==1].T
projected_if_cl = sim.residuals_opds_1rst[:nloop, sim.tel_2nd.pupil==1].T
# projected_if_cl = proj_1rst@(first_cl['src_opds'][:nloop, tel.pupil==1]).T
#%%

t = np.arange(projected_if_ol.shape[1]) *tel.samplingTime

f_ol, psd_ol = sim.OZItwin.psd(
    t,
    projected_if_ol.T,
    nperseg=min(5000, projected_if_ol.shape[1]),
)

f_cl, psd_cl = sim.OZItwin.psd(
    t,
    projected_if_cl.T,
    nperseg=min(5000, projected_if_cl.shape[1]),
)


#%%
fig_atg, ax_atg = plot_psd_aa( 
    f_cl,
    psd_cl*1e18,
    f_ol,
    psd_ol*1e18,
    label1="closed loop",
    label2="open loop",
    method=np.nansum,
    f_label="Frequency [Hz]",
    psd_label=r"PSD [nm$^2$/Hz]",
    fmin=None,
    fmax=None,
    etf_vmax=1.2,
    normalised=False,
    show_legend=True,
    one_column=True,
    dpi=300,
    save=False,
    savepath=master_folder+f"etf_first_stage/psds.psf",
    saveformat='all',
    journal_style=True,   # True: A&A final style ; False: working style with light grid
)
#%%
t = np.arange(projected_if_ol.shape[1]) *tel.samplingTime

f_ol, psd_ol = sim.OZItwin.psd(
    t,
    projected_if_ol.T,
    nperseg=min(1000, projected_if_ol.shape[0]),
)

f_cl, psd_cl = sim.OZItwin.psd(
    t,
    projected_if_cl.T,
    nperseg=min(1000, projected_if_cl.shape[0]),
)


#%%
fig_atg, ax_atg = plot_psd_aa( 
    f_cl,
    psd_cl,
    f_ol,
    psd_ol,
    label1="closed loop",
    label2="open loop",
    method=np.nansum,
    f_label="Frequency [Hz]",
    psd_label=r"PSD [nm$^2$/Hz]",
    fmin=None,
    fmax=None,
    normalised=False,
    show_legend=True,
    one_column=True,
    dpi=300,
    save=False,
    savepath=master_folder+f"etf_first_stage/psds.psf",
    saveformat='all',
    journal_style=True,   # True: A&A final style ; False: working style with light grid
)
#%%
from OOPAO.GainSensingCamera import GainSensingCamera
gsc = GainSensingCamera(wfs.mask, modes_pupil.reshape(80,80,-1))
og = []
wfs.focal_plane_camera.resolution = wfs.nRes
tel.resetOPD()
tel * wfs
wfs * wfs.focal_plane_camera
wfs.focal_plane_camera * gsc

for i in tqdm.tqdm(range(100)):

    tel.OPD = first_cl['src_opds'][np.random.random_integers(0,1000)]
    tel * wfs
    wfs * wfs.focal_plane_camera
    wfs.focal_plane_camera * gsc
    og.append(gsc.og)

#%%
# =============================================================================
# 10. Second-stage loop
# =============================================================================
second_ol = sim.run_second_stage_loop(
    nLoop=nloop,
    nmodes=SECOND_STAGE_NMODES,
    gainCL_2nd=0,
    leak_2nd=SECOND_LEAK,
    frame_delay_2nd=SECOND_FRAME_DELAY,
    progress=True,
)
second = sim.run_second_stage_loop(
    nLoop=nloop,
    nmodes=SECOND_STAGE_NMODES,
    gainCL_2nd=1,
    leak_2nd=SECOND_LEAK,
    frame_delay_2nd=SECOND_FRAME_DELAY,
    progress=True,
)

#%%
tel_2nd = sim.tel_2nd
dm_2nd = sim.dm_2nd
pupil = tel_2nd.pupil.ravel().astype(bool)

# projecteur 35 modes, pas le projecteur global
tel_2nd.resetOPD()
dm_2nd.coefs = 0
dm_2nd.coefs = sim.OZItwin.M2C[:,:35]#np.identity(dm.nValidAct) #OZItwin.M2C[:, :nmodes]
tel_2nd * dm_2nd

modes = tel_2nd.OPD.copy().reshape(tel_2nd.resolution**2, -1)
modes_pupil = modes[pupil]
modes_pupil /= modes_pupil.std(axis=0, keepdims=True)

# proj_1rst = np.linalg.pinv(modes_pupil)
cov_modes = modes_pupil.T @ modes_pupil
diag = np.diag(cov_modes)
diag = np.where(np.abs(diag) < 1e-30, 1.0, diag)

proj_2nd = np.linalg.pinv(modes_pupil)#(np.diag(1.0 / diag) @ modes_pupil.T).astype(np.float32)
#%%
opd_1rst_ol = (atm_OPDs_2nd-atm_OPDs_2nd.mean(axis=0))*sim.tel_2nd.pupil
opd_1rst_cl =( sim.residuals_opds_1rst-sim.residuals_opds_1rst.mean(axis=0))*sim.tel_2nd.pupil

opd_2nd_cl = (sim.opds_2nd-sim.opds_2nd.mean(axis=0))*sim.tel_2nd.pupil
#%%

proj_1rst_ol = proj_2nd@opd_1rst_ol[:nloop,tel_2nd.pupil].T
proj_1rst_cl = proj_2nd@opd_1rst_cl[:nloop,tel_2nd.pupil].T
proj_2nd_cl = proj_2nd@opd_2nd_cl[:nloop,tel_2nd.pupil].T
#%%

proj_1rst_ol = sim.OZItwin.proj_IF@opd_1rst_ol[:nloop].reshape(nloop,-1).T
proj_1rst_cl = sim.OZItwin.proj_IF@opd_1rst_cl[:nloop].reshape(nloop,-1).T
proj_2nd_cl = sim.OZItwin.proj_IF@opd_2nd_cl[:nloop].reshape(nloop,-1).T
#%%

psd_ol = sim.OZItwin.psd(
    t,
    proj_1rst_ol.T,
    nperseg=min(1000, projected_if_ol.shape[0]),
)

psd_cl1 = sim.OZItwin.psd(
    t,
    proj_1rst_cl.T,
    nperseg=min(1000, projected_if_cl.shape[0]),
)

psd_cl2 = sim.OZItwin.psd(
    t,
    proj_2nd_cl.T,
    nperseg=min(1000, projected_if_cl.shape[0]),
)

#%%
psds = [
    (psd_ol[0], psd_ol[1]*1e18),
    (psd_cl1[0], psd_cl1[1]*1e18),
    (psd_cl2[0], psd_cl2[1]*1e18)
]

labels = [
    "OL",
    "CL1 - OL2",
    "CL1 - CL2",

]

fig, ax = plot_n_psd_aa(
    psds,
    labels=labels,
    fmax=None,
    one_column=False,
    save=False,
    method=np.nanmean,
    savepath=master_folder+f"/psd_comparison",
    saveformat="all",
    f_label="Frequency [Hz]",
    psd_label=r"PSD [nm$^2$/modes]",
)