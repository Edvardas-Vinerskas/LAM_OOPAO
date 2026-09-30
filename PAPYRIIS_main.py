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
import os


PAPYRIIS_env = OOPAO_environment_PAPYRIIS()

#----------------------------------------------------Generate 2nd atmosphere#----------------------------------------------------
'''
savedir_atm = "PAPYRIIS_2stage_CNN_RL/generated_atm_2nd_stage"

print(f'_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}')
OPD_screen_1 = PAPYRIIS_env.generate_second_stage_atmosphere(nLoop=15000)

if not os.path.exists(savedir_atm):
    os.makedirs(savedir_atm)


np.savez_compressed(
    f"{savedir_atm}/atm_OPDs_2nd_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz",
    atm_OPDs_2nd=OPD_screen_1,
    r0=PAPYRIIS_env.atm_2nd.r0,
    L0=PAPYRIIS_env.atm_2nd.L0,
    windSpeed=PAPYRIIS_env.atm_2nd.windSpeed,
    fractionalR0=PAPYRIIS_env.atm_2nd.fractionalR0,
    windDirection=PAPYRIIS_env.atm_2nd.windDirection,
    altitude=PAPYRIIS_env.atm_2nd.altitude
)


#---------------------------------------------------Project atmosphere to 1st stage#----------------------------------------------------

atm_OPDs_2nd = np.load(f"{savedir_atm}/atm_OPDs_2nd_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz")
atm_OPDs_2nd = atm_OPDs_2nd["atm_OPDs_2nd"]
print(f"atm loaded: _r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}_multi_layer")
print(atm_OPDs_2nd.shape)
'''
savedir_proj = "PAPYRIIS_2stage_CNN_RL/projected_atm_1st_stage"
if not os.path.exists(savedir_proj):
    os.makedirs(savedir_proj)
'''
#TODO add tqdm
atm_OPDs_1st = PAPYRIIS_env.project_atmosphere_to_first_stage(atm_OPDs_2nd)



np.savez(f"{savedir_proj}/atm_OPDs_1st_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz",
        atm_OPDs_1st=atm_OPDs_1st,
        r0=PAPYRIIS_env.atm_2nd.r0,
        L0=PAPYRIIS_env.atm_2nd.L0,
        windSpeed=PAPYRIIS_env.atm_2nd.windSpeed,
        fractionalR0=PAPYRIIS_env.atm_2nd.fractionalR0,
        windDirection=PAPYRIIS_env.atm_2nd.windDirection,
        altitude=PAPYRIIS_env.atm_2nd.altitude
)


'''
#----------------------------------------------------1st stage CL#----------------------------------------------------

atm_OPD_1st = np.load(f"{savedir_proj}/atm_OPDs_1st_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz")
atm_OPD_1st = atm_OPD_1st["atm_OPDs_1st"]
print(f"atm loaded: _r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}_multi_layer")


#"PAPYRIIS_2stage_CNN_RL/~2026-06-23/PAPYRIIS_arcturus_noise_centralobs_quantisation_pwfs"
savedir_1st = "PAPYRIIS_2stage_CNN_RL/2026-09-23/PAPYRIIS_arcturus_v4_nophoton_cred"
if not os.path.exists(savedir_1st):
    os.makedirs(savedir_1st)

first_stage_results = PAPYRIIS_env.run_first_stage_loop(15000, atm_OPD_1st, gainCL = 0.4)

np.savez(f"{savedir_1st}/results_1st_stage_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz", **{
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

#----------------------------------------------------1st stage residual entering to 2nd (integrator)#----------------------------------------------------
#TODO I think the frame delay is set to 2 for 1st stage and 2 for 2nd stage (should be 3)
#TODO do a few different atmospheres with varying r_0 and varying boiling to check which one works best
#TODO go through the paper and ask yourself if you need any more layers in the simulation?
#TODO all of the current atmospheres are r_0 0.05 @ 500 nm, maybe try 0.075

#TODO definitely do a weaker atmosphere and see what you get (the performance is sussy baka especially since you will go to 1310 instead of 1600)
#TODO cameras are fixed (for vZWFS bits =None (reduces performance alot and even diverges), FWC = None (overflows))
#TODO the problem is that the on sky pupil crops below 90X90 which what CNN requires
    # I made the executive decision to use a 90x90 non-masked pupil with on_sky set to True for OZIRIIS
    # because the on_sky label is necessary
    # and I am using the on sky CNN
    # another problem this introduces is that the pupil is also shifted compared to 1st stage
    # but redoing the 1st stage with the proper pupil is hard so we leave it
    #TODO in conclusion: is_onsky = True + onsky CNN + 0 pupil mask (introducing central obstruction causes divergence after iteration 30)
#TODO you need to redo it without noise and not on sky
#TODO last one has EMCCD and slow_tt (one thing to include is bits in the cred2 hehe)
#PAPYRIIS_2stage_CNN_RL\~2026-06-01\PAPYRIIS_arcturus_nonoise\results_2nd_stage.npz

#"PAPYRIIS_2stage_CNN_RL/~2026-06-23/PAPYRIIS_arcturus_noise_quantisation_pwfs_calibration_pupil_EMCCD"
savedir_test = "PAPYRIIS_2stage_CNN_RL/2026-09-23/PAPYRIIS_arcturus_v4_nophoton_cred"
loaddir_test = savedir_test
print(f"{loaddir_test}/results_1st_stage_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz")

first_stage_results = np.load(f"{loaddir_test}/results_1st_stage_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz")
residuals_opds_1rst = first_stage_results['residuals_opds_1rst']
print(first_stage_results.files)
print(first_stage_results['telescope_pupil'].shape)
print(first_stage_results['reconstructed_cmd'].shape)
print(first_stage_results['src_opds'].shape)
print(first_stage_results['residuals_opds_1rst'].shape)


dm_commands = np.zeros((15000, 97))
reconstructed_cmd = np.zeros((15000, 97))
scnd_stage_strehl = np.zeros((15000))
tel_2nd_pupil = 0
src_opd = np.zeros((15000, 90, 90))
projector_kl_2nd = np.zeros((87, 8100))


obs, _ = PAPYRIIS_env.reset(residuals_opds_1rst)
aa = time.perf_counter()
for t in range(15000):
    action = 0 * obs.unsqueeze(0).unsqueeze(0)
    next_obs, INFO = PAPYRIIS_env.step(action.squeeze(), residuals_opds_1rst)
    if t%100 == 0:
        a = time.perf_counter()
        print(f'iteration {t}, 2nd strehl ratio {INFO['2nd_stage_strehl']}')

    if (t+1)%100 == 0:
        print(time.perf_counter() - a)
    dm_commands[t] =INFO["dm_commands"]
    reconstructed_cmd[t] =INFO["reconstructed_cmd"].detach().cpu().numpy()
    scnd_stage_strehl[t] =INFO["2nd_stage_strehl"]
    src_opd[t] =INFO["src_opd"]
    tel_2nd_pupil = INFO["telescope_pupil"]
    projector_kl_2nd = INFO["projector_kl_2nd"]

    obs = next_obs

print(time.perf_counter() - aa)
np.savez(f"{savedir_test}/results_2nd_stage_CL1OL2_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}.npz",
    # Concatenated across iterations
    all_2nd_stage_strehl    = scnd_stage_strehl,
    all_dm_commands         = dm_commands,
    all_reconstructed_cmd   = reconstructed_cmd,
    residual_opds_2nd       = src_opd,
    telescope_pupil         = tel_2nd_pupil,
    projector_kl_2nd        = projector_kl_2nd,
    )
