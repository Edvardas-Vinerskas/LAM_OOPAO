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
#TODO do a few different atmospheres with varying r_0 and varying boiling to check which one works best
#TODO go through the paper and ask yourself if you need any more layers in the simulation?
#TODO match the performance of the first stage (remember that you have a perfect pyramid and so maybe your gain should not be 0.7)
#TODO all of the current atmospheres are r_0 0.05 @ 500 nm, maybe try 0.075
#TODO check that your light propagation path has all of the components (you don't have the slow_tt and focal plane cameras but you don't really care?)
    #check og OZIRIIS
#TODO wfs cameras are ok!
# here seems to be where the actual generating of everythin starts
#----------------------------------------------------Generate 2nd atmosphere#----------------------------------------------------

print(f'_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}')
OPD_screen_1 = PAPYRIIS_env.generate_second_stage_atmosphere(nLoop=150000)

savedir_atm = "generated_atm_2nd_stage"
if not os.path.exists(savedir_atm):
    os.makedirs(savedir_atm)


np.savez(
    f"PAPYRIIS_2stage_CNN_RL/{savedir_atm}/atm_OPDs_2nd_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz",
    atm_OPDs_2nd=OPD_screen_1,
    r0=PAPYRIIS_env.atm_2nd.r0,
    L0=PAPYRIIS_env.atm_2nd.L0,
    windSpeed=PAPYRIIS_env.atm_2nd.windSpeed,
    fractionalR0=PAPYRIIS_env.atm_2nd.fractionalR0,
    windDirection=PAPYRIIS_env.atm_2nd.windDirection,
    altitude=PAPYRIIS_env.atm_2nd.altitude
)


#---------------------------------------------------Project atmosphere to 1st stage#----------------------------------------------------
'''
atm_OPDs_2nd = np.load(f"PAPYRIIS_2stage_CNN_RL/generated_atm_2nd_stage/atm_OPDs_2nd_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz")
atm_OPDs_2nd = atm_OPDs_2nd["atm_OPDs_2nd"]
print(f"atm loaded: _r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}_multi_layer")
print(atm_OPDs_2nd.shape)

atm_OPDs_1st = PAPYRIIS_env.project_atmosphere_to_first_stage(atm_OPDs_2nd)

np.savez(f"PAPYRIIS_2stage_CNN_RL/projected_atm_1st_stage/atm_OPDs_1st_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz",
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
'''
atm_OPD_1st = np.load(f"PAPYRIIS_2stage_CNN_RL/projected_atm_1st_stage/atm_OPDs_1st_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz")
atm_OPD_1st = atm_OPD_1st["atm_OPDs_1st"]
print(f"atm loaded: _r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}_multi_layer")


savedir_test = "PAPYRIIS_2stage_CNN_RL/~2026-06-23/PAPYRIIS_arcturus_noise_centralobs_quantisation_pwfs"

first_stage_results = PAPYRIIS_env.run_first_stage_loop(150000, atm_OPD_1st, gainCL = 0.4)

np.savez(f"{savedir_test}/results_1st_stage_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz", **{
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
#----------------------------------------------------1st stage residual entering to 2nd (integrator)#----------------------------------------------------
#TODO definitely do a weaker atmosphere and see what you get (the performance is sussy baka especially since you will go to 1310 instead of 1600)
#TODO cameras are fixed (bits =None (reduces performance alot and even diverges), FWC = None (overflows))
#TODO the problem is that the on sky pupil crops below 90X90 which what CNN requires
    # I made the executive decision to use a 90x90 non-masked pupil with on_sky set to True for OZIRIIS
    # because the on_sky label is necessary
    # and I am using the on sky CNN
    # another problem this introduces is that the pupil is also shifted compared to 1st stage
    # but redoing the 1st stage with the proper pupil is hard so we leave it
    #TODO in conclusion: is_onsky = True + onsky CNN + 0.1 pupil mask (it diverged around iteration 30)
#TODO you need to redo it without noise and not on sky
#TODO redo it with noise and on sky but slightly higher r_0
#TODO last one has EMCCD and slow_tt (one thing to include is bits in the cred2 hehe and also pyramid OGs)
#PAPYRIIS_2stage_CNN_RL\~2026-06-01\PAPYRIIS_arcturus_nonoise\results_2nd_stage.npz
savedir_test = "PAPYRIIS_2stage_CNN_RL/~2026-06-23/PAPYRIIS_arcturus_noise_quantisation_pwfs_calibration_pupil_EMCCD"
loaddir_test = savedir_test
print(f"{loaddir_test}/results_1st_stage_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz")

first_stage_results = np.load(f"{loaddir_test}/results_1st_stage_r0_{PAPYRIIS_env.atm_2nd.r0:.3f}_V0_{PAPYRIIS_env.atm_2nd.V0:.3f}_L0_{PAPYRIIS_env.atm_2nd.L0:.3f}_tboil_{PAPYRIIS_env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz")
residuals_opds_1rst = first_stage_results['residuals_opds_1rst']
print(first_stage_results.files)
print(first_stage_results['telescope_pupil'].shape)
print(first_stage_results['reconstructed_cmd'].shape)
print(first_stage_results['src_opds'].shape)
print(first_stage_results['residuals_opds_1rst'].shape)


dm_commands = np.zeros((30000, 97))
reconstructed_cmd = np.zeros((30000, 97))
scnd_stage_strehl = np.zeros((30000))
tel_2nd_pupil = 0
src_opd = np.zeros((30000, 90, 90))
projector_kl_2nd = np.zeros((87, 8100))


obs, _ = PAPYRIIS_env.reset(residuals_opds_1rst)
a = time.perf_counter()
for t in range(30000):
    action = 0 * obs.unsqueeze(0).unsqueeze(0)
    next_obs, INFO = PAPYRIIS_env.step(action.squeeze(), residuals_opds_1rst)
    print(f'2nd strehl ratio {INFO['2nd_stage_strehl']}')
    dm_commands[t] =INFO["dm_commands"]
    reconstructed_cmd[t] =INFO["reconstructed_cmd"].detach().cpu().numpy()
    scnd_stage_strehl[t] =INFO["2nd_stage_strehl"]
    src_opd[t] =INFO["src_opd"]
    tel_2nd_pupil = INFO["telescope_pupil"]
    projector_kl_2nd = INFO["projector_kl_2nd"]

    obs = next_obs

print(time.perf_counter() - a)
np.savez(f"{savedir_test}/results_2nd_stage_CL1OL2.npz",
    # Concatenated across iterations
    all_2nd_stage_strehl    = scnd_stage_strehl,
    all_dm_commands         = dm_commands,
    all_reconstructed_cmd   = reconstructed_cmd,
    residual_opds_2nd       = src_opd,
    telescope_pupil         = tel_2nd_pupil,
    projector_kl_2nd        = projector_kl_2nd,
    )
