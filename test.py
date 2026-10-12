# %%
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import sys
repo = '/home/evinerskas/LAM_OOPAO'
if repo not in sys.path:
    sys.path.insert(0, repo)




# %%
from PAPYRIIS_2stage_CNN_RL.OOPAO_PAPYRIIS_env import OOPAO_environment_PAPYRIIS


PAPYRIIS_env = OOPAO_environment_PAPYRIIS()

# %%

#TODO read Mathieu's paper
#TODO talk with Taissir, Mathieu and Francisco (today)
#TODO doktoranturos darymas tai ne sedejimas iki 20:00, o kolaboravimas su zmonemis draugeli


#TODO summarise all of the results so far in your slides
#TODO redo strehl calculation whatever good that would do (120 nm of NCPA)
#TODO rerun the gain scan using a different wavelength
#TODO modulation radius is effectively different? (modulation scan 3-6)
#TODO create a separate control theory project with all of theoretical control functions starting with leaky integrator
        #this could also be used for wind fitting and stuff like that
#TODO finish PO4AO gui
#TODO run with 10% rejection for the pyramid signal
#TODO there was a DM scaling issue for the 2nd stage somewhere?
    #ask claude to go through the code and identify how the influence function peak is determined in simulations
    #it should be the sign value
    #on one hand the DM will converge to its shape, on the other the frame to frame correction would be suboptimal due to incorrect scaling
    #on the third hand, the CNN uses a 1e-5 sign in example.ipynb so...


#TODO YOU NEED TO DO A SINGLE RUN WITH THE NEW ATM AND PROJECTIONS TO SANITY CHECK
#TODO reconstruct the phase using arctangent reconstruction from Mathieu's code!!!

#so let's think this through:
    #CNN outputs mode amplitudes from the observed signal -> these should be true
    #the modes are converted to control commands via M2C (I load it from Mathieu and Francisco, you can compare these)
    #they are converted to the 1st stage using the influence functions
    #so I just need to plot the KL variance in 2nd stage basis

#some more thinking:
    #the influence functions map a unit of control to a set OPD size
    #so a deformable mirror unit command on 2nd stage that corresponded to 16e-6
    #would then be converted to approx 4 unit commands in the 1st stage (say 4e-6)
    #if I then used an incorrect projection with 10-e6, then I would necessarily get
    #2.5 unit commands on the 1st stage resulting in a smaller residual
        #a quick check would be to change the 2nd dm influence function size
        #and compare it against another size to see if the 1st stage residuals change

#The solution has been found:
    #the way the 2nd stage deformable mirror was constructed was incorrect
    #it definitely had the wrong sign parameter
    #this did not affect the atmosphere since for that we had the alpao unit 
    #so this is it, this is where you die


#TODO I will just use sign = 1.6e-5 and it will be good enough
# %%

from functions import KL_variance_calculator

# %%
from functions import *

loaddir = 'bench_sky_04_15/arcturus_v7'
RL_telemetry_1st = np.load(f'{loaddir}/2026-04-16T01_19_12_telemetry_data_RLiter50.npy', allow_pickle = True)
CL1st_OL_2nd     = np.load(f'{loaddir}/2026-04-16T01_27_23_telemetry_2nd_data_CL1OL2_noCRED2.npy', allow_pickle = True)

dm_1st_inf, pupil_mask_1 = first_stage_dm_builder()
dm_1st_inf_reshaped = dm_1st_inf.reshape((100, 100, -1))


dm_2nd_inf, pupil_mask_2 = second_stage_dm_builder(sign = 1e-5)    # simulated data
dm_2nd_inf_16e6, _ = second_stage_dm_builder(sign = 1.6e-5)        # on sky data



# %%
print(np.abs(dm_2nd_inf[ :, 59]).max())
print(np.abs(dm_2nd_inf_16e6[:, 59]).max())

# %%

dm_1st_inf = dm_1st_inf.reshape(100, 100, 241)
dm_1st_inf_masked = dm_1st_inf[pupil_mask_1, :]

dm_2nd_inf = dm_2nd_inf.reshape(100, 100, 97)
dm_2nd_inf_masked = dm_2nd_inf[pupil_mask_2, :]

dm_2nd_inf_16e6 = dm_2nd_inf_16e6.reshape(100, 100, 97)
dm_2nd_inf_16e6_masked = dm_2nd_inf_16e6[pupil_mask_2, :]



IF = np.load('PAPYRIIS_2stage_CNN_RL/IF_dm2.npy') * 1e-06
IF_sim = dm_2nd_inf


amplitude_mean = np.ptp(IF, axis=0)
amplitude_mean_sim = np.ptp(IF_sim, axis = 0)
IF_sim_scaled = IF_sim * amplitude_mean / np.ptp(IF_sim)


IF_sim_scaled = IF_sim_scaled[pupil_mask_2, :]
# dm_2nd_inf_16e6_masked = IF_sim_scaled   # uncomment to use the IF_dm2 ptp-scaled DM for the on sky data instead of sign = 1.6e-5

#TODO I will just use sign = 1.6e-5 and it will be good enough

print(IF_sim_scaled.shape)
print(dm_2nd_inf_16e6_masked.shape)

plt.figure()
plt.plot(np.ptp(IF_sim_scaled, axis = 0), label = 'IF_sky')
plt.plot(np.ptp(dm_2nd_inf_16e6_masked, axis = 0), label = '16e-6')
plt.plot(np.ptp(dm_2nd_inf_masked, axis = 0), label = '10e-6')
plt.legend()
plt.show()


# %%
M2C_1st = - np.load('PAPYRIIS_2stage_CNN_RL/M2C_1rst.npy')
M2C_2nd = np.load('PAPYRIIS_2stage_CNN_RL/M2C_2nd.npy')
M2C_2nd  = -np.load('PAPYRIIS_2stage_CNN_RL/M2C_KL.npy')[:, :66]
#TODO these M2C matrices are definitely different
    #does not seem to affect the KL modes that much



# %%

C2M_2nd = np.linalg.pinv(M2C_2nd)
M2P_1st = dm_1st_inf_masked @ M2C_1st
M2P_2nd = dm_2nd_inf_masked @ M2C_2nd
M2P_2nd_16e6 = dm_2nd_inf_16e6_masked @ M2C_2nd


projector_kl_2nd_to_1st = np.linalg.pinv(dm_1st_inf_masked @ M2C_1st) @ (dm_2nd_inf_masked @ M2C_2nd)
projector_kl_2nd_to_1st_16e6 = np.linalg.pinv(dm_1st_inf_masked @ M2C_1st) @ (dm_2nd_inf_16e6_masked @ M2C_2nd)



# M2P_1st scaled to nm before computing covariance
GG_diag_1st = mode_covariance(M2P_1st * 1e9)
GG_diag_2nd = mode_covariance(M2P_2nd * 1e9)
GG_diag_2nd_16e6 = mode_covariance(M2P_2nd_16e6 * 1e9)

KL_frames     = [1, -1]  



next_states_1st = CL1st_OL_2nd.item()['slavedreconsCube'].squeeze() #1st stage residual as measured by 2nd stage
modes_1st_stage_2nd_base = next_states_1st @ C2M_2nd.T
modes_1st_stage_1st_base = next_states_1st @ C2M_2nd.T @ projector_kl_2nd_to_1st_16e6.T


coefs_var_1st_stage_2nd_base, modes_1st_stage_masked_2nd_base, coefs_var_1st_stage_2nd_base_totalerr = KL_variance_calculator(modes_1st_stage_2nd_base, GG_diag_2nd_16e6, KL_frames)
coefs_var_1st_stage_1st_base, modes_1st_stage_masked_1st_base, coefs_var_1st_stage_1st_base_totalerr = KL_variance_calculator(modes_1st_stage_1st_base, GG_diag_1st, KL_frames)


modes_atm = mode_calculator_fromDM(RL_telemetry_1st.item()['dmCmdCube'].squeeze(), M2C_1st)
coefs_var_atm, modes_atm_masked, coefs_var_atm_totalerr = KL_variance_calculator(modes_atm, GG_diag_1st, KL_frames)


simdir = "PAPYRIIS_2stage_CNN_RL/2026-09-23"
simulations = [
    {"label": "new interaction", "folder": f"{simdir}/PAPYRIIS_arcturus_v2_pushpull", "file_1st": "results_1st_stage_r0_0.070_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz", "file_2nd": "results_2nd_stage_CL1OL2_r0_0.070_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz"},
    # {"label": "old interaction", "folder": f"{simdir}/PAPYRIIS_arcturus_v2", "file_1st": "results_1st_stage_r0_0.070_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz", "file_2nd": "results_2nd_stage_CL1OL2.npz"},
    # {"label": "gain = 0.6", "folder": f"{simdir}/PAPYRIIS_arcturus_v2_pushpull_g06", "file_1st": "results_1st_stage_r0_0.070_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz", "file_2nd": "results_2nd_stage_CL1OL2_r0_0.070_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz"},
    # {"label": "gain = 0.7", "folder": f"{simdir}/PAPYRIIS_arcturus_v2_pushpull_g07", "file_1st": "results_1st_stage_r0_0.070_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz", "file_2nd": "results_2nd_stage_CL1OL2_r0_0.070_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz"},
]


# for sim in simulations:

dm_atm_coefs = np.load(f"{simdir}/PAPYRIIS_arcturus_v2_pushpull/results_1st_stage_r0_0.070_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz")['dm_commands']  # only this array is read from the npz
sim_modes_atm = mode_calculator_fromDM(dm_atm_coefs, M2C_1st)
sim_var_atm, sim_modes_atm_masked, sim_var_atm_totalerr = KL_variance_calculator(sim_modes_atm, GG_diag_1st, KL_frames)

path_2nd = f"{simdir}/PAPYRIIS_arcturus_v2_pushpull/results_2nd_stage_CL1OL2_r0_0.070_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz"
next_states_1st_sim = np.load(path_2nd)['all_reconstructed_cmd'].squeeze()

sim_modes_1st_base_2nd = next_states_1st_sim @ C2M_2nd.T
sim_modes_1st_base_1st = next_states_1st_sim @ C2M_2nd.T @ projector_kl_2nd_to_1st.T


sim_var_1st_base_2nd, sim_modes_1st_masked_base_2nd, sim_var_1st_base_2nd_totalerr = KL_variance_calculator(sim_modes_1st_base_2nd, GG_diag_2nd, KL_frames)
sim_var_1st_base_1st, sim_modes_1st_masked_base_1st, sim_var_1st_base_1st_totalerr = KL_variance_calculator(sim_modes_1st_base_1st, GG_diag_1st, KL_frames)


# %%

plt.close()
plt.figure(figsize=(19, 8))  # wider than the original (12, 8) to fit the legend on the right

plt.plot(coefs_var_atm, color="black", label=f"Atm, σ² = {coefs_var_atm_totalerr:,.0f} nm²")
plt.plot(sim_var_atm, ':', color="grey", label=f"Atm (sim,), σ² = {sim_var_atm_totalerr:,.0f} nm²")


plt.plot(coefs_var_1st_stage_1st_base, '--', color="blue", lw=2.5, label=f"1st stage (integrator)_1st_base, σ² = {coefs_var_1st_stage_1st_base_totalerr:,.0f} nm²")
# plt.plot(coefs_var_1st_stage_2nd_base, '--', color="orange", lw=2.5, label=f"1st stage (integrator)_2nd_base, σ² = {coefs_var_1st_stage_2nd_base_totalerr:,.0f} nm²")

plt.plot(sim_var_1st_base_1st, ':', color="blue", lw=2.5, label=f"1st stage (sim, base_1st), σ² = {sim_var_1st_base_1st_totalerr:,.0f} nm²")
# plt.plot(sim_var_1st_base_2nd, ':', color="orange", lw=2.5, label=f"1st stage (sim, base_2nd), σ² = {sim_var_1st_base_2nd_totalerr:,.0f} nm²")



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
plt.ylim(bottom = 5e-3)
plt.show()

# %%

#TODO compare this IF ptp with the simulated DM
#TODO how are the influence functions built inside OOPAO?
    #shouldn't the ptp be the same for every actuator? 
    #this is due to edge actuators

#TODO how is IF_dm2 used -> make a summary
    # normalised using np.ptp(IF_sim) and not np.ptp(IF_sim, axis = 0)
    # because it is not clear which actuator in simulation
    # corresponds to which actuator on sky
#TODO try plotting with the IF_dm2 for on sky results
#TODO I also need to use these new influence functions in the simulations?
    #can't really do that since the CNN deformable mirror was set up for 1e-5
    #you can still try and see what the CNN does

#TODO don't forget to also run the 2nd stage with a pyramid!!!
    #make environment copy

#TODO read this after toilet break:
    #1. IF_vZWFS is not used in simulations (only OZILinearity)
        #I think it was used for bench linearity tests
    #2. IF_dm2.npy is used in simulations
        #2.1 for interaction matrix building (you ofcourse have no use for the interaction matrix)
        #2.2 also used for ptp scaling of the synthetic dm.modes
IF = np.load('PAPYRIIS_2stage_CNN_RL/IF_dm2.npy') * 1e-06
IF_sim = PAPYRIIS_env.dm_2nd.modes


amplitude_mean = np.ptp(IF, axis=0)
amplitude_mean_sim = np.ptp(IF_sim, axis = 0)
IF_sim_scaled = IF_sim * amplitude_mean / np.ptp(IF_sim)

amplitude_mean_sim_scaled = np.ptp(IF_sim_scaled, axis = 0)


plt.figure()
plt.title('IF_sky vs sim')
plt.plot(amplitude_mean, label = 'sky')
plt.plot(amplitude_mean_sim, label = 'sim')
plt.plot(amplitude_mean_sim_scaled, label = 'sim_scaled')

plt.legend()

plt.show()


print(amplitude_mean.shape)
print(amplitude_mean_sim.shape)
# %%


M2C_2nd = np.load('PAPYRIIS_2stage_CNN_RL/M2C_2nd.npy')
M2C_2nd_KL = -np.load('PAPYRIIS_2stage_CNN_RL/M2C_KL.npy')[:, :66]



Gram_v1 = M2C_2nd.T @ M2C_2nd
v1 = np.sqrt(np.diag(Gram_v1))
Gram_v1 = Gram_v1 / np.outer(v1, v1)


Gram_v2 = M2C_2nd_KL.T @ M2C_2nd_KL
v2 = np.sqrt(np.diag(Gram_v2))
Gram_v2 = Gram_v2 / np.outer(v2, v2)

Gram_v3 = M2C_2nd.T @ M2C_2nd_KL
v3 = np.sqrt(np.abs(np.diag(Gram_v3)))
Gram_v3 = Gram_v3 / np.outer(v3, v3)


print(M2C_2nd.shape)
print(M2C_2nd_KL.shape)

mode = 2

#TODO it is still not clear to me how Mathieu's DM building results in the correct influence functions
    # and of course it seems that Francisco did not use those influence functions but only his own dm
    # so for simulations I use Francisco's stuff
    # while for on sky plotting I use the correct DM model (you can try both yours and Mathieu's)

#TODO do one run with M2C_KL and one with M2C from Mathieu for the CNN
#TODO ok, so on the M2C front is settled
    #TODO now I need to get the actual residual error from the vZWFS images and that will be that
    # then tune the simulations to get the same result more or less
    # and go on from there!
    # make a list of all the new needed simulations 
    # since the PAPYTWIN is more or less correct, the atm fit should be the target (calculate the error in first 10 modes for a baseline)
    # then figure out the CL1OL2 OPD from the vZWFS arctan reconstruction and you are set for life
    # implement your plan today!
    # might need to do it on the LAM server smh...
plt.close()
plt.figure()
plt.imshow(Gram_v1)
plt.colorbar()


plt.figure()
plt.imshow(Gram_v2)
plt.colorbar()

plt.figure()
plt.imshow(Gram_v3)
plt.colorbar()


# %%
