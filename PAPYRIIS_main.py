"""
This file is for simulating the PAPYRUS + OZIRIIS pipeline (CNN for vZWFS reconstruction)


little description of the CNN problem (central obstruction + sky pupil)
#TODO the problem is that the on sky pupil crops below 90X90 which what CNN requires
    # I made the executive decision to use a 90x90 non-masked pupil with on_sky set to True for OZIRIIS
    # because the on_sky label is necessary
    # and I am using the on sky CNN
    # another problem this introduces is that the pupil is also shifted compared to 1st stage
    # but redoing the 1st stage with the proper pupil is hard so we leave it
    #in conclusion: is_onsky = True + onsky CNN + 0 pupil mask (introducing central obstruction causes divergence after iteration 30)
    # cameras are fixed (for vZWFS bits =None (reduces performance alot and even diverges), FWC = None (overflows))
"""



import numpy as np
from functions import *
import time
from PAPYRIIS_2stage_CNN_RL.OOPAO_PAPYRIIS_env import OOPAO_environment_PAPYRIIS
import os

#----------------------------------------------------Generate 2nd atmosphere#----------------------------------------------------
def name_tag(env):
    return f"r0_{env.atm_2nd.r0:.3f}_V0_{env.atm_2nd.V0:.3f}_L0_{env.atm_2nd.L0:.3f}_tboil_{env.atm_2nd.t_boiling[0]:.3f}"


def atm_gen(savedir, env, sim_length):
    print(f"{name_tag(env)}")
    OPD_screen_1 = env.generate_second_stage_atmosphere(nLoop=sim_length)

    os.makedirs(savedir, exist_ok = True)


    np.savez_compressed(
        f"{savedir}/atm_OPDs_2nd_{name_tag(env)}_multi_layer.npz",
        atm_OPDs_2nd=OPD_screen_1,
        r0=env.atm_2nd.r0,
        L0=env.atm_2nd.L0,
        windSpeed=env.atm_2nd.windSpeed,
        fractionalR0=env.atm_2nd.fractionalR0,
        windDirection=env.atm_2nd.windDirection,
        altitude=env.atm_2nd.altitude
    )


#---------------------------------------------------Project atmosphere to 1st stage#----------------------------------------------------
def proj_1st_stage(loaddir, savedir, env):
    atm_OPDs_2nd = np.load(f"{loaddir}/atm_OPDs_2nd_{name_tag(env)}_multi_layer.npz")
    atm_OPDs_2nd = atm_OPDs_2nd["atm_OPDs_2nd"]
    print(f"atm loaded: _{name_tag(env)}_multi_layer")

    os.makedirs(savedir, exist_ok = True)

    atm_OPDs_1st = env.project_atmosphere_to_first_stage(atm_OPDs_2nd)


    np.savez(f"{savedir}/atm_OPDs_1st_{name_tag(env)}_multi_layer.npz",
            atm_OPDs_1st=atm_OPDs_1st,
            r0=env.atm_2nd.r0,
            L0=env.atm_2nd.L0,
            windSpeed=env.atm_2nd.windSpeed,
            fractionalR0=env.atm_2nd.fractionalR0,
            windDirection=env.atm_2nd.windDirection,
            altitude=env.atm_2nd.altitude
    )



#----------------------------------------------------1st stage CL#----------------------------------------------------

def run_1st_stage(loaddir, savedir, env, sim_length, gainCL):

    atm_OPD_1st = np.load(f"{loaddir}/atm_OPDs_1st_{name_tag(env)}_multi_layer.npz")
    atm_OPD_1st = atm_OPD_1st["atm_OPDs_1st"]
    print(f"atm loaded: _{name_tag(env)}_multi_layer")

    os.makedirs(savedir, exist_ok = True)

    first_stage_results = env.run_first_stage_loop(sim_length, atm_OPD_1st, gainCL = gainCL)


    np.savez(f"{savedir}/results_1st_stage_{name_tag(env)}_multi_layer.npz", **{
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
def run_2nd_stage_CL1OL2(loaddir, savedir, env, sim_length):
    print(f"{loaddir}/results_1st_stage_{name_tag(env)}_multi_layer.npz")
    first_stage_results = np.load(f"{loaddir}/results_1st_stage_{name_tag(env)}_multi_layer.npz")
    residuals_opds_1rst = first_stage_results['residuals_opds_1rst']

    os.makedirs(savedir, exist_ok = True)

    dm_commands = np.zeros((sim_length, 97))
    reconstructed_cmd = np.zeros((sim_length, 97))
    scnd_stage_strehl = np.zeros((sim_length))
    tel_2nd_pupil = 0
    src_opd = np.zeros((sim_length, 90, 90))
    projector_kl_2nd = np.zeros((87, 8100))


    obs, _ = env.reset(residuals_opds_1rst)
    t_start = time.perf_counter()
    for t in range(sim_length):
        action = 0 * obs.unsqueeze(0).unsqueeze(0)
        next_obs, info = env.step(action.squeeze(), residuals_opds_1rst)
        if t%100 == 0:
            t_block = time.perf_counter()
            print(f'iteration {t}, 2nd strehl ratio {info['2nd_stage_strehl']}')

        if (t+1)%100 == 0:
            print(time.perf_counter() - t_start)
        dm_commands[t] =info["dm_commands"]
        reconstructed_cmd[t] =info["reconstructed_cmd"].detach().cpu().numpy()
        scnd_stage_strehl[t] =info["2nd_stage_strehl"]
        src_opd[t] =info["src_opd"]
        tel_2nd_pupil = info["telescope_pupil"]
        projector_kl_2nd = info["projector_kl_2nd"]

        obs = next_obs

    print(time.perf_counter() - t_block)
    np.savez(f"{savedir}/results_2nd_stage_CL1OL2_{name_tag(env)}_multi_layer.npz",
        # Concatenated across iterations
        all_2nd_stage_strehl    = scnd_stage_strehl,
        all_dm_commands         = dm_commands,
        all_reconstructed_cmd   = reconstructed_cmd,
        residual_opds_2nd       = src_opd,
        telescope_pupil         = tel_2nd_pupil,
        projector_kl_2nd        = projector_kl_2nd,
        )



if __name__ == '__main__':
    PAPYRIIS_env = OOPAO_environment_PAPYRIIS()
    savedir_atm = "PAPYRIIS_2stage_CNN_RL/generated_atm_2nd_stage"
    savedir_proj = "PAPYRIIS_2stage_CNN_RL/projected_atm_1st_stage"
    savedir_1st = "PAPYRIIS_2stage_CNN_RL/2026-09-23/PAPYRIIS_arcturus_v4_nophoton_cred"
    savedir_2nd = "PAPYRIIS_2stage_CNN_RL/2026-09-23/PAPYRIIS_arcturus_v4_nophoton_cred" 
    simulation_length  = 15000
    gain_1st = 0.7


    #TODO I changed the wvl to "R" since optical gains depend on wavelength
    
    atm_gen(savedir = savedir_atm, env = PAPYRIIS_env, sim_length = simulation_length)
    proj_1st_stage(loaddir = savedir_atm, savedir = savedir_proj, env = PAPYRIIS_env)
    # run_1st_stage(loaddir = savedir_proj, savedir = savedir_1st, env = PAPYRIIS_env, gainCL = gain_1st, sim_length = simulation_length)
    # run_2nd_stage_CL1OL2(loaddir = savedir_1st, savedir = savedir_2nd, env = PAPYRIIS_env, sim_length = simulation_length)

    #TODO
        #the projection might be problematic in 3D?
        #but no, how it could be, pwfs reconstruction at high temporal frequencies was fine
        # YOU SHOULD STILL DO IT, it will not take long
        # and then do one analysis run for comparison innit
    #TODO redo the r_0 runs with different wind speeds (3 wind speeds per r_0, so 12 processes not including the ones I already have)
        #r_0 0.05 -> 2, 4
        #r_0 0.07 -> 2, 3, 4
        #r_0 0.085 -> 2, 3, 4
