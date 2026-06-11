"""
*when doing the temporal PSD you should also implement some kind of strehl condition
	problem is that then the temporal curve can get discontinuities
	the most straightforward solution seems to be just cutting out a continuous section of the timeseries where the strehl ratio is above your threshold
* write try except statements for when plots are non-sensical

"""

from scipy import signal
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors



#TODO don't forget to also change your atmosphere parameters when loading files (atm_OPD_array)
#TODO could you in fact rewrite this so unneccesary stuff is not loaded?
RL                   = True
integrator           = True
ideal                = False
only_2nd             = False
stage_2              = True
stage_1_plus_stage_2 = True
atm_RL               = True
atm_int              = False
atm_ideal            = False
plot_timeseries      = True
plot_tPSD            = True
use_1st_KL           = True
freq_lim             = 900
stage_1_freq_lim     = 250

#bench_loss_2/test_warmup50_1st250_2nd500_scaling1e2_noise_losspenalty01_reduceaction10_iters100_v1
#vZWFS_1st_2nd_noise_03_phnoise1_read_4_4_QE08_mag2_epis20_iters80_weight_decay=1e-6
#test_warmup50_2nd500_scaling1e2_noise_losspenalty01_reduceaction10_iters100_v10
#TODO change the atmosphere used in plotting
#TODO look into the policy output innit
CL_gain_pyr = 0.3
folder_name = "bench_sky_04_16"
frequency_1st = 400 #should be the same as 2nd if you are using CL1OL2 with 2nd stage looking at the 1st stage residuals
frequency_2nd = 400
frequency_atm = 200
directory_name_RL  = 'onsky_arcturus_1st200_2nd400_v8_linear_20260417-040839'#test_warmup50_1st250_2nd500_scaling1e2_noise_losspenalty01_reduceaction10_iters100_v1
#directory_name_RL_atm  = 'bench_loss_4/test_warmup20_1st150_2nd300_scaling1e2_noise_losspenalty01_reduceaction1_iters500_v15' #TODO fix atmosphere saving
#directory_name_int = 'test_warmup50_1st200_2nd400_scaling1e2_noise2ndstageonly_losspenalty01_iters300_v21_int' #vZWFS_integrator_phnoise1_read_4_4_QE08_mag2_12k
#directory_name_ideal = 'vZWFS_metrics_ideal'#vZWFS_metrics_ideal
label_RL = "RL"
label_int = "integrator"
label_ideal = "ideal"
KL_frame_thresh_RL = 1
KL_frame_end_RL    = -1 #18750
KL_frame_thresh_int = 1
KL_frame_end_int    = -1 #18750
KL_frame_thresh_ideal = 1
KL_frame_end_ideal    = -1
#modes for PSDs
mode_1 = 0
mode_2 = 10
mode_3 = 20
mode_4 = 30
mode_5 = 40
loaddir = 'bench_sky_04_16/onsky_arcturus_1st200_2nd400_v8_linear_20260417-040839'

RL_telemetry_2nd = np.load(f'{loaddir}/2026-04-17T04_14_34_telemetry_2nd_data_RLiter100.npy', allow_pickle = True)
RL_telemetry_1st = np.load(f'{loaddir}/2026-04-17T04_15_15_telemetry_data_RLiter100.npy', allow_pickle = True)

int_telemetry_2nd = np.load(f'{loaddir}/2026-04-17T04_18_06_telemetry_2nd_data_pythint.npy', allow_pickle = True)
#int_telemetry_1st = np.load(f'{loaddir}/2026-04-15T22_15_59_telemetry_data_pythint.npy', allow_pickle = True)

CL1st_OL_2nd = np.load(f'{loaddir}/2026-04-17T04_21_42_telemetry_2nd_data_CL1OL2.npy', allow_pickle = True)
Full_OL = np.load(f'{loaddir}/2026-04-17T04_24_07_telemetry_data_OL.npy', allow_pickle = True)
#Full_OL = np.load('bench_sky_04_15/2026-04-15T23_31_07_telemetry_data_OL500.npy', allow_pickle = True)
#print(testie.item()['s2m'])
print(RL_telemetry_1st.item().keys())
print(RL_telemetry_2nd.item().keys())
print(RL_telemetry_2nd.item()['slavedreconsCube'].shape)


valid_mask_1st_stage = np.load(f'{folder_name}/valid_mask_1st_stage.npy')
dm_x_1st_stage, dm_y_1st_stage = np.where(valid_mask_1st_stage)
dm_coords_1st_stage = (dm_x_1st_stage, dm_y_1st_stage)


valid_mask_2nd_stage = np.load(f'{folder_name}/valid_mask_2nd_stage.npy')
dm_x_2nd_stage, dm_y_2nd_stage = np.where(valid_mask_2nd_stage)
dm_coords_2nd_stage = (dm_x_2nd_stage, dm_y_2nd_stage)



#influence functions (control to phase)
dm_1st_inf = np.reshape(np.load(f'{folder_name}/Papyrus_influence_functions_in_nanometers.npy'), (80 * 80, 241))
dm_2nd_inf = np.load(f'{folder_name}/OZIRIIS_Zonal_80x80.npy')
dm_2nd_inf = np.moveaxis(dm_2nd_inf, 0, -1)


#TODO investigate this
#print(dm_1st_inf[dm_1st_inf < 0].shape)
#print(np.min(dm_2nd_inf))


#1st stage modes to control
#M2C_1st = np.load(f"{folder_name}/M2C_1st.npy")
M2C_1st = RL_telemetry_1st.item()['m2c']
#2nd stage modes to control
#M2C_2nd = np.load(f"{folder_name}/M2C_2nd.npy")
M2C_2nd = RL_telemetry_2nd.item()['m2c']


C2M_1st = np.linalg.pinv(M2C_1st)
C2M_2nd = np.linalg.pinv(M2C_2nd)


M2P_1st = dm_1st_inf @ M2C_1st
M2P_2nd = dm_2nd_inf @ M2C_2nd
projector_kl = np.linalg.pinv(M2P_1st) @ M2P_2nd



#RL
if RL:
    #residual_error_RL = np.load(f"{folder_name}/{directory_name_RL}/residual_error.npy") TODO I think you can extract this using influence functions

    cred_timestamp_RL = RL_telemetry_2nd.item()['timeStampDmCube']

    if stage_2:
        #strehl_array_2nd_RL = np.load(f"{folder_name}/{directory_name_RL}/strehl_array_2nd.npy") #psf fitting here
        next_states_2nd = RL_telemetry_2nd.item()['slavedreconsCube'].squeeze()
        modes_2nd_stage_RL = next_states_2nd @ C2M_2nd.T
        if use_1st_KL:
            modes_2nd_stage_RL = modes_2nd_stage_RL @ projector_kl.T




    if only_2nd:
        modes_1st_stage_RL = modes_2nd_stage_RL
    else:
        next_states_1st = CL1st_OL_2nd.item()['slavedreconsCube'].squeeze()
        modes_1st_stage_RL = next_states_1st @ C2M_2nd.T
        if use_1st_KL:
            modes_1st_stage_RL = modes_1st_stage_RL @ projector_kl.T
        #next_states_1st = RL_telemetry_1st.item()['modeCube'].squeeze() #this is already flattened, no need to apply a mask
        #modes_1st_stage_RL = next_states_1st# @ C2M_1st.T (it is already in modes)


    #dm_atm = np.load(f"{directory_name_RL_atm}/dm_atm.npy").squeeze()
    dm_atm = Full_OL.item()['modeCube'].squeeze()
    modes_atm_RL = dm_atm# @ C2M_1st.T
    #modes_atm_RL = modes_atm_RL

    

    dynamics_loss = np.load(f"{folder_name}/{directory_name_RL}/dynamics_loss_warmup.npy") 
    policy_loss = np.load(f"{folder_name}/{directory_name_RL}/policy_loss_warmup.npy") 


    #YOU SHOULD ALSO CHECK IF YOU ONLY HAVE AN EPISODE OF DATA IN ALL OF THESE BECAUSE I DON4T REMEMBER
    frequency_RL = frequency_2nd
    time_plot_RL = np.arange(0, 30000 / frequency_2nd, 1 / frequency_2nd) #time_plot = np.arange(0, iters * episode_length / frequency, 1/frequency)






#integrator
if integrator:
    #residual_error_int = np.load(f"{folder_name}/{directory_name_int}/residual_error.npy")

    cred_timestamp_int = int_telemetry_2nd.item()['timeStampDmCube']


    if stage_2:
        #strehl_array_2nd_int = np.load(f"{folder_name}/{directory_name_int}/strehl_array_2nd.npy")
        next_states_2nd = int_telemetry_2nd.item()['slavedreconsCube'].squeeze()
        modes_2nd_stage_int = next_states_2nd @ C2M_2nd.T
        if use_1st_KL:
            modes_2nd_stage_int = modes_2nd_stage_int @ projector_kl.T


    if only_2nd:
        modes_1st_stage_int = modes_2nd_stage_int
    else:
        next_states_1st = CL1st_OL_2nd.item()['slavedreconsCube'].squeeze()
        modes_1st_stage_int = next_states_1st @ C2M_2nd.T
        if use_1st_KL:
            modes_1st_stage_int = modes_1st_stage_int @ projector_kl.T
        #next_states_1st = int_telemetry_1st.item()['modeCube'].squeeze() #this is already flattened, no need to apply a mask
        #modes_1st_stage_int = next_states_1st# @ C2M_1st.T

    #dm_atm = np.load(f"{folder_name}/{directory_name_int}/dm_atm.npy").squeeze()[:5000]
    #modes_atm_int = dm_atm @ C2M_1st.T
    #modes_atm_int = modes_atm_int



    frequency_int = frequency_2nd
    time_plot_int = np.arange(0, 30000 / frequency_2nd, 1 / frequency_2nd)



if atm_RL:
    modes_atm = modes_atm_RL
    time_plot_atm = time_plot_RL
    f_samp_atm = frequency_atm

if atm_int:
    modes_atm = modes_atm_int
    time_plot_atm = time_plot_int
    f_samp_atm = frequency_atm



def welch_method_scipy(data, fs, nperseg=256):
    frequencies, psd = signal.welch(
        data,
        fs=fs,
        window='hann',  #windowing
        nperseg=nperseg,
        scaling='density'
    )
    return frequencies, psd



# ---------------------------------------------------Strehl---------------------------------------------------#
#plot comparison between RL, pyth int and 1st stage performance (CL1OL2)
#a few plots for seeing

time_plot = np.arange(0, 12.5, 0.25)

if RL:
    RL_PSF_tele = np.load(f'{loaddir}/results_RLiter100.npz')
    PSF_tele_1st = np.load(f'{loaddir}/results_CL1OL2.npz')
    strehl_RL_2nd = RL_PSF_tele['strehl']
    strehl_1st = PSF_tele_1st['strehl']
    seeing     = np.load(f'{loaddir}/results_OL.npz')
    seeing     = seeing['seeing']

if integrator:
    pythint_PSF_tele = np.load(f'{loaddir}/results_pythint.npz')
    strehl_pythint_2nd = pythint_PSF_tele['strehl']
    daoint_PSF_tele = np.load(f'{loaddir}/results_dao.npz')
    strehl_daoint_2nd = daoint_PSF_tele['strehl']

plt.figure()
plt.plot(time_plot, seeing, color = 'orange')
plt.title("Seeing over time")
plt.xlabel("Time (s)")
plt.ylabel('Seeing')


plt.figure()
if RL:
    if stage_2:
        if stage_1_plus_stage_2:
            plt.plot(time_plot, strehl_1st, '--', color="indianred", label=f"integrator 1st stage")
        plt.plot(time_plot, strehl_RL_2nd, color="red", label=f"RL 2nd stage")
    else: plt.plot(time_plot, strehl_1st, '--', color="indianred", label=f"integrator 1st stage")
if integrator:
    if stage_2:
        plt.plot(time_plot, strehl_pythint_2nd, color="blue", label=f"pyth integrator 2nd stage")

plt.title("Strehl ratio")
plt.xlabel("Time (s)")
plt.ylabel('Strehl ratio')
#plt.ylim(bottom=(sr_mean_RL - 0.5))
#plt.grid(True, alpha=0.5)
#plt.minorticks_on()
plt.legend()


plt.figure()
if RL:
    if stage_2:
        if stage_1_plus_stage_2:
            plt.plot(time_plot, strehl_1st, '--', color="indianred", label=f"integrator 1st stage")
        plt.plot(time_plot, strehl_RL_2nd, color="red", label=f"RL 2nd stage")
    else: plt.plot(time_plot, strehl_1st, '--', color="indianred", label=f"integrator 1st stage")
if integrator:
    if stage_2:
        plt.plot(time_plot, strehl_daoint_2nd, color="blue", label=f"dao integrator 2nd stage")


plt.title("Strehl ratio")
plt.xlabel("Time (s)")
plt.ylabel('Strehl ratio')
#plt.ylim(bottom=(sr_mean_RL - 0.5))
#plt.grid(True, alpha=0.5)
#plt.minorticks_on()
plt.legend()


# ---------------------------------------------------Jitter---------------------------------------------------#
#TODO plot the PSD plots for the jitter
#compare jitter from RL, pythint and daoint
frequency_psf = 400
jitter_x_timeseries_RL = RL_PSF_tele['cx']
jitter_y_timeseries_RL = RL_PSF_tele['cy']

jitter_x_timeseries_pythint = pythint_PSF_tele['cx']
jitter_y_timeseries_pythint = pythint_PSF_tele['cy']

jitter_x_timeseries_daoint = daoint_PSF_tele['cx']
jitter_y_timeseries_daoint = daoint_PSF_tele['cy']

jitter_x_timeseries_1st = PSF_tele_1st['cx']
jitter_y_timeseries_1st = PSF_tele_1st['cy']



jitter_x_PSD_freq_RL, jitter_x_PSD_RL = welch_method_scipy(jitter_x_timeseries_RL, frequency_psf)
jitter_y_PSD_freq_RL, jitter_y_PSD_RL = welch_method_scipy(jitter_y_timeseries_RL, frequency_psf)

jitter_x_PSD_freq_pythint, jitter_x_PSD_pythint = welch_method_scipy(jitter_x_timeseries_pythint, frequency_psf)
jitter_y_PSD_freq_pythint, jitter_y_PSD_pythint = welch_method_scipy(jitter_y_timeseries_pythint, frequency_psf)

jitter_x_PSD_freq_daoint, jitter_x_PSD_daoint = welch_method_scipy(jitter_x_timeseries_daoint, frequency_psf)
jitter_y_PSD_freq_daoint, jitter_y_PSD_daoint = welch_method_scipy(jitter_y_timeseries_daoint, frequency_psf)

jitter_x_PSD_freq_1st, jitter_x_PSD_1st = welch_method_scipy(jitter_x_timeseries_1st, frequency_psf)
jitter_y_PSD_freq_1st, jitter_y_PSD_1st = welch_method_scipy(jitter_y_timeseries_1st, frequency_psf)



plt.figure()
if RL:
    if stage_2:
        if stage_1_plus_stage_2:
            plt.plot(jitter_x_PSD_freq_1st, jitter_x_PSD_1st, '--', color="indianred", label=f"x jitter 1st stage")
        plt.plot(jitter_x_PSD_freq_RL, jitter_x_PSD_RL, color="red", label=f"x jitter 2nd stage (RL)")
    else: plt.plot(jitter_x_PSD_freq_1st, jitter_x_PSD_1st, color="indianred", label=f"x jitter 1st stage")
if integrator:
    if stage_2:
        plt.plot(jitter_x_PSD_freq_pythint, jitter_x_PSD_pythint, color="blue", label=f"x jitter 2nd stage (python integrator)")



plt.title(f"PSF jitter in x-axis (python integrator)")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("PSD")



plt.figure()
if RL:
    if stage_2:
        if stage_1_plus_stage_2:
            plt.plot(jitter_x_PSD_freq_1st, jitter_x_PSD_1st, '--', color="indianred", label=f"x jitter 1st stage")
        plt.plot(jitter_x_PSD_freq_RL, jitter_x_PSD_RL, color="red", label=f"x jitter 2nd stage (RL)")
    else: plt.plot(jitter_x_PSD_freq_1st, jitter_x_PSD_1st, color="indianred", label=f"x jitter 1st stage")
if integrator:
    if stage_2:
        plt.plot(jitter_x_PSD_freq_daoint, jitter_x_PSD_daoint, color="blue", label=f"x jitter 2nd stage (dao integrator)")



plt.title(f"PSF jitter in x-axis (dao integrator)")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("PSD")


#y axis
plt.figure()
if RL:
    if stage_2:
        if stage_1_plus_stage_2:
            plt.plot(jitter_y_PSD_freq_1st, jitter_y_PSD_1st, '--', color="indianred", label=f"y jitter 1st stage")
        plt.plot(jitter_y_PSD_freq_RL, jitter_y_PSD_RL, color="red", label=f"y jitter 2nd stage (RL)")
    else: plt.plot(jitter_y_PSD_freq_1st, jitter_y_PSD_1st, color="indianred", label=f"y jitter 1st stage")
if integrator:
    if stage_2:
        plt.plot(jitter_y_PSD_freq_pythint, jitter_y_PSD_pythint, color="blue", label=f"y jitter 2nd stage (python integrator)")



plt.title(f"PSF jitter in y-axis (python integrator)")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("PSD")



plt.figure()
if RL:
    if stage_2:
        if stage_1_plus_stage_2:
            plt.plot(jitter_y_PSD_freq_1st, jitter_y_PSD_1st, '--', color="indianred", label=f"y jitter 1st stage")
        plt.plot(jitter_y_PSD_freq_RL, jitter_y_PSD_RL, color="red", label=f"y jitter 2nd stage (RL)")
    else: plt.plot(jitter_y_PSD_freq_1st, jitter_y_PSD_1st, color="indianred", label=f"y jitter 1st stage")
if integrator:
    if stage_2:
        plt.plot(jitter_y_PSD_freq_daoint, jitter_y_PSD_daoint, color="blue", label=f"y jitter 2nd stage (dao integrator)")



plt.title(f"PSF jitter in y-axis (dao integrator)")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("PSD")


# ---------------------------------------------------SNR---------------------------------------------------#
#I guess theoretically you want the snr per dm actuator (i.e. how many pixels per actuator)

gain = 2.41  # e-/ADU

vzwfs_frame_cube = (RL_telemetry_2nd.item()['CRED2Cube'] - RL_telemetry_2nd.item()['credDark']) * gain
print(vzwfs_frame_cube[:, 0, 0])
vzwfs_frame_cube[:, 0, 0] = 0
N_frames = vzwfs_frame_cube.shape[0]
average_frame = np.mean(vzwfs_frame_cube, axis = 0) # assumes frame_cube already background subtracted
valid_pix_map = average_frame > np.max(average_frame) * 0.1 #select this value to get the pupils or get the SHM with the valid pixels

print(vzwfs_frame_cube[:, 0, 0])

signal_snr = average_frame[valid_pix_map]
background = average_frame[~valid_pix_map]

flux_per_pixel = np.mean(signal_snr) - np.median(background)
snr = flux_per_pixel / np.sqrt(flux_per_pixel + np.var(background) * N_frames)

print("snr per pixel", snr)
print("snr summed", snr * np.sqrt(signal_snr.shape[0]))





# ---------------------------------------------------Normalised Temporal PSD---------------------------------------------------#



#RL
if RL:
    modes_1st_stage_RL_masked = modes_1st_stage_RL[KL_frame_thresh_RL:KL_frame_end_RL, :]
    coefs_var_1st_stage_masked_RL = np.var(np.asarray(modes_1st_stage_RL_masked), axis = 0)
    coefs_var_1st_stage_RL = np.var(np.asarray(modes_1st_stage_RL), axis = 0)

    if stage_2:
        modes_2nd_stage_RL_masked = modes_2nd_stage_RL[KL_frame_thresh_RL:KL_frame_end_RL, :]
        coefs_var_2nd_stage_masked_RL = np.var(np.asarray(modes_2nd_stage_RL_masked), axis=0)
        coefs_var_2nd_stage_RL = np.var(np.asarray(modes_2nd_stage_RL), axis = 0)


#integrator
if integrator:
    modes_1st_stage_int_masked = modes_1st_stage_int[KL_frame_thresh_int:KL_frame_end_int, :]
    coefs_var_1st_stage_masked_int = np.var(np.asarray(modes_1st_stage_int_masked), axis = 0)
    coefs_var_1st_stage_int = np.var(np.asarray(modes_1st_stage_int), axis=0)

    if stage_2:
        modes_2nd_stage_int_masked = modes_2nd_stage_int[KL_frame_thresh_int:KL_frame_end_int, :]
        coefs_var_2nd_stage_masked_int = np.var(np.asarray(modes_2nd_stage_int_masked), axis = 0)
        coefs_var_2nd_stage_int = np.var(np.asarray(modes_2nd_stage_int), axis = 0)


#atmosphere
coefs_var_atm = np.var(np.asarray(modes_atm[KL_frame_thresh_int:, :]), axis = 0)


if RL:
    f_samp_RL = frequency_RL
    # tip timeseries
    residual_mode_1_curve_1st_RL_full = modes_1st_stage_RL[:, mode_1]
    residual_mode_2_curve_1st_RL_full = modes_1st_stage_RL[:, mode_2]
    residual_mode_3_curve_1st_RL_full = modes_1st_stage_RL[:, mode_3]
    residual_mode_4_curve_1st_RL_full = modes_1st_stage_RL[:, mode_4]
    residual_mode_5_curve_1st_RL_full = modes_1st_stage_RL[:, mode_5]

    if stage_2:
        residual_mode_1_curve_2nd_RL_full = modes_2nd_stage_RL[:, mode_1]
        residual_mode_2_curve_2nd_RL_full = modes_2nd_stage_RL[:, mode_2]
        residual_mode_3_curve_2nd_RL_full = modes_2nd_stage_RL[:, mode_3]
        residual_mode_4_curve_2nd_RL_full = modes_2nd_stage_RL[:, mode_4]
        residual_mode_5_curve_2nd_RL_full = modes_2nd_stage_RL[:, mode_5]


if integrator:
    f_samp_int = frequency_int
    # tip timeseries
    residual_mode_1_curve_1st_int_full = modes_1st_stage_int[:, mode_1]
    residual_mode_2_curve_1st_int_full = modes_1st_stage_int[:, mode_2]
    residual_mode_3_curve_1st_int_full = modes_1st_stage_int[:, mode_3]
    residual_mode_4_curve_1st_int_full = modes_1st_stage_int[:, mode_4]
    residual_mode_5_curve_1st_int_full = modes_1st_stage_int[:, mode_5]

    if stage_2:
        residual_mode_1_curve_2nd_int_full = modes_2nd_stage_int[:, mode_1]
        residual_mode_2_curve_2nd_int_full = modes_2nd_stage_int[:, mode_2]
        residual_mode_3_curve_2nd_int_full = modes_2nd_stage_int[:, mode_3]
        residual_mode_4_curve_2nd_int_full = modes_2nd_stage_int[:, mode_4]
        residual_mode_5_curve_2nd_int_full = modes_2nd_stage_int[:, mode_5]



if RL:
    #RL 1st and 2nd stage selected mode PSD calculation
    residual_mode_1_curve_1st_RL = modes_1st_stage_RL_masked[:, mode_1]
    residual_mode_2_curve_1st_RL = modes_1st_stage_RL_masked[:, mode_2]
    residual_mode_3_curve_1st_RL = modes_1st_stage_RL_masked[:, mode_3]
    residual_mode_4_curve_1st_RL = modes_1st_stage_RL_masked[:, mode_4]
    residual_mode_5_curve_1st_RL = modes_1st_stage_RL_masked[:, mode_5]

    if stage_2:
        residual_mode_1_curve_2nd_RL = modes_2nd_stage_RL_masked[:, mode_1]
        residual_mode_2_curve_2nd_RL = modes_2nd_stage_RL_masked[:, mode_2]
        residual_mode_3_curve_2nd_RL = modes_2nd_stage_RL_masked[:, mode_3]
        residual_mode_4_curve_2nd_RL = modes_2nd_stage_RL_masked[:, mode_4]
        residual_mode_5_curve_2nd_RL = modes_2nd_stage_RL_masked[:, mode_5]

if integrator:
    #int 1st and 2nd stage selected mode PSD calculation
    residual_mode_1_curve_1st_int = modes_1st_stage_int_masked[:, mode_1]
    residual_mode_2_curve_1st_int = modes_1st_stage_int_masked[:, mode_2]
    residual_mode_3_curve_1st_int = modes_1st_stage_int_masked[:, mode_3]
    residual_mode_4_curve_1st_int = modes_1st_stage_int_masked[:, mode_4]
    residual_mode_5_curve_1st_int = modes_1st_stage_int_masked[:, mode_5]

    if stage_2:
        residual_mode_1_curve_2nd_int = modes_2nd_stage_int_masked[:, mode_1]
        residual_mode_2_curve_2nd_int = modes_2nd_stage_int_masked[:, mode_2]
        residual_mode_3_curve_2nd_int = modes_2nd_stage_int_masked[:, mode_3]
        residual_mode_4_curve_2nd_int = modes_2nd_stage_int_masked[:, mode_4]
        residual_mode_5_curve_2nd_int = modes_2nd_stage_int_masked[:, mode_5]


#atmosphere modes for PSD
atm_mode_1_curve  = modes_atm[:, mode_1]
atm_mode_2_curve  = modes_atm[:, mode_2]
atm_mode_3_curve  = modes_atm[:, mode_3]
atm_mode_4_curve  = modes_atm[:, mode_4]
atm_mode_5_curve  = modes_atm[:, mode_5]

if RL:
    PSD_residual_mode_1_freq_t_1st_RL, PSD_residual_mode_1_1st_RL = welch_method_scipy(residual_mode_1_curve_1st_RL, frequency_1st)
    PSD_residual_mode_2_freq_t_1st_RL, PSD_residual_mode_2_1st_RL = welch_method_scipy(residual_mode_2_curve_1st_RL, frequency_1st)
    PSD_residual_mode_3_freq_t_1st_RL, PSD_residual_mode_3_1st_RL = welch_method_scipy(residual_mode_3_curve_1st_RL, frequency_1st)
    PSD_residual_mode_4_freq_t_1st_RL, PSD_residual_mode_4_1st_RL = welch_method_scipy(residual_mode_4_curve_1st_RL, frequency_1st)
    PSD_residual_mode_5_freq_t_1st_RL, PSD_residual_mode_5_1st_RL = welch_method_scipy(residual_mode_5_curve_1st_RL, frequency_1st)


    if stage_2:
        PSD_residual_mode_1_freq_t_2nd_RL, PSD_residual_mode_1_2nd_RL = welch_method_scipy(residual_mode_1_curve_2nd_RL, f_samp_RL)
        PSD_residual_mode_2_freq_t_2nd_RL, PSD_residual_mode_2_2nd_RL = welch_method_scipy(residual_mode_2_curve_2nd_RL, f_samp_RL)
        PSD_residual_mode_3_freq_t_2nd_RL, PSD_residual_mode_3_2nd_RL = welch_method_scipy(residual_mode_3_curve_2nd_RL, f_samp_RL)
        PSD_residual_mode_4_freq_t_2nd_RL, PSD_residual_mode_4_2nd_RL = welch_method_scipy(residual_mode_4_curve_2nd_RL, f_samp_RL)
        PSD_residual_mode_5_freq_t_2nd_RL, PSD_residual_mode_5_2nd_RL = welch_method_scipy(residual_mode_5_curve_2nd_RL, f_samp_RL)


if integrator:
    PSD_residual_mode_1_freq_t_1st_int, PSD_residual_mode_1_1st_int = welch_method_scipy(residual_mode_1_curve_1st_int, frequency_1st)
    PSD_residual_mode_2_freq_t_1st_int, PSD_residual_mode_2_1st_int = welch_method_scipy(residual_mode_2_curve_1st_int, frequency_1st)
    PSD_residual_mode_3_freq_t_1st_int, PSD_residual_mode_3_1st_int = welch_method_scipy(residual_mode_3_curve_1st_int, frequency_1st)
    PSD_residual_mode_4_freq_t_1st_int, PSD_residual_mode_4_1st_int = welch_method_scipy(residual_mode_4_curve_1st_int, frequency_1st)
    PSD_residual_mode_5_freq_t_1st_int, PSD_residual_mode_5_1st_int = welch_method_scipy(residual_mode_5_curve_1st_int, frequency_1st)

    if stage_2:
        PSD_residual_mode_1_freq_t_2nd_int, PSD_residual_mode_1_2nd_int = welch_method_scipy(residual_mode_1_curve_2nd_int, f_samp_int)
        PSD_residual_mode_2_freq_t_2nd_int, PSD_residual_mode_2_2nd_int = welch_method_scipy(residual_mode_2_curve_2nd_int, f_samp_int)
        PSD_residual_mode_3_freq_t_2nd_int, PSD_residual_mode_3_2nd_int = welch_method_scipy(residual_mode_3_curve_2nd_int, f_samp_int)
        PSD_residual_mode_4_freq_t_2nd_int, PSD_residual_mode_4_2nd_int = welch_method_scipy(residual_mode_4_curve_2nd_int, f_samp_int)
        PSD_residual_mode_5_freq_t_2nd_int, PSD_residual_mode_5_2nd_int = welch_method_scipy(residual_mode_5_curve_2nd_int, f_samp_int)



PSD_atm_mode_1_freq_t, PSD_atm_mode_1 = welch_method_scipy(atm_mode_1_curve, f_samp_atm)
PSD_atm_mode_2_freq_t, PSD_atm_mode_2 = welch_method_scipy(atm_mode_2_curve, f_samp_atm)
PSD_atm_mode_3_freq_t, PSD_atm_mode_3 = welch_method_scipy(atm_mode_3_curve, f_samp_atm)
PSD_atm_mode_4_freq_t, PSD_atm_mode_4 = welch_method_scipy(atm_mode_4_curve, f_samp_atm)
PSD_atm_mode_5_freq_t, PSD_atm_mode_5 = welch_method_scipy(atm_mode_5_curve, f_samp_atm)



if plot_tPSD:
    plt.figure()
    if RL:
        if stage_2:
            #if stage_1_plus_stage_2:
                #plt.plot(PSD_residual_mode_1_freq_t_1st_RL[np.where(PSD_residual_mode_1_freq_t_1st_RL <= stage_1_freq_lim)]
                 #        , PSD_residual_mode_1_freq_t_1st_RL[np.where(PSD_residual_mode_1_freq_t_1st_RL <= stage_1_freq_lim)]**(8/3) * PSD_residual_mode_1_1st_RL[np.where(PSD_residual_mode_1_freq_t_1st_RL <= stage_1_freq_lim)], '--', color="indianred", label=f"PSD_mode_{mode_1}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_RL, PSD_residual_mode_1_freq_t_2nd_RL**(8/3) * PSD_residual_mode_1_2nd_RL, color="red", label=f"PSD_mode_{mode_1}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_RL, PSD_residual_mode_1_freq_t_1st_RL**(8/3) * PSD_residual_mode_1_1st_RL, color="indianred", label=f"PSD_mode_{mode_1}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_int[np.where(PSD_residual_mode_1_freq_t_1st_int <= stage_1_freq_lim)]
                         , PSD_residual_mode_1_freq_t_1st_int[np.where(PSD_residual_mode_1_freq_t_1st_int <= stage_1_freq_lim)]**(8/3) * PSD_residual_mode_1_1st_int[np.where(PSD_residual_mode_1_freq_t_1st_int <= stage_1_freq_lim)], '--', color="cornflowerblue", label=f"PSD_mode_{mode_1}_1st_{label_int}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_int, PSD_residual_mode_1_freq_t_2nd_int**(8/3) * PSD_residual_mode_1_2nd_int, color="blue", label=f"PSD_mode_{mode_1}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_int, PSD_residual_mode_1_freq_t_1st_int**(8/3) * PSD_residual_mode_1_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_1}_1st_{label_int}")


    plt.plot(PSD_atm_mode_1_freq_t, PSD_atm_mode_1_freq_t**(8/3) * PSD_atm_mode_1, color="black", label=f"atm_PSD_mode_{mode_1}")
    plt.title(f"Energy normalised PSD mode_{mode_1}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD")


    plt.figure()
    if RL:
        if stage_2:
            #if stage_1_plus_stage_2:
               # plt.plot(PSD_residual_mode_2_freq_t_1st_RL[np.where(PSD_residual_mode_2_freq_t_1st_RL <= stage_1_freq_lim)]
                #         , PSD_residual_mode_2_freq_t_1st_RL[np.where(PSD_residual_mode_2_freq_t_1st_RL <= stage_1_freq_lim)]**(8/3) * PSD_residual_mode_2_1st_RL[np.where(PSD_residual_mode_2_freq_t_1st_RL <= stage_1_freq_lim)], '--', color="indianred", label=f"PSD_mode_{mode_2}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_RL, PSD_residual_mode_2_freq_t_2nd_RL**(8/3) * PSD_residual_mode_2_2nd_RL, color="red", label=f"PSD_mode_{mode_2}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_RL, PSD_residual_mode_2_freq_t_1st_RL**(8/3) * PSD_residual_mode_2_1st_RL, color="indianred", label=f"PSD_mode_{mode_2}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_2_freq_t_1st_int[np.where(PSD_residual_mode_2_freq_t_1st_int <= stage_1_freq_lim)]
                         ,PSD_residual_mode_2_freq_t_1st_int[np.where(PSD_residual_mode_2_freq_t_1st_int <= stage_1_freq_lim)]**(8/3) * PSD_residual_mode_2_1st_int[np.where(PSD_residual_mode_2_freq_t_1st_int <= stage_1_freq_lim)], '--', color="cornflowerblue", label=f"PSD_mode_{mode_2}_1st_{label_int}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_int, PSD_residual_mode_2_freq_t_2nd_int**(8/3) * PSD_residual_mode_2_2nd_int, color="blue", label=f"PSD_mode_{mode_2}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_int, PSD_residual_mode_2_freq_t_1st_int**(8/3) * PSD_residual_mode_2_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_2}_1st_{label_int}")


    plt.plot(PSD_atm_mode_2_freq_t, PSD_atm_mode_2_freq_t**(8/3) * PSD_atm_mode_2, color="black", label=f"atm_PSD_mode_{mode_2}")
    plt.title(f"Energy normalised PSD mode_{mode_2}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD")


    plt.figure()
    if RL:
        if stage_2:
            #if stage_1_plus_stage_2:
            #    plt.plot(PSD_residual_mode_3_freq_t_1st_RL[np.where(PSD_residual_mode_3_freq_t_1st_RL <= stage_1_freq_lim)]
             #            , PSD_residual_mode_3_freq_t_1st_RL[np.where(PSD_residual_mode_3_freq_t_1st_RL <= stage_1_freq_lim)]**(8/3) * PSD_residual_mode_3_1st_RL[np.where(PSD_residual_mode_3_freq_t_1st_RL <= stage_1_freq_lim)], '--', color="indianred", label=f"PSD_mode_{mode_3}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_RL, PSD_residual_mode_3_freq_t_2nd_RL**(8/3) * PSD_residual_mode_3_2nd_RL, color="red", label=f"PSD_mode_{mode_3}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_RL, PSD_residual_mode_3_freq_t_1st_RL**(8/3) * PSD_residual_mode_3_1st_RL, color="indianred", label=f"PSD_mode_{mode_3}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_3_freq_t_1st_int[np.where(PSD_residual_mode_3_freq_t_1st_int <= stage_1_freq_lim)]
                         ,PSD_residual_mode_3_freq_t_1st_int[np.where(PSD_residual_mode_3_freq_t_1st_int <= stage_1_freq_lim)]**(8/3) * PSD_residual_mode_3_1st_int[np.where(PSD_residual_mode_3_freq_t_1st_int <= stage_1_freq_lim)], '--', color="cornflowerblue", label=f"PSD_mode_{mode_3}_1st_{label_int}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_int,PSD_residual_mode_3_freq_t_2nd_int**(8/3) * PSD_residual_mode_3_2nd_int, color="blue", label=f"PSD_mode_{mode_3}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_int,PSD_residual_mode_3_freq_t_1st_int**(8/3) * PSD_residual_mode_3_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_3}_1st_{label_int}")


    plt.plot(PSD_atm_mode_3_freq_t,PSD_atm_mode_3_freq_t**(8/3) * PSD_atm_mode_3, color="black", label=f"atm_PSD_mode_{mode_3}")
    plt.title(f"Energy normalised PSD mode_{mode_3}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD")


    plt.figure()
    if RL:
        if stage_2:
           # if stage_1_plus_stage_2:
            #    plt.plot(PSD_residual_mode_4_freq_t_1st_RL[np.where(PSD_residual_mode_4_freq_t_1st_RL <= stage_1_freq_lim)]
            #             ,PSD_residual_mode_4_freq_t_1st_RL[np.where(PSD_residual_mode_4_freq_t_1st_RL <= stage_1_freq_lim)]**(8/3) * PSD_residual_mode_4_1st_RL[np.where(PSD_residual_mode_4_freq_t_1st_RL <= stage_1_freq_lim)], '--', color="indianred", label=f"PSD_mode_{mode_4}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_RL,PSD_residual_mode_4_freq_t_2nd_RL**(8/3) * PSD_residual_mode_4_2nd_RL, color="red", label=f"PSD_mode_{mode_4}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_RL,PSD_residual_mode_4_freq_t_1st_RL**(8/3) * PSD_residual_mode_4_1st_RL, color="indianred", label=f"PSD_mode_{mode_4}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_4_freq_t_1st_int[np.where(PSD_residual_mode_4_freq_t_1st_int <= stage_1_freq_lim)]
                         ,PSD_residual_mode_4_freq_t_1st_int[np.where(PSD_residual_mode_4_freq_t_1st_int <= stage_1_freq_lim)]**(8/3) * PSD_residual_mode_4_1st_int[np.where(PSD_residual_mode_4_freq_t_1st_int <= stage_1_freq_lim)], '--', color="cornflowerblue", label=f"PSD_mode_{mode_4}_1st_{label_int}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_int, PSD_residual_mode_4_freq_t_2nd_int**(8/3) * PSD_residual_mode_4_2nd_int, color="blue", label=f"PSD_mode_{mode_4}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_int,PSD_residual_mode_4_freq_t_1st_int**(8/3) * PSD_residual_mode_4_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_4}_1st_{label_int}")


    plt.plot(PSD_atm_mode_4_freq_t,PSD_atm_mode_4_freq_t**(8/3) * PSD_atm_mode_4, color="black", label=f"atm_PSD_mode_{mode_4}")
    plt.title(f"rEnergy normalised PSD mode_{mode_4}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD")


    plt.figure()
    if RL:
        if stage_2:
            #if stage_1_plus_stage_2:
            #    plt.plot(PSD_residual_mode_5_freq_t_1st_RL[np.where(PSD_residual_mode_5_freq_t_1st_RL <= stage_1_freq_lim)]
            #             ,PSD_residual_mode_5_freq_t_1st_RL[np.where(PSD_residual_mode_5_freq_t_1st_RL <= stage_1_freq_lim)]**(8/3) * PSD_residual_mode_5_1st_RL[np.where(PSD_residual_mode_5_freq_t_1st_RL <= stage_1_freq_lim)], '--', color="indianred", label=f"PSD_mode_{mode_5}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_RL, PSD_residual_mode_5_freq_t_2nd_RL**(8/3) * PSD_residual_mode_5_2nd_RL, color="red", label=f"PSD_mode_{mode_5}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_RL, PSD_residual_mode_5_freq_t_1st_RL**(8/3) * PSD_residual_mode_5_1st_RL, color="indianred", label=f"PSD_mode_{mode_5}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_5_freq_t_1st_int[np.where(PSD_residual_mode_5_freq_t_1st_int <= stage_1_freq_lim)]
                         ,PSD_residual_mode_5_freq_t_1st_int[np.where(PSD_residual_mode_5_freq_t_1st_int <= stage_1_freq_lim)]**(8/3) * PSD_residual_mode_5_1st_int[np.where(PSD_residual_mode_5_freq_t_1st_int <= stage_1_freq_lim)], '--', color="cornflowerblue", label=f"PSD_mode_{mode_5}_1st_{label_int}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_int, PSD_residual_mode_5_freq_t_2nd_int**(8/3) * PSD_residual_mode_5_2nd_int, color="blue", label=f"PSD_mode_{mode_5}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_int, PSD_residual_mode_5_freq_t_1st_int**(8/3) * PSD_residual_mode_5_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_5}_1st_{label_int}")


    plt.plot(PSD_atm_mode_5_freq_t, PSD_atm_mode_5_freq_t**(8/3) * PSD_atm_mode_5, color="black", label=f"atm_PSD_mode_{mode_5}")
    plt.title(f"Energy normalised PSD mode_{mode_5}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD")


plt.show()