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
matplotlib.use('Qt5Agg')
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
folder_name = "bench_sky_04_15"
frequency_1st = 400 #should be the same as 2nd if you are using CL1OL2 with 2nd stage looking at the 1st stage residuals
frequency_2nd = 400
directory_name_RL  = 'onsky_dubhe_1st200_2nd400_v1_20260415-220508'#test_warmup50_1st250_2nd500_scaling1e2_noise_losspenalty01_reduceaction10_iters100_v1
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
mode_2 = 7
mode_3 = 15
mode_4 = 25
mode_5 = 30


RL_telemetry_2nd = np.load('bench_sky_04_15/onsky_dubhe_1st200_2nd400_v1_20260415-220508/2026-04-15T22_11_21_telemetry_2nd_data_RLiter100.npy', allow_pickle = True)
RL_telemetry_1st = np.load('bench_sky_04_15/onsky_dubhe_1st200_2nd400_v1_20260415-220508/2026-04-15T22_10_57_telemetry_data_RLiter100.npy', allow_pickle = True)

int_telemetry_2nd = np.load('bench_sky_04_15/onsky_dubhe_1st200_2nd400_v1_20260415-220508/2026-04-15T22_16_26_telemetry_2nd_data_pythint.npy', allow_pickle = True)
int_telemetry_1st = np.load('bench_sky_04_15/onsky_dubhe_1st200_2nd400_v1_20260415-220508/2026-04-15T22_15_59_telemetry_data_pythint.npy', allow_pickle = True)

CL1st_OL_2nd = np.load('bench_sky_04_15/onsky_dubhe_1st200_2nd400_v1_20260415-220508/2026-04-15T22_21_49_telemetry_2nd_data_CL1OL2.npy', allow_pickle = True)
Full_OL = np.load('bench_sky_04_15/onsky_dubhe_1st200_2nd400_v1_20260415-220508/2026-04-15T22_23_10_telemetry_data_OL.npy', allow_pickle = True)
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
    f_samp_atm = frequency_RL / 2

if atm_int:
    modes_atm = modes_atm_int
    time_plot_atm = time_plot_int
    f_samp_atm = frequency_int



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




# ---------------------------------------------------Latency---------------------------------------------------#
#TODO FOR THE LATENCY RESTRICT IT TO SOME RANGE (say -1000 to 1000) and then give it the same number of bins
#TODO also calculate the statistical things

cred_timestamp_array_RL = np.asarray(cred_timestamp_RL)
dates_np_RL = cred_timestamp_array_RL.astype('datetime64[us]')
diffs_RL = np.diff(dates_np_RL) / np.timedelta64(1, 'us') - 2500 #in microseconds
diffs_RL_median = np.median(diffs_RL)
diffs_RL_mean = np.mean(diffs_RL)
diffs_RL_std = np.std(diffs_RL)

cred_timestamp_array_int = np.asarray(cred_timestamp_int)
dates_np_int = cred_timestamp_array_int.astype('datetime64[us]')
diffs_int = np.diff(dates_np_int) / np.timedelta64(1, 'us') - 2500 #in microseconds
diffs_int_median = np.median(diffs_int)
diffs_int_mean = np.mean(diffs_int)
diffs_int_std = np.std(diffs_int)


time_diff = np.load("bench_sky_04_15/onsky_dubhe_1st200_2nd400_v1_20260415-220508/timdeff_list.npy")
time_diff = np.reshape(time_diff, time_diff.shape[0] * time_diff.shape[1])
time_diff = (time_diff - 0.0025) * 1e3


plt.figure()
plt.subplot(131)
plt.title("time difference between frames RL")
plt.xlabel('Latency in µs')
plt.hist(diffs_RL, bins=5000, color = 'black')
plt.minorticks_on()
plt.subplot(132)
plt.title("time difference between frames int")
plt.xlabel('Latency in µs')
plt.minorticks_on()
plt.hist(diffs_int, bins=5000, color = 'black')
plt.subplot(133)
plt.title("time difference between frames RL mine")
plt.xlabel('Latency in µs')
plt.minorticks_on()
plt.hist(time_diff[30000:60000], bins=5000, color = 'black')
plt.show()




# ---------------------------------------------------Strehl---------------------------------------------------#
"""if RL:
    if stage_2:
        sr_mean_RL = np.mean(strehl_array_2nd_RL)
    else:
        sr_mean_RL = np.mean(strehl_array_1st_RL)

    kernel = np.ones(30) / 30

    #pad sr
    pad_left = len(kernel) // 2
    pad_right = len(kernel) - pad_left - 1

    sr_padded_1st_RL = np.pad(strehl_array_1st_RL, (pad_left, pad_right), mode='constant', constant_values=sr_mean_RL)
    sr_running_1st_RL = np.convolve(sr_padded_1st_RL, kernel, mode='valid')

    if stage_2:
        sr_padded_2nd_RL = np.pad(strehl_array_2nd_RL, (pad_left, pad_right), mode='constant', constant_values=sr_mean_RL)
        sr_running_2nd_RL = np.convolve(sr_padded_2nd_RL, kernel, mode='valid')

if integrator:
    if stage_2:
        sr_mean_int = np.mean(strehl_array_2nd_int)
    else:
        sr_mean_int = np.mean(strehl_array_1st_int)

    kernel = np.ones(30) / 30

    #pad sr
    pad_left = len(kernel) // 2
    pad_right = len(kernel) - pad_left - 1

    sr_padded_1st_int = np.pad(strehl_array_1st_int, (pad_left, pad_right), mode='constant', constant_values=sr_mean_int)
    sr_running_1st_int = np.convolve(sr_padded_1st_int, kernel, mode='valid')

    if stage_2:
        sr_padded_2nd_int = np.pad(strehl_array_2nd_int, (pad_left, pad_right), mode='constant', constant_values=sr_mean_int)
        sr_running_2nd_int = np.convolve(sr_padded_2nd_int, kernel, mode='valid')

if ideal:
    if stage_2:
        sr_mean_ideal = np.mean(strehl_array_2nd_ideal)
    else:
        sr_mean_ideal = np.mean(strehl_array_1st_ideal)

    kernel = np.ones(30) / 30

    #pad sr
    pad_left = len(kernel) // 2
    pad_right = len(kernel) - pad_left - 1

    sr_padded_1st_ideal = np.pad(strehl_array_1st_ideal, (pad_left, pad_right), mode='constant', constant_values=sr_mean_ideal)
    sr_running_1st_ideal = np.convolve(sr_padded_1st_ideal, kernel, mode='valid')

    if stage_2:
        sr_padded_2nd_ideal = np.pad(strehl_array_2nd_ideal, (pad_left, pad_right), mode='constant', constant_values=sr_mean_ideal)
        sr_running_2nd_ideal = np.convolve(sr_padded_2nd_ideal, kernel, mode='valid')

plt.figure()
if integrator:  plt.plot(time_plot_int, strehl_array_1st_int[:len(time_plot_int)], color = '#ff7f0e', label=f"1st_{label_int}")
if ideal:       plt.plot(time_plot_ideal, strehl_array_1st_ideal[:len(time_plot_ideal)], color = 'black', label=f"1st_{label_ideal}")
if RL:          plt.plot(time_plot_RL, strehl_array_1st_RL, color = '#1f77b4', label=f"1st_{label_RL}")

if not stage_2:
    if RL: plt.plot(time_plot_RL, sr_running_1st_RL, label=f"running_avg_1st_{label_RL}")
if stage_2:
    #if RL: plt.plot(time_plot_RL, strehl_array_2nd_RL, color = '#003f5c', label=f"2nd_{label_RL}")
    if RL: plt.plot(time_plot_RL, sr_running_2nd_RL, color = 'red', label=f"running_avg_2nd_{label_RL}")

    #if integrator: plt.plot(time_plot_int, strehl_array_2nd_int, color = '#ffa600', label=f"2nd_{label_int}")
    if integrator: plt.plot(time_plot_int, sr_running_2nd_int, color='darkturquoise', label=f"running_avg_2nd_{label_int}")

    #if ideal: plt.plot(time_plot_ideal, strehl_array_2nd_ideal, color = 'black', label=f"2nd_{label_ideal}")
    if ideal: plt.plot(time_plot_ideal, sr_running_2nd_ideal, color='black',label=f"running_avg_2nd_{label_ideal}")


plt.title("Strehl ratio")
plt.xlabel("time s")
#plt.ylim(bottom=(sr_mean_RL - 0.5))
plt.grid(True, alpha=0.5)
plt.minorticks_on()
plt.legend()

"""
# ---------------------------------------------------Zernike/KL decomposition---------------------------------------------------#
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


#ideal
if ideal:
    modes_1st_stage_ideal_masked = modes_1st_stage_ideal[KL_frame_thresh_ideal:, :]
    coefs_var_1st_stage_masked_ideal = np.var(np.asarray(modes_1st_stage_ideal_masked), axis = 0)
    coefs_var_1st_stage_ideal = np.var(np.asarray(modes_1st_stage_ideal), axis=0)

    if stage_2:
        modes_2nd_stage_ideal_masked = modes_2nd_stage_ideal[KL_frame_thresh_ideal:, :]
        coefs_var_2nd_stage_masked_ideal = np.var(np.asarray(modes_2nd_stage_ideal_masked), axis = 0)
        coefs_var_2nd_stage_ideal = np.var(np.asarray(modes_2nd_stage_ideal), axis = 0)


#atmosphere
coefs_var_atm = np.var(np.asarray(modes_atm[KL_frame_thresh_int:, :]), axis = 0)


plt.figure()
#plt.plot(coefs_var_atm, color="black",label=f"KL coeffs for atmospheric phase")
if RL:
    if stage_2:
        if stage_1_plus_stage_2:
            plt.plot(coefs_var_1st_stage_masked_RL, '--', color="indianred", label=f"KL coeffs 1st stage {label_RL} ")
        plt.plot(coefs_var_2nd_stage_masked_RL, color="red", label=f"KL coeffs 2nd stage {label_RL}")
    else:
        plt.plot(coefs_var_1st_stage_masked_RL, color="indianred", label=f"KL coeffs {label_RL}")

if integrator:
    if stage_2:
        if stage_1_plus_stage_2:
            plt.plot(coefs_var_1st_stage_masked_int, '--', color="cornflowerblue", label=f"KL coeffs 1st stage {label_int}")
        plt.plot(coefs_var_2nd_stage_masked_int, color="blue", label=f"KL coeffs 2nd stage {label_int}")
    else:
        plt.plot(coefs_var_1st_stage_masked_int, color="cornflowerblue", label=f"KL coeffs {label_int}")

if ideal:
    if stage_2:
        if stage_1_plus_stage_2:
            plt.plot(coefs_var_1st_stage_masked_ideal, '--', color="seagreen", label=f"KL coeffs 1st stage {label_ideal}")
        plt.plot(coefs_var_2nd_stage_masked_ideal, color="green", label=f"KL coeffs 2nd stage {label_ideal}")
    else:
        plt.plot(coefs_var_1st_stage_masked_ideal, color="seagreen", label=f"KL coeffs {label_ideal}")

plt.title(f"KL coefficients RL vs integrator", fontsize = 18)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("KL coefficients", fontsize = 14)
plt.ylabel("KL coefficient variance", fontsize = 14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.ylim(bottom = 5e-6)
#plt.tight_layout()
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
#plt.savefig("KL_decomp_on_sky_04_02.png", dpi=300)



# ---------------------------------------------------Temporal PSD---------------------------------------------------#
# temporal PSD calculation from the std
def welch_method_scipy(data, fs, nperseg=256):
    frequencies, psd = signal.welch(
        data,
        fs=fs,
        window='hann',  #windowing
        nperseg=nperseg,
        scaling='density'
    )
    return frequencies, psd

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


if ideal:
    f_samp_ideal = frequency_ideal
    # tip timeseries
    residual_mode_1_curve_1st_ideal_full = modes_1st_stage_ideal[:, mode_1]
    residual_mode_2_curve_1st_ideal_full = modes_1st_stage_ideal[:, mode_2]
    residual_mode_3_curve_1st_ideal_full = modes_1st_stage_ideal[:, mode_3]
    residual_mode_4_curve_1st_ideal_full = modes_1st_stage_ideal[:, mode_4]
    residual_mode_5_curve_1st_ideal_full = modes_1st_stage_ideal[:, mode_5]

    if stage_2:
        residual_mode_1_curve_2nd_ideal_full = modes_2nd_stage_ideal[:, mode_1]
        residual_mode_2_curve_2nd_ideal_full = modes_2nd_stage_ideal[:, mode_2]
        residual_mode_3_curve_2nd_ideal_full = modes_2nd_stage_ideal[:, mode_3]
        residual_mode_4_curve_2nd_ideal_full = modes_2nd_stage_ideal[:, mode_4]
        residual_mode_5_curve_2nd_ideal_full = modes_2nd_stage_ideal[:, mode_5]



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

if ideal:
    #ideal 1st and 2nd stage selected mode PSD calculation
    residual_mode_1_curve_1st_ideal = modes_1st_stage_ideal_masked[:, mode_1]
    residual_mode_2_curve_1st_ideal = modes_1st_stage_ideal_masked[:, mode_2]
    residual_mode_3_curve_1st_ideal = modes_1st_stage_ideal_masked[:, mode_3]
    residual_mode_4_curve_1st_ideal = modes_1st_stage_ideal_masked[:, mode_4]
    residual_mode_5_curve_1st_ideal = modes_1st_stage_ideal_masked[:, mode_5]

    if stage_2:
        residual_mode_1_curve_2nd_ideal = modes_2nd_stage_ideal_masked[:, mode_1]
        residual_mode_2_curve_2nd_ideal = modes_2nd_stage_ideal_masked[:, mode_2]
        residual_mode_3_curve_2nd_ideal = modes_2nd_stage_ideal_masked[:, mode_3]
        residual_mode_4_curve_2nd_ideal = modes_2nd_stage_ideal_masked[:, mode_4]
        residual_mode_5_curve_2nd_ideal = modes_2nd_stage_ideal_masked[:, mode_5]


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


if ideal:
    #tip
    PSD_residual_mode_1_freq_t_1st_ideal, PSD_residual_mode_1_1st_ideal = welch_method_scipy(residual_mode_1_curve_1st_ideal, frequency_1st)
    PSD_residual_mode_2_freq_t_1st_ideal, PSD_residual_mode_2_1st_ideal = welch_method_scipy(residual_mode_2_curve_1st_ideal, frequency_1st)
    PSD_residual_mode_3_freq_t_1st_ideal, PSD_residual_mode_3_1st_ideal = welch_method_scipy(residual_mode_3_curve_1st_ideal, frequency_1st)
    PSD_residual_mode_4_freq_t_1st_ideal, PSD_residual_mode_4_1st_ideal = welch_method_scipy(residual_mode_4_curve_1st_ideal, frequency_1st)
    PSD_residual_mode_5_freq_t_1st_ideal, PSD_residual_mode_5_1st_ideal = welch_method_scipy(residual_mode_5_curve_1st_ideal, frequency_1st)

    if stage_2:
        PSD_residual_mode_1_freq_t_2nd_ideal, PSD_residual_mode_1_2nd_ideal = welch_method_scipy(residual_mode_1_curve_2nd_ideal, f_samp_ideal)
        PSD_residual_mode_2_freq_t_2nd_ideal, PSD_residual_mode_2_2nd_ideal = welch_method_scipy(residual_mode_2_curve_2nd_ideal, f_samp_ideal)
        PSD_residual_mode_3_freq_t_2nd_ideal, PSD_residual_mode_3_2nd_ideal = welch_method_scipy(residual_mode_3_curve_2nd_ideal, f_samp_ideal)
        PSD_residual_mode_4_freq_t_2nd_ideal, PSD_residual_mode_4_2nd_ideal = welch_method_scipy(residual_mode_4_curve_2nd_ideal, f_samp_ideal)
        PSD_residual_mode_5_freq_t_2nd_ideal, PSD_residual_mode_5_2nd_ideal = welch_method_scipy(residual_mode_5_curve_2nd_ideal, f_samp_ideal)


PSD_atm_mode_1_freq_t, PSD_atm_mode_1 = welch_method_scipy(atm_mode_1_curve, f_samp_atm)
PSD_atm_mode_2_freq_t, PSD_atm_mode_2 = welch_method_scipy(atm_mode_2_curve, f_samp_atm)
PSD_atm_mode_3_freq_t, PSD_atm_mode_3 = welch_method_scipy(atm_mode_3_curve, f_samp_atm)
PSD_atm_mode_4_freq_t, PSD_atm_mode_4 = welch_method_scipy(atm_mode_4_curve, f_samp_atm)
PSD_atm_mode_5_freq_t, PSD_atm_mode_5 = welch_method_scipy(atm_mode_5_curve, f_samp_atm)


if plot_timeseries:
    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_RL, residual_mode_1_curve_1st_RL_full, color="indianred", label=f"mode_{mode_1}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_1_curve_2nd_RL_full, color="red", label=f"mode_{mode_1}_2nd_stage_{label_RL}")
        else: plt.plot(time_plot_RL, residual_mode_1_curve_1st_RL_full, color="indianred", label=f"mode_{mode_1}_1st_stage_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_int, residual_mode_1_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_1}_1st_stage_{label_int}")
            plt.plot(time_plot_int, residual_mode_1_curve_2nd_int_full, color="blue", label=f"mode_{mode_1}_2nd_stage_{label_int}")
        else: plt.plot(time_plot_int, residual_mode_1_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_1}_1st_stage_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_ideal, residual_mode_1_curve_1st_ideal_full, label=f"mode_{mode_1}_1st_stage_{label_ideal}")
            plt.plot(time_plot_ideal, residual_mode_1_curve_2nd_ideal_full, label=f"mode_{mode_1}_2nd_stage_{label_ideal}")
        else: plt.plot(time_plot_ideal, residual_mode_1_curve_1st_ideal_full, label=f"mode_{mode_1}_1st_stage_{label_ideal}")
    #plt.plot(time_plot_atm, atm_mode_1_curve, color="black", label=f"atm_mode_{mode_1}_curve")
    plt.title(f"residual/atm timeseries mode_{mode_1}, gain {CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_1}")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_RL, residual_mode_2_curve_1st_RL_full, color="indianred", label=f"mode_{mode_2}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_2_curve_2nd_RL_full, color="red", label=f"mode_{mode_2}_2nd_stage_{label_RL}")
        else: plt.plot(time_plot_RL, residual_mode_2_curve_1st_RL_full, color="indianred", label=f"mode_{mode_2}_1st_stage_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_int, residual_mode_2_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_2}_1st_stage_{label_int}")
            plt.plot(time_plot_int, residual_mode_2_curve_2nd_int_full, color="blue", label=f"mode_{mode_2}_2nd_stage_{label_int}")
        else: plt.plot(time_plot_int, residual_mode_2_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_2}_1st_stage_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_ideal, residual_mode_2_curve_1st_ideal_full, label=f"mode_{mode_2}_1st_stage_{label_ideal}")
            plt.plot(time_plot_ideal, residual_mode_2_curve_2nd_ideal_full, label=f"mode_{mode_2}_2nd_stage_{label_ideal}")
        else: plt.plot(time_plot_ideal, residual_mode_2_curve_1st_ideal_full, label=f"mode_{mode_2}_1st_stage_{label_ideal}")
    #plt.plot(time_plot_atm, atm_mode_2_curve, color="black", label=f"atm_mode_{mode_2}_curve")
    plt.title(f"residual/atm timeseries mode_{mode_2}, gain {CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_2}")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_RL, residual_mode_3_curve_1st_RL_full, color="indianred", label=f"mode_{mode_3}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_3_curve_2nd_RL_full, color="red", label=f"mode_{mode_3}_2nd_stage_{label_RL}")
        else: plt.plot(time_plot_RL, residual_mode_3_curve_1st_RL_full, color="indianred", label=f"mode_{mode_3}_1st_stage_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_int, residual_mode_3_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_3}_1st_stage_{label_int}")
            plt.plot(time_plot_int, residual_mode_3_curve_2nd_int_full, color="blue", label=f"mode_{mode_3}_2nd_stage_{label_int}")
        else: plt.plot(time_plot_int, residual_mode_3_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_3}_1st_stage_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_ideal, residual_mode_3_curve_1st_ideal_full, label=f"mode_{mode_3}_1st_stage__{label_ideal}")
            plt.plot(time_plot_ideal, residual_mode_3_curve_2nd_ideal_full, label=f"mode_{mode_3}_2nd_stage_{label_ideal}")
        else: plt.plot(time_plot_ideal, residual_mode_3_curve_1st_ideal_full, label=f"mode_{mode_3}_1st_stage_{label_ideal}")
    #plt.plot(time_plot_atm, atm_mode_3_curve, color="black", label=f"atm_mode_{mode_3}_curve")
    plt.title(f"residual/atm timeseries mode_{mode_3}, gain {CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_3}")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_RL, residual_mode_4_curve_1st_RL_full, color="indianred", label=f"mode_{mode_4}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_4_curve_2nd_RL_full, color="red", label=f"mode__{mode_4}_2nd_stage_{label_RL}")
        else: plt.plot(time_plot_RL, residual_mode_4_curve_1st_RL_full, color="indianred", label=f"mode_{mode_4}_1st_stage_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_int, residual_mode_4_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_4}_1st_stage_{label_int}")
            plt.plot(time_plot_int, residual_mode_4_curve_2nd_int_full, color="blue", label=f"mode_{mode_4}_2nd_stage_{label_int}")
        else: plt.plot(time_plot_int, residual_mode_4_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_4}_1st_stage_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_ideal, residual_mode_4_curve_1st_ideal_full, label=f"mode_{mode_4}_1st_stage_{label_ideal}")
            plt.plot(time_plot_ideal, residual_mode_4_curve_2nd_ideal_full, label=f"mode_{mode_4}_2nd_stage_{label_ideal}")
        else: plt.plot(time_plot_ideal, residual_mode_4_curve_1st_ideal_full, label=f"mode_{mode_4}_1st_stage_{label_ideal}")
    #plt.plot(time_plot_atm, atm_mode_4_curve, color="black", label=f"atm_mode_{mode_4}_curve")
    plt.title(f"residual/atm timeseries mode_{mode_4}, gain {CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_4}")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_RL, residual_mode_5_curve_1st_RL_full, color="indianred", label=f"mode_{mode_5}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_5_curve_2nd_RL_full, color="red", label=f"mode_{mode_5}_2nd_stage_{label_RL}")
        else: plt.plot(time_plot_RL, residual_mode_5_curve_1st_RL_full, color="indianred", label=f"mode_{mode_5}_1st_stage_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_int, residual_mode_5_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_5}_1st_stage_{label_int}")
            plt.plot(time_plot_int, residual_mode_5_curve_2nd_int_full, color="blue", label=f"mode_{mode_5}_2nd_stage_{label_int}")
        else: plt.plot(time_plot_int, residual_mode_5_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_5}_1st_stage_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_ideal, residual_mode_5_curve_1st_ideal_full, label=f"mode_{mode_5}_1st_stage_{label_ideal}")
            plt.plot(time_plot_ideal, residual_mode_5_curve_2nd_ideal_full, label=f"mode_{mode_5}_2nd_stage_{label_ideal}")
        else: plt.plot(time_plot_ideal, residual_mode_5_curve_1st_ideal_full, label=f"mode_{mode_5}_1st_stage_{label_ideal}")
    #plt.plot(time_plot_atm, atm_mode_5_curve, color="black", label=f"atm_mode_{mode_5}_curve")
    plt.title(f"residual/atm timeseries mode_{mode_5}, gain {CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_5}")


if plot_tPSD:
    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_RL[np.where(PSD_residual_mode_1_freq_t_1st_RL <= stage_1_freq_lim)]
                         , PSD_residual_mode_1_1st_RL[np.where(PSD_residual_mode_1_freq_t_1st_RL <= stage_1_freq_lim)], '--', color="indianred", label=f"PSD_mode_{mode_1}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_RL, PSD_residual_mode_1_2nd_RL, color="red", label=f"PSD_mode_{mode_1}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_RL, PSD_residual_mode_1_1st_RL, color="indianred", label=f"PSD_mode_{mode_1}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_int[np.where(PSD_residual_mode_1_freq_t_1st_int <= stage_1_freq_lim)]
                         , PSD_residual_mode_1_1st_int[np.where(PSD_residual_mode_1_freq_t_1st_int <= stage_1_freq_lim)], '--', color="cornflowerblue", label=f"PSD_mode_{mode_1}_1st_{label_int}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_int, PSD_residual_mode_1_2nd_int, color="blue", label=f"PSD_mode_{mode_1}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_int, PSD_residual_mode_1_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_1}_1st_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_ideal[np.where(PSD_residual_mode_1_freq_t_1st_ideal <= stage_1_freq_lim)]
                         , PSD_residual_mode_1_1st_ideal[np.where(PSD_residual_mode_1_freq_t_1st_ideal <= stage_1_freq_lim)], '--', label=f"PSD_mode_{mode_1}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_ideal, PSD_residual_mode_1_2nd_ideal, label=f"PSD_mode_{mode_1}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_ideal, PSD_residual_mode_1_1st_ideal, label=f"PSD_mode_{mode_1}_1st_{label_ideal}")


    plt.plot(PSD_atm_mode_1_freq_t, PSD_atm_mode_1, color="black", label=f"atm_PSD_mode_{mode_1}")
    plt.title(f"residual PSD mode_{mode_1}, gain {CL_gain_pyr}")
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
                plt.plot(PSD_residual_mode_2_freq_t_1st_RL[np.where(PSD_residual_mode_2_freq_t_1st_RL <= stage_1_freq_lim)]
                         , PSD_residual_mode_2_1st_RL[np.where(PSD_residual_mode_2_freq_t_1st_RL <= stage_1_freq_lim)], '--', color="indianred", label=f"PSD_mode_{mode_2}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_RL, PSD_residual_mode_2_2nd_RL, color="red", label=f"PSD_mode_{mode_2}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_RL, PSD_residual_mode_2_1st_RL, color="indianred", label=f"PSD_mode_{mode_2}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_2_freq_t_1st_int[np.where(PSD_residual_mode_2_freq_t_1st_int <= stage_1_freq_lim)]
                         , PSD_residual_mode_2_1st_int[np.where(PSD_residual_mode_2_freq_t_1st_int <= stage_1_freq_lim)], '--', color="cornflowerblue", label=f"PSD_mode_{mode_2}_1st_{label_int}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_int, PSD_residual_mode_2_2nd_int, color="blue", label=f"PSD_mode_{mode_2}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_int, PSD_residual_mode_2_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_2}_1st_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_2_freq_t_1st_ideal[np.where(PSD_residual_mode_2_freq_t_1st_ideal <= stage_1_freq_lim)]
                         , PSD_residual_mode_2_1st_ideal[np.where(PSD_residual_mode_2_freq_t_1st_ideal <= stage_1_freq_lim)], '--', label=f"PSD_mode_{mode_2}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_ideal, PSD_residual_mode_2_2nd_ideal, label=f"PSD_mode_{mode_2}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_ideal, PSD_residual_mode_2_1st_ideal, label=f"PSD_mode_{mode_2}_1st_{label_ideal}")


    plt.plot(PSD_atm_mode_2_freq_t, PSD_atm_mode_2, color="black", label=f"atm_PSD_mode_{mode_2}")
    plt.title(f"residual PSD mode_{mode_2}, gain {CL_gain_pyr}")
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
                plt.plot(PSD_residual_mode_3_freq_t_1st_RL[np.where(PSD_residual_mode_3_freq_t_1st_RL <= stage_1_freq_lim)]
                         , PSD_residual_mode_3_1st_RL[np.where(PSD_residual_mode_3_freq_t_1st_RL <= stage_1_freq_lim)], '--', color="indianred", label=f"PSD_mode_{mode_3}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_RL, PSD_residual_mode_3_2nd_RL, color="red", label=f"PSD_mode_{mode_3}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_RL, PSD_residual_mode_3_1st_RL, color="indianred", label=f"PSD_mode_{mode_3}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_3_freq_t_1st_int[np.where(PSD_residual_mode_3_freq_t_1st_int <= stage_1_freq_lim)]
                         , PSD_residual_mode_3_1st_int[np.where(PSD_residual_mode_3_freq_t_1st_int <= stage_1_freq_lim)], '--', color="cornflowerblue", label=f"PSD_mode_{mode_3}_1st_{label_int}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_int, PSD_residual_mode_3_2nd_int, color="blue", label=f"PSD_mode_{mode_3}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_int, PSD_residual_mode_3_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_3}_1st_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_3_freq_t_1st_ideal[np.where(PSD_residual_mode_3_freq_t_1st_ideal <= stage_1_freq_lim)]
                         , PSD_residual_mode_3_1st_ideal[np.where(PSD_residual_mode_3_freq_t_1st_ideal <= stage_1_freq_lim)], '--', label=f"PSD_mode_{mode_3}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_ideal, PSD_residual_mode_3_2nd_ideal, label=f"PSD_mode_{mode_3}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_ideal, PSD_residual_mode_3_1st_ideal, label=f"PSD_mode_{mode_3}_1st_{label_ideal}")


    plt.plot(PSD_atm_mode_3_freq_t, PSD_atm_mode_3, color="black", label=f"atm_PSD_mode_{mode_3}")
    plt.title(f"residual PSD mode_{mode_3}, gain {CL_gain_pyr}")
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
                plt.plot(PSD_residual_mode_4_freq_t_1st_RL[np.where(PSD_residual_mode_4_freq_t_1st_RL <= stage_1_freq_lim)]
                         , PSD_residual_mode_4_1st_RL[np.where(PSD_residual_mode_4_freq_t_1st_RL <= stage_1_freq_lim)], '--', color="indianred", label=f"PSD_mode_{mode_4}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_RL, PSD_residual_mode_4_2nd_RL, color="red", label=f"PSD_mode_{mode_4}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_RL, PSD_residual_mode_4_1st_RL, color="indianred", label=f"PSD_mode_{mode_4}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_4_freq_t_1st_int[np.where(PSD_residual_mode_4_freq_t_1st_int <= stage_1_freq_lim)]
                         , PSD_residual_mode_4_1st_int[np.where(PSD_residual_mode_4_freq_t_1st_int <= stage_1_freq_lim)], '--', color="cornflowerblue", label=f"PSD_mode_{mode_4}_1st_{label_int}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_int, PSD_residual_mode_4_2nd_int, color="blue", label=f"PSD_mode_{mode_4}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_int, PSD_residual_mode_4_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_4}_1st_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_4_freq_t_1st_ideal[np.where(PSD_residual_mode_4_freq_t_1st_ideal <= stage_1_freq_lim)]
                         , PSD_residual_mode_4_1st_ideal[np.where(PSD_residual_mode_4_freq_t_1st_ideal <= stage_1_freq_lim)], '--', color="indianred", label=f"PSD_mode_{mode_4}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_ideal, PSD_residual_mode_4_2nd_ideal, label=f"PSD_mode_{mode_4}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_ideal, PSD_residual_mode_4_1st_ideal, color="indianred", label=f"PSD_mode_{mode_4}_1st_{label_ideal}")

    plt.plot(PSD_atm_mode_4_freq_t, PSD_atm_mode_4, color="black", label=f"atm_PSD_mode_{mode_4}")
    plt.title(f"residual PSD mode_{mode_4}, gain {CL_gain_pyr}")
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
                plt.plot(PSD_residual_mode_5_freq_t_1st_RL[np.where(PSD_residual_mode_5_freq_t_1st_RL <= stage_1_freq_lim)]
                         , PSD_residual_mode_5_1st_RL[np.where(PSD_residual_mode_5_freq_t_1st_RL <= stage_1_freq_lim)], '--', color="indianred", label=f"PSD_mode_{mode_5}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_RL, PSD_residual_mode_5_2nd_RL, color="red", label=f"PSD_mode_{mode_5}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_RL, PSD_residual_mode_5_1st_RL, color="indianred", label=f"PSD_mode_{mode_5}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_5_freq_t_1st_int[np.where(PSD_residual_mode_5_freq_t_1st_int <= stage_1_freq_lim)]
                         , PSD_residual_mode_5_1st_int[np.where(PSD_residual_mode_5_freq_t_1st_int <= stage_1_freq_lim)], '--', color="cornflowerblue", label=f"PSD_mode_{mode_5}_1st_{label_int}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_int, PSD_residual_mode_5_2nd_int, color="blue", label=f"PSD_mode_{mode_5}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_int, PSD_residual_mode_5_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_5}_1st_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_5_freq_t_1st_ideal[np.where(PSD_residual_mode_5_freq_t_1st_ideal <= stage_1_freq_lim)]
                         , PSD_residual_mode_5_1st_ideal[np.where(PSD_residual_mode_5_freq_t_1st_ideal <= stage_1_freq_lim)], '--', label=f"PSD_mode_{mode_5}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_ideal, PSD_residual_mode_5_2nd_ideal, label=f"PSD_mode_{mode_5}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_ideal, PSD_residual_mode_5_1st_ideal, label=f"PSD_mode_{mode_5}_1st_{label_ideal}")

    plt.plot(PSD_atm_mode_5_freq_t, PSD_atm_mode_5, color="black", label=f"atm_PSD_mode_{mode_5}")
    plt.title(f"residual PSD mode_{mode_5}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD")



#---------------------------------------------------Cumulative PSD---------------------------------------------------#



if plot_tPSD:
    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_RL[np.where(PSD_residual_mode_1_freq_t_1st_RL <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_1_1st_RL[np.where(PSD_residual_mode_1_freq_t_1st_RL <= stage_1_freq_lim)]), '--', color="indianred", label=f"PSD_mode_{mode_1}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_RL, np.cumsum(PSD_residual_mode_1_2nd_RL), color="red", label=f"PSD_mode_{mode_1}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_RL, np.cumsum(PSD_residual_mode_1_1st_RL), color="indianred", label=f"PSD_mode_{mode_1}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_int[np.where(PSD_residual_mode_1_freq_t_1st_int <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_1_1st_int[np.where(PSD_residual_mode_1_freq_t_1st_int <= stage_1_freq_lim)]), '--', color="cornflowerblue", label=f"PSD_mode_{mode_1}_1st_{label_int}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_int, np.cumsum(PSD_residual_mode_1_2nd_int), color="blue", label=f"PSD_mode_{mode_1}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_int, np.cumsum(PSD_residual_mode_1_1st_int), color="cornflowerblue", label=f"PSD_mode_{mode_1}_1st_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_ideal[np.where(PSD_residual_mode_1_freq_t_1st_ideal <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_1_1st_ideal[np.where(PSD_residual_mode_1_freq_t_1st_ideal <= stage_1_freq_lim)]), '--', label=f"PSD_mode_{mode_1}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_ideal, np.cumsum(PSD_residual_mode_1_2nd_ideal), label=f"PSD_mode_{mode_1}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_ideal, np.cumsum(PSD_residual_mode_1_1st_ideal), label=f"PSD_mode_{mode_1}_1st_{label_ideal}")



    plt.title(f"Cumulative PSD {mode_1}, gain {CL_gain_pyr}")
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
                plt.plot(PSD_residual_mode_2_freq_t_1st_RL[np.where(PSD_residual_mode_2_freq_t_1st_RL <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_2_1st_RL[np.where(PSD_residual_mode_2_freq_t_1st_RL <= stage_1_freq_lim)]), '--', color="indianred", label=f"PSD_mode_{mode_2}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_RL, np.cumsum(PSD_residual_mode_2_2nd_RL), color="red", label=f"PSD_mode_{mode_2}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_RL, np.cumsum(PSD_residual_mode_2_1st_RL), color="indianred", label=f"PSD_mode_{mode_2}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_2_freq_t_1st_int[np.where(PSD_residual_mode_2_freq_t_1st_int <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_2_1st_int[np.where(PSD_residual_mode_2_freq_t_1st_int <= stage_1_freq_lim)]), '--', color="cornflowerblue", label=f"PSD_mode_{mode_2}_1st_{label_int}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_int, np.cumsum(PSD_residual_mode_2_2nd_int), color="blue", label=f"PSD_mode_{mode_2}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_int, np.cumsum(PSD_residual_mode_2_1st_int), color="cornflowerblue", label=f"PSD_mode_{mode_2}_1st_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_2_freq_t_1st_ideal[np.where(PSD_residual_mode_2_freq_t_1st_ideal <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_2_1st_ideal[np.where(PSD_residual_mode_2_freq_t_1st_ideal <= stage_1_freq_lim)]), '--', label=f"PSD_mode_{mode_2}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_ideal, np.cumsum(PSD_residual_mode_2_2nd_ideal), label=f"PSD_mode_{mode_2}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_ideal, np.cumsum(PSD_residual_mode_2_1st_ideal), label=f"PSD_mode_{mode_2}_1st_{label_ideal}")



    plt.title(f"Cumulative PSD {mode_2}, gain {CL_gain_pyr}")
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
                plt.plot(PSD_residual_mode_3_freq_t_1st_RL[np.where(PSD_residual_mode_3_freq_t_1st_RL <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_3_1st_RL[np.where(PSD_residual_mode_3_freq_t_1st_RL <= stage_1_freq_lim)]), '--', color="indianred", label=f"PSD_mode_{mode_3}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_RL, np.cumsum(PSD_residual_mode_3_2nd_RL), color="red", label=f"PSD_mode_{mode_3}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_RL, np.cumsum(PSD_residual_mode_3_1st_RL), color="indianred", label=f"PSD_mode_{mode_3}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_3_freq_t_1st_int[np.where(PSD_residual_mode_3_freq_t_1st_int <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_3_1st_int[np.where(PSD_residual_mode_3_freq_t_1st_int <= stage_1_freq_lim)]), '--', color="cornflowerblue", label=f"PSD_mode_{mode_3}_1st_{label_int}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_int, np.cumsum(PSD_residual_mode_3_2nd_int), color="blue", label=f"PSD_mode_{mode_3}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_int, np.cumsum(PSD_residual_mode_3_1st_int), color="cornflowerblue", label=f"PSD_mode_{mode_3}_1st_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_3_freq_t_1st_ideal[np.where(PSD_residual_mode_3_freq_t_1st_ideal <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_3_1st_ideal[np.where(PSD_residual_mode_3_freq_t_1st_ideal <= stage_1_freq_lim)]), '--', label=f"PSD_mode_{mode_3}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_ideal, np.cumsum(PSD_residual_mode_3_2nd_ideal), label=f"PSD_mode_{mode_3}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_ideal, np.cumsum(PSD_residual_mode_3_1st_ideal), label=f"PSD_mode_{mode_3}_1st_{label_ideal}")



    plt.title(f"Cumulative PSD {mode_3}, gain {CL_gain_pyr}")
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
                plt.plot(PSD_residual_mode_4_freq_t_1st_RL[np.where(PSD_residual_mode_4_freq_t_1st_RL <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_4_1st_RL[np.where(PSD_residual_mode_4_freq_t_1st_RL <= stage_1_freq_lim)]), '--', color="indianred", label=f"PSD_mode_{mode_4}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_RL, np.cumsum(PSD_residual_mode_4_2nd_RL), color="red", label=f"PSD_mode_{mode_4}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_RL, np.cumsum(PSD_residual_mode_4_1st_RL), color="indianred", label=f"PSD_mode_{mode_4}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_4_freq_t_1st_int[np.where(PSD_residual_mode_4_freq_t_1st_int <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_4_1st_int[np.where(PSD_residual_mode_4_freq_t_1st_int <= stage_1_freq_lim)]), '--', color="cornflowerblue", label=f"PSD_mode_{mode_4}_1st_{label_int}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_int, np.cumsum(PSD_residual_mode_4_2nd_int), color="blue", label=f"PSD_mode_{mode_4}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_int, np.cumsum(PSD_residual_mode_4_1st_int), color="cornflowerblue", label=f"PSD_mode_{mode_4}_1st_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_4_freq_t_1st_ideal[np.where(PSD_residual_mode_4_freq_t_1st_ideal <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_4_1st_ideal[np.where(PSD_residual_mode_4_freq_t_1st_ideal <= stage_1_freq_lim)]), '--', color="indianred", label=f"PSD_mode_{mode_4}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_ideal, np.cumsum(PSD_residual_mode_4_2nd_ideal), label=f"PSD_mode_{mode_4}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_ideal, np.cumsum(PSD_residual_mode_4_1st_ideal), color="indianred", label=f"PSD_mode_{mode_4}_1st_{label_ideal}")


    plt.title(f"Cumulative PSD {mode_4}, gain {CL_gain_pyr}")
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
                plt.plot(PSD_residual_mode_5_freq_t_1st_RL[np.where(PSD_residual_mode_5_freq_t_1st_RL <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_5_1st_RL[np.where(PSD_residual_mode_5_freq_t_1st_RL <= stage_1_freq_lim)]), '--', color="indianred", label=f"PSD_mode_{mode_5}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_RL, np.cumsum(PSD_residual_mode_5_2nd_RL), color="red", label=f"PSD_mode_{mode_5}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_RL, np.cumsum(PSD_residual_mode_5_1st_RL), color="indianred", label=f"PSD_mode_{mode_5}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_5_freq_t_1st_int[np.where(PSD_residual_mode_5_freq_t_1st_int <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_5_1st_int[np.where(PSD_residual_mode_5_freq_t_1st_int <= stage_1_freq_lim)]), '--', color="cornflowerblue", label=f"PSD_mode_{mode_5}_1st_{label_int}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_int, np.cumsum(PSD_residual_mode_5_2nd_int), color="blue", label=f"PSD_mode_{mode_5}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_int, np.cumsum(PSD_residual_mode_5_1st_int), color="cornflowerblue", label=f"PSD_mode_{mode_5}_1st_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_5_freq_t_1st_ideal[np.where(PSD_residual_mode_5_freq_t_1st_ideal <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_5_1st_ideal[np.where(PSD_residual_mode_5_freq_t_1st_ideal <= stage_1_freq_lim)]), '--', label=f"PSD_mode_{mode_5}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_ideal, np.cumsum(PSD_residual_mode_5_2nd_ideal), label=f"PSD_mode_{mode_5}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_ideal, np.cumsum(PSD_residual_mode_5_1st_ideal), label=f"PSD_mode_{mode_5}_1st_{label_ideal}")


    plt.title(f"Cumulative PSD {mode_5}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD")







# --------------------------------------------------- Ratio ---------------------------------------------------#

if RL:
    PSD_mode_1_ratio_RL = PSD_residual_mode_1_1st_RL/PSD_residual_mode_1_2nd_RL
    PSD_mode_2_ratio_RL = PSD_residual_mode_2_1st_RL/PSD_residual_mode_2_2nd_RL
    PSD_mode_3_ratio_RL = PSD_residual_mode_3_1st_RL/PSD_residual_mode_3_2nd_RL
    PSD_mode_4_ratio_RL = PSD_residual_mode_4_1st_RL/PSD_residual_mode_4_2nd_RL
    PSD_mode_5_ratio_RL = PSD_residual_mode_5_1st_RL/PSD_residual_mode_5_2nd_RL

if integrator:
    PSD_mode_1_ratio_int = PSD_residual_mode_1_1st_int/PSD_residual_mode_1_2nd_int
    PSD_mode_2_ratio_int = PSD_residual_mode_2_1st_int/PSD_residual_mode_2_2nd_int
    PSD_mode_3_ratio_int = PSD_residual_mode_3_1st_int/PSD_residual_mode_3_2nd_int
    PSD_mode_4_ratio_int = PSD_residual_mode_4_1st_int/PSD_residual_mode_4_2nd_int
    PSD_mode_5_ratio_int = PSD_residual_mode_5_1st_int/PSD_residual_mode_5_2nd_int


if ideal:
    PSD_mode_1_ratio_ideal = PSD_residual_mode_1_1st_ideal/PSD_residual_mode_1_2nd_ideal
    PSD_mode_2_ratio_ideal = PSD_residual_mode_2_1st_ideal/PSD_residual_mode_2_2nd_ideal
    PSD_mode_3_ratio_ideal = PSD_residual_mode_3_1st_ideal/PSD_residual_mode_3_2nd_ideal
    PSD_mode_4_ratio_ideal = PSD_residual_mode_4_1st_ideal/PSD_residual_mode_4_2nd_ideal
    PSD_mode_5_ratio_ideal = PSD_residual_mode_5_1st_ideal/PSD_residual_mode_5_2nd_ideal



plt.figure()
if RL:
    if stage_2:
        plt.plot(PSD_residual_mode_1_freq_t_2nd_RL, PSD_mode_1_ratio_RL, color="red", label=f"PSD_mode_{mode_1}_2nd_{label_RL}")
    else: plt.plot(PSD_residual_mode_1_freq_t_1st_RL, PSD_residual_mode_1_1st_RL, color="indianred", label=f"PSD_mode_{mode_1}_1st_{label_RL}")
if integrator:
    if stage_2:
        plt.plot(PSD_residual_mode_1_freq_t_2nd_int, PSD_mode_1_ratio_int, color="blue", label=f"PSD_mode_{mode_1}_2nd_{label_int}")
    else: plt.plot(PSD_residual_mode_1_freq_t_1st_int, PSD_residual_mode_1_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_1}_1st_{label_int}")
if ideal:
    if stage_2:
        plt.plot(PSD_residual_mode_1_freq_t_2nd_ideal, PSD_mode_1_ratio_ideal, label=f"PSD_mode_{mode_1}_2nd_{label_ideal}")
    else: plt.plot(PSD_residual_mode_1_freq_t_1st_ideal, PSD_residual_mode_1_1st_ideal, label=f"PSD_mode_{mode_1}_1st_{label_ideal}")

plt.title(f"Gain of 2nd stage over 1st stage {mode_1}, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("ratio")



plt.figure()
if RL:
    if stage_2:
        plt.plot(PSD_residual_mode_2_freq_t_2nd_RL, PSD_mode_2_ratio_RL, color="red", label=f"PSD_mode_{mode_2}_2nd_{label_RL}")
    else: plt.plot(PSD_residual_mode_2_freq_t_1st_RL, PSD_residual_mode_2_1st_RL, color="indianred", label=f"PSD_mode_{mode_2}_1st_{label_RL}")
if integrator:
    if stage_2:
        plt.plot(PSD_residual_mode_2_freq_t_2nd_int, PSD_mode_2_ratio_int, color="blue", label=f"PSD_mode_{mode_2}_2nd_{label_int}")
    else: plt.plot(PSD_residual_mode_2_freq_t_1st_int, PSD_residual_mode_2_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_2}_1st_{label_int}")
if ideal:
    if stage_2:
        plt.plot(PSD_residual_mode_2_freq_t_2nd_ideal, PSD_mode_2_ratio_ideal, label=f"PSD_mode_{mode_2}_2nd_{label_ideal}")
    else: plt.plot(PSD_residual_mode_2_freq_t_1st_ideal, PSD_residual_mode_2_1st_ideal, label=f"PSD_mode_{mode_2}_1st_{label_ideal}")


plt.title(f"Gain of 2nd stage over 1st stage {mode_2}, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("ratio")



plt.figure()
if RL:
    if stage_2:
        plt.plot(PSD_residual_mode_3_freq_t_2nd_RL, PSD_mode_3_ratio_RL, color="red", label=f"PSD_mode_{mode_3}_2nd_{label_RL}")
    else: plt.plot(PSD_residual_mode_3_freq_t_1st_RL, PSD_residual_mode_3_1st_RL, color="indianred", label=f"PSD_mode_{mode_3}_1st_{label_RL}")
if integrator:
    if stage_2:
        plt.plot(PSD_residual_mode_3_freq_t_2nd_int, PSD_mode_3_ratio_int, color="blue", label=f"PSD_mode_{mode_3}_2nd_{label_int}")
    else: plt.plot(PSD_residual_mode_3_freq_t_1st_int, PSD_residual_mode_3_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_3}_1st_{label_int}")
if ideal:
    if stage_2:
        plt.plot(PSD_residual_mode_3_freq_t_2nd_ideal, PSD_mode_3_ratio_ideal, label=f"PSD_mode_{mode_3}_2nd_{label_ideal}")
    else: plt.plot(PSD_residual_mode_3_freq_t_1st_ideal, PSD_residual_mode_3_1st_ideal, label=f"PSD_mode_{mode_3}_1st_{label_ideal}")



plt.title(f"Gain of 2nd stage over 1st stage {mode_3}, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("ratio")





plt.figure()
if RL:
    if stage_2:
        plt.plot(PSD_residual_mode_4_freq_t_2nd_RL, PSD_mode_4_ratio_RL, color="red", label=f"PSD_mode_{mode_4}_2nd_{label_RL}")
    else: plt.plot(PSD_residual_mode_4_freq_t_1st_RL, PSD_residual_mode_4_1st_RL, color="indianred", label=f"PSD_mode_{mode_4}_1st_{label_RL}")
if integrator:
    if stage_2:
        plt.plot(PSD_residual_mode_4_freq_t_2nd_int, PSD_mode_4_ratio_int, color="blue", label=f"PSD_mode_{mode_4}_2nd_{label_int}")
    else: plt.plot(PSD_residual_mode_4_freq_t_1st_int, PSD_residual_mode_4_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_4}_1st_{label_int}")
if ideal:
    if stage_2:
        plt.plot(PSD_residual_mode_4_freq_t_2nd_ideal, PSD_mode_4_ratio_ideal, label=f"PSD_mode_{mode_4}_2nd_{label_ideal}")
    else: plt.plot(PSD_residual_mode_4_freq_t_1st_ideal, PSD_residual_mode_4_1st_ideal, color="indianred", label=f"PSD_mode_{mode_4}_1st_{label_ideal}")

plt.title(f"Gain of 2nd stage over 1st stage {mode_4}, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("ratio")



plt.figure()
if RL:
    if stage_2:
        plt.plot(PSD_residual_mode_5_freq_t_2nd_RL, PSD_mode_5_ratio_RL, color="red", label=f"PSD_mode_{mode_5}_2nd_{label_RL}")
    else: plt.plot(PSD_residual_mode_5_freq_t_1st_RL, PSD_residual_mode_5_1st_RL, color="indianred", label=f"PSD_mode_{mode_5}_1st_{label_RL}")
if integrator:
    if stage_2:
        plt.plot(PSD_residual_mode_5_freq_t_2nd_int, PSD_mode_5_ratio_int, color="blue", label=f"PSD_mode_{mode_5}_2nd_{label_int}")
    else: plt.plot(PSD_residual_mode_5_freq_t_1st_int, PSD_residual_mode_5_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_5}_1st_{label_int}")
if ideal:
    if stage_2:
        plt.plot(PSD_residual_mode_5_freq_t_2nd_ideal, PSD_mode_5_ratio_ideal, label=f"PSD_mode_{mode_5}_2nd_{label_ideal}")
    else: plt.plot(PSD_residual_mode_5_freq_t_1st_ideal, PSD_residual_mode_5_1st_ideal, label=f"PSD_mode_{mode_5}_1st_{label_ideal}")


plt.title(f"Gain of 2nd stage over 1st stage {mode_5}, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("ratio")





# ---------------------------------------------------temporal Error transfer function---------------------------------------------------#

#to select for the common frequencies since the 1st stage OL was running at 200Hz on sky
idx_CL = np.isin(np.round(PSD_residual_mode_1_freq_t_2nd_RL, 6), np.round(PSD_atm_mode_1_freq_t, 6))
idx_atm = np.isin(np.round(PSD_atm_mode_1_freq_t, 6), np.round(PSD_residual_mode_1_freq_t_2nd_RL, 6))
f2_selected  = PSD_residual_mode_1_freq_t_2nd_RL[idx_CL]
psd2_selected = PSD_residual_mode_1_2nd_RL[idx_CL]

if plot_tPSD:
    if RL:
        tETF_mode_1_1st_RL = PSD_residual_mode_1_1st_RL[idx_CL] / PSD_atm_mode_1[idx_atm]
        tETF_mode_2_1st_RL = PSD_residual_mode_2_1st_RL[idx_CL] / PSD_atm_mode_2[idx_atm]
        tETF_mode_3_1st_RL = PSD_residual_mode_3_1st_RL[idx_CL] / PSD_atm_mode_3[idx_atm]
        tETF_mode_4_1st_RL = PSD_residual_mode_4_1st_RL[idx_CL] / PSD_atm_mode_4[idx_atm]
        tETF_mode_5_1st_RL = PSD_residual_mode_5_1st_RL[idx_CL] / PSD_atm_mode_5[idx_atm]

        if stage_2:
            tETF_mode_1_2nd_RL = PSD_residual_mode_1_2nd_RL[idx_CL] / PSD_atm_mode_1[idx_atm]
            tETF_mode_2_2nd_RL = PSD_residual_mode_2_2nd_RL[idx_CL] / PSD_atm_mode_2[idx_atm]
            tETF_mode_3_2nd_RL = PSD_residual_mode_3_2nd_RL[idx_CL] / PSD_atm_mode_3[idx_atm]
            tETF_mode_4_2nd_RL = PSD_residual_mode_4_2nd_RL[idx_CL] / PSD_atm_mode_4[idx_atm]
            tETF_mode_5_2nd_RL = PSD_residual_mode_5_2nd_RL[idx_CL] / PSD_atm_mode_5[idx_atm]


    if integrator:
        tETF_mode_1_1st_int = PSD_residual_mode_1_1st_int[idx_CL] / PSD_atm_mode_1[idx_atm]
        tETF_mode_2_1st_int = PSD_residual_mode_2_1st_int[idx_CL] / PSD_atm_mode_2[idx_atm]
        tETF_mode_3_1st_int = PSD_residual_mode_3_1st_int[idx_CL] / PSD_atm_mode_3[idx_atm]
        tETF_mode_4_1st_int = PSD_residual_mode_4_1st_int[idx_CL] / PSD_atm_mode_4[idx_atm]
        tETF_mode_5_1st_int = PSD_residual_mode_5_1st_int[idx_CL] / PSD_atm_mode_5[idx_atm]

        if stage_2:
            tETF_mode_1_2nd_int = PSD_residual_mode_1_2nd_int[idx_CL] / PSD_atm_mode_1[idx_atm]
            tETF_mode_2_2nd_int = PSD_residual_mode_2_2nd_int[idx_CL] / PSD_atm_mode_2[idx_atm]
            tETF_mode_3_2nd_int = PSD_residual_mode_3_2nd_int[idx_CL] / PSD_atm_mode_3[idx_atm]
            tETF_mode_4_2nd_int = PSD_residual_mode_4_2nd_int[idx_CL] / PSD_atm_mode_4[idx_atm]
            tETF_mode_5_2nd_int = PSD_residual_mode_5_2nd_int[idx_CL] / PSD_atm_mode_5[idx_atm]


    if ideal:
        tETF_mode_1_1st_ideal = PSD_residual_mode_1_1st_ideal[idx_CL] / PSD_atm_mode_1[idx_atm]
        tETF_mode_2_1st_ideal = PSD_residual_mode_2_1st_ideal[idx_CL] / PSD_atm_mode_2[idx_atm]
        tETF_mode_3_1st_ideal = PSD_residual_mode_3_1st_ideal[idx_CL] / PSD_atm_mode_3[idx_atm]
        tETF_mode_4_1st_ideal = PSD_residual_mode_4_1st_ideal[idx_CL] / PSD_atm_mode_4[idx_atm]
        tETF_mode_5_1st_ideal = PSD_residual_mode_5_1st_ideal[idx_CL] / PSD_atm_mode_5[idx_atm]

        if stage_2:
            tETF_mode_1_2nd_ideal = PSD_residual_mode_1_2nd_ideal[idx_CL] / PSD_atm_mode_1[idx_atm]
            tETF_mode_2_2nd_ideal = PSD_residual_mode_2_2nd_ideal[idx_CL] / PSD_atm_mode_2[idx_atm]
            tETF_mode_3_2nd_ideal = PSD_residual_mode_3_2nd_ideal[idx_CL] / PSD_atm_mode_3[idx_atm]
            tETF_mode_4_2nd_ideal = PSD_residual_mode_4_2nd_ideal[idx_CL] / PSD_atm_mode_4[idx_atm]
            tETF_mode_5_2nd_ideal = PSD_residual_mode_5_2nd_ideal[idx_CL] / PSD_atm_mode_5[idx_atm]



    #tip
    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_RL[idx_CL], tETF_mode_1_1st_RL, '--', color="indianred", label=f"ETF mode_{mode_1}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_RL[idx_CL], tETF_mode_1_2nd_RL, color="red", label=f"ETF mode_{mode_1}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_RL, tETF_mode_1_1st_RL, color="indianred", label=f"ETF mode_{mode_1}_1st_{label_RL}")

    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_int[idx_CL], tETF_mode_1_1st_int, '--', color="cornflowerblue", label=f"ETF mode_{mode_1}_1st_{label_int}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_int[idx_CL], tETF_mode_1_2nd_int, color="blue", label=f"ETF mode_{mode_1}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_int, tETF_mode_1_1st_int, color="cornflowerblue", label=f"ETF mode_{mode_1}_1st_{label_int}")


    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_ideal[idx_CL], tETF_mode_1_1st_ideal, '--', label=f"ETF mode_{mode_1}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_ideal[idx_CL], tETF_mode_1_2nd_ideal, label=f"ETF mode_{mode_1}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_ideal, tETF_mode_1_1st_ideal, label=f"ETF mode_{mode_1}_1st_{label_ideal}")
    plt.title("temporal error transfer functions")
    plt.ylabel("ETF")
    plt.xlabel("frequency Hz")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(right=np.max(freq_lim))
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()


    #tilt
    plt.figure()


    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_2_freq_t_1st_RL[idx_CL], tETF_mode_2_1st_RL, '--', color="indianred", label=f"ETF mode_{mode_2}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_RL[idx_CL], tETF_mode_2_2nd_RL, color="red", label=f"ETF mode_{mode_2}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_RL, tETF_mode_2_1st_RL, color="indianred", label=f"ETF mode_{mode_2}_1st_{label_RL}")

    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_2_freq_t_1st_int[idx_CL], tETF_mode_2_1st_int, '--', color="cornflowerblue", label=f"ETF mode_{mode_2}_1st_{label_int}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_int[idx_CL], tETF_mode_2_2nd_int, color="blue", label=f"ETF mode_{mode_2}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_int, tETF_mode_2_1st_int, color="cornflowerblue", label=f"ETF mode_{mode_2}_1st_{label_int}")


    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_2_freq_t_1st_ideal[idx_CL], tETF_mode_2_1st_ideal, '--', label=f"ETF mode_{mode_2}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_ideal[idx_CL], tETF_mode_2_2nd_ideal, label=f"ETF mode_{mode_2}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_ideal, tETF_mode_2_1st_ideal, label=f"ETF mode_{mode_2}_1st_{label_ideal}")
    plt.title("temporal error transfer functions")
    plt.ylabel("ETF")
    plt.xlabel("frequency Hz")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(right=np.max(freq_lim))
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()

    #100
    plt.figure()

    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_3_freq_t_1st_RL[idx_CL], tETF_mode_3_1st_RL, '--', color="indianred", label=f"ETF mode_{mode_3}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_RL[idx_CL], tETF_mode_3_2nd_RL, color="red", label=f"ETF mode_{mode_3}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_RL, tETF_mode_3_1st_RL, color="indianred", label=f"ETF mode_{mode_3}_1st_{label_RL}")


    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_3_freq_t_1st_int[idx_CL], tETF_mode_3_1st_int, '--', color="cornflowerblue", label=f"ETF mode_{mode_3}_1st_{label_int}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_int[idx_CL], tETF_mode_3_2nd_int, color="blue", label=f"ETF mode_{mode_3}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_int, tETF_mode_3_1st_int, color="cornflowerblue", label=f"ETF mode_{mode_3}_1st_{label_int}")


    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_3_freq_t_1st_ideal[idx_CL], tETF_mode_3_1st_ideal, '--', label=f"ETF mode_{mode_3}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_ideal[idx_CL], tETF_mode_3_2nd_ideal, label=f"ETF mode_{mode_3}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_ideal, tETF_mode_3_1st_ideal, label=f"ETF mode_{mode_3}_1st_{label_ideal}")

    plt.title("temporal error transfer functions")
    plt.ylabel("ETF")
    plt.xlabel("frequency Hz")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(right=np.max(freq_lim))
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()

    #200
    plt.figure()

    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_4_freq_t_1st_RL[idx_CL], tETF_mode_4_1st_RL, '--', color="indianred", label=f"ETF mode_{mode_4}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_RL[idx_CL], tETF_mode_4_2nd_RL, color="red", label=f"ETF mode_{mode_4}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_RL, tETF_mode_4_1st_RL, color="indianred", label=f"ETF mode_{mode_4}_1st_{label_RL}")

    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_4_freq_t_1st_int[idx_CL], tETF_mode_4_1st_int, '--', color="cornflowerblue", label=f"ETF mode_{mode_4}_1st_{label_int}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_int[idx_CL], tETF_mode_4_2nd_int, color="blue", label=f"ETF mode_{mode_4}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_int, tETF_mode_4_1st_int, color="cornflowerblue", label=f"ETF mode_{mode_4}_1st_{label_int}")

    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_4_freq_t_1st_ideal[idx_CL], tETF_mode_4_1st_ideal, '--', label=f"ETF mode_{mode_4}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_ideal[idx_CL], tETF_mode_4_2nd_ideal, label=f"ETF mode_{mode_4}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_ideal, tETF_mode_4_1st_ideal, label=f"ETF mode_{mode_4}_1st_{label_ideal}")

    plt.title("temporal error transfer functions")
    plt.ylabel("ETF")
    plt.xlabel("frequency Hz")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(right=np.max(freq_lim))
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()


    #5
    plt.figure()


    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_5_freq_t_1st_RL[idx_CL], tETF_mode_5_1st_RL, '--', color="indianred", label=f"ETF mode_{mode_5}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_RL[idx_CL], tETF_mode_5_2nd_RL, color="red", label=f"ETF mode_{mode_5}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_RL, tETF_mode_5_1st_RL, color="indianred", label=f"ETF mode_{mode_5}_1st_{label_RL}")

    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_5_freq_t_1st_int[idx_CL], tETF_mode_5_1st_int, '--', color="cornflowerblue", label=f"ETF mode_{mode_5}_1st_{label_int}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_int[idx_CL], tETF_mode_5_2nd_int, color="blue", label=f"ETF mode_{mode_5}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_int, tETF_mode_5_1st_int, color="cornflowerblue", label=f"ETF mode_{mode_5}_1st_{label_int}")

    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_5_freq_t_1st_ideal[idx_CL], tETF_mode_5_1st_ideal, '--', label=f"ETF mode_{mode_5}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_ideal[idx_CL], tETF_mode_5_2nd_ideal, label=f"ETF mode_{mode_5}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_ideal, tETF_mode_5_1st_ideal, label=f"ETF mode_{mode_5}_1st_{label_ideal}")


    plt.title("temporal error transfer functions")
    plt.ylabel("ETF")
    plt.xlabel("frequency Hz")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(right=np.max(freq_lim))
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()




plt.show()





























