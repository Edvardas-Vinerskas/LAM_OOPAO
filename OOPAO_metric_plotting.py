"""
*when doing the temporal PSD you should also implement some kind of strehl condition
	problem is that then the temporal curve can get discontinuities
	the most straightforward solution seems to be just cutting out a continuous section of the timeseries where the strehl ratio is above your threshold
* write try except statements for when plots are non-sensical

"""

from scipy import signal
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.DeformableMirror import DeformableMirror

from po4ao_edw.OOPAO_environment_PWFS import OOPAO_environment_PWFS
from po4ao_edw.OOPAO_environment_ZWFS import OOPAO_environment_ZWFS


env = OOPAO_environment_ZWFS()
#env = OOPAO_environment_PWFS()

"""DM = DeformableMirror(telescope    = env.TEL,
                      nSubap       = env.N_SUBAPERTURE_pyr,
                      mechCoupling = env.MECH_COUPLING)

full_M2C = compute_KL_basis(env.TEL,
                            env.ATM,
                            DM,
                            lim=0)


KL_basis_dm = (DM.modes @ full_M2C) * np.tile(env.TEL.pupil.flatten()[:, None], full_M2C.shape[1])
projector_kl = np.linalg.pinv(KL_basis_dm)"""


#TODO don't forget to also change your atmosphere parameters when loading files (atm_OPD_array)
#TODO could you in fact rewrite this so unneccesary stuff is not loaded?
RL                   = True
integrator           = True
ideal                = True
stage_2              = True
stage_1_plus_stage_2 = False
atm_RL               = True
atm_int              = False
atm_ideal            = False
plot_timeseries      = True
plot_tPSD            = True
freq_lim             = 900

directory_name_RL  = 'vZWFS_1st_2nd_noise_03_phnoise1_read_4_4_QE08_mag2_epis20_iters40_warmupperc05'#vZWFS_1st_2nd_noise_03_phnoise1_read_4_4_QE08_mag2_epis20_eplen1500_noonline
directory_name_int = 'vZWFS_integrator_phnoise1_read_4_4_QE08_mag2_12k' #vZWFS_integrator_phnoise1_read_4_4_QE08_mag2_12k
directory_name_ideal = 'vZWFS_metrics_ideal'#vZWFS_metrics_ideal
label_RL = "RL"
label_int = "perfectvZWFS"
label_ideal = "ideal"
KL_frame_thresh = 1000
KL_frame_end    = -1 #21000 or 6000
#modes for PSDs
mode_1 = 1
mode_2 = 10
mode_3 = 20
mode_4 = 40
mode_5 = 80




#RL
if RL:
    residual_error_RL = np.load(f"temp_save_dir/{directory_name_RL}/residual_error.npy")
    strehl_array_1st_RL = np.load(f"temp_save_dir/{directory_name_RL}/strehl_array_1st.npy")

    if stage_2:
        strehl_array_2nd_RL = np.load(f"temp_save_dir/{directory_name_RL}/strehl_array_2nd.npy")
        modes_2nd_stage_RL = np.load(f"temp_save_dir/{directory_name_RL}/modes_2nd_stage.npy")


    modes_1st_stage_RL = np.load(f"temp_save_dir/{directory_name_RL}/modes_1st_stage.npy")
    modes_atm_RL = np.load(f"temp_save_dir/{directory_name_RL}/modes_atm.npy")


    total_err_array_RL = np.load(f"temp_save_dir/{directory_name_RL}/total_err_array.npy") #not useful for now

    if RL == True:
        dynamics_loss = np.load(f"temp_save_dir/{directory_name_RL}/dynamics_loss.npy") #not useful for now
        policy_loss = np.load(f"temp_save_dir/{directory_name_RL}/policy_loss.npy") #not useful for now

    #YOU SHOULD ALSO CHECK IF YOU ONLY HAVE AN EPISODE OF DATA IN ALL OF THESE BECAUSE I DON4T REMEMBER
    frequency_RL = np.load(f"temp_save_dir/{directory_name_RL}/frequency.npy")
    time_plot_RL = np.load(f"temp_save_dir/{directory_name_RL}/time_array.npy")






#integrator
if integrator:
    residual_error_int = np.load(f"temp_save_dir/{directory_name_int}/residual_error.npy")
    strehl_array_1st_int = np.load(f"temp_save_dir/{directory_name_int}/strehl_array_1st.npy")


    if stage_2:
        strehl_array_2nd_int = np.load(f"temp_save_dir/{directory_name_int}/strehl_array_2nd.npy")
        modes_2nd_stage_int = np.load(f"temp_save_dir/{directory_name_int}/modes_2nd_stage.npy")


    modes_1st_stage_int = np.load(f"temp_save_dir/{directory_name_int}/modes_1st_stage.npy")
    modes_atm_int = np.load(f"temp_save_dir/{directory_name_int}/modes_atm.npy")


    total_err_array_int = np.load(f"temp_save_dir/{directory_name_int}/total_err_array.npy") #not useful for now

    #YOU SHOULD ALSO CHECK IF YOU ONLY HAVE AN EPISODE OF DATA IN ALL OF THESE BECAUSE I DON4T REMEMBER
    frequency_int = np.load(f"temp_save_dir/{directory_name_int}/frequency.npy")
    time_plot_int = np.load(f"temp_save_dir/{directory_name_int}/time_array.npy")



#ideal
if ideal:
    residual_error_ideal = np.load(f"temp_save_dir/{directory_name_ideal}/residual_error.npy")
    strehl_array_1st_ideal = np.load(f"temp_save_dir/{directory_name_ideal}/strehl_array_1st.npy")

    if stage_2:
        strehl_array_2nd_ideal = np.load(f"temp_save_dir/{directory_name_ideal}/strehl_array_2nd.npy")
        modes_2nd_stage_ideal = np.load(f"temp_save_dir/{directory_name_ideal}/modes_2nd_stage.npy")


    modes_1st_stage_ideal = np.load(f"temp_save_dir/{directory_name_ideal}/modes_1st_stage.npy")
    modes_atm_ideal = np.load(f"temp_save_dir/{directory_name_ideal}/modes_atm.npy")


    total_err_array_ideal = np.load(f"temp_save_dir/{directory_name_ideal}/total_err_array.npy") #not useful for now

    #YOU SHOULD ALSO CHECK IF YOU ONLY HAVE AN EPISODE OF DATA IN ALL OF THESE BECAUSE I DON4T REMEMBER
    frequency_ideal = np.load(f"temp_save_dir/{directory_name_ideal}/frequency.npy")
    time_plot_ideal = np.load(f"temp_save_dir/{directory_name_ideal}/time_array.npy")

if atm_RL:
    modes_atm = modes_atm_RL
    time_plot_atm = time_plot_RL
    f_samp_atm = frequency_RL

if atm_int:
    modes_atm = modes_atm_int
    time_plot_atm = time_plot_int
    f_samp_atm = frequency_int

if atm_ideal:
    modes_atm = modes_atm_ideal
    time_plot_atm = time_plot_ideal
    f_samp_atm = frequency_ideal

# ---------------------------------------------------Loss---------------------------------------------------#
if RL:
    plt.figure()
    plt.subplot(121)
    plt.title("dynamics_loss warmup")
    plt.plot(dynamics_loss)
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.yscale('log')
    plt.subplot(122)
    plt.title("policy_loss warmup")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.plot(policy_loss)
    plt.yscale('log')



# ---------------------------------------------------Strehl---------------------------------------------------#
if RL:
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


# ---------------------------------------------------Zernike/KL decomposition---------------------------------------------------#
#RL
if RL:
    modes_1st_stage_RL_masked = modes_1st_stage_RL[KL_frame_thresh:KL_frame_end, :]
    coefs_var_1st_stage_masked_RL = np.var(np.asarray(modes_1st_stage_RL_masked), axis = 0)
    coefs_var_1st_stage_RL = np.var(np.asarray(modes_1st_stage_RL), axis = 0)

    if stage_2:
        modes_2nd_stage_RL_masked = modes_2nd_stage_RL[KL_frame_thresh:KL_frame_end, :]
        coefs_var_2nd_stage_masked_RL = np.var(np.asarray(modes_2nd_stage_RL_masked), axis=0)
        coefs_var_2nd_stage_RL = np.var(np.asarray(modes_2nd_stage_RL), axis = 0)


#integrator
if integrator:
    modes_1st_stage_int_masked = modes_1st_stage_int[KL_frame_thresh:, :]
    coefs_var_1st_stage_masked_int = np.var(np.asarray(modes_1st_stage_int_masked), axis = 0)
    coefs_var_1st_stage_int = np.var(np.asarray(modes_1st_stage_int), axis=0)

    if stage_2:
        modes_2nd_stage_int_masked = modes_2nd_stage_int[KL_frame_thresh:, :]
        coefs_var_2nd_stage_masked_int = np.var(np.asarray(modes_2nd_stage_int_masked), axis = 0)
        coefs_var_2nd_stage_int = np.var(np.asarray(modes_2nd_stage_int), axis = 0)


#ideal
if ideal:
    modes_1st_stage_ideal_masked = modes_1st_stage_ideal[KL_frame_thresh:, :]
    coefs_var_1st_stage_masked_ideal = np.var(np.asarray(modes_1st_stage_ideal_masked), axis = 0)
    coefs_var_1st_stage_ideal = np.var(np.asarray(modes_1st_stage_ideal), axis=0)

    if stage_2:
        modes_2nd_stage_ideal_masked = modes_2nd_stage_ideal[KL_frame_thresh:, :]
        coefs_var_2nd_stage_masked_ideal = np.var(np.asarray(modes_2nd_stage_ideal_masked), axis = 0)
        coefs_var_2nd_stage_ideal = np.var(np.asarray(modes_2nd_stage_ideal), axis = 0)


#atmosphere
coefs_var_atm = np.var(np.asarray(modes_atm[KL_frame_thresh:, :]), axis = 0)


plt.figure()
plt.plot(coefs_var_atm, color="black",label=f"KL coeffs for atmospheric phase")
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

plt.title(f"KL coefficients for corrected vs atmosphere phase")
plt.yscale("log")
plt.xscale("log")
plt.tight_layout()
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()



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
    PSD_residual_mode_1_freq_t_1st_RL, PSD_residual_mode_1_1st_RL = welch_method_scipy(residual_mode_1_curve_1st_RL, f_samp_RL)
    PSD_residual_mode_2_freq_t_1st_RL, PSD_residual_mode_2_1st_RL = welch_method_scipy(residual_mode_2_curve_1st_RL, f_samp_RL)
    PSD_residual_mode_3_freq_t_1st_RL, PSD_residual_mode_3_1st_RL = welch_method_scipy(residual_mode_3_curve_1st_RL, f_samp_RL)
    PSD_residual_mode_4_freq_t_1st_RL, PSD_residual_mode_4_1st_RL = welch_method_scipy(residual_mode_4_curve_1st_RL, f_samp_RL)
    PSD_residual_mode_5_freq_t_1st_RL, PSD_residual_mode_5_1st_RL = welch_method_scipy(residual_mode_5_curve_1st_RL, f_samp_RL)


    if stage_2:
        PSD_residual_mode_1_freq_t_2nd_RL, PSD_residual_mode_1_2nd_RL = welch_method_scipy(residual_mode_1_curve_2nd_RL, f_samp_RL)
        PSD_residual_mode_2_freq_t_2nd_RL, PSD_residual_mode_2_2nd_RL = welch_method_scipy(residual_mode_2_curve_2nd_RL, f_samp_RL)
        PSD_residual_mode_3_freq_t_2nd_RL, PSD_residual_mode_3_2nd_RL = welch_method_scipy(residual_mode_3_curve_2nd_RL, f_samp_RL)
        PSD_residual_mode_4_freq_t_2nd_RL, PSD_residual_mode_4_2nd_RL = welch_method_scipy(residual_mode_4_curve_2nd_RL, f_samp_RL)
        PSD_residual_mode_5_freq_t_2nd_RL, PSD_residual_mode_5_2nd_RL = welch_method_scipy(residual_mode_5_curve_2nd_RL, f_samp_RL)


if integrator:
    PSD_residual_mode_1_freq_t_1st_int, PSD_residual_mode_1_1st_int = welch_method_scipy(residual_mode_1_curve_1st_int, f_samp_int)
    PSD_residual_mode_2_freq_t_1st_int, PSD_residual_mode_2_1st_int = welch_method_scipy(residual_mode_2_curve_1st_int, f_samp_int)
    PSD_residual_mode_3_freq_t_1st_int, PSD_residual_mode_3_1st_int = welch_method_scipy(residual_mode_3_curve_1st_int, f_samp_int)
    PSD_residual_mode_4_freq_t_1st_int, PSD_residual_mode_4_1st_int = welch_method_scipy(residual_mode_4_curve_1st_int, f_samp_int)
    PSD_residual_mode_5_freq_t_1st_int, PSD_residual_mode_5_1st_int = welch_method_scipy(residual_mode_5_curve_1st_int, f_samp_int)

    if stage_2:
        PSD_residual_mode_1_freq_t_2nd_int, PSD_residual_mode_1_2nd_int = welch_method_scipy(residual_mode_1_curve_2nd_int, f_samp_int)
        PSD_residual_mode_2_freq_t_2nd_int, PSD_residual_mode_2_2nd_int = welch_method_scipy(residual_mode_2_curve_2nd_int, f_samp_int)
        PSD_residual_mode_3_freq_t_2nd_int, PSD_residual_mode_3_2nd_int = welch_method_scipy(residual_mode_3_curve_2nd_int, f_samp_int)
        PSD_residual_mode_4_freq_t_2nd_int, PSD_residual_mode_4_2nd_int = welch_method_scipy(residual_mode_4_curve_2nd_int, f_samp_int)
        PSD_residual_mode_5_freq_t_2nd_int, PSD_residual_mode_5_2nd_int = welch_method_scipy(residual_mode_5_curve_2nd_int, f_samp_int)


if ideal:
    #tip
    PSD_residual_mode_1_freq_t_1st_ideal, PSD_residual_mode_1_1st_ideal = welch_method_scipy(residual_mode_1_curve_1st_ideal, f_samp_ideal)
    PSD_residual_mode_2_freq_t_1st_ideal, PSD_residual_mode_2_1st_ideal = welch_method_scipy(residual_mode_2_curve_1st_ideal, f_samp_ideal)
    PSD_residual_mode_3_freq_t_1st_ideal, PSD_residual_mode_3_1st_ideal = welch_method_scipy(residual_mode_3_curve_1st_ideal, f_samp_ideal)
    PSD_residual_mode_4_freq_t_1st_ideal, PSD_residual_mode_4_1st_ideal = welch_method_scipy(residual_mode_4_curve_1st_ideal, f_samp_ideal)
    PSD_residual_mode_5_freq_t_1st_ideal, PSD_residual_mode_5_1st_ideal = welch_method_scipy(residual_mode_5_curve_1st_ideal, f_samp_ideal)

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
                plt.plot(time_plot_RL, residual_mode_1_curve_1st_RL_full, label=f"mode_{mode_1}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_1_curve_2nd_RL_full, label=f"mode_{mode_1}_2nd_stage_{label_RL}")
        else: plt.plot(time_plot_RL, residual_mode_1_curve_1st_RL_full, label=f"mode_{mode_1}_1st_stage_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_int, residual_mode_1_curve_1st_int_full, label=f"mode_{mode_1}_1st_stage_{label_int}")
            plt.plot(time_plot_int, residual_mode_1_curve_2nd_int_full, label=f"mode_{mode_1}_2nd_stage_{label_int}")
        else: plt.plot(time_plot_int, residual_mode_1_curve_1st_int_full, label=f"mode_{mode_1}_1st_stage_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_ideal, residual_mode_1_curve_1st_ideal_full, label=f"mode_{mode_1}_1st_stage_{label_ideal}")
            plt.plot(time_plot_ideal, residual_mode_1_curve_2nd_ideal_full, label=f"mode_{mode_1}_2nd_stage_{label_ideal}")
        else: plt.plot(time_plot_ideal, residual_mode_1_curve_1st_ideal_full, label=f"mode_{mode_1}_1st_stage_{label_ideal}")
    plt.plot(time_plot_atm, atm_mode_1_curve, color="black", label=f"atm_mode_{mode_1}_curve")
    plt.title(f"residual/atm timeseries mode_{mode_1}, gain {env.CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_1}")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_RL, residual_mode_2_curve_1st_RL_full, label=f"mode_{mode_2}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_2_curve_2nd_RL_full, label=f"mode_{mode_2}_2nd_stage_{label_RL}")
        else: plt.plot(time_plot_RL, residual_mode_2_curve_1st_RL_full, label=f"mode_{mode_2}_1st_stage_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_int, residual_mode_2_curve_1st_int_full, label=f"mode_{mode_2}_1st_stage_{label_int}")
            plt.plot(time_plot_int, residual_mode_2_curve_2nd_int_full, label=f"mode_{mode_2}_2nd_stage_{label_int}")
        else: plt.plot(time_plot_int, residual_mode_2_curve_1st_int_full, label=f"mode_{mode_2}_1st_stage_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_ideal, residual_mode_2_curve_1st_ideal_full, label=f"mode_{mode_2}_1st_stage_{label_ideal}")
            plt.plot(time_plot_ideal, residual_mode_2_curve_2nd_ideal_full, label=f"mode_{mode_2}_2nd_stage_{label_ideal}")
        else: plt.plot(time_plot_ideal, residual_mode_2_curve_1st_ideal_full, label=f"mode_{mode_2}_1st_stage_{label_ideal}")
    plt.plot(time_plot_atm, atm_mode_2_curve, color="black", label=f"atm_mode_{mode_2}_curve")
    plt.title(f"residual/atm timeseries mode_{mode_2}, gain {env.CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_2}")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_RL, residual_mode_3_curve_1st_RL_full, label=f"mode_{mode_3}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_3_curve_2nd_RL_full, label=f"mode_{mode_3}_2nd_stage_{label_RL}")
        else: plt.plot(time_plot_RL, residual_mode_3_curve_1st_RL_full, label=f"mode_{mode_3}_1st_stage_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_int, residual_mode_3_curve_1st_int_full, label=f"mode_{mode_3}_1st_stage_{label_int}")
            plt.plot(time_plot_int, residual_mode_3_curve_2nd_int_full, label=f"mode_{mode_3}_2nd_stage_{label_int}")
        else: plt.plot(time_plot_int, residual_mode_3_curve_1st_int_full, label=f"mode_{mode_3}_1st_stage_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_ideal, residual_mode_3_curve_1st_ideal_full, label=f"mode_{mode_3}_1st_stage__{label_ideal}")
            plt.plot(time_plot_ideal, residual_mode_3_curve_2nd_ideal_full, label=f"mode_{mode_3}_2nd_stage_{label_ideal}")
        else: plt.plot(time_plot_ideal, residual_mode_3_curve_1st_ideal_full, label=f"mode_{mode_3}_1st_stage_{label_ideal}")
    plt.plot(time_plot_atm, atm_mode_3_curve, color="black", label=f"atm_mode_{mode_3}_curve")
    plt.title(f"residual/atm timeseries mode_{mode_3}, gain {env.CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_3}")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_RL, residual_mode_4_curve_1st_RL_full, label=f"mode_{mode_4}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_4_curve_2nd_RL_full, label=f"mode__{mode_4}_2nd_stage_{label_RL}")
        else: plt.plot(time_plot_RL, residual_mode_4_curve_1st_RL_full, label=f"mode_{mode_4}_1st_stage_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_int, residual_mode_4_curve_1st_int_full, label=f"mode_{mode_4}_1st_stage_{label_int}")
            plt.plot(time_plot_int, residual_mode_4_curve_2nd_int_full, label=f"mode_{mode_4}_2nd_stage_{label_int}")
        else: plt.plot(time_plot_int, residual_mode_4_curve_1st_int_full, label=f"mode_{mode_4}_1st_stage_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_ideal, residual_mode_4_curve_1st_ideal_full, label=f"mode_{mode_4}_1st_stage_{label_ideal}")
            plt.plot(time_plot_ideal, residual_mode_4_curve_2nd_ideal_full, label=f"mode_{mode_4}_2nd_stage_{label_ideal}")
        else: plt.plot(time_plot_ideal, residual_mode_4_curve_1st_ideal_full, label=f"mode_{mode_4}_1st_stage_{label_ideal}")
    plt.plot(time_plot_atm, atm_mode_4_curve, color="black", label=f"atm_mode_{mode_4}_curve")
    plt.title(f"residual/atm timeseries mode_{mode_4}, gain {env.CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_4}")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_RL, residual_mode_5_curve_1st_RL_full, label=f"mode_{mode_5}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_5_curve_2nd_RL_full, label=f"mode_{mode_5}_2nd_stage_{label_RL}")
        else: plt.plot(time_plot_RL, residual_mode_5_curve_1st_RL_full, label=f"mode_{mode_5}_1st_stage_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_int, residual_mode_5_curve_1st_int_full, label=f"mode_{mode_5}_1st_stage_{label_int}")
            plt.plot(time_plot_int, residual_mode_5_curve_2nd_int_full, label=f"mode_{mode_5}_2nd_stage_{label_int}")
        else: plt.plot(time_plot_int, residual_mode_5_curve_1st_int_full, label=f"mode_{mode_5}_1st_stage_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_ideal, residual_mode_5_curve_1st_ideal_full, label=f"mode_{mode_5}_1st_stage_{label_ideal}")
            plt.plot(time_plot_ideal, residual_mode_5_curve_2nd_ideal_full, label=f"mode_{mode_5}_2nd_stage_{label_ideal}")
        else: plt.plot(time_plot_ideal, residual_mode_5_curve_1st_ideal_full, label=f"mode_{mode_5}_1st_stage_{label_ideal}")
    plt.plot(time_plot_atm, atm_mode_5_curve, color="black", label=f"atm_mode_{mode_5}_curve")
    plt.title(f"residual/atm timeseries mode_{mode_5}, gain {env.CL_gain_pyr}")
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
                plt.plot(PSD_residual_mode_1_freq_t_1st_RL[np.where(PSD_residual_mode_1_freq_t_1st_RL <= 250)]
                         , PSD_residual_mode_1_1st_RL[np.where(PSD_residual_mode_1_freq_t_1st_RL <= 250)], '--', label=f"PSD_mode_{mode_1}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_RL, PSD_residual_mode_1_2nd_RL, label=f"PSD_mode_{mode_1}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_RL, PSD_residual_mode_1_1st_RL, label=f"PSD_mode_{mode_1}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_int[np.where(PSD_residual_mode_1_freq_t_1st_int <= 250)]
                         , PSD_residual_mode_1_1st_int[np.where(PSD_residual_mode_1_freq_t_1st_int <= 250)], '--', label=f"PSD_mode_{mode_1}_1st_{label_int}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_int, PSD_residual_mode_1_2nd_int, label=f"PSD_mode_{mode_1}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_int, PSD_residual_mode_1_1st_int, label=f"PSD_mode_{mode_1}_1st_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_ideal[np.where(PSD_residual_mode_1_freq_t_1st_ideal <= 250)]
                         , PSD_residual_mode_1_1st_ideal[np.where(PSD_residual_mode_1_freq_t_1st_ideal <= 250)], '--', label=f"PSD_mode_{mode_1}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_ideal, PSD_residual_mode_1_2nd_ideal, label=f"PSD_mode_{mode_1}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_ideal, PSD_residual_mode_1_1st_ideal, label=f"PSD_mode_{mode_1}_1st_{label_ideal}")


    plt.plot(PSD_atm_mode_1_freq_t, PSD_atm_mode_1, color="black", label=f"atm_PSD_mode_{mode_1}")
    plt.title(f"residual PSD mode_{mode_1}, gain {env.CL_gain_pyr}")
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
                plt.plot(PSD_residual_mode_2_freq_t_1st_RL[np.where(PSD_residual_mode_2_freq_t_1st_RL <= 250)]
                         , PSD_residual_mode_2_1st_RL[np.where(PSD_residual_mode_2_freq_t_1st_RL <= 250)], '--', label=f"PSD_mode_{mode_2}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_RL, PSD_residual_mode_2_2nd_RL, label=f"PSD_mode_{mode_2}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_RL, PSD_residual_mode_2_1st_RL, label=f"PSD_mode_{mode_2}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_2_freq_t_1st_int[np.where(PSD_residual_mode_2_freq_t_1st_int <= 250)]
                         , PSD_residual_mode_2_1st_int[np.where(PSD_residual_mode_2_freq_t_1st_int <= 250)], '--', label=f"PSD_mode_{mode_2}_1st_{label_int}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_int, PSD_residual_mode_2_2nd_int, label=f"PSD_mode_{mode_2}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_int, PSD_residual_mode_2_1st_int, label=f"PSD_mode_{mode_2}_1st_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_2_freq_t_1st_ideal[np.where(PSD_residual_mode_2_freq_t_1st_ideal <= 250)]
                         , PSD_residual_mode_2_1st_ideal[np.where(PSD_residual_mode_2_freq_t_1st_ideal <= 250)], '--', label=f"PSD_mode_{mode_2}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_ideal, PSD_residual_mode_2_2nd_ideal, label=f"PSD_mode_{mode_2}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_ideal, PSD_residual_mode_2_1st_ideal, label=f"PSD_mode_{mode_2}_1st_{label_ideal}")


    plt.plot(PSD_atm_mode_2_freq_t, PSD_atm_mode_2, color="black", label=f"atm_PSD_mode_{mode_2}")
    plt.title(f"residual PSD mode_{mode_2}, gain {env.CL_gain_pyr}")
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
                plt.plot(PSD_residual_mode_3_freq_t_1st_RL[np.where(PSD_residual_mode_3_freq_t_1st_RL <= 250)]
                         , PSD_residual_mode_3_1st_RL[np.where(PSD_residual_mode_3_freq_t_1st_RL <= 250)], '--', label=f"PSD_mode_{mode_3}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_RL, PSD_residual_mode_3_2nd_RL, label=f"PSD_mode_{mode_3}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_RL, PSD_residual_mode_3_1st_RL, label=f"PSD_mode_{mode_3}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_3_freq_t_1st_int[np.where(PSD_residual_mode_3_freq_t_1st_int <= 250)]
                         , PSD_residual_mode_3_1st_int[np.where(PSD_residual_mode_3_freq_t_1st_int <= 250)], '--', label=f"PSD_mode_{mode_3}_1st_{label_int}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_int, PSD_residual_mode_3_2nd_int, label=f"PSD_mode_{mode_3}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_int, PSD_residual_mode_3_1st_int, label=f"PSD_mode_{mode_3}_1st_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_3_freq_t_1st_ideal[np.where(PSD_residual_mode_3_freq_t_1st_ideal <= 250)]
                         , PSD_residual_mode_3_1st_ideal[np.where(PSD_residual_mode_3_freq_t_1st_ideal <= 250)], '--', label=f"PSD_mode_{mode_3}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_ideal, PSD_residual_mode_3_2nd_ideal, label=f"PSD_mode_{mode_3}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_ideal, PSD_residual_mode_3_1st_ideal, label=f"PSD_mode_{mode_3}_1st_{label_ideal}")


    plt.plot(PSD_atm_mode_3_freq_t, PSD_atm_mode_3, color="black", label=f"atm_PSD_mode_{mode_3}")
    plt.title(f"residual PSD mode_{mode_3}, gain {env.CL_gain_pyr}")
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
                plt.plot(PSD_residual_mode_4_freq_t_1st_RL[np.where(PSD_residual_mode_4_freq_t_1st_RL <= 250)]
                         , PSD_residual_mode_4_1st_RL[np.where(PSD_residual_mode_4_freq_t_1st_RL <= 250)], '--', label=f"PSD_mode_{mode_4}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_RL, PSD_residual_mode_4_2nd_RL, label=f"PSD_mode_{mode_4}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_RL, PSD_residual_mode_4_1st_RL, label=f"PSD_mode_{mode_4}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_4_freq_t_1st_int[np.where(PSD_residual_mode_4_freq_t_1st_int <= 250)]
                         , PSD_residual_mode_4_1st_int[np.where(PSD_residual_mode_4_freq_t_1st_int <= 250)], '--', label=f"PSD_mode_{mode_4}_1st_{label_int}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_int, PSD_residual_mode_4_2nd_int, label=f"PSD_mode_{mode_4}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_int, PSD_residual_mode_4_1st_int, label=f"PSD_mode_{mode_4}_1st_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_4_freq_t_1st_ideal[np.where(PSD_residual_mode_4_freq_t_1st_ideal <= 250)]
                         , PSD_residual_mode_4_1st_ideal[np.where(PSD_residual_mode_4_freq_t_1st_ideal <= 250)], '--', label=f"PSD_mode_{mode_4}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_ideal, PSD_residual_mode_4_2nd_ideal, label=f"PSD_mode_{mode_4}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_ideal, PSD_residual_mode_4_1st_ideal, label=f"PSD_mode_{mode_4}_1st_{label_ideal}")

    plt.plot(PSD_atm_mode_4_freq_t, PSD_atm_mode_4, color="black", label=f"atm_PSD_mode_{mode_4}")
    plt.title(f"residual PSD mode_{mode_4}, gain {env.CL_gain_pyr}")
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
                plt.plot(PSD_residual_mode_5_freq_t_1st_RL[np.where(PSD_residual_mode_5_freq_t_1st_RL <= 250)]
                         , PSD_residual_mode_5_1st_RL[np.where(PSD_residual_mode_5_freq_t_1st_RL <= 250)], '--', label=f"PSD_mode_{mode_5}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_RL, PSD_residual_mode_5_2nd_RL, label=f"PSD_mode_{mode_5}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_RL, PSD_residual_mode_5_1st_RL, label=f"PSD_mode_{mode_5}_1st_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_5_freq_t_1st_int[np.where(PSD_residual_mode_5_freq_t_1st_int <= 250)]
                         , PSD_residual_mode_5_1st_int[np.where(PSD_residual_mode_5_freq_t_1st_int <= 250)], '--', label=f"PSD_mode_{mode_5}_1st_{label_int}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_int, PSD_residual_mode_5_2nd_int, label=f"PSD_mode_{mode_5}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_int, PSD_residual_mode_5_1st_int, label=f"PSD_mode_{mode_5}_1st_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_5_freq_t_1st_ideal[np.where(PSD_residual_mode_5_freq_t_1st_ideal <= 250)]
                         , PSD_residual_mode_5_1st_ideal[np.where(PSD_residual_mode_5_freq_t_1st_ideal <= 250)], '--', label=f"PSD_mode_{mode_5}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_ideal, PSD_residual_mode_5_2nd_ideal, label=f"PSD_mode_{mode_5}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_ideal, PSD_residual_mode_5_1st_ideal, label=f"PSD_mode_{mode_5}_1st_{label_ideal}")

    plt.plot(PSD_atm_mode_5_freq_t, PSD_atm_mode_5, color="black", label=f"atm_PSD_mode_{mode_5}")
    plt.title(f"residual PSD mode_{mode_5}, gain {env.CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD")


# ---------------------------------------------------temporal Error transfer function---------------------------------------------------#
if plot_tPSD:
    if RL:
        tETF_mode_1_1st_RL = PSD_residual_mode_1_1st_RL / PSD_atm_mode_1
        tETF_mode_2_1st_RL = PSD_residual_mode_2_1st_RL / PSD_atm_mode_2
        tETF_mode_3_1st_RL = PSD_residual_mode_3_1st_RL / PSD_atm_mode_3
        tETF_mode_4_1st_RL = PSD_residual_mode_4_1st_RL / PSD_atm_mode_4
        tETF_mode_5_1st_RL = PSD_residual_mode_5_1st_RL / PSD_atm_mode_5

        if stage_2:
            tETF_mode_1_2nd_RL = PSD_residual_mode_1_2nd_RL / PSD_atm_mode_1
            tETF_mode_2_2nd_RL = PSD_residual_mode_2_2nd_RL / PSD_atm_mode_2
            tETF_mode_3_2nd_RL = PSD_residual_mode_3_2nd_RL / PSD_atm_mode_3
            tETF_mode_4_2nd_RL = PSD_residual_mode_4_2nd_RL / PSD_atm_mode_4
            tETF_mode_5_2nd_RL = PSD_residual_mode_5_2nd_RL / PSD_atm_mode_5


    if integrator:
        tETF_mode_1_1st_int = PSD_residual_mode_1_1st_int / PSD_atm_mode_1
        tETF_mode_2_1st_int = PSD_residual_mode_2_1st_int / PSD_atm_mode_2
        tETF_mode_3_1st_int = PSD_residual_mode_3_1st_int / PSD_atm_mode_3
        tETF_mode_4_1st_int = PSD_residual_mode_4_1st_int / PSD_atm_mode_3
        tETF_mode_5_1st_int = PSD_residual_mode_5_1st_int / PSD_atm_mode_3

        if stage_2:
            tETF_mode_1_2nd_int = PSD_residual_mode_1_2nd_int / PSD_atm_mode_1
            tETF_mode_2_2nd_int = PSD_residual_mode_2_2nd_int / PSD_atm_mode_2
            tETF_mode_3_2nd_int = PSD_residual_mode_3_2nd_int / PSD_atm_mode_3
            tETF_mode_4_2nd_int = PSD_residual_mode_4_2nd_int / PSD_atm_mode_4
            tETF_mode_5_2nd_int = PSD_residual_mode_5_2nd_int / PSD_atm_mode_5


    if ideal:
        tETF_mode_1_1st_ideal = PSD_residual_mode_1_1st_ideal / PSD_atm_mode_1
        tETF_mode_2_1st_ideal = PSD_residual_mode_2_1st_ideal / PSD_atm_mode_2
        tETF_mode_3_1st_ideal = PSD_residual_mode_3_1st_ideal / PSD_atm_mode_3
        tETF_mode_4_1st_ideal = PSD_residual_mode_4_1st_ideal / PSD_atm_mode_4
        tETF_mode_5_1st_ideal = PSD_residual_mode_5_1st_ideal / PSD_atm_mode_5

        if stage_2:
            tETF_mode_1_2nd_ideal = PSD_residual_mode_1_2nd_ideal / PSD_atm_mode_1
            tETF_mode_2_2nd_ideal = PSD_residual_mode_2_2nd_ideal / PSD_atm_mode_2
            tETF_mode_3_2nd_ideal = PSD_residual_mode_3_2nd_ideal / PSD_atm_mode_3
            tETF_mode_4_2nd_ideal = PSD_residual_mode_4_2nd_ideal / PSD_atm_mode_4
            tETF_mode_5_2nd_ideal = PSD_residual_mode_5_2nd_ideal / PSD_atm_mode_5



    #tip
    plt.figure()
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_ideal[np.where(PSD_residual_mode_1_freq_t_1st_ideal <= 250)]
                         , tETF_mode_1_1st_ideal[np.where(PSD_residual_mode_1_freq_t_1st_ideal <= 250)], '--', label=f"ETF mode_{mode_1}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_ideal, tETF_mode_1_2nd_ideal, label=f"ETF mode_{mode_1}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_ideal, tETF_mode_1_1st_ideal, label=f"ETF mode_{mode_1}_1st_{label_ideal}")

    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_int[np.where(PSD_residual_mode_1_freq_t_1st_int <= 250)]
                         , tETF_mode_1_1st_int[np.where(PSD_residual_mode_1_freq_t_1st_int <= 250)], '--', label=f"ETF mode_{mode_1}_1st_{label_int}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_int, tETF_mode_1_2nd_int, label=f"ETF mode_{mode_1}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_int, tETF_mode_1_1st_int, label=f"ETF mode_{mode_1}_1st_{label_int}")

    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_RL[np.where(PSD_residual_mode_1_freq_t_1st_RL <= 250)]
                         , tETF_mode_1_1st_RL[np.where(PSD_residual_mode_1_freq_t_1st_RL <= 250)], '--', label=f"ETF mode_{mode_1}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_RL, tETF_mode_1_2nd_RL, label=f"ETF mode_{mode_1}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_RL, tETF_mode_1_1st_RL, label=f"ETF mode_{mode_1}_1st_{label_RL}")
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
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_2_freq_t_1st_ideal[np.where(PSD_residual_mode_2_freq_t_1st_ideal <= 250)]
                         , tETF_mode_2_1st_ideal[np.where(PSD_residual_mode_2_freq_t_1st_ideal <= 250)], '--', label=f"ETF mode_{mode_2}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_ideal, tETF_mode_2_2nd_ideal, label=f"ETF mode_{mode_2}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_ideal, tETF_mode_2_1st_ideal, label=f"ETF mode_{mode_2}_1st_{label_ideal}")

    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_2_freq_t_1st_int[np.where(PSD_residual_mode_2_freq_t_1st_int <= 250)]
                         , tETF_mode_2_1st_int[np.where(PSD_residual_mode_2_freq_t_1st_int <= 250)], '--', label=f"ETF mode_{mode_2}_1st_{label_int}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_int, tETF_mode_2_2nd_int, label=f"ETF mode_{mode_2}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_int, tETF_mode_2_1st_int, label=f"ETF mode_{mode_2}_1st_{label_int}")

    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_2_freq_t_1st_RL[np.where(PSD_residual_mode_2_freq_t_1st_RL <= 250)]
                         , tETF_mode_2_1st_RL[np.where(PSD_residual_mode_2_freq_t_1st_RL <= 250)], '--', label=f"ETF mode_{mode_2}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_RL, tETF_mode_2_2nd_RL, label=f"ETF mode_{mode_2}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_RL, tETF_mode_2_1st_RL, label=f"ETF mode_{mode_2}_1st_{label_RL}")
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
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_3_freq_t_1st_ideal[np.where(PSD_residual_mode_3_freq_t_1st_ideal <= 250)]
                         , tETF_mode_3_1st_ideal[np.where(PSD_residual_mode_3_freq_t_1st_ideal <= 250)], '--', label=f"ETF mode_{mode_3}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_ideal, tETF_mode_3_2nd_ideal, label=f"ETF mode_{mode_3}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_ideal, tETF_mode_3_1st_ideal, label=f"ETF mode_{mode_3}_1st_{label_ideal}")

    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_3_freq_t_1st_int[np.where(PSD_residual_mode_3_freq_t_1st_int <= 250)]
                         , tETF_mode_3_1st_int[np.where(PSD_residual_mode_3_freq_t_1st_int <= 250)], '--', label=f"ETF mode_{mode_3}_1st_{label_int}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_int, tETF_mode_3_2nd_int, label=f"ETF mode_{mode_3}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_int, tETF_mode_3_1st_int, label=f"ETF mode_{mode_3}_1st_{label_int}")

    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_3_freq_t_1st_RL[np.where(PSD_residual_mode_3_freq_t_1st_RL <= 250)]
                         , tETF_mode_3_1st_RL[np.where(PSD_residual_mode_3_freq_t_1st_RL <= 250)], '--', label=f"ETF mode_{mode_3}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_RL, tETF_mode_3_2nd_RL, label=f"ETF mode_{mode_3}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_RL, tETF_mode_3_1st_RL, label=f"ETF mode_{mode_3}_1st_{label_RL}")
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
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_4_freq_t_1st_ideal[np.where(PSD_residual_mode_4_freq_t_1st_ideal <= 250)]
                         , tETF_mode_4_1st_ideal[np.where(PSD_residual_mode_4_freq_t_1st_ideal <= 250)], '--', label=f"ETF mode_{mode_4}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_ideal, tETF_mode_4_2nd_ideal, label=f"ETF mode_{mode_4}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_ideal, tETF_mode_4_1st_ideal, label=f"ETF mode_{mode_4}_1st_{label_ideal}")

    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_4_freq_t_1st_int[np.where(PSD_residual_mode_4_freq_t_1st_int <= 250)]
                         , tETF_mode_4_1st_int[np.where(PSD_residual_mode_4_freq_t_1st_int <= 250)], '--', label=f"ETF mode_{mode_4}_1st_{label_int}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_int, tETF_mode_4_2nd_int, label=f"ETF mode_{mode_4}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_int, tETF_mode_4_1st_int, label=f"ETF mode_{mode_4}_1st_{label_int}")

    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_4_freq_t_1st_RL[np.where(PSD_residual_mode_4_freq_t_1st_RL <= 250)]
                         , tETF_mode_4_1st_RL[np.where(PSD_residual_mode_4_freq_t_1st_RL <= 250)], '--', label=f"ETF mode_{mode_4}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_RL, tETF_mode_4_2nd_RL, label=f"ETF mode_{mode_4}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_RL, tETF_mode_4_1st_RL, label=f"ETF mode_{mode_4}_1st_{label_RL}")
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
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_5_freq_t_1st_ideal[np.where(PSD_residual_mode_5_freq_t_1st_ideal <= 250)]
                         , tETF_mode_5_1st_ideal[np.where(PSD_residual_mode_5_freq_t_1st_ideal <= 250)], '--', label=f"ETF mode_{mode_5}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_ideal, tETF_mode_5_2nd_ideal, label=f"ETF mode_{mode_5}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_ideal, tETF_mode_5_1st_ideal, label=f"ETF mode_{mode_5}_1st_{label_ideal}")

    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_5_freq_t_1st_int[np.where(PSD_residual_mode_5_freq_t_1st_int <= 250)]
                         , tETF_mode_5_1st_int[np.where(PSD_residual_mode_5_freq_t_1st_int <= 250)], '--', label=f"ETF mode_{mode_5}_1st_{label_int}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_int, tETF_mode_5_2nd_int, label=f"ETF mode_{mode_5}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_int, tETF_mode_5_1st_int, label=f"ETF mode_{mode_5}_1st_{label_int}")

    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_5_freq_t_1st_RL[np.where(PSD_residual_mode_5_freq_t_1st_RL <= 250)]
                         , tETF_mode_5_1st_RL[np.where(PSD_residual_mode_5_freq_t_1st_RL <= 250)], '--', label=f"ETF mode_{mode_5}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_RL, tETF_mode_5_2nd_RL, label=f"ETF mode_{mode_5}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_RL, tETF_mode_5_1st_RL, label=f"ETF mode_{mode_5}_1st_{label_RL}")
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





























