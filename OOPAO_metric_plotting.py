"""
*when doing the temporal PSD you should also implement some kind of strehl condition
	problem is that then the temporal curve can get discontinuities
	the most straightforward solution seems to be just cutting out a continuous section of the timeseries where the strehl ratio is above your threshold


"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.DeformableMirror import DeformableMirror

from po4ao_edw.OOPAO_environment_PWFS import OOPAO_environment_PWFS
from po4ao_edw.OOPAO_environment_ZWFS import OOPAO_environment_ZWFS


env = OOPAO_environment_ZWFS()
#env = OOPAO_environment_PWFS()

DM_fake = DeformableMirror(telescope = env.TEL,
                           nSubap    = 40,
                           mechCoupling = env.MECH_COUPLING)

M2C_KL_fake = compute_KL_basis(tel = env.TEL, atm = env.ATM, dm = DM_fake)
M2C_fake    = M2C_KL_fake[:, :400]
modes_fake  = DM_fake.modes @ M2C_fake
modes_fake_inv = np.linalg.pinv(np.squeeze(modes_fake[env.TEL.pupilLogical, :]))


#TODO don't forget to also change your atmosphere parameters when loading files (atm_OPD_array)
#TODO could you in fact rewrite this so unneccesary stuff is not loaded?
RL         = True
integrator = True
ideal      = True
stage_2    = True
directory_name_RL  = 'vZWFS_1st_2nd_noise_03_wooftw_seed_chang_gaussian_noise'
directory_name_int = 'vZWFS_metrics_integrator_2'
directory_name_ideal = 'vZWFS_1st_2nd_noise_03_wooftw_seed_chang_dyn_mask_2'
label_RL = "gaussian_noise"
label_int = "integrator_2"
label_ideal = "dyn_mask"
mask_thresh = 0.5



#RL
residual_error_RL = np.load(f"temp_save_dir/{directory_name_RL}/residual_error.npy")
strehl_array_1st_RL = np.load(f"temp_save_dir/{directory_name_RL}/strehl_array_1st.npy")

if stage_2:
    strehl_array_2nd_RL = np.load(f"temp_save_dir/{directory_name_RL}/strehl_array_2nd.npy")

#tel_psf_array_RL = np.load(f"temp_save_dir/{directory_name_RL}/tel_psf_array.npy")
residual_OPD_array_RL = np.load(f"temp_save_dir/{directory_name_RL}/residual_OPD_array.npy") #for use in spatial PSD/KL modes var and correlation
#you can later do the spatial PSD when you save the required atm parameter
atm_OPD_array_RL = np.load(f"temp_save_dir/{directory_name_RL}/atm_OPD_array.npy") #not sure where I would use it
total_err_array_RL = np.load(f"temp_save_dir/{directory_name_RL}/total_err_array.npy") #not useful for now
dynamics_loss = np.load(f"temp_save_dir/{directory_name_RL}/dynamics_loss.npy") #not useful for now
policy_loss = np.load(f"temp_save_dir/{directory_name_RL}/policy_loss.npy") #not useful for now

#YOU SHOULD ALSO CHECK IF YOU ONLY HAVE AN EPISODE OF DATA IN ALL OF THESE BECAUSE I DON4T REMEMBER
frequency_RL = np.load(f"temp_save_dir/{directory_name_RL}/frequency.npy")
time_plot_RL = np.load(f"temp_save_dir/{directory_name_RL}/time_array.npy")






#integrator
residual_error_int = np.load(f"temp_save_dir/{directory_name_int}/residual_error.npy")
strehl_array_1st_int = np.load(f"temp_save_dir/{directory_name_int}/strehl_array_1st.npy")


if stage_2:
    strehl_array_2nd_int = np.load(f"temp_save_dir/{directory_name_int}/strehl_array_2nd.npy")

#tel_psf_array_int = np.load(f"temp_save_dir/{directory_name_int}/tel_psf_array.npy")
residual_OPD_array_int = np.load(f"temp_save_dir/{directory_name_int}/residual_OPD_array.npy") #for use in spatial PSD/KL modes var and correlation
#you can later do the spatial PSD when you save the required atm parameter
atm_OPD_array_int = np.load(f"temp_save_dir/{directory_name_int}/atm_OPD_array.npy") #not sure where I would use it
total_err_array_int = np.load(f"temp_save_dir/{directory_name_int}/total_err_array.npy") #not useful for now

#YOU SHOULD ALSO CHECK IF YOU ONLY HAVE AN EPISODE OF DATA IN ALL OF THESE BECAUSE I DON4T REMEMBER
frequency_int = np.load(f"temp_save_dir/{directory_name_int}/frequency.npy")
time_plot_int = np.load(f"temp_save_dir/{directory_name_int}/time_array.npy")



#ideal
residual_error_ideal = np.load(f"temp_save_dir/{directory_name_ideal}/residual_error.npy")
strehl_array_1st_ideal = np.load(f"temp_save_dir/{directory_name_ideal}/strehl_array_1st.npy")

if stage_2:
    strehl_array_2nd_ideal = np.load(f"temp_save_dir/{directory_name_ideal}/strehl_array_2nd.npy")

#tel_psf_array_ideal = np.load(f"temp_save_dir/{directory_name_ideal}/tel_psf_array.npy")
residual_OPD_array_ideal = np.load(f"temp_save_dir/{directory_name_ideal}/residual_OPD_array.npy") #for use in spatial PSD/KL modes var and correlation
#you can later do the spatial PSD when you save the required atm parameter
atm_OPD_array_ideal = np.load(f"temp_save_dir/{directory_name_ideal}/atm_OPD_array.npy") #not sure where I would use it
total_err_array_ideal = np.load(f"temp_save_dir/{directory_name_ideal}/total_err_array.npy") #not useful for now

#YOU SHOULD ALSO CHECK IF YOU ONLY HAVE AN EPISODE OF DATA IN ALL OF THESE BECAUSE I DON4T REMEMBER
frequency_ideal = np.load(f"temp_save_dir/{directory_name_ideal}/frequency.npy")
time_plot_ideal = np.load(f"temp_save_dir/{directory_name_ideal}/time_array.npy")

atm_OPD_array = atm_OPD_array_int
time_plot_atm = time_plot_int
f_samp_atm = frequency_int

# ---------------------------------------------------Loss---------------------------------------------------#
if RL:
    plt.figure()
    plt.subplot(121)
    plt.title("dynamics_loss warmup")
    plt.plot(dynamics_loss)
    plt.grid(True)
    plt.yscale('log')
    plt.subplot(122)
    plt.title("policy_loss warmup")
    plt.grid(True)
    plt.plot(policy_loss)
    plt.yscale('log')



# ---------------------------------------------------Strehl---------------------------------------------------#
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


plt.figure()
if RL:          plt.plot(time_plot_RL, strehl_array_1st_RL, label=f"strehl_1st_{label_RL}")
if integrator:  plt.plot(time_plot_int, strehl_array_1st_int, label=f"strehl_1st_{label_int}")
#if ideal:       plt.plot(time_plot_ideal, strehl_array_1st_ideal, label=f"strehl_1st_{label_ideal}")

if not stage_2:
    if RL: plt.plot(time_plot_RL, sr_running_1st_RL, label=f"running_strehl_1st_{label_RL}")
if stage_2:
    if RL: plt.plot(time_plot_RL, strehl_array_2nd_RL, color = '#003f5c', label=f"strehl_2nd_{label_RL}")
    if RL: plt.plot(time_plot_RL, sr_running_2nd_RL, color = 'red', label=f"running_strehl_2nd_{label_RL}")

    if integrator: plt.plot(time_plot_int, strehl_array_2nd_int, color = '#ffa600', label=f"strehl_2nd_{label_int}")
    if ideal: plt.plot(time_plot_ideal, strehl_array_2nd_ideal, label=f"strehl_2nd_{label_ideal}")


plt.title("Strehl ratio")
plt.xlabel("time s")
plt.ylim(bottom=(sr_mean_RL - 0.5))
plt.grid(True)
plt.legend()


# ---------------------------------------------------Zernike/KL decomposition---------------------------------------------------#
#try subtracting modes from the atmosphere and see what you get
#check how the atmosphere is updated, how is the atmosphere extrapolated
#delete the wind or smth
#RL
if stage_2:
    strehl_mask_RL = strehl_array_2nd_RL
else:
    strehl_mask_RL = strehl_array_1st_RL

coefficient_matrix_res_list_RL        = []
coefficient_matrix_res_list_RL_full   = []
residual_OPD_array_05_RL = residual_OPD_array_RL[strehl_mask_RL > mask_thresh]
for i in range(len(residual_OPD_array_05_RL)):
    final_residual_phase_RL = 2 * np.pi * residual_OPD_array_05_RL[i] / env.SRC.wavelength
    coefficient_matrix_res_RL = modes_fake_inv @ final_residual_phase_RL[np.where(env.TEL.pupil > 0)]
    coefficient_matrix_res_list_RL.append(coefficient_matrix_res_RL)

for i in range(len(residual_OPD_array_RL)):
    final_residual_phase_RL_full = 2 * np.pi * residual_OPD_array_RL[i] / env.SRC.wavelength
    coefficient_matrix_res_RL_full = modes_fake_inv @ final_residual_phase_RL_full[np.where(env.TEL.pupil > 0)]
    coefficient_matrix_res_list_RL_full.append(coefficient_matrix_res_RL_full)

coefficient_matrix_res_var_RL       = np.var(np.asarray(coefficient_matrix_res_list_RL), axis = 0)


#integrator
if stage_2:
    strehl_mask_int = strehl_array_2nd_int
else:
    strehl_mask_int = strehl_array_1st_int
coefficient_matrix_res_list_int        = []
coefficient_matrix_res_list_int_full   = []
residual_OPD_array_05_int = residual_OPD_array_int[strehl_mask_int > mask_thresh]
for i in range(len(residual_OPD_array_05_int)):
    final_residual_phase_int = 2 * np.pi * residual_OPD_array_05_int[i] / env.SRC.wavelength
    coefficient_matrix_res_int = modes_fake_inv @ final_residual_phase_int[np.where(env.TEL.pupil > 0)]
    coefficient_matrix_res_list_int.append(coefficient_matrix_res_int)

for i in range(len(residual_OPD_array_int)):
    final_residual_phase_int_full = 2 * np.pi * residual_OPD_array_int[i] / env.SRC.wavelength
    coefficient_matrix_res_int_full = modes_fake_inv @ final_residual_phase_int_full[np.where(env.TEL.pupil > 0)]
    coefficient_matrix_res_list_int_full.append(coefficient_matrix_res_int_full)

coefficient_matrix_res_var_int        = np.var(np.asarray(coefficient_matrix_res_list_int), axis = 0)


#ideal
if stage_2:
    strehl_mask_ideal = strehl_array_2nd_ideal
else:
    strehl_mask_ideal = strehl_array_1st_ideal
coefficient_matrix_res_list_ideal        = []
coefficient_matrix_res_list_ideal_full   = []
residual_OPD_array_05_ideal = residual_OPD_array_ideal[strehl_mask_ideal > mask_thresh]
for i in range(len(residual_OPD_array_05_ideal)):
    final_residual_phase_ideal = 2 * np.pi * residual_OPD_array_05_ideal[i] / env.SRC.wavelength
    coefficient_matrix_res_ideal = modes_fake_inv @ final_residual_phase_ideal[np.where(env.TEL.pupil > 0)]
    coefficient_matrix_res_list_ideal.append(coefficient_matrix_res_ideal)

for i in range(len(residual_OPD_array_ideal)):
    final_residual_phase_ideal_full = 2 * np.pi * residual_OPD_array_ideal[i] / env.SRC.wavelength
    coefficient_matrix_res_ideal_full = modes_fake_inv @ final_residual_phase_ideal_full[np.where(env.TEL.pupil > 0)]
    coefficient_matrix_res_list_ideal_full.append(coefficient_matrix_res_ideal_full)
coefficient_matrix_res_var_ideal        = np.var(np.asarray(coefficient_matrix_res_list_ideal), axis = 0)



#atmosphere
temp_atm_coef_list = []
for i in range(len(atm_OPD_array)):
    atmosphere_phase = 2 * np.pi * atm_OPD_array[i] / env.SRC.wavelength
    coefficient_matrix_atmosphere = modes_fake_inv @ atmosphere_phase[np.where(env.TEL.pupil > 0)]
    temp_atm_coef_list.append(coefficient_matrix_atmosphere)

coefficient_matrix_atm_var = np.var(np.asarray(temp_atm_coef_list), axis = 0)
#coefficient_matrix_atm_var = np.load('./temp_save_dir/atm_KL_mode_coef_var.npy')




plt.figure()
if RL:         plt.plot(coefficient_matrix_res_var_RL, color="red", label=f"KL coeffs for residual phase_{label_RL}")
if integrator: plt.plot(coefficient_matrix_res_var_int, color="blue", label=f"KL coeffs for residual phase_{label_int}")
if ideal:      plt.plot(coefficient_matrix_res_var_ideal, color="green", label=f"KL coeffs for residual phase_{label_ideal}")
plt.title(f"KL coefficients for corrected vs atmospher phase")
plt.plot(coefficient_matrix_atm_var, color="black",
         label=f"KL coeffs for atmospheric phase")
plt.yscale("log")
plt.xscale("log")
plt.tight_layout()
plt.grid(True)
plt.legend()



# ---------------------------------------------------Temporal PSD---------------------------------------------------#
# temporal PSD calculation from the std
f_samp_RL = frequency_RL
f_samp_int = frequency_int
f_samp_ideal = frequency_ideal

def welch_method_scipy(data, fs, nperseg=256):
    frequencies, psd = signal.welch(
        data,
        fs=fs,
        window='hann',  #windowing
        nperseg=nperseg,
        scaling='density'
    )
    return frequencies, psd

coefficient_matrix_res_list_RL_full = np.asarray(coefficient_matrix_res_list_RL_full)
coefficient_matrix_res_list_int_full = np.asarray(coefficient_matrix_res_list_int_full)
coefficient_matrix_res_list_ideal_full = np.asarray(coefficient_matrix_res_list_ideal_full)
residual_tip_curve_RL_full    = coefficient_matrix_res_list_RL_full[:, 0]
residual_tip_curve_int_full   = coefficient_matrix_res_list_int_full[:, 0]
residual_tip_curve_ideal_full = coefficient_matrix_res_list_ideal_full[:, 0]


coefficient_matrix_res_list_RL = np.asarray(coefficient_matrix_res_list_RL)
residual_tip_curve_RL  = coefficient_matrix_res_list_RL[:, 0]
residual_tilt_curve_RL = coefficient_matrix_res_list_RL[:, 1]
residual_100_curve_RL  = coefficient_matrix_res_list_RL[:, 100]
residual_200_curve_RL  = coefficient_matrix_res_list_RL[:, 200]


coefficient_matrix_res_list_int = np.asarray(coefficient_matrix_res_list_int)
residual_tip_curve_int  = coefficient_matrix_res_list_int[:, 0]
residual_tilt_curve_int = coefficient_matrix_res_list_int[:, 1]
residual_100_curve_int  = coefficient_matrix_res_list_int[:, 100]
residual_200_curve_int  = coefficient_matrix_res_list_int[:, 200]


coefficient_matrix_res_list_ideal = np.asarray(coefficient_matrix_res_list_ideal)
residual_tip_curve_ideal  = coefficient_matrix_res_list_ideal[:, 0]
residual_tilt_curve_ideal = coefficient_matrix_res_list_ideal[:, 1]
residual_100_curve_ideal  = coefficient_matrix_res_list_ideal[:, 100]
residual_200_curve_ideal  = coefficient_matrix_res_list_ideal[:, 200]


temp_atm_coef_array = np.asarray(temp_atm_coef_list)
atm_tip_curve  = temp_atm_coef_array[:, 0]
atm_tilt_curve = temp_atm_coef_array[:, 1]
atm_100_curve  = temp_atm_coef_array[:, 100]
atm_200_curve  = temp_atm_coef_array[:, 200]

# tip
PSD_residual_tip_freq_t_RL, PSD_residual_tip_RL = welch_method_scipy(residual_tip_curve_RL, f_samp_RL)
PSD_residual_tip_freq_t_int, PSD_residual_tip_int = welch_method_scipy(residual_tip_curve_int, f_samp_int)
PSD_residual_tip_freq_t_ideal, PSD_residual_tip_ideal = welch_method_scipy(residual_tip_curve_ideal, f_samp_ideal)
PSD_atm_tip_freq_t, PSD_atm_tip = welch_method_scipy(atm_tip_curve, f_samp_atm)

# tilt
PSD_residual_tilt_freq_t_RL, PSD_residual_tilt_RL = welch_method_scipy(residual_tilt_curve_RL, f_samp_RL)
PSD_residual_tilt_freq_t_int, PSD_residual_tilt_int = welch_method_scipy(residual_tilt_curve_int, f_samp_int)
PSD_residual_tilt_freq_t_ideal, PSD_residual_tilt_ideal = welch_method_scipy(residual_tilt_curve_ideal, f_samp_ideal)
PSD_atm_tilt_freq_t, PSD_atm_tilt = welch_method_scipy(atm_tilt_curve, f_samp_atm)

# modes 100 and 200
PSD_residual_100_freq_t_RL, PSD_residual_100_RL = welch_method_scipy(residual_100_curve_RL, f_samp_RL)
PSD_residual_100_freq_t_int, PSD_residual_100_int = welch_method_scipy(residual_100_curve_int, f_samp_int)
PSD_residual_100_freq_t_ideal, PSD_residual_100_ideal = welch_method_scipy(residual_100_curve_ideal, f_samp_ideal)

PSD_residual_200_freq_t_RL, PSD_residual_200_RL = welch_method_scipy(residual_200_curve_RL, f_samp_RL)
PSD_residual_200_freq_t_int, PSD_residual_200_int = welch_method_scipy(residual_200_curve_int, f_samp_int)
PSD_residual_200_freq_t_ideal, PSD_residual_200_ideal = welch_method_scipy(residual_200_curve_ideal, f_samp_ideal)

PSD_atm_100_freq_t, PSD_atm_100 = welch_method_scipy(atm_100_curve, f_samp_atm)
PSD_atm_200_freq_t, PSD_atm_200 = welch_method_scipy(atm_200_curve, f_samp_atm)

plt.figure()
if RL:         plt.plot(PSD_residual_tip_freq_t_RL, PSD_residual_tip_RL, label=f"residual_PSD_tip_{label_RL}")
if integrator: plt.plot(PSD_residual_tip_freq_t_int, PSD_residual_tip_int, label=f"residual_PSD_tip_{label_int}")
if ideal:      plt.plot(PSD_residual_tip_freq_t_ideal, PSD_residual_tip_ideal, label=f"residual_PSD_tip_{label_ideal}")
plt.plot(PSD_atm_tip_freq_t, PSD_atm_tip, label="atm_PSD_tip")
plt.title(f"residual PSD tip, gain {env.CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True)
plt.legend()
plt.ylabel("PSD")

plt.figure()
if RL:         plt.plot(time_plot_RL, residual_tip_curve_RL_full, label=f"residual_tip_curve_{label_RL}")
if integrator: plt.plot(time_plot_int, residual_tip_curve_int_full, label=f"residual_tip_curve_{label_int}")
if ideal:      plt.plot(time_plot_ideal, residual_tip_curve_ideal_full, label=f"residual_tip_curve_{label_ideal}")
plt.plot(time_plot_atm, atm_tip_curve, label="atm_tip_curve")
plt.title(f"residual/atm timeseries tip, gain {env.CL_gain_pyr}")
plt.xlabel("time (s)")
plt.grid(True)
plt.legend()
plt.ylabel("residual tip")

plt.figure()
if RL:         plt.plot(PSD_residual_tilt_freq_t_RL, PSD_residual_tilt_RL, label=f"residual_PSD_tilt_{label_RL}")
if integrator: plt.plot(PSD_residual_tilt_freq_t_int, PSD_residual_tilt_int, label=f"residual_PSD_tilt_{label_int}")
if ideal:      plt.plot(PSD_residual_tilt_freq_t_ideal, PSD_residual_tilt_ideal, label=f"residual_PSD_tilt_{label_ideal}")
plt.plot(PSD_atm_tilt_freq_t, PSD_atm_tilt, label="atm_PSD_tilt")
plt.title(f"residual PSD tilt, gain {env.CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True)
plt.legend()
plt.ylabel("PSD")

plt.figure()
if RL:         plt.plot(PSD_residual_100_freq_t_RL, PSD_residual_100_RL, label=f"PSD_residual_100_{label_RL}")
if integrator: plt.plot(PSD_residual_100_freq_t_int, PSD_residual_100_int, label=f"PSD_residual_100_{label_int}")
if ideal:      plt.plot(PSD_residual_100_freq_t_ideal, PSD_residual_100_ideal, label=f"PSD_residual_100_{label_ideal}")
plt.plot(PSD_atm_100_freq_t, PSD_atm_100, label="atm_PSD_100")
plt.title(f"residual PSD 100, gain {env.CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True)
plt.legend()
plt.ylabel("PSD")

plt.figure()
if RL:         plt.plot(PSD_residual_200_freq_t_RL, PSD_residual_200_RL, label=f"PSD_residual_200_{label_RL}")
if integrator: plt.plot(PSD_residual_200_freq_t_int, PSD_residual_200_int, label=f"PSD_residual_200_{label_int}")
if ideal:      plt.plot(PSD_residual_200_freq_t_ideal, PSD_residual_200_ideal, label=f"PSD_residual_200_{label_ideal}")
plt.plot(PSD_atm_200_freq_t, PSD_atm_200, label="atm_PSD_200")
plt.title(f"residual PSD 200, gain {env.CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True)
plt.legend()
plt.ylabel("PSD")

# ---------------------------------------------------temporal Error transfer function---------------------------------------------------#
tETF_tip_RL = PSD_residual_tip_RL / PSD_atm_tip
tETF_tilt_RL = PSD_residual_tilt_RL / PSD_atm_tilt
tETF_100_RL = PSD_residual_100_RL / PSD_atm_100
tETF_200_RL = PSD_residual_200_RL / PSD_atm_200

tETF_tip_int = PSD_residual_tip_int / PSD_atm_tip
tETF_tilt_int = PSD_residual_tilt_int / PSD_atm_tilt
tETF_100_int = PSD_residual_100_int / PSD_atm_100
tETF_200_int = PSD_residual_200_int / PSD_atm_200

tETF_tip_ideal = PSD_residual_tip_ideal / PSD_atm_tip
tETF_tilt_ideal = PSD_residual_tilt_ideal / PSD_atm_tilt
tETF_100_ideal = PSD_residual_100_ideal / PSD_atm_100
tETF_200_ideal = PSD_residual_200_ideal / PSD_atm_200

plt.figure()
if RL:         plt.plot(PSD_residual_tip_freq_t_RL, tETF_tip_RL, label=f"ETF tip_{label_RL}")
if integrator: plt.plot(PSD_residual_tip_freq_t_int, tETF_tip_int, label=f"ETF tip_{label_int}")
if ideal:      plt.plot(PSD_residual_tip_freq_t_ideal, tETF_tip_ideal, label=f"ETF tip_{label_ideal}")
#plt.plot(PSD_residual_tilt_freq_t_RL, tETF_tilt_RL, label="ETF tilt_RL")
#plt.plot(PSD_residual_100_freq_t_RL, tETF_100_RL, label="ETF 100_RL")
#plt.plot(PSD_residual_100_freq_t_int, tETF_100_int, label="ETF 100_int")
#plt.plot(PSD_residual_100_freq_t_ideal, tETF_100_ideal, label="ETF 10_ideal")
#plt.plot(PSD_residual_200_freq_t_RL, tETF_200_RL, label="ETF 200_RL")
plt.title("temporal error transfer functions")
plt.ylabel("ETF")
plt.xlabel("frequency Hz")
plt.xscale("log")
plt.yscale("log")
plt.xlim(right=np.max(PSD_residual_tip_freq_t_RL))
plt.grid(True)
plt.legend()



plt.show()





























