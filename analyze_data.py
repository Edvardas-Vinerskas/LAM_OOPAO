
import time
import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import convolve
from numpy.fft import fft, ifft


# def richardson_lucy(observed, psf, iterations=100):
#     estimate = np.full_like(observed, 0.5)
#     psf_mirror = psf[::-1]
#     for _ in range(iterations):
#         relative_blur = observed / (convolve(estimate, psf, mode="same") + 1e-10)
#         estimate *= convolve(relative_blur, psf_mirror, mode="same")
#     return estimate


def door_function(length, width):
    door = np.zeros(length)
    center = length // 2
    half_width = width // 2
    door[center - half_width : center + half_width] = 1.0
    return door / np.sum(door)  # Normalize


def deconvolve_fft(observed, kernel, epsilon=1e-3):
    O = fft(observed)
    K = fft(kernel)
    K_conj = np.conj(K)
    deconv = O * K_conj / (K * K_conj + epsilon)
    return np.real(ifft(deconv))


# Load data
RL_telemetry_2nd = np.load('bench_sky_04_15/onsky_arcturus_1st200_2nd400_v6_20260416-005743/2026-04-16T01_03_00_telemetry_2nd_data_RL_iter60.npy', allow_pickle = True)
int_telemetry_2nd = np.load('bench_sky_04_15/onsky_arcturus_1st200_2nd400_v6_20260416-005743/2026-04-16T01_07_14_telemetry_2nd_data_pythint.npy', allow_pickle = True) #2026-04-16T01_08_54_telemetry_2nd_data_dao      #2026-04-16T01_07_14_telemetry_2nd_data_pythint
time_diff = np.load('bench_sky_04_15/onsky_arcturus_1st200_2nd400_v6_20260416-005743/timdeff_list.npy')
time_diff = time_diff.reshape(120000)


ii = 0
timestamp_wfs   = RL_telemetry_2nd.item()['timeStampcredCube']           #np.load("timestamps/WFS_timestamps.npy")  # Timestamps of WFS data
timestamp_dm    = RL_telemetry_2nd.item()['timeStampDmCube']       #np.load("timestamps/DM_timestamps.npy")  # Reference timestamps

valid_pixels = RL_telemetry_2nd.item()['validPixels'].copy()
valid_pixels[valid_pixels < 1e-5] = 0
valid_pixels[valid_pixels >= 1e-5] = 1

frame_wfs       = (RL_telemetry_2nd.item()['CRED2Cube'] - RL_telemetry_2nd.item()['credDark']) * valid_pixels[np.newaxis, :, :]      #np.load("images/WFS_frames.npy")  # Associated values
cred_dark       = RL_telemetry_2nd.item()['credDark']

time_plot = np.arange(0, 30000)



timestamp_wfs = np.asarray(timestamp_wfs)
timestamp_wfs = timestamp_wfs.astype('datetime64[us]')
timestamp_wfs = timestamp_wfs.astype(np.int64)# / np.timedelta64(1, 'us')


timestamp_dm = np.asarray(timestamp_dm)
timestamp_dm = timestamp_dm.astype('datetime64[us]')
timestamp_dm = timestamp_dm.astype(np.int64)# / np.timedelta64(1, 'us')
'''
#so you calculate the difference between the entries in wfs and dm timestamps and then also wfs - dm
#1st will give you the frequency of the loop
#2nd one will give you the delay between wfs and dm
plt.figure()
plt.scatter(time_plot[ii * 1000 :(ii +1) * 1000], diffs_wfs[ii * 1000 :(ii +1) * 1000], label = 'wfs')
plt.scatter(time_plot[ii * 1000 :(ii +1) * 1000], diffs_dm[ii * 1000 :(ii +1) * 1000], label = 'dm')
#plt.hist(diffs_dm - diffs_wfs, bins=5000, color = 'black')
plt.legend()
plt.show()'''

'''#for frame skipping
cred_cube = RL_telemetry_2nd.item()['CRED2Cube']
print(cred_cube[:, 0, 0])
plt.figure()
#plt.plot(np.diff(cred_cube[:, 0, 0]))
plt.plot(time_diff)
plt.show()'''



# Keep 1 over 2 frames to sample up and down on DM commands
timestamp_dm = timestamp_dm[::2] 

# Step 1: For each timestamp_wfs, find the index j of timestamp_dm such that
# timestamp_dm[j] <= timestamp_wfs[i] < timestamp_dm[j+1]
# Use searchsorted to find the right edge
indices = np.searchsorted(timestamp_dm, timestamp_wfs, side="right") - 1


# Make sure indices are within bounds
valid = (indices >= 0) & (indices < len(timestamp_dm) - 1)


# Filter valid values
valid_indices = indices[valid]
valid_timestamps_wfs = timestamp_wfs[valid]
valid_frame_wfs = frame_wfs[valid, ...]

# Step 2: Compute wrapped time (offset from previous timestamp_dm)
wrapped_time = valid_timestamps_wfs - timestamp_dm[valid_indices]
print(valid_timestamps_wfs)
print(timestamp_dm[valid_indices])
print(wrapped_time[:10])
#indices are where the wfs timestamps should go in between timestamp_dm
#valid is a mask that just cuts off indices that are outside timestamp_dm
#valid_indices is thus just indices where timestamp_wfs is within timestamp_dm
 #timestamp_dm[valid_indices] replicates all of the valid timestamp_dm so that the shapes between valid_timestamps_wfs and timestamp_dm are the same



# Sort array
sorted_time = wrapped_time.argsort()



rms = np.sum(
    (valid_frame_wfs[sorted_time] - valid_frame_wfs[sorted_time][0, ...]) ** 2,
    axis=(1, 2),
)


# from scipy.interpolate import UnivariateSpline

#sorts the time differences between wfs and dm
x = wrapped_time[sorted_time] * 1e-3
y = rms


#only leaves the unique time differences
_, unique_indices = np.unique(x, return_index=True)
x_unique = x[unique_indices]
rms_unique = rms[unique_indices]
print(x_unique.shape[0])
print(x_unique.shape[0] // 2)
err

# spline = UnivariateSpline(
#     x_unique[: x_unique.shape[0] // 2],
#     rms_unique[: x_unique.shape[0] // 2],
#     s=50 * np.var(rms_unique[: x_unique.shape[0] // 2]),
# )

#t = np.linspace(0, 20, 1000)
# fitted = spline(t)

# Step 3: Plot
plt.figure(figsize=(8, 5))
plt.scatter(
    x_unique[: x_unique.shape[0] // 2],
    rms_unique[: x_unique.shape[0] // 2],
    s=10,
    alpha=0.7,
)
# plt.plot(t,fitted,color='orange')
plt.xlabel("Relative Time Since DM command - milliseconds")
plt.ylabel("RMS between 2 successive WFS image")
plt.title(
    "Latency plot - percentage of valid measurements = "
    + str(np.round(100 * np.sum(valid) / valid.shape[0], 3))
)
plt.grid(True)
plt.ylim([0, np.max(rms)])
plt.show()


# Save dtaa
#np.save("data_processed/timestamp.npy", x_unique[: x_unique.shape[0] // 2])
#np.save("data_processed/rms.npy", rms_unique[: x_unique.shape[0] // 2])


"""
door = door_function(len(t), width=int(1/(t[1]-t[0])))
recovered = richardson_lucy(fitted, door, iterations=20)

plt.figure(figsize=(8, 5))
#plt.scatter(t,door, s=10, alpha=0.7)
plt.plot(t,fitted,color='orange')
plt.plot(t,recovered ,color='red')
plt.xlabel("Relative Time Since DM command - milliseconds")
plt.ylabel("RMS between 2 successive WFS image")
plt.title('Latency plot - percentage of valid measurements = '+str(np.round(100*np.sum(valid)/valid.shape[0],3)))
plt.grid(True)
#plt.ylim([0,np.max(rms)])
plt.show(block=False)

"""
