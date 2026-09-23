import OOPAO
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.signal import correlate, correlation_lags
from OOPAO.Zernike import Zernike
from OOPAO.Telescope import Telescope
import torch
"""
This code was used for seeing how to get the lag between wfs and deformable mirror through covariance
it was not very successful

"""



#note that the modeCube and dmCmdCube indices are aligned, i.e. DM(t) = DM(t-1) - res(t), so from the cross correlation 
#so from the cross correlation you should directly read off the delay which should be smth like 1.32 frames (1.6/5 = 0.32 + 1 frame)
#but it is not 1.32 = 6.6ms delay :( and instead the peak is at smth like 0.035 or 7 frames after! very interesting
#   these 7 (or 6, 8) frames seem to be consistent across modes interestingly enough
#the data cross correlation at least has the same shape as the dummy data cross correlation
#the problem is probably the fact that the correlation peak depends on the broadness of the DM command autocorrelation 
#   as the autocorrelation is effectively what is happening if you look at the formula res = atm - DM
#   the broadness of the DM autocorrelation is several frames large and using the dummy dataset
    #   it is also broad because the DM signal lacks high frequency content obviously, so funny innit
#   you can accurately see a peak up to 3 frames, and in my data set it is definitely less than 2
#   I am also not sure how to reconstruct it with 1.32 frame delay, which is what it should be.
#   So the best thing is to just use the 1.32 frame delay somehow... and stop bothering with the cross-correlation



path = "bench_sky_04_15/onsky_arcturus_1st200_2nd400_v10_20260416-020659"
filename = "2026-04-16T02_11_05_telemetry_2nd_data_RLiter70"
open_loop = "2026-04-16T02_19_39_telemetry_data_OL" #for later comparison

tester = np.load(f"{path}/{filename}.npy", allow_pickle=True)
print(tester.item().keys())


#TODO here I tried to redo the figure from Jalo's paper figure 9
#for dao it works, for the python stuff there is slight misalignment that is weirder and RL is weird, not sure why
#maybe you should do a bunch of different residuals from total dm commands (like 10 difference maybe) and see which one is the best
#RL is very weird when comparing slavedreconsCube against total DM commands
#you might have done it wrong as well, a - are actions applied to the DM, so PO4AO output,
# o are the observations obtained from slavedreconsCube, I think I did the inverse of this
n = 20060
d = 50
res_modes = tester.item()['slavedreconsCube'].squeeze()[n:n + d] #observations
dm_commands = tester.item()['dmCmdCube'].squeeze()[n:n + d] #actions
dm_commands_t = dm_commands[2:]
dm_commands_t_1 = dm_commands[1:-1]
dm_commands_t_2 = dm_commands[:-2]
residual_dm = -((0.95 * dm_commands_t_1) + dm_commands_t)/0.5
residual_dm_2 = -((0.95 * dm_commands_t_2) + dm_commands_t_1)/0.5



#so here you used the reconstructed commands
#but you need the output of the PO4AO thing, not the reconstructed commands lmao
#you get that by taking the total DM commands and subtracting the previous one from the current one
n = 3000
d = 1

actions = (res_modes[0:-2] - np.mean(res_modes[0:-2], axis = 0)).T
residual_dm = (residual_dm - np.mean(residual_dm, axis = 0)).T
residual_dm_2 = (residual_dm_2 - np.mean(residual_dm_2, axis = 0)).T

applied_commands = np.concatenate((residual_dm, residual_dm_2), axis = 0)

print(actions.shape)
print(residual_dm.shape)
print(residual_dm_2.shape)
print(applied_commands.shape)


corr = correlate(actions[50] - actions[50].mean(), residual_dm[50] - residual_dm[50].mean(), mode='full', method='fft')
# corr_shifted = correlate(R - R.mean(), D - D.mean(), mode='full', method='fft')
lags = correlation_lags(len(actions[50]), len(residual_dm[50]), mode='full')

xxxx = np.arange(0, len(tester.item()['dmCmdCounter']), 1)

plt.figure()
plt.plot(actions[20])
plt.plot(residual_dm[20])
plt.figure()
plt.plot(lags, corr)
plt.figure()
plt.scatter(xxxx, tester.item()['dmCmdCounter'])




result = actions @ np.linalg.pinv(applied_commands)


plt.figure()
plt.imshow(result)
plt.show()

errrr
#TODO here you have work on the cross correlation
c2m = np.linalg.pinv(tester.item()['m2c'])


DM_shape_modes = tester.item()['dmCmdCube'].squeeze() @ c2m.T
DM_shape_modes = DM_shape_modes - np.mean(DM_shape_modes, axis = 0)

res_modes = tester.item()['modeCube'].squeeze()
res_modes = res_modes - np.mean(res_modes, axis = 0)

f_sample = 200  # Hz

corr = correlate(res_modes[:, 190], DM_shape_modes[:, 190], mode='full', method='fft')
lags = correlation_lags(len(res_modes[:, 10]), len(DM_shape_modes[:, 10]), mode='full')# / f_sample

idx_0 = np.argmin(np.abs(lags))
print(lags[idx_0])

corr_short = corr[idx_0 - 100:idx_0 + 100]
lags_short = lags[idx_0 - 100:idx_0 + 100]

lag = lags_short[np.argmax(corr_short)]  # integer-sample delay estimate
print(f"Estimated lag: {lag} samples ({lag * 1e3:.2f} ms)")
print(np.argmax(corr_short))

plt.figure()
plt.plot(lags, corr)
plt.axvline(0, color = 'k', ls = '--', lw = 0.8)
plt.axvline(lag, color = 'red', ls = '--', lw = 0.8)
plt.xlabel('lag [s]')
plt.ylabel('cross-correlation')



# Build a toy closed loop with a KNOWN delay to calibrate your sign convention
N = 5000
true_delay = 2  # samples, chosen by you
rng = np.random.default_rng(50)
atm = np.cumsum(rng.normal(size=N))  # toy "turbulence", any colored process works

D = np.zeros(N)
R = np.zeros(N)
g = 0.5
for k in range(N):
    dm_on_mirror = D[k - true_delay] if k - true_delay >= 0 else 0.0
    R[k] = atm[k] - dm_on_mirror
    D[k] = (D[k-1] if k > 0 else 0.0) + g * R[k]

corr = correlate(D - D.mean(), D - D.mean(), mode='full', method='fft')
corr_shifted = correlate(R - R.mean(), D - D.mean(), mode='full', method='fft')
lags = correlation_lags(len(R), len(D), mode='full')

# search on |corr|, not raw max, since the true lag may be a trough
best_lag = lags[np.argmin(corr)]
print(best_lag, np.sign(corr[np.argmax(np.abs(corr))]))
#so the correlation should be negative no matter the array order, so why is mine positive?

plt.figure()
plt.plot(lags, corr)
plt.axvline(0, color = 'k', ls = '--', lw = 0.8)
plt.axvline(best_lag, color = 'red', ls = '--', lw = 0.8)
plt.xlabel('lag [s]')
plt.ylabel('cross-correlation')
plt.show()
