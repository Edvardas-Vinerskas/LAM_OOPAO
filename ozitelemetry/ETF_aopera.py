# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:01:34 2026

@author: mmotte
"""

"""
Compute ETF in continuous (Laplace) and discrete (Z) transforms, for comparison.
Compute the noise-transfer function (NTF) in continuous transform.
Compare with the analytical formulas for bandwidths.
"""

import numpy as np
import matplotlib.pyplot as plt
from aopera.control import closed_loop_transfer, bandwidth, noise_transfer, bandwidth_noise

Fao = 1000 # AO loop frequency [Hz]
ki = 0.4 # integrator gain
frame_delay = 3 # pure frame delay (WFS integration excluded)
db = 0 # choosen cutoff of the ETF [dB]
dbn = -4.5 # chosen cutoff of the NTF [dB]
leak = 0.94

freq = np.logspace(0, np.log10(2*Fao), num=5000)

#%% TRANSFER FUNCTION
cl_cont = closed_loop_transfer(freq, Fao, ki, frame_delay, discrete=False, leak=leak)
etf2_cont = np.abs(cl_cont)**2
cl_disc = closed_loop_transfer(freq, Fao, ki, frame_delay, discrete=True, leak=leak)
etf2_disc = np.abs(cl_disc)**2

bw = bandwidth(ki, frame_delay, Fao, db=db)

#%% NOISE TRANSFER
ntf2 = np.abs(noise_transfer(freq, Fao, ki, frame_delay, discrete=True, leak=leak))**2

vld = np.where(freq<(Fao/2))
noise_propa = 2*np.trapezoid(ntf2[vld], freq[vld])/Fao # numerical integration

bwn = bandwidth_noise(ki, frame_delay, Fao, db=dbn)

print('bilateral integral of |NTF|²/Fao : %.2f'%noise_propa)
print('4 * BW / Fao                     : %.2f'%(4*bw/Fao))
print('2 * BW_noise / Fao               : %.2f'%(2*bwn/Fao))

#%% PLOT
plt.figure(1)
plt.clf()
plt.loglog(freq, etf2_cont, lw=2, label='ETF continuous')
plt.loglog(freq, etf2_disc, lw=2, label='ETF discrete')
plt.loglog(freq, ntf2, lw=2, label='NTF discrete')
plt.scatter(bw, 10**(db/10), c='k', label='bandwidth')
plt.axvline(bw, c='k', ls=':', lw=2)
plt.scatter(bwn, 10**(dbn/10), c='gray', label='bandwidth noise')
plt.axvline(bwn, c='gray', ls=':', lw=2)
plt.axvline(Fao/2, c='r', ls='--', label='F / 2')
plt.ylim(1e-3, 10)
plt.xlim(min(freq), max(freq))
plt.grid()
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.title('Transfer function')
