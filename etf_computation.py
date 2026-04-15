# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:19:53 2026

@author: mmotte
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:25:37 2026

@author: mmotte
"""

import numpy as np
import sys
import os
from scipy.signal import welch
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from OZITelemetry import OZITele
import Class_ETFs
from plot_functions import plot_psd_aa
#%%




cl_tele = OZITele(tele_path=None, is_onsky=True)
ol_tele = OZITele(tele_path=None, is_onsky=True)

cl_tele.reconstruct_all_phase(parallel=True, parall_njob=4)
ol_tele.reconstruct_all_phase(parallel=True, parall_njob=4)

#%%
#%%
cl_tele.reconstruct_all_phase(iteration = 15, damping=0.3)
ol_tele.reconstruct_all_phase(iteration = 15, damping=0.3)
#%%
cl_tele.project_OPDs()
ol_tele.project_OPDs()
#%%
cl_tele.compute_all_PSD() #
ol_tele.compute_all_PSD() #
#%% 
freq_atan_cl, modal_psd_atan_cl = cl_tele.psd_modal
freq_atan_ol, modal_psd_atan_ol = ol_tele.psd_modal

freq_cl, modal_psd_cl = cl_tele.psd_cmd_modal
freq_ol, modal_psd_ol = ol_tele.psd_cmd_modal
#%%
fig_atg, ax_atg = plot_psd_aa( 
    freq_atan_cl,
    modal_psd_atan_cl[:,2:40],
    f2=freq_atan_ol,
    psd2=modal_psd_atan_ol[:,2:40],
    label1="open loop",
    label2="closed loop",
    method=np.nansum,
    f_unit="Hz",
    psd_unit=r"nm$^2$/Hz",
    fmin=None,
    fmax=None,
    normalised=False,
    show_legend=True,
    one_column=True,
    dpi=300,
    save=False,
    savepath="mean_psd_aa.pdf",
    saveformat=None,
    journal_style=True,   # True: A&A final style ; False: working style with light grid
)
#%%
fig_atg, ax_atg = plot_psd_aa( 
    freq_cl,
    modal_psd_cl[:,2:40],
    f2=freq_ol,
    psd2=modal_psd_ol[:,2:40],
    label1="open loop",
    label2="closed loop",
    method=np.nansum,
    f_unit="Hz",
    psd_unit=r"nm$^2$/Hz",
    fmin=None,
    fmax=None,
    normalised=False,
    show_legend=True,
    one_column=True,
    dpi=300,
    save=False,
    savepath="mean_psd_aa.pdf",
    saveformat=None,
    journal_style=True,   # True: A&A final style ; False: working style with light grid
)
#%%
pj = cl_tele.reconstruct_phase(cl_tele.img_ZWFS1[0], cl_tele.img_ZWFS2[0], damping=0.3)

#%%
plt.imshow(cl_tele.vzwfs.zwfs2.sin_phase)

