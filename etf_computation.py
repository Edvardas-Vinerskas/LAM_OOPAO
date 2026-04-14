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
from scipy.signal import welch
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from OZITelemetry import OZITele
import Class_ETFs
from plot_functions import plot_psd_aa
#%%
file_adress = None # you can add manually you file but better if it is none as you will 
cl_tele = OZITele(tele_path=file_adress, is_onsky=True) 
ol_tele = OZITele(tele_path=file_adress, is_onsky=True) 

#%%
cl_tele.reconstruct_all_phase()
ol_tele.reconstruct_all_phase()
#%%
cl_tele.project_OPDs()
ol_tele.project_OPDs()
#%%
cl_tele.compute_all_PSD() #
ol_tele.compute_all_PSD() #
#%% 
freq_atan_cl, modal_psd_atan_cl = cl_tele.psd_modal
freq_atan_ol, modal_psd_atan_ol = ol_tele.psd_modal

freq_cl, modal_psd_cl = cl_tele.psd_modal
freq_ol, modal_psd_ol = ol_tele.psd_modal
#%%
fig_atg, ax_atg = plot_psd_aa( 
    freq_atan_cl,
    modal_psd_atan_cl,
    f2=freq_atan_ol,
    psd2=modal_psd_atan_ol,
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
pj = cl_tele.reconstruct_phase(cl_tele.img_ZWFS1[0], cl_tele.img_ZWFS2[0])

#%%
plt.imshow(cl_tele.vzwfs.zwfs2.sin_phase)

