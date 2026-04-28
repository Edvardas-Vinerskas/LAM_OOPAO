# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:30:29 2026

@author: mmotte
"""

import numpy as np


import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import numpy as np
from astropy.io import fits
from numpy.fft import fft2, fftshift
from maoppy.utils import circavg
from maoppy.instrument import papyrus
from maoppy.psfmodel import Psfao, Turbulent
from maoppy.psffit import psffit
from scipy.ndimage import gaussian_filter
from tkinter.filedialog import askopenfilename

from papylib.constant import TELESCOPE, DM
from papylib.image import structured_bkg, strehl_ratio, otf_diffraction, clean_data, center_cog
import tkinter as tk
from tkinter import filedialog

class PSFTele:
    """
    Analyze OZIRIIS telemetry data, reconstruct wavefront phase, and compute
    modal or influence-function PSDs from telemetry cubes.
    """

    def __init__(self,tele_path:str = None, is_onsky:bool=True, is_cl:bool = True, crop_img = 100, temporal_crop = None):
        """
        Initialize the telemetry analysis object from a saved telemetry file.

        Parameters
        ----------
        tele_path : str, optional
            Path to the input ``.npy`` telemetry file. If ``None``, a file
            selection dialog is opened.
        is_onsky : bool, optional
            Whether the telemetry corresponds to on-sky data. This changes the
            source definition and pupil preparation workflow.
        CNN : bool, optional
            Fallback flag indicating whether the stored reconstruction command
            cube comes from a CNN-based reconstructor when this information is
            not found in the telemetry file.

        Raises
        ------
        ValueError
            If no file is selected when ``tele_path`` is not provided.
        """
        self.tag = 'psf'
        self.temporal_crop = slice(None) if temporal_crop is None else slice(temporal_crop[0], temporal_crop[1])
        self.is_onsky = is_onsky
        self.is_cl = is_cl
        self._center_cog = center_cog
        self.rad2arcsec = 180/(2*np.pi)*3600


        #%% FIXED PARAMETERS
        self.sampling_calib = 2.84 # from CRED3 data 05/03/2025
        self.wvl_sky = 1310e-9 # central wavelength
        self.wvl_calib = 1550e-9 # internal laser source
        if tele_path is None:
            tele_path = self._choose_file()

            if tele_path:
                print("Selected file:", tele_path)
            else:
                raise ValueError('No files selected')
        
        self.tele_path = tele_path
        data = self._load(self.tele_path).astype(np.float32)

        self.img_raw = data[:-1][self.temporal_crop] 
        self.dark = data[-1]
        self.frame_count = self.img_raw[:,0,0]
        self.crop_img = crop_img
        self.short_exp_psf = self.img_raw-self.dark
        img_raw_mean = self.img_raw.mean(axis=0)
        self.img_raw_mean, self.dark = self._center_cog(img_raw_mean, self.dark, crop_img)
        self.long_exp_PSF = self.img_raw_mean- self.dark
        
        if self.is_onsky:
            self.sampling = self.sampling_calib * DM.D_CALIB/DM.D_SKY * self.wvl_sky/self.wvl_calib
            papyrus.occ = TELESCOPE.OBSTRUCTION
            self.wvl = self.wvl_sky
        else:
            self.sampling = self.sampling_calib
            papyrus.occ = 0
            self.wvl = self.wvl_calib
    def psf_analysis(self, psf = None, elevation_deg = 80):
        nb_act_lin = 1 + DM.D_CALIB/DM.PITCH
        papyrus.Nact = round(nb_act_lin * DM.D_SKY/DM.D_CALIB * np.sqrt(195/DM.NACT))
        if psf is None:
            psf = self.long_exp_PSF
        ron = 10
        weights = 1/(gaussian_filter(psf, 2)+ron**2)

        if self.is_cl:
            psfmodel = Psfao((self.crop_img,self.crop_img), system=papyrus, samp=self.sampling)
            psfparam_guess = [0.09, 1e-4, 0.4, 0.5, 1, 0, 1.5]
            fixed = [False, ]*7
        else:
            psfmodel = Turbulent((self.crop_img,self.crop_img), system=papyrus, samp=self.sampling)
            psfparam_guess = [0.09, 30]
            fixed = [False, ]*2
            
        self.out = psffit(psf, psfmodel, psfparam_guess, weights=weights, fixed=fixed, max_nfev=30)
        def get_otf(img):
            return np.abs(fftshift(fft2(fftshift(img))))
        self.otf_fit_avg = circavg(get_otf(self.out.psf), center=(self.crop_img//2,self.crop_img//2))
        self.psf_fit = self.out.psf
        self.psf = (psf-self.out.flux_bck[1])/self.out.flux_bck[0]

        self.otf_avg = circavg(get_otf(self.psf), center=(self.crop_img//2,self.crop_img//2))

        self.SR = strehl_ratio(self.psf, self.sampling)
        self.otf_diff = otf_diffraction(self.crop_img, self.sampling, sky=self.is_onsky)
        self.otf_diff_avg = circavg(self.otf_diff, center=(self.crop_img//2,self.crop_img//2))
        self.r0_zenith = self.out.x[0]/np.cos(np.pi/2-elevation_deg*np.pi/180)**(3/5)
        self.seeing = self.rad2arcsec*self.wvl/self.r0_zenith
        self.seeing_550 = self.seeing * (550e-9/self.wvl)**(-1/5)
        self.SR_otf= psfmodel.strehlOTF(self.out.x)
    def _choose_file(self):
        """
        Open a file dialog and return the selected telemetry file path.

        Returns
        -------
        str
            Path to the selected file, or an empty string if no file is chosen.
        """
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.askopenfilename(
            title="Select a file",
            filetypes=[("Python files", "*.fits"), ("All files", "*.*")]
        )
        
        root.destroy()
        return file_path
    
    def _delete_img(self):
        del self.img, self.img_raw
    def _load(self,path):
        if path.endswith('.npy'):
            out = np.load(path)
        elif path.endswith('.fits'):
            f = fits.open(path)
            out = f[0].data
            f.close()
        else:
            raise ValueError('Do not know how to open frequired extension')
        
        return out