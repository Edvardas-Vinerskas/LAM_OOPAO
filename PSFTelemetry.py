# -*- coding: utf-8 -*-
"""Reorganized telemetry utilities.

This file keeps the original runtime behavior while grouping methods by
initialization, public workflows, computation/projection, utilities, and
special methods.
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import numpy as np
from astropy.io import fits
from numpy.fft import fft2, fftshift
from maoppy.utils import circavg, compute_otf
from maoppy.instrument import papyrus
from maoppy.psfmodel import Psfao, Turbulent, PsfaoPolychromatic
from maoppy.psffit import psffit
from scipy.ndimage import gaussian_filter
from tkinter.filedialog import askopenfilename
from papylib.constant import TELESCOPE, DM
from papylib.image import structured_bkg, strehl_ratio, otf_diffraction, clean_data, center_cog
import tkinter as tk
from tkinter import filedialog


class PSFTele:
    """Load PSF telemetry cubes and run PSF-oriented image analysis."""

    def __init__(self, tele_path: str=None, is_onsky: bool=True, is_cl: bool=True, crop_img=100, temporal_crop=None, dark_path = None):
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
        self.rad2arcsec = 180 / (2 * np.pi) * 3600
        self.sampling_calib = 2.86
        self.wvl_sky = 1.31e-06
        self.wvl_calib = 1.55e-06
        if tele_path is None:
            tele_path = self._choose_file()
            if tele_path:
                print('Selected file:', tele_path)
            else:
                raise ValueError('No files selected')
        self.tele_path = tele_path
        data = self._load(self.tele_path).astype(np.float32)
        self.img_raw = data[:-1][self.temporal_crop]
        if dark_path is None:
            self._dark = data[-1]
        else:
            self._dark = self._load(dark_path).astype(np.float32).mean(axis=0)
        self.frame_count = self.img_raw[:, 0, 0]
        self.img_raw[:,0,:] = 0
        self._dark[0,:] = 0
        self.crop_img = crop_img
        self.short_exp_psf = self.img_raw - self._dark
        img_raw_mean = self.img_raw.mean(axis=0)
        self.img_raw_mean, self.dark = self._center_cog(img_raw_mean, self._dark, crop_img)
        self.long_exp_PSF = self.img_raw_mean - self.dark
        nb_act_lin = 16
        nb_mode_control = 195 # number of KL modes controlled
        
        if self.is_onsky:
            self.sampling = self.sampling_calib * DM.D_CALIB / DM.D_SKY * self.wvl_sky / self.wvl_calib
            papyrus.occ = TELESCOPE.OBSTRUCTION
            papyrus.Nact = round(nb_act_lin * 37.5/35.5 * np.sqrt(nb_mode_control/241))
            ron = 10
            self.wvl = self.wvl_sky
        else:
            self.sampling = self.sampling_calib
            papyrus.occ = 0
            
            self.wvl = self.wvl_calib


    def psf_analysis(self, psf=None, elevation_deg=90, sampling=None, band = [1050e-9,1450e-9]):
        """Fit a PSF model and store image quality metrics on the instance.
    
        Parameters
        ----------
        psf : ndarray, optional
            PSF image to analyze. If omitted, the long-exposure PSF is used.
        elevation_deg : float, optional
            Elevation angle used to convert fitted r0 into zenith seeing.
        sampling : float, optional
            Pixel sampling used by the PSF model. If omitted, ``self.sampling`` is used.
        """
        psf = self._resolve_psf_input(psf)
    
        results = self.analyze_psf(
            psf=psf,
            elevation_deg=elevation_deg,
            sampling=sampling,
            band = band
        )
    
        self.out = results["out"]
        self.psf_fit = results["psf_fit"]
        self.psf = results["psf"]
        self.otf_fit_avg = results["otf_fit_avg"]
        self.otf_avg = results["otf_avg"]
        self.SR = results["SR"]
        self.otf_diff = results["otf_diff"]
        self.r0_zenith = results["r0_zenith"]
        self.seeing = results["seeing"]
        self.seeing_550 = results["seeing_550"]
        self.SR_otf = results["SR_otf"]
        
    def analyze_psf(self, psf, elevation_deg=90, sampling=None, polychromatic = True, band = [1050e-9,1450e-9]):
        """Analyze a PSF and return the results without modifying the instance.
    
        Parameters
        ----------
        psf : ndarray
            PSF image to analyze.
        elevation_deg : float, optional
            Elevation angle used to convert fitted r0 into zenith seeing.
        sampling : float, optional
            Pixel sampling used by the PSF model. If omitted, ``self.sampling`` is used.
    
        Returns
        -------
        dict
            PSF analysis results.
        """
        sampling = self._resolve_sampling_input(sampling)

        ny, nx = psf.shape
        if ny != nx:
            raise ValueError(f"PSF image must be square. Got shape {psf.shape}.")
        
        
        weights = self._compute_psf_fit_weights(psf)
        if self.is_onsky:
            D = DM.D_SKY
        else: 
            D = DM.D_CALIB 
        if polychromatic:
            self.wvl_sky = np.array(band)
            self.sampling = self.sampling_calib * DM.D_CALIB / D * self.wvl_sky / self.wvl_calib
        else:
            self.wvl_sky = 1310e-9
            self.sampling = [self.sampling_calib * DM.D_CALIB / D * self.wvl_sky / self.wvl_calib]
        
        if self.is_cl:
            psfmodel, psfparam_guess, fixed = self._build_psf_model(sampling=self.sampling,
                                                                    crop_img=nx,
                                                                    )
            
        if self.is_cl:
            psfmodel, psfparam_guess, fixed = self._build_psf_model(sampling=self.sampling,
                                                                    crop_img=nx,
                                                                    )
        else:
            psfmodel, psfparam_guess, fixed = self._build_psf_model(sampling=self.sampling.mean(),
                                                                    crop_img=nx,
                                                                    )
        weights = self._compute_psf_fit_weights(psf)
    
        out = self._fit_psf_model(
            psf=psf,
            psfmodel=psfmodel,
            psfparam_guess=psfparam_guess,
            weights=weights,
            fixed=fixed,
        )
    
        psf_fit = out.psf
        psf_normalized = self._normalize_fitted_psf(psf, out.flux_bck)
    
        otf_fit_avg = self._compute_otf_radial_average(psf_fit, nx)
        otf_avg = self._compute_otf_radial_average(psf_normalized, nx)
    
        SR = np.round(np.abs(compute_otf(out.psf)).sum() / np.abs(psfmodel.otfDiffraction).sum() * 100, decimals = 1)#strehl_ratio(psf_normalized, self.sampling )
    
        otf_diff = psfmodel.otfDiffraction
        
    
        seeing_metrics = self._compute_seeing_metrics(
            fitted_r0=out.x[0],
            elevation_deg=elevation_deg,
            wavelength=self.wvl,
            rad2arcsec=self.rad2arcsec,
        )
    
        SR_otf = psfmodel.strehlOTF(out.x)
    
        return {
            "out": out,
            "psf_fit": psf_fit,
            "psf": psf_normalized,
            "otf_fit_avg": otf_fit_avg,
            "otf_avg": otf_avg,
            "SR": SR,
            "otf_diff": otf_diff,
            "r0_zenith": seeing_metrics["r0_zenith"],
            "seeing": seeing_metrics["seeing"],
            "seeing_550": seeing_metrics["seeing_550"],
            "SR_otf": SR_otf,
            "sampling":  self.sampling,
            "elevation_deg": elevation_deg,
        }
    def _resolve_psf_input(self, psf):
        """Return the input PSF or the stored long-exposure PSF."""
        if psf is None:
            return self.long_exp_PSF
        return psf
    
    
    def _resolve_sampling_input(self, sampling):
        """Return the input sampling or the instance default sampling."""
        if sampling is None:
            return self.sampling_calib
        return sampling
    
    
    def _build_psf_model(self, sampling, crop_img=None):
        """Build the PSF model and initial fitting parameters.
    
        Parameters
        ----------
        sampling : float
            Pixel sampling used by the PSF model.
    
        Returns
        -------
        psfmodel
            PSF model instance used by ``psffit``.
        psfparam_guess : list
            Initial parameter guess passed to ``psffit``.
        fixed : list[bool]
            Boolean mask indicating which parameters are fixed.
        """
        nb_act_lin = 1 + DM.D_CALIB / DM.PITCH
        papyrus.Nact = round(
            nb_act_lin * DM.D_SKY / DM.D_CALIB * np.sqrt(195 / DM.NACT)
        )
        if crop_img is None:
            crop_img = self.crop_img
        if self.is_cl:
            psfmodel = PsfaoPolychromatic(
                (crop_img, crop_img),
                system=papyrus,
                samp=sampling,
            )
            psfparam_guess = [0.05, 1e-4, 0.4, 0.5, 1, 0, 1.5]
            fixed = [False, ]*7
            psfmodel.bounds[1][0] = 0.4 # min seeing set to 0.72"
            psfmodel.bounds[0][3] = 0.5 # min alpha
            
        else:
            
            psfmodel = Turbulent((crop_img, crop_img), system=papyrus, samp=sampling)
            psfparam_guess = [0.09, 30]
            fixed = [False, ]*2
    
        return psfmodel, psfparam_guess, fixed
    
    
    @staticmethod
    def _compute_psf_fit_weights(psf, ron=10):
        """Compute PSF fitting weights from a smoothed PSF image.
    
        Parameters
        ----------
        psf : ndarray
            PSF image used for fitting.
        ron : float, optional
            Read-out noise term added to the smoothed image.
    
        Returns
        -------
        ndarray
            Pixel weights passed to ``psffit``.
        """
        return 1 / (gaussian_filter(psf, 2) + ron ** 2)
    
    
    @staticmethod
    def _fit_psf_model(psf, psfmodel, psfparam_guess, weights, fixed):
        """Fit a PSF model without modifying the class instance."""
        return psffit(
            psf,
            psfmodel,
            psfparam_guess,
            weights=weights,
            fixed=fixed,
            max_nfev=60,
        )
    
    
    @staticmethod
    def _normalize_fitted_psf(psf, flux_bck):
        """Normalize a fitted PSF using the fitted flux and background terms.
    
        Parameters
        ----------
        psf : ndarray
            Raw PSF image.
        flux_bck : sequence
            Fitted flux and background parameters returned by ``psffit``.
    
        Returns
        -------
        ndarray
            Normalized PSF image.
        """
        return (psf - flux_bck[1]) / flux_bck[0]
    
    
    @staticmethod
    def _compute_otf(img):
        """Compute the absolute centered optical transfer function of an image."""
        return np.abs(fftshift(fft2(fftshift(img))))
    
    
    @classmethod
    def _compute_otf_radial_average(cls, img, crop_img):
        """Compute the circular average of an image OTF."""
        return circavg(
            cls._compute_otf(img),
            center=(crop_img // 2, crop_img // 2),
        )
    
    
    @staticmethod
    def _compute_seeing_metrics(fitted_r0, elevation_deg, wavelength, rad2arcsec):
        """Compute zenith r0 and seeing metrics from a fitted r0 value.
    
        Parameters
        ----------
        fitted_r0 : float
            Fitted Fried parameter returned by the PSF model.
        elevation_deg : float
            Elevation angle in degrees.
        wavelength : float
            Observation wavelength in meters.
        rad2arcsec : float
            Conversion factor from radians to arcseconds.
    
        Returns
        -------
        dict
            Dictionary containing ``r0_zenith``, ``seeing``, and ``seeing_550``.
        """
        r0_zenith = fitted_r0 / np.cos(
            np.pi / 2 - elevation_deg * np.pi / 180
        ) ** (3 / 5)
    
        seeing = rad2arcsec * wavelength / r0_zenith
        seeing_550 = seeing * (550e-9 / wavelength) ** (-1 / 5)
    
        return {
            "r0_zenith": r0_zenith,
            "seeing": seeing,
            "seeing_550": seeing_550,
        }
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
        file_path = filedialog.askopenfilename(title='Select a file', filetypes=[('Python files', '*.fits'), ('All files', '*.*')])
        root.destroy()
        return file_path

    def _load(self, path):
        """Load telemetry data from a supported file.

        Parameters
        ----------
        path : str
            Path to a ``.npy`` or ``.fits`` file.
        
        Returns
        -------
        ndarray
            Loaded data array."""
        if path.endswith('.npy'):
            out = np.load(path)
        elif path.endswith('.fits'):
            f = fits.open(path)
            out = f[0].data
            f.close()
        else:
            raise ValueError('Do not know how to open required extension')
        return out

    def _delete_img(self):
        """Delete image cubes from the instance to release memory."""
        del self.img, self.img_raw
