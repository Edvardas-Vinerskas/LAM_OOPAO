# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 13:38:53 2026

@author: mmotte
"""

import numpy as np


import logging  # For logging messages
import tqdm

from scipy.signal import welch
from scipy.interpolate import interp1d
from skimage.transform import resize
logging.basicConfig(level=logging.INFO)  # Set up logging
logger = logging.getLogger(__name__)  # Create logger
import tkinter as tk
from tkinter import filedialog
import os
import sys
global HERE
HERE = os.path.dirname(os.path.abspath(__file__))
from OOPAO.Zernike import Zernike
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from parallel_utils import  _import_oopao_symbols
class PAPYtele:
    """
    Analyze PAPYRUS telemetry data, reconstruct wavefront phase, and compute
    modal or influence-function PSDs from telemetry cubes.
    """

    def __init__(self,tele_path:str = None, OG:float|np.ndarray=None, temporal_crop = None):
        """
        Initialize the telemetry analysis object from a saved telemetry file.

        Parameters
        ----------
        tele_path : str, optional
            Path to the input ``.npy`` telemetry file. If ``None``, a file
            selection dialog is opened.

        Raises
        ------
        ValueError
            If no file is selected when ``tele_path`` is not provided.
        """
        self.tag = 'ozi'
        self.temporal_crop = slice(None) if temporal_crop is None else slice(temporal_crop[0], temporal_crop[1])
        if tele_path is None:
            tele_path = self._choose_file()

            if tele_path:
                print("Selected file:", tele_path)
            else:
                raise ValueError('No files selected')
        self.tele_path = tele_path
        if OG is None: 
            self.OG = 1.
        else:
            self.OG = OG
        self.data = np.load(self.tele_path, allow_pickle=True)

        self.dmshape = self.data.item()['dmCmdCube'].astype(np.float32)[self.temporal_crop]
        self.rec_cmd = self.data.item()['modeCube'].astype(np.float32)[self.temporal_crop][...,0]/self.OG
        self.ts = self.data.item()['timeStampOcamCube'][self.temporal_crop]  # list[datetime]
        self.t0 = self.ts[0]
        self.time = np.array([(t - self.t0).total_seconds() for t in self.ts], dtype=float)
        self.M2C = self.data.item()['m2c']
 
        self.proj_IF = np.load(HERE+'\projector_IF_sky.npy')
        self.proj_modes= np.load(HERE+'\projector_M2C_sky.npy')
        pupil = self.proj_IF.sum(axis = 0)
        self.pupil_mask = pupil!=0

        self.IF = np.load('IF_DM1.npy')
        
        self.M2phase = self.IF[self.pupil_mask,:]@self.M2C
        self.modes_std = self.M2phase.std(axis = 0)
        self.IF_std = self.IF[self.pupil_mask,:].std(axis = 0)
        self.initialise_OOPAO_objects()

    def compute_all_PSD(self, npsg = None):
        """
        Compute all available PSD products for reconstructed phases and commands.

        Parameters
        ----------
        npsg : int, optional
            Requested segment length for PSD estimation.
        """
        self.PSD_cmd_IFs(npsg)
        self.PSD_cmd_modal(npsg)
    def PSD_cmd_IFs(self, npsg = None):
        """
        Compute PSDs of the reconstructed command vectors in actuator space.

        Parameters
        ----------
        npsg : int, optional
            Requested segment length for PSD estimation.
        """
        if npsg is None:
            npsg = self.time.size
        
        self.psd_cmd_IFs = self.psd(self.time, (self.M2C@self.rec_cmd.T).T*self.IF_std[None,:], nperseg=npsg)
    def PSD_cmd_modal(self, npsg = None):
        """
        Compute PSDs of the reconstructed command vectors in modal space.

        Parameters
        ----------
        npsg : int, optional
            Requested segment length for PSD estimation. 
        """
        if npsg is None:
            npsg = self.time.size
        logger.info('Computing modal PSD from cmd')
        self.psd_cmd_modal = self.psd(self.time, self.rec_cmd*self.modes_std[None,:], nperseg=npsg)
    def _uniform_resample(self, t, x, fs=None):
        """
        Interpolate irregularly sampled time series onto a uniform time grid.

        Parameters
        ----------
        t : ndarray
            One-dimensional time stamps in seconds, assumed monotonic.
        x : ndarray
            Time series array with samples along the first axis.
        fs : float, optional
            Target sampling frequency. If ``None``, it is inferred from the
            median time step.

        Returns
        -------
        tu : ndarray
            Uniformly sampled time vector.
        xu : ndarray
            Interpolated time series on the uniform grid.
        fs : float
            Sampling frequency used for resampling.
        """
        t = np.asarray(t).ravel()
        x = np.asarray(x)
        
        dt = np.diff(t)
        dt_med = np.nanmedian(dt)
        if fs is None:
            fs = 1.0 / dt_med

        # Uniform grid (use overlap of available time span)
        t0, t1 = t[0], t[-1]
        Nu = int(np.floor((t1 - t0) * fs)) + 1
        tu = t0 + np.arange(Nu) / fs

        # Interpolate each mode
        f = interp1d(t, x, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate")
        xu = f(tu)
        return tu, xu, fs
    def psd(self, t, a,fs=None, nperseg=4096, noverlap=None, detrend="constant", window="hann"):
        """
        Estimate power spectral densities using Welch's method.

        Parameters
        ----------
        t : ndarray
            Time stamps associated with the samples.
        a : ndarray
            Input time series, with one signal per column.
        fs : float, optional
            Sampling frequency. If ``None``, it is inferred after uniform
            resampling.
        nperseg : int, optional
            Segment length used by Welch's method.
        noverlap : int, optional
            Number of overlapping points between segments. If ``None``, half of
            ``nperseg`` is used.
        detrend : str or callable, optional
            Detrending strategy passed to ``scipy.signal.welch``.
        window : str or tuple or array_like, optional
            Window specification passed to ``scipy.signal.welch``.

        Returns
        -------
        f : ndarray
            Frequency array.
        psd : ndarray
            Power spectral density estimates with shape ``(n_freq, n_signals)``.
        """
        # Resample to uniform grids
        
        _, a_u, fs = self._uniform_resample(t, a, fs=fs)
    
        fs = fs
        # Ensure same fs



        if noverlap is None:
            noverlap = nperseg // 2

        M = a_u.shape[1]

        psd = []

        # Welch per mode (robust, avoids huge memory)
        for m in range(M):
            
            f, P = welch(a_u[:, m], fs=fs, window=window,
                            nperseg=nperseg, noverlap=noverlap,
                            detrend=detrend, return_onesided=True, scaling="density")

            psd.append(P)


        psd = np.stack(psd, axis=1)  # (F, M)

        
        return  f.astype(np.float32),psd.astype(np.float32)
    def initialise_OOPAO_objects(self):
        Source, Telescope, _, _, DeformableMirror, MisRegistration, _ = _import_oopao_symbols()
        
        self.src = Source(optBand='V', magnitude=0)
   
        
        pupil =self.pupil_mask.reshape(-1,np.int64(self.pupil_mask.shape[0]**(1/2)))[::-1,:]
        self.tel = Telescope(pupil.shape[0],1.52, pupil = pupil) 
        
        self.src*self.tel
        
        # self.cam = Detector(psf_sampling=self.psf_sampling)

        param = np.load(HERE+'\dm_second_stage_misreg_dict.npy', allow_pickle=True).item()
        m = MisRegistration(param)

        self.dm_ozi = DeformableMirror(telescope = self.tel,
                                nSubap=10,
                                mechCoupling=0.35,
                                print_dm_properties=False,
                                pitch=0.11,
                                misReg = m,
                                sign=-1e-5)
        if_path = os.path.join(HERE, "IF_vZWFS.npy")
        if not os.path.exists(if_path):
            raise FileNotFoundError(
                f"Fichier d'influence functions introuvable : {if_path}"
            )

        # IF = np.load(if_path)
        # if IF[0,...].shape != self.tel.pupil.shape:
        #     IF = self._rescale_matrix(IF, self.tel.pupil.shape[0], self.tel.pupil.shape[1])
        # # self.IF_ozi = np.load(HERE+'\IF_dm2.npy').reshape(97,-1).T*1e-6
        # self.IF_ozi = IF.reshape(97, -1).T.astype(np.float32) 

        # self.IF_ozi_std = self.IF_ozi[(self.tel.pupil==1).ravel(),:].std(axis = 0)
        # amplitude_mean = -np.ptp(self.IF_ozi_std,axis =0)
        self.IF_ozi =np.load('C:/Users/mmotte/OZITelemetry/ozitelemetry/IF_dm2.npy')*1e-6
       
        print(self.IF_ozi.shape)
        amplitude_mean = np.ptp(self.IF_ozi,axis =0)
        self.dm_ozi.modes*=amplitude_mean/np.ptp(self.dm_ozi.modes)
        #self.dm_ozi.modes = self.IF_ozi.copy() 
    def _rescale_matrix(self,A, j,k, anti_aliasing = True):
        """
        Resample a 2D, 3D, or 4D array to a target spatial shape.

        Parameters
        ----------
        A : ndarray
            Input array to resize. Supported dimensions are 2, 3, and 4.
        j : int
            Target size along the first spatial axis.
        k : int
            Target size along the second spatial axis.
        anti_aliasing : bool, optional
            Whether to apply anti-aliasing during resizing.

        Returns
        -------
        ndarray
            Resized array with the requested spatial dimensions.
        """
        if A.ndim ==3:
            l, m, n = A.shape
            return resize(A, (l, j, k), order=5, anti_aliasing=anti_aliasing)
        elif A.ndim ==4:
            o ,l, m, n = A.shape
            return resize(A, (o,l, j, k), order=5, anti_aliasing=anti_aliasing)
        elif A.ndim == 2:
            m, n = A.shape
            return resize(A, (j, k), order=5, anti_aliasing=anti_aliasing)
    def _compute_proj_dm(self, modal_basis, tel, dm, return_modes = False):
        dm.coefs = modal_basis
        tel*dm
        modes= tel.OPD.copy()
        modes = modes.reshape((tel.resolution**2, modes.shape[-1]))/tel.OPD[tel.pupil, :].std(axis=0)  # Flatten for projection
        cov_modes = modes.T @ modes  # Compute mode covariance
        if return_modes:
            return np.diag(1 / np.diag(cov_modes)) @ modes.T, modes  # Pseudo-inverse projection matrix
        else:
            return np.diag(1 / np.diag(cov_modes)) @ modes.T  # Pseudo-inverse projection matrix
    def _compute_proj_OPDs(self, modes, tel):
        

        modes = modes.reshape((tel.resolution**2, modes.shape[-1]))/modes[tel.pupil, :].std(axis=0)  # Flatten for projection
        cov_modes = modes.T @ modes  # Compute mode covariance
        return np.diag(1 / np.diag(cov_modes)) @ modes.T  # Pseudo-inverse projection matrix
    def compute_Zernike_basis(self, nmodes = 30):
        Zer_basis = Zernike(self.tel, J= nmodes)
        Zer_basis.computeZernike(self.tel)
        self.Zer_modes = Zer_basis.modesFullRes.copy()
        self.proj_Zer = self._compute_proj_OPDs(self.Zer_modes, self.tel)
    def OPDs_map_from_cmd(self):
        OPD_map = self.IF@(self.M2C@self.rec_cmd.T)
        return OPD_map.T.reshape(-1, self.tel.pupil.shape[0], self.tel.pupil.shape[0])
    def project_on_OZIRIIS(self, modes, proj = None):
        if proj is None:
            proj, modes = self._compute_proj_dm(modes, self.tel, self.dm_ozi, return_modes = True)
        rec_modes = np.zeros((self.rec_cmd.shape[0], modes.shape[-1]))
        for i in tqdm.tqdm(range(self.rec_cmd.shape[0])):
            opds = self.IF@(self.M2C@self.rec_cmd[i])
            rec_modes[i] = proj@opds
        
        return rec_modes, modes, proj
    def _project_phase(self,projector, phase):
        """
        Apply a projector to a stack of phase or OPD maps.

        Parameters
        ----------
        projector : ndarray
            Projection matrix of shape ``(n_coefficients, n_pixels)``.
        phase : ndarray
            Stack of 2D maps to project.

        Returns
        -------
        ndarray
            Projected coefficients for each input frame.
        """
        projected_phase = np.zeros((phase.shape[0],projector.shape[0])).astype(np.float32)
        for i in tqdm.tqdm(range(phase.shape[0])):
            projected_phase[i]=projector@phase[i].ravel()
        return projected_phase.astype(np.float32)
    def __matmul__(self, obj):
        if obj.tag == "ozi":
            
            self.ozi_if, self.ozi_if_modes, self.ozi_proj_if = self.project_on_OZIRIIS(np.identity(97))#, proj=np.linalg.pinv(self.dm_ozi.modes))
            self.ozi_modes, self.ozi_KL_basis, self.ozi_proj_modes = self.project_on_OZIRIIS(obj.M2C)
            print('First stage command projected on second stage')
        else:
            raise ValueError('Entered object not an OZItele')
    
    def _choose_file(self):
        from qtpy.QtWidgets import QApplication, QFileDialog
    
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
    
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select OZIRIIS linearity .npz file",
            "",
            "NPZ files (*.npz);;All files (*.*)",
        )

        return file_path
