# -*- coding: utf-8 -*-
"""Reorganized telemetry utilities.

This file keeps the original runtime behavior while grouping methods by
initialization, public workflows, computation/projection, utilities, and
special methods.
"""

import numpy as np
import scipy as scp
from Pupil_selection import reference_intensities
from skimage.transform import resize
import logging
import tqdm
from scipy.signal import welch
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import tkinter as tk
from tkinter import filedialog
import os
import sys
global HERE
from pathlib import Path

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from parameterFile_papyriis import initializeParameterFile
from parallel_utils_twin import _reconstruct_phase_worker, _import_all_oopao_symbols, _simulate_psf_chunk_worker
from OOPAO.Zernike import Zernike


class OZIRIIS:
    def __init__(self, is_onsky: bool=True,  param = None, controlled_modes = 35):
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
        self.tag = 'ozi'
        self.is_onsky = is_onsky
        if param is None:
            self.param = initializeParameterFile()
        else:
            self.param = param
        self.M2C = np.load(str(HERE) +'\M2C_2nd.npy')
        self.psf_sampling = 2.56
        self.initialise_OOPAO_objects()
        self.compute_projectors()

    
        
        
    def initialise_OOPAO_objects(self):
        """Initialise the OOPAO optical objects used by reconstruction and simulation."""
        Source, Telescope, ZWFS, ZWFS2, DeformableMirror, MisRegistration, Detector, Atmosphere  = _import_all_oopao_symbols()
        if self.is_onsky:
            self.src = Source(optBand='H', magnitude=-2.5)
            self.src.wavelength = 1.6e-06
            self.src.bandwidth = 2e-07
            
        else:
            self.src = Source(optBand='H', magnitude=-2.5)
            self.src.wavelength = 1.55e-06
            self.src.bandwidth = 0
            
        
        self.initialise_tel_objects()
        self.src * self.tel 
        if self.is_onsky:
            dia = self.param['d_z_sky']
        else:
            dia = self.param['d_z_calib'] 
        self.vzwfs = ZWFS2(tel = self.tel, diameter = dia, phase_shift = self.param['depth'], zpf = self.param['mask_px_size'])
        self.zwfs1 = self.vzwfs.zwfs1
        self.zwfs2 = self.vzwfs.zwfs2
        self.cam = Detector(psf_sampling=self.psf_sampling)
        param = np.load(str(HERE) + '\dm_second_stage_misreg_dict.npy', allow_pickle=True).item()
        m = MisRegistration(param)
        
        self.dm = DeformableMirror(telescope=self.tel, nSubap=10, mechCoupling=0.35, print_dm_properties=False, pitch=0.11, misReg=m)
   
        if_path = os.path.join(HERE, 'IF_vZWFS.npy')
        self.IF = np.load(os.path.join(HERE, 'IF_dm2.npy')) * 1e-06

        amplitude_mean = np.ptp(self.IF, axis=0)
        self.dm.modes *= amplitude_mean / np.ptp(self.dm.modes)

        self.src_imaging = Source(optBand='IR1310', magnitude=-2.5)
        self.atm=Atmosphere(telescope     = self.tel,\
                       r0            = self.param['r0'],\
                       L0            = self.param['L0'],\
                       windSpeed     = self.param['windSpeed'],\
                       fractionalR0  = self.param['fractionnalR0'],\
                       windDirection = self.param['windDirection'],\
                       altitude      = self.param['altitude'])
            
        self.M2Phase = (self.dm.modes@ self.M2C)
        self.Phase2Modes = np.linalg.pinv(self.M2Phase)
    
    def initialise_tel_objects(self):
        Source, Telescope, ZWFS, ZWFS2, DeformableMirror, MisRegistration, Detector, _  = _import_all_oopao_symbols()
        ratio_sky_calib = 37.5/35.5
        self.tel_calib = Telescope(resolution    = self.param['resolution_2nd'],
                        diameter            = self.param['diameter']*ratio_sky_calib,
                        samplingTime        = self.param['samplingTime_2nd'],
                        centralObstruction  = 0)


        tel_sky = Telescope(resolution      = self.param['resolution_2nd_sky'] ,
                        diameter            = self.param['diameter'],
                        samplingTime        = self.param['samplingTime_2nd'],
                        centralObstruction  = 0.3)
        pupil_sky = np.pad(tel_sky.pupil.copy(), pad_width=self.param['pad_sky'])
        self.tel_sky = Telescope(resolution      = pupil_sky.shape[0],
                        diameter            = self.param['diameter'],
                        samplingTime        = self.param['samplingTime_2nd'],
                        pupil = pupil_sky)

        if self.is_onsky:
            self.tel = self.tel_sky
        else:
            self.tel = self.tel_calib

        
    def compute_synth_IM(self, stroke_nm=12):
        """Compute a synthetic interaction matrix from DM modal strokes."""
        M2Phase = self.IF * 1e9 @ self.M2C
        std_phase = np.std(M2Phase, axis=0)
        eps = 1e-12
        M2C = self.M2C.copy()
        stroke = stroke_nm / std_phase
        IM1 = []
        IM2 = []
        for i in range(M2C.shape[-1]):
            self.dm.coefs = stroke[i] * M2C[:, i]
            self.tel * self.dm
            self.tel * self.vzwfs
            img1_pos = self.zwfs1.img_ZWFS
            img2_pos = self.zwfs2.img_ZWFS
            self.dm.coefs = -stroke[i] * M2C[:, i]
            self.tel * self.dm
            self.tel * self.vzwfs
            IM1.append((img1_pos - self.zwfs1.img_ZWFS) / (2 * stroke[i]))
            IM2.append((img2_pos - self.zwfs2.img_ZWFS) / (2 * stroke[i]))
        return np.array(IM1)[:, self.tel.pupil ==1].T, np.array(IM2)[:, self.tel.pupil ==1].T
    
    def _rescale_matrix(self, A, j, k, anti_aliasing=True):
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
        if A.ndim == 3:
            l, m, n = A.shape
            return resize(A, (l, j, k), order=5, anti_aliasing=anti_aliasing)
        elif A.ndim == 4:
            o, l, m, n = A.shape
            return resize(A, (o, l, j, k), order=5, anti_aliasing=anti_aliasing)
        elif A.ndim == 2:
            m, n = A.shape
            return resize(A, (j, k), order=5, anti_aliasing=anti_aliasing)
        
    def _crop_pupil(self, pupil = None):
        if pupil is None:
            pupil = self.tel.pupil
        where_pupil = np.where(pupil)
        x_, x = where_pupil[0].min(),where_pupil[0].max()
        y_, y = where_pupil[1].min(),where_pupil[1].max()
        final_mask = pupil[x_:x+1,y_:y+1]
        
        return final_mask, (x_, x,y_, y)
    def _crop_opd(self, OPDs:np.ndarray= None, pupil = None):
        if OPDs is None:
            OPDs = self.OPDs
        if pupil is None:
            pupil = self.tel.pupil
        final_mask, _ = self._crop_pupil(pupil)
        if OPDs.ndim == 3:
            OPDs_crop = np.zeros((OPDs.shape[0], final_mask.shape[0], final_mask.shape[1])).astype(np.float32)
            OPDs_crop[:, final_mask] = OPDs[:, pupil]
        else:
            OPDs_crop = np.zeros_like(final_mask).astype(np.float32)
            OPDs_crop[final_mask] = OPDs[pupil]
        return final_mask, OPDs_crop
    
    
    def reconstruct_phase(self, method='atan', damping=0.5, iteration=10):
        return self.vzwfs.reconstructor(iteration=iteration, damping_iteration=damping, reconstructor=method)
    
    def nonlinea_reconstr(self, method='atan', damping=0.5, iteration=10):
        return self.Phase2Modes@self.reconstruct_phase(method, damping, iteration)[self.tel.pupil==1]
    
    def _compute_proj_dm(self, modal_basis, tel, dm, keep_pupil_only = False, filtering = None):
        """
        Calcule un projecteur à partir des modes DM générés dans OOPAO.
        """
        modal_basis = np.asarray(modal_basis, dtype=np.float32)

        if modal_basis.ndim != 2:
            raise ValueError(
                f"modal_basis doit être 2D. Shape reçue : {modal_basis.shape}"
            )

        dm.coefs = modal_basis
        tel * dm

        modes = tel.OPD.copy()
        modes = modes.reshape((tel.resolution**2, modes.shape[-1]))

        std = tel.OPD[tel.pupil, :].std(axis=0)
        std = np.where(np.abs(std) < 1e-30, 1.0, std)

        modes = modes / std[None, :]

        cov_modes = modes.T @ modes
        diag = np.diag(cov_modes)
        diag = np.where(np.abs(diag) < 1e-30, 1.0, diag)

        return (np.diag(1.0 / diag) @ modes.T).astype(np.float32)

    def _compute_proj_OPDs(self, modes, tel, keep_pupil_only = False, filtering = None):
        """
        Calcule un projecteur à partir de modes OPD déjà définis.
        """
        modes = np.asarray(modes, dtype=np.float32)

        if modes.ndim != 3:
            raise ValueError(
                "modes doit avoir la shape (resolution, resolution, n_modes). "
                f"Shape reçue : {modes.shape}"
            )

        if modes.shape[:2] != tel.pupil.shape:
            raise ValueError(
                "Shape spatiale des modes incompatible avec la pupille. "
                f"modes.shape[:2]={modes.shape[:2]}, pupil.shape={tel.pupil.shape}"
            )

        std = modes[tel.pupil, :].std(axis=0)
        std = np.where(np.abs(std) < 1e-30, 1.0, std)

        modes_flat = modes.reshape((tel.resolution**2, modes.shape[-1]))
        modes_flat = modes_flat / std[None, :]

        cov_modes = modes_flat.T @ modes_flat
        diag = np.diag(cov_modes)
        diag = np.where(np.abs(diag) < 1e-30, 1.0, diag)

        return (np.diag(1.0 / diag) @ modes_flat.T).astype(np.float32)
    def _export_reconstruction_setup(self):
        """Return the minimal reconstruction setup required by worker processes."""
    
        if self.is_onsky:
            diameter = self.param['d_z_sky']
        else:
            diameter = self.param['d_z_calib']
    
        return {
            'is_onsky': self.is_onsky,
            'is_nb': False,
    
            'submask0': self.tel.pupil.copy(),
            'submask1': self.tel.pupil.copy(),
            'pupil0': self.tel.pupil.copy(),
            'pupil1': self.tel.pupil.copy(),
    
            # paramètres identiques au cas non parallèle
            'diameter': diameter,
            'phase_shift': self.param['depth'],
            'zpf': self.param['mask_px_size'],
    
            # longueur d’onde pour cohérence complète
            'wavelength': self.src.wavelength,
            'bandwidth': self.src.bandwidth,
        }
    def create_images_from_phase(self, phase):
        if phase.ndim == 3:
            self.img_ZWFS1 = np.zeros_like(phase)
            self.img_ZWFS2 = np.zeros_like(phase)
            print('Propagating the phase \n')
            for i in tqdm.tqdm(range(phase.shape[0])):
               
                self.img_ZWFS1[i], self.img_ZWFS2[i] =  self.create_images_from_phase(phase[i])
            return self.img_ZWFS1, self.img_ZWFS2
        elif phase.ndim == 2:
            self.vzwfs.wfs_measure(phase_in=phase)
            img_ZWFS1= self.vzwfs.zwfs1.img_ZWFS.copy()
            img_ZWFS2 = self.vzwfs.zwfs2.img_ZWFS.copy()
            return img_ZWFS1, img_ZWFS2
    def filter_OPDs(self, OPD=None, nmodes=None, modal_basis='KL'):
        """Filter OPD maps on the requested modal basis."""
        if nmodes is None:
            nmodes = self.M2C.shape[-1]
        if OPD is None:
            OPD = self.OPDs.copy()
        if modal_basis == 'KL':
            self.tel.resetOPD()
            self.dm.coefs = 0
            self.dm.coefs = self.M2C
            self.tel * self.dm
            modes = self.tel.OPD.copy()
            modes = modes / self.tel.OPD[self.tel.pupil, :].std(axis=0)
            self.tel.resetOPD()
            self.dm.coefs = 0
            self.tel * self.dm
            OPDs_on_modes = self._project_phase(self.proj_M2C, OPD)
            filtered_OPD = 0
            for i in tqdm.tqdm(range(nmodes)):
                filtered_OPD += OPDs_on_modes[:, None, None, i] * modes[None, :, :, i]
        elif modal_basis == 'IF':
            filtered_OPD = 0
            IF_inv = np.linalg.pinv(self.dm.modes)
            filtered_OPD = (self.dm.modes @ (IF_inv @ OPD.reshape(self.OPDs.shape[0], -1).T)).T.reshape(OPD.shape[0], OPD.shape[1], OPD.shape[2])
        elif modal_basis == 'Zer':
            opd_on_zer = self._project_phase(self.proj_Zer, OPD)
            filtered_OPD = (opd_on_zer[:, None, None, :nmodes] * self.Zer_modes[None, :, :, :nmodes]).sum(axis=-1)
        return filtered_OPD * (self.tel.pupil == 1)
    
    def _project_phase(self, projector, phase):
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
        projected_phase = np.zeros((phase.shape[0], projector.shape[0])).astype(np.float32)
        for i in tqdm.tqdm(range(phase.shape[0])):
            projected_phase[i] = projector @ phase[i].ravel()
        return projected_phase.astype(np.float32)
    
    
    def reconstruct_all_phase(self, method='atan', iteration=10, damping=0.5, parallel=True, parall_njob=4):
        """
        Reconstruct phase maps for the full telemetry sequence.

        Parameters
        ----------
        method : str, optional
            Reconstruction method passed to each frame reconstruction.
        iteration : int, optional
            Requested number of iterations for reconstruction.
        damping : float, optional
            Damping factor applied during each frame reconstruction.
        """
        if parallel:
            setup = self._export_reconstruction_setup()
            n_frames = self.img_ZWFS1.shape[0]
            gen = Parallel(
                    n_jobs=parall_njob,
                    prefer='processes',
                    return_as='generator'
                )(
                    delayed(_reconstruct_phase_worker)(
                        self.img_ZWFS1[i],
                        self.img_ZWFS2[i],
                        setup,
                        method,
                        damping,
                        iteration
                    )
                    for i in range(n_frames)
                )
            self.phase = np.asarray(list(tqdm.tqdm(gen, total=n_frames, desc=f'Phase reconstruction ({method})')), dtype=np.float32)
        else:
            self.phase = np.zeros((self.img_ZWFS1.shape[0], self.tel.pupil.shape[0], self.tel.pupil.shape[1])).astype(np.float32)
            logger.info(f'Computing phase for each frame using {method} reconstruction')
            for i in tqdm.tqdm(range(self.img_ZWFS1.shape[0])):
                self.vzwfs.zwfs1.img_ZWFS,self.vzwfs.zwfs2.img_ZWFS = self.img_ZWFS1[i], self.img_ZWFS2[i]
                self.phase[i] = self.reconstruct_phase(method, damping, iteration).astype(np.float32)
        self._phase2OPD()
        self.has_recontructed_phase = True
    def psd(self, t, a, fs=None, nperseg=4096, noverlap=None, detrend='constant', window='hann'):
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
        _, a_u, fs = self._uniform_resample(t, a, fs=fs)
        fs = fs
        if noverlap is None:
            noverlap = nperseg // 2
        M = a_u.shape[1]
        psd = []
        for m in range(M):
            f, P = welch(a_u[:, m], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, detrend=detrend, return_onesided=True, scaling='density')
            psd.append(P)
        psd = np.stack(psd, axis=1)
        return (f.astype(np.float32), psd.astype(np.float32))
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
        t0, t1 = (t[0], t[-1])
        Nu = int(np.floor((t1 - t0) * fs)) + 1
        tu = t0 + np.arange(Nu) / fs
        f = interp1d(t, x, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
        xu = f(tu)
        return (tu, xu, fs)
    def _phase2OPD(self, phase=None):
        """Convert reconstructed phase maps into OPD maps using the source wavelength."""
        if phase is None:
            phase = self.phase
        self.OPDs = (phase / (2 * np.pi) * self.src.wavelength).astype(np.float32)
        
    def compute_projectors(self):
        """
        Compute projection matrices onto modal commands and influence functions.

        
        """
        self.proj_M2C = self._compute_proj_dm(self.M2C, self.tel, self.dm)
        self.proj_IF = self._compute_proj_dm(np.identity(self.dm.modes.shape[-1]), self.tel, self.dm)