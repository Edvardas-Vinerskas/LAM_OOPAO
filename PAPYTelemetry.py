# -*- coding: utf-8 -*-
"""Reorganized telemetry utilities.

This file keeps the original runtime behavior while grouping methods by
initialization, public workflows, computation/projection, utilities, and
special methods.
"""

import numpy as np
import logging
import tqdm
from scipy.signal import welch
from scipy.interpolate import interp1d
from skimage.transform import resize
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import tkinter as tk
from tkinter import filedialog
from joblib import Parallel, delayed
import os
import sys
global HERE
from pathlib import Path
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from OOPAO.Zernike import Zernike
from parallel_utils import _reconstruct_phase_worker, _import_oopao_symbols, _simulate_psf_chunk_worker
from parallel_utils import _import_oopao_symbols


class PAPYtele:
    """Analyze PAPYRUS telemetry commands and project first-stage telemetry products."""

    def __init__(self, tele_path: str=None, is_onsky = True, OG: float | np.ndarray=None, temporal_crop=None, extract_values = True):
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
        self.tag = 'papy'
        self.temporal_crop = slice(None) if temporal_crop is None else slice(temporal_crop[0], temporal_crop[1])
        if tele_path is None:
            tele_path = self._choose_file()
            if tele_path:
                print('Selected file:', tele_path)
            else:
                raise ValueError('No files selected')
        self.tele_path = tele_path
        if OG is None:
            self.OG = 1.0
        else:
            self.OG = OG
        self.data = np.load(self.tele_path, allow_pickle=True)
        self.frame_count = self.data.item()['ocamCounter'].astype(np.float32)[self.temporal_crop]
        self.dmshape = self.data.item()['dmCmdCube'].astype(np.float32)[self.temporal_crop][...,0]
        self.dmCL = self.data.item()['dmCLCube'].astype(np.float32)[self.temporal_crop][...,0]
        self.dmFlat = self.data.item()['dmFlat'].astype(np.float32)[self.temporal_crop][...,0]
        self.dmOffset = self.data.item()['dmOffset'].astype(np.float32)[self.temporal_crop][...,0]
        self.rec_cmd = self.data.item()['modeCube'].astype(np.float32)[self.temporal_crop][..., 0]
        self.rec_cmd_with_og = self.data.item()['modeCube'].astype(np.float32)[self.temporal_crop][..., 0]/self.OG 

        self.ts = self.data.item()['timeStampOcamCube'][self.temporal_crop]
        self.loop_gain = self.data.item()['lpGain'][0][0]
        self.loop_leak = self.data.item()['lpLeak'][0][0]
        self.modu_radius = self.data.item()['modulatorRadius']
        self.t0 = self.ts[0]
        self.time = np.array([(t - self.t0).total_seconds() for t in self.ts], dtype=float)
        self.M2C = self.data.item()['m2c']
        self.C2M = np.linalg.pinv(self.M2C)
        self.proj_IF = np.load(str(HERE) +   '/projector_IF_sky.npy')
        self.proj_M2C = np.load(str(HERE) +   '/projector_M2C_sky.npy')
        pupil = self.proj_IF.sum(axis=0)
        self.pupil_mask = pupil != 0
        self.is_onsky = is_onsky
        self.IF = np.load('IF_DM1.npy')
        self.M2phase = self.IF[self.pupil_mask, :] @ self.M2C
        self.modes_std = self.M2phase.std(axis=0)
        self.IF_std = self.IF[self.pupil_mask, :].std(axis=0)
        if extract_values:
            self.initialise_OOPAO_objects()
            self.compute_projectors()
            self.OPDs_map_from_cmd()
            self.psf_sampling = 2.55
            # self.project_OPDs()
    def initialise_OOPAO_objects(self):
        """Initialise OOPAO objects needed to project PAPYRUS telemetry onto OZIRIIS modes."""
        Source, Telescope, _, _, DeformableMirror, MisRegistration, _ = _import_oopao_symbols()
        from parameterFile_papytwin import initializeParameterFile
        param = initializeParameterFile()
        print(param['nActuator'])
        self.src = Source(optBand='V', magnitude=0)
        self.tel = self._tel_first_stage(param)
        self.dm = self._DM_first_stage(param)
        self.tel_ozi = self._tel_second_stage(self.tel)
        self.src * self.tel
        self.src * self.tel_ozi
        param = np.load(str(HERE) +   '/dm_second_stage_misreg_dict.npy', allow_pickle=True).item()
        m = MisRegistration(param)
        
        self.dm_ozi = DeformableMirror(telescope=self.tel, nSubap=10, mechCoupling=0.35, print_dm_properties=False, pitch=0.11, misReg=m)
        
        self.IF_ozi = np.load(str(HERE) + '/IF_dm2.npy') * 1e-6
        print(self.IF_ozi.shape)
        amplitude_mean = np.ptp(self.IF_ozi, axis=0)
        self.dm_ozi.modes *= amplitude_mean / np.ptp(self.dm_ozi.modes)
        self.src_imaging = Source(optBand='IR1310', magnitude=0)
        
    def _tel_first_stage(self, param):
        _, Telescope, _, _, _, MisRegistration, _ = _import_oopao_symbols()
        T152onDM_size       = 35.5 # mm
        PapyrusOnDM_size    = 37.5 # mm 
        ratio_sky_calib = T152onDM_size/PapyrusOnDM_size

        
        # create a temporary Telescope object
        tel_calib = Telescope(resolution    = int(np.round(param['nSubaperture']*param['nPixelPerSubap'])),
                        diameter            = param['diameter']/ratio_sky_calib,
                        samplingTime        = param['samplingTime'],
                        centralObstruction  = 0,
                        fov                 = 0)
        
        n_extra_pix = (param['resolution']-tel_calib.resolution)//2
        pupil_calib = np.pad(tel_calib.pupil,[n_extra_pix,n_extra_pix])
        print(pupil_calib.shape)

        # create a temporary Telescope object
        pupil_sky = self.pupil_mask.reshape(80,80)
        
        
        
        # redefine the pupil padding to accomodate for the calibration or sky mode
        tel = Telescope(resolution          = param['resolution'],
                        diameter            = param['diameter'] * param['resolution']/tel_calib.resolution,
                        samplingTime        = param['samplingTime'],
                        centralObstruction  = 0,
                        fov                 = 0)    
        
        if self.is_onsky:
            tel.pupil = pupil_sky
        else:
            tel.pupil = pupil_calib
        return tel
        
    def _tel_second_stage(self, tel_first_stage):
        _, Telescope, _, _, _, MisRegistration, _ = _import_oopao_symbols()
        papypupil,_ = self._crop_pupil(tel_first_stage.pupil.copy())
        pupil = np.pad(papypupil, pad_width=((5,3),(3,5))) #hard coded
        tel_2nd = Telescope(pupil.shape[0], 1.52, pupil=pupil)
        return tel_2nd
    
    def _DM_first_stage(self, param):
        _, _, _, _, DeformableMirror, MisRegistration, _ = _import_oopao_symbols()
        
        T152onDM_size       = 35.5 # mm
        param_misreg = np.load(str(HERE)+'/dm_first_stage_misreg_dict.npy', allow_pickle=True).item()
        misReg          = MisRegistration(param_misreg)
        pitch           = 2.5 #mm
        DM_diag_size    = param['nActuator'] * pitch #mm
        scale_T152DM = DM_diag_size / T152onDM_size
        D_T152 = 1.52
        
        x = np.linspace(-scale_T152DM * D_T152/2, scale_T152DM * D_T152/2, param['nActuator'])
        [X,Y] = np.meshgrid(x,x)
        
        DM_coordinates = np.asarray([X.reshape(17**2),Y.reshape(17**2)]).T
        dist           = np.sqrt(DM_coordinates[:,0]**2 + DM_coordinates[:,1]**2)
        DM_coordinates = DM_coordinates[dist <= D_T152/2 + 2.2 *pitch * D_T152 / T152onDM_size, :]
        DM_pitch       = pitch * D_T152 / T152onDM_size
        
        # hardcoded for now
        alpao_unit     = 30*7591.024876#1.507e5#
        
        param['dm_coordinates'] = DM_coordinates
        param['pitch']          = DM_pitch
        
        dm=DeformableMirror(telescope    = self.tel,
                            nSubap       = 16,
                            mechCoupling = param['mechanicalCoupling'],
                            misReg       = misReg, 
                            coordinates  = DM_coordinates,
                            pitch        = DM_pitch,
                            modes        = None,
                            flip_lr      = True,
                            sign         = -1/alpao_unit)
        return dm
    def compute_Zernike_basis(self, nmodes=195):
        """Compute a Zernike basis and its corresponding OPD projector."""
        Zer_basis = Zernike(self.tel, J=nmodes)
        Zer_basis.computeZernike(self.tel)
        self.Zer_modes = Zer_basis.modesFullRes.copy()
        self.proj_Zer = self._compute_proj_OPDs(self.Zer_modes, self.tel)

    def OPDs_map_from_cmd(self, OG:float|np.ndarray = None):
        """Reconstruct OPD maps from PAPYRUS modal commands."""
        if OG is None:
            rec_cmd = self.rec_cmd_with_og.copy()
        else:
            rec_cmd = self.rec_cmd/OG

        OPD_map = (self.dm.modes @ (self.M2C @ rec_cmd.T))*self.tel.pupil.reshape(-1,1)
        self.OPDs_from_cmd = OPD_map.T.reshape(-1, self.tel.pupil.shape[0], self.tel.pupil.shape[0])
        return self.OPDs_from_cmd
    
    def compute_projectors(self, keep_pupil_only = False):
        """
        Compute projection matrices onto modal commands and influence functions.

        
        """
        self._projector_keep_pupil_status = keep_pupil_only
        self.proj_M2C = self._compute_proj_dm(self.M2C, self.tel, self.dm,keep_pupil_only)
        self.proj_IF = self._compute_proj_dm(np.identity(self.dm.modes.shape[-1]), self.tel, self.dm,keep_pupil_only)
        
    def project_on_OZIRIIS(self, modes, proj=None):
        """Project first-stage commands onto an OZIRIIS modal basis."""
        if proj is None:
            proj, modes = self._compute_proj_dm(modes, self.tel_ozi, self.dm_ozi, return_modes=True)
        rec_modes = np.zeros((self.rec_cmd_with_og.shape[0], modes.shape[-1]))
        for i in tqdm.tqdm(range(self.rec_cmd_with_og.shape[0])):
            opds = self.dm.modes @ (self.M2C @ self.rec_cmd_with_og[i])
            opds_ = np.zeros_like(self.tel_ozi.pupil).astype(np.float32)
            opds_[self.tel_ozi.pupil] = opds[self.tel.pupil.ravel()]
            rec_modes[i] = proj @ opds_.ravel()
        return (rec_modes, modes, proj)

    def PSD_cmd_IFs(self, npsg=None):
        """
        Compute PSDs of the reconstructed command vectors in actuator space.

        Parameters
        ----------
        npsg : int, optional
            Requested segment length for PSD estimation.
        """
        if npsg is None:
            npsg = self.time.size
        self.psd_cmd_IFs = self.psd(self.time, (self.proj_IF@self.OPDs_from_cmd.reshape(self.OPDs_from_cmd.shape[0],-1).T).T, nperseg=npsg)

    def PSD_cmd_modal(self, npsg=None):
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
        self.psd_cmd_modal = self.psd(self.time, (self.proj_M2C@self.OPDs_from_cmd.reshape(self.OPDs_from_cmd.shape[0],-1).T).T, nperseg=npsg)
        
    def correct_DM_model_issue(self, IM_bench,*, IM_synth=None, rec_synth=None):
        if (IM_synth is None) and (rec_synth is None):
            raise ValueError(
                "IM_synth or rec_synth must be provided"
            )
        if (IM_synth is not None) == (rec_synth is not None):
            logger.warning(
                "IM_synth and rec_synth both provided, ising rec_synth by default"
            )
        if rec_synth is not None:
            factor =np.diag((rec_synth@IM_bench))[2:50].mean()
        else:
            from OOPAO.calibration.CalibrationVault import CalibrationVault
            rec_synth = CalibrationVault(IM_synth).M
            factor =np.diag((rec_synth@IM_bench))[2:50].mean()
        try : 
            self.initial_rec_cmd = self.initial_rec_cmd.copy()
            print('Bench rec_cmd already saved in the variable initial_rec_cmd')
        except:
            self.initial_rec_cmd = self.rec_cmd.copy()
            print('Bench rec_cmd is now saved in the variable initial_rec_cmd')
        self.rec_cmd = factor*self.initial_rec_cmd.copy()
        return self.rec_cmd 
    def compute_all_PSD(self, npsg=None):
        """
        Compute all available PSD products for reconstructed phases and commands.

        Parameters
        ----------
        npsg : int, optional
            Requested segment length for PSD estimation.
        """
        self.PSD_cmd_IFs(npsg)
        self.PSD_cmd_modal(npsg)
        
    def modal_PSD(self, projected_phase):
        """Return variance per projected mode after mean removal."""
        projected_phase -= projected_phase.mean(axis=0)
        modal_psd = projected_phase.var(axis=0)
        return modal_psd
    
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
        final_mask = self._crop_pupil()
        if OPDs.ndim == 3:
            OPDs_crop = np.zeros((OPDs.shape[0], final_mask.shape[0], final_mask.shape[1])).astype(np.float32)
            OPDs_crop[:, final_mask] = OPDs[:, pupil]
        else:
            OPDs_crop = np.zeros_like(final_mask).astype(np.float32)
            OPDs_crop[final_mask] = OPDs[pupil]
        return final_mask, OPDs_crop
    
    def simulate_PSF(self, imaging_wvl=True, sampling=None, ncpa=None, parallel=True, parall_njob=4, chunk_size=100, img_size=100):
        """Simulate PSF images from reconstructed OPD maps."""
        if ncpa is not None:
            if ncpa.size != self.M2C.shape[-1]:
                raise ValueError('ncpa must be an array of the size of the number of modes in the M2C')
        
        opd_ncpa = self._compute_ncpa_opd(ncpa)
        pupil = self.tel.pupil
        OPDs = self.OPDs_from_cmd
        if sampling is None:
            sampling = np.copy(self.psf_sampling)
        if parallel:
            if imaging_wvl:
                wvl = self.src_imaging.wavelength
            else:
                wvl = self.src.wavelength
            print(wvl)
            n_frames = OPDs.shape[0]
            opd_ncpa *= 2 * np.pi / wvl
 
            chunks = [OPDs[i:i + chunk_size] * 2 * np.pi / wvl for i in range(0, n_frames, chunk_size)]
            gen = Parallel(n_jobs=parall_njob, prefer='processes', return_as='generator')((delayed(_simulate_psf_chunk_worker)(chunk, pupil, sampling, img_size, opd_ncpa) for chunk in chunks))
            psf_chunks = list(tqdm.tqdm(gen, total=len(chunks), desc='PSF simulation'))
            self.simulated_psf = np.concatenate(psf_chunks, axis=0).astype(np.float32)
        else:
            _, _, _, _, _, _, Detector = _import_oopao_symbols()
            self.cam = Detector(psf_sampling=sampling * self.submasks[0].sum() ** (1 / 2) / self.tel.pupil.shape[0])
            if imaging_wvl:
                self.src_imaging * self.tel
            else:
                self.src * self.tel
            self.simulated_psf = []
            for i in tqdm.tqdm(range(OPDs.shape[0]), desc='PSF simulation'):
                self.tel.OPD = OPDs[i] + opd_ncpa
                self.tel * self.cam
                self.simulated_psf.append(self.cam.frame.copy().astype(np.float32))
            self.simulated_psf = np.asarray(self.simulated_psf, dtype=np.float32)
        
    def _compute_ncpa_opd(self, ncpa=None):
        """Compute the static NCPA OPD map from modal coefficients."""
        self.tel.OPD = self.tel.OPD * 0
        self.dm_ozi.coefs = 0
        if ncpa is not None:
            self.dm_ozi.coefs = self.M2C @ ncpa
        else:
            self.dm_ozi.coefs = self.dm_ozi.coefs * 0
        self.tel * self.dm_ozi
        return self.tel.OPD.copy().astype(np.float32)
    
    def compute_lost_frames(self):
        """Detect frame counter discontinuities in the telemetry sequence."""
        self.lost_frames = np.diff(self.frame_count) - 1
        self.where_lost_frames = np.append(False, self.lost_frames > 0)
    
    def _compute_proj_dm(self, modal_basis, tel, dm, return_modes=False, keep_pupil_only = False, filtering = None):
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
        inv =  (np.diag(1.0 / diag) @ modes.T).astype(np.float32)
        if return_modes:
            inv, modes
        else:
            return inv
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
        

        modes_flat = modes.reshape((tel.resolution**2, modes.shape[-1]))
        modes_flat = modes_flat / std[None, :]

        cov_modes = modes_flat.T @ modes_flat
        diag = np.diag(cov_modes)
        diag = np.where(np.abs(diag) < 1e-30, 1.0, diag)
    
        return (np.diag(1.0 / diag) @ modes_flat.T).astype(np.float32)
    
    def project_OPDs(self, remove_mean = False):
        """
        Project reconstructed OPD maps onto influence functions and modes.

        Raises
        ------
        RuntimeError
            If phase reconstruction has not been computed yet.
        """

        if self._projector_keep_pupil_status:
    
        
            if remove_mean:
                mean = self.OPDs[:,self.tel.pupil==1].mean(axis=0)
            else:
                mean = 0
            self.OPDs_on_IFs = self._project_phase(self.proj_IF, self.OPDs[:,self.tel.pupil==1]-mean)
            self.OPDs_on_modes = self._project_phase(self.proj_M2C, self.OPDs[:,self.tel.pupil==1]-mean)
            self.has_projected_phase = True
        else: 
            if remove_mean:
                mean = self.OPDs.mean(axis=0)
            else:
                mean = 0
            self.OPDs_on_IFs = self._project_phase(self.proj_IF, self.OPDs-mean)
            self.OPDs_on_modes = self._project_phase(self.proj_M2C, self.OPDs-mean)
            self.has_projected_phase = True
            
    
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
        projected_phase = projector @ phase.reshape(phase.shape[0],-1).T
        return projected_phase.astype(np.float32).T

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
    def _get_psf_for_analysis(self):
        """Return the simulated PSF image used for PSF analysis.
    
        If ``self.simulated_psf`` is a cube, the long-exposure PSF is estimated as
        the temporal average of the simulated frames.
        """
        if not hasattr(self, "simulated_psf"):
            raise RuntimeError("Must run simulate_PSF before PSF analysis.")
    
        if self.simulated_psf.ndim == 3:
            return self.simulated_psf.mean(axis=0)
    
        if self.simulated_psf.ndim == 2:
            return self.simulated_psf
    
        raise ValueError(
            "simulated_psf must be either a 2D image or a 3D temporal cube."
        )
    
    
    def _safe_store_psf_analysis_result(self, name, value):
        """Store a PSF analysis result without overwriting existing attributes."""
        if not hasattr(self, "psf_analysis_results"):
            self.psf_analysis_results = {}
    
        self.psf_analysis_results[name] = value
    
        if not hasattr(self, name):
            setattr(self, name, value)
        else:
            safe_name = f"simulated_{name}"
            setattr(self, safe_name, value)
    def _choose_file(self):
        from qtpy.QtWidgets import QApplication, QFileDialog
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        file_path, _ = QFileDialog.getOpenFileName(None, 'Select OZIRIIS linearity .npz file', '', 'NPZ files (*.npz);;All files (*.*)')
        return file_path

    def __matmul__(self, obj):
        """Project compatible telemetry products using the ``@`` operator."""
        if obj.tag == 'ozi':
            self.ozi_if, self.ozi_if_modes, self.ozi_proj_if = self.project_on_OZIRIIS(np.identity(97))
            self.ozi_modes, self.ozi_KL_basis, self.ozi_proj_modes = self.project_on_OZIRIIS(obj.M2C)
            print('First stage command projected on second stage')
        else:
            raise ValueError('Entered object not an OZItele')
    def __mul__(self, obj):
        """Analyze the simulated PSF using a PSFTele-like analyzer.
    
        Parameters
        ----------
        obj : object
            PSF telemetry analyzer. It must expose an ``analyze_psf`` method.
    
        Returns
        -------
        OZITele
            Current instance, updated with PSF analysis results.
    
        Raises
        ------
        TypeError
            If the provided object cannot analyze PSFs.
        RuntimeError
            If no simulated PSF is available.
        """
        if not obj.tag == 'psf':
            raise ValueError('The multiplied object must be a PSF object')
    
        psf = self._get_psf_for_analysis()
    
        results = obj.analyze_psf(
            psf=psf,
            sampling=obj.sampling,
        )
    
        for name, value in results.items():
            self._safe_store_psf_analysis_result(name, value)
    
        print("Simulated PSF analyzed with PSFTele")
    
        return self