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
from parallel_utils import _reconstruct_phase_worker, _import_oopao_symbols, _simulate_psf_chunk_worker
from OOPAO.Zernike import Zernike


class OZITele:
    """Analyze OZIRIIS telemetry, reconstruct OPDs, project phases, and compute PSD products."""

    def __init__(self, tele_path: str=None, is_onsky: bool=True, CNN=False, narrow_band=False, temporal_crop=None, is_cl=True, extract_values=True, threshold_for_pupil_selection = None):
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
        self.temporal_crop = slice(None) if temporal_crop is None else slice(temporal_crop[0], temporal_crop[1])
        if tele_path is None:
            tele_path = self._choose_file()
            if tele_path:
                print('Selected file:', tele_path)
            else:
                raise ValueError('No files selected')
        self.is_nb = narrow_band
        self.tele_path = tele_path
        data = np.load(self.tele_path, allow_pickle=True)
        self.loop_gain = data.item()['lpGain'][0][0]
        self.loop_leak = data.item()['lpLeak'][0][0]
        self.controlled_modes = data.item()['nmodes'][0][0]
        self.off_mask = data.item()['validPixels'].astype(np.float32)
        self.img_raw = data.item()['CRED2Cube'].astype(np.float32)[self.temporal_crop]
        self.dark = data.item()['credDark'].astype(np.float32)
        self.dmshape = data.item()['dmCmdCube'].astype(np.float32)[self.temporal_crop][...,0]
        self.dmCL = data.item()['dmCLCube'].astype(np.float32)[self.temporal_crop][...,0]
        self.reconstructed_cube = data.item()['slavedreconsCube'].astype(np.float32)[self.temporal_crop]
        self.full_reconstructed_cube = data.item()['FullreconsCube'].astype(np.float32)[self.temporal_crop]
        try:
            self.is_onsky = np.bool_(data.item()['is_onsky'][0][0])
            logger.info('Onsky status found in datas, not taking into account entered value')
        except:
            logger.info('Onsky status not found in datas, taking into account entered value')
            self.is_onsky = is_onsky
        if self.is_onsky:
            self.M2C = data.item()['m2c_sky']
            self.IM = data.item()['IntMat_sky']
        else:
            self.M2C = data.item()['m2c']
            self.IM = data.item()['IntMat']
        self.C2M = np.linalg.pinv(self.M2C)
        self.psf_sampling = 2.8
        self.frame_count = self.img_raw[:, 0, 0]
        try:
            self.is_cl = np.bool_(data.item()['loop_status'][0][0])
            logger.info('Closed loop status found in datas, not taking into account entered value')
        except:
            logger.info('Closed loop status not found in datas, taking into account entered value')
            self.is_cl = is_cl
        try:
            self.reconstruction_method = data.item()['reconstruction_method'].astype(np.float32)[0][0]
            logger.info('Reconstruction method found in datas, not taking into account entered value')
            if self.reconstruction_method == 0:
                self.CNN = False
            else:
                self.CNN = True
        except:
            logger.warn('Reconstruction method not found in datas, taking into account entered value')
            self.CNN = CNN
        if self.CNN:
            self.rec_cmd = self.reconstructed_cube.copy()
        else:
            self.rec_cmd = self.full_reconstructed_cube.copy()
        self.rec_cmd = self.rec_cmd.reshape(self.rec_cmd.shape[0], 97)
        self.rec_cmd_modal = np.zeros((self.rec_cmd.shape[0], self.C2M.shape[0]))
        for i in range(self.rec_cmd.shape[0]):
            self.rec_cmd_modal[i, :] = self.C2M @ self.rec_cmd[i]
        self.ts_dm = data.item()['timeStampcredCube'][self.temporal_crop]
        self.img = self.img_raw - self.dark
        self.extract_values = extract_values
        self.img[:, :1, :] = 0
        self.img /= self.img.sum(axis=(1, 2))[:, None, None]
        self.ts = data.item()['timeStampcredCube'][self.temporal_crop]
        self.t0 = self.ts[0]
        self.time = np.array([(t - self.t0).total_seconds() for t in self.ts], dtype=float)
        if threshold_for_pupil_selection is None:
            use_tuner = True
            threshold_for_pupil_selection = 0.25
        else:
            
            use_tuner = False
        if extract_values:
            if self.is_onsky:
                self.positions_calib = [np.array([35, 20, 125, 110]), np.array([127, 19, 216, 109])]
                self.initial_positions, self.initial_pupils, self.initial_submasks, self.global_masks = reference_intensities(self.off_mask, crop = threshold_for_pupil_selection, use_tuner=use_tuner )
                minr, minc, maxr, maxc = self.positions_calib[0]
                self.initial_submasks[0] = self.global_masks[0][minr:maxr, minc:maxc]
                minr, minc, maxr, maxc = self.positions_calib[1]
                self.initial_submasks[1] = self.global_masks[1][minr:maxr, minc:maxc]
                self.initial_pupils[0] = np.zeros_like(self.initial_submasks[0]).astype(np.float32)
                self.initial_pupils[1] = np.zeros_like(self.initial_submasks[1]).astype(np.float32)
                self.initial_pupils[0][self.initial_submasks[0]] = self.off_mask[self.global_masks[0]]
                self.initial_pupils[1][self.initial_submasks[1]] = self.off_mask[self.global_masks[1]]
                self.submasks = [None, None]
                self.pupils = [None, None]
                pupil2 = self._rescale_matrix(self.initial_pupils[1], self.initial_pupils[0].shape[0], self.initial_pupils[0].shape[1])
                self.pupils[1], _ = self._pad_to_square(pupil2)
                self.pupils[0], _ = self._pad_to_square(self.initial_pupils[0])
                self.submasks[1], _ = self._pad_to_square(self.initial_submasks[0])
                self.submasks[0] = self.submasks[1]
            else:
                self.positions_calib = [np.array([35, 20, 125, 110]), np.array([127, 19, 216, 109])]
                self.initial_positions, self.initial_pupils, self.initial_submasks, self.global_masks = reference_intensities(self.off_mask, crop = threshold_for_pupil_selection, use_tuner=use_tuner )
                self.submasks = [None, None]
                self.pupils = [None, None]
                pupil2 = self._rescale_matrix(self.initial_pupils[1], self.initial_pupils[0].shape[0], self.initial_pupils[0].shape[1])
                self.pupils[1], _ = self._pad_to_square(pupil2)
                self.pupils[0], _ = self._pad_to_square(self.initial_pupils[0])
                self.submasks[1], _ = self._pad_to_square(self.initial_submasks[0])
                self.submasks[0] = self.submasks[1]
            self._initialise_OOPAO_objects()
            self.compute_projectors()
            self.extract_Zimages()
            self.has_recontructed_phase = False
            self.has_projected_phase = False
            self.has_recompute_rec_cmd = False

    def _initialise_OOPAO_objects(self):
        """Initialise the OOPAO optical objects used by reconstruction and simulation."""
        Source, Telescope, ZWFS, ZWFS2, DeformableMirror, MisRegistration, Detector = _import_oopao_symbols()
        if self.is_onsky and ~self.is_nb:
            self.src1 = Source(optBand='H', magnitude=-2.5)
            self.src1.wavelength = 1.6e-06
            self.src1.bandwidth = 2e-07
            self.src2 = Source(optBand='H', magnitude=-2.5)
            self.src2.wavelength = 1.6e-06
            self.src2.bandwidth = 2e-07
        else:
            self.src1 = Source(optBand='H', magnitude=-2.5)
            self.src1.wavelength = 1.55e-06
            self.src1.bandwidth = 0
            self.src2 = Source(optBand='H', magnitude=-2.5)
            self.src2.wavelength = 1.55e-06
            self.src2.bandwidth = 0.0
        self.tel1 = Telescope(self.submasks[0].shape[0], 1.52, pupil=self.submasks[0])
        self.tel1.pupilReflectivity = np.sqrt(self.pupils[0]) * self.submasks[0]
        self.tel1.pupilReflectivity[~np.isfinite(self.tel1.pupilReflectivity)] = 0
        self.src1 * self.tel1
        self.tel2 = Telescope(self.submasks[1].shape[0], 1.52, pupil=self.submasks[1])
        self.tel2.pupilReflectivity = np.sqrt(self.pupils[1]) * self.submasks[1]
        self.tel2.pupilReflectivity[~np.isfinite(self.tel2.pupilReflectivity)] = 0
        self.src2 * self.tel2
        self.vzwfs = self._build_vzwfs_class()
        self.zwfs1 = self.vzwfs.zwfs1
        self.zwfs2 = self.vzwfs.zwfs2
        self.cam = Detector(psf_sampling=self.psf_sampling)
        param = np.load(str(HERE) + '\dm_second_stage_misreg_dict.npy', allow_pickle=True).item()
        m = MisRegistration(param)
        self.dm1 = DeformableMirror(telescope=self.tel1, nSubap=10, mechCoupling=0.35, print_dm_properties=False, pitch=0.11, misReg=m)
        self.dm2 = DeformableMirror(telescope=self.tel2, nSubap=10, mechCoupling=0.35, print_dm_properties=False, pitch=0.11, misReg=m)
        # if_path = os.path.join(HERE, 'IF_vZWFS.npy')
        self.IF = np.load(os.path.join(HERE, 'IF_dm2.npy')) * 1e-06
        print(self.IF.shape)
        amplitude_mean = np.ptp(self.IF, axis=0)
        self.dm1.modes *= amplitude_mean / np.ptp(self.dm1.modes)
        self.dm2.modes *= amplitude_mean / np.ptp(self.dm1.modes)
        self.src_imaging = Source(optBand='IR1310', magnitude=-2.5)

    def _build_vzwfs_class(self):
        """Build the paired ZWFS object using the current telescope geometry."""
        Source, Telescope, ZWFS, ZWFS2, DeformableMirror, MisRegistration, Detector = _import_oopao_symbols()
        if self.is_onsky:
            diam = 1.96
        else:
            diam = 2.14
        zwfs1 = ZWFS(self.tel1, diameter=diam, phase_shift=0.33, zpf=30, phase_shift_unit='pi')
        zwfs2 = ZWFS(self.tel2, diameter=diam, phase_shift=-0.74, zpf=30, phase_shift_unit='pi')
        return ZWFS2(ZWFS1=zwfs1, ZWFS2=zwfs2)

    def _export_reconstruction_setup(self):
        """Return the reconstruction setup required by worker processes."""
    
        if self.is_onsky:
            diam = 1.96
        else:
            diam = 2.14
    
        return {
            "is_onsky": self.is_onsky,
            "is_nb": self.is_nb,
    
            "submask0": self.submasks[0].copy(),
            "submask1": self.submasks[1].copy(),
            "pupil0": self.pupils[0].copy(),
            "pupil1": self.pupils[1].copy(),
    
            "diam": diam,
            "phase_shift_1": 0.33,
            "phase_shift_2": -0.74,
            "zpf": 30,
            "phase_shift_unit": "pi",
    
            # copie explicite de ce que contient réellement le non-parallèle
            "src1_optBand": self.src1.optBand if hasattr(self.src1, "optBand") else "H",
            "src2_optBand": self.src2.optBand if hasattr(self.src2, "optBand") else "H",
            "src1_wavelength": self.src1.wavelength,
            "src2_wavelength": self.src2.wavelength,
            "src1_bandwidth": self.src1.bandwidth,
            "src2_bandwidth": self.src2.bandwidth,
        }
    def _export_psf_setup(self, img_wvl=False):
        """Return the minimal PSF simulation setup required by worker processes."""
        return {'is_onsky': self.is_onsky, 'imaging_wvl': img_wvl, 'is_nb': self.is_nb, 'submask0': self.submasks[0], 'pupil0': self.pupils[0], 'psf_sampling': self.psf_sampling}
    
    # def _compute_proj_dm(self, modal_basis, tel, dm):
    #     """Compute a diagonal-normalized projection matrix from DM-generated modes."""
    #     dm.coefs = modal_basis
    #     tel * dm
    #     modes = tel.OPD.copy()
    #     modes = modes.reshape((tel.resolution ** 2, modes.shape[-1])) / tel.OPD[tel.pupil, :].std(axis=0)
        
    #     return np.linalg.pinv(modes)
    # def _compute_proj_OPDs(self, modes, tel):
    #     """Compute a diagonal-normalized projection matrix from OPD modes."""
    #     modes = modes.reshape((tel.resolution ** 2, modes.shape[-1])) / modes[tel.pupil, :].std(axis=0)
    #     cov_modes = modes.T @ modes
    #     return np.linalg.pinv(modes)
    
    # def _compute_proj_dm(self, modal_basis, tel, dm, keep_pupil_only = False, filtering = None):
    #     """Compute a diagonal-normalized projection matrix from DM-generated modes."""
    #     dm.coefs = modal_basis
    #     tel * dm
    #     modes = tel.OPD.copy()
    #     if keep_pupil_only:
    #         modes = modes[tel.pupil, :] / modes[tel.pupil, :].std(axis=0)
    #     else:
    #         modes = modes.reshape((tel.resolution ** 2, modes.shape[-1]))/ modes[tel.pupil, :].std(axis=0)
    #     return np.linalg.pinv(modes, rcond=filtering)
    # def _compute_proj_OPDs(self, modes, tel, keep_pupil_only = False, filtering = None):
    #     """Compute a diagonal-normalized projection matrix from OPD modes."""
    #     if keep_pupil_only:
    #         modes = modes[tel.pupil, :] / modes[tel.pupil, :].std(axis=0)
    #     else:
    #         modes = modes.reshape((tel.resolution ** 2, modes.shape[-1]))/ modes[tel.pupil, :].std(axis=0)
    #     return np.linalg.pinv(modes, rcond=filtering)
    
    
    def compute_projectors(self, keep_pupil_only = False):
        """
        Compute projection matrices onto modal commands and influence functions.

        
        """
        self._projector_keep_pupil_status = keep_pupil_only
        self.proj_M2C = self._compute_proj_dm(self.M2C, self.tel1, self.dm1,keep_pupil_only)
        self.proj_IF = self._compute_proj_dm(np.identity(self.dm1.modes.shape[-1]), self.tel1, self.dm1,keep_pupil_only, filtering=1/30)
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
    def extract_Zimages(self):
        """
        Extract and format the two ZWFS image streams from the raw image cube.

        The images are cropped from the valid-pixel regions, rescaled when
        needed, and padded to square arrays so they match the internal optical
        model geometry.
        """
        images_z1 = np.zeros((self.img.shape[0], self.initial_submasks[0].shape[0], self.initial_submasks[0].shape[1]))
        images_z2 = np.zeros((self.img.shape[0], self.initial_submasks[1].shape[0], self.initial_submasks[1].shape[1]))
        images_z1[:, self.initial_submasks[0]] = self.img[:, self.global_masks[0]]
        images_z2[:, self.initial_submasks[1]] = self.img[:, self.global_masks[1]]
        self.img_ZWFS2 = []
        logger.info('Extracting the signal of the ZWFSs')
        for i in tqdm.tqdm(range(images_z2.shape[0])):
            self.img_ZWFS2.append(self._rescale_matrix(images_z2[i], self.pupils[0].shape[0], self.pupils[0].shape[1]))
        self.img_ZWFS2 = np.array(self.img_ZWFS2)
        self.img_ZWFS2, _ = self._pad_to_square(self.img_ZWFS2)
        self.img_ZWFS1, _ = self._pad_to_square(images_z1)

    def compute_lost_frames(self):
        """Detect frame counter discontinuities in the telemetry sequence."""
        self.lost_frames = np.diff(self.frame_count) - 1
        self.where_lost_frames = np.append(False, self.lost_frames > 0)


    # def smooth_lost_frames(self):
    #     """Compute lost-frame metadata before any optional smoothing workflow."""
    #     self.compute_lost_frames()
    #     return None

    def compute_Zernike_basis(self, nmodes=30):
        """Compute a Zernike basis and its corresponding OPD projector."""
        Zer_basis = Zernike(self.tel1, J=nmodes)
        Zer_basis.computeZernike(self.tel1)
        self.Zer_modes = Zer_basis.modesFullRes.copy()
        self.proj_Zer = self._compute_proj_OPDs(self.Zer_modes, self.tel1)

    def compute_synth_IM(self, stroke_nm=12):
        """Compute a synthetic interaction matrix from DM modal strokes."""
        M2Phase = self.IF * 1000000000.0 @ self.M2C
        std_phase = np.std(M2Phase, axis=0)
        eps = 1e-12
        M2C = self.M2C.copy()
        stroke = stroke_nm / std_phase
        IM1 = []
        for i in range(M2C.shape[-1]):
            self.dm1.coefs = stroke[i] * M2C[:, i]
            self.tel1 * self.dm1
            self.tel1 * self.zwfs1
            img1_pos = self.zwfs1.img_ZWFS
            self.dm1.coefs = -stroke[i] * M2C[:, i]
            self.tel1 * self.dm1
            self.tel1 * self.zwfs1
            IM1.append((img1_pos - self.zwfs1.img_ZWFS) / (2 * stroke[i]))
        return np.array(IM1)[:, self.submasks[0]].T

    def compute_SR(self, wavelength=None, ncpa=None):
        """Estimate Strehl ratio statistics from reconstructed phase variance."""
        opd_ncpa = self._compute_ncpa_opd(ncpa)
        if wavelength is None:
            wavelength = self.src1.wavelength
        phase_var = (self.phase[:, self.tel1.pupil == 1] + opd_ncpa[self.tel1.pupil == 1] * 2 * np.pi / self.src1.wavelength).var(axis=1) * (self.src1.wavelength / wavelength) ** 2
        SR = np.exp(-phase_var)
        SR_mean = np.exp(-phase_var.mean())
        print(f'At {wavelength * 1000000000.0:.0f} nm, the average SR is about {SR_mean * 100:.1f}%')
        return (SR, SR_mean)

    def reconstruct_phase(self, im1, im2, method='atan', damping=0.5, iteration=10, modes_filtering=False, modal_basis=None, nmodes=None):
        """
        Reconstruct a single phase map from a pair of ZWFS images.

        Parameters
        ----------
        im1 : ndarray
            Image from the first ZWFS channel.
        im2 : ndarray
            Image from the second ZWFS channel.
        method : str, optional
            Reconstruction method passed to the underlying reconstructor.
        damping : float, optional
            Damping factor applied during the iterative reconstruction.
        iteration : int, optional
            Requested number of iterations.
        Returns
        -------
        ndarray
            Reconstructed phase map.
        """
        self.vzwfs.zwfs1.img_ZWFS = im1
        self.vzwfs.zwfs2.img_ZWFS = im2
        if modes_filtering:
            if modal_basis is None:
                raise AttributeError("modal_basis must be provided when modes_filtering=True. Expected 'KL', 'Zer', or an array of shape (im1.shape[0], im1.shape[1], N).")
            if isinstance(modal_basis, str):
                allowed_basis = {'KL', 'Zer'}
                if nmodes is None:
                    nmodes = self.M2C.shape[-1]
                if modal_basis not in allowed_basis:
                    raise ValueError(f'Invalid modal_basis string: {modal_basis!r}. Expected one of {sorted(allowed_basis)}.')
                elif modal_basis == 'KL':
                    modal_basis = self._compute_KL_basis_from_cmd(self.M2C, self.tel1, self.dm1)
                    self.KL_basis = modal_basis.copy()
                elif modal_basis == 'Zer':
                    self.compute_Zernike_basis(nmodes)
                    modal_basis = self.Zer_modes.copy()
            elif isinstance(modal_basis, np.ndarray):
                if modal_basis.ndim != 3:
                    raise ValueError(f'modal_basis array must be 3-dimensional with shape (im1.shape[0], im1.shape[1], N). Got shape {modal_basis.shape}.')
                expected_shape_2d = im1.shape[:2]
                if modal_basis.shape[:2] != expected_shape_2d:
                    raise ValueError(f'Invalid modal_basis spatial shape. Expected first two dimensions {expected_shape_2d}, got {modal_basis.shape[:2]}.')
                if nmodes is not None and modal_basis.shape[2] < nmodes:
                    raise ValueError(f'modal_basis contains only {modal_basis.shape[2]} modes, but nmodes={nmodes} was requested.')
            else:
                raise TypeError(f"modal_basis must be either a string or a numpy.ndarray when modes_filtering=True. Expected 'KL', 'Zer', or an array of shape (im1.shape[0], im1.shape[1], N). Got {type(modal_basis).__name__}.")
            if nmodes is None:
                nmodes = modal_basis.shape[-1]
            return self.vzwfs.reconstructor(iteration=iteration, damping_iteration=damping, reconstructor=method, filter_modes=modes_filtering, modal_basis=modal_basis[..., :nmodes])
        else:
            return self.vzwfs.reconstructor(iteration=iteration, damping_iteration=damping, reconstructor=method)

    def reconstruct_all_phase(self, method='atan', iteration=10, damping=0.5, modes_filtering=False, modal_basis=None, nmodes=None, parallel=True, parall_njob=4):
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
            if modes_filtering:
                if modal_basis is None:
                    raise AttributeError("modal_basis must be provided when modes_filtering=True. Expected 'KL', 'Zer', or an array of shape (im1.shape[0], im1.shape[1], N).")
                if nmodes is None:
                    nmodes = self.M2C.shape[-1]
                if isinstance(modal_basis, str):
                    allowed_basis = {'KL', 'Zer'}
                    if modal_basis not in allowed_basis:
                        raise ValueError(f'Invalid modal_basis string: {modal_basis!r}. Expected one of {sorted(allowed_basis)}.')
                    elif modal_basis == 'KL':
                        modal_basis = self._compute_KL_basis_from_cmd(self.M2C, self.tel1, self.dm1)
                        self.KL_basis = modal_basis.copy()
                    elif modal_basis == 'Zer':
                        self.compute_Zernike_basis(nmodes)
                        modal_basis = self.Zer_modes.copy()
                elif isinstance(modal_basis, np.ndarray):
                    if modal_basis.ndim != 3:
                        raise ValueError(f'modal_basis array must be 3-dimensional with shape (im1.shape[0], im1.shape[1], N). Got shape {modal_basis.shape}.')
                    expected_shape_2d = self.submasks[0].shape[:2]
                    if modal_basis.shape[:2] != expected_shape_2d:
                        raise ValueError(f'Invalid modal_basis spatial shape. Expected first two dimensions {expected_shape_2d}, got {modal_basis.shape[:2]}.')
                    if nmodes is None:
                        nmodes = modal_basis.shape[-1]
                    if nmodes is not None and modal_basis.shape[2] < nmodes:
                        raise ValueError(f'modal_basis contains only {modal_basis.shape[2]} modes, but nmodes={nmodes} was requested.')
                else:
                    raise TypeError(f"modal_basis must be either a string or a numpy.ndarray when modes_filtering=True. Expected 'KL', 'Zer', or an array of shape (im1.shape[0], im1.shape[1], N). Got {type(modal_basis).__name__}.")
                gen = Parallel(n_jobs=parall_njob, prefer='processes', return_as='generator')((delayed(_reconstruct_phase_worker)(self.img_ZWFS1[i], self.img_ZWFS2[i], setup, method, damping, iteration, modes_filtering, modal_basis, nmodes) for i in range(n_frames)))
            else:
                gen = Parallel(n_jobs=parall_njob, prefer='processes', return_as='generator')((delayed(_reconstruct_phase_worker)(self.img_ZWFS1[i], self.img_ZWFS2[i], setup, method, damping, iteration) for i in range(n_frames)))
            self.phase = np.asarray(list(tqdm.tqdm(gen, total=n_frames, desc=f'Phase reconstruction ({method})')), dtype=np.float32)
        else:
            self.phase = np.zeros((self.img_ZWFS1.shape[0], self.tel1.pupil.shape[0], self.tel1.pupil.shape[1])).astype(np.float32)
            logger.info(f'Computing phase for each frame using {method} reconstruction')
            for i in tqdm.tqdm(range(self.img_ZWFS1.shape[0])):
                self.phase[i] = self.reconstruct_phase(self.img_ZWFS1[i], self.img_ZWFS2[i], method, damping, iteration).astype(np.float32)
        self._phase2OPD()
        self.has_recontructed_phase = True

    def _phase2OPD(self, phase=None):
        """Convert reconstructed phase maps into OPD maps using the source wavelength."""
        if phase is None:
            phase = self.phase
        self.OPDs = (phase / (2 * np.pi) * self.src1.wavelength).astype(np.float32)

    def OPDs_map_from_cmd(self):
        """Reconstruct OPD maps from PAPYRUS modal commands."""
        OPD_map = self.dm1.modes @ self.rec_cmd.T
        self.OPDs_from_cmd = OPD_map.T.reshape(-1, self.tel1.pupil.shape[0], self.tel1.pupil.shape[0])
        return self.OPDs_from_cmd

    def filter_OPDs(self, OPD=None, nmodes=None, modal_basis='KL'):
        """Filter OPD maps on the requested modal basis."""
        if nmodes is None:
            nmodes = self.M2C.shape[-1]
        if OPD is None:
            OPD = self.OPDs.copy()
        if modal_basis == 'KL':
            self.dm1.coefs = self.M2C
            self.tel1 * self.dm1
            modes = self.tel1.OPD.copy()
            modes = modes / self.tel1.OPD[self.tel1.pupil, :].std(axis=0)
            self.dm1.coefs = 0
            self.tel1 * self.dm1
            OPDs_on_modes = self._project_phase(self.proj_M2C, OPD)
            filtered_OPD = 0
            for i in tqdm.tqdm(range(nmodes)):
                filtered_OPD += OPDs_on_modes[:, None, None, i] * modes[None, :, :, i]
        elif modal_basis == 'IF':
            filtered_OPD = 0
            IF_inv = np.linalg.pinv(self.dm1.modes)
            filtered_OPD = (self.dm1.modes @ (IF_inv @ OPD.reshape(self.OPDs.shape[0], -1).T)).T.reshape(OPD.shape[0], OPD.shape[1], OPD.shape[2])
        elif modal_basis == 'Zer':
            opd_on_zer = self._project_phase(self.proj_Zer, OPD)
            filtered_OPD = (opd_on_zer[:, None, None, :nmodes] * self.Zer_modes[None, :, :, :nmodes]).sum(axis=-1)
        return filtered_OPD * (self.tel1.pupil == 1)

    def project_OPDs(self, remove_mean = False, only_ctrl_modes = False):
        """
        Project reconstructed OPD maps onto influence functions and modes.

        Raises
        ------
        RuntimeError
            If phase reconstruction has not been computed yet.
        """

        if self.has_recontructed_phase:
            if only_ctrl_modes:
                OPDs = self.filter_OPDs(nmodes = self.controlled_modes)
            else:
                OPDs = self.OPDs
            if self._projector_keep_pupil_status:
        
            
                if remove_mean:
                    mean = self.OPDs[:,self.tel1.pupil==1].mean(axis=0)
                else:
                    mean = 0
                self.OPDs_on_IFs = self._project_phase(self.proj_IF, OPDs[:,self.tel1.pupil==1]-mean)
                self.OPDs_on_modes = self._project_phase(self.proj_M2C, OPDs[:,self.tel1.pupil==1]-mean)
                self.has_projected_phase = True
            else: 
                if remove_mean:
                    mean = OPDs.mean(axis=0)
                else:
                    mean = 0
                self.OPDs_on_IFs = self._project_phase(self.proj_IF, OPDs-mean)
                self.OPDs_on_modes = self._project_phase(self.proj_M2C, OPDs-mean)
                self.has_projected_phase = True
        else:

            raise RuntimeError('Must compute phase before projection')
    def _crop_pupil(self, pupil = None):
        if pupil is None:
            pupil = self.tel1.pupil&self.tel2.pupil
        where_pupil = np.where(pupil)
        x_, x = where_pupil[0].min(),where_pupil[0].max()
        y_, y = where_pupil[1].min(),where_pupil[1].max()
        final_mask = pupil[x_:x+1,y_:y+1]
        return final_mask, (x_, x,y_, y)
    def _crop_opd(self, OPDs:np.ndarray= None):
        if OPDs is None:
            OPDs = self.OPDs
        final_mask, _ = self._crop_pupil()
        if OPDs.ndim == 3:
            OPDs_crop = np.zeros((OPDs.shape[0], final_mask.shape[0], final_mask.shape[1])).astype(np.float32)
            OPDs_crop[:, final_mask] = OPDs[:, self.tel1.pupil&self.tel2.pupil]
        else:
            OPDs_crop = np.zeros_like(final_mask).astype(np.float32)
            OPDs_crop[final_mask] = OPDs[self.tel1.pupil&self.tel2.pupil]
        return final_mask, OPDs_crop
        
    def project_on_PAPYRUS(self, projector, OPDs=None):
        """Project reconstructed second-stage OPDs onto a PAPYRUS projector."""
    
        if OPDs is None:
            OPDs = self.OPDs.copy()
    
        N = int(projector.shape[1] ** 0.5)
    
        projector_cube = projector.reshape(-1, N, N)
        projector_mat = projector_cube.reshape(projector_cube.shape[0], -1)
    
        pupil_1rst_full = np.sum(projector_cube, axis=0) != 0
    
        _, OPDs_crop = self._crop_opd(OPDs)
    
        pupil_1rst_crop, crop = self._crop_pupil(pupil_1rst_full)
        x_min, x_max, y_min, y_max = crop
    
        rec_modes = np.zeros((self.rec_cmd.shape[0], projector_cube.shape[0]), dtype=np.float32)
    
        for i in tqdm.tqdm(range(self.rec_cmd.shape[0])):
    
            phase_crop = self._rescale_matrix(
                OPDs_crop[i],
                pupil_1rst_crop.shape[0],
                pupil_1rst_crop.shape[1]
            )
    
            phase_full = np.pad(
                phase_crop,
                pad_width=(
                    (x_min, N - x_max - 1),
                    (y_min, N - y_max - 1)
                )
            )
            rec_modes[i] = projector_mat @ phase_full.ravel()
    
        return rec_modes
    def linear_reconstruction(self, nmodes=None, IM=None):
        """Recompute command vectors using a synthetic or provided interaction matrix."""
        if nmodes is None:
            nmodes = self.controlled_modes
        if IM is None:
            IM = self.compute_synth_IM()
        self.linear_reconstructor = self.M2C[:, :nmodes] @ np.linalg.pinv(IM[:, :nmodes])
        if ~self.has_recompute_rec_cmd:
            self.initial_rec_cmd = self.rec_cmd.copy()
        self.rec_cmd = (self.linear_reconstructor @ self.img_ZWFS1[:, self.submasks[0]].T).T
        self.has_recompute_rec_cmd = True

    

    def simulate_PSF(self, OPDs = None, imaging_wvl=True, sampling=None, ncpa=None, parallel=True, parall_njob=4, chunk_size=100, img_size=100):
        """Simulate PSF images from reconstructed OPD maps."""
        if ncpa is not None:
            if ncpa.size != self.M2C.shape[-1]:
                raise ValueError('ncpa must be an array of the size of the number of modes in the M2C')
        
        opd_ncpa = self._compute_ncpa_opd(ncpa)
        if OPDs is None:
            if not self.has_recontructed_phase:
                raise RuntimeError('Must compute phase before projection')
            OPDs = self.OPDs
        pupil, OPDs = self._crop_opd(OPDs)
        _, opd_ncpa = self._crop_opd(opd_ncpa)
        if sampling is None:
            sampling = np.copy(self.psf_sampling)
        else:
            logger.info('Changing sampling internal variable')
            self.psf_sampling = sampling
        if parallel:
            if imaging_wvl:
                wvl = self.src_imaging.wavelength
            else:
                wvl = self.src1.wavelength
            print(wvl)
            n_frames = OPDs.shape[0]
            opd_ncpa *= 2 * np.pi / wvl
            print(np.ptp(opd_ncpa))
            chunks = [OPDs[i:i + chunk_size] * 2 * np.pi / wvl for i in range(0, n_frames, chunk_size)]
            gen = Parallel(n_jobs=parall_njob, prefer='processes', return_as='generator')((delayed(_simulate_psf_chunk_worker)(chunk, pupil, sampling, img_size, opd_ncpa) for chunk in chunks))
            psf_chunks = list(tqdm.tqdm(gen, total=len(chunks), desc='PSF simulation'))
            self.simulated_psf = np.concatenate(psf_chunks, axis=0).astype(np.float32)
        else:
            _, _, _, _, _, _, Detector = _import_oopao_symbols()
            self.cam = Detector(psf_sampling=sampling * self.submasks[0].sum() ** (1 / 2) / self.tel1.pupil.shape[0])
            if imaging_wvl:
                self.src_imaging * self.tel1
            else:
                self.src1 * self.tel1
            self.simulated_psf = []
            for i in tqdm.tqdm(range(self.OPDs.shape[0]), desc='PSF simulation'):
                self.tel1.OPD = self.OPDs[i] + opd_ncpa
                self.tel1 * self.cam
                self.simulated_psf.append(self.cam.frame.copy().astype(np.float32))
            self.simulated_psf = np.asarray(self.simulated_psf, dtype=np.float32)

    def PSD_IFs(self, npsg=None):
        """
        Compute PSDs of reconstructed OPDs projected onto influence functions.

        Parameters
        ----------
        npsg : int, optional
            Requested segment length for PSD estimation. 

        Raises
        ------
        RuntimeError
            If OPDs have not been projected yet.
        """
        if npsg is None:
            npsg = self.time.size
        if self.has_projected_phase:
            logger.info('Computing IFs PSD')
            self.psd_IFs = self.psd(self.time, self.OPDs_on_IFs, nperseg=npsg)
        else:
            raise RuntimeError('Must project phase before PSDs')

    def PSD_modal(self, npsg=None):
        """
        Compute PSDs of reconstructed OPDs projected onto modal coefficients.

        Parameters
        ----------
        npsg : int, optional
            Requested segment length for PSD estimation.

        Raises
        ------
        RuntimeError
            If OPDs have not been projected yet.
        """
        if npsg is None:
            npsg = self.time.size
        if self.has_projected_phase:
            logger.info('Computing modal PSD')
            self.psd_modal = self.psd(self.time, self.OPDs_on_modes, nperseg=npsg)
        else:
            raise RuntimeError('Must project phase before PSDs')

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
        OPDs = self.OPDs_map_from_cmd()
        if self._projector_keep_pupil_status:
            self.psd_cmd_IFs = self.psd(self.time, self._project_phase(self.proj_IF, OPDs[:,self.tel1.pupil==1]), nperseg=npsg)
        else:
            self.psd_cmd_IFs = self.psd(self.time, self._project_phase(self.proj_IF, OPDs), nperseg=npsg)
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
        OPDs = self.OPDs_map_from_cmd()
        if self._projector_keep_pupil_status:
            self.psd_cmd_modal = self.psd(self.time, self._project_phase(self.proj_M2C, OPDs[:,self.tel1.pupil==1]), nperseg=npsg)
        else:
            self.psd_cmd_modal = self.psd(self.time, self._project_phase(self.proj_M2C, OPDs), nperseg=npsg)

    def compute_all_PSD(self, npsg=None):
        """
        Compute all available PSD products for reconstructed phases and commands.

        Parameters
        ----------
        npsg : int, optional
            Requested segment length for PSD estimation.
        """
        self.PSD_IFs(npsg)
        self.PSD_modal(npsg)
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

    def _compute_KL_basis_from_cmd(self, modal_basis, tel, dm):
        """Generate a normalized KL-like OPD basis from command vectors."""
        dm.coefs = modal_basis
        tel * dm
        return tel.OPD.copy() / tel.OPD.copy()[tel.pupil == 1, :].std(axis=0)

    def _compute_ncpa_opd(self, ncpa=None):
        """Compute the static NCPA OPD map from modal coefficients."""
        self.tel1.OPD = self.tel1.OPD * 0
        self.dm1.coefs = 0
        if ncpa is not None:
            self.dm1.coefs = self.M2C @ ncpa
        else:
            self.dm1.coefs = self.dm1.coefs * 0
        self.tel1 * self.dm1
        return self.tel1.OPD.copy().astype(np.float32)

    

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

    def _pad_to_square(self, arr: np.ndarray):
        """
        Pad a 2D or 3D array with zeros to obtain square spatial dimensions.

        Parameters
        ----------
        arr : ndarray
            Input array with shape ``(M, N)`` or ``(L, M, N)``.

        Returns
        -------
        padded : ndarray
            Zero-padded square array.
        padded_cr : list[int]
            Padding convention offsets describing the applied crop/pad values.

        Raises
        ------
        ValueError
            If the array is not 2D or 3D.
        TypeError
            If the dtype is unsupported for constant padding.
        """
        if arr.ndim == 2:
            M, N = arr.shape
        elif arr.ndim == 3:
            L, M, N = arr.shape
        else:
            raise ValueError('Only 2 or 3d arrays works')
        size = max(M, N)
        pad_top = (size - M) // 2
        pad_bottom = size - M - pad_top
        pad_left = (size - N) // 2
        pad_right = size - N - pad_left
        if arr.ndim == 2:
            pad_width = [(pad_top, pad_bottom), (pad_left, pad_right)]
        else:
            pad_width = [(0, 0), (pad_top, pad_bottom), (pad_left, pad_right)]
        if np.issubdtype(arr.dtype, np.bool_):
            const_value = False
        elif np.issubdtype(arr.dtype, np.integer):
            const_value = 0
        elif np.issubdtype(arr.dtype, np.floating):
            const_value = 0.0
        else:
            raise TypeError(f'Unsupported type {arr.dtype} ')
        padded = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=const_value)
        padded_cr = [-pad_bottom, -pad_left, pad_top, pad_right]
        return (padded, padded_cr)

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

    def _delete_img(self):
        """Delete image cubes from the instance to release memory."""
        del self.img, self.img_raw

    def __matmul__(self, obj):
        """Project compatible telemetry products using the ``@`` operator."""
        if obj.tag == 'papy':
            if self.has_recontructed_phase:
                self.papy_if = self.project_on_PAPYRUS(obj.proj_IF)
                self.papy_modes = self.project_on_PAPYRUS(obj.proj_M2C)
            else:
                raise RuntimeError('Must compute phase before projection')
            print('Second stage projected on first stage')
        else:
            raise ValueError('Entered object not a PAPYtele')
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
