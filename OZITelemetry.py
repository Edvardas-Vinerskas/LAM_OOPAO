# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 13:38:53 2026

@author: mmotte
"""

import numpy as np


from Pupil_selection import reference_intensities
from skimage.transform import resize
import logging  # For logging messages
import tqdm

from scipy.signal import welch
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
logging.basicConfig(level=logging.INFO)  # Set up logging
logger = logging.getLogger(__name__)  # Create logger
import tkinter as tk
from tkinter import filedialog
import os
import sys


global HERE
HERE = os.path.dirname(os.path.abspath(__file__))

if HERE not in sys.path:
    sys.path.insert(0, HERE)
from parallel_utils import _reconstruct_phase_worker, _import_oopao_symbols, _simulate_psf_chunk_worker
from OOPAO.Zernike import Zernike


class OZITele:
    """
    Analyze OZIRIIS telemetry data, reconstruct wavefront phase, and compute
    modal or influence-function PSDs from telemetry cubes.
    """

    def __init__(self,tele_path:str = None, is_onsky:bool=True, CNN = False, narrow_band = False, temporal_crop = None):
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
        self.temporal_crop = slice(None) if temporal_crop is None else slice(temporal_crop[0], temporal_crop[1])
        if tele_path is None:
            tele_path = self._choose_file()

            if tele_path:
                print("Selected file:", tele_path)
            else:
                raise ValueError('No files selected')
        self.is_nb = narrow_band
        self.tele_path = tele_path
        data = np.load(self.tele_path, allow_pickle=True)
        self.off_mask = data.item()['validPixels'].astype(np.float32)
        self.img_raw = data.item()['CRED2Cube'].astype(np.float32)[self.temporal_crop]
        self.dark = data.item()['credDark'].astype(np.float32)
        self.dmshape = data.item()['dmCmdCube'].astype(np.float32)[self.temporal_crop]
        self.reconstructed_cube = data.item()['slavedreconsCube'].astype(np.float32)[self.temporal_crop]
        self.full_reconstructed_cube = data.item()['FullreconsCube'].astype(np.float32)[self.temporal_crop]
        self.M2C = data.item()['m2c']
        self.C2M =np.linalg.pinv(self.M2C)
        self.psf_sampling = 3
        self.frame_count = self.img_raw[:,0,0]
        # img /=img.sum(axis = (1,2))[:,None,None]
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

        # t_cl = np.arange(0, a_cl.shape[0],1)*1/400
        # t_ol = np.arange(0, a_ol.shape[0],1)*1/400
        # Conversion modale (vectorisée)
        for i in range(self.rec_cmd.shape[0]):
            self.rec_cmd_modal[i,:] = self.C2M@self.rec_cmd[i]
       
        self.img = self.img_raw-self.dark
        self.M2C = data.item()['m2c']
        self.img[:,:1,:] = 0
        self.img /=self.img.sum(axis = (1,2))[:,None,None]
        self.ts = data.item()['timeStampcredCube'][self.temporal_crop]  # list[datetime]
        self.t0 = self.ts[0]
        self.time = np.array([(t - self.t0).total_seconds() for t in self.ts], dtype=float)
        self.is_onsky = is_onsky
        if self.is_onsky:
            
            self.positions_calib = [np.array([ 35,  20, 125, 110]), np.array([127,  19, 216, 109])]
            self.initial_positions, self.initial_pupils, self.initial_submasks, self.global_masks = reference_intensities(self.off_mask)
            minr, minc, maxr, maxc = self.positions_calib[0]
            self.initial_submasks[0] = self.global_masks[0][minr:maxr, minc:maxc]
            minr, minc, maxr, maxc = self.positions_calib[1]
            self.initial_submasks[1] = self.global_masks[1][minr:maxr, minc:maxc]
            self.initial_pupils[0] = np.zeros_like(self.initial_submasks[0]).astype(np.float32)
            self.initial_pupils[1] = np.zeros_like(self.initial_submasks[1]).astype(np.float32)
            self.initial_pupils[0][self.initial_submasks[0]] = self.off_mask[self.global_masks[0]]
            self.initial_pupils[1][self.initial_submasks[1]] = self.off_mask[self.global_masks[1]]
            self.submasks = [None,None]
            self.pupils = [None,None]
            pupil2 = self._rescale_matrix(self.initial_pupils[1], self.initial_pupils[0].shape[0], self.initial_pupils[0].shape[1])
            
            
            self.pupils[1],_ = self._pad_to_square(pupil2)
            self.pupils[0],_ = self._pad_to_square(self.initial_pupils[0])
            self.submasks[1],_ = self._pad_to_square(self.initial_pupils[0])
            self.submasks[0] = self.submasks[1]
           
        else:
            self.positions_calib = [np.array([ 35,  20, 125, 110]), np.array([127,  19, 216, 109])]
            self.initial_positions, self.initial_pupils, self.initial_submasks, self.global_masks = reference_intensities(self.off_mask)
            
            self.submasks = [None,None]
            self.pupils = [None,None]
            pupil2 = self._rescale_matrix(self.initial_pupils[1], self.initial_pupils[0].shape[0], self.initial_pupils[0].shape[1])
            
            
            self.pupils[1],_ = self._pad_to_square(pupil2)
            self.pupils[0],_ = self._pad_to_square(self.initial_pupils[0])
            self.submasks[1],_ = self._pad_to_square(self.initial_pupils[0])
            self.submasks[0] = self.submasks[1]
            
        self._initialise_OOPAO_objects()
        
        self.compute_projectors()
        self.extract_Zimages()
        self.has_recontructed_phase = False
        self.has_projected_phase = False
    def _initialise_OOPAO_objects(self):
        Source, Telescope, ZWFS, ZWFS2, DeformableMirror, MisRegistration, Detector = _import_oopao_symbols()
        if self.is_onsky and (~self.is_nb):
            self.src1 = Source(optBand='H', magnitude=-2.5)
            self.src1.wavelength = 1.6e-6
 
            self.src1.bandwidth = 0.2e-6
            self.src2 = Source(optBand='H', magnitude=-2.5)
            self.src2.wavelength = 1.6e-6
            self.src2.bandwidth = 0.2e-6
        else: 
            self.src1 = Source(optBand='H', magnitude=-2.5)
            self.src1.wavelength = 1.550e-6
            self.src1.bandwidth = 0
            self.src2 = Source(optBand='H', magnitude=-2.5)
            self.src2.wavelength = 1.550e-6
            self.src2.bandwidth = 0e-6
        self.tel1 = Telescope(self.submasks[0].shape[0],1.52, pupil = self.submasks[0]) 
        self.tel1.pupilReflectivity = np.sqrt(self.pupils[0])
        self.tel1.pupilReflectivity[~np.isfinite(self.tel1.pupilReflectivity)]=0
        self.src1*self.tel1
        self.tel2 = Telescope(self.submasks[1].shape[0],1.52, pupil = self.submasks[1]) 
        self.tel2.pupilReflectivity = np.sqrt(self.pupils[1])
        self.tel2.pupilReflectivity[~np.isfinite(self.tel2.pupilReflectivity)]=0
        self.src2*self.tel2


        self.vzwfs = self._build_vzwfs_class()
        self.zwfs1 = self.vzwfs.zwfs1
        self.zwfs2 = self.vzwfs.zwfs2
        self.cam = Detector(psf_sampling=self.psf_sampling)

        param = np.load(HERE+'\dm_second_stage_misreg_dict.npy', allow_pickle=True).item()
        m = MisRegistration(param)

        self.dm1 = DeformableMirror(telescope = self.tel1,
                                nSubap=10,
                                mechCoupling=0.35,
                                print_dm_properties=False,
                                pitch=0.11,
                                misReg = m,
                                sign=-1e-5)
        self.dm2 = DeformableMirror(telescope = self.tel2,
                                nSubap=10,
                                mechCoupling=0.35,
                                print_dm_properties=False,
                                pitch=0.11,
                                misReg = m,
                                sign=-1e-5)

        self.IF = np.load(HERE+'\IF_dm2.npy').reshape(97,-1).T*1e-6
        self.M2phase = self.IF@self.M2C
        self.modes_std = self.M2phase.std(axis = 0)
        self.IF_std = self.IF.std(axis = 0)
        amplitude_mean = np.ptp(self.IF,axis =0)

        self.dm1.modes*=amplitude_mean/np.ptp(self.dm1.modes)
        self.dm2.modes*=amplitude_mean/np.ptp(self.dm2.modes)
        self.src_imaging = Source(optBand='IR1310', magnitude=-2.5)
    def _build_vzwfs_class(self):
        Source, Telescope, ZWFS, ZWFS2, DeformableMirror, MisRegistration,Detector = _import_oopao_symbols()
        if self.is_onsky:
            diam = 2
        else:
            diam = 2.14
        zwfs1 = ZWFS(self.tel1, diameter = diam, phase_shift=0.33, zpf = 30, phase_shift_unit='pi' )
        zwfs2 = ZWFS(self.tel2, diameter = diam, phase_shift=-0.74, zpf = 30, phase_shift_unit='pi' )
        return ZWFS2(ZWFS1=zwfs1, ZWFS2=zwfs2)
    def _export_reconstruction_setup(self):
        return {
            "is_onsky": self.is_onsky,
            "is_nb": self.is_nb,
            "submask0": self.submasks[0],
            "submask1": self.submasks[1],
            "pupil0": self.pupils[0],
            "pupil1": self.pupils[1],
        }
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
    def _compute_proj_dm(self, modal_basis, tel, dm):
        dm = modal_basis
        tel*dm
        modes= tel.OPD.copy()
        modes = modes.reshape((tel.resolution**2, modes.shape[-1]))/tel.OPD[tel.pupil, :].std(axis=0)  # Flatten for projection
        cov_modes = modes.T @ modes  # Compute mode covariance
        return np.diag(1 / np.diag(cov_modes)) @ modes.T  # Pseudo-inverse projection matrix
    def _compute_proj_OPDs(self, modes, tel):
        

        modes = modes.reshape((tel.resolution**2, modes.shape[-1]))/modes[tel.pupil, :].std(axis=0)  # Flatten for projection
        cov_modes = modes.T @ modes  # Compute mode covariance
        return np.diag(1 / np.diag(cov_modes)) @ modes.T  # Pseudo-inverse projection matrix
    def compute_projectors(self):
        """
        Compute projection matrices onto modal commands and influence functions.

        
        """
        
        self.proj_M2C = self._compute_proj_dm(self.M2C, self.tel1, self.dm1)# np.diag(1 / np.diag(cov_modes)) @ modes.T  # Pseudo-inverse projection matrix
        self.proj_IF = self._compute_proj_dm(np.identity(self.dm1.modes.shape[-1]), self.tel1, self.dm1)
    
    def extract_Zimages(self):
        """
        Extract and format the two ZWFS image streams from the raw image cube.

        The images are cropped from the valid-pixel regions, rescaled when
        needed, and padded to square arrays so they match the internal optical
        model geometry.
        """
        images_z1 = np.zeros((self.img.shape[0],self.initial_submasks[0].shape[0],self.initial_submasks[0].shape[1]))
        images_z2 =  np.zeros((self.img.shape[0],self.initial_submasks[1].shape[0],self.initial_submasks[1].shape[1]))
        images_z1[:,self.initial_submasks[0]] = (self.img)[:,self.global_masks[0]]
        images_z2[:,self.initial_submasks[1]] = (self.img)[:,self.global_masks[1]]
        self.img_ZWFS2 = []
        logger.info('Extracting the signal of the ZWFSs')
        for i in tqdm.tqdm(range(images_z2.shape[0])):
            self.img_ZWFS2.append(self._rescale_matrix(images_z2[i], self.pupils[0].shape[0], self.pupils[0].shape[1]))
        self.img_ZWFS2 = np.array(self.img_ZWFS2)
        self.img_ZWFS2, _ =  self._pad_to_square(self.img_ZWFS2)
        self.img_ZWFS1,_ =  self._pad_to_square(images_z1)
        
    def reconstruct_phase(self, im1,im2, method = 'atan', damping = 0.5, iteration = 10, parallel = False):
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
        return self.vzwfs.reconstructor(
            iteration=iteration,
            damping_iteration=damping,
            reconstructor=method
        )
        
    
    def reconstruct_all_phase(self, method = 'atan', iteration = 10, damping = 0.5, parallel = True, parall_njob = 4):
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
                prefer="processes",
                return_as="generator"
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
    
            self.phase = np.asarray(
                list(tqdm.tqdm(gen, total=n_frames, desc=f"Phase reconstruction ({method})")),
                dtype=np.float32
            )
 
        else:
            self.phase = np.zeros((self.img_ZWFS1.shape[0], self.tel1.pupil.shape[0], self.tel1.pupil.shape[1])).astype(np.float32)
            logger.info(f'Computing phase for each frame using {method} reconstruction')
            for i in tqdm.tqdm(range(self.img_ZWFS1.shape[0])):
                self.phase[i] = self.reconstruct_phase(self.img_ZWFS1[i], self.img_ZWFS2[i], method, damping, iteration).astype(np.float32)
        self._phase2OPD()
        self.has_recontructed_phase = True
    def _phase2OPD(self, phase = None):
        self.OPDs= (self.phase/(2*np.pi)*self.src1.wavelength).astype(np.float32)
    def compute_Zernike_basis(self, nmodes = 30):
        Zer_basis = Zernike(self.tel1, J= nmodes)
        Zer_basis.computeZernike(self.tel1)
        self.Zer_modes = Zer_basis.modesFullRes.copy()
        self.proj_Zer = self._compute_proj_OPDs(self.Zer_modes, self.tel1)

    def project_OPDs(self):
        
        """
        Project reconstructed OPD maps onto influence functions and modes.

        Raises
        ------
        RuntimeError
            If phase reconstruction has not been computed yet.
        """
        if self.has_recontructed_phase:
            self.OPDs_on_IFs = self._project_phase(self.proj_IF, self.OPDs)
            self.OPDs_on_modes = self._project_phase(self.proj_M2C, self.OPDs)
            self.has_projected_phase = True
        else:
            raise RuntimeError('Must compute phase before projection')
        
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
    def _pad_to_square(self,arr: np.ndarray):
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
            raise ValueError("Only 2 or 3d arrays works")

        size = max(M, N)
        pad_top = (size - M) // 2
        pad_bottom = size - M - pad_top
        pad_left = (size - N) // 2
        pad_right = size - N - pad_left

        # Padding selon dimension
        if arr.ndim == 2:
            pad_width = [(pad_top, pad_bottom), (pad_left, pad_right)]
        else:  # 3D
            pad_width = [(0, 0), (pad_top, pad_bottom), (pad_left, pad_right)]

        # Détermination du remplissage constant selon le type
        if np.issubdtype(arr.dtype, np.bool_):
            const_value = False
        elif np.issubdtype(arr.dtype, np.integer):
            const_value = 0
        elif np.issubdtype(arr.dtype, np.floating):
            const_value = 0.0
        else:
            raise TypeError(f"Unsupported type {arr.dtype} ")

        padded = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=const_value)

        # Conserve la même convention que ton code précédent
        padded_cr = [-pad_bottom, -pad_left, pad_top, pad_right]

        return padded, padded_cr
    def PSD_IFs(self, npsg = None):
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
            self.psd_IFs = self._psd(self.time, self.OPDs_on_IFs, nperseg=npsg)
        else:
            raise RuntimeError('Must project phase before PSDs')
    def PSD_modal(self, npsg = None):
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
            self.psd_modal = self._psd(self.time, self.OPDs_on_modes, nperseg=npsg)
        else:
            raise RuntimeError('Must project phase before PSDs')
            
    def compute_all_PSD(self, npsg = None):
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
        
        self.psd_cmd_IFs = self._psd(self.time, self.rec_cmd*self.IF_std[None,:], nperseg=npsg)
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
        self.psd_cmd_modal = self._psd(self.time, self.rec_cmd_modal*self.modes_std[None,:], nperseg=npsg)
    def _export_psf_setup(self, img_wvl = False):
        return {
            "is_onsky": self.is_onsky,
            "imaging_wvl":img_wvl,
            "is_nb": self.is_nb,
            "submask0": self.submasks[0],
            "pupil0": self.pupils[0],
            "psf_sampling": self.psf_sampling,
        }

    def _compute_ncpa_opd(self, ncpa=None):
        self.tel1.OPD = self.tel1.OPD * 0
    
        if ncpa is not None:
            self.dm1.coefs = self.M2C @ ncpa
        else:
            self.dm1.coefs = self.dm1.coefs * 0
    
        self.tel1 * self.dm1
        return self.tel1.OPD.copy().astype(np.float32)
    def simulate_PSF(self, imaging_wvl = True, ncpa=None, parallel=True, parall_njob=4, chunk_size=100):
        if ncpa is not None:
            if ncpa.size != self.M2C.shape[-1]:
                raise ValueError('ncpa must be an array of the size of the number of modes in the M2C')
    
        if not self.has_recontructed_phase:
            raise RuntimeError('Must compute phase before projection')
    
        opd_ncpa = self._compute_ncpa_opd(ncpa)
        
        if parallel:
            setup = self._export_psf_setup(imaging_wvl)
            n_frames = self.OPDs.shape[0]
    
            chunks = [
                self.OPDs[i:i + chunk_size]
                for i in range(0, n_frames, chunk_size)
            ]
    
            gen = Parallel(
                n_jobs=parall_njob,
                prefer="processes",
                return_as="generator"
            )(
                delayed(_simulate_psf_chunk_worker)(
                    chunk,
                    setup,
                    opd_ncpa
                )
                for chunk in chunks
            )
    
            psf_chunks = list(
                tqdm.tqdm(gen, total=len(chunks), desc="PSF simulation")
            )
            self.simulated_psf = np.concatenate(psf_chunks, axis=0).astype(np.float32)
    
        else:
            if imaging_wvl:
                self.src_imaging*self.tel1
            else:
                self.src1*self.tel1
            self.simulated_psf = []
            for i in tqdm.tqdm(range(self.OPDs.shape[0]), desc="PSF simulation"):
                self.tel1.OPD = self.OPDs[i] + opd_ncpa
                self.tel1 * self.cam
                self.simulated_psf.append(self.cam.frame.copy().astype(np.float32))
    
            self.simulated_psf = np.asarray(self.simulated_psf, dtype=np.float32)
            
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
    def compute_SR(self, wavelength = None, ncpa = None):
        opd_ncpa = self._compute_ncpa_opd(ncpa)
        if wavelength is None:
            wavelength = self.src1.wavelength
        phase_var = (self.phase[:,self.tel1.pupil ==1]+opd_ncpa[self.tel1.pupil ==1]*2*np.pi/self.src1.wavelength).var(axis=1)*(self.src1.wavelength/wavelength)**2
        

        SR = np.exp(-phase_var)#.mean())

        SR_mean = np.exp(-phase_var.mean())
        
        print(f"At {wavelength*1e9:.0f} nm, the average SR is about {SR_mean*100:.1f}%" )
        return SR, SR_mean
        
    def _psd(self, t, a,fs=None, nperseg=4096, noverlap=None, detrend="constant", window="hann"):
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
        print(np.isnan(t).any())
        print(np.isnan(a).any())
        _, a_u, fs = self._uniform_resample(t, a, fs=fs)
        print(fs)
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

        
        return f.astype(np.float32),psd.astype(np.float32)
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
            filetypes=[("Python files", "*.npy"), ("All files", "*.*")]
        )
        
        root.destroy()
        return file_path
    def _delete_img(self):
        del self.img, self.img_raw
