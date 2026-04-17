# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 13:38:53 2026

@author: mmotte
"""

import numpy as np


from Pupil_selection import reference_intensities
from skimage.transform import resize
import logging  # For logging messages


from scipy.signal import welch
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO)  # Set up logging
logger = logging.getLogger(__name__)  # Create logger
import tkinter as tk
from tkinter import filedialog
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


class PAPYtele:
    """
    Analyze PAPYRUS telemetry data, reconstruct wavefront phase, and compute
    modal or influence-function PSDs from telemetry cubes.
    """

    def __init__(self,tele_path:str = None):
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
        if tele_path is None:
            tele_path = self._choose_file()

            if tele_path:
                print("Selected file:", tele_path)
            else:
                raise ValueError('No files selected')
        self.tele_path = tele_path
        data = np.load(self.tele_path, allow_pickle=True)

        self.dmshape = data.item()['dmCmdCube'].astype(np.float32)
        self.rec_cmd = data.item()['modeCube'].astype(np.float32)
        self.ts = data.item()['timeStampOcamCube']  # list[datetime]
        self.t0 = self.ts[0]
        self.time = np.array([(t - self.t0).total_seconds() for t in self.ts], dtype=float)
        self.M2C = data.item()['m2c']
        self.C2M =np.linalg.pinv(self.M2C)
        self.proj_IF = np.load('projector_IF_sky.npy')
        pupil = self.proj_IF.sum(axis = 0)
        self.pupil_mask = pupil!=0
        self.IF = np.load('IF_DM1.npy')[self.pupil_mask,:]
        
        self.M2phase = self.IF@self.M2C
        self.modes_std = self.M2phase.std(axis = 0)
        self.IF_std = self.IF.std(axis = 0)

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
        
        self.psd_cmd_IFs = self._psd(self.time, (self.M2C@self.rec_cmd)*self.IF_std[None,:], nperseg=npsg)
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
        self.psd_cmd_modal = self._psd(self.time, self.rec_cmd*self.modes_std[None,:], nperseg=npsg)
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

        
        return f,psd
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
