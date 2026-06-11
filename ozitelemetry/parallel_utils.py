# -*- coding: utf-8 -*-
import numpy as np


import os
import sys


import contextlib
import importlib
@contextlib.contextmanager
def _silent_oopao_import():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_get_terminal_size = os.get_terminal_size

    devnull_out = open(os.devnull, "w", encoding="utf-8", errors="ignore")
    devnull_err = open(os.devnull, "w", encoding="utf-8", errors="ignore")

    try:
        sys.stdout = devnull_out
        sys.stderr = devnull_err
        os.get_terminal_size = lambda *args, **kwargs: os.terminal_size((80, 24))
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        os.get_terminal_size = old_get_terminal_size
        devnull_out.close()
        devnull_err.close()
@contextlib.contextmanager
def _suppress_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    f_out = open(os.devnull, "w", encoding="utf-8", errors="ignore")
    f_err = open(os.devnull, "w", encoding="utf-8", errors="ignore")

    try:
        sys.stdout = f_out
        sys.stderr = f_err
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        f_out.close()
        f_err.close()
def _import_oopao_symbols():
    with _silent_oopao_import():
        Source = importlib.import_module("OOPAO.Source").Source
        Telescope = importlib.import_module("OOPAO.Telescope").Telescope
        ZWFS = importlib.import_module("OOPAO.ZWFS").ZWFS
        ZWFS2 = importlib.import_module("OOPAO.ZWFS2").ZWFS2
        DeformableMirror = importlib.import_module("OOPAO.DeformableMirror").DeformableMirror
        MisRegistration = importlib.import_module("OOPAO.MisRegistration").MisRegistration
        Detector = importlib.import_module("OOPAO.Detector").Detector
        
    return Source, Telescope, ZWFS, ZWFS2, DeformableMirror, MisRegistration, Detector
def _import_all_oopao_symbols():
    with _silent_oopao_import():
        Source = importlib.import_module("OOPAO.Source").Source
        Telescope = importlib.import_module("OOPAO.Telescope").Telescope
        ZWFS = importlib.import_module("OOPAO.ZWFS").ZWFS
        ZWFS2 = importlib.import_module("OOPAO.ZWFS2").ZWFS2
        DeformableMirror = importlib.import_module("OOPAO.DeformableMirror").DeformableMirror
        MisRegistration = importlib.import_module("OOPAO.MisRegistration").MisRegistration
        Detector = importlib.import_module("OOPAO.Detector").Detector
        Atmosphere = importlib.import_module("OOPAO.Atmosphere").Atmosphere
    return Source, Telescope, ZWFS, ZWFS2, DeformableMirror, MisRegistration, Detector, Atmosphere

def _build_vzwfs_from_setup(setup):
    Source, Telescope, ZWFS, ZWFS2, _, _, _ = _import_oopao_symbols()

    src1 = Source(optBand=setup.get("src1_optBand", "H"), magnitude=-2.5)
    src1.wavelength = setup["src1_wavelength"]
    src1.bandwidth = setup["src1_bandwidth"]

    src2 = Source(optBand=setup.get("src2_optBand", "H"), magnitude=-2.5)
    src2.wavelength = setup["src2_wavelength"]
    src2.bandwidth = setup["src2_bandwidth"]

    tel1 = Telescope(
        setup["submask0"].shape[0],
        1.52,
        pupil=setup["submask0"]
    )
    tel1.pupilReflectivity = np.sqrt(setup["pupil0"]) * setup["submask0"]
    tel1.pupilReflectivity[~np.isfinite(tel1.pupilReflectivity)] = 0
    src1 * tel1

    tel2 = Telescope(
        setup["submask1"].shape[0],
        1.52,
        pupil=setup["submask1"]
    )
    tel2.pupilReflectivity = np.sqrt(setup["pupil1"]) * setup["submask1"]
    tel2.pupilReflectivity[~np.isfinite(tel2.pupilReflectivity)] = 0
    src2 * tel2

    zwfs1 = ZWFS(
        tel1,
        diameter=setup["diam"],
        phase_shift=setup["phase_shift_1"],
        zpf=setup["zpf"],
        phase_shift_unit=setup["phase_shift_unit"]
    )

    zwfs2 = ZWFS(
        tel2,
        diameter=setup["diam"],
        phase_shift=setup["phase_shift_2"],
        zpf=setup["zpf"],
        phase_shift_unit=setup["phase_shift_unit"]
    )

    return ZWFS2(ZWFS1=zwfs1, ZWFS2=zwfs2)

def _reconstruct_phase_worker(im1, im2, setup, method='atan', damping=0.5, iteration=10, modes_filtering = False, modal_basis = None, nmodes = None):
    with _suppress_output():
        vzwfs = _build_vzwfs_from_setup(setup)
        vzwfs.zwfs1.img_ZWFS = im1
        vzwfs.zwfs2.img_ZWFS = im2
        if modes_filtering:
            phase = vzwfs.reconstructor(
                iteration=iteration,
                damping_iteration=damping,
                reconstructor=method,
                filter_modes = modes_filtering,
                modal_basis = modal_basis[...,:nmodes]
            )
        else:
            phase = vzwfs.reconstructor(
                iteration=iteration,
                damping_iteration=damping,
                reconstructor=method
            )
    return phase

def _build_psf_objects_from_setup(setup):
    Source, Telescope, _, _, _, _, Detector = _import_oopao_symbols()
    if setup["imaging_wvl"]: 
        src1 = Source(optBand='IR1310', magnitude=-2.5)
    else:
        if setup["is_onsky"] and (~setup["is_nb"]):
            src1 = Source(optBand='H', magnitude=-2.5)
            src1.wavelength = 1.6e-6
            src1.bandwidth = 0.2e-6
        else:
            src1 = Source(optBand='IR1310', magnitude=-2.5)
            src1.wavelength = 1.550e-6
            src1.bandwidth = 0e-6

    tel1 = Telescope(setup["submask0"].shape[0], 1.52, pupil=setup["submask0"])
    tel1.pupilReflectivity = np.sqrt(setup["pupil0"])
    tel1.pupilReflectivity[~np.isfinite(tel1.pupilReflectivity)] = 0
    src1 * tel1

    cam = Detector(psf_sampling=setup["psf_sampling"])
    return tel1, cam

def _simulate_psf_chunk_worker(opd_chunk, pupil, sampling, nsize, opd_ncpa):
    with _suppress_output():
        out = []
        for opd in opd_chunk:
            out.append(MFT_psf(opd+opd_ncpa, pupil, sampling, nsize))

    return np.asarray(out, dtype=np.float32)

def MFT_psf(
    phi,
    pupil,
    sampling=3,
    nimg=100,
    normalize=False,
):
    """
    Simule une PSF par Matrix Fourier Transform avec :
    - un sampling focal arbitraire (même non entier),
    - un nombre de pixels imposé en sortie.

    Parameters
    ----------
    phi : 2D ndarray
        Carte de phase en radians.
    pupil : 2D ndarray
        Fonction pupille.
    sampling : float
        Sampling focal en pixels par lambda/D.
    nimg : int or tuple of int
        Taille de l'image PSF de sortie.
        - si int : image carrée nimg x nimg
        - si tuple : (ny_img, nx_img)
    normalize : bool
        Si True, normalise la PSF par son maximum.

    Returns
    -------
    psf : 2D ndarray
        PSF simulée.
    x_foc : 1D ndarray
        Coordonnées focales en lambda/D sur l'axe x.
    y_foc : 1D ndarray
        Coordonnées focales en lambda/D sur l'axe y.
    """

    phi = np.asarray(phi)
    pupil = np.asarray(pupil)

    if phi.shape != pupil.shape:
        raise ValueError("phi et pupil doivent avoir la même taille.")

    if np.isscalar(nimg):
        ny_img = int(nimg)
        nx_img = int(nimg)
    else:
        ny_img, nx_img = nimg

    ny, nx = phi.shape

    # Champ complexe dans la pupille
    field = pupil * np.exp(1j * phi)

    # Nombre de pixels sur le diamètre de la pupille
    yy, xx = np.where(pupil > 0)
    if len(xx) == 0 or len(yy) == 0:
        raise ValueError("La pupille semble vide.")

    Dpix = max(xx.max() - xx.min() + 1, yy.max() - yy.min() + 1)

    # Coordonnées pupille normalisées en unités de D
    # (0 au centre géométrique)
    x_pup = (np.arange(nx) - (nx - 1) / 2) / Dpix
    y_pup = (np.arange(ny) - (ny - 1) / 2) / Dpix

    # Coordonnées focales en lambda/D
    # Sampling = nb de pixels par lambda/D
    x_foc = (np.arange(nx_img) - (nx_img - 1) / 2) / sampling
    y_foc = (np.arange(ny_img) - (ny_img - 1) / 2) / sampling

    # Matrices de Fourier
    Mx = np.exp(-2j * np.pi * np.outer(x_foc, x_pup))
    My = np.exp(-2j * np.pi * np.outer(y_foc, y_pup))

    # Transformée de Fourier 2D par produit matriciel
    focal = My @ field @ Mx.T

    # PSF
    psf = np.abs(focal) ** 2

    if normalize:
        max_psf = np.nanmax(psf)
        if max_psf > 0:
            psf = psf / max_psf

    return psf
def _simulate_psf_chunk_worker_depreciated(opd_chunk, setup, opd_ncpa):
    with _suppress_output():
        tel1, cam = _build_psf_objects_from_setup(setup)

        out = []
        for opd in opd_chunk:
            tel1.OPD = opd + opd_ncpa
            tel1 * cam
            out.append(cam.frame.copy().astype(np.float32))

    return np.asarray(out, dtype=np.float32)