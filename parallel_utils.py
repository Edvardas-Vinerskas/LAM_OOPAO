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


def _build_vzwfs_from_setup(setup):
    Source, Telescope, ZWFS, ZWFS2, _, _, _ = _import_oopao_symbols()

    
    if setup["is_onsky"] and (~setup["is_nb"]):
        src1 = Source(optBand='H', magnitude=-2.5)
        src1.wavelength = 1.6e-6
        src1.bandwidth = 0.2e-6

        src2 = Source(optBand='H', magnitude=-2.5)
        src2.wavelength = 1.6e-6
        src2.bandwidth = 0.2e-6
        diam = 2
    else:
        src1 = Source(optBand='IR1310', magnitude=-2.5)
        src2 = Source(optBand='IR1310', magnitude=-2.5)
        src1.wavelength = 1.550e-6
        src1.bandwidth = 0e-6
        src2.wavelength = 1.550e-6
        src2.bandwidth = 0e-6

    tel1 = Telescope(setup["submask0"].shape[0], 1.52, pupil=setup["submask0"])
    tel1.pupilReflectivity = np.sqrt(setup["pupil0"])
    tel1.pupilReflectivity[~np.isfinite(tel1.pupilReflectivity)]=0

    src1 * tel1

    tel2 = Telescope(setup["submask1"].shape[0], 1.52, pupil=setup["submask1"])
    tel2.pupilReflectivity = np.sqrt(setup["pupil1"])
    tel2.pupilReflectivity[~np.isfinite(tel2.pupilReflectivity)]=0
    src2 * tel2
    
    zwfs1 = ZWFS(tel1, diameter=diam, phase_shift=0.33, zpf=30, phase_shift_unit='pi')
    zwfs2 = ZWFS(tel2, diameter=diam, phase_shift=-0.74, zpf=30, phase_shift_unit='pi')

    return ZWFS2(ZWFS1=zwfs1, ZWFS2=zwfs2)

def _reconstruct_phase_worker(im1, im2, setup, method='atan', damping=0.5, iteration=10):
    with _suppress_output():
        vzwfs = _build_vzwfs_from_setup(setup)
        vzwfs.zwfs1.img_ZWFS = im1
        vzwfs.zwfs2.img_ZWFS = im2
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


def _simulate_psf_chunk_worker(opd_chunk, setup, opd_ncpa):
    with _suppress_output():
        tel1, cam = _build_psf_objects_from_setup(setup)

        out = []
        for opd in opd_chunk:
            tel1.OPD = opd + opd_ncpa
            tel1 * cam
            out.append(cam.frame.copy().astype(np.float32))

    return np.asarray(out, dtype=np.float32)