# -*- coding: utf-8 -*-

import numpy as np

from OOPAO.Source import Source

import time


def run_ao_loop_2nd_stage(
    OZItwin,
    residuals_opds_1rst,
    nLoop,
    nmodes,
    calib_CL_2nd,
    gainCL=0.0,
    leak=0.98,
    frame_delay=2,
    verbose=True,
):
    """
    Run the second-stage AO loop using residual OPDs from the first stage.

    Parameters
    ----------
    OZItwin : object
        Object containing atm, tel, dm, src, vzwfs, M2C.
    residuals_opds_1rst : np.ndarray
        Residual OPDs from the first AO stage.
        Shape typically: (nLoop, tel.resolution, tel.resolution).
    nLoop : int
        Number of loop iterations.
    nmodes : int
        Number of modes kept in OZItwin.M2C[:, :nmodes].
    calib_CL_2nd : np.ndarray
        Calibration/reconstruction matrix for the second-stage WFS.
    gainCL : float
        Closed-loop gain.
    leak : float
        DM leak factor.
    frame_delay : int
        Frame delay. Currently supports frame_delay = 1 or 2.
    verbose : bool
        If True, print loop progress.

    Returns
    -------
    results : dict
        Dictionary containing loop outputs.
    """

    # Allocate memory
    SR_NGS = np.zeros(nLoop)
    SR_SRC = np.zeros(nLoop)

    total = np.zeros(nLoop)
    residual_SRC = np.zeros(nLoop)
    residual_NGS = np.zeros(nLoop)

    dm_commands = np.zeros((nLoop, OZItwin.dm.nValidAct))

    # Initial WFS signal
    wfsSignal = np.zeros(OZItwin.vzwfs.zwfs1.nSignal)

    # Reconstructor
    reconstructor = OZItwin.M2C[:, :nmodes] @ calib_CL_2nd

    n_cmd = reconstructor.shape[0]
    reconstructed_cmd = np.zeros((nLoop, n_cmd))

    # OPD storage
    opds = np.zeros_like(residuals_opds_1rst)

    OPD_NGS_all = np.zeros_like(residuals_opds_1rst)
    OPD_SRC_all = np.zeros_like(residuals_opds_1rst)

    # Initialization
    OZItwin.dm.coefs = 0
    OZItwin.atm.initializeAtmosphere(OZItwin.tel)

    pupil_mask = np.where(OZItwin.tel.pupil > 0)

    for i in range(nLoop):

        # Update phase screen from first-stage residual OPD
        OZItwin.atm.update(residuals_opds_1rst[i])

        # Save input phase variance
        total[i] = np.std(OZItwin.tel.OPD[pupil_mask]) * 1e9

        # Propagate source through atmosphere, telescope and DM
        OZItwin.atm * OZItwin.src * OZItwin.tel * OZItwin.dm

        # Propagate to vector ZWFS
        OZItwin.tel * OZItwin.vzwfs

        # Save NGS residual
        residual_NGS[i] = np.std(OZItwin.tel.OPD[pupil_mask]) * 1e9
        OPD_NGS_all[i] = OZItwin.tel.mean_removed_OPD.copy()
        opds[i] = OZItwin.tel.OPD.copy()

        # Propagate SRC through atmosphere, telescope and DM
        OZItwin.atm * OZItwin.src * OZItwin.tel * OZItwin.dm

        # Save DM commands
        dm_commands[i, :] = OZItwin.dm.coefs.copy()

        # Save SRC residual
        residual_SRC[i] = np.std(OZItwin.tel.OPD[pupil_mask]) * 1e9
        OPD_SRC_all[i] = OZItwin.tel.mean_removed_OPD.copy()

        # Frame delay = 1:
        # use the current WFS signal before computing the command
        if frame_delay == 1:
            wfsSignal = OZItwin.vzwfs.zwfs1.signal.copy()

        # Reconstruct command
        reconstructed_cmd[i] = reconstructor @ wfsSignal

        # Apply command on DM
        OZItwin.dm.coefs = (
            leak * OZItwin.dm.coefs
            - gainCL * reconstructed_cmd[i]
        )

        # Frame delay = 2:
        # update WFS signal after computing the command
        if frame_delay == 2:
            wfsSignal = OZItwin.vzwfs.zwfs1.signal.copy()

        if verbose:
            print(
                f"\rLoop {i+1}/{nLoop} "
                f"NGS: {residual_NGS[i]:.3f} "
                f"-- SRC: {residual_SRC[i]:.3f}",
                end="",
                flush=True,
            )

    if verbose:
        print()

    results = {
        "SR_NGS": SR_NGS,
        "SR_SRC": SR_SRC,
        "total": total,
        "residual_SRC": residual_SRC,
        "residual_NGS": residual_NGS,
        "dm_commands": dm_commands,
        "wfsSignal_last": wfsSignal,
        "reconstructor": reconstructor,
        "reconstructed_cmd": reconstructed_cmd,
        "opds": opds,
        "OPD_NGS_all": OPD_NGS_all,
        "OPD_SRC_all": OPD_SRC_all,
        "gainCL": gainCL,
        "leak": leak,
        "frame_delay": frame_delay,
        "nmodes": nmodes,
    }

    return results

def run_ao_loop_from_opds(
    atm,
    tel,
    dm,
    wfs,
    ngs,
    calib,
    M2C_CL,
    atm_OPDs_1rst,
    nLoop,
    OZItwin=None,
    src_band="IR1310",
    src_mag=0,
    gainCL=0.0,
    leak=0.995,
    frame_delay=0,
    photonNoise=False,
    display=False,
    verbose=True,
):
    """
    Run an AO loop using precomputed atmospheric OPDs.

    Parameters
    ----------
    atm : OOPAO Atmosphere
        Atmosphere object.
    tel : OOPAO Telescope
        Telescope object.
    dm : OOPAO DeformableMirror
        Deformable mirror object.
    wfs : OOPAO WFS
        Wavefront sensor object.
    ngs : OOPAO Source
        Natural guide star source.
    calib : object
        Calibration object containing calib.M.
    M2C_CL : np.ndarray
        Modal-to-command or control matrix used before calib.M.
    atm_OPDs_1rst : np.ndarray
        Array of atmospheric OPDs, shape typically:
        (nLoop, tel.resolution, tel.resolution).
    nLoop : int
        Number of loop iterations.
    OZItwin : object, optional
        Object containing OZItwin.tel.samplingTime.
        If None, ratio_samp is set to 1.
    src_band : str
        Scientific source band.
    src_mag : float
        Scientific source magnitude.
    gainCL : float
        Closed-loop gain.
    leak : float
        Leak factor. Currently kept as parameter, but not applied in the original code.
    frame_delay : int
        Frame delay applied to the WFS signal buffer.
    photonNoise : bool
        Whether photon noise is enabled for the WFS camera.
    display : bool
        Display flag. Currently kept for compatibility with the original script.
    verbose : bool
        If True, print loop progress.

    Returns
    -------
    results : dict
        Dictionary containing all saved outputs from the loop.
    """

    # Initialize atmosphere on telescope
    atm.initializeAtmosphere(tel)

    # Create scientific source
    src = Source(src_band, src_mag)
    src * tel

    # Initialize telescope and DM commands
    tel.resetOPD()
    dm.coefs = 0

    ngs * tel * dm * wfs
    wfs * wfs.focal_plane_camera

    # Initialize atmosphere propagation
    atm * ngs * tel
    atm * src * tel

    # Sampling ratio
    if OZItwin is None:
        ratio_samp = 1
    else:
        ratio_samp = 1 / OZItwin.tel.samplingTime / (1 / tel.samplingTime)

    ratio_samp = int(ratio_samp)

    if ratio_samp < 1:
        raise ValueError("ratio_samp must be >= 1.")

    n_wfs_samples = int(nLoop / ratio_samp)

    # Allocate memory
    SR_NGS = np.zeros(nLoop)
    SR_SRC = np.zeros(nLoop)

    total = np.zeros(nLoop)
    residual_SRC = np.zeros(nLoop)
    residual_NGS = np.zeros(nLoop)

    dm_commands = np.zeros((nLoop, dm.nValidAct))

    wfsSignal = np.zeros(
        (n_wfs_samples + frame_delay, wfs.nSignal)
    )

    reconstructor = M2C_CL @ calib.M

    n_cmd = reconstructor.shape[0]
    reconstructed_cmd = np.zeros((n_wfs_samples, n_cmd))

    opds = np.zeros(
        (nLoop, tel.pupil.shape[0], tel.pupil.shape[1])
    )
    opds_res = np.zeros(
        (nLoop, tel.pupil.shape[0], tel.pupil.shape[1])
    )

    OPD_SRC_all = np.zeros(
        (nLoop, tel.pupil.shape[0], tel.pupil.shape[1])
    )

    # Loop parameters
    wfs.cam.photonNoise = photonNoise

    k = 0
    pupil_mask = np.where(tel.pupil > 0)

    for i in range(nLoop):

        # Update phase screen
        atm.update(atm_OPDs_1rst[i] * tel.pupil)

        # Save phase variance before correction
        total[i] = np.std(tel.OPD[pupil_mask]) * 1e9

        # Propagate NGS through atmosphere, telescope, DM, WFS
        atm * ngs * tel * dm * wfs

        opds[i] = dm.OPD.copy()
        opds_res[i] = tel.OPD.copy()

        wfs * wfs.focal_plane_camera

        # Save residuals on NGS
        residual_NGS[i] = np.std(tel.OPD[pupil_mask]) * 1e9

        # WFS sampling
        if i % ratio_samp == 0 and k < n_wfs_samples:

            wfsSignal[k + frame_delay] = wfs.signal.copy()

            reconstructed_cmd[k] = reconstructor @ wfsSignal[k]

            # Apply DM command
            dm.coefs = dm.coefs - gainCL * reconstructed_cmd[k]

            k += 1

        # Propagate SRC through atmosphere, telescope and DM
        atm * src * tel * dm

        dm_commands[i, :] = dm.coefs.copy()

        # Save residuals on SRC
        residual_SRC[i] = np.std(tel.OPD[pupil_mask]) * 1e9

        OPD_SRC_all[i] = tel.mean_removed_OPD.copy()

        if verbose:
            print(
                f"\rLoop {i+1}/{nLoop} "
                f"NGS: {residual_NGS[i]:.3f} "
                f"-- SRC: {residual_SRC[i]:.3f}",
                end="",
                flush=True,
            )

    if verbose:
        print()

    results = {
        "SR_NGS": SR_NGS,
        "SR_SRC": SR_SRC,
        "total": total,
        "residual_SRC": residual_SRC,
        "residual_NGS": residual_NGS,
        "dm_commands": dm_commands,
        "wfsSignal": wfsSignal,
        "reconstructor": reconstructor,
        "reconstructed_cmd": reconstructed_cmd,
        "opds": opds,
        "opds_res": opds_res,
        "OPD_SRC_all": OPD_SRC_all,
        "src": src,
        "ratio_samp": ratio_samp,
        "gainCL": gainCL,
        "leak": leak,
        "frame_delay": frame_delay,
    }

    return results
def compute_etf(
    f1,
    psd1,
    f2,
    psd2,
    method=np.nansum,
    fmin=None,
    fmax=None,
    normalised=False,
):
    
    f1 = np.asarray(f1).ravel()
    psd1 = np.asarray(psd1)

    if psd1.ndim == 1:
        m1 = psd1
    else:
        m1 = method(psd1, axis=1)

    if normalised:
        m1 = m1 / np.nanmax(m1)

    

    f2 = np.asarray(f2).ravel()
    psd2 = np.asarray(psd2)

    if psd2.ndim == 1:
        m2 = psd2
    else:
        m2 = method(psd2, axis=1)

    if normalised:
        m2 = m2 / np.nanmax(m2)

    # valid positive points only
    valid1 = np.isfinite(f1) & np.isfinite(m1) & (f1 > 0) & (m1 > 0)
    valid2 = np.isfinite(f2) & np.isfinite(m2) & (f2 > 0) & (m2 > 0)

    if np.count_nonzero(valid1) < 2 or np.count_nonzero(valid2) < 2:
        raise ValueError("Not enough valid positive points to compute ETF.")

    f1v = f1[valid1]
    m1v = m1[valid1]
    f2v = f2[valid2]
    m2v = m2[valid2]

    # sort frequencies
    idx1 = np.argsort(f1v)
    idx2 = np.argsort(f2v)
    f1v, m1v = f1v[idx1], m1v[idx1]
    f2v, m2v = f2v[idx2], m2v[idx2]

    # common frequency range
    f_low = max(np.nanmin(f1v), np.nanmin(f2v))
    f_high = min(np.nanmax(f1v), np.nanmax(f2v))
    common = (f1v >= f_low) & (f1v <= f_high)

    f_ratio = f1v[common]
    m1_ratio = m1v[common]

    if f_ratio.size < 2:
        raise ValueError("No overlapping frequency range to compute ETF.")

    # interpolate PSD2 on f1 grid in log-log space
    logf2 = np.log10(f2v)
    logm2 = np.log10(m2v)
    logf_ratio = np.log10(f_ratio)

    logm2_interp = np.interp(logf_ratio, logf2, logm2)
    m2_interp = 10 ** logm2_interp

    etf = m1_ratio / m2_interp

    valid_etf = np.isfinite(etf) & (etf > 0)
    f_etf = f_ratio[valid_etf]
    etf = etf[valid_etf]
    m1_ratio = m1_ratio[valid_etf]
    m2_interp = m2_interp[valid_etf]

    if f_etf.size < 2:
        raise ValueError("Not enough strictly positive ETF points.")

    # optional frequency crop
    if fmin is not None:
        keep = f_etf >= fmin
        f_etf = f_etf[keep]
        etf = etf[keep]
        m1_ratio = m1_ratio[keep]
        m2_interp = m2_interp[keep]

    if fmax is not None:
        keep = f_etf <= fmax
        f_etf = f_etf[keep]
        etf = etf[keep]
        m1_ratio = m1_ratio[keep]
        m2_interp = m2_interp[keep]

    return f_etf, etf, m1_ratio, m2_interp