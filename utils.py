# -*- coding: utf-8 -*-

import numpy as np

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