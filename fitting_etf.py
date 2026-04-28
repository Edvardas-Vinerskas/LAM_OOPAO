# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:01:34 2026

@author: mmotte
"""

import numpy as np
from scipy.optimize import least_squares
from aopera.control import closed_loop_transfer


import numpy as np
from scipy.optimize import least_squares
from aopera.control import closed_loop_transfer


def fit_etf_discrete(
    freq_exp,
    etf_exp,
    Fao=None,
    ki=None,
    frame_delay=None,
    leak=None,
    p0=None,
    bounds=None,
    fit_in_log=True,
    fit_freq_window=None,
    positive_freq_only=True,
    max_nfev=5000,
):
    """
    Fit an experimental ETF with the theoretical discrete ETF model.

    Parameters
    ----------
    freq_exp : array-like
        Experimental frequency vector [Hz].
    etf_exp : array-like
        Experimental ETF values (typically |ETF|^2).
    Fao, ki, frame_delay, leak : float or None, optional
        If not None, parameter is fixed during the fit.
        If None, parameter is fitted.
    p0 : dict or None, optional
        Initial guesses for free parameters, e.g.
        {'Fao': 1000, 'ki': 0.4, 'frame_delay': 3, 'leak': 0.94}
    bounds : dict or None, optional
        Bounds for free parameters, e.g.
        {
            'Fao': (100, 5000),
            'ki': (0.01, 2.0),
            'frame_delay': (0.0, 10.0),
            'leak': (0.0, 1.0),
        }
    fit_in_log : bool, optional
        If True, fit log10(ETF). Recommended.
    fit_freq_window : tuple/list or None, optional
        Frequency crop used ONLY for the fit, e.g. (10, 250).
        The returned freq_fit and etf_fit are still computed on the full input grid.
    positive_freq_only : bool, optional
        Remove non-positive frequencies and invalid ETF values.
    max_nfev : int, optional
        Maximum number of function evaluations for least_squares.

    Returns
    -------
    freq_fit : ndarray
        Full frequency vector used to evaluate the final fitted ETF.
    etf_fit : ndarray
        Best-fit theoretical discrete ETF on freq_fit.
    param_fit : dict
        Best-fit parameters:
        {'Fao': ..., 'ki': ..., 'frame_delay': ..., 'leak': ...}
    fit_error : dict
        Error metrics computed on the cropped fit region:
        {
            'rmse': ...,
            'rmse_log': ...,
            'sse': ...,
            'cost': ...
        }
    """

    freq_exp = np.asarray(freq_exp, dtype=float).ravel()
    etf_exp = np.asarray(etf_exp, dtype=float).ravel()

    if freq_exp.shape != etf_exp.shape:
        raise ValueError("freq_exp and etf_exp must have the same shape.")

    # ------------------------------------------------------------------
    # Clean full data
    # ------------------------------------------------------------------
    mask_full = np.isfinite(freq_exp) & np.isfinite(etf_exp)
    if positive_freq_only:
        mask_full &= (freq_exp > 0) & (etf_exp > 0)

    freq_full = freq_exp[mask_full]
    etf_full = etf_exp[mask_full]

    if freq_full.size < 4:
        raise ValueError("Not enough valid points in input arrays.")

    idx_full = np.argsort(freq_full)
    freq_full = freq_full[idx_full]
    etf_full = etf_full[idx_full]

    # ------------------------------------------------------------------
    # Build crop mask for fit only
    # ------------------------------------------------------------------
    mask_fit = np.ones_like(freq_full, dtype=bool)

    if fit_freq_window is not None:
        if len(fit_freq_window) != 2:
            raise ValueError("fit_freq_window must be a tuple/list like (fmin, fmax).")

        fmin, fmax = fit_freq_window

        if fmin is not None:
            mask_fit &= (freq_full >= fmin)
        if fmax is not None:
            mask_fit &= (freq_full <= fmax)

    freq_data = freq_full[mask_fit]
    etf_data = etf_full[mask_fit]

    if freq_data.size < 4:
        raise ValueError("Not enough valid points in the selected fit frequency window.")

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------
    fixed_params = {
        'Fao': Fao,
        'ki': ki,
        'frame_delay': frame_delay,
        'leak': leak,
    }

    default_p0 = {
        'Fao': max(freq_full) if max(freq_full) > 0 else 1000.0,
        'ki': 0.4,
        'frame_delay': 3.0,
        'leak': 0.94,
    }

    default_bounds = {
        'Fao': (max(np.max(freq_full) / 2, 1.0), max(np.max(freq_full) * 10, 10.0)),
        'ki': (1e-4, 10.0),
        'frame_delay': (0.0, 20.0),
        'leak': (0.0, 1.0),
    }

    if p0 is not None:
        default_p0.update(p0)
    if bounds is not None:
        default_bounds.update(bounds)

    free_names = [name for name, value in fixed_params.items() if value is None]

    def unpack_params(x=None):
        params = fixed_params.copy()
        if x is not None:
            for name, value in zip(free_names, x):
                params[name] = float(value)
        return params

    def evaluate_model(freq, params):
        cl_disc = closed_loop_transfer(
            freq,
            params['Fao'],
            params['ki'],
            params['frame_delay'],
            discrete=True,
            leak=params['leak']
        )
        return np.abs(cl_disc) ** 2

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    if len(free_names) == 0:
        param_fit = unpack_params()
    else:
        x0 = np.array([default_p0[name] for name in free_names], dtype=float)
        lb = np.array([default_bounds[name][0] for name in free_names], dtype=float)
        ub = np.array([default_bounds[name][1] for name in free_names], dtype=float)

        def residuals(x):
            params = unpack_params(x)
            etf_model = evaluate_model(freq_data, params)

            etf_model = np.maximum(etf_model, 1e-20)
            etf_obs = np.maximum(etf_data, 1e-20)

            if fit_in_log:
                return np.log10(etf_model) - np.log10(etf_obs)
            return etf_model - etf_obs

        result = least_squares(
            residuals,
            x0=x0,
            bounds=(lb, ub),
            max_nfev=max_nfev,
        )

        param_fit = unpack_params(result.x)

    # ------------------------------------------------------------------
    # Final model evaluated on FULL frequency grid
    # ------------------------------------------------------------------
    freq_fit = freq_full.copy()
    etf_fit = evaluate_model(freq_fit, param_fit)

    # ------------------------------------------------------------------
    # Error metrics on FIT REGION ONLY
    # ------------------------------------------------------------------
    etf_fit_crop = evaluate_model(freq_data, param_fit)

    diff_lin = etf_fit_crop - etf_data
    rmse = np.sqrt(np.mean(diff_lin**2))
    sse = np.sum(diff_lin**2)

    etf_fit_safe = np.maximum(etf_fit_crop, 1e-20)
    etf_data_safe = np.maximum(etf_data, 1e-20)
    diff_log = np.log10(etf_fit_safe) - np.log10(etf_data_safe)
    rmse_log = np.sqrt(np.mean(diff_log**2))

    fit_error = {
        'rmse': rmse,
        'rmse_log': rmse_log,
        'sse': sse,
        'cost': 0.5 * np.sum((diff_log if fit_in_log else diff_lin) ** 2),
        'n_points_fit': freq_data.size,
        'fit_freq_window': fit_freq_window,
    }

    return freq_fit, etf_fit, param_fit, fit_error