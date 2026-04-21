# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:01:34 2026

@author: mmotte
"""

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
    positive_freq_only : bool, optional
        Remove non-positive frequencies and invalid ETF values.
    max_nfev : int, optional
        Maximum number of function evaluations for least_squares.

    Returns
    -------
    freq_fit : ndarray
        Frequency vector actually used for the fit.
    etf_fit : ndarray
        Best-fit theoretical discrete ETF on freq_fit.
    param_fit : dict
        Best-fit parameters:
        {'Fao': ..., 'ki': ..., 'frame_delay': ..., 'leak': ...}
    fit_error : dict
        Error metrics:
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
    # Clean data
    # ------------------------------------------------------------------
    mask = np.isfinite(freq_exp) & np.isfinite(etf_exp)
    if positive_freq_only:
        mask &= (freq_exp > 0) & (etf_exp > 0)

    freq_fit = freq_exp[mask]
    etf_data = etf_exp[mask]

    if freq_fit.size < 4:
        raise ValueError("Not enough valid points for fitting.")

    # Sort by frequency
    idx = np.argsort(freq_fit)
    freq_fit = freq_fit[idx]
    etf_data = etf_data[idx]

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
        'Fao': max(freq_fit) if max(freq_fit) > 0 else 1000.0,
        'ki': 0.4,
        'frame_delay': 3.0,
        'leak': 0.94,
    }

    default_bounds = {
        'Fao': (max(np.max(freq_fit) / 2, 1.0), max(np.max(freq_fit) * 10, 10.0)),
        'ki': (1e-4, 10.0),
        'frame_delay': (0.0, 20.0),
        'leak': (0.0, 1.0),
    }

    if p0 is not None:
        default_p0.update(p0)
    if bounds is not None:
        default_bounds.update(bounds)

    free_names = [name for name, value in fixed_params.items() if value is None]

    if len(free_names) == 0:
        # Nothing to fit, just evaluate the model
        param_fit = fixed_params.copy()
        cl_disc = closed_loop_transfer(
            freq_fit,
            param_fit['Fao'],
            param_fit['ki'],
            param_fit['frame_delay'],
            discrete=True,
            leak=param_fit['leak']
        )
        etf_fit = np.abs(cl_disc) ** 2
    else:
        x0 = np.array([default_p0[name] for name in free_names], dtype=float)
        lb = np.array([default_bounds[name][0] for name in free_names], dtype=float)
        ub = np.array([default_bounds[name][1] for name in free_names], dtype=float)

        def unpack_params(x):
            params = fixed_params.copy()
            for name, value in zip(free_names, x):
                params[name] = float(value)
            return params

        def model_etf(x):
            params = unpack_params(x)
            cl_disc = closed_loop_transfer(
                freq_fit,
                params['Fao'],
                params['ki'],
                params['frame_delay'],
                discrete=True,
                leak=params['leak']
            )
            return np.abs(cl_disc) ** 2

        def residuals(x):
            etf_model = model_etf(x)

            # Safety against numerical issues
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
        etf_fit = model_etf(result.x)

    # ------------------------------------------------------------------
    # Error metrics
    # ------------------------------------------------------------------
    diff_lin = etf_fit - etf_data
    rmse = np.sqrt(np.mean(diff_lin**2))
    sse = np.sum(diff_lin**2)

    etf_fit_safe = np.maximum(etf_fit, 1e-20)
    etf_data_safe = np.maximum(etf_data, 1e-20)
    diff_log = np.log10(etf_fit_safe) - np.log10(etf_data_safe)
    rmse_log = np.sqrt(np.mean(diff_log**2))

    fit_error = {
        'rmse': rmse,
        'rmse_log': rmse_log,
        'sse': sse,
        'cost': 0.5 * np.sum((diff_log if fit_in_log else diff_lin) ** 2),
    }

    return freq_fit, etf_fit, param_fit, fit_error