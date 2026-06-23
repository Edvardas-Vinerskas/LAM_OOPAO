# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:08:07 2026

@author: mmotte
"""
import numpy as np  # Import NumPy for numerical operations


import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from pathlib import Path
import pickle
from matplotlib import colors
from matplotlib.ticker import MaxNLocator


def _block_average_1d(x, N):
    
    x = np.asarray(x).ravel()

    if N is None or N <= 1:
        return x

    n = len(x) // N
    if n == 0:
        return x

    x_trim = x[:n * N]
    return x_trim.reshape(n, N).mean(axis=1)


def _smooth_time_series_block(time, y, N):

    time = np.asarray(time).ravel()
    y = np.asarray(y).ravel()

    if len(time) != len(y):
        raise ValueError("time and y must have the same length.")

    if N is None or N <= 1:
        return time, y

    n = len(y) // N
    if n == 0:
        return time, y

    time_trim = time[:n * N]
    y_trim = y[:n * N]

    time_s = time_trim.reshape(n, N).mean(axis=1)
    y_s = y_trim.reshape(n, N).mean(axis=1)

    return time_s, y_s


def _legend_loc_from_mode(legend_loc):
    """Return a Matplotlib-compatible legend location.

    Accepted values include Matplotlib locations and a few convenient aliases:
    - "free", "libre", "auto", "best" -> "best"
    - "upper right", "haut droite", "top right" -> "upper right"
    - "upper left", "haut gauche", "top left" -> "upper left"
    - "lower right", "bas droite", "bottom right" -> "lower right"
    - "lower left", "bas gauche", "bottom left" -> "lower left"
    """
    if legend_loc is None:
        return "best"

    loc = str(legend_loc).strip().lower().replace("_", "-")
    loc = " ".join(loc.replace("-", " " ).split())

    aliases = {
        "free": "best",
        "libre": "best",
        "auto": "best",
        "best": "best",
        "upper right": "upper right",
        "top right": "upper right",
        "haut droite": "upper right",
        "haut a droite": "upper right",
        "haut à droite": "upper right",
        "upper left": "upper left",
        "top left": "upper left",
        "haut gauche": "upper left",
        "haut a gauche": "upper left",
        "haut à gauche": "upper left",
        "lower right": "lower right",
        "bottom right": "lower right",
        "bas droite": "lower right",
        "bas a droite": "lower right",
        "bas à droite": "lower right",
        "lower left": "lower left",
        "bottom left": "lower left",
        "bas gauche": "lower left",
        "bas a gauche": "lower left",
        "bas à gauche": "lower left",
        "center": "center",
        "center left": "center left",
        "center right": "center right",
        "upper center": "upper center",
        "lower center": "lower center",
    }

    if loc not in aliases:
        raise ValueError(
            "legend_loc must be one of: 'free'/'best', 'upper right', "
            "'upper left', 'lower right', or 'lower left'. "
            "French aliases such as 'haut droite' and 'bas gauche' are also accepted."
        )

    return aliases[loc]



def _axis_scale_from_mode(scale="log"):
    """Return Matplotlib x/y scales from a compact scale mode.

    Accepted values are:
    - "linear": linear x and linear y
    - "xlog"  : logarithmic x and linear y
    - "ylog"  : linear x and logarithmic y
    - "log" or "loglog" : logarithmic x and logarithmic y
    """

    if scale is None:
        scale = "log"

    scale = str(scale).strip().lower()

    if scale == "linear":
        return "linear", "linear"
    if scale == "xlog":
        return "log", "linear"
    if scale == "ylog":
        return "linear", "log"
    if scale in {"log", "loglog"}:
        return "log", "log"

    raise ValueError(
        "scale must be one of 'linear', 'xlog', 'ylog', 'log', or 'loglog'."
    )


def _valid_xy_for_scale(x, y, xscale, yscale):
    """Finite filtering, with positivity enforced only on logarithmic axes."""

    valid = np.isfinite(x) & np.isfinite(y)

    if xscale == "log":
        valid &= (x > 0)

    if yscale == "log":
        valid &= (y > 0)

    return valid

def plot_psd_aa(
    f1,
    psd1,
    f2=None,
    psd2=None,
    label1="Closed loop",
    label2="Open loop",
    method=np.nansum,
    f_label="Hz",
    psd_label=r"nm$^2$/Hz",
    fmin=None,
    fmax=None,
    ylim = None,
    scale="log",
    normalised=False,
    compute_etf=True,
    name_etf = "ETF",
    show_legend=True,
    legend_loc="lower left",
    one_column=True,
    dpi=300,
    save=False,
    savepath="mean_psd_aa.pdf",
    saveformat=None,
    journal_style=True,   # True: A&A final style ; False: working style with light grid
    etf_scale=None,       # None: same scale as PSD; otherwise "linear", "xlog", "ylog", "log"/"loglog"
    etf_ref_one = True,
    etf_vmax = None,
):
    # ---------- input ----------
    label_fs = 9
    tick_fs = 8
    legend_fs = 8

    xscale, yscale = _axis_scale_from_mode(scale)
    etf_xscale, etf_yscale = _axis_scale_from_mode(scale if etf_scale is None else etf_scale)

    f1 = np.asarray(f1).ravel()
    psd1 = np.asarray(psd1)

    if psd1.ndim == 1:
        m1 = psd1
    else:
        m1 = method(psd1, axis=1)

    if normalised:
        m1 = m1 / np.nanmax(m1)

    has_second_curve = (f2 is not None) and (psd2 is not None)
    if has_second_curve:
        f2 = np.asarray(f2).ravel()
        psd2 = np.asarray(psd2)

        if psd2.ndim == 1:
            m2 = psd2
        else:
            m2 = method(psd2, axis=1)

        if normalised:
            m2 = m2 / np.nanmax(m2)
    else:
        m2 = None

    if compute_etf and not has_second_curve:
        raise ValueError(
            "compute_etf=True requires both f2 and psd2 to be provided."
        )

    # ---------- figure size ----------
    width_in = 88 / 25.4 if one_column else 180 / 25.4

    if compute_etf and has_second_curve:
        height_in = width_in * 0.95
        fig, axes = plt.subplots(
            2, 1,
            figsize=(width_in, height_in),
            dpi=dpi,
            constrained_layout=True,
            sharex=(xscale == etf_xscale),
            gridspec_kw={"height_ratios": [1.0, 2.2]}
        )
        ax_etf, ax = axes
    else:
        height_in = width_in * 0.72
        fig, ax = plt.subplots(
            figsize=(width_in, height_in),
            constrained_layout=True,
            dpi=dpi
        )
        ax_etf = None
        axes = [ax]

    # ---------- curves ----------
    col1 = "black"
    col2 = "#355C9A"   # muted dark blue

    valid1_plot = _valid_xy_for_scale(f1, m1, xscale, yscale)

    ax.plot(
        f1[valid1_plot], m1[valid1_plot],
        color=col1,
        lw=1.6,
        ls="-",
        label=label1,
        zorder=3,
        solid_capstyle="round",
    )

    if has_second_curve:
        valid2_plot = _valid_xy_for_scale(f2, m2, xscale, yscale)

        ax.plot(
            f2[valid2_plot], m2[valid2_plot],
            color=col2,
            lw=1.6,
            ls=(0, (7, 3)),
            label=label2,
            zorder=3,
            dash_capstyle="butt",
        )

    # ---------- ETF subplot ----------
    if compute_etf and has_second_curve:
        # valid points for interpolation in log-log space
        valid1 = np.isfinite(f1) & np.isfinite(m1) & (f1 > 0) & (m1 > 0)
        valid2 = np.isfinite(f2) & np.isfinite(m2) & (f2 > 0) & (m2 > 0)

        if np.count_nonzero(valid1) < 2 or np.count_nonzero(valid2) < 2:
            raise ValueError("Not enough valid positive points to compute ETF.")

        f1v = f1[valid1]
        m1v = m1[valid1]
        f2v = f2[valid2]
        m2v = m2[valid2]

        # sort in case the input is not strictly ordered
        idx1 = np.argsort(f1v)
        idx2 = np.argsort(f2v)
        f1v, m1v = f1v[idx1], m1v[idx1]
        f2v, m2v = f2v[idx2], m2v[idx2]

        # restrict to common frequency range
        f_low = max(np.nanmin(f1v), np.nanmin(f2v))
        f_high = min(np.nanmax(f1v), np.nanmax(f2v))
        common = (f1v >= f_low) & (f1v <= f_high)

        f_ratio = f1v[common]
        m1_ratio = m1v[common]

        if f_ratio.size < 2:
            raise ValueError("No overlapping frequency range to compute ETF.")

        # interpolate m2 on f1 grid in log-log space
        logf2 = np.log10(f2v)
        logm2 = np.log10(m2v)
        logf_ratio = np.log10(f_ratio)

        logm2_interp = np.interp(logf_ratio, logf2, logm2)
        m2_interp = 10**logm2_interp

        etf = m1_ratio / m2_interp

        valid_etf = _valid_xy_for_scale(f_ratio, etf, etf_xscale, etf_yscale)
        f_ratio_plot = f_ratio[valid_etf]
        etf_plot = etf[valid_etf]
        
        if f_ratio_plot.size < 2:
            raise ValueError("Not enough valid ETF points to display with the requested ETF scale.")
         
        ax_etf.plot(
            f_ratio_plot, etf_plot,
            color="black",
            lw=1.2,
            zorder=3,
        )
        
        # reference line ETF = 1 over the displayed x-range
        if etf_ref_one:
            ax_etf.plot(
                f_ratio_plot,
                np.ones_like(f_ratio_plot),
                color="0.4",
                lw=0.9,
                ls="--",
                zorder=2,
            )

        ax_etf.set_xscale(etf_xscale)
        ax_etf.set_yscale(etf_yscale)
        ax_etf.set_ylabel(name_etf, fontsize=label_fs)
        ax_etf.tick_params(
            which="major", direction="in", length=5, width=1.0, labelsize=tick_fs, pad=4
        )
        ax_etf.tick_params(
            which="minor", direction="in", length=3, width=0.8
        )

        for spine in ax_etf.spines.values():
            spine.set_linewidth(1.0)

        if journal_style:
            ax_etf.grid(False)
        else:
            ax_etf.grid(True, which="major", color="0.88", lw=0.6)
            ax_etf.grid(True, which="minor", color="0.93", lw=0.4)

    # ---------- axes ----------
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(ylim)
    # ---------- limits ----------
    if fmin is not None or fmax is not None:
        ax.set_xlim(left=fmin, right=fmax)
        if ax_etf is not None:
            ax_etf.set_xlim(left=fmin, right=fmax)
        if etf_vmax is not None:
            ax_etf.set_ylim(None, top = etf_vmax)
    # ---------- labels ----------
    ax.set_xlabel(f"{f_label}", fontsize=label_fs)
    if normalised:
        ax.set_ylabel("Normalised PSD", fontsize=label_fs)
    else:
        ax.set_ylabel(f"{psd_label}", fontsize=label_fs)

    # ---------- ticks / frame ----------
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.tick_params(which="major", direction="in", length=6, width=1.0, labelsize=tick_fs, pad=6)
    ax.tick_params(which="minor", direction="in", length=3.5, width=0.8)

    # ---------- grid ----------
    if journal_style:
        ax.grid(False)
    else:
        ax.grid(True, which="major", color="0.88", lw=0.6)
        ax.grid(True, which="minor", color="0.93", lw=0.4)

    # ---------- legend ----------
    if show_legend and has_second_curve:
        ax.legend(
            frameon=False,
            fontsize=legend_fs,
            loc=_legend_loc_from_mode(legend_loc),
            handlelength=3.2,
            borderaxespad=0.4,
        )

    # ---------- save ----------
    if save:
        path = Path(savepath)

        # default format
        if saveformat is None:
            if path.suffix:
                saveformat = path.suffix.lower().lstrip(".")
            else:
                saveformat = "pdf"
                path = path.with_suffix(".pdf")

        # create parent directory automatically if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # helper function to save in one format
        def _save_one(fmt):
            fmt = fmt.lower().lstrip(".")

            if fmt == "fig":
                # Python-specific serialized matplotlib figure
                fig_path = path.with_suffix(".fig")
                with open(fig_path, "wb") as f:
                    pickle.dump(fig, f)

            elif fmt in {"pdf", "eps", "svg"}:
                out = path.with_suffix(f".{fmt}")
                fig.savefig(
                    out,
                    format=fmt,
                    bbox_inches="tight",
                    pad_inches=0.02,
                    dpi=max(dpi, 600),
                    transparent=False,
                )

            elif fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
                out = path.with_suffix(f".{fmt}")
                fig.savefig(
                    out,
                    format=fmt,
                    dpi=max(dpi, 600),
                    bbox_inches="tight",
                    pad_inches=0.02,
                    transparent=False,
                )

            else:
                raise ValueError(
                    f"Unsupported save format '{fmt}'. "
                    "Use pdf, eps, png, tiff, jpg, jpeg, svg, fig, or all."
                )

        # save all requested formats
        if saveformat.lower() == "all":
            for fmt in ("png", "pdf", "fig"):
                _save_one(fmt)
        else:
            _save_one(saveformat)

    return fig, (ax_etf, ax) if ax_etf is not None else ax

#%%


def plot_sr_aa(
    time_cl,
    SR_cl,
    time_ol,
    SR_ol,
    SR_cl_cam,
    SR_ol_cam,
    label_cl="Closed loop",
    label_ol="Open loop",
    lambda_wfs_nm=1600,
    lambda_img_nm=None,
    time_unit="s",
    one_column=True,
    dpi=300,
    save=False,
    savepath="strehl_vs_time.pdf",
    saveformat=None,
    journal_style=True,
    show_legend=True,
    legend_loc="lower right",
    smooth_N=None,              # <--- nouvelle option
):
   

    # --- Lissage éventuel ---
    time_cl_s, SR_cl_s = _smooth_time_series_block(time_cl, SR_cl, smooth_N)
    time_ol_s, SR_ol_s = _smooth_time_series_block(time_ol, SR_ol, smooth_N)
    time_cl_cam_s, SR_cl_cam_s = _smooth_time_series_block(time_cl, SR_cl_cam, smooth_N)
    time_ol_cam_s, SR_ol_cam_s = _smooth_time_series_block(time_ol, SR_ol_cam, smooth_N)

    # --- Paramètres de style A&A ---
    label_fs = 9
    tick_fs = 8
    legend_fs = 8

    width_in = 88 / 25.4 if one_column else 180 / 25.4
    height_in = width_in * 1.15

    fig, axes = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(width_in, height_in),
        dpi=dpi,
        constrained_layout=True
    )

    ax1, ax2 = axes

    # Couleurs / styles
    col_cl = "black"
    col_ol = "#355C9A"
    ls_cl = "-"
    ls_ol = (0, (7, 3))

    # =========================
    # (a) SR à lambda_wfs_nm
    # =========================
    ax1.plot(
        time_cl_s, SR_cl_s,
        color=col_cl, lw=1.5, ls=ls_cl,
        label=label_cl, zorder=3, solid_capstyle="round"
    )
    ax1.plot(
        time_ol_s, SR_ol_s,
        color=col_ol, lw=1.5, ls=ls_ol,
        label=label_ol, zorder=3, dash_capstyle="butt"
    )

    ax1.set_ylabel("SR", fontsize=label_fs)
    ax1.set_ylim(0, 1.02)

    title1 = rf"(a) SR at {lambda_wfs_nm:.0f} nm (vZWFS)"
    

    ax1.text(
        0.02, 0.95, title1,
        transform=ax1.transAxes,
        ha="left", va="top",
        fontsize=label_fs
    )

    if show_legend:
        ax1.legend(
            frameon=False,
            fontsize=legend_fs,
            loc=_legend_loc_from_mode(legend_loc),
            handlelength=3.2,
            borderaxespad=0.4,
        )

    # =========================
    # (b) SR à lambda_img_nm
    # =========================
    ax2.plot(
        time_cl_cam_s, SR_cl_cam_s,
        color=col_cl, lw=1.5, ls=ls_cl,
        label=label_cl, zorder=3, solid_capstyle="round"
    )
    ax2.plot(
        time_ol_cam_s, SR_ol_cam_s,
        color=col_ol, lw=1.5, ls=ls_ol,
        label=label_ol, zorder=3, dash_capstyle="butt"
    )

    ax2.set_ylabel("SR", fontsize=label_fs)
    ax2.set_xlabel(f"Time [{time_unit}]", fontsize=label_fs)
    ax2.set_ylim(0, 1.02)

    if lambda_img_nm is None:
        title2 = r"(b) SR at imaging wavelength"
    else:
        title2 = rf"(b) SR at {lambda_img_nm:.0f} nm"

    

    ax2.text(
        0.02, 0.95, title2,
        transform=ax2.transAxes,
        ha="left", va="top",
        fontsize=label_fs
    )

    # --- Ticks / cadre / grille ---
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        ax.tick_params(
            which="major",
            direction="in",
            length=5,
            width=1.0,
            labelsize=tick_fs,
            pad=4,
            top=True,
            right=True,
        )
        ax.tick_params(
            which="minor",
            direction="in",
            length=3,
            width=0.8,
            top=True,
            right=True,
        )

        if journal_style:
            ax.grid(False)
        else:
            ax.grid(True, which="major", color="0.88", lw=0.6)
            ax.grid(True, which="minor", color="0.93", lw=0.4)

    # --- Sauvegarde ---
    if save:
        path = Path(savepath)

        if saveformat is None:
            if path.suffix:
                saveformat = path.suffix.lower().lstrip(".")
            else:
                saveformat = "pdf"
                path = path.with_suffix(".pdf")

        path.parent.mkdir(parents=True, exist_ok=True)

        def _save_one(fmt):
            fmt = fmt.lower().lstrip(".")

            if fmt == "fig":
                fig_path = path.with_suffix(".fig")
                with open(fig_path, "wb") as f:
                    pickle.dump(fig, f)

            elif fmt in {"pdf", "eps", "svg"}:
                out = path.with_suffix(f".{fmt}")
                fig.savefig(
                    out,
                    format=fmt,
                    bbox_inches="tight",
                    pad_inches=0.02,
                    transparent=False,
                )

            elif fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
                out = path.with_suffix(f".{fmt}")
                fig.savefig(
                    out,
                    format=fmt,
                    dpi=max(dpi, 300),
                    bbox_inches="tight",
                    pad_inches=0.02,
                    transparent=False,
                )

            else:
                raise ValueError(
                    f"Unsupported save format '{fmt}'. "
                    "Use pdf, eps, png, tiff, jpg, jpeg, svg, fig, or all."
                )

        if saveformat.lower() == "all":
            for fmt in ("png", "pdf", "fig"):
                _save_one(fmt)
        else:
            _save_one(saveformat)

    return fig, axes




#%%
def plot_psf_grid_aa(
    psf_groups,
    wvl,
    telescope_diameter,
    sampling,
    vmin=1e-6,
    vmax=1,
    cmap="inferno",
    xlabel=r"[arcsec]",
    ylabel=r"[arcsec]",
    cbar_label="Normalized intensity",
    nx=None,
    titles=None,
    row_labels=None,
    col_labels=None,
    figsize=None,
    dpi=1200,
    origin="lower",
    save=False,
    savepath="psf_grid.pdf",
    saveformat=None,
    share_colorbar=True,
    individual_colorbars=False,
    row_colorbars=False,
    row_cbar_orientation="vertical",
    row_cbar_fraction=0.046,
    row_cbar_pad=0.04,
    row_cbar_right=0.88,
    hide_inner_labels=True,
    normalize_each=True,
    norm_mode="log",
    linthresh=1e-4,
    linscale=1.0,
    norm_clip=True,
    cbar_position=(0.18, 0.085, 0.64, 0.025),
    layout_rect=(0.0, 0.13, 1.0, 1.0),
    wspace=0.25,
    hspace=0.32,
):
    """
    Plot several PSFs as a grid with a style adapted to Astronomy & Astrophysics.

    Parameters
    ----------
    psf_groups : list[list[np.ndarray or None]]
        List of rows. Each row contains 2D PSFs or None.
        A None entry creates an empty hidden cell.

    wvl : float
        Wavelength in meters.

    telescope_diameter : float
        Telescope diameter in meters.

    sampling : float
        Sampling factor.

    vmin, vmax : float, list, or nested list
        Normalization limits.
        Can be:
        - scalar: same value for all PSFs
        - list of length nrows: one value per row
        - nested list matching psf_groups: one value per PSF

    cmap : str
        Matplotlib colormap.

    xlabel, ylabel : str
        Axis labels.

    cbar_label : str or list or nested list
        Colorbar label.
        Can be scalar, row-wise, or per-PSF.

    nx : int or None
        Central square crop size. If None, each image is kept at full size.

    titles : list[list[str or None]] or None
        Individual subplot titles, with the same structure as psf_groups.

    row_labels : list[str] or None
        Labels displayed on the first column, one per row.

    col_labels : list[str] or None
        Labels displayed above columns.
        Ignored for a cell if titles[i][j] is already defined.

    figsize : tuple or None
        Figure size in inches.

    dpi : int
        DPI used when saving raster formats.

    origin : str
        Origin passed to imshow.

    save : bool
        If True, save the figure.

    savepath : str
        Output path.

    saveformat : str or None
        Save format. If None, inferred from savepath.
        Can be "png", "pdf", "fig", "svg", "eps", "jpg", "jpeg", "tif", "tiff", or "all".

    share_colorbar : bool
        If True, use one common horizontal colorbar at the bottom.

    individual_colorbars : bool
        If True and share_colorbar=False, add one colorbar per subplot.

    row_colorbars : bool
        If True and share_colorbar=False, add one colorbar per row.

    row_cbar_orientation : {"vertical", "horizontal"}
        Orientation of row colorbars.

    row_cbar_fraction : float
        Fraction parameter passed to fig.colorbar for row colorbars.

    row_cbar_pad : float
        Padding parameter passed to fig.colorbar for row colorbars.

    row_cbar_right : float
        Right margin used when row_colorbars=True and row_cbar_orientation="vertical".
        Lower values leave more room for colorbar tick labels.

    hide_inner_labels : bool
        If True, only outer axes keep their labels.

    normalize_each : bool
        If True, each PSF is normalized by its sum and then by its maximum.
        Use False for residuals or PSF differences.

    norm_mode : {"linear", "log", "symlog"} or list or nested list
        Normalization mode.
        Can be scalar, row-wise, or per-PSF.

    linthresh : float or list or nested list
        Linear threshold for SymLogNorm.

    linscale : float or list or nested list
        Linear scale factor for SymLogNorm.

    norm_clip : bool
        If True, clip values outside vmin/vmax in the normalization.

    cbar_position : tuple
        Position of the shared bottom colorbar:
        (left, bottom, width, height), in figure-relative coordinates.

    layout_rect : tuple
        Rectangle reserved for subplots in tight_layout:
        (left, bottom, right, top).

    wspace, hspace : float
        Horizontal and vertical spacing between subplots.

    Returns
    -------
    fig, axes
        Matplotlib figure and 2D axes array.
    """

    if not isinstance(psf_groups, (list, tuple)) or len(psf_groups) == 0:
        raise ValueError("psf_groups must be a non-empty list of rows.")

    nrows = len(psf_groups)
    ncols = max(len(row) for row in psf_groups)

    if ncols == 0:
        raise ValueError("psf_groups does not contain any column.")

    n_colorbar_modes = sum(
        [
            bool(share_colorbar),
            bool(individual_colorbars),
            bool(row_colorbars),
        ]
    )

    if n_colorbar_modes > 1:
        raise ValueError(
            "share_colorbar, individual_colorbars, and row_colorbars are mutually exclusive. "
            "Choose only one colorbar mode."
        )

    if row_cbar_orientation not in {"vertical", "horizontal"}:
        raise ValueError(
            "row_cbar_orientation must be either 'vertical' or 'horizontal'."
        )

    valid_psfs = [
        psf
        for row in psf_groups
        for psf in row
        if psf is not None
    ]

    if len(valid_psfs) == 0:
        raise ValueError("No valid PSF to plot: all entries are None.")

    rc_params = {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    }

    pix_scale = wvl / telescope_diameter / sampling
    rad2arcsec = 180 / np.pi * 3600
    pix_scale_arcsec = pix_scale * rad2arcsec

    if figsize is None:
        figsize = (
            max(3.35, 2.4 * ncols),
            max(2.5, 2.3 * nrows),
        )

    def _is_sequence(obj):
        return isinstance(obj, (list, tuple, np.ndarray)) and not isinstance(obj, str)

    def _expand_to_grid(spec, name):
        """
        Expand a scalar, row-wise list, or nested list to the psf_groups grid shape.
        """
        if isinstance(spec, str) or np.isscalar(spec):
            return [
                [spec for _ in range(len(psf_groups[i]))]
                for i in range(nrows)
            ]

        if not _is_sequence(spec):
            raise TypeError(
                f"{name} must be a scalar, a string, a row-wise list, "
                f"or a nested list matching psf_groups."
            )

        if len(spec) != nrows:
            raise ValueError(
                f"{name} must have length nrows={nrows} when provided as a list."
            )

        # Row-wise case:
        # e.g. vmin=[1e-4, -1e-2]
        # e.g. norm_mode=["log", "symlog"]
        if all((not _is_sequence(x)) or isinstance(x, str) for x in spec):
            return [
                [spec[i] for _ in range(len(psf_groups[i]))]
                for i in range(nrows)
            ]

        # Per-PSF nested case:
        # e.g. vmin=[[1e-4, 1e-4], [-1e-2, -1e-3]]
        grid = []

        for i in range(nrows):
            row_spec = spec[i]

            if not _is_sequence(row_spec) or isinstance(row_spec, str):
                raise TypeError(
                    f"{name}[{i}] must be a sequence matching psf_groups[{i}]."
                )

            if len(row_spec) != len(psf_groups[i]):
                raise ValueError(
                    f"{name}[{i}] must have length {len(psf_groups[i])} "
                    f"to match psf_groups[{i}]."
                )

            grid.append(list(row_spec))

        return grid

    norm_mode_grid = _expand_to_grid(norm_mode, "norm_mode")
    vmin_grid = _expand_to_grid(vmin, "vmin")
    vmax_grid = _expand_to_grid(vmax, "vmax")
    linthresh_grid = _expand_to_grid(linthresh, "linthresh")
    linscale_grid = _expand_to_grid(linscale, "linscale")
    cbar_label_grid = _expand_to_grid(cbar_label, "cbar_label")

    def _build_norm(i, j):
        mode = str(norm_mode_grid[i][j]).lower()

        vmin_ij = vmin_grid[i][j]
        vmax_ij = vmax_grid[i][j]
        linthresh_ij = linthresh_grid[i][j]
        linscale_ij = linscale_grid[i][j]

        if mode in {"linear", "lin"}:
            return Normalize(
                vmin=vmin_ij,
                vmax=vmax_ij,
                clip=norm_clip,
            )

        if mode == "log":
            if vmin_ij is None or vmax_ij is None:
                raise ValueError(
                    f"vmin and vmax must be provided for log normalization at ({i}, {j})."
                )

            if vmin_ij <= 0 or vmax_ij <= 0:
                raise ValueError(
                    f"vmin and vmax must be strictly positive for log normalization at ({i}, {j})."
                )

            return LogNorm(
                vmin=vmin_ij,
                vmax=vmax_ij,
                clip=norm_clip,
            )

        if mode == "symlog":
            if linthresh_ij is None or linthresh_ij <= 0:
                raise ValueError(
                    f"linthresh must be strictly positive for symlog normalization at ({i}, {j})."
                )

            return SymLogNorm(
                linthresh=linthresh_ij,
                linscale=linscale_ij,
                vmin=vmin_ij,
                vmax=vmax_ij,
                clip=norm_clip,
                base=10,
            )

        raise ValueError(
            f"Unsupported norm_mode='{mode}' at ({i}, {j}). "
            "Use 'linear', 'log', or 'symlog'."
        )

    def _center_crop(arr, crop_size, i, j):
        ny0, nx0 = arr.shape

        if crop_size > ny0 or crop_size > nx0:
            raise ValueError(
                f"nx={crop_size} is too large for the PSF at position ({i}, {j}) "
                f"with shape {arr.shape}."
            )

        cy, cx = ny0 // 2, nx0 // 2
        half = crop_size // 2

        y_start = cy - half
        x_start = cx - half
        y_end = y_start + crop_size
        x_end = x_start + crop_size

        return arr[y_start:y_end, x_start:x_end]

    def _prepare_image(psf, i, j):
        psf = np.asarray(psf, dtype=float)

        if psf.ndim != 2:
            raise ValueError(
                f"The PSF at position ({i}, {j}) must be a 2D array."
            )

        if not np.any(np.isfinite(psf)):
            raise ValueError(
                f"The PSF at position ({i}, {j}) does not contain any finite value."
            )

        psf = np.nan_to_num(
            psf,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if nx is not None:
            psf = _center_crop(psf, nx, i, j)

        if normalize_each:
            psf_sum = np.sum(psf)

            if psf_sum <= 0:
                raise ValueError(
                    f"The PSF at position ({i}, {j}) has a zero or negative sum. "
                    "Use normalize_each=False for residuals or difference maps."
                )

            image = psf / psf_sum

            peak = np.max(image)

            if peak <= 0:
                raise ValueError(
                    f"The PSF at position ({i}, {j}) has a zero or negative maximum."
                )

            image = image / peak

        else:
            image = psf.copy()

        mode = str(norm_mode_grid[i][j]).lower()
        vmin_ij = vmin_grid[i][j]

        # LogNorm cannot handle values <= 0.
        # We avoid NaNs by clipping values below vmin to vmin.
        if mode == "log":
            if vmin_ij is None or vmin_ij <= 0:
                raise ValueError(
                    f"vmin must be strictly positive for log normalization at ({i}, {j})."
                )

            image = np.clip(image, vmin_ij, None)

        ny, nx_img = image.shape

        x_axis = (np.arange(nx_img) - (nx_img - 1) / 2) * pix_scale_arcsec
        y_axis = (np.arange(ny) - (ny - 1) / 2) * pix_scale_arcsec

        extent = [
            x_axis[0],
            x_axis[-1],
            y_axis[0],
            y_axis[-1],
        ]

        return image, extent

    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
        )

        images = []
        image_axes = []

        row_images = [[] for _ in range(nrows)]
        row_axes = [[] for _ in range(nrows)]

        for i in range(nrows):
            row = psf_groups[i]

            for j in range(ncols):
                ax = axes[i, j]

                if j >= len(row) or row[j] is None:
                    ax.set_visible(False)
                    continue

                image, extent = _prepare_image(row[j], i, j)
                norm_ij = _build_norm(i, j)

                im = ax.imshow(
                    image,
                    norm=norm_ij,
                    cmap=cmap,
                    extent=extent,
                    origin=origin,
                )

                images.append(im)
                image_axes.append(ax)

                row_images[i].append(im)
                row_axes[i].append(ax)

                title_set = False

                if titles is not None:
                    if i < len(titles) and j < len(titles[i]):
                        if titles[i][j] is not None:
                            ax.set_title(titles[i][j])
                            title_set = True

                if not title_set and col_labels is not None:
                    if i == 0 and j < len(col_labels):
                        ax.set_title(col_labels[j])

                if hide_inner_labels:
                    if i == nrows - 1:
                        ax.set_xlabel(xlabel)
                    else:
                        ax.set_xlabel("")
                        ax.set_xticklabels([])

                    if j == 0:
                        if row_labels is not None and i < len(row_labels):
                            ax.set_ylabel(row_labels[i])
                        else:
                            ax.set_ylabel(ylabel)
                    else:
                        ax.set_ylabel("")
                        ax.set_yticklabels([])

                else:
                    ax.set_xlabel(xlabel)

                    if row_labels is not None and j == 0 and i < len(row_labels):
                        ax.set_ylabel(row_labels[i])
                    else:
                        ax.set_ylabel(ylabel)

        if len(images) == 0:
            raise ValueError("No valid image has been plotted.")

        if share_colorbar:
            fig.tight_layout(rect=layout_rect)
            fig.subplots_adjust(wspace=wspace, hspace=hspace)

            cax = fig.add_axes(cbar_position)

            cbar = fig.colorbar(
                images[0],
                cax=cax,
                orientation="horizontal",
            )
            cbar.set_label(cbar_label_grid[0][0])
            cbar.ax.tick_params(direction="in", pad=2, labelsize=7)

        else:
            fig.tight_layout()
            fig.subplots_adjust(wspace=wspace, hspace=hspace)

            if individual_colorbars:
                for im, ax in zip(images, image_axes):
                    cbar = fig.colorbar(
                        im,
                        ax=ax,
                        fraction=0.046,
                        pad=0.04,
                    )
                    cbar.set_label(cbar_label_grid[0][0])
                    cbar.ax.tick_params(direction="in", pad=2, labelsize=7)

            elif row_colorbars:
                if row_cbar_orientation == "vertical":
                    fig.subplots_adjust(right=row_cbar_right)

                for i in range(nrows):
                    if len(row_images[i]) == 0:
                        continue

                    cbar = fig.colorbar(
                        row_images[i][0],
                        ax=row_axes[i],
                        orientation=row_cbar_orientation,
                        fraction=row_cbar_fraction,
                        pad=row_cbar_pad,
                    )

                    # Use the label of the first valid PSF in the row.
                    first_valid_j = None
                    for j in range(len(psf_groups[i])):
                        if psf_groups[i][j] is not None:
                            first_valid_j = j
                            break

                    if first_valid_j is not None:
                        cbar.set_label(cbar_label_grid[i][first_valid_j])
                    else:
                        cbar.set_label(cbar_label)

                    if row_cbar_orientation == "vertical":
                        cbar.ax.tick_params(
                            axis="y",
                            direction="in",
                            pad=2,
                            labelsize=7,
                        )
                    else:
                        cbar.ax.tick_params(
                            axis="x",
                            direction="in",
                            pad=2,
                            labelsize=7,
                        )

        if save:
            path = Path(savepath)

            if saveformat is None:
                if path.suffix:
                    saveformat = path.suffix.lower().lstrip(".")
                else:
                    saveformat = "pdf"
                    path = path.with_suffix(".pdf")

            path.parent.mkdir(parents=True, exist_ok=True)

            def _save_one(fmt):
                fmt = fmt.lower().lstrip(".")

                if fmt == "fig":
                    fig_path = path.with_suffix(".fig")
                    with open(fig_path, "wb") as f:
                        pickle.dump(fig, f)

                elif fmt in {"pdf", "eps", "svg"}:
                    out = path.with_suffix(f".{fmt}")
                    fig.savefig(
                        out,
                        format=fmt,
                        bbox_inches="tight",
                        pad_inches=0.02,
                        transparent=False,
                    )

                elif fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
                    out = path.with_suffix(f".{fmt}")
                    fig.savefig(
                        out,
                        format=fmt,
                        dpi=max(dpi, 300),
                        bbox_inches="tight",
                        pad_inches=0.02,
                        transparent=False,
                    )

                else:
                    raise ValueError(
                        f"Unsupported save format '{fmt}'. "
                        "Use pdf, eps, png, tiff, jpg, jpeg, svg, fig, or all."
                    )

            if saveformat.lower() == "all":
                for fmt in ("png", "pdf", "fig"):
                    _save_one(fmt)
            else:
                _save_one(saveformat)

    return fig, axes
# def plot_psf_grid_aa(
#     psf_groups,
#     wvl,
#     telescope_diameter,
#     sampling,
#     vmin=1e-6,
#     vmax=1,
#     cmap="inferno",
#     xlabel=r"[arcsec]",
#     ylabel=r"[arcsec]",
#     cbar_label="Normalized intensity",
#     nx=None,
#     titles=None,
#     row_labels=None,
#     col_labels=None,
#     figsize=None,
#     dpi=1200,
#     origin="lower",
#     save=False,
#     savepath="psf_grid.pdf",
#     saveformat=None,
#     share_colorbar=True,
#     individual_colorbars=False,
#     row_colorbars=False,
#     row_cbar_orientation="vertical",
#     row_cbar_fraction=0.046,
#     row_cbar_pad=0.04,
#     hide_inner_labels=True,
#     normalize_each=True,
#     use_log_norm=True,
#     cbar_position=(0.18, 0.085, 0.64, 0.025),
#     layout_rect=(0.0, 0.13, 1.0, 1.0),
#     wspace=0.25,
#     hspace=0.32,
# ):
#     """
#     Plot several PSFs as a grid with a style adapted to Astronomy & Astrophysics.

#     Parameters
#     ----------
#     psf_groups : list[list[np.ndarray or None]]
#         List of rows. Each row contains 2D PSFs or None.
#         A None entry creates an empty hidden cell.

#     wvl : float
#         Wavelength in meters.

#     telescope_diameter : float
#         Telescope diameter in meters.

#     sampling : float
#         Sampling factor.

#     vmin, vmax : float
#         Display normalization limits.

#     cmap : str
#         Matplotlib colormap.

#     xlabel, ylabel : str
#         Axis labels.

#     cbar_label : str
#         Colorbar label.

#     nx : int or None
#         Central square crop size. If None, each image is kept at full size.

#     titles : list[list[str or None]] or None
#         Individual subplot titles, with the same structure as psf_groups.

#     row_labels : list[str] or None
#         Labels displayed on the first column, one per row.

#     col_labels : list[str] or None
#         Labels displayed above columns.
#         Ignored for a cell if titles[i][j] is already defined.

#     figsize : tuple or None
#         Figure size in inches.

#     dpi : int
#         DPI used when saving raster formats.

#     origin : str
#         Origin passed to imshow.

#     save : bool
#         If True, save the figure.

#     savepath : str
#         Output path.

#     saveformat : str or None
#         Save format. If None, inferred from savepath.
#         Can be "png", "pdf", "fig", "svg", "eps", "jpg", "jpeg", "tif", "tiff", or "all".

#     share_colorbar : bool
#         If True, use one common horizontal colorbar at the bottom.

#     individual_colorbars : bool
#         If True and share_colorbar=False, add one colorbar per subplot.

#     row_colorbars : bool
#         If True and share_colorbar=False, add one colorbar per row.

#     row_cbar_orientation : {"vertical", "horizontal"}
#         Orientation of row colorbars.

#     row_cbar_fraction : float
#         Fraction parameter passed to fig.colorbar for row colorbars.

#     row_cbar_pad : float
#         Padding parameter passed to fig.colorbar for row colorbars.

#     hide_inner_labels : bool
#         If True, only outer axes keep their labels.

#     normalize_each : bool
#         If True, each PSF is normalized by its sum and then by its maximum.
#         This reproduces the behavior of the original plot_psf_aa function.

#         If False, images are displayed as provided.

#     use_log_norm : bool
#         If True, use LogNorm(vmin, vmax).
#         If False, use linear normalization with vmin/vmax.

#     cbar_position : tuple
#         Position of the shared bottom colorbar:
#         (left, bottom, width, height), in figure-relative coordinates.

#     layout_rect : tuple
#         Rectangle reserved for subplots in tight_layout:
#         (left, bottom, right, top).
#         The bottom value should be above the shared colorbar vertical position.

#     wspace, hspace : float
#         Horizontal and vertical spacing between subplots.

#     Returns
#     -------
#     fig, axes
#         Matplotlib figure and 2D axes array.
#     """

#     if not isinstance(psf_groups, (list, tuple)) or len(psf_groups) == 0:
#         raise ValueError("psf_groups must be a non-empty list of rows.")

#     nrows = len(psf_groups)
#     ncols = max(len(row) for row in psf_groups)

#     if ncols == 0:
#         raise ValueError("psf_groups does not contain any column.")

#     n_colorbar_modes = sum(
#         [
#             bool(share_colorbar),
#             bool(individual_colorbars),
#             bool(row_colorbars),
#         ]
#     )

#     if n_colorbar_modes > 1:
#         raise ValueError(
#             "share_colorbar, individual_colorbars, and row_colorbars are mutually exclusive. "
#             "Choose only one colorbar mode."
#         )

#     if row_cbar_orientation not in {"vertical", "horizontal"}:
#         raise ValueError(
#             "row_cbar_orientation must be either 'vertical' or 'horizontal'."
#         )

#     valid_psfs = [
#         psf
#         for row in psf_groups
#         for psf in row
#         if psf is not None
#     ]

#     if len(valid_psfs) == 0:
#         raise ValueError("No valid PSF to plot: all entries are None.")

#     rc_params = {
#         "font.size": 8,
#         "axes.labelsize": 8,
#         "axes.titlesize": 8,
#         "xtick.labelsize": 7,
#         "ytick.labelsize": 7,
#         "legend.fontsize": 7,
#         "font.family": "serif",
#         "mathtext.fontset": "cm",
#         "axes.linewidth": 0.8,
#         "xtick.direction": "in",
#         "ytick.direction": "in",
#         "xtick.top": True,
#         "ytick.right": True,
#     }

#     pix_scale = wvl / telescope_diameter / sampling
#     rad2arcsec = 180 / np.pi * 3600
#     pix_scale_arcsec = pix_scale * rad2arcsec

   

#     if figsize is None:
#         figsize = (
#             max(3.35, 2.4 * ncols),
#             max(2.5, 2.3 * nrows),
#         )
#     def _as_row_values(value, name):
#         """
#         Convert a scalar or a row-wise list/tuple/array into a list of length nrows.
#         """
#         if np.isscalar(value):
#             return [value] * nrows
    
#         if isinstance(value, (list, tuple, np.ndarray)):
#             if len(value) != nrows:
#                 raise ValueError(
#                     f"{name} must be either a scalar or a list/tuple/array "
#                     f"with length nrows={nrows}."
#                 )
#             return list(value)
    
#         raise TypeError(
#             f"{name} must be either a scalar or a list/tuple/array."
#         )
    
    
#     vmin_rows = _as_row_values(vmin, "vmin")
#     vmax_rows = _as_row_values(vmax, "vmax")

#     def _center_crop(arr, crop_size, i, j):
#         ny0, nx0 = arr.shape

#         if crop_size > ny0 or crop_size > nx0:
#             raise ValueError(
#                 f"nx={crop_size} is too large for the PSF at position ({i}, {j}) "
#                 f"with shape {arr.shape}."
#             )

#         cy, cx = ny0 // 2, nx0 // 2
#         half = crop_size // 2

#         y_start = cy - half
#         x_start = cx - half
#         y_end = y_start + crop_size
#         x_end = x_start + crop_size

#         return arr[y_start:y_end, x_start:x_end]

#     def _prepare_image(psf, i, j, vmin_row=None):
#         psf = np.asarray(psf, dtype=float)

#         if psf.ndim != 2:
#             raise ValueError(
#                 f"The PSF at position ({i}, {j}) must be a 2D array."
#             )

#         if not np.any(np.isfinite(psf)):
#             raise ValueError(
#                 f"The PSF at position ({i}, {j}) does not contain any finite value."
#             )

#         psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)

#         if nx is not None:
#             psf = _center_crop(psf, nx, i, j)

#         if normalize_each:
#             psf_sum = np.nansum(psf)

#             if psf_sum <= 0:
#                 raise ValueError(
#                     f"The PSF at position ({i}, {j}) has a zero or negative sum."
#                 )

#             image = psf / psf_sum

#             peak = np.nanmax(image)
#             if peak <= 0:
#                 raise ValueError(
#                     f"The PSF at position ({i}, {j}) has a zero or negative maximum."
#                 )

#             image = image / peak

#         else:
#             image = psf.copy()

#         if use_log_norm:
#             # LogNorm does not support values <= 0.
#             # Instead of replacing non-positive values by NaN, clip them to vmin.
#             if vmin_row is None:
#                 raise ValueError("vmin_row must be provided when use_log_norm=True.")
        
#             if vmin_row <= 0:
#                 raise ValueError("vmin must be strictly positive when use_log_norm=True.")
        
#             image = np.clip(image, vmin_row, None)

#         ny, nx_img = image.shape

#         x_axis = (np.arange(nx_img) - (nx_img - 1) / 2) * pix_scale_arcsec
#         y_axis = (np.arange(ny) - (ny - 1) / 2) * pix_scale_arcsec

#         extent = [
#             x_axis[0],
#             x_axis[-1],
#             y_axis[0],
#             y_axis[-1],
#         ]

#         return image, extent

#     with plt.rc_context(rc_params):
#         fig, axes = plt.subplots(
#             nrows,
#             ncols,
#             figsize=figsize,
#             squeeze=False,
#         )

#         images = []
#         image_axes = []

#         row_images = [[] for _ in range(nrows)]
#         row_axes = [[] for _ in range(nrows)]

#         for i in range(nrows):
#             row = psf_groups[i]

#             for j in range(ncols):
#                 ax = axes[i, j]

#                 if j >= len(row) or row[j] is None:
#                     ax.set_visible(False)
#                     continue

#                 image, extent = _prepare_image(row[j], i, j, vmin_row=vmin_rows[i])

#                 if use_log_norm:
#                     norm = LogNorm(vmin=vmin_rows[i], vmax=vmax_rows[i])
#                 else:
#                     norm = plt.Normalize(vmin=vmin_rows[i], vmax=vmax_rows[i])
                
#                 im = ax.imshow(
#                     image,
#                     norm=norm,
#                     cmap=cmap,
#                     extent=extent,
#                     origin=origin,
#                 )

#                 images.append(im)
#                 image_axes.append(ax)

#                 row_images[i].append(im)
#                 row_axes[i].append(ax)

#                 title_set = False

#                 if titles is not None:
#                     if i < len(titles) and j < len(titles[i]):
#                         if titles[i][j] is not None:
#                             ax.set_title(titles[i][j])
#                             title_set = True

#                 if not title_set and col_labels is not None:
#                     if i == 0 and j < len(col_labels):
#                         ax.set_title(col_labels[j])

#                 if hide_inner_labels:
#                     if i == nrows - 1:
#                         ax.set_xlabel(xlabel)
#                     else:
#                         ax.set_xlabel("")
#                         ax.set_xticklabels([])

#                     if j == 0:
#                         if row_labels is not None and i < len(row_labels):
#                             ax.set_ylabel(row_labels[i])
#                         else:
#                             ax.set_ylabel(ylabel)
#                     else:
#                         ax.set_ylabel("")
#                         ax.set_yticklabels([])

#                 else:
#                     ax.set_xlabel(xlabel)

#                     if row_labels is not None and j == 0 and i < len(row_labels):
#                         ax.set_ylabel(row_labels[i])
#                     else:
#                         ax.set_ylabel(ylabel)

#         if len(images) == 0:
#             raise ValueError("No valid image has been plotted.")

#         if share_colorbar:
#             fig.tight_layout(rect=layout_rect)
#             fig.subplots_adjust(wspace=wspace, hspace=hspace)

#             cax = fig.add_axes(cbar_position)

#             cbar = fig.colorbar(
#                 images[0],
#                 cax=cax,
#                 orientation="horizontal",
#             )
#             cbar.set_label(cbar_label)

#         else:
#             fig.tight_layout()
#             fig.subplots_adjust(wspace=wspace, hspace=hspace)

#             if individual_colorbars:
#                 for im, ax in zip(images, image_axes):
#                     cbar = fig.colorbar(
#                         im,
#                         ax=ax,
#                         fraction=0.046,
#                         pad=0.04,
#                     )
#                     cbar.set_label(cbar_label)

#             elif row_colorbars:
#                 if row_cbar_orientation == "vertical":
#                     fig.subplots_adjust(right=0.88)
            
#                 for i in range(nrows):
#                     if len(row_images[i]) == 0:
#                         continue
            
#                     cbar = fig.colorbar(
#                         row_images[i][0],
#                         ax=row_axes[i],
#                         orientation=row_cbar_orientation,
#                         fraction=row_cbar_fraction,
#                         pad=row_cbar_pad,
#                     )
#                     cbar.set_label(cbar_label)
            
#                     if row_cbar_orientation == "vertical":
#                         cbar.ax.tick_params(
#                             axis="y",
#                             direction="in",
#                             pad=2,
#                             labelsize=7,
#                         )
            
#                     else:
#                         cbar.ax.tick_params(
#                             axis="x",
#                             direction="in",
#                             pad=2,
#                             labelsize=7,
#                         )

#         if save:
#             path = Path(savepath)

#             if saveformat is None:
#                 if path.suffix:
#                     saveformat = path.suffix.lower().lstrip(".")
#                 else:
#                     saveformat = "pdf"
#                     path = path.with_suffix(".pdf")

#             path.parent.mkdir(parents=True, exist_ok=True)

#             def _save_one(fmt):
#                 fmt = fmt.lower().lstrip(".")

#                 if fmt == "fig":
#                     fig_path = path.with_suffix(".fig")
#                     with open(fig_path, "wb") as f:
#                         pickle.dump(fig, f)

#                 elif fmt in {"pdf", "eps", "svg"}:
#                     out = path.with_suffix(f".{fmt}")
#                     fig.savefig(
#                         out,
#                         format=fmt,
#                         bbox_inches="tight",
#                         pad_inches=0.02,
#                         transparent=False,
#                     )

#                 elif fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
#                     out = path.with_suffix(f".{fmt}")
#                     fig.savefig(
#                         out,
#                         format=fmt,
#                         dpi=max(dpi, 300),
#                         bbox_inches="tight",
#                         pad_inches=0.02,
#                         transparent=False,
#                     )

#                 else:
#                     raise ValueError(
#                         f"Unsupported save format '{fmt}'. "
#                         "Use pdf, eps, png, tiff, jpg, jpeg, svg, fig, or all."
#                     )

#             if saveformat.lower() == "all":
#                 for fmt in ("png", "pdf", "fig"):
#                     _save_one(fmt)
#             else:
#                 _save_one(saveformat)

#     return fig, axes
#%%
# def plot_psf_aa(
#     psf,
#     wvl,
#     telescope_diameter,
#     sampling,
#     vmin=1e-6,
#     vmax=1,
#     cmap="inferno",
#     xlabel=r"[arcsec]",
#     ylabel=r"[arcsec]",
#     cbar_label="Normalized intensity",
#     nx = None,
#     title=None,
#     figsize=(3.35, 3.1),   # ~ largeur colonne A&A en pouces
#     dpi=1200,
#     origin="lower",
#     save=False,
#     savepath="psf.pdf",
#     saveformat=None,
#     normalise = True,
# ):
    

#     # Style adapté à A&A
#     plt.rcParams.update({
#         "font.size": 8,
#         "axes.labelsize": 8,
#         "axes.titlesize": 8,
#         "xtick.labelsize": 7,
#         "ytick.labelsize": 7,
#         "legend.fontsize": 7,
#         "font.family": "serif",
#         "mathtext.fontset": "cm",
#         "axes.linewidth": 0.8,
#         "xtick.direction": "in",
#         "ytick.direction": "in",
#         "xtick.top": True,
#         "ytick.right": True,
#     })

#     # Échelle en arcsec
#     pix_scale = wvl / telescope_diameter / sampling
    
#     rad2arcsec = 180/(2*np.pi)*3600
#     if normalise:
#         psf_norm = psf/psf.sum()
#     else:
#         psf_norm = psf.copy()

#     if nx is None:
#         nx = psf.shape[0]
#     else:
#         ctr = psf_norm.shape[0]//2
#         psf_norm = psf_norm[ctr-nx//2:ctr+nx//2,ctr-nx//2:ctr+nx//2]
#     axis = np.linspace(-nx // 2, nx // 2, nx) * pix_scale * rad2arcsec
#     # Normalisation
#     maxi = np.max(psf_norm)
#     norm = LogNorm(vmin=vmin, vmax=vmax)

#     # Figure
#     fig, ax = plt.subplots(figsize=figsize)

#     im = ax.imshow(
#         psf_norm / maxi,
#         norm=norm,
#         cmap=cmap,
#         extent=[axis[0], axis[-1], axis[0], axis[-1]],
#         origin=origin,
#     )

#     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     cbar.set_label(cbar_label)

#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)

#     if title is not None:
#         ax.set_title(title)

#     fig.tight_layout()

#     # Sauvegarde optionnelle
#     if save:
#         path = Path(savepath)

#         if saveformat is None:
#             if path.suffix:
#                 saveformat = path.suffix.lower().lstrip(".")
#             else:
#                 saveformat = "pdf"
#                 path = path.with_suffix(".pdf")

#         path.parent.mkdir(parents=True, exist_ok=True)

#         def _save_one(fmt):
#             fmt = fmt.lower().lstrip(".")

#             if fmt == "fig":
#                 fig_path = path.with_suffix(".fig")
#                 with open(fig_path, "wb") as f:
#                     pickle.dump(fig, f)

#             elif fmt in {"pdf", "eps", "svg"}:
#                 out = path.with_suffix(f".{fmt}")
#                 fig.savefig(
#                     out,
#                     format=fmt,
#                     bbox_inches="tight",
#                     pad_inches=0.02,
#                     transparent=False,
#                 )

#             elif fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
#                 out = path.with_suffix(f".{fmt}")
#                 fig.savefig(
#                     out,
#                     format=fmt,
#                     dpi=max(dpi, 300),
#                     bbox_inches="tight",
#                     pad_inches=0.02,
#                     transparent=False,
#                 )

#             else:
#                 raise ValueError(
#                     f"Unsupported save format '{fmt}'. "
#                     "Use pdf, eps, png, tiff, jpg, jpeg, svg, fig, or all."
#                 )

#         if saveformat.lower() == "all":
#             for fmt in ("png", "pdf", "fig"):
#                 _save_one(fmt)
#         else:
#             _save_one(saveformat)

#     return fig, ax

#%%




def plot_frame_count_aa(
    frame_count,
    time=None,
    x_unit="Iteration",
    one_column=True,
    dpi=300,
    save=False,
    savepath="frame_count_lost_frames.pdf",
    saveformat=None,
    journal_style=True,
    show_cumulative=True,
    legend_loc="upper left",
    lost_ymin=None,
    lost_ymax=None,
):
   

    frame_count = np.asarray(frame_count).ravel()

    if frame_count.size < 2:
        raise ValueError("frame_count must contain at least 2 elements.")

    if time is None:
        x = np.arange(frame_count.size)
        x_label = x_unit
    else:
        x = np.asarray(time).ravel()
        if x.size != frame_count.size:
            raise ValueError("time and frame_count must have the same length.")
        x_label = x_unit

    frame_from_zero = frame_count - frame_count[0]
    frame_diff = np.diff(frame_count)
    lost_frames = np.clip(frame_diff - 1, 0, None)
    x_lost = x[1:]

    if show_cumulative:
        lost_cum = np.cumsum(lost_frames)

    label_fs = 9
    tick_fs = 8
    legend_fs = 8

    width_in = 88 / 25.4 if one_column else 180 / 25.4
    height_in = width_in * 1

    fig, axes = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(width_in, height_in),
        dpi=dpi,
        constrained_layout=True
    )

    ax1, ax2 = axes

    col_main = "black"
    col_alt = "#355C9A"

    ax1.plot(
        x, frame_from_zero,
        color=col_main,
        lw=1.4,
        ls="-",
        zorder=3,
        solid_capstyle="round",
    )

    ax1.set_ylabel("Count", fontsize=label_fs)
    ax1.text(
        0.02, 0.95,
        r"(a)",
        transform=ax1.transAxes,
        ha="left", va="top",
        fontsize=label_fs
    )

    

    # (b) lost frames
    line_lost, = ax2.plot(
        x_lost, lost_frames,
        color=col_alt,
        lw=1.1,
        ls="-",
        drawstyle="steps-mid",
        zorder=3,
    )
    ax2.set_ylabel("Lost frames", fontsize=label_fs)
    ax2.set_xlabel(x_label, fontsize=label_fs)
    ax2.text(
        0.02, 0.95,
        r"(b)",
        transform=ax2.transAxes,
        ha="left", va="top",
        fontsize=label_fs
    )
    ax2_right = None
    if show_cumulative:
        ax2_right = ax2.twinx()
    
        lost_cum = np.cumsum(lost_frames)
    
        line_cum, = ax2_right.plot(
            x_lost, lost_cum,
            color=col_main,
            lw=1.0,
            ls=(0, (7, 3)),
            zorder=2,
        )
    
        ax2_right.set_ylabel("Cumulative lost", fontsize=label_fs, labelpad=8)
    
        ax2_right.tick_params(
            which="major",
            direction="in",
            length=4.5,
            width=1.0,
            labelsize=tick_fs,
            pad=3,
            top=True,
            right=True,
        )
        ax2_right.tick_params(
            which="minor",
            direction="in",
            length=2.5,
            width=0.8,
            top=True,
            right=True,
        )
    
        for spine in ax2_right.spines.values():
            spine.set_linewidth(1.0)

        ax2.legend(
            [line_lost, line_cum],
            ["Lost/step", "Cumulative"],
            frameon=False,
            fontsize=legend_fs,
            loc=_legend_loc_from_mode(legend_loc),
            handlelength=2.8,
            borderaxespad=0.3,
        )

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        ax.tick_params(
            which="major",
            direction="in",
            length=5,
            width=1.0,
            labelsize=tick_fs,
            pad=4,
            top=True,
            right=True,
        )
        ax.tick_params(
            which="minor",
            direction="in",
            length=3,
            width=0.8,
            top=True,
            right=True,
        )
        
        if journal_style:
            ax.grid(False)
        else:
            ax.grid(True, which="major", color="0.88", lw=0.6)
            ax.grid(True, which="minor", color="0.93", lw=0.4)

    ax1.yaxis.set_label_coords(-0.14, 0.5)
    ax2.yaxis.set_label_coords(-0.14, 0.5)
    
    if ax2_right is not None:
        ax2_right.yaxis.set_label_coords(1.10, 0.5)  
    ax1.set_xlim(x[0], x[-1])
    fig.subplots_adjust(
        left=0.20,
        right=0.98,
        bottom=0.16,
        top=0.97,
        hspace=0.20
    )
    if save:
        path = Path(savepath)

        if saveformat is None:
            if path.suffix:
                saveformat = path.suffix.lower().lstrip(".")
            else:
                saveformat = "pdf"
                path = path.with_suffix(".pdf")

        path.parent.mkdir(parents=True, exist_ok=True)

        def _save_one(fmt):
            fmt = fmt.lower().lstrip(".")

            if fmt == "fig":
                fig_path = path.with_suffix(".fig")
                with open(fig_path, "wb") as f:
                    pickle.dump(fig, f)

            elif fmt in {"pdf", "eps", "svg"}:
                out = path.with_suffix(f".{fmt}")
                fig.savefig(
                    out,
                    format=fmt,
                    bbox_inches="tight",
                    pad_inches=0.02,
                    transparent=False,
                )

            elif fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
                out = path.with_suffix(f".{fmt}")
                fig.savefig(
                    out,
                    format=fmt,
                    dpi=max(dpi, 300),
                    bbox_inches="tight",
                    pad_inches=0.02,
                    transparent=False,
                )

            else:
                raise ValueError(
                    f"Unsupported save format '{fmt}'. "
                    "Use pdf, eps, png, tiff, jpg, jpeg, svg, fig, or all."
                )

        if saveformat.lower() == "all":
            for fmt in ("png", "pdf", "fig"):
                _save_one(fmt)
        else:
            _save_one(saveformat)

    return fig, axes

#%%
def plot_phase_map_aa(
    phase,
    telescope_diameter=None,   # [m] ; if None -> pixel coordinates
    cmap="seismic",
    nan_color="0.85",
    vmin=None,
    vmax=None,
    symmetric=True,
    xlabel=None,
    ylabel=None,
    cbar_label=r"Phase [rad]",
    title=None,
    one_column=True,
    dpi=300,
    origin="lower",
    interpolation="none",
    journal_style=True,
    save=False,
    savepath="phase_map_aa.pdf",
    saveformat=None,
):
    """
    Plot a 2D phase map in a style suitable for A&A publication.

    Parameters
    ----------
    phase : 2D ndarray
        Phase map to display. Can contain NaNs.
    telescope_diameter : float or None, optional
        Telescope diameter in meters. If provided, axes are expressed in meters
        assuming the full array spans the full pupil diameter. If None, axes are
        shown in pixels.
    cmap : str, optional
        Colormap. Default is 'seismic'.
    nan_color : color spec, optional
        Color used for NaN pixels.
    vmin, vmax : float, optional
        Color scale limits. If None, automatically inferred.
    symmetric : bool, optional
        If True, enforce a symmetric color scale around 0.
    xlabel, ylabel : str or None
        Axis labels. If None, automatic labels are used.
    cbar_label : str
        Colorbar label.
    title : str, optional
        Figure title.
    one_column : bool
        If True, use ~8.8 cm width; else ~18 cm.
    dpi : int
        Figure dpi.
    origin : {'lower', 'upper'}
        Image origin.
    interpolation : str
        Interpolation passed to imshow.
    journal_style : bool
        If True, no grid, restrained styling for final paper figure.
    save : bool
        If True, save figure.
    savepath : str
        Output file path.
    saveformat : str, optional
        'pdf', 'png', 'svg', 'eps', 'fig', or 'all'.

    Returns
    -------
    fig, ax
    """

    phase = np.asarray(phase)

    if phase.ndim != 2:
        raise ValueError("phase must be a 2D array.")

    ny, nx = phase.shape

    # -------- Style A&A --------
    label_fs = 9
    tick_fs = 8
    title_fs = 9

    width_in = 88 / 25.4 if one_column else 180 / 25.4
    height_in = width_in * 0.92

    fig, ax = plt.subplots(
        figsize=(width_in, height_in),
        dpi=dpi,
        constrained_layout=True
    )

    # -------- NaN-safe array --------
    phase_ma = np.ma.masked_invalid(phase)

    if phase_ma.count() == 0:
        raise ValueError("phase contains only NaN values.")

    # -------- Color scale --------
    finite_vals = phase[np.isfinite(phase)]

    if vmin is None or vmax is None:
        if symmetric:
            vmax_auto = np.nanmax(np.abs(finite_vals))
            vmin_auto = -vmax_auto
        else:
            vmin_auto = np.nanmin(finite_vals)
            vmax_auto = np.nanmax(finite_vals)

        if vmin is None:
            vmin = vmin_auto
        if vmax is None:
            vmax = vmax_auto

    if symmetric:
        vmax_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vmax_abs, vmax_abs
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=nan_color)

    # -------- Extent / labels --------
    if telescope_diameter is None:
        extent = [-0.5, nx - 0.5, -0.5, ny - 0.5]
        if xlabel is None:
            xlabel = r"$x$ [pix]"
        if ylabel is None:
            ylabel = r"$y$ [pix]"
    else:
        D = float(telescope_diameter)
        if D <= 0:
            raise ValueError("telescope_diameter must be > 0.")

        dx = D / nx
        dy = D / ny

        extent = [
            -D / 2 + dx / 2,
            +D / 2 - dx / 2,
            -D / 2 + dy / 2,
            +D / 2 - dy / 2,
        ]

        if xlabel is None:
            xlabel = r"$x$ [m]"
        if ylabel is None:
            ylabel = r"$y$ [m]"

    # -------- Image --------
    im = ax.imshow(
        phase_ma,
        cmap=cmap_obj,
        norm=norm,
        origin=origin,
        extent=extent,
        interpolation=interpolation,
        aspect="equal",
    )

    # -------- Colorbar --------
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=label_fs)
    cbar.ax.tick_params(labelsize=tick_fs, direction="in", length=4, width=0.8)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()

    # -------- Labels --------
    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)

    if title is not None:
        ax.set_title(title, fontsize=title_fs)

    # -------- Frame / ticks --------
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.tick_params(
        which="major",
        direction="in",
        length=5,
        width=1.0,
        labelsize=tick_fs,
        pad=4,
        top=True,
        right=True,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        length=3,
        width=0.8,
        top=True,
        right=True,
    )

    if journal_style:
        ax.grid(False)
    else:
        ax.grid(True, which="major", color="0.88", lw=0.6)
        ax.grid(True, which="minor", color="0.93", lw=0.4)

    # -------- Save --------
    if save:
        path = Path(savepath)

        if saveformat is None:
            if path.suffix:
                saveformat = path.suffix.lower().lstrip(".")
            else:
                saveformat = "pdf"
                path = path.with_suffix(".pdf")

        path.parent.mkdir(parents=True, exist_ok=True)

        def _save_one(fmt):
            fmt = fmt.lower().lstrip(".")

            if fmt == "fig":
                fig_path = path.with_suffix(".fig")
                with open(fig_path, "wb") as f:
                    pickle.dump(fig, f)

            elif fmt in {"pdf", "eps", "svg"}:
                out = path.with_suffix(f".{fmt}")
                fig.savefig(
                    out,
                    format=fmt,
                    bbox_inches="tight",
                    pad_inches=0.02,
                    transparent=False,
                )

            elif fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
                out = path.with_suffix(f".{fmt}")
                fig.savefig(
                    out,
                    format=fmt,
                    dpi=max(dpi, 300),
                    bbox_inches="tight",
                    pad_inches=0.02,
                    transparent=False,
                )

            else:
                raise ValueError(
                    f"Unsupported save format '{fmt}'. "
                    "Use pdf, eps, png, tiff, jpg, jpeg, svg, fig, or all."
                )

        if saveformat.lower() == "all":
            for fmt in ("png", "pdf", "fig"):
                _save_one(fmt)
        else:
            _save_one(saveformat)

    return fig, ax

#%%
from scipy.integrate import cumulative_trapezoid

def plot_cumulative_psd_aa(
    f1,
    psd1,
    f2=None,
    psd2=None,
    label1="Closed loop",
    label2="Open loop",
    method=np.nansum,
    cumulative="forward",
    f_label="Hz",
    y_unit=r"nm$^2$",
    ylabel=None,
    fmin=None,
    fmax=None,
    ymin=None,
    ymax=None,
    scale="log",
    xscale=None,
    yscale=None,
    normalised=False,
    show_legend=True,
    legend_loc="best",
    one_column=True,
    dpi=300,
    save=False,
    savepath="cumulative_psd_aa.pdf",
    saveformat=None,
    journal_style=True,
):
    """
    Plot cumulative PSD in A&A style.

    Parameters
    ----------
    f1, psd1 : array-like
        Frequency vector and PSD array for curve 1.
        psd1 can be 1D or 2D. If 2D, `method(psd1, axis=1)` is applied.
    f2, psd2 : array-like, optional
        Same for curve 2.
    cumulative : {"forward", "reverse"}
        - "forward" : C(f) = integral from f_min to f
        - "reverse" : C(f) = integral from f to f_max
    normalised : bool
        If True, each cumulative curve is divided by its maximum.
    """

    # ---------- style ----------
    label_fs = 9
    tick_fs = 8
    legend_fs = 8

    xscale_from_mode, yscale_from_mode = _axis_scale_from_mode(scale)
    if xscale is None:
        xscale = xscale_from_mode
    if yscale is None:
        yscale = yscale_from_mode

    if xscale not in {"linear", "log"}:
        raise ValueError("xscale must be 'linear' or 'log'.")

    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'.")

    # ---------- helpers ----------
    def _reduce_psd(f, psd):
        f = np.asarray(f).ravel()
        psd = np.asarray(psd)
    
        if psd.ndim == 1:
            y = psd
        else:
            y = method(psd, axis=1)
    
        valid = _valid_xy_for_scale(f, y, xscale, yscale)
    
        f = f[valid]
        y = y[valid]
    
        if f.size < 2:
            raise ValueError("Not enough valid points to compute cumulative PSD.")
    
        idx = np.argsort(f)
        return f[idx], y[idx]

    def _cumulative_from_psd(f, y, mode="forward"):
        if mode == "forward":
            c = cumulative_trapezoid(y, f, initial=0)
        elif mode == "reverse":
            # integral from current frequency to highest frequency
            c = -cumulative_trapezoid(y[::-1], f[::-1], initial=0)[::-1]
        else:
            raise ValueError("`cumulative` must be 'forward' or 'reverse'.")

        # avoid tiny negative numerical artefacts
        c = np.where(np.isfinite(c), c, np.nan)
        c[c < 0] = np.maximum(c[c < 0], 0)

        return c

    # ---------- curve 1 ----------
    f1r, y1 = _reduce_psd(f1, psd1)
    c1 = _cumulative_from_psd(f1r, y1, mode=cumulative)

    if normalised:
        max1 = np.nanmax(c1)
        if max1 > 0:
            c1 = c1 / max1

    # ---------- curve 2 ----------
    has_second_curve = (f2 is not None) and (psd2 is not None)
    if has_second_curve:
        f2r, y2 = _reduce_psd(f2, psd2)
        c2 = _cumulative_from_psd(f2r, y2, mode=cumulative)

        if normalised:
            max2 = np.nanmax(c2)
            if max2 > 0:
                c2 = c2 / max2
    else:
        c2 = None

    # ---------- figure size ----------
    width_in = 88 / 25.4 if one_column else 180 / 25.4
    height_in = width_in * 0.72

    fig, ax = plt.subplots(
        figsize=(width_in, height_in),
        dpi=dpi,
        constrained_layout=True
    )

    # ---------- curves ----------
    col1 = "black"
    col2 = "#355C9A"   # muted dark blue
    valid1_plot = _valid_xy_for_scale(f1r, c1, xscale, yscale)
    
    f1p = f1r[valid1_plot]
    c1p = c1[valid1_plot]
    ax.plot(
        f1p, c1p,
        color=col1,
        lw=1.6,
        ls="-",
        label=label1,
        zorder=3,
        solid_capstyle="round",
    )

    if has_second_curve:
        valid2_plot = _valid_xy_for_scale(f2r, c2, xscale, yscale)
    
        f2p = f2r[valid2_plot]
        c2p = c2[valid2_plot]
    
        ax.plot(
            f2p, c2p,
            color=col2,
            lw=1.6,
            ls=(0, (7, 3)),
            label=label2,
            zorder=3,
            dash_capstyle="butt",
        )

    # ---------- axes ----------
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    
    if fmin is not None or fmax is not None:
        ax.set_xlim(left=fmin, right=fmax)
    
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    ax.set_xlabel(f"{f_label}", fontsize=label_fs)

    if ylabel is None:
        if normalised:
            ylabel = "Normalised cumulative PSD"
        else:
            ylabel = f"{y_unit}"
    ax.set_ylabel(ylabel, fontsize=label_fs)

    # ---------- frame / ticks ----------
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.tick_params(
        which="major",
        direction="in",
        length=6,
        width=1.0,
        labelsize=tick_fs,
        pad=6,
        top=True,
        right=True,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        length=3.5,
        width=0.8,
        top=True,
        right=True,
    )

    # ---------- grid ----------
    if journal_style:
        ax.grid(False)
    else:
        ax.grid(True, which="major", color="0.88", lw=0.6)
        ax.grid(True, which="minor", color="0.93", lw=0.4)

    # ---------- legend ----------
    if show_legend and has_second_curve:
        ax.legend(
            frameon=False,
            fontsize=legend_fs,
            loc=_legend_loc_from_mode(legend_loc),
            handlelength=3.2,
            borderaxespad=0.4,
        )

    # ---------- save ----------
    if save:
        path = Path(savepath)

        if saveformat is None:
            if path.suffix:
                saveformat = path.suffix.lower().lstrip(".")
            else:
                saveformat = "pdf"
                path = path.with_suffix(".pdf")

        path.parent.mkdir(parents=True, exist_ok=True)

        def _save_one(fmt):
            fmt = fmt.lower().lstrip(".")

            if fmt == "fig":
                fig_path = path.with_suffix(".fig")
                with open(fig_path, "wb") as f:
                    pickle.dump(fig, f)

            elif fmt in {"pdf", "eps", "svg"}:
                out = path.with_suffix(f".{fmt}")
                fig.savefig(
                    out,
                    format=fmt,
                    bbox_inches="tight",
                    pad_inches=0.02,
                    transparent=False,
                )

            elif fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
                out = path.with_suffix(f".{fmt}")
                fig.savefig(
                    out,
                    format=fmt,
                    dpi=max(dpi, 300),
                    bbox_inches="tight",
                    pad_inches=0.02,
                    transparent=False,
                )

            else:
                raise ValueError(
                    f"Unsupported save format '{fmt}'. "
                    "Use pdf, eps, png, tiff, jpg, jpeg, svg, fig, or all."
                )

        if saveformat.lower() == "all":
            # Pour une figure de publication, PDF est la sortie de référence
            for fmt in ("pdf", "png", "fig"):
                _save_one(fmt)
        else:
            _save_one(saveformat)

    return fig, ax
#%%
def plot_etf_fit_aa(
    f_etf,
    etf,
    f_fit=None,
    etf_fit=None,
    label_data="Measured ETF",
    label_fit="Discrete fit",
    xlabel="Frequency [Hz]",
    ylabel="ETF",
    title=None,
    one_column=True,
    dpi=300,
    journal_style=True,
    show_legend=True,
    legend_loc="best",
    xscale="log",
    yscale="log",
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    fit_params=None,
    annotate_params=False,
    save=False,
    savepath="etf_fit_aa.pdf",
    saveformat=None,
):
    """
    Plot an ETF and its fit in a style suitable for A&A publication.

    Parameters
    ----------
    f_etf : array_like
        Frequency vector for the measured ETF.
    etf : array_like
        Measured ETF values.
    f_fit : array_like or None
        Frequency vector for the fitted ETF.
    etf_fit : array_like or None
        Fitted ETF values.
    label_data : str
        Label for measured ETF.
    label_fit : str
        Label for fit.
    xlabel, ylabel : str
        Axis labels.
    title : str or None
        Optional title.
    one_column : bool
        If True, figure width is 88 mm, else 180 mm.
    dpi : int
        Figure dpi.
    journal_style : bool
        If True, no background grid.
    show_legend : bool
        Whether to show the legend.
    xscale, yscale : {"log", "linear"}
        Axis scales.
    xmin, xmax, ymin, ymax : float or None
        Axis limits.
    fit_params : dict or None
        Optional dictionary of fit parameters to annotate.
    annotate_params : bool
        If True, print fit parameters inside the figure.
    save : bool
        If True, save the figure.
    savepath : str
        Output file path.
    saveformat : str or None
        'pdf', 'png', 'svg', 'eps', 'fig', or 'all'.
    """

    # -------- Input sanitation --------
    f_etf = np.asarray(f_etf).ravel()
    etf = np.asarray(etf).ravel()

    if f_etf.size != etf.size:
        raise ValueError("f_etf and etf must have the same length.")

    valid = np.isfinite(f_etf) & np.isfinite(etf)
    if xscale == "log":
        valid &= (f_etf > 0)
    if yscale == "log":
        valid &= (etf > 0)

    f_etf = f_etf[valid]
    etf = etf[valid]

    if f_etf.size < 2:
        raise ValueError("Not enough valid ETF points to plot.")

    has_fit = (f_fit is not None) and (etf_fit is not None)
    if has_fit:
        f_fit = np.asarray(f_fit).ravel()
        etf_fit = np.asarray(etf_fit).ravel()

        if f_fit.size != etf_fit.size:
            raise ValueError("f_fit and etf_fit must have the same length.")

        valid_fit = np.isfinite(f_fit) & np.isfinite(etf_fit)
        if xscale == "log":
            valid_fit &= (f_fit > 0)
        if yscale == "log":
            valid_fit &= (etf_fit > 0)

        f_fit = f_fit[valid_fit]
        etf_fit = etf_fit[valid_fit]

        if f_fit.size < 2:
            raise ValueError("Not enough valid fit points to plot.")

    # -------- Style A&A --------
    label_fs = 9
    tick_fs = 8
    legend_fs = 8
    title_fs = 9

    width_in = 88 / 25.4 if one_column else 180 / 25.4
    height_in = width_in * 0.72

    fig, ax = plt.subplots(
        figsize=(width_in, height_in),
        dpi=dpi,
        constrained_layout=True
    )

    # -------- Curves --------
    col_data = "black"
    col_fit = "#355C9A"

    ax.plot(
        f_etf, etf,
        color=col_data,
        lw=1.2,
        ls="none",
        marker="o",
        ms=2.8,
        mew=0.0,
        label=label_data,
        zorder=2,
    )

    if has_fit:
        ax.plot(
            f_fit, etf_fit,
            color=col_fit,
            lw=1.5,
            ls="-",
            label=label_fit,
            zorder=5,
        )

    # -------- Reference line ETF=1 --------
    if xscale == "log":
        xref = np.logspace(np.log10(np.min(f_etf)), np.log10(np.max(f_etf)), 200)
    else:
        xref = np.linspace(np.min(f_etf), np.max(f_etf), 200)

    ax.plot(
        xref,
        np.ones_like(xref),
        color="0.5",
        lw=0.9,
        ls="--",
        zorder=1,
    )

    # -------- Scales / limits --------
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    if xmin is not None or xmax is not None:
        ax.set_xlim(left=xmin, right=xmax)
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    # -------- Labels --------
    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)

    if title is not None:
        ax.set_title(title, fontsize=title_fs)

    # -------- Annotation --------
    if annotate_params and fit_params is not None:
        txt = "\n".join([rf"{k} = {v}" for k, v in fit_params.items()])
        ax.text(
            0.04, 0.96, txt,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.8")
        )

    # -------- Ticks / frame --------
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.tick_params(
        which="major",
        direction="in",
        length=5,
        width=1.0,
        labelsize=tick_fs,
        pad=4,
        top=True,
        right=True,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        length=3,
        width=0.8,
        top=True,
        right=True,
    )

    # -------- Grid --------
    if journal_style:
        ax.grid(False)
    else:
        ax.grid(True, which="major", color="0.88", lw=0.6)
        ax.grid(True, which="minor", color="0.93", lw=0.4)

    # -------- Legend --------
    if show_legend and has_fit:
        ax.legend(
            frameon=False,
            fontsize=legend_fs,
            loc=_legend_loc_from_mode(legend_loc),
            handlelength=2.8,
            borderaxespad=0.3,
        )

    # -------- Save --------
    if save:
        path = Path(savepath)

        if saveformat is None:
            if path.suffix:
                saveformat = path.suffix.lower().lstrip(".")
            else:
                saveformat = "pdf"
                path = path.with_suffix(".pdf")

        path.parent.mkdir(parents=True, exist_ok=True)

        def _save_one(fmt):
            fmt = fmt.lower().lstrip(".")

            if fmt == "fig":
                fig_path = path.with_suffix(".fig")
                with open(fig_path, "wb") as f:
                    pickle.dump(fig, f)

            elif fmt in {"pdf", "eps", "svg"}:
                out = path.with_suffix(f".{fmt}")
                fig.savefig(
                    out,
                    format=fmt,
                    bbox_inches="tight",
                    pad_inches=0.02,
                    transparent=False,
                )

            elif fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
                out = path.with_suffix(f".{fmt}")
                fig.savefig(
                    out,
                    format=fmt,
                    dpi=max(dpi, 300),
                    bbox_inches="tight",
                    pad_inches=0.02,
                    transparent=False,
                )

            else:
                raise ValueError(
                    f"Unsupported save format '{fmt}'. "
                    "Use pdf, eps, png, tiff, jpg, jpeg, svg, fig, or all."
                )

        if saveformat.lower() == "all":
            for fmt in ("png", "pdf", "fig"):
                _save_one(fmt)
        else:
            _save_one(saveformat)

    return fig, ax

def plot_phase_comparison_aa(
    phase1,
    phase2,
    telescope_diameter=None,
    labels=("Map 1", "Map 2"),
    diff_label=None,
    diff_mode="error",        # "subtract", "error", or "none"
    cmap="seismic",
    diff_cmap="seismic",
    error_cmap="cividis",
    nan_color="0.85",
    vmin=None,
    vmax=None,
    diff_vmin=None,
    diff_vmax=None,
    symmetric=True,
    symmetric_diff=True,
    same_color_scale=True,
    xlabel=None,
    ylabel=None,
    cbar_label=r"OPD [nm]",
    diff_cbar_label=None,
    one_column=False,
    dpi=300,
    origin="lower",
    interpolation="none",
    journal_style=True,
    error_vmax_percentile=99.5,
    save=False,
    savepath="phase_comparison_aa.pdf",
    saveformat=None,
):
    """
    Compare two 2D phase maps in an A&A-compatible style.

    Parameters
    ----------
    phase1, phase2 : 2D ndarray
        Phase maps to compare. NaNs are allowed.

    telescope_diameter : float or None
        If provided, the full array is assumed to span the full pupil diameter.
        If None, pixel coordinates are used.

    labels : tuple of str
        Titles for phase1 and phase2.

    diff_label : str or None
        Title for the third panel. If None, chosen automatically.

    diff_mode : {"subtract", "error", "none"}
        - "subtract" : show phase2 - phase1
        - "error"    : show abs(phase2 - phase1) / abs(phase1)
        - "none"     : show only phase1 and phase2

    cmap : str
        Colormap for phase1 and phase2. Default is "seismic".

    diff_cmap : str
        Colormap for signed difference map. Default is "seismic".

    error_cmap : str
        Colormap for positive error map. Default is "cividis".

    nan_color : color
        Color used for NaN pixels.

    vmin, vmax : float or None
        Color limits for phase1 and phase2.

    diff_vmin, diff_vmax : float or None
        Color limits for the third panel.
        For diff_mode="error", diff_vmax can be set manually to override
        error_vmax_percentile.

    symmetric : bool
        If True, phase1 and phase2 use a color scale symmetric around zero.

    symmetric_diff : bool
        If True, signed difference uses a color scale symmetric around zero.
        Ignored for diff_mode="error", because the error is positive.

    same_color_scale : bool
        If True, phase1 and phase2 share the same color scale.

    cbar_label : str
        Colorbar label for phase maps.

    diff_cbar_label : str or None
        Colorbar label for the third panel.

    error_vmax_percentile : float or None
        Percentile used to set the upper display limit of the error colorbar.
        Example: 99.5 sets vmax to the 99.5th percentile.
        The data are not modified; values above vmax are simply saturated.
        Set to None to use the full error range.

    save, savepath, saveformat :
        Same convention as the other A&A plotting functions.

    Returns
    -------
    fig, axes
    """

    phase1 = np.asarray(phase1)
    phase2 = np.asarray(phase2)
    
    if phase1.ndim != 2 or phase2.ndim != 2:
        raise ValueError("phase1 and phase2 must be 2D arrays.")
    if diff_mode in {"subtract", "error"}:
        if phase1.shape != phase2.shape:
            raise ValueError("phase1 and phase2 must have the same shape.")

    if diff_mode not in {"subtract", "error", "none"}:
        raise ValueError("diff_mode must be 'subtract', 'error', or 'none'.")

    ny, nx = phase1.shape

    show_diff = diff_mode in {"subtract", "error"}
    ncols = 3 if show_diff else 2

    # ---------------------------
    # NaN-safe arrays
    # ---------------------------
    phase1_ma = np.ma.masked_invalid(phase1)
    phase2_ma = np.ma.masked_invalid(phase2)

    if phase1_ma.count() == 0:
        raise ValueError("phase1 contains only NaN values.")

    if phase2_ma.count() == 0:
        raise ValueError("phase2 contains only NaN values.")

    # ---------------------------
    # Difference / error map
    # ---------------------------
    if show_diff:
        if diff_mode == "subtract":
            diff = phase2 - phase1

            if diff_label is None:
                diff_label = rf"{labels[1]} $-$ {labels[0]}"

            if diff_cbar_label is None:
                diff_cbar_label = cbar_label

            diff_cmap_use = diff_cmap
            symmetric_diff_use = symmetric_diff
            cbar_extend = "neither"

        elif diff_mode == "error":
            # Relative error:
            # abs(phase2 - phase1) / abs(phase1)
            phase1_abs = np.abs(phase1)

            valid = np.isfinite(phase1) & np.isfinite(phase2)

            if not np.any(valid):
                raise ValueError("No valid finite pixels to compute the error map.")

            ref_scale = np.nanmax(phase1_abs[valid])

            if not np.isfinite(ref_scale) or ref_scale <= 0:
                raise ValueError("Cannot compute relative error: invalid reference scale.")

            # Automatic floor to avoid divisions by values too close to zero.
            # Pixels where abs(phase1) is below this floor are kept as NaN.
            floor = max(1e-10, 1e-6 * ref_scale)

            diff = np.full_like(phase1, np.nan, dtype=float)

            valid_rel = valid & (phase1_abs > floor)

            diff[valid_rel] = 100*(
                np.abs(phase2[valid_rel] - phase1[valid_rel])
                / phase1_abs[valid_rel]
            )

            finite_error = diff[np.isfinite(diff)]

            if finite_error.size == 0:
                raise ValueError(
                    "The error map contains only NaN values. "
                    "The reference phase may be too close to zero everywhere."
                )

            # Robust display limit only.
            # The data are NOT clipped or modified.
            if error_vmax_percentile is not None and diff_vmax is None:
                diff_vmax = np.nanpercentile(finite_error, error_vmax_percentile)

                if not np.isfinite(diff_vmax) or diff_vmax <= 0:
                    diff_vmax = None

            if diff_label is None:
                diff_label = "Relative error"

            if diff_cbar_label is None:
                diff_cbar_label = "Relative error"

            diff_cmap_use = error_cmap
            symmetric_diff_use = False

            # For a positive error map, the colorbar should start at zero.
            if diff_vmin is None:
                diff_vmin = 0.0

            # Indicate that values above vmax are saturated.
            cbar_extend = "max" if diff_vmax is not None else "neither"

        diff_ma = np.ma.masked_invalid(diff)

        if diff_ma.count() == 0:
            raise ValueError("The difference/error map contains only NaN values.")

    else:
        diff = None
        diff_ma = None
        cbar_extend = "neither"

    # ---------------------------
    # A&A-like style
    # ---------------------------
    label_fs = 9
    tick_fs = 8
    title_fs = 9

    width_in = 88 / 25.4 if one_column else 180 / 25.4

    if ncols == 3:
        height_in = width_in * 0.34
    else:
        height_in = width_in * 0.42

    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(width_in, height_in),
        dpi=dpi,
        constrained_layout=True,
        squeeze=False,
    )

    axes = axes.ravel()

    # ---------------------------
    # Extent and labels
    # ---------------------------
    if telescope_diameter is None:
        extent = [-0.5, nx - 0.5, -0.5, ny - 0.5]

        if xlabel is None:
            xlabel = r"$x$ [pix]"

        if ylabel is None:
            ylabel = r"$y$ [pix]"

    else:
        D = float(telescope_diameter)

        if D <= 0:
            raise ValueError("telescope_diameter must be > 0.")

        dx = D / nx
        dy = D / ny

        extent = [
            -D / 2 + dx / 2,
            +D / 2 - dx / 2,
            -D / 2 + dy / 2,
            +D / 2 - dy / 2,
        ]

        if xlabel is None:
            xlabel = r"$x$ [m]"

        if ylabel is None:
            ylabel = r"$y$ [m]"

    # ---------------------------
    # Normalization helper
    # ---------------------------
    def _make_norm(data, vmin_in, vmax_in, symmetric_in):
        data = np.asarray(data)
        finite = data[np.isfinite(data)]

        if finite.size == 0:
            raise ValueError("Cannot define color scale from all-NaN data.")

        if vmin_in is None or vmax_in is None:
            if symmetric_in:
                vmax_auto = np.nanmax(np.abs(finite))
                vmin_auto = -vmax_auto
            else:
                vmin_auto = np.nanmin(finite)
                vmax_auto = np.nanmax(finite)

            if vmin_in is None:
                vmin_in = vmin_auto

            if vmax_in is None:
                vmax_in = vmax_auto

        if symmetric_in:
            vmax_abs = max(abs(vmin_in), abs(vmax_in))

            if vmax_abs == 0:
                vmax_abs = 1.0

            vmin_in = -vmax_abs
            vmax_in = +vmax_abs

            norm = colors.TwoSlopeNorm(
                vmin=vmin_in,
                vcenter=0.0,
                vmax=vmax_in,
            )

        else:
            if vmax_in == vmin_in:
                vmax_in = vmin_in + 1.0

            norm = colors.Normalize(
                vmin=vmin_in,
                vmax=vmax_in,
            )

        return norm

    # ---------------------------
    # Phase color scales
    # ---------------------------
    if same_color_scale:
        finite_phase = np.concatenate(
            [
                phase1[np.isfinite(phase1)].ravel(),
                phase2[np.isfinite(phase2)].ravel(),
            ]
        )

        norm1 = _make_norm(finite_phase, vmin, vmax, symmetric)
        norm2 = norm1

    else:
        norm1 = _make_norm(phase1, vmin, vmax, symmetric)
        norm2 = _make_norm(phase2, vmin, vmax, symmetric)

    if show_diff:
        norm_diff = _make_norm(
            diff,
            diff_vmin,
            diff_vmax,
            symmetric_diff_use,
        )

    # ---------------------------
    # Colormaps
    # ---------------------------
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=nan_color)

    if show_diff:
        diff_cmap_obj = plt.get_cmap(diff_cmap_use).copy()
        diff_cmap_obj.set_bad(color=nan_color)

    # ---------------------------
    # Images
    # ---------------------------
    im1 = axes[0].imshow(
        phase1_ma,
        cmap=cmap_obj,
        norm=norm1,
        origin=origin,
        extent=extent,
        interpolation=interpolation,
        aspect="equal",
    )

    im2 = axes[1].imshow(
        phase2_ma,
        cmap=cmap_obj,
        norm=norm2,
        origin=origin,
        extent=extent,
        interpolation=interpolation,
        aspect="equal",
    )

    axes[0].set_title(rf"(a) {labels[0]}", fontsize=title_fs)
    axes[1].set_title(rf"(b) {labels[1]}", fontsize=title_fs)

    if show_diff:
        im3 = axes[2].imshow(
            diff_ma,
            cmap=diff_cmap_obj,
            norm=norm_diff,
            origin=origin,
            extent=extent,
            interpolation=interpolation,
            aspect="equal",
        )

        axes[2].set_title(rf"(c) {diff_label}", fontsize=title_fs)

    # ---------------------------
    # Axes styling
    # ---------------------------
    for i, ax in enumerate(axes):
        ax.set_xlabel(xlabel, fontsize=label_fs)

        if i == 0:
            ax.set_ylabel(ylabel, fontsize=label_fs)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        ax.tick_params(
            which="major",
            direction="in",
            length=5,
            width=1.0,
            labelsize=tick_fs,
            pad=4,
            top=True,
            right=True,
        )

        ax.tick_params(
            which="minor",
            direction="in",
            length=3,
            width=0.8,
            top=True,
            right=True,
        )

        if journal_style:
            ax.grid(False)
        else:
            ax.grid(True, which="major", color="0.88", lw=0.6)
            ax.grid(True, which="minor", color="0.93", lw=0.4)
    fig.canvas.draw()
    # ---------------------------
    # Colorbars
    # ---------------------------
    if same_color_scale:
        cbar12 = fig.colorbar(
            im2,
            ax=axes[:2],
            fraction=0.046,
            pad=0.035,
        )

        cbar12.set_label(cbar_label, fontsize=label_fs)
        cbar12.ax.tick_params(
            labelsize=tick_fs,
            direction="in",
            length=4,
            width=0.8,
        )
        cbar12.locator = MaxNLocator(nbins=5)
        cbar12.update_ticks()
        cbar12.ax.yaxis.set_label_coords(3, 0.5)

    else:
        for im, ax in zip([im1, im2], axes[:2]):
            cbar = fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.035,
            )

            cbar.set_label(cbar_label, fontsize=label_fs, labelpad = 2)
            cbar.ax.tick_params(
                labelsize=tick_fs,
                direction="in",
                length=4,
                width=0.8,
            )
            cbar.locator = MaxNLocator(nbins=5)
            cbar.update_ticks()

    if show_diff:
        cbar3 = fig.colorbar(
            im3,
            ax=axes[2],
            fraction=0.046,
            pad=0.035,
            extend=cbar_extend,
        )
        
        cbar3.set_label(diff_cbar_label, fontsize=label_fs)
        cbar3.ax.yaxis.set_label_coords(5, 0.545)
        
        cbar3.ax.tick_params(
            labelsize=tick_fs,
            direction="in",
            length=4,
            width=0.8,
        )
        cbar3.locator = MaxNLocator(nbins=5)
        cbar3.update_ticks()
    

    # ---------------------------
    # Save
    # ---------------------------
    if save:
        path = Path(savepath)

        if saveformat is None:
            if path.suffix:
                saveformat = path.suffix.lower().lstrip(".")
            else:
                saveformat = "pdf"
                path = path.with_suffix(".pdf")

        path.parent.mkdir(parents=True, exist_ok=True)

        def _save_one(fmt):
            fmt = fmt.lower().lstrip(".")

            if fmt == "fig":
                fig_path = path.with_suffix(".fig")
                with open(fig_path, "wb") as f:
                    pickle.dump(fig, f)

            elif fmt in {"pdf", "eps", "svg"}:
                out = path.with_suffix(f".{fmt}")
                fig.savefig(
                    out,
                    format=fmt,
                    bbox_inches="tight",
                    pad_inches=0.02,
                    transparent=False,
                )

            elif fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
                out = path.with_suffix(f".{fmt}")
                fig.savefig(
                    out,
                    format=fmt,
                    dpi=max(dpi, 300),
                    bbox_inches="tight",
                    pad_inches=0.02,
                    transparent=False,
                )

            else:
                raise ValueError(
                    f"Unsupported save format '{fmt}'. "
                    "Use pdf, eps, png, tiff, jpg, jpeg, svg, fig, or all."
                )

        if saveformat.lower() == "all":
            for fmt in ("pdf", "png", "fig"):
                _save_one(fmt)
        else:
            _save_one(saveformat)

    return fig, axes
#%%
def plot_n_psd_aa(
    psd_list,
    labels=None,
    method=np.nansum,
    f_label="Hz",
    psd_label=r"nm$^2$/Hz",
    xlabel=None,
    ylabel=None,
    fmin=None,
    fmax=None,
    ymin=None,
    ymax=None,
    scale="log",
    normalised=False,
    one_column=False,
    dpi=300,
    save=False,
    savepath="n_psd_aa.pdf",
    saveformat=None,
    journal_style=True,
    show_legend=True,
    legend_ncol=None,
    legend_loc="best",
    use_color=True,
    curve_styles=None,
):
    """
    Plot N PSDs in an A&A-compatible style.

    Parameters
    ----------
    psd_list : list of tuple
        List of PSDs to plot. Each element must be a tuple (freq, psd).

    labels : list of str, optional
        Labels associated with each PSD.

    curve_styles : list of dict or dict, optional
        Custom style for each curve.

        Accepted forms:

        1. List of dictionaries, same order as psd_list:

            curve_styles = [
                {"color": "black", "ls": "-", "lw": 1.3},
                {"color": "black", "ls": "--", "lw": 1.3},
            ]

        2. Dictionary indexed by curve number:

            curve_styles = {
                0: {"color": "black", "ls": "-"},
                1: {"color": "black", "ls": "--"},
            }

        3. Dictionary indexed by label:

            curve_styles = {
                "ZWFS - IM": {"color": "#1f77b4", "ls": "-"},
                "vZWFS - Atan": {"color": "#1f77b4", "ls": "--"},
            }

        Supported style keys include:
            color, ls, linestyle, lw, linewidth, alpha, zorder

        If curve_styles is None, the original automatic style is used.

    legend_loc : str
        Legend location. Accepted values include:
        'upper right', 'upper left', 'lower right', 'lower left',
        and 'free'/'best' for automatic placement.
    """

    xscale, yscale = _axis_scale_from_mode(scale)

    if not isinstance(psd_list, (list, tuple)):
        raise TypeError("psd_list must be a list of tuples: [(freq, psd), ...].")

    if len(psd_list) == 0:
        raise ValueError("psd_list must contain at least one PSD.")

    n_psd = len(psd_list)

    if labels is None:
        labels = [rf"PSD {i+1}" for i in range(n_psd)]

    if len(labels) != n_psd:
        raise ValueError("labels must have the same length as psd_list.")

    # ---------- A&A style ----------
    label_fs = 9
    tick_fs = 8
    legend_fs = 7 if n_psd > 6 else 8

    width_in = 88 / 25.4 if one_column else 180 / 25.4

    if one_column:
        height_in = width_in * 0.78
    else:
        height_in = width_in * 0.55

    fig, ax = plt.subplots(
        figsize=(width_in, height_in),
        dpi=dpi,
        constrained_layout=True,
    )

    linestyles = [
        "-",
        (0, (7, 3)),
        (0, (3, 2)),
        (0, (1, 2)),
        (0, (5, 2, 1, 2)),
        (0, (9, 3, 2, 3)),
    ]

    if use_color:
        colors_cycle = [
            "black",
            "#355C9A",
            "#8A3B12",
            "#3B6E3B",
            "#6A4C93",
            "#666666",
            "#A23E48",
            "#2F4F4F",
        ]
    else:
        colors_cycle = ["black", "0.25", "0.40", "0.55", "0.70", "0.15"]

    default_lw = 1.5 if n_psd <= 4 else 1.1 if n_psd <= 10 else 0.9

    def _prepare_psd(freq, psd):
        freq = np.asarray(freq).ravel()
        psd = np.asarray(psd)

        if psd.ndim == 1:
            y = psd.ravel()
        elif psd.ndim == 2:
            y = method(psd, axis=1)
        else:
            raise ValueError("Each PSD must be either 1D or 2D.")

        if freq.size != y.size:
            raise ValueError(
                "Frequency vector and reduced PSD must have the same length. "
                "If PSD is 2D, the function assumes method(psd, axis=1)."
            )

        valid = _valid_xy_for_scale(freq, y, xscale, yscale)
        freq = freq[valid]
        y = y[valid]

        if freq.size < 2:
            raise ValueError(
                "Each PSD must contain at least two valid points "
                "for the requested axis scale."
            )

        idx = np.argsort(freq)
        freq = freq[idx]
        y = y[idx]

        if normalised:
            ymax_local = np.nanmax(y)
            if ymax_local > 0:
                y = y / ymax_local

        return freq, y

    def _get_curve_style(i, label):
        """
        Return the plotting style for curve i.

        The style is first built from the automatic defaults, then optionally
        updated by curve_styles.
        """

        style = {
            "color": colors_cycle[i % len(colors_cycle)],
            "lw": default_lw,
            "ls": linestyles[i % len(linestyles)],
            "alpha": 1.0,
            "zorder": 3,
            "solid_capstyle": "round",
            "dash_capstyle": "butt",
        }

        if curve_styles is None:
            return style

        if isinstance(curve_styles, (list, tuple)):
            if len(curve_styles) != n_psd:
                raise ValueError(
                    "If curve_styles is a list or tuple, it must have "
                    "the same length as psd_list."
                )
            custom = curve_styles[i]

        elif isinstance(curve_styles, dict):
            if label in curve_styles:
                custom = curve_styles[label]
            elif i in curve_styles:
                custom = curve_styles[i]
            else:
                custom = None

        else:
            raise TypeError(
                "curve_styles must be None, a list/tuple of dictionaries, "
                "or a dictionary indexed by curve index or label."
            )

        if custom is None:
            return style

        if not isinstance(custom, dict):
            raise TypeError(
                "Each custom curve style must be a dictionary, "
                f"got {type(custom)} for curve {i}."
            )

        # Accept Matplotlib aliases.
        custom = custom.copy()

        if "linestyle" in custom and "ls" not in custom:
            custom["ls"] = custom.pop("linestyle")

        if "linewidth" in custom and "lw" not in custom:
            custom["lw"] = custom.pop("linewidth")

        allowed_keys = {
            "color",
            "lw",
            "ls",
            "alpha",
            "zorder",
            "solid_capstyle",
            "dash_capstyle",
        }

        unknown_keys = set(custom) - allowed_keys
        if unknown_keys:
            raise ValueError(
                "Unsupported style key(s) in curve_styles: "
                f"{sorted(unknown_keys)}. "
                "Supported keys are: color, ls, linestyle, lw, linewidth, "
                "alpha, zorder, solid_capstyle, dash_capstyle."
            )

        style.update(custom)
        return style

    for i, ((freq, psd), label) in enumerate(zip(psd_list, labels)):
        f_plot, y_plot = _prepare_psd(freq, psd)
        style = _get_curve_style(i, label)

        ax.plot(
            f_plot,
            y_plot,
            label=label,
            **style,
        )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    if fmin is not None or fmax is not None:
        ax.set_xlim(left=fmin, right=fmax)

    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    if xlabel is None:
        xlabel = f"{f_label}"

    if ylabel is None:
        if normalised:
            ylabel = "Normalised PSD"
        else:
            ylabel = f"{psd_label}"

    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.tick_params(
        which="major",
        direction="in",
        length=5,
        width=1.0,
        labelsize=tick_fs,
        pad=4,
        top=True,
        right=True,
    )

    ax.tick_params(
        which="minor",
        direction="in",
        length=3,
        width=0.8,
        top=True,
        right=True,
    )

    if journal_style:
        ax.grid(False)
    else:
        ax.grid(True, which="major", color="0.88", lw=0.6)
        ax.grid(True, which="minor", color="0.93", lw=0.4)

    if show_legend:
        if legend_ncol is None:
            if n_psd <= 4:
                legend_ncol = 1
            elif n_psd <= 8:
                legend_ncol = 2
            else:
                legend_ncol = 3

        ax.legend(
            frameon=False,
            fontsize=legend_fs,
            loc=_legend_loc_from_mode(legend_loc),
            ncol=legend_ncol,
            handlelength=3.0,
            columnspacing=1.0,
            borderaxespad=0.4,
        )

    if save:
        path = Path(savepath)

        if saveformat is None:
            if path.suffix:
                saveformat = path.suffix.lower().lstrip(".")
            else:
                saveformat = "pdf"
                path = path.with_suffix(".pdf")

        path.parent.mkdir(parents=True, exist_ok=True)

        def _save_one(fmt):
            fmt = fmt.lower().lstrip(".")

            if fmt == "fig":
                fig_path = path.with_suffix(".fig")
                with open(fig_path, "wb") as f:
                    pickle.dump(fig, f)

            elif fmt in {"pdf", "eps", "svg"}:
                out = path.with_suffix(f".{fmt}")
                fig.savefig(
                    out,
                    format=fmt,
                    bbox_inches="tight",
                    pad_inches=0.02,
                    transparent=False,
                )

            elif fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
                out = path.with_suffix(f".{fmt}")
                fig.savefig(
                    out,
                    format=fmt,
                    dpi=max(dpi, 300),
                    bbox_inches="tight",
                    pad_inches=0.02,
                    transparent=False,
                )

            else:
                raise ValueError(
                    f"Unsupported save format '{fmt}'. "
                    "Use pdf, eps, png, tiff, jpg, jpeg, svg, fig, or all."
                )

        if saveformat.lower() == "all":
            for fmt in ("pdf", "png", "fig"):
                _save_one(fmt)
        else:
            _save_one(saveformat)

    return fig, ax
#%%


# def plot_psf_grid_aa(
#     psf_groups,
#     wvl,
#     telescope_diameter,
#     sampling,
#     vmin=1e-6,
#     vmax=1,
#     cmap="inferno",
#     xlabel=r"[arcsec]",
#     ylabel=r"[arcsec]",
#     cbar_label="Normalized intensity",
#     nx=None,
#     titles=None,
#     row_labels=None,
#     col_labels=None,
#     figsize=None,
#     dpi=1200,
#     origin="lower",
#     save=False,
#     savepath="psf_grid.pdf",
#     saveformat=None,
#     share_colorbar=True,
#     individual_colorbars=False,
#     hide_inner_labels=True,
#     normalize_each=True,
#     use_log_norm=True,
#     cbar_position=(0.18, 0.085, 0.64, 0.025),
#     layout_rect=(0.0, 0.13, 1.0, 1.0),
#     wspace=0.25,
#     hspace=0.32,
# ):
#     """
#     Trace plusieurs PSF sous forme de grille avec un style adapté à Astronomy & Astrophysics.

#     Parameters
#     ----------
#     psf_groups : list[list[np.ndarray or None]]
#         Liste de lignes. Chaque ligne contient des PSF 2D ou None.
#         Une entrée None produit une cellule vide sans axe visible.

#     wvl : float
#         Longueur d'onde en mètres.

#     telescope_diameter : float
#         Diamètre du télescope en mètres.

#     sampling : float
#         Facteur d'échantillonnage.

#     vmin, vmax : float
#         Bornes de normalisation de l'image affichée.

#     cmap : str
#         Colormap Matplotlib.

#     xlabel, ylabel : str
#         Labels des axes.

#     cbar_label : str
#         Label de la colorbar.

#     nx : int or None
#         Taille du crop central carré. Si None, chaque image est gardée entière.

#     titles : list[list[str or None]] or None
#         Titres individuels, même structure que psf_groups.

#     row_labels : list[str] or None
#         Labels affichés sur la première colonne, un par ligne.

#     col_labels : list[str] or None
#         Labels affichés au-dessus des colonnes.
#         Ignorés pour une case si titles[i][j] existe déjà.

#     figsize : tuple or None
#         Taille de figure en pouces.

#     share_colorbar : bool
#         Si True, utilise une colorbar commune horizontale en bas.

#     individual_colorbars : bool
#         Si True et share_colorbar=False, ajoute une colorbar par axe.

#     hide_inner_labels : bool
#         Si True, seuls les axes extérieurs gardent les labels.

#     normalize_each : bool
#         Si True :
#             chaque image est normalisée par sa somme puis par son maximum.
#             C'est le comportement de ta fonction plot_psf_aa originale.

#         Si False :
#             les images sont affichées telles quelles.
#             À utiliser pour des PSF ou résidus déjà normalisés.

#     use_log_norm : bool
#         Si True, utilise LogNorm(vmin, vmax).
#         Si False, utilise une normalisation linéaire avec vmin/vmax.

#     cbar_position : tuple
#         Position de la colorbar commune :
#         (left, bottom, width, height), en coordonnées relatives à la figure.

#     layout_rect : tuple
#         Rectangle réservé aux subplots dans tight_layout :
#         (left, bottom, right, top).
#         Le bottom doit être supérieur à la position verticale de la colorbar.

#     wspace, hspace : float
#         Espacement horizontal et vertical entre les subplots.

#     Returns
#     -------
#     fig, axes
#         Figure Matplotlib et tableau 2D d'axes.
#     """

#     if not isinstance(psf_groups, (list, tuple)) or len(psf_groups) == 0:
#         raise ValueError("psf_groups doit être une liste non vide de lignes.")

#     nrows = len(psf_groups)
#     ncols = max(len(row) for row in psf_groups)

#     if ncols == 0:
#         raise ValueError("psf_groups ne contient aucune colonne.")

#     if share_colorbar and individual_colorbars:
#         raise ValueError(
#             "share_colorbar=True et individual_colorbars=True sont incompatibles. "
#             "Une colorbar commune ou des colorbars individuelles, pas les deux."
#         )

#     valid_psfs = [
#         psf
#         for row in psf_groups
#         for psf in row
#         if psf is not None
#     ]

#     if len(valid_psfs) == 0:
#         raise ValueError("Aucune PSF valide à tracer : toutes les entrées sont None.")

#     rc_params = {
#         "font.size": 8,
#         "axes.labelsize": 8,
#         "axes.titlesize": 8,
#         "xtick.labelsize": 7,
#         "ytick.labelsize": 7,
#         "legend.fontsize": 7,
#         "font.family": "serif",
#         "mathtext.fontset": "cm",
#         "axes.linewidth": 0.8,
#         "xtick.direction": "in",
#         "ytick.direction": "in",
#         "xtick.top": True,
#         "ytick.right": True,
#     }

#     pix_scale = wvl / telescope_diameter / sampling
#     rad2arcsec = 180 / np.pi * 3600
#     pix_scale_arcsec = pix_scale * rad2arcsec

#     if use_log_norm:
#         norm = LogNorm(vmin=vmin, vmax=vmax)
#     else:
#         norm = plt.Normalize(vmin=vmin, vmax=vmax)

#     if figsize is None:
#         figsize = (
#             max(3.35, 2.4 * ncols),
#             max(2.5, 2.3 * nrows),
#         )

#     def _center_crop(arr, crop_size, i, j):
#         ny0, nx0 = arr.shape

#         if crop_size > ny0 or crop_size > nx0:
#             raise ValueError(
#                 f"nx={crop_size} est trop grand pour la PSF en position ({i}, {j}) "
#                 f"de taille {arr.shape}."
#             )

#         cy, cx = ny0 // 2, nx0 // 2
#         half = crop_size // 2

#         y_start = cy - half
#         x_start = cx - half
#         y_end = y_start + crop_size
#         x_end = x_start + crop_size

#         return arr[y_start:y_end, x_start:x_end]

#     def _prepare_image(psf, i, j):
#         psf = np.asarray(psf, dtype=float)

#         if psf.ndim != 2:
#             raise ValueError(
#                 f"La PSF en position ({i}, {j}) doit être un array 2D."
#             )

#         if not np.any(np.isfinite(psf)):
#             raise ValueError(
#                 f"La PSF en position ({i}, {j}) ne contient aucune valeur finie."
#             )

#         psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)

#         if nx is not None:
#             psf = _center_crop(psf, nx, i, j)

#         if normalize_each:
#             psf_sum = np.sum(psf)

#             if psf_sum <= 0:
#                 raise ValueError(
#                     f"La PSF en position ({i}, {j}) a une somme nulle ou négative."
#                 )

#             image = psf / psf_sum

#             peak = np.max(image)
#             if peak <= 0:
#                 raise ValueError(
#                     f"La PSF en position ({i}, {j}) a un maximum nul ou négatif."
#                 )

#             image = image / peak

#         else:
#             image = psf.copy()

#         if use_log_norm:
#             # LogNorm ne supporte pas les valeurs <= 0.
#             image = np.where(image > 0, image, np.nan)

#         ny, nx_img = image.shape

#         x_axis = (np.arange(nx_img) - (nx_img - 1) / 2) * pix_scale_arcsec
#         y_axis = (np.arange(ny) - (ny - 1) / 2) * pix_scale_arcsec

#         extent = [
#             x_axis[0],
#             x_axis[-1],
#             y_axis[0],
#             y_axis[-1],
#         ]

#         return image, extent

#     with plt.rc_context(rc_params):
#         fig, axes = plt.subplots(
#             nrows,
#             ncols,
#             figsize=figsize,
#             squeeze=False,
#         )

#         images = []
#         image_axes = []

#         for i in range(nrows):
#             row = psf_groups[i]

#             for j in range(ncols):
#                 ax = axes[i, j]

#                 if j >= len(row) or row[j] is None:
#                     ax.set_visible(False)
#                     continue

#                 image, extent = _prepare_image(row[j], i, j)

#                 im = ax.imshow(
#                     image,
#                     norm=norm,
#                     cmap=cmap,
#                     extent=extent,
#                     origin=origin,
#                 )

#                 images.append(im)
#                 image_axes.append(ax)

#                 # Titre individuel prioritaire
#                 title_set = False
#                 if titles is not None:
#                     if i < len(titles) and j < len(titles[i]):
#                         if titles[i][j] is not None:
#                             ax.set_title(titles[i][j])
#                             title_set = True

#                 # Label de colonne si pas déjà un titre individuel
#                 if not title_set and col_labels is not None:
#                     if i == 0 and j < len(col_labels):
#                         ax.set_title(col_labels[j])

#                 # Labels d'axes
#                 if hide_inner_labels:
#                     if i == nrows - 1:
#                         ax.set_xlabel(xlabel)
#                     else:
#                         ax.set_xlabel("")
#                         ax.set_xticklabels([])

#                     if j == 0:
#                         if row_labels is not None and i < len(row_labels):
#                             ax.set_ylabel(row_labels[i])
#                         else:
#                             ax.set_ylabel(ylabel)
#                     else:
#                         ax.set_ylabel("")
#                         ax.set_yticklabels([])
#                 else:
#                     ax.set_xlabel(xlabel)

#                     if row_labels is not None and j == 0 and i < len(row_labels):
#                         ax.set_ylabel(row_labels[i])
#                     else:
#                         ax.set_ylabel(ylabel)

#         if len(images) == 0:
#             raise ValueError("Aucune image valide n'a été tracée.")

#         if share_colorbar:
#             fig.tight_layout(rect=layout_rect)
#             fig.subplots_adjust(wspace=wspace, hspace=hspace)

#             cax = fig.add_axes(cbar_position)

#             cbar = fig.colorbar(
#                 images[0],
#                 cax=cax,
#                 orientation="horizontal",
#             )
#             cbar.set_label(cbar_label)

#         else:
#             fig.tight_layout()
#             fig.subplots_adjust(wspace=wspace, hspace=hspace)

#             if individual_colorbars:
#                 for im, ax in zip(images, image_axes):
#                     cbar = fig.colorbar(
#                         im,
#                         ax=ax,
#                         fraction=0.046,
#                         pad=0.04,
#                     )
#                     cbar.set_label(cbar_label)

#         if save:
#             path = Path(savepath)

#             if saveformat is None:
#                 if path.suffix:
#                     saveformat = path.suffix.lower().lstrip(".")
#                 else:
#                     saveformat = "pdf"
#                     path = path.with_suffix(".pdf")

#             path.parent.mkdir(parents=True, exist_ok=True)

#             def _save_one(fmt):
#                 fmt = fmt.lower().lstrip(".")

#                 if fmt == "fig":
#                     fig_path = path.with_suffix(".fig")
#                     with open(fig_path, "wb") as f:
#                         pickle.dump(fig, f)

#                 elif fmt in {"pdf", "eps", "svg"}:
#                     out = path.with_suffix(f".{fmt}")
#                     fig.savefig(
#                         out,
#                         format=fmt,
#                         bbox_inches="tight",
#                         pad_inches=0.02,
#                         transparent=False,
#                     )

#                 elif fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
#                     out = path.with_suffix(f".{fmt}")
#                     fig.savefig(
#                         out,
#                         format=fmt,
#                         dpi=max(dpi, 300),
#                         bbox_inches="tight",
#                         pad_inches=0.02,
#                         transparent=False,
#                     )

#                 else:
#                     raise ValueError(
#                         f"Unsupported save format '{fmt}'. "
#                         "Use pdf, eps, png, tiff, jpg, jpeg, svg, fig, or all."
#                     )

#             if saveformat.lower() == "all":
#                 for fmt in ("png", "pdf", "fig"):
#                     _save_one(fmt)
#             else:
#                 _save_one(saveformat)

#     return fig, axes
#%%
def plot_curves_aa(
    x_list,
    y_list,
    labels=None,
    curve_styles=None,
    xlabel=None,
    ylabel=None,
    title=None,
    xlim=None,
    ylim=None,
    scale="linear",
    figure_width="one_column",
    figsize=None,
    height_ratio=0.72,
    dpi=300,
    journal_style=True,
    grid=False,
    grid_kwargs=None,
    label_fs=9,
    tick_fs=8,
    legend_fs=8,
    title_fs=9,
    tick_length=5,
    tick_width=1.0,
    minor_tick_length=3,
    minor_tick_width=0.8,
    spine_width=1.0,
    ticks_top=True,
    ticks_right=True,
    major_nbins=None,
    show_legend=True,
    legend_loc="best",
    legend_ncol=1,
    legend_handlelength=3.0,
    legend_frameon=False,
    legend_kwargs=None,
    use_labellines=False,
    labellines_fallback="legend",
    labelline_kwargs=None,
    save=False,
    savepath="curves_aa.pdf",
    saveformat=None,
    sort_by_x=False,
    min_points=2,
    ax=None,
):
    """
    Plot one or several generic curves using an A&A-compatible publication style.

    This function is designed to be inserted in a plotting utility module that
    already imports NumPy, Matplotlib, pathlib.Path, pickle, and MaxNLocator.
    It reuses the existing helper functions `_axis_scale_from_mode`,
    `_valid_xy_for_scale`, and `_legend_loc_from_mode` when available.

    The default style follows common Astronomy & Astrophysics figure constraints:
    compact one-column width, readable labels after reduction, no decorative
    background, no grid in publication mode, inward ticks, sufficiently thick
    lines, restrained colors, and line styles that remain distinguishable in
    grayscale.

    Parameters
    ----------
    x_list, y_list : list of array-like
        Lists containing the x and y values for each curve. The two lists must
        have the same length. For each curve, `x_list[i]` and `y_list[i]` are
        converted to one-dimensional arrays and must contain the same number of
        elements.

    labels : list of str or None, optional
        Curve labels. If None, labels are generated as "Curve 1", "Curve 2",
        etc. Labels are used both for legends and for dictionary-based
        `curve_styles` lookup.

    curve_styles : list of dict, dict, or None, optional
        Optional per-curve Matplotlib style customisation.

        Accepted forms are:

        1. List or tuple of dictionaries, one dictionary per curve:

            curve_styles = [
                {"color": "black", "lw": 1.6, "ls": "-"},
                {"color": "#355C9A", "lw": 1.6, "ls": "--"},
            ]

        2. Dictionary indexed by curve number:

            curve_styles = {
                0: {"marker": "o", "ms": 3.0},
                2: {"zorder": 5},
            }

        3. Dictionary indexed by label:

            curve_styles = {
                "Model A": {"color": "black", "ls": "-"},
                "Model B": {"color": "#355C9A", "ls": "--"},
            }

        Supported keys are all common Matplotlib `Axes.plot` keyword arguments,
        including color, lw, linewidth, ls, linestyle, marker, markersize, ms,
        alpha, zorder, drawstyle, solid_capstyle, dash_capstyle, markerfacecolor,
        markeredgecolor, markeredgewidth, and similar parameters.

    xlabel, ylabel : str or None, optional
        Axis labels. Include units explicitly, for example "Frequency [Hz]".

    title : str or None, optional
        Optional axis title.

    xlim, ylim : tuple or None, optional
        Axis limits as `(min, max)`. Use None for automatic limits.

    scale : {"linear", "xlog", "ylog", "log", "loglog"}, optional
        Axis scale mode. For logarithmic axes, non-finite values and non-positive
        values on logarithmic axes are removed before plotting. If fewer than
        `min_points` valid points remain for a curve, a clear ValueError is
        raised.

    figure_width : {"one_column", "intermediate", "full_page"} or float, optional
        Figure width preset or custom width in millimetres.

        - "one_column": 88 mm
        - "intermediate": 120 mm
        - "full_page": 180 mm
        - float: custom width in millimetres

    figsize : tuple or None, optional
        Custom figure size in inches. If provided, it overrides `figure_width`
        and `height_ratio`.

    height_ratio : float, optional
        Figure height divided by figure width when `figsize` is None.

    dpi : int, optional
        Figure dpi. Bitmap exports are saved with at least 300 dpi.

    journal_style : bool, optional
        If True, use final publication style: no grid and no background
        decoration. If False, a light grid can be enabled for working plots.

    grid : bool, optional
        Whether to show a grid. In publication mode (`journal_style=True`), the
        grid remains disabled regardless of this value.

    grid_kwargs : dict or None, optional
        Extra keyword arguments passed to `Axes.grid` when a grid is enabled.

    label_fs, tick_fs, legend_fs, title_fs : float, optional
        Font sizes for axis labels, ticks, legend, and title.

    tick_length, tick_width : float, optional
        Major tick length and width.

    minor_tick_length, minor_tick_width : float, optional
        Minor tick length and width.

    spine_width : float, optional
        Axis spine width.

    ticks_top, ticks_right : bool, optional
        Whether to draw ticks on the top and right sides.

    major_nbins : int or None, optional
        If provided, apply `MaxNLocator(nbins=major_nbins)` to major ticks on
        linear axes.

    show_legend : bool, optional
        Whether to display a legend when `use_labellines=False`, or as fallback
        when line labels cannot be used.

    legend_loc : str or None, optional
        Legend location. Existing aliases accepted by `_legend_loc_from_mode`
        are supported, including "best", "free", "upper right", "upper left",
        "lower right", and "lower left".

    legend_ncol : int, optional
        Number of legend columns.

    legend_handlelength : float, optional
        Legend handle length.

    legend_frameon : bool, optional
        Whether to draw a legend frame. Default is False for a sober publication
        style.

    legend_kwargs : dict or None, optional
        Extra keyword arguments passed to `Axes.legend`.

    use_labellines : bool, optional
        If True, place labels directly on the curves using the optional
        `labellines` package.

    labellines_fallback : {"legend", "error"}, optional
        Behaviour if `use_labellines=True` but `labellines` is not installed.

        - "legend": fall back to a standard legend.
        - "error": raise an ImportError with a clear message.

    labelline_kwargs : dict or None, optional
        Extra keyword arguments passed to `labellines.labelLines`.

    save : bool, optional
        If True, save the figure.

    savepath : str or pathlib.Path, optional
        Output file path. If no suffix is present and `saveformat` is None,
        ".pdf" is appended.

    saveformat : str or None, optional
        Output format. Accepted values are "pdf", "eps", "svg", "png", "tif",
        "tiff", "jpg", "jpeg", "fig", and "all". If None, the suffix of
        `savepath` is used, or "pdf" if no suffix is provided.

    sort_by_x : bool, optional
        If True, sort each curve by increasing x after filtering. This is useful
        for lines, but should be disabled if the input order has a special
        meaning.

    min_points : int, optional
        Minimum number of valid points required per curve after filtering.

    ax : matplotlib.axes.Axes or None, optional
        Existing axis on which to draw. If None, a new figure and axis are
        created.

    Returns
    -------
    fig, ax : tuple
        The Matplotlib figure and axis.

    Examples
    --------
    Minimal example with three curves:

    >>> x = np.linspace(0.1, 10.0, 300)
    >>> y1 = np.sin(x) / x
    >>> y2 = np.cos(x) / (1.0 + 0.1 * x)
    >>> y3 = 0.2 * np.exp(-0.2 * x)
    >>>
    >>> fig, ax = plot_curves_aa(
    ...     [x, x, x],
    ...     [y1, y2, y3],
    ...     labels=["Sinc-like", "Damped cosine", "Exponential"],
    ...     xlabel="Time [s]",
    ...     ylabel="Amplitude",
    ...     figure_width="one_column",
    ...     save=True,
    ...     savepath="example_curves_aa.pdf",
    ... )
    """

    # ------------------------------------------------------------------
    # Validate global inputs.
    # ------------------------------------------------------------------
    if not isinstance(x_list, (list, tuple)):
        raise TypeError("x_list must be a list or tuple of array-like objects.")

    if not isinstance(y_list, (list, tuple)):
        raise TypeError("y_list must be a list or tuple of array-like objects.")

    if len(x_list) == 0:
        raise ValueError("x_list and y_list must contain at least one curve.")

    if len(x_list) != len(y_list):
        raise ValueError("x_list and y_list must have the same length.")

    n_curves = len(x_list)

    if labels is None:
        labels = [f"Curve {i + 1}" for i in range(n_curves)]
    else:
        if len(labels) != n_curves:
            raise ValueError("labels must have the same length as x_list and y_list.")
        labels = [str(label) for label in labels]

    if labellines_fallback not in {"legend", "error"}:
        raise ValueError("labellines_fallback must be either 'legend' or 'error'.")

    if min_points < 1:
        raise ValueError("min_points must be at least 1.")

    # Reuse the existing helper if the function is inserted in the current file.
    xscale, yscale = _axis_scale_from_mode(scale)

    # ------------------------------------------------------------------
    # Figure size.
    # ------------------------------------------------------------------
    if figsize is None:
        if isinstance(figure_width, str):
            width_key = figure_width.strip().lower().replace("-", "_").replace(" ", "_")
            width_map_mm = {
                "one_column": 88.0,
                "single_column": 88.0,
                "column": 88.0,
                "intermediate": 120.0,
                "medium": 120.0,
                "full_page": 180.0,
                "full_width": 180.0,
                "page": 180.0,
            }

            if width_key not in width_map_mm:
                raise ValueError(
                    "figure_width must be 'one_column', 'intermediate', "
                    "'full_page', or a custom width in millimetres."
                )

            width_mm = width_map_mm[width_key]

        else:
            width_mm = float(figure_width)
            if width_mm <= 0:
                raise ValueError("Custom figure_width must be positive.")

        width_in = width_mm / 25.4
        height_in = width_in * float(height_ratio)
        figsize = (width_in, height_in)

    # ------------------------------------------------------------------
    # Create or reuse the axis.
    # ------------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize,
            dpi=dpi,
            constrained_layout=True,
        )
    else:
        fig = ax.figure

    # ------------------------------------------------------------------
    # Default A&A-compatible style cycle.
    # Avoid red/green as a pair and do not rely on color alone.
    # ------------------------------------------------------------------
    default_colors = [
        "black",
        "#355C9A",  # muted dark blue
        "#8A5A00",  # muted brown/ochre
        "#6A4C93",  # muted purple
        "0.35",
        "#2F4F4F",  # dark slate
        "0.55",
        "#4C72B0",  # restrained blue
    ]

    default_linestyles = [
        "-",
        (0, (7, 3)),
        (0, (3, 2)),
        (0, (1, 2)),
        (0, (5, 2, 1, 2)),
        (0, (9, 3, 2, 3)),
    ]

    default_markers = [
        None,
        None,
        None,
        None,
        "o",
        "s",
        "^",
        "D",
    ]

    if n_curves <= 4:
        default_lw = 1.6
    elif n_curves <= 10:
        default_lw = 1.3
    else:
        default_lw = 1.1

    def _normalise_style_aliases(style):
        """Return a copy of a Matplotlib style dictionary with common aliases."""
        style = dict(style)

        if "linestyle" in style and "ls" not in style:
            style["ls"] = style.pop("linestyle")

        if "linewidth" in style and "lw" not in style:
            style["lw"] = style.pop("linewidth")

        if "markersize" in style and "ms" not in style:
            style["ms"] = style.pop("markersize")

        if "markeredgewidth" in style and "mew" not in style:
            style["mew"] = style.pop("markeredgewidth")

        return style

    def _get_curve_style(i, label):
        """Build the final style dictionary for curve i."""
        style = {
            "color": default_colors[i % len(default_colors)],
            "lw": default_lw,
            "ls": default_linestyles[i % len(default_linestyles)],
            "marker": default_markers[i % len(default_markers)],
            "ms": 3.0,
            "alpha": 1.0,
            "zorder": 3 + i,
            "solid_capstyle": "round",
            "dash_capstyle": "butt",
        }

        if curve_styles is None:
            return style

        if isinstance(curve_styles, (list, tuple)):
            if len(curve_styles) != n_curves:
                raise ValueError(
                    "If curve_styles is a list or tuple, it must have the same "
                    "length as x_list and y_list."
                )
            custom = curve_styles[i]

        elif isinstance(curve_styles, dict):
            if label in curve_styles:
                custom = curve_styles[label]
            elif i in curve_styles:
                custom = curve_styles[i]
            else:
                custom = None

        else:
            raise TypeError(
                "curve_styles must be None, a list or tuple of dictionaries, "
                "or a dictionary indexed by curve index or label."
            )

        if custom is None:
            return style

        if not isinstance(custom, dict):
            raise TypeError(
                f"curve_styles for curve {i} must be a dictionary, "
                f"got {type(custom)}."
            )

        custom = _normalise_style_aliases(custom)
        style.update(custom)
        return style

    def _prepare_curve(x, y, i):
        """Convert, validate, filter, and optionally sort one curve."""
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        if x.size != y.size:
            raise ValueError(
                f"Curve {i} has incompatible dimensions: "
                f"x has {x.size} points but y has {y.size} points."
            )

        if x.size == 0:
            raise ValueError(f"Curve {i} is empty.")

        valid = _valid_xy_for_scale(x, y, xscale, yscale)

        x_valid = x[valid]
        y_valid = y[valid]

        if x_valid.size < min_points:
            if xscale == "log" or yscale == "log":
                raise ValueError(
                    f"Curve {i} has fewer than {min_points} valid points after "
                    "removing non-finite values and non-positive values required "
                    "by logarithmic axes."
                )

            raise ValueError(
                f"Curve {i} has fewer than {min_points} valid finite points."
            )

        if sort_by_x:
            order = np.argsort(x_valid)
            x_valid = x_valid[order]
            y_valid = y_valid[order]

        return x_valid, y_valid

    # ------------------------------------------------------------------
    # Plot curves.
    # ------------------------------------------------------------------
    plotted_lines = []

    for i, (x, y, label) in enumerate(zip(x_list, y_list, labels)):
        x_plot, y_plot = _prepare_curve(x, y, i)
        style = _get_curve_style(i, label)

        line, = ax.plot(
            x_plot,
            y_plot,
            label=label,
            **style,
        )
        plotted_lines.append(line)

    # ------------------------------------------------------------------
    # Axes, limits, and labels.
    # ------------------------------------------------------------------
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    if xlim is not None:
        if len(xlim) != 2:
            raise ValueError("xlim must be a tuple or list of length 2.")
        ax.set_xlim(xlim)

    if ylim is not None:
        if len(ylim) != 2:
            raise ValueError("ylim must be a tuple or list of length 2.")
        ax.set_ylim(ylim)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fs)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fs)

    if title is not None:
        ax.set_title(title, fontsize=title_fs)

    # ------------------------------------------------------------------
    # Ticks and frame.
    # ------------------------------------------------------------------
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    ax.tick_params(
        which="major",
        direction="in",
        length=tick_length,
        width=tick_width,
        labelsize=tick_fs,
        pad=4,
        top=ticks_top,
        right=ticks_right,
    )

    ax.tick_params(
        which="minor",
        direction="in",
        length=minor_tick_length,
        width=minor_tick_width,
        top=ticks_top,
        right=ticks_right,
    )

    if major_nbins is not None:
        if xscale == "linear":
            ax.xaxis.set_major_locator(MaxNLocator(nbins=major_nbins))
        if yscale == "linear":
            ax.yaxis.set_major_locator(MaxNLocator(nbins=major_nbins))

    # ------------------------------------------------------------------
    # Grid policy.
    # ------------------------------------------------------------------
    if journal_style:
        ax.grid(False)
    else:
        if grid:
            if grid_kwargs is None:
                grid_kwargs = {
                    "which": "major",
                    "color": "0.88",
                    "lw": 0.6,
                }
            ax.grid(True, **grid_kwargs)
        else:
            ax.grid(False)

    # ------------------------------------------------------------------
    # Legend or labels on lines.
    # ------------------------------------------------------------------
    legend_was_drawn = False

    if use_labellines:
        try:
            from labellines import labelLines
            labellines_available = True
        except ImportError:
            labellines_available = False

        if labellines_available:
            if labelline_kwargs is None:
                labelline_kwargs = {}

            default_labelline_kwargs = {
                "align": True,
                "fontsize": legend_fs,
                "zorder": 10,
            }
            default_labelline_kwargs.update(labelline_kwargs)

            labelLines(plotted_lines, **default_labelline_kwargs)

        else:
            if labellines_fallback == "error":
                raise ImportError(
                    "use_labellines=True requires the optional 'labellines' "
                    "package. Install it or set labellines_fallback='legend'."
                )

            if show_legend:
                legend_was_drawn = True

    elif show_legend:
        legend_was_drawn = True

    if legend_was_drawn:
        if legend_kwargs is None:
            legend_kwargs = {}

        ax.legend(
            frameon=legend_frameon,
            fontsize=legend_fs,
            loc=_legend_loc_from_mode(legend_loc),
            ncol=legend_ncol,
            handlelength=legend_handlelength,
            borderaxespad=0.4,
            **legend_kwargs,
        )

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------
    if save:
        path = Path(savepath)

        if saveformat is None:
            if path.suffix:
                saveformat = path.suffix.lower().lstrip(".")
            else:
                saveformat = "pdf"
                path = path.with_suffix(".pdf")

        saveformat = str(saveformat).lower().lstrip(".")
        path.parent.mkdir(parents=True, exist_ok=True)

        def _save_one(fmt):
            """Save the figure in one supported format."""
            fmt = fmt.lower().lstrip(".")

            if fmt == "fig":
                fig_path = path.with_suffix(".fig")
                with open(fig_path, "wb") as f:
                    pickle.dump(fig, f)

            elif fmt in {"pdf", "eps", "svg"}:
                out = path.with_suffix(f".{fmt}")
                fig.savefig(
                    out,
                    format=fmt,
                    bbox_inches="tight",
                    pad_inches=0.02,
                    transparent=False,
                )

            elif fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
                out = path.with_suffix(f".{fmt}")
                fig.savefig(
                    out,
                    format=fmt,
                    dpi=max(dpi, 300),
                    bbox_inches="tight",
                    pad_inches=0.02,
                    transparent=False,
                )

            else:
                raise ValueError(
                    f"Unsupported save format '{fmt}'. "
                    "Use pdf, eps, svg, png, tif, tiff, jpg, jpeg, fig, or all."
                )

        if saveformat == "all":
            for fmt in ("pdf", "eps", "png", "fig"):
                _save_one(fmt)
        else:
            _save_one(saveformat)

    return fig, ax

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import pickle


def plot_opd_grid_aa(
    opd_groups,
    titles=None,
    row_labels=None,
    col_labels=None,
    telescope_diameter=1.52,
    unit_factor=1e9,
    unit_label="nm",
    cmap="seismic",
    nan_color="0.85",
    vmin=None,
    vmax=None,
    symmetric=True,
    same_color_scale=True,
    xlabel=r"$x$ [m]",
    ylabel=r"$y$ [m]",
    cbar_label=None,
    figsize=None,
    dpi=300,
    origin="lower",
    interpolation="none",
    hide_inner_labels=True,
    share_colorbar=True,
    individual_colorbars=False,
    cbar_position=(0.18, 0.09, 0.64, 0.025),
    layout_rect=(0.0, 0.13, 1.0, 1.0),
    wspace=0.25,
    hspace=0.28,
    journal_style=True,
    save=False,
    savepath="opd_grid_aa.pdf",
    saveformat=None,
):
    """
    Plot several OPD maps as a grid with an A&A-like style.

    Parameters
    ----------
    opd_groups : list[list[np.ndarray or None]]
        List of rows. Each row contains 2D OPD maps or None.
        A None entry creates an empty hidden cell.

    titles : list[list[str or None]] or None
        Individual subplot titles with the same structure as opd_groups.

    row_labels : list[str] or None
        Labels displayed on the first column, one per row.

    col_labels : list[str] or None
        Labels displayed above the columns if no individual title is set.

    telescope_diameter : float
        Telescope diameter in meters.

    unit_factor : float
        Multiplicative factor applied before plotting.
        Example: 1e9 converts OPD from meters to nanometers.

    unit_label : str
        Display unit after applying unit_factor.

    cmap : str
        Matplotlib colormap.

    nan_color : color
        Color used for NaN pixels.

    vmin, vmax : float or None
        Display color scale limits.
        If same_color_scale=True, they apply to the common scale.
        If same_color_scale=False, they are used independently per map.

    symmetric : bool
        If True, enforce a symmetric color scale around zero.

    same_color_scale : bool
        If True, all maps share the same color scale.

    xlabel, ylabel : str
        Axis labels.

    cbar_label : str or None
        Colorbar label.

    figsize : tuple or None
        Figure size in inches.

    dpi : int
        Figure dpi.

    origin : {"lower", "upper"}
        Image origin.

    interpolation : str
        Interpolation passed to imshow.

    hide_inner_labels : bool
        If True, only outer axes keep their labels.

    share_colorbar : bool
        If True, use one common horizontal colorbar.

    individual_colorbars : bool
        If True and share_colorbar=False, use one colorbar per subplot.

    cbar_position : tuple
        Position of the shared colorbar:
        (left, bottom, width, height).

    layout_rect : tuple
        Rectangle reserved for subplots in tight_layout.

    wspace, hspace : float
        Horizontal and vertical spacing.

    journal_style : bool
        If True, no background grid.

    save : bool
        If True, save the figure.

    savepath : str
        Output path.

    saveformat : str or None
        Output format.

    Returns
    -------
    fig, axes
    """

    if not isinstance(opd_groups, (list, tuple)) or len(opd_groups) == 0:
        raise ValueError("opd_groups must be a non-empty list of rows.")

    nrows = len(opd_groups)
    ncols = max(len(row) for row in opd_groups)

    if ncols == 0:
        raise ValueError("opd_groups does not contain any column.")

    if share_colorbar and individual_colorbars:
        raise ValueError(
            "share_colorbar=True and individual_colorbars=True are mutually exclusive."
        )

    valid_opds = [
        np.asarray(opd, dtype=float)
        for row in opd_groups
        for opd in row
        if opd is not None
    ]

    if len(valid_opds) == 0:
        raise ValueError("No valid OPD map to plot: all entries are None.")

    shapes = [opd.shape for opd in valid_opds]
    if any(len(shape) != 2 for shape in shapes):
        raise ValueError("All OPD maps must be 2D arrays.")

    if len(set(shapes)) != 1:
        raise ValueError(f"All OPD maps must have the same shape. Got {shapes}.")

    ny, nx = valid_opds[0].shape

    if cbar_label is None:
        cbar_label = rf"OPD [{unit_label}]"

    rc_params = {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    }

    if figsize is None:
        figsize = (
            max(3.35, 2.5 * ncols),
            max(2.5, 2.4 * nrows),
        )

    D = float(telescope_diameter)
    if D <= 0:
        raise ValueError("telescope_diameter must be > 0.")

    dx = D / nx
    dy = D / ny

    extent = [
        -D / 2 + dx / 2,
        +D / 2 - dx / 2,
        -D / 2 + dy / 2,
        +D / 2 - dy / 2,
    ]

    def _make_norm(data, vmin_in, vmax_in, symmetric_in):
        data = np.asarray(data, dtype=float)
        finite = data[np.isfinite(data)]

        if finite.size == 0:
            raise ValueError("Cannot define color scale from an all-NaN OPD map.")

        if vmin_in is None or vmax_in is None:
            if symmetric_in:
                vmax_auto = np.nanmax(np.abs(finite))
                vmin_auto = -vmax_auto
            else:
                vmin_auto = np.nanmin(finite)
                vmax_auto = np.nanmax(finite)

            if vmin_in is None:
                vmin_in = vmin_auto
            if vmax_in is None:
                vmax_in = vmax_auto

        if symmetric_in:
            vmax_abs = max(abs(vmin_in), abs(vmax_in))
            if vmax_abs == 0:
                vmax_abs = 1.0

            vmin_in = -vmax_abs
            vmax_in = +vmax_abs

            return colors.TwoSlopeNorm(
                vmin=vmin_in,
                vcenter=0.0,
                vmax=vmax_in,
            )

        if vmax_in == vmin_in:
            vmax_in = vmin_in + 1.0

        return colors.Normalize(vmin=vmin_in, vmax=vmax_in)

    finite_all = np.concatenate(
        [
            (np.asarray(opd, dtype=float) * unit_factor)[
                np.isfinite(np.asarray(opd, dtype=float) * unit_factor)
            ].ravel()
            for row in opd_groups
            for opd in row
            if opd is not None
        ]
    )

    if same_color_scale:
        common_norm = _make_norm(finite_all, vmin, vmax, symmetric)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=nan_color)

    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            dpi=dpi,
            squeeze=False,
        )

        images = []
        image_axes = []

        for i in range(nrows):
            row = opd_groups[i]

            for j in range(ncols):
                ax = axes[i, j]

                if j >= len(row) or row[j] is None:
                    ax.set_visible(False)
                    continue

                opd = np.asarray(row[j], dtype=float) * unit_factor
                opd_ma = np.ma.masked_invalid(opd)

                if opd_ma.count() == 0:
                    raise ValueError(f"OPD map at position ({i}, {j}) contains only NaNs.")

                if same_color_scale:
                    norm = common_norm
                else:
                    norm = _make_norm(opd, vmin, vmax, symmetric)

                im = ax.imshow(
                    opd_ma,
                    cmap=cmap_obj,
                    norm=norm,
                    origin=origin,
                    extent=extent,
                    interpolation=interpolation,
                    aspect="equal",
                )

                images.append(im)
                image_axes.append(ax)

                title_set = False
                if titles is not None:
                    if i < len(titles) and j < len(titles[i]):
                        if titles[i][j] is not None:
                            ax.set_title(titles[i][j])
                            title_set = True

                if not title_set and col_labels is not None:
                    if i == 0 and j < len(col_labels):
                        ax.set_title(col_labels[j])

                if hide_inner_labels:
                    if i == nrows - 1:
                        ax.set_xlabel(xlabel)
                    else:
                        ax.set_xlabel("")
                        ax.set_xticklabels([])

                    if j == 0:
                        if row_labels is not None and i < len(row_labels):
                            ax.set_ylabel(row_labels[i])
                        else:
                            ax.set_ylabel(ylabel)
                    else:
                        ax.set_ylabel("")
                        ax.set_yticklabels([])
                else:
                    ax.set_xlabel(xlabel)

                    if row_labels is not None and j == 0 and i < len(row_labels):
                        ax.set_ylabel(row_labels[i])
                    else:
                        ax.set_ylabel(ylabel)

                for spine in ax.spines.values():
                    spine.set_linewidth(1.0)

                ax.tick_params(
                    which="major",
                    direction="in",
                    length=5,
                    width=1.0,
                    labelsize=7,
                    pad=4,
                    top=True,
                    right=True,
                )

                ax.tick_params(
                    which="minor",
                    direction="in",
                    length=3,
                    width=0.8,
                    top=True,
                    right=True,
                )

                if journal_style:
                    ax.grid(False)
                else:
                    ax.grid(True, which="major", color="0.88", lw=0.6)
                    ax.grid(True, which="minor", color="0.93", lw=0.4)

        if len(images) == 0:
            raise ValueError("No valid OPD image has been plotted.")

        if share_colorbar:
            fig.tight_layout(rect=layout_rect)
            fig.subplots_adjust(wspace=wspace, hspace=hspace)

            cax = fig.add_axes(cbar_position)
            cbar = fig.colorbar(
                images[0],
                cax=cax,
                orientation="horizontal",
            )
            cbar.set_label(cbar_label)
            cbar.ax.tick_params(direction="in", pad=2, labelsize=7)
            cbar.locator = MaxNLocator(nbins=5)
            cbar.update_ticks()

        else:
            fig.tight_layout()
            fig.subplots_adjust(wspace=wspace, hspace=hspace)

            if individual_colorbars:
                for im, ax in zip(images, image_axes):
                    cbar = fig.colorbar(
                        im,
                        ax=ax,
                        fraction=0.046,
                        pad=0.04,
                    )
                    cbar.set_label(cbar_label)
                    cbar.ax.tick_params(direction="in", pad=2, labelsize=7)
                    cbar.locator = MaxNLocator(nbins=5)
                    cbar.update_ticks()

        if save:
            path = Path(savepath)

            if saveformat is None:
                if path.suffix:
                    saveformat = path.suffix.lower().lstrip(".")
                else:
                    saveformat = "pdf"
                    path = path.with_suffix(".pdf")

            path.parent.mkdir(parents=True, exist_ok=True)

            def _save_one(fmt):
                fmt = fmt.lower().lstrip(".")

                if fmt == "fig":
                    fig_path = path.with_suffix(".fig")
                    with open(fig_path, "wb") as f:
                        pickle.dump(fig, f)

                elif fmt in {"pdf", "eps", "svg"}:
                    out = path.with_suffix(f".{fmt}")
                    fig.savefig(
                        out,
                        format=fmt,
                        bbox_inches="tight",
                        pad_inches=0.02,
                        transparent=False,
                    )

                elif fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
                    out = path.with_suffix(f".{fmt}")
                    fig.savefig(
                        out,
                        format=fmt,
                        dpi=max(dpi, 300),
                        bbox_inches="tight",
                        pad_inches=0.02,
                        transparent=False,
                    )

                else:
                    raise ValueError(
                        f"Unsupported save format '{fmt}'. "
                        "Use pdf, eps, png, tiff, jpg, jpeg, svg, fig, or all."
                    )

            if saveformat.lower() == "all":
                for fmt in ("pdf", "png", "fig"):
                    _save_one(fmt)
            else:
                _save_one(saveformat)

    return fig, axes


#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import pickle


def plot_lq_depth_aa(
    depth33,
    depth76,
    lq_phase33,
    lq_int33,
    lq_phase76,
    lq_int76,
    method=np.nanmedian,
    mode_index=None,
    stroke_index=None,
    normalise_to_min=True,
    x_abs=True,
    yscale="linear",
    xlabel=r"Phase-shift depth [$\pi$ rad]",
    phase_ylabel="Phase LQ / min",
    intensity_ylabel="Intensity LQ / min",
    titles=(r"(a) ZWFS33", r"(b) ZWFS76"),
    one_column=False,
    dpi=300,
    journal_style=True,
    show_legend=True,
    legend_loc="best",
    save=False,
    savepath="lq_vs_depth_aa.pdf",
    saveformat=None,
):
    """
    Plot phase and intensity LQ criteria as a function of mask phase-shift depth.

    Parameters
    ----------
    depth33, depth76 : array-like
        Depth vectors for ZWFS33 and ZWFS76.

    lq_phase33, lq_int33, lq_phase76, lq_int76 : ndarray
        LQ arrays with shape (n_depth, n_modes, n_stroke).

    method : callable
        Reduction method used over modes and/or strokes.
        Typical choices: np.nanmedian, np.nanmean, np.nanmin.

    mode_index : int or None
        If not None, only this mode is plotted.

    stroke_index : int or None
        If not None, only this stroke index is plotted.

    normalise_to_min : bool
        If True, each curve is divided by its own finite positive minimum.

    x_abs : bool
        If True, plot abs(depth). Useful because ZWFS76 depth is negative.

    yscale : {"linear", "log"}
        Y scale for both left and right y axes.

    one_column : bool
        If True, use 88 mm width. If False, use 180 mm width.

    save : bool
        If True, save the figure.

    savepath : str
        Output path.

    saveformat : str or None
        "pdf", "png", "fig", "eps", "svg", "all", or None.
    """

    # -------------------------------------------------------------------------
    # Input conversion
    # -------------------------------------------------------------------------
    depth33 = np.asarray(depth33).ravel()
    depth76 = np.asarray(depth76).ravel()

    lq_phase33 = np.asarray(lq_phase33, dtype=float)
    lq_int33 = np.asarray(lq_int33, dtype=float)
    lq_phase76 = np.asarray(lq_phase76, dtype=float)
    lq_int76 = np.asarray(lq_int76, dtype=float)

    expected_ndim = 3
    for name, arr in [
        ("lq_phase33", lq_phase33),
        ("lq_int33", lq_int33),
        ("lq_phase76", lq_phase76),
        ("lq_int76", lq_int76),
    ]:
        if arr.ndim != expected_ndim:
            raise ValueError(
                f"{name} must have shape (n_depth, n_modes, n_stroke). "
                f"Got shape {arr.shape}."
            )

    if lq_phase33.shape[0] != depth33.size:
        raise ValueError(
            "depth33 length must match lq_phase33.shape[0]."
        )

    if lq_phase76.shape[0] != depth76.size:
        raise ValueError(
            "depth76 length must match lq_phase76.shape[0]."
        )

    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be either 'linear' or 'log'.")

    # -------------------------------------------------------------------------
    # Reduction helper
    # -------------------------------------------------------------------------
    def _reduce_lq(arr):
        """
        Reduce a LQ cube into one curve versus depth.
        """

        data = arr

        if mode_index is not None:
            data = data[:, mode_index, :]
        else:
            # keep all modes
            pass

        if stroke_index is not None:
            if mode_index is not None:
                data = data[:, stroke_index]
            else:
                data = data[:, :, stroke_index]

        # Now reduce whatever is left except depth axis.
        if data.ndim == 1:
            curve = data
        else:
            axes = tuple(range(1, data.ndim))
            curve = method(data, axis=axes)

        return np.asarray(curve, dtype=float).ravel()

    def _normalise(y):
        """
        Normalise by finite positive minimum.
        """

        y = np.asarray(y, dtype=float)
        y_norm = y.copy()

        if not normalise_to_min:
            return y_norm

        valid = np.isfinite(y_norm)

        if yscale == "log":
            valid &= y_norm > 0

        if not np.any(valid):
            raise ValueError(
                "Cannot normalise: no valid finite value found."
            )

        ymin = np.nanmin(y_norm[valid])

        if ymin == 0:
            valid_nonzero = valid & (y_norm != 0)
            if not np.any(valid_nonzero):
                raise ValueError(
                    "Cannot normalise to minimum: all valid values are zero."
                )
            ymin = np.nanmin(np.abs(y_norm[valid_nonzero]))

        y_norm = y_norm / ymin

        return y_norm

    def _prepare_xy(depth, y):
        """
        Prepare x and y arrays, removing invalid values.
        """

        x = np.abs(depth) if x_abs else depth.copy()
        y = np.asarray(y, dtype=float)

        valid = np.isfinite(x) & np.isfinite(y)

        if yscale == "log":
            valid &= y > 0

        x = x[valid]
        y = y[valid]

        order = np.argsort(x)

        return x[order], y[order]

    # -------------------------------------------------------------------------
    # Build curves
    # -------------------------------------------------------------------------
    phase33 = _normalise(_reduce_lq(lq_phase33))
    inten33 = _normalise(_reduce_lq(lq_int33))
    phase76 = _normalise(_reduce_lq(lq_phase76))
    inten76 = _normalise(_reduce_lq(lq_int76))

    x33_phase, phase33 = _prepare_xy(depth33, phase33)
    x33_int, inten33 = _prepare_xy(depth33, inten33)

    x76_phase, phase76 = _prepare_xy(depth76, phase76)
    x76_int, inten76 = _prepare_xy(depth76, inten76)

    # -------------------------------------------------------------------------
    # A&A style
    # -------------------------------------------------------------------------
    label_fs = 9
    tick_fs = 8
    legend_fs = 8
    title_fs = 9

    width_in = 88 / 25.4 if one_column else 180 / 25.4
    height_in = width_in * 0.42

    rc_params = {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    }

    col_phase = "black"
    col_int = "#355C9A"

    with plt.rc_context(rc_params):

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(width_in, height_in),
            dpi=dpi,
            constrained_layout=True,
        )

        ax33, ax76 = axes
        ax33_r = ax33.twinx()
        ax76_r = ax76.twinx()

        # ---------------------------------------------------------------------
        # ZWFS33
        # ---------------------------------------------------------------------
        line33_phase, = ax33.plot(
            x33_phase,
            phase33,
            color=col_phase,
            lw=1.5,
            ls="-",
            label="Phase LQ",
            zorder=3,
            solid_capstyle="round",
        )

        line33_int, = ax33_r.plot(
            x33_int,
            inten33,
            color=col_int,
            lw=1.5,
            ls=(0, (7, 3)),
            label="Intensity LQ",
            zorder=3,
            dash_capstyle="butt",
        )

        ax33.set_title(titles[0], fontsize=title_fs)

        # ---------------------------------------------------------------------
        # ZWFS76
        # ---------------------------------------------------------------------
        line76_phase, = ax76.plot(
            x76_phase,
            phase76,
            color=col_phase,
            lw=1.5,
            ls="-",
            label="Phase LQ",
            zorder=3,
            solid_capstyle="round",
        )

        line76_int, = ax76_r.plot(
            x76_int,
            inten76,
            color=col_int,
            lw=1.5,
            ls=(0, (7, 3)),
            label="Intensity LQ",
            zorder=3,
            dash_capstyle="butt",
        )

        ax76.set_title(titles[1], fontsize=title_fs)

        # ---------------------------------------------------------------------
        # Axes formatting
        # ---------------------------------------------------------------------
        for ax, ax_r in [(ax33, ax33_r), (ax76, ax76_r)]:

            ax.set_xlabel(xlabel, fontsize=label_fs)
            ax.set_ylabel(phase_ylabel, fontsize=label_fs)
            ax_r.set_ylabel(intensity_ylabel, fontsize=label_fs, labelpad=8)

            ax.set_yscale(yscale)
            ax_r.set_yscale(yscale)

            for a in [ax, ax_r]:
                for spine in a.spines.values():
                    spine.set_linewidth(1.0)

                a.tick_params(
                    which="major",
                    direction="in",
                    length=5,
                    width=1.0,
                    labelsize=tick_fs,
                    pad=4,
                    top=True,
                    right=True,
                )

                a.tick_params(
                    which="minor",
                    direction="in",
                    length=3,
                    width=0.8,
                    top=True,
                    right=True,
                )

                if journal_style:
                    a.grid(False)
                else:
                    a.grid(True, which="major", color="0.88", lw=0.6)
                    a.grid(True, which="minor", color="0.93", lw=0.4)

            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            if yscale == "linear":
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax_r.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # Hide duplicated y labels in the middle if desired:
        # ax76.set_ylabel("")

        # ---------------------------------------------------------------------
        # Legends
        # ---------------------------------------------------------------------
        if show_legend:
            ax33.legend(
                [line33_phase, line33_int],
                ["Phase LQ", "Intensity LQ"],
                frameon=False,
                fontsize=legend_fs,
                loc=legend_loc,
                handlelength=3.0,
                borderaxespad=0.4,
            )

            ax76.legend(
                [line76_phase, line76_int],
                ["Phase LQ", "Intensity LQ"],
                frameon=False,
                fontsize=legend_fs,
                loc=legend_loc,
                handlelength=3.0,
                borderaxespad=0.4,
            )

        # ---------------------------------------------------------------------
        # Save
        # ---------------------------------------------------------------------
        if save:
            path = Path(savepath)

            if saveformat is None:
                if path.suffix:
                    saveformat = path.suffix.lower().lstrip(".")
                else:
                    saveformat = "pdf"
                    path = path.with_suffix(".pdf")

            saveformat = str(saveformat).lower().lstrip(".")
            path.parent.mkdir(parents=True, exist_ok=True)

            def _save_one(fmt):
                fmt = fmt.lower().lstrip(".")

                if fmt == "fig":
                    fig_path = path.with_suffix(".fig")
                    with open(fig_path, "wb") as f:
                        pickle.dump(fig, f)

                elif fmt in {"pdf", "eps", "svg"}:
                    out = path.with_suffix(f".{fmt}")
                    fig.savefig(
                        out,
                        format=fmt,
                        bbox_inches="tight",
                        pad_inches=0.02,
                        transparent=False,
                    )

                elif fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
                    out = path.with_suffix(f".{fmt}")
                    fig.savefig(
                        out,
                        format=fmt,
                        dpi=max(dpi, 300),
                        bbox_inches="tight",
                        pad_inches=0.02,
                        transparent=False,
                    )

                else:
                    raise ValueError(
                        f"Unsupported save format '{fmt}'. "
                        "Use pdf, eps, svg, png, tif, tiff, jpg, jpeg, fig, or all."
                    )

            if saveformat == "all":
                for fmt in ("pdf", "png", "fig"):
                    _save_one(fmt)
            else:
                _save_one(saveformat)

    return fig, (ax33, ax33_r, ax76, ax76_r)
#%%
def plot_image_row_aa(
    image_groups,
    titles=None,
    cmap="inferno",
    cbar_label="Intensity",
    vmin=None,
    vmax=None,
    norm_mode="linear",
    origin="lower",
    interpolation="none",
    aspect="equal",
    hide_ticks=True,
    show_axes_frame=True,
    figure_width="full_page",
    figsize=None,
    dpi=300,
    journal_style=True,
    save=False,
    savepath="image_row_aa.pdf",
    saveformat=None,
    left=0.025,
    right=0.985,
    bottom=0.08,
    top=0.86,
    image_cbar_gap=0.001,
    panel_gap=0.018,
    cbar_width=0.008,
    title_fs=8,
    tick_fs=7,
    label_fs=8,
    single_colorbar=False,
    single_cbar_gap=0.008,
):
    """
    Plot one row of rectangular images with either independent colorbars
    or one shared colorbar.

    Parameters
    ----------
    single_colorbar : bool
        If False, each image has its own compact colorbar.
        If True, all images share the same normalization and one colorbar
        is placed to the right of the last panel.

    single_cbar_gap : float
        Gap between the last image and the shared colorbar in figure coordinates.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, LogNorm
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from pathlib import Path
    import pickle

    if not isinstance(image_groups, (list, tuple)) or len(image_groups) != 1:
        raise ValueError("image_groups must contain exactly one row: [[img1, img2, ...]].")

    images_in = image_groups[0]
    ncols = len(images_in)

    if ncols == 0:
        raise ValueError("image_groups[0] must contain at least one image.")

    images = []
    for j, img in enumerate(images_in):
        arr = np.asarray(img, dtype=float)

        if arr.ndim != 2:
            raise ValueError(f"Image {j} must be a 2D array. Got shape {arr.shape}.")

        if not np.any(np.isfinite(arr)):
            raise ValueError(f"Image {j} contains no finite value.")

        images.append(arr)

    if norm_mode not in {"linear", "log"}:
        raise ValueError("norm_mode must be either 'linear' or 'log'.")

    def _is_sequence(obj):
        return isinstance(obj, (list, tuple, np.ndarray)) and not isinstance(obj, str)

    def _expand(values, name):
        if values is None or np.isscalar(values):
            return [values] * ncols

        if _is_sequence(values):
            if len(values) == 1 and _is_sequence(values[0]):
                values = values[0]

            if len(values) != ncols:
                raise ValueError(f"{name} must have length {ncols}.")

            return list(values)

        raise TypeError(f"{name} must be None, scalar, list, or nested list.")

    vmin_list = _expand(vmin, "vmin")
    vmax_list = _expand(vmax, "vmax")

    if titles is None:
        titles_row = [None] * ncols
    else:
        if len(titles) != 1 or len(titles[0]) != ncols:
            raise ValueError("titles must have the same structure as image_groups.")
        titles_row = titles[0]

    if isinstance(figure_width, str):
        width_key = figure_width.strip().lower().replace("-", "_").replace(" ", "_")
        width_map_mm = {
            "one_column": 88.0,
            "single_column": 88.0,
            "column": 88.0,
            "intermediate": 120.0,
            "medium": 120.0,
            "full_page": 180.0,
            "full_width": 180.0,
            "page": 180.0,
        }

        if width_key not in width_map_mm:
            raise ValueError(
                "figure_width must be 'one_column', 'intermediate', "
                "'full_page', or a custom width in millimetres."
            )

        width_mm = width_map_mm[width_key]
    else:
        width_mm = float(figure_width)

    if width_mm <= 0:
        raise ValueError("figure_width must be positive.")

    width_in = width_mm / 25.4

    if figsize is None:
        height_in = width_in * 0.46
        figsize = (width_in, height_in)

    rc_params = {
        "font.size": 8,
        "axes.labelsize": label_fs,
        "axes.titlesize": title_fs,
        "xtick.labelsize": tick_fs,
        "ytick.labelsize": tick_fs,
        "legend.fontsize": 7,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    }

    def _compute_scientific_scale(values):
        finite_abs = np.abs(values[np.isfinite(values)])
        finite_abs = finite_abs[finite_abs > 0]

        if finite_abs.size == 0:
            return 0, 1.0

        exponent = int(np.floor(np.log10(np.nanmax(finite_abs))))
        scale = 10.0 ** exponent

        return exponent, scale

    if single_colorbar:
        all_finite = np.concatenate([arr[np.isfinite(arr)] for arr in images])

        vmin_common = vmin if np.isscalar(vmin) or vmin is None else np.nanmin([
            np.nanmin(arr[np.isfinite(arr)]) for arr in images
        ])
        vmax_common = vmax if np.isscalar(vmax) or vmax is None else np.nanmax([
            np.nanmax(arr[np.isfinite(arr)]) for arr in images
        ])

        if vmin_common is None:
            if norm_mode == "log":
                positive = all_finite[all_finite > 0]
                if positive.size == 0:
                    raise ValueError("No positive values available for log normalization.")
                vmin_common = np.nanmin(positive)
            else:
                vmin_common = np.nanmin(all_finite)

        if vmax_common is None:
            vmax_common = np.nanmax(all_finite)

        if vmax_common == vmin_common:
            vmax_common = vmin_common + 1.0

        vmin_list = [vmin_common] * ncols
        vmax_list = [vmax_common] * ncols

    with plt.rc_context(rc_params):
        fig = plt.figure(figsize=figsize, dpi=dpi)

        fig_w, fig_h = figsize
        panel_height = top - bottom

        if panel_height <= 0:
            raise ValueError("top must be larger than bottom.")

        image_widths = []
        for arr in images:
            ny, nx = arr.shape
            image_widths.append(panel_height * (nx / ny) * (fig_h / fig_w))

        if single_colorbar:
            total_width = (
                sum(image_widths)
                + (ncols - 1) * panel_gap
                + single_cbar_gap
                + cbar_width
            )
        else:
            total_width = (
                sum(image_widths)
                + ncols * cbar_width
                + ncols * image_cbar_gap
                + (ncols - 1) * panel_gap
            )

        available_width = right - left

        if total_width > available_width:
            raise ValueError(
                "Layout too wide. Reduce top-bottom height, cbar_width, "
                "image_cbar_gap, panel_gap, single_cbar_gap, or use a wider figure."
            )

        x0 = left + 0.5 * (available_width - total_width)

        axes = []
        cbar_axes = []
        im_last = None
        formatter_values_last = None

        for j, arr in enumerate(images):
            image_width = image_widths[j]
            ax = fig.add_axes([x0, bottom, image_width, panel_height])

            finite = arr[np.isfinite(arr)]

            vmin_j = vmin_list[j]
            vmax_j = vmax_list[j]

            if vmin_j is None:
                vmin_j = np.nanmin(finite)

            if vmax_j is None:
                vmax_j = np.nanmax(finite)

            if vmax_j == vmin_j:
                vmax_j = vmin_j + 1.0

            if norm_mode == "linear":
                img_plot = arr
                norm = Normalize(vmin=vmin_j, vmax=vmax_j)
                formatter_values = np.array([vmin_j, vmax_j], dtype=float)
            else:
                positive = finite[finite > 0]

                if positive.size == 0:
                    raise ValueError(f"Image {j} has no positive values for log normalization.")

                if vmin_j <= 0:
                    vmin_j = np.nanmin(positive)

                img_plot = np.clip(arr, vmin_j, None)
                norm = LogNorm(vmin=vmin_j, vmax=vmax_j)
                formatter_values = np.array([vmin_j, vmax_j], dtype=float)

            im = ax.imshow(
                img_plot,
                cmap=cmap,
                norm=norm,
                origin=origin,
                interpolation=interpolation,
                aspect=aspect,
            )

            im_last = im
            formatter_values_last = formatter_values

            if titles_row[j] is not None:
                ax.set_title(titles_row[j], fontsize=title_fs, pad=3)

            if hide_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.tick_params(
                    which="major",
                    direction="in",
                    length=4,
                    width=0.8,
                    labelsize=tick_fs,
                    pad=3,
                    top=True,
                    right=True,
                )

            if not show_axes_frame:
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                for spine in ax.spines.values():
                    spine.set_linewidth(0.8)

            if journal_style:
                ax.grid(False)

            axes.append(ax)

            if not single_colorbar:
                cax = fig.add_axes([
                    x0 + image_width + image_cbar_gap,
                    bottom,
                    cbar_width,
                    panel_height,
                ])

                cbar = fig.colorbar(im, cax=cax, orientation="vertical")
                cbar.set_label(cbar_label, fontsize=label_fs, labelpad=3)

                exponent, scale = _compute_scientific_scale(formatter_values)

                cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                cbar.ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda x, pos, s=scale: f"{x / s:.1f}")
                )

                cbar.ax.tick_params(
                    axis="y",
                    which="major",
                    direction="in",
                    length=3.5,
                    width=0.8,
                    labelsize=tick_fs,
                    pad=2,
                )

                cbar.ax.text(
                    1.05,
                    1.01,
                    rf"$\times 10^{{{exponent}}}$",
                    transform=cbar.ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=tick_fs,
                )

                cbar_axes.append(cax)

                x0 += image_width + image_cbar_gap + cbar_width + panel_gap
            else:
                x0 += image_width + panel_gap

        if single_colorbar:
            x_last = axes[-1].get_position().x1
            cax = fig.add_axes([
                x_last + single_cbar_gap,
                bottom,
                cbar_width,
                panel_height,
            ])

            cbar = fig.colorbar(im_last, cax=cax, orientation="vertical")
            cbar.set_label(cbar_label, fontsize=label_fs, labelpad=3)

            exponent, scale = _compute_scientific_scale(formatter_values_last)

            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            cbar.ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, pos, s=scale: f"{x / s:.1f}")
            )

            cbar.ax.tick_params(
                axis="y",
                which="major",
                direction="in",
                length=3.5,
                width=0.8,
                labelsize=tick_fs,
                pad=2,
            )

            cbar.ax.text(
                1.05,
                1.01,
                rf"$\times 10^{{{exponent}}}$",
                transform=cbar.ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=tick_fs,
            )

            cbar_axes.append(cax)

        if save:
            path = Path(savepath)

            if saveformat is None:
                if path.suffix:
                    saveformat = path.suffix.lower().lstrip(".")
                else:
                    saveformat = "pdf"
                    path = path.with_suffix(".pdf")

            saveformat = str(saveformat).lower().lstrip(".")
            path.parent.mkdir(parents=True, exist_ok=True)

            def _save_one(fmt):
                fmt = fmt.lower().lstrip(".")

                if fmt == "fig":
                    fig_path = path.with_suffix(".fig")
                    with open(fig_path, "wb") as f:
                        pickle.dump(fig, f)

                elif fmt in {"pdf", "eps", "svg"}:
                    out = path.with_suffix(f".{fmt}")
                    fig.savefig(
                        out,
                        format=fmt,
                        bbox_inches="tight",
                        pad_inches=0.02,
                        transparent=False,
                    )

                elif fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
                    out = path.with_suffix(f".{fmt}")
                    fig.savefig(
                        out,
                        format=fmt,
                        dpi=max(dpi, 300),
                        bbox_inches="tight",
                        pad_inches=0.02,
                        transparent=False,
                    )

                else:
                    raise ValueError(
                        f"Unsupported save format '{fmt}'. "
                        "Use pdf, eps, svg, png, tif, tiff, jpg, jpeg, fig, or all."
                    )

            if saveformat == "all":
                for fmt in ("pdf", "png", "fig"):
                    _save_one(fmt)
            else:
                _save_one(saveformat)

    return fig, axes, cbar_axes