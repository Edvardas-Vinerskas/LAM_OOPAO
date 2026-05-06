# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:08:07 2026

@author: mmotte
"""
import numpy as np  # Import NumPy for numerical operations


import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import LogNorm
from pathlib import Path
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

    # ---------- limits ----------
    if fmin is not None or fmax is not None:
        ax.set_xlim(left=fmin, right=fmax)
        if ax_etf is not None:
            ax_etf.set_xlim(left=fmin, right=fmax)

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
def plot_psf_aa(
    psf,
    wvl,
    telescope_diameter,
    sampling,
    vmin=1e-6,
    vmax=1,
    cmap="inferno",
    xlabel=r"[arcsec]",
    ylabel=r"[arcsec]",
    cbar_label="Normalized intensity",
    nx = None,
    title=None,
    figsize=(3.35, 3.1),   # ~ largeur colonne A&A en pouces
    dpi=1200,
    origin="lower",
    save=False,
    savepath="psf.pdf",
    saveformat=None,
):
    

    # Style adapté à A&A
    plt.rcParams.update({
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
    })

    # Échelle en arcsec
    pix_scale = wvl / telescope_diameter / sampling
    
    rad2arcsec = 180/(2*np.pi)*3600
    
    psf_norm = psf/psf.sum()
    if nx is None:
        nx = psf.shape[0]
    else:
        ctr = psf_norm.shape[0]//2
        psf_norm = psf_norm[ctr-nx//2:ctr+nx//2,ctr-nx//2:ctr+nx//2]
    axis = np.linspace(-nx // 2, nx // 2, nx) * pix_scale * rad2arcsec
    # Normalisation
    maxi = np.max(psf_norm)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Figure
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        psf_norm / maxi,
        norm=norm,
        cmap=cmap,
        extent=[axis[0], axis[-1], axis[0], axis[-1]],
        origin=origin,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()

    # Sauvegarde optionnelle
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
):
    """
    Plot N PSDs in an A&A-compatible style.

    Parameters
    ----------
    psd_list : list of tuple
        List of PSDs to plot. Each element must be a tuple (freq, psd).
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
        constrained_layout=True
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

    lw = 1.5 if n_psd <= 4 else 1.1 if n_psd <= 10 else 0.9

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
            raise ValueError("Each PSD must contain at least two valid points for the requested axis scale.")

        idx = np.argsort(freq)
        freq = freq[idx]
        y = y[idx]

        if normalised:
            ymax_local = np.nanmax(y)
            if ymax_local > 0:
                y = y / ymax_local

        return freq, y

    for i, ((freq, psd), label) in enumerate(zip(psd_list, labels)):
        f_plot, y_plot = _prepare_psd(freq, psd)

        ax.plot(
            f_plot,
            y_plot,
            color=colors_cycle[i % len(colors_cycle)],
            lw=lw,
            ls=linestyles[i % len(linestyles)],
            label=label,
            zorder=3,
            solid_capstyle="round",
            dash_capstyle="butt",
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
