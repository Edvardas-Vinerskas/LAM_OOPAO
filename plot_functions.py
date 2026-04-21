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

def plot_psd_aa(
    f1,
    psd1,
    f2=None,
    psd2=None,
    label1="Closed loop",
    label2="Open loop",
    method=np.nansum,
    f_unit="Hz",
    psd_unit=r"nm$^2$/Hz",
    fmin=None,
    fmax=None,
    normalised=False,
    compute_etf=True,
    show_legend=True,
    one_column=True,
    dpi=300,
    save=False,
    savepath="mean_psd_aa.pdf",
    saveformat=None,
    journal_style=True,   # True: A&A final style ; False: working style with light grid
):
    # ---------- input ----------
    label_fs = 9
    tick_fs = 8
    legend_fs = 8
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
            sharex=True,
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

    ax.loglog(
        f1, m1,
        color=col1,
        lw=1.6,
        ls="-",
        label=label1,
        zorder=3,
        solid_capstyle="round",
    )

    if has_second_curve:
        ax.loglog(
            f2, m2,
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

        valid_etf = np.isfinite(etf) & (etf > 0)
        f_ratio_plot = f_ratio[valid_etf]
        etf_plot = etf[valid_etf]
        
        if f_ratio_plot.size < 2:
            raise ValueError("Not enough strictly positive ETF points to display in log-log scale.")
        
        ax_etf.loglog(
            f_ratio_plot, etf_plot,
            color="black",
            lw=1.2,
            zorder=3,
        )
        
        # reference line ETF = 1 over the displayed x-range
        ax_etf.plot(
            f_ratio_plot,
            np.ones_like(f_ratio_plot),
            color="0.4",
            lw=0.9,
            ls="--",
            zorder=2,
        )

        ax_etf.set_ylabel("ETF", fontsize=label_fs)
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

    # ---------- limits ----------
    if fmin is not None or fmax is not None:
        ax.set_xlim(left=fmin, right=fmax)

    # ---------- labels ----------
    ax.set_xlabel(f"Frequency [{f_unit}]", fontsize=label_fs)
    if normalised:
        ax.set_ylabel("Normalised PSD", fontsize=label_fs)
    else:
        ax.set_ylabel(f"PSD [{psd_unit}]", fontsize=label_fs)

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
            loc="lower left",
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
            loc="lower right",
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
            loc="upper left",
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