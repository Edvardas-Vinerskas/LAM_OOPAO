# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:08:07 2026

@author: mmotte
"""
import numpy as np  # Import NumPy for numerical operations


import matplotlib.pyplot as plt


from pathlib import Path
#%%
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:08:07 2026

@author: mmotte
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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

        if saveformat is None:
            if path.suffix:
                saveformat = path.suffix.lower().lstrip(".")
            else:
                saveformat = "pdf"
                path = path.with_suffix(".pdf")
    
        # create parent directory automatically if neede
        path.parent.mkdir(parents=True, exist_ok=True)

        if saveformat in {"pdf", "eps", "svg"}:
            fig.savefig(
                path,
                format=saveformat,
                bbox_inches="tight",
                pad_inches=0.02,
                transparent=False,
            )
        elif saveformat in {"png", "tif", "tiff", "jpg", "jpeg"}:
            fig.savefig(
                path,
                format=saveformat,
                dpi=max(dpi, 300),
                bbox_inches="tight",
                pad_inches=0.02,
                transparent=False,
            )
        else:
            raise ValueError(
                f"Unsupported save format '{saveformat}'. "
                "Use pdf, eps, png, tiff, jpg, jpeg, or svg."
            )

    return fig, (ax_etf, ax) if ax_etf is not None else ax