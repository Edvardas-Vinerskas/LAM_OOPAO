# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:08:07 2026

@author: mmotte
"""
import numpy as np  # Import NumPy for numerical operations


import matplotlib.pyplot as plt


from pathlib import Path
#%%
def plot_psd_aa( 
    f1,
    psd1,
    f2=None,
    psd2=None,
    label1="open loop",
    label2="closed loop",
    method=np.nansum,
    f_unit="Hz",
    psd_unit=r"nm$^2$/Hz",
    fmin=None,
    fmax=None,
    normalised=False,
    show_legend=True,
    one_column=True,
    dpi=300,
    save=False,
    savepath="mean_psd_aa.pdf",
    saveformat=None,
    journal_style=True,   # True: A&A final style ; False: working style with light grid
):
    # ---------- input ----------
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

    # ---------- figure size ----------
    width_in = 88 / 25.4 if one_column else 180 / 25.4
    height_in = width_in * 0.72
    fig, ax = plt.subplots(figsize=(width_in, height_in), constrained_layout=True, dpi=dpi)

    # ---------- curves ----------
    # Strong distinction by BOTH color and linestyle
    col1 = "black"
    col2 = "#355C9A"   # muted dark blue, cleaner than navy

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

    # ---------- limits ----------
    if fmin is not None or fmax is not None:
        ax.set_xlim(left=fmin, right=fmax)

    # ---------- labels ----------
    ax.set_xlabel(f"Frequency [{f_unit}]", fontsize=10)
    if normalised:
        ax.set_ylabel("Normalised PSD", fontsize=10)
    else:
        ax.set_ylabel(f"PSD [{psd_unit}]", fontsize=10)

    # ---------- ticks / frame ----------
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.tick_params(which="major", direction="in", length=6, width=1.0, labelsize=9, pad=6)
    ax.tick_params(which="minor", direction="in", length=3.5, width=0.8)

    # ---------- grid ----------
    if journal_style:
        # final A&A version
        ax.grid(False)
    else:
        # working / discussion version
        ax.grid(True, which="major", color="0.88", lw=0.6)
        ax.grid(True, which="minor", color="0.93", lw=0.4)

    # ---------- legend ----------
    if show_legend and has_second_curve:
        ax.legend(
            frameon=False,
            fontsize=9,
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

    return fig, ax