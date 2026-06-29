# -*- coding: utf-8 -*-

import numpy as np

from OOPAO.Source import Source

import time



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

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.patches import Circle
from numpy.fft import fft2, fftshift
from maoppy.utils import circavg

def plot_simulated_psf_analysis_from_tele(
    tele,
    psf_obj=None,
    papyrus=None,
    nx=None,
    wvl=None,
    WVL_0=1550e-9,
    rad2arcsec=180 / np.pi * 3600,
    hfov=3,
    lw=2,
    log_vmin=1e-4,
    log_vmax=1,
    sym_linthresh=1e-3,
    save_path=None,
):
    """
    Trace les diagnostics PSF à partir d'un objet OZITele après `tele * psf`.

    Version auto-contenue : fonctionne si sampling et wvl sont des scalaires
    ou des arrays.
    """

    def compute_otf_local(img):
        return fftshift(fft2(fftshift(img)))

    def get_from_tele(name, default=None, required=True):
        if hasattr(tele, "psf_analysis_results") and name in tele.psf_analysis_results:
            return tele.psf_analysis_results[name]

        if hasattr(tele, name):
            return getattr(tele, name)

        if required:
            raise AttributeError(
                f"Impossible de trouver '{name}' dans tele.psf_analysis_results "
                f"ou dans les attributs de tele. Avez-vous bien exécuté `cl_tele * psf` ?"
            )

        return default

    def as_1d_array(x):
        if x is None:
            return None

        x = np.asarray(x)

        if x.ndim == 0:
            return x.reshape(1)

        return x.ravel()

    def first_scalar(x, default=np.nan):
        if x is None:
            return default

        try:
            arr = np.asarray(x).ravel()
            if arr.size == 0:
                return default
            return float(arr[0])
        except Exception:
            return default

    def format_scalar_or_array(x, precision=4, unit=""):
        if x is None:
            return "None"

        arr = np.asarray(x)

        if arr.ndim == 0:
            return f"{float(arr):.{precision}f}{unit}"

        arr = arr.ravel()

        if arr.size == 1:
            return f"{float(arr[0]):.{precision}f}{unit}"

        return (
            f"[{np.nanmin(arr):.{precision}f}, "
            f"{np.nanmax(arr):.{precision}f}]{unit} "
            f"({arr.size} values)"
        )

    def get_sampling():
        sampling_local = get_from_tele(
            "sampling",
            default=None,
            required=False,
        )

        if sampling_local is None:
            sampling_local = get_from_tele(
                "psf_sampling",
                default=None,
                required=False,
            )

        if sampling_local is None and psf_obj is not None:
            if hasattr(psf_obj, "sampling"):
                sampling_local = psf_obj.sampling
            elif hasattr(psf_obj, "psf_sampling"):
                sampling_local = psf_obj.psf_sampling

        if sampling_local is None:
            raise AttributeError(
                "Impossible de trouver le sampling. "
                "Vérifiez que `cl_tele * psf` a bien été exécuté, "
                "ou passez explicitement `psf_obj=psf`."
            )

        return sampling_local

    def get_wavelength(default=None):
        if wvl is not None:
            return wvl

        if hasattr(tele, "wvl_psf"):
            return np.asarray(tele.wvl_psf)

        if hasattr(tele, "wavelength"):
            return np.asarray(tele.wavelength)

        if hasattr(tele, "wvl"):
            return np.asarray(tele.wvl)

        if psf_obj is not None:
            if hasattr(psf_obj, "wvl_psf"):
                return np.asarray(psf_obj.wvl_psf)
            if hasattr(psf_obj, "wvl"):
                return np.asarray(psf_obj.wvl)

        return default

    # ------------------------------------------------------------------
    # 1. Récupération des données
    # ------------------------------------------------------------------
    out = get_from_tele("out")
    img_norm = get_from_tele("psf")
    psf_fit = get_from_tele("psf_fit")

    sampling = get_sampling()
    sampling_arr = as_1d_array(sampling)
    sampling_ref = first_scalar(sampling_arr)

    wvl_used = get_wavelength(default=WVL_0)
    wvl_arr = as_1d_array(wvl_used)
    wvl_ref = first_scalar(wvl_arr, default=WVL_0)

    otf_diff = get_from_tele(
        "otf_diff",
        default=None,
        required=False,
    )

    SR_otf = get_from_tele(
        "SR_otf",
        default=np.nan,
        required=False,
    )

    seeing_550 = get_from_tele(
        "seeing_550",
        default=np.nan,
        required=False,
    )

    if nx is None:
        nx = img_norm.shape[0]

    # ------------------------------------------------------------------
    # 2. PSF de diffraction
    # ------------------------------------------------------------------
    psf_diff = None
    psfmodel = None

    if hasattr(out, "psfmodel"):
        psfmodel = out.psfmodel
    elif psf_obj is not None and hasattr(psf_obj, "psfmodel"):
        psfmodel = psf_obj.psfmodel

    if psfmodel is not None and hasattr(psfmodel, "psfDiffraction"):
        psf_diff = psfmodel.psfDiffraction

    if psf_diff is None and hasattr(tele, "simulated_psf_diff"):
        if tele.simulated_psf_diff.ndim == 4:
            psf_diff = tele.simulated_psf_diff.mean(axis=(0, 1))
        elif tele.simulated_psf_diff.ndim == 3:
            psf_diff = tele.simulated_psf_diff.mean(axis=0)
        elif tele.simulated_psf_diff.ndim == 2:
            psf_diff = tele.simulated_psf_diff

    if psf_diff is None:
        psf_diff = np.zeros_like(img_norm)
        psf_diff[nx // 2, nx // 2] = 1.0

    # ------------------------------------------------------------------
    # 3. MTF / OTF
    # ------------------------------------------------------------------
    mtf_avg = circavg(
        np.abs(compute_otf_local(img_norm)),
        center=(nx // 2, nx // 2),
    )

    mtf_fit_avg = circavg(
        np.abs(compute_otf_local(psf_fit)),
        center=(nx // 2, nx // 2),
    )

    if otf_diff is not None:
        mtf_diff_avg = circavg(
            np.abs(otf_diff),
            center=(nx // 2, nx // 2),
        )
        otf_diff_for_sr = otf_diff
    else:
        otf_diff_for_sr = compute_otf_local(psf_diff)
        mtf_diff_avg = circavg(
            np.abs(otf_diff_for_sr),
            center=(nx // 2, nx // 2),
        )

    denom = np.abs(otf_diff_for_sr).sum()

    if denom > 0:
        sr_otf_sum = np.abs(compute_otf_local(psf_fit)).sum() / denom * 100
    else:
        sr_otf_sum = np.nan

    # ------------------------------------------------------------------
    # 4. Axes
    # ------------------------------------------------------------------
    use_arcsec = papyrus is not None and hasattr(papyrus, "pixel_mas")

    if use_arcsec:
        axis = np.linspace(-nx // 2, nx // 2, nx) * papyrus.pixel_mas * 1e-3
        x_label = "[arcsec]"
    else:
        axis = np.arange(nx) - nx // 2
        x_label = "[pix]"

    dxdy = getattr(out, "dxdy", np.array([0.0, 0.0]))
    dxdy = np.asarray(dxdy).ravel()

    if dxdy.size < 2:
        dxdy = np.array([0.0, 0.0])

    if use_arcsec:
        dx, dy = dxdy[:2] * papyrus.pixel_mas * 1e-3
    else:
        dx, dy = dxdy[:2]

    cxcy = (nx // 2 + dxdy[1], nx // 2 + dxdy[0])

    maxi_data = np.nanmax(img_norm)
    maxi_diff = np.nanmax(psf_diff)

    if not np.isfinite(maxi_data) or maxi_data <= 0:
        maxi_data = 1.0

    if not np.isfinite(maxi_diff) or maxi_diff <= 0:
        maxi_diff = 1.0

    # ------------------------------------------------------------------
    # 5. Figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    def set_img_panel(ax, tab, title, cmap="Spectral_r", norm=None):
        if norm is None:
            norm = LogNorm(vmin=log_vmin, vmax=log_vmax)

        im = ax.imshow(
            tab / maxi_data,
            norm=norm,
            cmap=cmap,
            extent=[axis[0], axis[-1], axis[0], axis[-1]],
            origin="lower",
        )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if (
            use_arcsec
            and hasattr(papyrus, "D")
            and hasattr(papyrus, "Nact")
            and np.isfinite(wvl_ref)
        ):
            corr_radius = rad2arcsec * (wvl_ref / papyrus.D * papyrus.Nact / 2)

            corr_zone = Circle(
                [dx, -dy],
                corr_radius,
                fc="none",
                ec="k",
                ls=":",
            )

            ax.add_artist(corr_zone)

        ax.set_title(title)
        ax.set_xlabel(x_label)

        if use_arcsec:
            ax.set_xlim(-hfov + dx, hfov + dx)
            ax.set_ylim(-hfov - dy, hfov - dy)

    # Images
    set_img_panel(axes[0], img_norm, "data")
    set_img_panel(axes[1], psf_fit, "fit")

    set_img_panel(
        axes[2],
        psf_fit - img_norm,
        "fit - data",
        cmap="RdBu",
        norm=SymLogNorm(
            linthresh=sym_linthresh,
            vmin=-1,
            vmax=1,
        ),
    )

    # Profils PSF
    axes[3].set_title("PSF")

    axes[3].semilogy(
        circavg(psf_diff / maxi_diff, center=(nx // 2, nx // 2)),
        lw=lw,
        label="diffrac.",
        c="grey",
    )

    axes[3].semilogy(
        circavg(img_norm / maxi_diff, center=cxcy),
        lw=lw,
        label="data",
    )

    axes[3].semilogy(
        circavg(psf_fit / maxi_diff, center=cxcy),
        lw=lw,
        label="fit",
    )

    if hasattr(out, "flux_bck"):
        flux_bck = np.asarray(out.flux_bck).ravel()

        if flux_bck.size >= 2 and flux_bck[0] != 0:
            axes[3].axhline(
                flux_bck[1] / flux_bck[0],
                c="C1",
                ls="--",
                label="bck fit",
            )

    # Coupure AO pour sampling scalaire ou vectoriel
    if papyrus is not None and hasattr(papyrus, "Nact"):
        if sampling_arr.size == 1:
            axes[3].axvline(
                papyrus.Nact / 2 * float(sampling_arr[0]),
                c="k",
                ls=":",
                label="AO",
            )
        else:
            for k, samp_k in enumerate(sampling_arr):
                axes[3].axvline(
                    papyrus.Nact / 2 * float(samp_k),
                    c="k",
                    ls=":",
                    alpha=0.25,
                    label="AO" if k == 0 else None,
                )

    axes[3].grid()
    axes[3].set_xlim(0, nx // 2)
    axes[3].set_ylim(1e-5, 1)
    axes[3].set_xlabel("Position [pix]")
    axes[3].legend()

    # MTF / OTF
    axes[4].set_title("OTF")

    axes[4].loglog(
        mtf_diff_avg,
        lw=lw,
        label="diffrac.",
        c="grey",
    )

    axes[4].loglog(
        mtf_avg,
        lw=lw,
        label="data",
    )

    axes[4].loglog(
        mtf_fit_avg,
        lw=lw,
        label="fit",
    )

    axes[4].set_xlabel("Frequency [1/pix]")
    axes[4].set_ylim(1e-3, 1.5)
    axes[4].set_xlim(right=nx // 2)
    axes[4].grid()
    axes[4].legend()

    # Texte
    axes[5].axis("off")

    SR_otf_scalar = np.nanmean(SR_otf)
    seeing_550_scalar = first_scalar(seeing_550, default=np.nan)

    sampling_txt = format_scalar_or_array(
        sampling_arr.mean(),
        precision=4,
        unit=" pix/?D",
    )

    wvl_txt = format_scalar_or_array(
        wvl_arr.mean() * 1e9,
        precision=1,
        unit=" nm",
    )


    text = (
        f"Sampling : {sampling_txt}\n"
        f"Wavelength : {wvl_txt}\n\n"
    )

    if np.isfinite(SR_otf_scalar):
        text += f"Strehl OTF : {100 * SR_otf_scalar:.1f} %\n"
    else:
        text += "Strehl OTF stored : nan\n"

    if np.isfinite(sr_otf_sum):
        text += f"Strehl OTF sum : {sr_otf_sum:.1f} %\n\n"
    else:
        text += "Strehl OTF sum : nan\n\n"

    if np.isfinite(seeing_550_scalar):
        text += f"Seeing : {seeing_550_scalar:.2f} \" @ 550 nm\n"

    axes[5].text(
        -0.05,
        0.1,
        text,
        size=14,
        va="bottom",
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            format=save_path.split(".")[-1],
            bbox_inches="tight",
            pad_inches=0.1,
        )

    diagnostics = {
        "img_norm": img_norm,
        "psf_fit": psf_fit,
        "psf_diff": psf_diff,
        "mtf_avg": mtf_avg,
        "mtf_fit_avg": mtf_fit_avg,
        "mtf_diff_avg": mtf_diff_avg,
        "sampling": sampling,
        "sampling_arr": sampling_arr,
        "wvl": wvl_used,
        "wvl_arr": wvl_arr,
        "SR_otf": SR_otf,
        "SR_otf_sum_percent": sr_otf_sum,
        "seeing_550": seeing_550,
        "out": out,
    }

    return fig, axes, diagnostics