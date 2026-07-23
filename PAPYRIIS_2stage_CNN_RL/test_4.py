"""
ao_metrics.py
=============
Post-processing and visualisation for two-stage AO simulation results.

Produces:
  • RL training-loss curves  (dynamics + policy)
  • Strehl-ratio time series with running-average overlay
  • KL-coefficient variance spectra
  • Per-mode temporal PSDs  (raw + cumulative)
  • Per-mode residual time series  (optional)

Usage
-----
  python ao_metrics.py

All tunable parameters live in the Config dataclass at the top of the file.
Nothing executes on import; the script is safe to use as a module.
"""

#TODO this is OOPAO metrics plotting as rewritten by claude

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")          # set backend before pyplot is imported
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION  — every tunable parameter in one place
# ══════════════════════════════════════════════════════════════════════════════



@dataclass
class Config:
    """Single source of truth for all run parameters."""

    # ── paths ──────────────────────────────────────────────────────────────
    base_dir:       Path = Path("PAPYRIIS_2stage_CNN_RL")
    run_date:       str  = "~2026-06-01"
    run_name_rl:    str  = "PAPYRIIS_arcturus_noise"
    run_name_ideal: str  = "PAPYRIIS_arcturus_nonoise"
    atm_filename:   str  = ("generated_atm_2nd_stage/"
                             "atm_OPDs_2nd_r0_0.050_V0_4.121.npz")
    atm_npz_key:    str  = "atm_OPDs_2nd"

    # ── AO system constants ────────────────────────────────────────────────
    wavelength:  float = 1.6e-6   # science wavelength [m]
    n_frames:    int   = 30_000   # trailing frames to load and analyse
    dm2_grid:    int   = 90       # 2nd-stage DM pupil grid size [pixels]

    # ── loop frequencies [Hz] ──────────────────────────────────────────────
    freq_rl:    int = 400
    freq_int:   int = 400
    freq_ideal: int = 400
    freq_atm:   int = 400

    # ── cosmetic (label only) ──────────────────────────────────────────────
    cl_gain_pyr: float = 0.3

    # ── KL mode indices to display ─────────────────────────────────────────
    plot_modes: Tuple[int, ...] = (1, 10, 20, 30, 40)

    # ── frequency axis limits ──────────────────────────────────────────────
    stage1_freq_lim: float = 250.0   # dashed 1st-stage PSD cut-off [Hz]
    freq_lim:        float = 900.0

    # ── Welch PSD ──────────────────────────────────────────────────────────
    welch_nperseg: int = 256

    # ── Strehl running-average window [frames] ─────────────────────────────
    sr_window: int = 500

    # ── Strehl threshold for stable-segment PSD (0 = use full series) ─────
    sr_threshold: float = 0.0

    # ── controller/figure switches ─────────────────────────────────────────
    use_rl:         bool = True
    use_integrator: bool = True
    use_ideal:      bool = True
    overlay_stage1: bool = True    # dashed 1st-stage line on 2nd-stage figs
    plot_timeseries: bool = False
    plot_psd:        bool = True
    plot_cumulative: bool = True

    # ── derived paths (read-only) ──────────────────────────────────────────
    @property
    def run_dir(self) -> Path:
        return self.base_dir / self.run_date

    @property
    def rl_dir(self) -> Path:
        return self.run_dir / self.run_name_rl

    @property
    def ideal_dir(self) -> Path:
        return self.run_dir / self.run_name_ideal


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA CONTAINERS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StageData:
    """Processed data for one controller × one AO stage."""
    strehl:    np.ndarray    # shape (T,)
    modes:     np.ndarray    # shape (T, n_modes)
    frequency: float         # loop frequency [Hz]
    label:     str
    color:     str
    linestyle: str = "-"

    @property
    def time_axis(self) -> np.ndarray:
        return np.arange(len(self.strehl)) / self.frequency


@dataclass
class ControllerResult:
    """Stage-1 and stage-2 data for a single controller."""
    name:   str
    stage1: Optional[StageData] = None
    stage2: Optional[StageData] = None


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SIGNAL-PROCESSING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def strehl_from_opd(opds: np.ndarray,
                    pupil_mask: np.ndarray,
                    wavelength: float) -> np.ndarray:
    """
    Maréchal approximation:  SR = exp(−σ²_φ),  φ = 2π · OPD / λ  [rad].

    Parameters
    ----------
    opds       : (T, H, W) or (T, N_pupil) — OPD maps [m]
    pupil_mask : boolean mask matching the spatial dims of opds
    wavelength : science wavelength [m]
    """
    phases = (2.0 * np.pi / wavelength) * opds[:, pupil_mask]
    return np.exp(-phases.var(axis=1))


def running_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Causal running mean; edges padded with the global mean so the
    output length equals the input length.
    """
    kernel = np.ones(window) / window
    pad_l  = window // 2
    pad_r  = window - pad_l - 1
    padded = np.pad(arr, (pad_l, pad_r), mode="constant",
                    constant_values=arr.mean())
    return np.convolve(padded, kernel, mode="valid")


def welch_psd(ts: np.ndarray,
              fs: float,
              nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Return (frequencies [Hz], one-sided PSD) via Welch's method."""
    return signal.welch(ts, fs=fs, window="hann",
                        nperseg=nperseg, scaling="density")


def find_stable_segment(strehl: np.ndarray,
                        threshold: float) -> Tuple[int, int]:
    """
    Return (start, end) of the longest contiguous run with strehl ≥ threshold.

    If threshold is 0 (or no frames pass), returns (0, len(strehl)) so the
    caller always receives a valid slice without special-casing.
    """
    if threshold <= 0.0:
        return 0, len(strehl)

    above   = strehl >= threshold
    changes = np.diff(above.astype(np.int8), prepend=0, append=0)
    starts  = np.where(changes == 1)[0]
    ends    = np.where(changes == -1)[0]

    if len(starts) == 0:
        warnings.warn("No frames above Strehl threshold; using full series.")
        return 0, len(strehl)

    best = int(np.argmax(ends - starts))
    return int(starts[best]), int(ends[best])


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_tail(path: Path, n: int) -> Dict[str, np.ndarray]:
    """Load an .npz and return the last *n* rows of every array."""
    raw = np.load(path)
    return {
        k: (raw[k][-n:] if raw[k].ndim >= 1 else raw[k])
        for k in raw.files
    }



def load_calibration(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (C2M_1st, C2M_2nd) — pseudo-inverse command-to-mode matrices.
    C2M · cmd_vector → modal coefficients.
    """
    M2C_1st = np.load(cfg.base_dir / "M2C_1rst.npy")
    M2C_2nd = np.load(cfg.base_dir / "M2C_KL.npy")
    return np.linalg.pinv(M2C_1st), np.linalg.pinv(M2C_2nd)


def load_pupil_and_projector(
        cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (pupil_mask_2nd, projector_kl_2nd).

    projector_kl_2nd : (n_modes, n_pupil_pix) — projects pupil-plane OPD
                       onto KL modes via  modes = OPD_flat @ projector.T
    """
    raw        = np.load(cfg.rl_dir / "results_2nd_stage.npz")
    pupil_mask = raw["telescope_pupil"].astype(bool)   # (H, W)
    proj_raw   = raw["projector_kl_2nd"]               # (n_modes, H*W)
    n_modes    = proj_raw.shape[0]
    projector  = proj_raw.reshape(n_modes,
                                   cfg.dm2_grid,
                                   cfg.dm2_grid)[:, pupil_mask]
    return pupil_mask, projector


def load_atmosphere_modes(cfg: Config,
                           pupil_mask: np.ndarray,
                           projector_kl: np.ndarray) -> np.ndarray:
    """
    Return open-loop atmospheric KL-mode time series, shape (n_frames, n_modes).
    """
    opds = np.load(cfg.base_dir / cfg.atm_filename
                   )[cfg.atm_npz_key][-cfg.n_frames:]        # (T, H, W)
    return opds[:, pupil_mask] @ projector_kl.T              # (T, n_modes)


def _build_stage(npz: Dict[str, np.ndarray],
                 C2M_2nd: np.ndarray,
                 pupil_mask: np.ndarray,
                 cfg: Config,
                 frequency: float,
                 label: str,
                 color: str,
                 linestyle: str = "-",
                 strehl_key: Optional[str] = "all_2nd_stage_strehl",
                 opd_key: Optional[str]    = None,
                 cmd_key: str              = "all_reconstructed_cmd",
                 ) -> StageData:
    """
    Construct a StageData from a pre-loaded .npz dict.

    Strehl is read directly if *strehl_key* is present, otherwise computed
    from the OPD array named by *opd_key* via the Maréchal approximation.
    """
    if strehl_key and strehl_key in npz:
        strehl = npz[strehl_key]
    elif opd_key and opd_key in npz:
        strehl = strehl_from_opd(npz[opd_key], pupil_mask, cfg.wavelength)
    else:
        raise KeyError(
            f"Cannot find a Strehl source. "
            f"Tried keys '{strehl_key}' and '{opd_key}'. "
            f"Available: {list(npz)}"
        )

    modes = npz[cmd_key] @ C2M_2nd.T      # (T, n_modes)
    return StageData(strehl=strehl, modes=modes, frequency=frequency,
                     label=label, color=color, linestyle=linestyle)


def load_controllers(cfg: Config,
                     C2M_2nd: np.ndarray,
                     pupil_mask: np.ndarray) -> Dict[str, ControllerResult]:
    """
    Load all controller results and return a dict keyed by short name.

    Note: the 1st-stage (CL1OL2) recording is shared across all controllers
    — it is loaded once here and referenced by each ControllerResult.
    """

    # ── shared 1st-stage (same file for RL, integrator, and ideal) ─────────
    s1_npz    = _load_tail(cfg.rl_dir / "results_2nd_stage_CL1OL2.npz",
                            cfg.n_frames)
    stage1_shared = _build_stage(
        s1_npz, C2M_2nd, pupil_mask, cfg,
        frequency  = cfg.freq_rl,
        label      = "1st stage (integrator)",
        color      = "indianred",
        linestyle  = "--",
        strehl_key = None,
        opd_key    = "residual_opds_2nd",
    )

    controllers: Dict[str, ControllerResult] = {}

    if cfg.use_rl:
        d = _load_tail(cfg.rl_dir / "results_2nd_stage.npz", cfg.n_frames)
        controllers["rl"] = ControllerResult(
            name   = "RL",
            stage1 = stage1_shared,
            stage2 = _build_stage(d, C2M_2nd, pupil_mask, cfg,
                                   frequency  = cfg.freq_rl,
                                   label      = "RL",
                                   color      = "red",
                                   strehl_key = "all_2nd_stage_strehl"),
        )

    if cfg.use_integrator:
        d = _load_tail(cfg.rl_dir / "results_2nd_stage_int.npz", cfg.n_frames)
        controllers["int"] = ControllerResult(
            name   = "Integrator",
            stage1 = stage1_shared,
            stage2 = _build_stage(d, C2M_2nd, pupil_mask, cfg,
                                   frequency  = cfg.freq_int,
                                   label      = "Integrator",
                                   color      = "blue",
                                   strehl_key = "all_2nd_stage_strehl"),
        )

    if cfg.use_ideal:
        path = cfg.ideal_dir / "results_2nd_stage_r0_0.050_V0_4.121.npz"
        d    = _load_tail(path, cfg.n_frames)
        controllers["ideal"] = ControllerResult(
            name   = "No-noise RL",
            stage1 = stage1_shared,
            stage2 = _build_stage(d, C2M_2nd, pupil_mask, cfg,
                                   frequency  = cfg.freq_ideal,
                                   label      = "No-noise RL",
                                   color      = "darkgreen",
                                   strehl_key = None,
                                   opd_key    = "all_src_opd"),
        )

    return controllers


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PLOT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _style(ax: plt.Axes, *,
           title: str = "",
           xlabel: str = "",
           ylabel: str = "",
           log_x: bool = False,
           log_y: bool = False) -> None:
    """Apply consistent styling to an Axes object."""
    if title:  ax.set_title(title,  fontsize=14)
    if xlabel: ax.set_xlabel(xlabel, fontsize=12)
    if ylabel: ax.set_ylabel(ylabel, fontsize=12)
    if log_x:  ax.set_xscale("log")
    if log_y:  ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.4)
    ax.minorticks_on()
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=10)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FIGURE-LEVEL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_losses(cfg: Config) -> None:
    """Dynamics and policy loss curves from RL warm-up."""
    try:
        dyn_loss = np.load(cfg.rl_dir / "dynamics_loss.npy")
        pol_loss = np.load(cfg.rl_dir / "policy_loss.npy")
    except FileNotFoundError as exc:
        warnings.warn(f"Cannot load loss files: {exc}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("RL training losses", fontsize=14)
    for ax, data, name in zip(
            [ax1, ax2],
            [dyn_loss, pol_loss],
            ["Dynamics loss", "Policy loss"]):
        ax.plot(data)
        _style(ax, title=name, xlabel="Gradient step",
               ylabel="Loss", log_x=True, log_y=True)
    fig.tight_layout()


def plot_strehl(controllers: Dict[str, ControllerResult],
                cfg: Config) -> None:
    """
    Strehl ratio time series + running-average overlay for every controller.

    Raw instantaneous Strehl is shown faintly; the running mean is bold.
    The legend entry includes the time-averaged Strehl ratio.
    """
    fig, ax = plt.subplots(figsize=(11, 4))

    for ctrl in controllers.values():
        sd = ctrl.stage2
        if sd is None:
            continue
        avg = running_mean(sd.strehl, cfg.sr_window)
        ax.plot(sd.time_axis, sd.strehl,
                color=sd.color, alpha=0.2, linewidth=0.5)
        ax.plot(sd.time_axis, avg,
                color=sd.color,
                label=f"{sd.label}  (mean SR = {sd.strehl.mean():.3f})")

    _style(ax,
           title="Strehl ratio — 2nd stage",
           xlabel="Time (s)",
           ylabel="Strehl ratio")
    fig.tight_layout()


def plot_kl_variance(controllers: Dict[str, ControllerResult],
                     modes_atm: np.ndarray,
                     cfg: Config) -> None:
    """
    Log–log KL-mode variance spectrum for each controller.

    The open-loop atmospheric variance is shown as a black dashed reference.
    When cfg.overlay_stage1 is True, the shared 1st-stage variance is also
    overlaid (dashed) on the same axes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.var(modes_atm, axis=0), "k--", label="Atmosphere (open loop)")

    for ctrl in controllers.values():
        if cfg.overlay_stage1 and ctrl.stage1 is not None:
            s1 = ctrl.stage1
            ax.plot(np.var(s1.modes, axis=0),
                    color=s1.color, linestyle=s1.linestyle,
                    label=f"1st stage — {ctrl.name}")
        if ctrl.stage2 is not None:
            s2 = ctrl.stage2
            ax.plot(np.var(s2.modes, axis=0),
                    color=s2.color, linestyle=s2.linestyle,
                    label=f"2nd stage — {s2.label}")

    _style(ax,
           title="KL-coefficient variance spectrum",
           xlabel="KL mode index",
           ylabel="Variance  [arb. units]",
           log_x=True, log_y=True)
    fig.tight_layout()


def _psd_single_mode(controllers: Dict[str, ControllerResult],
                     modes_atm: np.ndarray,
                     mode_idx: int,
                     cfg: Config,
                     cumulative: bool = False) -> None:
    """
    Draw one temporal (or cumulative) PSD figure for a single KL mode.

    For each controller:
      • 2nd-stage PSD is drawn with its primary colour.
      • 1st-stage PSD is drawn dashed, clipped to cfg.stage1_freq_lim.

    The longest contiguous segment where SR ≥ cfg.sr_threshold is used for
    the PSD; if the threshold is 0 the full series is used.

    Bad data (e.g. a mode with zero variance) is caught and skipped.
    """
    kind  = "Cumulative PSD" if cumulative else "Temporal PSD"
    fig, ax = plt.subplots(figsize=(8, 5))

    # ── atmosphere reference ────────────────────────────────────────────────
    fa, pa = welch_psd(modes_atm[:, mode_idx], cfg.freq_atm, cfg.welch_nperseg)
    ax.plot(fa, np.cumsum(pa) if cumulative else pa,
            "k-", label=f"Atmosphere — mode {mode_idx}")

    for ctrl in controllers.values():
        # ── 2nd-stage ───────────────────────────────────────────────────────
        if ctrl.stage2 is not None:
            s2    = ctrl.stage2
            i0, i1 = find_stable_segment(s2.strehl, cfg.sr_threshold)
            try:
                f2, p2 = welch_psd(s2.modes[i0:i1, mode_idx],
                                    s2.frequency, cfg.welch_nperseg)
                ax.plot(f2, np.cumsum(p2) if cumulative else p2,
                        color=s2.color, linestyle=s2.linestyle,
                        label=f"2nd — {s2.label}")
            except Exception as exc:
                warnings.warn(f"PSD failed [{ctrl.name} stage2 mode {mode_idx}]: {exc}")

        # ── 1st-stage dashed overlay ────────────────────────────────────────
        if cfg.overlay_stage1 and ctrl.stage1 is not None:
            s1      = ctrl.stage1
            i0, i1  = find_stable_segment(s1.strehl, cfg.sr_threshold)
            try:
                f1, p1 = welch_psd(s1.modes[i0:i1, mode_idx],
                                    s1.frequency, cfg.welch_nperseg)
                mask   = f1 <= cfg.stage1_freq_lim
                ax.plot(f1[mask],
                        (np.cumsum(p1) if cumulative else p1)[mask],
                        color=s1.color, linestyle=s1.linestyle,
                        label=f"1st — {s1.label}")
            except Exception as exc:
                warnings.warn(f"PSD failed [{ctrl.name} stage1 mode {mode_idx}]: {exc}")

    _style(ax,
           title=f"{kind} — mode {mode_idx}  (gain = {cfg.cl_gain_pyr})",
           xlabel="Frequency (Hz)",
           ylabel="PSD",
           log_x=True, log_y=True)
    fig.tight_layout()


def plot_psds(controllers: Dict[str, ControllerResult],
              modes_atm: np.ndarray,
              cfg: Config,
              cumulative: bool = False) -> None:
    """Generate one PSD figure per entry in cfg.plot_modes."""
    for mode_idx in cfg.plot_modes:
        _psd_single_mode(controllers, modes_atm, mode_idx, cfg,
                         cumulative=cumulative)


def plot_timeseries(controllers: Dict[str, ControllerResult],
                    cfg: Config) -> None:
    """
    Raw residual time-series for each mode in cfg.plot_modes.
    One figure per mode; both stages overlaid when cfg.overlay_stage1 is True.
    """
    for mode_idx in cfg.plot_modes:
        fig, ax = plt.subplots(figsize=(11, 4))

        for ctrl in controllers.values():
            stages: List[Optional[StageData]] = []
            if cfg.overlay_stage1:
                stages.append(ctrl.stage1)
            stages.append(ctrl.stage2)

            for sd in stages:
                if sd is None:
                    continue
                ax.plot(sd.time_axis, sd.modes[:, mode_idx],
                        color=sd.color, linestyle=sd.linestyle,
                        alpha=0.85, label=sd.label)

        _style(ax,
               title=f"Residual time series — mode {mode_idx}",
               xlabel="Time (s)",
               ylabel=f"KL mode {mode_idx}  coefficient")
        fig.tight_layout()


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg = Config()

    # ── calibration matrices ────────────────────────────────────────────────
    _C2M_1st, C2M_2nd = load_calibration(cfg)
    # Note: C2M_1st retained here for future use (e.g. 1st-stage projection)

    # ── pupil geometry ──────────────────────────────────────────────────────
    pupil_mask, projector_kl = load_pupil_and_projector(cfg)

    # ── open-loop atmospheric modes ─────────────────────────────────────────
    modes_atm = load_atmosphere_modes(cfg, pupil_mask, projector_kl)

    # ── controller results ──────────────────────────────────────────────────
    controllers = load_controllers(cfg, C2M_2nd, pupil_mask)

    # ── plots ───────────────────────────────────────────────────────────────
    if cfg.use_rl:
        plot_training_losses(cfg)

    plot_strehl(controllers, cfg)
    plot_kl_variance(controllers, modes_atm, cfg)

    if cfg.plot_psd:
        plot_psds(controllers, modes_atm, cfg, cumulative=False)

    if cfg.plot_cumulative:
        plot_psds(controllers, modes_atm, cfg, cumulative=True)

    if cfg.plot_timeseries:
        plot_timeseries(controllers, cfg)

    plt.show()


if __name__ == "__main__":
    main()