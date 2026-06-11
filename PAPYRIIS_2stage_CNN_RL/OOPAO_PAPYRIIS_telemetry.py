"""
ao_metrics.py
=============
Post-processing and visualisation for two-stage AO simulation results.
 
Produces:
  * RL training-loss curves  (dynamics + policy)
  * Strehl-ratio time series with running-average overlay
  * KL-coefficient variance spectra
  * Per-mode temporal PSDs  (raw + cumulative)
  * Per-mode residual time series  (optional)
  * tETFsn #TODO need to implement
 
"""
 
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

#TODO don't forget to also change your atmosphere parameters when loading files (atm_OPD_array)
#TODO could you in fact rewrite this so unneccesary stuff is not loaded?
#TODO change the atmosphere used in plotting
#atm for open loop data
#1st stage residuals
#2nd stage residuals
#M2C for projection
#strehl calculation from OPDs
#RL, int, first stage, OL
#influence functions of the 2 deformable mirrors
#fitting error + temporal error?
#TODO include the ideal case (i.e. no noise and no bullshit)

#TODO delete if not in use
only_2nd             = False
stage_2              = True
stage_1_plus_stage_2 = True
atm_RL               = True
atm_int              = False
use_1st_KL           = False
label_RL = "RL"
label_int = "integrator"
label_ideal = "nonoiseRL"


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
    plot_tpsd:       bool = True
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
 

#this is supposed to get you the longest possible strehl timeseries that above a certain threshold
#i.e. when the loop is optimally closed so that you see the best results
#for on-sky this is not really needed, UNLESS the closed loop failed
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

#TODO look through these loaders because they are strange
def _load_tail(path: Path, n: int) -> Dict[str, np.ndarray]:
    """Load an .npz and return the last *n* rows of every array."""
    raw = np.load(path)
    return {k: raw[k][-n:] for k in raw.files}
 
 
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


#TODO you should get all of the KL modes from the deformable mirror commands just like irl
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

errr
PAPYRIIS_env = OOPAO_environment_PAPYRIIS()

atm_OPDs_2nd = np.load("PAPYRIIS_2stage_CNN_RL/generated_atm_2nd_stage/atm_OPDs_2nd_r0_0.050_V0_4.121.npz")
atm_OPDs_2nd = atm_OPDs_2nd["atm_OPDs_2nd"][-30000:]

results_2nd_stage_RL    = np.load(f'{loaddir}/results_2nd_stage.npz')
results_2nd_stage_int   = np.load(f'{loaddir}/results_2nd_stage_int.npz')
results_2nd_stage_ideal = np.load("PAPYRIIS_2stage_CNN_RL/~2026-06-01/PAPYRIIS_arcturus_nonoise/results_2nd_stage_r0_0.050_V0_4.121.npz")

#results_1st_stage_og       = np.load(f'{loaddir}/results_1st_stage_r0_0.050_V0_4.121.npz')    #TODO fix this later
results_1st_stage       = np.load(f'{loaddir}/results_2nd_stage_CL1OL2.npz')
print(results_2nd_stage_RL.files)
print(results_1st_stage.files)

M2C_1st             = np.load("PAPYRIIS_2stage_CNN_RL/M2C_1rst.npy")
M2C_2nd             = np.load('PAPYRIIS_2stage_CNN_RL/M2C_KL.npy')

C2M_1st = np.linalg.pinv(M2C_1st)
C2M_2nd = np.linalg.pinv(M2C_2nd)


"""dm_1st_inf = PAPYRIIS_env.dm_1st.modes
dm_2nd_inf = PAPYRIIS_env.dm_2nd.modes

print(dm_1st_inf.shape)
print(dm_2nd_inf.shape)

errr
M2OPD_1st = dm_1st_inf @ M2C_1st
M2OPD_2nd = dm_2nd_inf @ M2C_2nd
projector_kl = np.linalg.pinv(M2P_1st) @ M2P_2nd"""

pupil_mask_1st      = results_1st_stage['telescope_pupil'].astype(bool)
pupil_mask_2nd      = results_2nd_stage_RL["telescope_pupil"].astype(bool)

#projector_kl_1st    = results_1st_stage_og['projector_kl_1st'].reshape(-1, 80, 80)[:, pupil_mask_1st]    #from OPD to modes
projector_kl_2nd    = results_2nd_stage_RL["projector_kl_2nd"].reshape(-1, 90, 90)[:, pupil_mask_2nd]


#strehl = OPDs["telescope_pupil"]


#RL
if RL:

    if stage_2:
        #residual_opds_2nd_RL   = results_2nd_stage_RL["residual_opds_2nd"][-30000:]
        #residual_phase_2nd_RL  = 2 * np.pi * residual_opds_2nd_RL / (1.6e-06)
        #strehl_array_2nd_RL = np.exp(-np.var(residual_phase_2nd_RL[:, pupil_mask_2nd], axis=1))
        strehl_array_2nd_RL = results_2nd_stage_RL["all_2nd_stage_strehl"][-30000:]


        next_states_2nd_RL = results_2nd_stage_RL["all_reconstructed_cmd"][-30000:]
        modes_2nd_stage_RL = next_states_2nd_RL @ C2M_2nd.T



    if only_2nd:
        modes_1st_stage_RL = modes_2nd_stage_RL
    else:
        residual_opds_1st_RL   = results_1st_stage["residual_opds_2nd"][-30000:]
        residual_phase_1st_RL  = 2 * np.pi * residual_opds_1st_RL / (1.6e-06)
        strehl_array_1st_RL = np.exp(-np.var(residual_phase_1st_RL[:, pupil_mask_2nd], axis=1))
        next_states_1st = results_1st_stage["all_reconstructed_cmd"][-30000:]
        modes_1st_stage_RL = next_states_1st @ C2M_2nd.T



    dm_atm = atm_OPDs_2nd[:, pupil_mask_2nd]
    modes_atm_RL = dm_atm @ projector_kl_2nd.T 
    

    dynamics_loss = np.load(f"{folder_name}/{directory_name_RL}/dynamics_loss.npy") 
    policy_loss = np.load(f"{folder_name}/{directory_name_RL}/policy_loss.npy") 



    frequency_RL = frequency_2nd
    time_plot_RL = np.arange(0, 30000 / frequency_2nd, 1 / frequency_2nd) #time_plot = np.arange(0, iters * episode_length / frequency, 1/frequency)



#integrator
if integrator:
    if stage_2:
        #residual_opds_2nd_int   = results_2nd_stage_int["residual_opds_2nd"][-30000:]
        #residual_phase_2nd_int  = 2 * np.pi * residual_opds_2nd_int / (1.6e-06)
        #strehl_array_2nd_int = np.exp(-np.var(residual_phase_2nd_int[:, pupil_mask_2nd], axis=1))
        strehl_array_2nd_int = results_2nd_stage_int["all_2nd_stage_strehl"][-30000:]

        next_states_2nd = results_2nd_stage_int["all_reconstructed_cmd"][-30000:]
        modes_2nd_stage_int = next_states_2nd @ C2M_2nd.T



    if only_2nd:
        modes_1st_stage_int = modes_2nd_stage_int
    else:
        residual_opds_1st_int   = results_1st_stage["residual_opds_2nd"][-30000:]
        residual_phase_1st_int  = 2 * np.pi * residual_opds_1st_int / (1.6e-06)
        strehl_array_1st_int = np.exp(-np.var(residual_phase_1st_int[:, pupil_mask_2nd], axis=1))
        next_states_1st = results_1st_stage["all_reconstructed_cmd"][-30000:]
        modes_1st_stage_int = next_states_1st @ C2M_2nd.T



    frequency_int = frequency_2nd
    time_plot_int = np.arange(0, 30000 / frequency_2nd, 1 / frequency_2nd)


#ideal
if ideal:
    if stage_2:
        residual_opds_2nd_ideal   = results_2nd_stage_ideal["all_src_opd"][-30000:]
        residual_phase_2nd_ideal  = 2 * np.pi * residual_opds_2nd_ideal / (1.6e-06)
        strehl_array_2nd_ideal = np.exp(-np.var(residual_phase_2nd_ideal[:, pupil_mask_2nd], axis=1))

        next_states_2nd = results_2nd_stage_ideal["all_reconstructed_cmd"][-30000:]
        modes_2nd_stage_ideal = next_states_2nd @ C2M_2nd.T



    if only_2nd:
        modes_1st_stage_ideal = modes_2nd_stage_ideal
    else:
        residual_opds_1st_ideal   = results_1st_stage["residual_opds_2nd"][-30000:]
        residual_phase_1st_ideal  = 2 * np.pi * residual_opds_1st_ideal / (1.6e-06)
        strehl_array_1st_ideal = np.exp(-np.var(residual_phase_1st_ideal[:, pupil_mask_2nd], axis=1))
        next_states_1st = results_1st_stage["all_reconstructed_cmd"][-30000:]
        modes_1st_stage_ideal = next_states_1st @ C2M_2nd.T



    frequency_ideal = frequency_2nd
    time_plot_ideal = np.arange(0, 30000 / frequency_2nd, 1 / frequency_2nd)



if atm_RL:
    modes_atm = modes_atm_RL
    time_plot_atm = time_plot_RL
    f_samp_atm = frequency_atm

if atm_int:
    modes_atm = modes_atm_int
    time_plot_atm = time_plot_int
    f_samp_atm = frequency_atm



# ---------------------------------------------------Loss---------------------------------------------------#
if RL:
    plt.figure()
    plt.subplot(121)
    plt.title("dynamics_loss warmup")
    plt.plot(dynamics_loss)
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.xscale('log')
    plt.yscale('log')
    plt.subplot(122)
    plt.title("policy_loss warmup")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.plot(policy_loss)
    plt.xscale('log')
    plt.yscale('log')





# ---------------------------------------------------Strehl---------------------------------------------------#
if RL:
    if stage_2:
        sr_mean_RL = np.mean(strehl_array_2nd_RL)
    else:
        sr_mean_RL = np.mean(strehl_array_1st_RL)

    kernel = np.ones(500) / 500

    #pad sr
    pad_left = len(kernel) // 2
    pad_right = len(kernel) - pad_left - 1

    sr_padded_1st_RL = np.pad(strehl_array_1st_RL, (pad_left, pad_right), mode='constant', constant_values=np.mean(strehl_array_1st_RL))
    sr_running_1st_RL = np.convolve(sr_padded_1st_RL, kernel, mode='valid')

    if stage_2:
        sr_padded_2nd_RL = np.pad(strehl_array_2nd_RL, (pad_left, pad_right), mode='constant', constant_values=sr_mean_RL)
        sr_running_2nd_RL = np.convolve(sr_padded_2nd_RL, kernel, mode='valid')

if integrator:
    if stage_2:
        sr_mean_int = np.mean(strehl_array_2nd_int)
    else:
        sr_mean_int = np.mean(strehl_array_1st_int)

    kernel = np.ones(500) / 500

    #pad sr
    pad_left = len(kernel) // 2
    pad_right = len(kernel) - pad_left - 1

    sr_padded_1st_int = np.pad(strehl_array_1st_int, (pad_left, pad_right), mode='constant', constant_values=np.mean(strehl_array_1st_int))
    sr_running_1st_int = np.convolve(sr_padded_1st_int, kernel, mode='valid')

    if stage_2:
        sr_padded_2nd_int = np.pad(strehl_array_2nd_int, (pad_left, pad_right), mode='constant', constant_values=sr_mean_int)
        sr_running_2nd_int = np.convolve(sr_padded_2nd_int, kernel, mode='valid')

if ideal:
    if stage_2:
        sr_mean_ideal = np.mean(strehl_array_2nd_ideal)
    else:
        sr_mean_ideal = np.mean(strehl_array_1st_ideal)

    kernel = np.ones(500) / 500

    #pad sr
    pad_left = len(kernel) // 2
    pad_right = len(kernel) - pad_left - 1

    sr_padded_1st_ideal = np.pad(strehl_array_1st_ideal, (pad_left, pad_right), mode='constant', constant_values=sr_mean_ideal)
    sr_running_1st_ideal = np.convolve(sr_padded_1st_ideal, kernel, mode='valid')

    if stage_2:
        sr_padded_2nd_ideal = np.pad(strehl_array_2nd_ideal, (pad_left, pad_right), mode='constant', constant_values=sr_mean_ideal)
        sr_running_2nd_ideal = np.convolve(sr_padded_2nd_ideal, kernel, mode='valid')

plt.figure()
#if integrator:  plt.plot(time_plot_int, strehl_array_1st_int[:len(time_plot_int)], color = '#ff7f0e', label=f"1st_{label_int}")
#if ideal:       plt.plot(time_plot_ideal, strehl_array_1st_ideal[:len(time_plot_ideal)], color = 'black', label=f"1st_{label_ideal}")
if RL:          plt.plot(time_plot_RL, strehl_array_1st_RL, color = '#1f77b4', label=f"1st_{label_RL}")

if not stage_2:
    if RL: plt.plot(time_plot_RL, sr_running_1st_RL, label=f"running_avg_1st_{label_RL}")
if stage_2:
    #if RL: plt.plot(time_plot_RL, strehl_array_2nd_RL, color = '#003f5c', label=f"2nd_{label_RL}")
    if RL: plt.plot(time_plot_RL, sr_running_2nd_RL, color = 'red', label=f"running_avg_2nd_{label_RL}")

    #if integrator: plt.plot(time_plot_int, strehl_array_2nd_int, color = '#ffa600', label=f"2nd_{label_int}")
    if integrator: plt.plot(time_plot_int, sr_running_2nd_int, color='darkturquoise', label=f"running_avg_2nd_{label_int}")

    #if ideal: plt.plot(time_plot_ideal, strehl_array_2nd_ideal, color = 'black', label=f"2nd_{label_ideal}")
    if ideal: plt.plot(time_plot_ideal, sr_running_2nd_ideal, color='black',label=f"running_avg_2nd_{label_ideal}")


plt.title("Strehl ratio")
plt.xlabel("time s")
#plt.ylim(bottom=(sr_mean_RL - 0.5))
plt.grid(True, alpha=0.5)
plt.minorticks_on()
plt.legend()


# ---------------------------------------------------Zernike/KL decomposition---------------------------------------------------#
#RL
if RL:
    modes_1st_stage_RL_masked = modes_1st_stage_RL[KL_frame_thresh_RL:KL_frame_end_RL, :]
    coefs_var_1st_stage_masked_RL = np.var(np.asarray(modes_1st_stage_RL_masked), axis = 0)
    coefs_var_1st_stage_RL = np.var(np.asarray(modes_1st_stage_RL), axis = 0)

    if stage_2:
        modes_2nd_stage_RL_masked = modes_2nd_stage_RL[KL_frame_thresh_RL:KL_frame_end_RL, :]
        coefs_var_2nd_stage_masked_RL = np.var(np.asarray(modes_2nd_stage_RL_masked), axis=0)
        coefs_var_2nd_stage_RL = np.var(np.asarray(modes_2nd_stage_RL), axis = 0)


#integrator
if integrator:
    modes_1st_stage_int_masked = modes_1st_stage_int[KL_frame_thresh_int:KL_frame_end_int, :]
    coefs_var_1st_stage_masked_int = np.var(np.asarray(modes_1st_stage_int_masked), axis = 0)
    coefs_var_1st_stage_int = np.var(np.asarray(modes_1st_stage_int), axis=0)

    if stage_2:
        modes_2nd_stage_int_masked = modes_2nd_stage_int[KL_frame_thresh_int:KL_frame_end_int, :]
        coefs_var_2nd_stage_masked_int = np.var(np.asarray(modes_2nd_stage_int_masked), axis = 0)
        coefs_var_2nd_stage_int = np.var(np.asarray(modes_2nd_stage_int), axis = 0)


#ideal
if ideal:
    modes_1st_stage_ideal_masked = modes_1st_stage_ideal[KL_frame_thresh_ideal:, :]
    coefs_var_1st_stage_masked_ideal = np.var(np.asarray(modes_1st_stage_ideal_masked), axis = 0)
    coefs_var_1st_stage_ideal = np.var(np.asarray(modes_1st_stage_ideal), axis=0)

    if stage_2:
        modes_2nd_stage_ideal_masked = modes_2nd_stage_ideal[KL_frame_thresh_ideal:, :]
        coefs_var_2nd_stage_masked_ideal = np.var(np.asarray(modes_2nd_stage_ideal_masked), axis = 0)
        coefs_var_2nd_stage_ideal = np.var(np.asarray(modes_2nd_stage_ideal), axis = 0)


#atmosphere
coefs_var_atm = np.var(np.asarray(modes_atm[KL_frame_thresh_int:, :]), axis = 0)


plt.figure()
#plt.plot(coefs_var_atm, color="black",label=f"KL coeffs for atmospheric phase")
if RL:
    if stage_2:
        if stage_1_plus_stage_2:
            plt.plot(coefs_var_1st_stage_masked_RL, '--', color="indianred", label=f"KL coeffs 1st stage_int")
        plt.plot(coefs_var_2nd_stage_masked_RL, color="red", label=f"KL coeffs 2nd stage {label_RL}")
    else:
        plt.plot(coefs_var_1st_stage_masked_RL, color="indianred", label=f"KL coeffs {label_RL}")

if integrator:
    if stage_2:
        plt.plot(coefs_var_2nd_stage_masked_int, color="blue", label=f"KL coeffs 2nd stage {label_int}")
    else:
        plt.plot(coefs_var_1st_stage_masked_int, color="cornflowerblue", label=f"KL coeffs {label_int}")

if ideal:
    if stage_2:
        plt.plot(coefs_var_2nd_stage_masked_ideal, color="green", label=f"KL coeffs 2nd stage {label_ideal}")
    else:
        plt.plot(coefs_var_1st_stage_masked_ideal, color="seagreen", label=f"KL coeffs {label_ideal}")

plt.title(f"KL coefficients RL vs integrator", fontsize = 18)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("KL coefficients", fontsize = 14)
plt.ylabel("KL coefficient variance", fontsize = 14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.ylim(bottom = 5e-6)
#plt.tight_layout()
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
#plt.savefig("KL_decomp_on_sky_04_02.png", dpi=300)



# ---------------------------------------------------Temporal PSD---------------------------------------------------#
# temporal PSD calculation from the std
def welch_method_scipy(data, fs, nperseg=256):
    frequencies, psd = signal.welch(
        data,
        fs=fs,
        window='hann',  #windowing
        nperseg=nperseg,
        scaling='density'
    )
    return frequencies, psd

if RL:
    f_samp_RL = frequency_RL
    # tip timeseries
    f = modes_1st_stage_RL[:, mode_1]
    residual_mode_2_curve_1st_RL_full = modes_1st_stage_RL[:, mode_2]
    residual_mode_3_curve_1st_RL_full = modes_1st_stage_RL[:, mode_3]
    residual_mode_4_curve_1st_RL_full = modes_1st_stage_RL[:, mode_4]
    residual_mode_5_curve_1st_RL_full = modes_1st_stage_RL[:, mode_5]

    if stage_2:
        residual_mode_1_curve_2nd_RL_full = modes_2nd_stage_RL[:, mode_1]
        residual_mode_2_curve_2nd_RL_full = modes_2nd_stage_RL[:, mode_2]
        residual_mode_3_curve_2nd_RL_full = modes_2nd_stage_RL[:, mode_3]
        residual_mode_4_curve_2nd_RL_full = modes_2nd_stage_RL[:, mode_4]
        residual_mode_5_curve_2nd_RL_full = modes_2nd_stage_RL[:, mode_5]


if integrator:
    f_samp_int = frequency_int
    # tip timeseries
    residual_mode_1_curve_1st_int_full = modes_1st_stage_int[:, mode_1]
    residual_mode_2_curve_1st_int_full = modes_1st_stage_int[:, mode_2]
    residual_mode_3_curve_1st_int_full = modes_1st_stage_int[:, mode_3]
    residual_mode_4_curve_1st_int_full = modes_1st_stage_int[:, mode_4]
    residual_mode_5_curve_1st_int_full = modes_1st_stage_int[:, mode_5]

    if stage_2:
        residual_mode_1_curve_2nd_int_full = modes_2nd_stage_int[:, mode_1]
        residual_mode_2_curve_2nd_int_full = modes_2nd_stage_int[:, mode_2]
        residual_mode_3_curve_2nd_int_full = modes_2nd_stage_int[:, mode_3]
        residual_mode_4_curve_2nd_int_full = modes_2nd_stage_int[:, mode_4]
        residual_mode_5_curve_2nd_int_full = modes_2nd_stage_int[:, mode_5]


if ideal:
    f_samp_ideal = frequency_ideal
    # tip timeseries
    residual_mode_1_curve_1st_ideal_full = modes_1st_stage_ideal[:, mode_1]
    residual_mode_2_curve_1st_ideal_full = modes_1st_stage_ideal[:, mode_2]
    residual_mode_3_curve_1st_ideal_full = modes_1st_stage_ideal[:, mode_3]
    residual_mode_4_curve_1st_ideal_full = modes_1st_stage_ideal[:, mode_4]
    residual_mode_5_curve_1st_ideal_full = modes_1st_stage_ideal[:, mode_5]

    if stage_2:
        residual_mode_1_curve_2nd_ideal_full = modes_2nd_stage_ideal[:, mode_1]
        residual_mode_2_curve_2nd_ideal_full = modes_2nd_stage_ideal[:, mode_2]
        residual_mode_3_curve_2nd_ideal_full = modes_2nd_stage_ideal[:, mode_3]
        residual_mode_4_curve_2nd_ideal_full = modes_2nd_stage_ideal[:, mode_4]
        residual_mode_5_curve_2nd_ideal_full = modes_2nd_stage_ideal[:, mode_5]



if RL:
    #RL 1st and 2nd stage selected mode PSD calculation
    residual_mode_1_curve_1st_RL = modes_1st_stage_RL_masked[:, mode_1]
    residual_mode_2_curve_1st_RL = modes_1st_stage_RL_masked[:, mode_2]
    residual_mode_3_curve_1st_RL = modes_1st_stage_RL_masked[:, mode_3]
    residual_mode_4_curve_1st_RL = modes_1st_stage_RL_masked[:, mode_4]
    residual_mode_5_curve_1st_RL = modes_1st_stage_RL_masked[:, mode_5]

    if stage_2:
        residual_mode_1_curve_2nd_RL = modes_2nd_stage_RL_masked[:, mode_1]
        residual_mode_2_curve_2nd_RL = modes_2nd_stage_RL_masked[:, mode_2]
        residual_mode_3_curve_2nd_RL = modes_2nd_stage_RL_masked[:, mode_3]
        residual_mode_4_curve_2nd_RL = modes_2nd_stage_RL_masked[:, mode_4]
        residual_mode_5_curve_2nd_RL = modes_2nd_stage_RL_masked[:, mode_5]

if integrator:
    #int 1st and 2nd stage selected mode PSD calculation
    residual_mode_1_curve_1st_int = modes_1st_stage_int_masked[:, mode_1]
    residual_mode_2_curve_1st_int = modes_1st_stage_int_masked[:, mode_2]
    residual_mode_3_curve_1st_int = modes_1st_stage_int_masked[:, mode_3]
    residual_mode_4_curve_1st_int = modes_1st_stage_int_masked[:, mode_4]
    residual_mode_5_curve_1st_int = modes_1st_stage_int_masked[:, mode_5]

    if stage_2:
        residual_mode_1_curve_2nd_int = modes_2nd_stage_int_masked[:, mode_1]
        residual_mode_2_curve_2nd_int = modes_2nd_stage_int_masked[:, mode_2]
        residual_mode_3_curve_2nd_int = modes_2nd_stage_int_masked[:, mode_3]
        residual_mode_4_curve_2nd_int = modes_2nd_stage_int_masked[:, mode_4]
        residual_mode_5_curve_2nd_int = modes_2nd_stage_int_masked[:, mode_5]

if ideal:
    #ideal 1st and 2nd stage selected mode PSD calculation
    residual_mode_1_curve_1st_ideal = modes_1st_stage_ideal_masked[:, mode_1]
    residual_mode_2_curve_1st_ideal = modes_1st_stage_ideal_masked[:, mode_2]
    residual_mode_3_curve_1st_ideal = modes_1st_stage_ideal_masked[:, mode_3]
    residual_mode_4_curve_1st_ideal = modes_1st_stage_ideal_masked[:, mode_4]
    residual_mode_5_curve_1st_ideal = modes_1st_stage_ideal_masked[:, mode_5]

    if stage_2:
        residual_mode_1_curve_2nd_ideal = modes_2nd_stage_ideal_masked[:, mode_1]
        residual_mode_2_curve_2nd_ideal = modes_2nd_stage_ideal_masked[:, mode_2]
        residual_mode_3_curve_2nd_ideal = modes_2nd_stage_ideal_masked[:, mode_3]
        residual_mode_4_curve_2nd_ideal = modes_2nd_stage_ideal_masked[:, mode_4]
        residual_mode_5_curve_2nd_ideal = modes_2nd_stage_ideal_masked[:, mode_5]


#atmosphere modes for PSD
atm_mode_1_curve  = modes_atm[:, mode_1]
atm_mode_2_curve  = modes_atm[:, mode_2]
atm_mode_3_curve  = modes_atm[:, mode_3]
atm_mode_4_curve  = modes_atm[:, mode_4]
atm_mode_5_curve  = modes_atm[:, mode_5]

if RL:
    PSD_residual_mode_1_freq_t_1st_RL, PSD_residual_mode_1_1st_RL = welch_method_scipy(residual_mode_1_curve_1st_RL, frequency_1st)
    PSD_residual_mode_2_freq_t_1st_RL, PSD_residual_mode_2_1st_RL = welch_method_scipy(residual_mode_2_curve_1st_RL, frequency_1st)
    PSD_residual_mode_3_freq_t_1st_RL, PSD_residual_mode_3_1st_RL = welch_method_scipy(residual_mode_3_curve_1st_RL, frequency_1st)
    PSD_residual_mode_4_freq_t_1st_RL, PSD_residual_mode_4_1st_RL = welch_method_scipy(residual_mode_4_curve_1st_RL, frequency_1st)
    PSD_residual_mode_5_freq_t_1st_RL, PSD_residual_mode_5_1st_RL = welch_method_scipy(residual_mode_5_curve_1st_RL, frequency_1st)


    if stage_2:
        PSD_residual_mode_1_freq_t_2nd_RL, PSD_residual_mode_1_2nd_RL = welch_method_scipy(residual_mode_1_curve_2nd_RL, f_samp_RL)
        PSD_residual_mode_2_freq_t_2nd_RL, PSD_residual_mode_2_2nd_RL = welch_method_scipy(residual_mode_2_curve_2nd_RL, f_samp_RL)
        PSD_residual_mode_3_freq_t_2nd_RL, PSD_residual_mode_3_2nd_RL = welch_method_scipy(residual_mode_3_curve_2nd_RL, f_samp_RL)
        PSD_residual_mode_4_freq_t_2nd_RL, PSD_residual_mode_4_2nd_RL = welch_method_scipy(residual_mode_4_curve_2nd_RL, f_samp_RL)
        PSD_residual_mode_5_freq_t_2nd_RL, PSD_residual_mode_5_2nd_RL = welch_method_scipy(residual_mode_5_curve_2nd_RL, f_samp_RL)


if integrator:
    PSD_residual_mode_1_freq_t_1st_int, PSD_residual_mode_1_1st_int = welch_method_scipy(residual_mode_1_curve_1st_int, frequency_1st)
    PSD_residual_mode_2_freq_t_1st_int, PSD_residual_mode_2_1st_int = welch_method_scipy(residual_mode_2_curve_1st_int, frequency_1st)
    PSD_residual_mode_3_freq_t_1st_int, PSD_residual_mode_3_1st_int = welch_method_scipy(residual_mode_3_curve_1st_int, frequency_1st)
    PSD_residual_mode_4_freq_t_1st_int, PSD_residual_mode_4_1st_int = welch_method_scipy(residual_mode_4_curve_1st_int, frequency_1st)
    PSD_residual_mode_5_freq_t_1st_int, PSD_residual_mode_5_1st_int = welch_method_scipy(residual_mode_5_curve_1st_int, frequency_1st)

    if stage_2:
        PSD_residual_mode_1_freq_t_2nd_int, PSD_residual_mode_1_2nd_int = welch_method_scipy(residual_mode_1_curve_2nd_int, f_samp_int)
        PSD_residual_mode_2_freq_t_2nd_int, PSD_residual_mode_2_2nd_int = welch_method_scipy(residual_mode_2_curve_2nd_int, f_samp_int)
        PSD_residual_mode_3_freq_t_2nd_int, PSD_residual_mode_3_2nd_int = welch_method_scipy(residual_mode_3_curve_2nd_int, f_samp_int)
        PSD_residual_mode_4_freq_t_2nd_int, PSD_residual_mode_4_2nd_int = welch_method_scipy(residual_mode_4_curve_2nd_int, f_samp_int)
        PSD_residual_mode_5_freq_t_2nd_int, PSD_residual_mode_5_2nd_int = welch_method_scipy(residual_mode_5_curve_2nd_int, f_samp_int)


if ideal:
    #tip
    PSD_residual_mode_1_freq_t_1st_ideal, PSD_residual_mode_1_1st_ideal = welch_method_scipy(residual_mode_1_curve_1st_ideal, frequency_1st)
    PSD_residual_mode_2_freq_t_1st_ideal, PSD_residual_mode_2_1st_ideal = welch_method_scipy(residual_mode_2_curve_1st_ideal, frequency_1st)
    PSD_residual_mode_3_freq_t_1st_ideal, PSD_residual_mode_3_1st_ideal = welch_method_scipy(residual_mode_3_curve_1st_ideal, frequency_1st)
    PSD_residual_mode_4_freq_t_1st_ideal, PSD_residual_mode_4_1st_ideal = welch_method_scipy(residual_mode_4_curve_1st_ideal, frequency_1st)
    PSD_residual_mode_5_freq_t_1st_ideal, PSD_residual_mode_5_1st_ideal = welch_method_scipy(residual_mode_5_curve_1st_ideal, frequency_1st)

    if stage_2:
        PSD_residual_mode_1_freq_t_2nd_ideal, PSD_residual_mode_1_2nd_ideal = welch_method_scipy(residual_mode_1_curve_2nd_ideal, f_samp_ideal)
        PSD_residual_mode_2_freq_t_2nd_ideal, PSD_residual_mode_2_2nd_ideal = welch_method_scipy(residual_mode_2_curve_2nd_ideal, f_samp_ideal)
        PSD_residual_mode_3_freq_t_2nd_ideal, PSD_residual_mode_3_2nd_ideal = welch_method_scipy(residual_mode_3_curve_2nd_ideal, f_samp_ideal)
        PSD_residual_mode_4_freq_t_2nd_ideal, PSD_residual_mode_4_2nd_ideal = welch_method_scipy(residual_mode_4_curve_2nd_ideal, f_samp_ideal)
        PSD_residual_mode_5_freq_t_2nd_ideal, PSD_residual_mode_5_2nd_ideal = welch_method_scipy(residual_mode_5_curve_2nd_ideal, f_samp_ideal)


PSD_atm_mode_1_freq_t, PSD_atm_mode_1 = welch_method_scipy(atm_mode_1_curve, f_samp_atm)
PSD_atm_mode_2_freq_t, PSD_atm_mode_2 = welch_method_scipy(atm_mode_2_curve, f_samp_atm)
PSD_atm_mode_3_freq_t, PSD_atm_mode_3 = welch_method_scipy(atm_mode_3_curve, f_samp_atm)
PSD_atm_mode_4_freq_t, PSD_atm_mode_4 = welch_method_scipy(atm_mode_4_curve, f_samp_atm)
PSD_atm_mode_5_freq_t, PSD_atm_mode_5 = welch_method_scipy(atm_mode_5_curve, f_samp_atm)


if plot_timeseries:
    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_RL, residual_mode_1_curve_1st_RL_full, color="indianred", label=f"mode_{mode_1}_1st_stage")
            plt.plot(time_plot_RL, residual_mode_1_curve_2nd_RL_full, color="red", label=f"mode_{mode_1}_2nd_stage_{label_RL}")
        else: plt.plot(time_plot_RL, residual_mode_1_curve_1st_RL_full, color="indianred", label=f"mode_{mode_1}_1st_stage_{label_RL}")
    if integrator:
        if stage_2:
            plt.plot(time_plot_int, residual_mode_1_curve_2nd_int_full, color="blue", label=f"mode_{mode_1}_2nd_stage_{label_int}")
        else: plt.plot(time_plot_int, residual_mode_1_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_1}_1st_stage_{label_int}")
    if ideal:
        if stage_2:
            plt.plot(time_plot_ideal, residual_mode_1_curve_2nd_ideal_full, label=f"mode_{mode_1}_2nd_stage_{label_ideal}")
        else: plt.plot(time_plot_ideal, residual_mode_1_curve_1st_ideal_full, label=f"mode_{mode_1}_1st_stage_{label_ideal}")
    #plt.plot(time_plot_atm, atm_mode_1_curve, color="black", label=f"atm_mode_{mode_1}_curve")
    plt.title(f"residual/atm timeseries mode_{mode_1}, gain {CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_1}")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_RL, residual_mode_2_curve_1st_RL_full, color="indianred", label=f"mode_{mode_2}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_2_curve_2nd_RL_full, color="red", label=f"mode_{mode_2}_2nd_stage_{label_RL}")
        else: plt.plot(time_plot_RL, residual_mode_2_curve_1st_RL_full, color="indianred", label=f"mode_{mode_2}_1st_stage_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_int, residual_mode_2_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_2}_1st_stage_{label_int}")
            plt.plot(time_plot_int, residual_mode_2_curve_2nd_int_full, color="blue", label=f"mode_{mode_2}_2nd_stage_{label_int}")
        else: plt.plot(time_plot_int, residual_mode_2_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_2}_1st_stage_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_ideal, residual_mode_2_curve_1st_ideal_full, label=f"mode_{mode_2}_1st_stage_{label_ideal}")
            plt.plot(time_plot_ideal, residual_mode_2_curve_2nd_ideal_full, label=f"mode_{mode_2}_2nd_stage_{label_ideal}")
        else: plt.plot(time_plot_ideal, residual_mode_2_curve_1st_ideal_full, label=f"mode_{mode_2}_1st_stage_{label_ideal}")
    #plt.plot(time_plot_atm, atm_mode_2_curve, color="black", label=f"atm_mode_{mode_2}_curve")
    plt.title(f"residual/atm timeseries mode_{mode_2}, gain {CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_2}")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_RL, residual_mode_3_curve_1st_RL_full, color="indianred", label=f"mode_{mode_3}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_3_curve_2nd_RL_full, color="red", label=f"mode_{mode_3}_2nd_stage_{label_RL}")
        else: plt.plot(time_plot_RL, residual_mode_3_curve_1st_RL_full, color="indianred", label=f"mode_{mode_3}_1st_stage_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_int, residual_mode_3_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_3}_1st_stage_{label_int}")
            plt.plot(time_plot_int, residual_mode_3_curve_2nd_int_full, color="blue", label=f"mode_{mode_3}_2nd_stage_{label_int}")
        else: plt.plot(time_plot_int, residual_mode_3_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_3}_1st_stage_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_ideal, residual_mode_3_curve_1st_ideal_full, label=f"mode_{mode_3}_1st_stage__{label_ideal}")
            plt.plot(time_plot_ideal, residual_mode_3_curve_2nd_ideal_full, label=f"mode_{mode_3}_2nd_stage_{label_ideal}")
        else: plt.plot(time_plot_ideal, residual_mode_3_curve_1st_ideal_full, label=f"mode_{mode_3}_1st_stage_{label_ideal}")
    #plt.plot(time_plot_atm, atm_mode_3_curve, color="black", label=f"atm_mode_{mode_3}_curve")
    plt.title(f"residual/atm timeseries mode_{mode_3}, gain {CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_3}")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_RL, residual_mode_4_curve_1st_RL_full, color="indianred", label=f"mode_{mode_4}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_4_curve_2nd_RL_full, color="red", label=f"mode__{mode_4}_2nd_stage_{label_RL}")
        else: plt.plot(time_plot_RL, residual_mode_4_curve_1st_RL_full, color="indianred", label=f"mode_{mode_4}_1st_stage_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_int, residual_mode_4_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_4}_1st_stage_{label_int}")
            plt.plot(time_plot_int, residual_mode_4_curve_2nd_int_full, color="blue", label=f"mode_{mode_4}_2nd_stage_{label_int}")
        else: plt.plot(time_plot_int, residual_mode_4_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_4}_1st_stage_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_ideal, residual_mode_4_curve_1st_ideal_full, label=f"mode_{mode_4}_1st_stage_{label_ideal}")
            plt.plot(time_plot_ideal, residual_mode_4_curve_2nd_ideal_full, label=f"mode_{mode_4}_2nd_stage_{label_ideal}")
        else: plt.plot(time_plot_ideal, residual_mode_4_curve_1st_ideal_full, label=f"mode_{mode_4}_1st_stage_{label_ideal}")
    #plt.plot(time_plot_atm, atm_mode_4_curve, color="black", label=f"atm_mode_{mode_4}_curve")
    plt.title(f"residual/atm timeseries mode_{mode_4}, gain {CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_4}")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_RL, residual_mode_5_curve_1st_RL_full, color="indianred", label=f"mode_{mode_5}_1st_stage_{label_RL}")
            plt.plot(time_plot_RL, residual_mode_5_curve_2nd_RL_full, color="red", label=f"mode_{mode_5}_2nd_stage_{label_RL}")
        else: plt.plot(time_plot_RL, residual_mode_5_curve_1st_RL_full, color="indianred", label=f"mode_{mode_5}_1st_stage_{label_RL}")
    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_int, residual_mode_5_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_5}_1st_stage_{label_int}")
            plt.plot(time_plot_int, residual_mode_5_curve_2nd_int_full, color="blue", label=f"mode_{mode_5}_2nd_stage_{label_int}")
        else: plt.plot(time_plot_int, residual_mode_5_curve_1st_int_full, color="cornflowerblue", label=f"mode_{mode_5}_1st_stage_{label_int}")
    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(time_plot_ideal, residual_mode_5_curve_1st_ideal_full, label=f"mode_{mode_5}_1st_stage_{label_ideal}")
            plt.plot(time_plot_ideal, residual_mode_5_curve_2nd_ideal_full, label=f"mode_{mode_5}_2nd_stage_{label_ideal}")
        else: plt.plot(time_plot_ideal, residual_mode_5_curve_1st_ideal_full, label=f"mode_{mode_5}_1st_stage_{label_ideal}")
    #plt.plot(time_plot_atm, atm_mode_5_curve, color="black", label=f"atm_mode_{mode_5}_curve")
    plt.title(f"residual/atm timeseries mode_{mode_5}, gain {CL_gain_pyr}")
    plt.xlabel("time (s)")
    plt.grid(True, alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel(f"residual mode_{mode_5}")


if plot_tPSD:
    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_RL[np.where(PSD_residual_mode_1_freq_t_1st_RL <= stage_1_freq_lim)]
                         , PSD_residual_mode_1_1st_RL[np.where(PSD_residual_mode_1_freq_t_1st_RL <= stage_1_freq_lim)], '--', color="indianred", label=f"PSD_mode_{mode_1}_1st_int")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_RL, PSD_residual_mode_1_2nd_RL, color="red", label=f"PSD_mode_{mode_1}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_RL, PSD_residual_mode_1_1st_RL, color="indianred", label=f"PSD_mode_{mode_1}_1st_{label_RL}")
    if integrator:
        if stage_2:
            plt.plot(PSD_residual_mode_1_freq_t_2nd_int, PSD_residual_mode_1_2nd_int, color="blue", label=f"PSD_mode_{mode_1}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_int, PSD_residual_mode_1_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_1}_1st_{label_int}")
    if ideal:
        if stage_2:
            plt.plot(PSD_residual_mode_1_freq_t_2nd_ideal, PSD_residual_mode_1_2nd_ideal, label=f"PSD_mode_{mode_1}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_ideal, PSD_residual_mode_1_1st_ideal, label=f"PSD_mode_{mode_1}_1st_{label_ideal}")


    plt.plot(PSD_atm_mode_1_freq_t, PSD_atm_mode_1, color="black", label=f"atm_PSD_mode_{mode_1}")
    plt.title(f"residual PSD mode_{mode_1}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_2_freq_t_1st_RL[np.where(PSD_residual_mode_2_freq_t_1st_RL <= stage_1_freq_lim)]
                         , PSD_residual_mode_2_1st_RL[np.where(PSD_residual_mode_2_freq_t_1st_RL <= stage_1_freq_lim)], '--', color="indianred", label=f"PSD_mode_{mode_2}_1st_int")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_RL, PSD_residual_mode_2_2nd_RL, color="red", label=f"PSD_mode_{mode_2}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_RL, PSD_residual_mode_2_1st_RL, color="indianred", label=f"PSD_mode_{mode_2}_1st_{label_RL}")
    if integrator:
        if stage_2:
            plt.plot(PSD_residual_mode_2_freq_t_2nd_int, PSD_residual_mode_2_2nd_int, color="blue", label=f"PSD_mode_{mode_2}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_int, PSD_residual_mode_2_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_2}_1st_{label_int}")
    if ideal:
        if stage_2:
            plt.plot(PSD_residual_mode_2_freq_t_2nd_ideal, PSD_residual_mode_2_2nd_ideal, label=f"PSD_mode_{mode_2}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_ideal, PSD_residual_mode_2_1st_ideal, label=f"PSD_mode_{mode_2}_1st_{label_ideal}")


    plt.plot(PSD_atm_mode_2_freq_t, PSD_atm_mode_2, color="black", label=f"atm_PSD_mode_{mode_2}")
    plt.title(f"residual PSD mode_{mode_2}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_3_freq_t_1st_RL[np.where(PSD_residual_mode_3_freq_t_1st_RL <= stage_1_freq_lim)]
                         , PSD_residual_mode_3_1st_RL[np.where(PSD_residual_mode_3_freq_t_1st_RL <= stage_1_freq_lim)], '--', color="indianred", label=f"PSD_mode_{mode_3}_1st_int")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_RL, PSD_residual_mode_3_2nd_RL, color="red", label=f"PSD_mode_{mode_3}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_RL, PSD_residual_mode_3_1st_RL, color="indianred", label=f"PSD_mode_{mode_3}_1st_{label_RL}")
    if integrator:
        if stage_2:
            plt.plot(PSD_residual_mode_3_freq_t_2nd_int, PSD_residual_mode_3_2nd_int, color="blue", label=f"PSD_mode_{mode_3}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_int, PSD_residual_mode_3_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_3}_1st_{label_int}")
    if ideal:
        if stage_2:
            plt.plot(PSD_residual_mode_3_freq_t_2nd_ideal, PSD_residual_mode_3_2nd_ideal, label=f"PSD_mode_{mode_3}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_ideal, PSD_residual_mode_3_1st_ideal, label=f"PSD_mode_{mode_3}_1st_{label_ideal}")


    plt.plot(PSD_atm_mode_3_freq_t, PSD_atm_mode_3, color="black", label=f"atm_PSD_mode_{mode_3}")
    plt.title(f"residual PSD mode_{mode_3}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_4_freq_t_1st_RL[np.where(PSD_residual_mode_4_freq_t_1st_RL <= stage_1_freq_lim)]
                         , PSD_residual_mode_4_1st_RL[np.where(PSD_residual_mode_4_freq_t_1st_RL <= stage_1_freq_lim)], '--', color="indianred", label=f"PSD_mode_{mode_4}_1st_int")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_RL, PSD_residual_mode_4_2nd_RL, color="red", label=f"PSD_mode_{mode_4}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_RL, PSD_residual_mode_4_1st_RL, color="indianred", label=f"PSD_mode_{mode_4}_1st_{label_RL}")
    if integrator:
        if stage_2:
            plt.plot(PSD_residual_mode_4_freq_t_2nd_int, PSD_residual_mode_4_2nd_int, color="blue", label=f"PSD_mode_{mode_4}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_int, PSD_residual_mode_4_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_4}_1st_{label_int}")
    if ideal:
        if stage_2:
            plt.plot(PSD_residual_mode_4_freq_t_2nd_ideal, PSD_residual_mode_4_2nd_ideal, label=f"PSD_mode_{mode_4}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_ideal, PSD_residual_mode_4_1st_ideal, color="indianred", label=f"PSD_mode_{mode_4}_1st_{label_ideal}")

    plt.plot(PSD_atm_mode_4_freq_t, PSD_atm_mode_4, color="black", label=f"atm_PSD_mode_{mode_4}")
    plt.title(f"residual PSD mode_{mode_4}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_5_freq_t_1st_RL[np.where(PSD_residual_mode_5_freq_t_1st_RL <= stage_1_freq_lim)]
                         , PSD_residual_mode_5_1st_RL[np.where(PSD_residual_mode_5_freq_t_1st_RL <= stage_1_freq_lim)], '--', color="indianred", label=f"PSD_mode_{mode_5}_1st_int")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_RL, PSD_residual_mode_5_2nd_RL, color="red", label=f"PSD_mode_{mode_5}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_RL, PSD_residual_mode_5_1st_RL, color="indianred", label=f"PSD_mode_{mode_5}_1st_{label_RL}")
    if integrator:
        if stage_2:
            plt.plot(PSD_residual_mode_5_freq_t_2nd_int, PSD_residual_mode_5_2nd_int, color="blue", label=f"PSD_mode_{mode_5}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_int, PSD_residual_mode_5_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_5}_1st_{label_int}")
    if ideal:
        if stage_2:
            plt.plot(PSD_residual_mode_5_freq_t_2nd_ideal, PSD_residual_mode_5_2nd_ideal, label=f"PSD_mode_{mode_5}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_ideal, PSD_residual_mode_5_1st_ideal, label=f"PSD_mode_{mode_5}_1st_{label_ideal}")

    plt.plot(PSD_atm_mode_5_freq_t, PSD_atm_mode_5, color="black", label=f"atm_PSD_mode_{mode_5}")
    plt.title(f"residual PSD mode_{mode_5}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD")



#---------------------------------------------------Cumulative PSD---------------------------------------------------#



if plot_tPSD:
    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_RL[np.where(PSD_residual_mode_1_freq_t_1st_RL <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_1_1st_RL[np.where(PSD_residual_mode_1_freq_t_1st_RL <= stage_1_freq_lim)]), '--', color="indianred", label=f"PSD_mode_{mode_1}_1st_int")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_RL, np.cumsum(PSD_residual_mode_1_2nd_RL), color="red", label=f"PSD_mode_{mode_1}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_RL, np.cumsum(PSD_residual_mode_1_1st_RL), color="indianred", label=f"PSD_mode_{mode_1}_1st_{label_RL}")
    if integrator:
        if stage_2:
            plt.plot(PSD_residual_mode_1_freq_t_2nd_int, np.cumsum(PSD_residual_mode_1_2nd_int), color="blue", label=f"PSD_mode_{mode_1}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_int, np.cumsum(PSD_residual_mode_1_1st_int), color="cornflowerblue", label=f"PSD_mode_{mode_1}_1st_{label_int}")
    if ideal:
        if stage_2:
            plt.plot(PSD_residual_mode_1_freq_t_2nd_ideal, np.cumsum(PSD_residual_mode_1_2nd_ideal), label=f"PSD_mode_{mode_1}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_ideal, np.cumsum(PSD_residual_mode_1_1st_ideal), label=f"PSD_mode_{mode_1}_1st_{label_ideal}")



    plt.title(f"Cumulative PSD {mode_1}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_2_freq_t_1st_RL[np.where(PSD_residual_mode_2_freq_t_1st_RL <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_2_1st_RL[np.where(PSD_residual_mode_2_freq_t_1st_RL <= stage_1_freq_lim)]), '--', color="indianred", label=f"PSD_mode_{mode_2}_1st_int")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_RL, np.cumsum(PSD_residual_mode_2_2nd_RL), color="red", label=f"PSD_mode_{mode_2}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_RL, np.cumsum(PSD_residual_mode_2_1st_RL), color="indianred", label=f"PSD_mode_{mode_2}_1st_{label_RL}")
    if integrator:
        if stage_2:
            plt.plot(PSD_residual_mode_2_freq_t_2nd_int, np.cumsum(PSD_residual_mode_2_2nd_int), color="blue", label=f"PSD_mode_{mode_2}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_int, np.cumsum(PSD_residual_mode_2_1st_int), color="cornflowerblue", label=f"PSD_mode_{mode_2}_1st_{label_int}")
    if ideal:
        if stage_2:
            plt.plot(PSD_residual_mode_2_freq_t_2nd_ideal, np.cumsum(PSD_residual_mode_2_2nd_ideal), label=f"PSD_mode_{mode_2}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_ideal, np.cumsum(PSD_residual_mode_2_1st_ideal), label=f"PSD_mode_{mode_2}_1st_{label_ideal}")



    plt.title(f"Cumulative PSD {mode_2}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_3_freq_t_1st_RL[np.where(PSD_residual_mode_3_freq_t_1st_RL <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_3_1st_RL[np.where(PSD_residual_mode_3_freq_t_1st_RL <= stage_1_freq_lim)]), '--', color="indianred", label=f"PSD_mode_{mode_3}_1st_int")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_RL, np.cumsum(PSD_residual_mode_3_2nd_RL), color="red", label=f"PSD_mode_{mode_3}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_RL, np.cumsum(PSD_residual_mode_3_1st_RL), color="indianred", label=f"PSD_mode_{mode_3}_1st_{label_RL}")
    if integrator:
        if stage_2:
            plt.plot(PSD_residual_mode_3_freq_t_2nd_int, np.cumsum(PSD_residual_mode_3_2nd_int), color="blue", label=f"PSD_mode_{mode_3}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_int, np.cumsum(PSD_residual_mode_3_1st_int), color="cornflowerblue", label=f"PSD_mode_{mode_3}_1st_{label_int}")
    if ideal:
        if stage_2:
            plt.plot(PSD_residual_mode_3_freq_t_2nd_ideal, np.cumsum(PSD_residual_mode_3_2nd_ideal), label=f"PSD_mode_{mode_3}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_ideal, np.cumsum(PSD_residual_mode_3_1st_ideal), label=f"PSD_mode_{mode_3}_1st_{label_ideal}")



    plt.title(f"Cumulative PSD {mode_3}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_4_freq_t_1st_RL[np.where(PSD_residual_mode_4_freq_t_1st_RL <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_4_1st_RL[np.where(PSD_residual_mode_4_freq_t_1st_RL <= stage_1_freq_lim)]), '--', color="indianred", label=f"PSD_mode_{mode_4}_1st_int")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_RL, np.cumsum(PSD_residual_mode_4_2nd_RL), color="red", label=f"PSD_mode_{mode_4}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_RL, np.cumsum(PSD_residual_mode_4_1st_RL), color="indianred", label=f"PSD_mode_{mode_4}_1st_{label_RL}")
    if integrator:
        if stage_2:
            plt.plot(PSD_residual_mode_4_freq_t_2nd_int, np.cumsum(PSD_residual_mode_4_2nd_int), color="blue", label=f"PSD_mode_{mode_4}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_int, np.cumsum(PSD_residual_mode_4_1st_int), color="cornflowerblue", label=f"PSD_mode_{mode_4}_1st_{label_int}")
    if ideal:
        if stage_2:
            plt.plot(PSD_residual_mode_4_freq_t_2nd_ideal, np.cumsum(PSD_residual_mode_4_2nd_ideal), label=f"PSD_mode_{mode_4}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_ideal, np.cumsum(PSD_residual_mode_4_1st_ideal), color="indianred", label=f"PSD_mode_{mode_4}_1st_{label_ideal}")


    plt.title(f"Cumulative PSD {mode_4}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD")


    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_5_freq_t_1st_RL[np.where(PSD_residual_mode_5_freq_t_1st_RL <= stage_1_freq_lim)]
                         , np.cumsum(PSD_residual_mode_5_1st_RL[np.where(PSD_residual_mode_5_freq_t_1st_RL <= stage_1_freq_lim)]), '--', color="indianred", label=f"PSD_mode_{mode_5}_1st_int")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_RL, np.cumsum(PSD_residual_mode_5_2nd_RL), color="red", label=f"PSD_mode_{mode_5}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_RL, np.cumsum(PSD_residual_mode_5_1st_RL), color="indianred", label=f"PSD_mode_{mode_5}_1st_{label_RL}")
    if integrator:
        if stage_2:
            plt.plot(PSD_residual_mode_5_freq_t_2nd_int, np.cumsum(PSD_residual_mode_5_2nd_int), color="blue", label=f"PSD_mode_{mode_5}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_int, np.cumsum(PSD_residual_mode_5_1st_int), color="cornflowerblue", label=f"PSD_mode_{mode_5}_1st_{label_int}")
    if ideal:
        if stage_2:
            plt.plot(PSD_residual_mode_5_freq_t_2nd_ideal, np.cumsum(PSD_residual_mode_5_2nd_ideal), label=f"PSD_mode_{mode_5}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_ideal, np.cumsum(PSD_residual_mode_5_1st_ideal), label=f"PSD_mode_{mode_5}_1st_{label_ideal}")


    plt.title(f"Cumulative PSD {mode_5}, gain {CL_gain_pyr}")
    plt.xlabel("frequency (Hz)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.ylabel("PSD")







"""# --------------------------------------------------- Ratio ---------------------------------------------------#

if RL:
    PSD_mode_1_ratio_RL = PSD_residual_mode_1_1st_RL/PSD_residual_mode_1_2nd_RL
    PSD_mode_2_ratio_RL = PSD_residual_mode_2_1st_RL/PSD_residual_mode_2_2nd_RL
    PSD_mode_3_ratio_RL = PSD_residual_mode_3_1st_RL/PSD_residual_mode_3_2nd_RL
    PSD_mode_4_ratio_RL = PSD_residual_mode_4_1st_RL/PSD_residual_mode_4_2nd_RL
    PSD_mode_5_ratio_RL = PSD_residual_mode_5_1st_RL/PSD_residual_mode_5_2nd_RL

if integrator:
    PSD_mode_1_ratio_int = PSD_residual_mode_1_1st_int/PSD_residual_mode_1_2nd_int
    PSD_mode_2_ratio_int = PSD_residual_mode_2_1st_int/PSD_residual_mode_2_2nd_int
    PSD_mode_3_ratio_int = PSD_residual_mode_3_1st_int/PSD_residual_mode_3_2nd_int
    PSD_mode_4_ratio_int = PSD_residual_mode_4_1st_int/PSD_residual_mode_4_2nd_int
    PSD_mode_5_ratio_int = PSD_residual_mode_5_1st_int/PSD_residual_mode_5_2nd_int


if ideal:
    PSD_mode_1_ratio_ideal = PSD_residual_mode_1_1st_ideal/PSD_residual_mode_1_2nd_ideal
    PSD_mode_2_ratio_ideal = PSD_residual_mode_2_1st_ideal/PSD_residual_mode_2_2nd_ideal
    PSD_mode_3_ratio_ideal = PSD_residual_mode_3_1st_ideal/PSD_residual_mode_3_2nd_ideal
    PSD_mode_4_ratio_ideal = PSD_residual_mode_4_1st_ideal/PSD_residual_mode_4_2nd_ideal
    PSD_mode_5_ratio_ideal = PSD_residual_mode_5_1st_ideal/PSD_residual_mode_5_2nd_ideal



plt.figure()
if RL:
    if stage_2:
        plt.plot(PSD_residual_mode_1_freq_t_2nd_RL, PSD_mode_1_ratio_RL, color="red", label=f"PSD_mode_{mode_1}_2nd_{label_RL}")
    else: plt.plot(PSD_residual_mode_1_freq_t_1st_RL, PSD_residual_mode_1_1st_RL, color="indianred", label=f"PSD_mode_{mode_1}_1st_{label_RL}")
if integrator:
    if stage_2:
        plt.plot(PSD_residual_mode_1_freq_t_2nd_int, PSD_mode_1_ratio_int, color="blue", label=f"PSD_mode_{mode_1}_2nd_{label_int}")
    else: plt.plot(PSD_residual_mode_1_freq_t_1st_int, PSD_residual_mode_1_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_1}_1st_{label_int}")
if ideal:
    if stage_2:
        plt.plot(PSD_residual_mode_1_freq_t_2nd_ideal, PSD_mode_1_ratio_ideal, label=f"PSD_mode_{mode_1}_2nd_{label_ideal}")
    else: plt.plot(PSD_residual_mode_1_freq_t_1st_ideal, PSD_residual_mode_1_1st_ideal, label=f"PSD_mode_{mode_1}_1st_{label_ideal}")

plt.title(f"Gain of 2nd stage over 1st stage {mode_1}, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("ratio")



plt.figure()
if RL:
    if stage_2:
        plt.plot(PSD_residual_mode_2_freq_t_2nd_RL, PSD_mode_2_ratio_RL, color="red", label=f"PSD_mode_{mode_2}_2nd_{label_RL}")
    else: plt.plot(PSD_residual_mode_2_freq_t_1st_RL, PSD_residual_mode_2_1st_RL, color="indianred", label=f"PSD_mode_{mode_2}_1st_{label_RL}")
if integrator:
    if stage_2:
        plt.plot(PSD_residual_mode_2_freq_t_2nd_int, PSD_mode_2_ratio_int, color="blue", label=f"PSD_mode_{mode_2}_2nd_{label_int}")
    else: plt.plot(PSD_residual_mode_2_freq_t_1st_int, PSD_residual_mode_2_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_2}_1st_{label_int}")
if ideal:
    if stage_2:
        plt.plot(PSD_residual_mode_2_freq_t_2nd_ideal, PSD_mode_2_ratio_ideal, label=f"PSD_mode_{mode_2}_2nd_{label_ideal}")
    else: plt.plot(PSD_residual_mode_2_freq_t_1st_ideal, PSD_residual_mode_2_1st_ideal, label=f"PSD_mode_{mode_2}_1st_{label_ideal}")


plt.title(f"Gain of 2nd stage over 1st stage {mode_2}, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("ratio")



plt.figure()
if RL:
    if stage_2:
        plt.plot(PSD_residual_mode_3_freq_t_2nd_RL, PSD_mode_3_ratio_RL, color="red", label=f"PSD_mode_{mode_3}_2nd_{label_RL}")
    else: plt.plot(PSD_residual_mode_3_freq_t_1st_RL, PSD_residual_mode_3_1st_RL, color="indianred", label=f"PSD_mode_{mode_3}_1st_{label_RL}")
if integrator:
    if stage_2:
        plt.plot(PSD_residual_mode_3_freq_t_2nd_int, PSD_mode_3_ratio_int, color="blue", label=f"PSD_mode_{mode_3}_2nd_{label_int}")
    else: plt.plot(PSD_residual_mode_3_freq_t_1st_int, PSD_residual_mode_3_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_3}_1st_{label_int}")
if ideal:
    if stage_2:
        plt.plot(PSD_residual_mode_3_freq_t_2nd_ideal, PSD_mode_3_ratio_ideal, label=f"PSD_mode_{mode_3}_2nd_{label_ideal}")
    else: plt.plot(PSD_residual_mode_3_freq_t_1st_ideal, PSD_residual_mode_3_1st_ideal, label=f"PSD_mode_{mode_3}_1st_{label_ideal}")



plt.title(f"Gain of 2nd stage over 1st stage {mode_3}, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("ratio")





plt.figure()
if RL:
    if stage_2:
        plt.plot(PSD_residual_mode_4_freq_t_2nd_RL, PSD_mode_4_ratio_RL, color="red", label=f"PSD_mode_{mode_4}_2nd_{label_RL}")
    else: plt.plot(PSD_residual_mode_4_freq_t_1st_RL, PSD_residual_mode_4_1st_RL, color="indianred", label=f"PSD_mode_{mode_4}_1st_{label_RL}")
if integrator:
    if stage_2:
        plt.plot(PSD_residual_mode_4_freq_t_2nd_int, PSD_mode_4_ratio_int, color="blue", label=f"PSD_mode_{mode_4}_2nd_{label_int}")
    else: plt.plot(PSD_residual_mode_4_freq_t_1st_int, PSD_residual_mode_4_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_4}_1st_{label_int}")
if ideal:
    if stage_2:
        plt.plot(PSD_residual_mode_4_freq_t_2nd_ideal, PSD_mode_4_ratio_ideal, label=f"PSD_mode_{mode_4}_2nd_{label_ideal}")
    else: plt.plot(PSD_residual_mode_4_freq_t_1st_ideal, PSD_residual_mode_4_1st_ideal, color="indianred", label=f"PSD_mode_{mode_4}_1st_{label_ideal}")

plt.title(f"Gain of 2nd stage over 1st stage {mode_4}, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("ratio")



plt.figure()
if RL:
    if stage_2:
        plt.plot(PSD_residual_mode_5_freq_t_2nd_RL, PSD_mode_5_ratio_RL, color="red", label=f"PSD_mode_{mode_5}_2nd_{label_RL}")
    else: plt.plot(PSD_residual_mode_5_freq_t_1st_RL, PSD_residual_mode_5_1st_RL, color="indianred", label=f"PSD_mode_{mode_5}_1st_{label_RL}")
if integrator:
    if stage_2:
        plt.plot(PSD_residual_mode_5_freq_t_2nd_int, PSD_mode_5_ratio_int, color="blue", label=f"PSD_mode_{mode_5}_2nd_{label_int}")
    else: plt.plot(PSD_residual_mode_5_freq_t_1st_int, PSD_residual_mode_5_1st_int, color="cornflowerblue", label=f"PSD_mode_{mode_5}_1st_{label_int}")
if ideal:
    if stage_2:
        plt.plot(PSD_residual_mode_5_freq_t_2nd_ideal, PSD_mode_5_ratio_ideal, label=f"PSD_mode_{mode_5}_2nd_{label_ideal}")
    else: plt.plot(PSD_residual_mode_5_freq_t_1st_ideal, PSD_residual_mode_5_1st_ideal, label=f"PSD_mode_{mode_5}_1st_{label_ideal}")


plt.title(f"Gain of 2nd stage over 1st stage {mode_5}, gain {CL_gain_pyr}")
plt.xlabel("frequency (Hz)")
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which='both', alpha=0.5)
plt.minorticks_on()
plt.legend()
plt.ylabel("ratio")


plt.show()
"""

# ---------------------------------------------------temporal Error transfer function---------------------------------------------------#
"""
#to select for the common frequencies since the 1st stage OL was running at 200Hz on sky
idx_CL = np.isin(np.round(PSD_residual_mode_1_freq_t_2nd_RL, 6), np.round(PSD_atm_mode_1_freq_t, 6))
idx_atm = np.isin(np.round(PSD_atm_mode_1_freq_t, 6), np.round(PSD_residual_mode_1_freq_t_2nd_RL, 6))
f2_selected  = PSD_residual_mode_1_freq_t_2nd_RL[idx_CL]
psd2_selected = PSD_residual_mode_1_2nd_RL[idx_CL]

if plot_tPSD:
    if RL:
        tETF_mode_1_1st_RL = PSD_residual_mode_1_1st_RL[idx_CL] / PSD_atm_mode_1[idx_atm]
        tETF_mode_2_1st_RL = PSD_residual_mode_2_1st_RL[idx_CL] / PSD_atm_mode_2[idx_atm]
        tETF_mode_3_1st_RL = PSD_residual_mode_3_1st_RL[idx_CL] / PSD_atm_mode_3[idx_atm]
        tETF_mode_4_1st_RL = PSD_residual_mode_4_1st_RL[idx_CL] / PSD_atm_mode_4[idx_atm]
        tETF_mode_5_1st_RL = PSD_residual_mode_5_1st_RL[idx_CL] / PSD_atm_mode_5[idx_atm]

        if stage_2:
            tETF_mode_1_2nd_RL = PSD_residual_mode_1_2nd_RL[idx_CL] / PSD_atm_mode_1[idx_atm]
            tETF_mode_2_2nd_RL = PSD_residual_mode_2_2nd_RL[idx_CL] / PSD_atm_mode_2[idx_atm]
            tETF_mode_3_2nd_RL = PSD_residual_mode_3_2nd_RL[idx_CL] / PSD_atm_mode_3[idx_atm]
            tETF_mode_4_2nd_RL = PSD_residual_mode_4_2nd_RL[idx_CL] / PSD_atm_mode_4[idx_atm]
            tETF_mode_5_2nd_RL = PSD_residual_mode_5_2nd_RL[idx_CL] / PSD_atm_mode_5[idx_atm]


    if integrator:
        tETF_mode_1_1st_int = PSD_residual_mode_1_1st_int[idx_CL] / PSD_atm_mode_1[idx_atm]
        tETF_mode_2_1st_int = PSD_residual_mode_2_1st_int[idx_CL] / PSD_atm_mode_2[idx_atm]
        tETF_mode_3_1st_int = PSD_residual_mode_3_1st_int[idx_CL] / PSD_atm_mode_3[idx_atm]
        tETF_mode_4_1st_int = PSD_residual_mode_4_1st_int[idx_CL] / PSD_atm_mode_4[idx_atm]
        tETF_mode_5_1st_int = PSD_residual_mode_5_1st_int[idx_CL] / PSD_atm_mode_5[idx_atm]

        if stage_2:
            tETF_mode_1_2nd_int = PSD_residual_mode_1_2nd_int[idx_CL] / PSD_atm_mode_1[idx_atm]
            tETF_mode_2_2nd_int = PSD_residual_mode_2_2nd_int[idx_CL] / PSD_atm_mode_2[idx_atm]
            tETF_mode_3_2nd_int = PSD_residual_mode_3_2nd_int[idx_CL] / PSD_atm_mode_3[idx_atm]
            tETF_mode_4_2nd_int = PSD_residual_mode_4_2nd_int[idx_CL] / PSD_atm_mode_4[idx_atm]
            tETF_mode_5_2nd_int = PSD_residual_mode_5_2nd_int[idx_CL] / PSD_atm_mode_5[idx_atm]


    if ideal:
        tETF_mode_1_1st_ideal = PSD_residual_mode_1_1st_ideal[idx_CL] / PSD_atm_mode_1[idx_atm]
        tETF_mode_2_1st_ideal = PSD_residual_mode_2_1st_ideal[idx_CL] / PSD_atm_mode_2[idx_atm]
        tETF_mode_3_1st_ideal = PSD_residual_mode_3_1st_ideal[idx_CL] / PSD_atm_mode_3[idx_atm]
        tETF_mode_4_1st_ideal = PSD_residual_mode_4_1st_ideal[idx_CL] / PSD_atm_mode_4[idx_atm]
        tETF_mode_5_1st_ideal = PSD_residual_mode_5_1st_ideal[idx_CL] / PSD_atm_mode_5[idx_atm]

        if stage_2:
            tETF_mode_1_2nd_ideal = PSD_residual_mode_1_2nd_ideal[idx_CL] / PSD_atm_mode_1[idx_atm]
            tETF_mode_2_2nd_ideal = PSD_residual_mode_2_2nd_ideal[idx_CL] / PSD_atm_mode_2[idx_atm]
            tETF_mode_3_2nd_ideal = PSD_residual_mode_3_2nd_ideal[idx_CL] / PSD_atm_mode_3[idx_atm]
            tETF_mode_4_2nd_ideal = PSD_residual_mode_4_2nd_ideal[idx_CL] / PSD_atm_mode_4[idx_atm]
            tETF_mode_5_2nd_ideal = PSD_residual_mode_5_2nd_ideal[idx_CL] / PSD_atm_mode_5[idx_atm]



    #tip
    plt.figure()
    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_RL[idx_CL], tETF_mode_1_1st_RL, '--', color="indianred", label=f"ETF mode_{mode_1}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_RL[idx_CL], tETF_mode_1_2nd_RL, color="red", label=f"ETF mode_{mode_1}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_RL, tETF_mode_1_1st_RL, color="indianred", label=f"ETF mode_{mode_1}_1st_{label_RL}")

    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_int[idx_CL], tETF_mode_1_1st_int, '--', color="cornflowerblue", label=f"ETF mode_{mode_1}_1st_{label_int}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_int[idx_CL], tETF_mode_1_2nd_int, color="blue", label=f"ETF mode_{mode_1}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_int, tETF_mode_1_1st_int, color="cornflowerblue", label=f"ETF mode_{mode_1}_1st_{label_int}")


    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_1_freq_t_1st_ideal[idx_CL], tETF_mode_1_1st_ideal, '--', label=f"ETF mode_{mode_1}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_1_freq_t_2nd_ideal[idx_CL], tETF_mode_1_2nd_ideal, label=f"ETF mode_{mode_1}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_1_freq_t_1st_ideal, tETF_mode_1_1st_ideal, label=f"ETF mode_{mode_1}_1st_{label_ideal}")
    plt.title("temporal error transfer functions")
    plt.ylabel("ETF")
    plt.xlabel("frequency Hz")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(right=np.max(freq_lim))
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()


    #tilt
    plt.figure()


    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_2_freq_t_1st_RL[idx_CL], tETF_mode_2_1st_RL, '--', color="indianred", label=f"ETF mode_{mode_2}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_RL[idx_CL], tETF_mode_2_2nd_RL, color="red", label=f"ETF mode_{mode_2}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_RL, tETF_mode_2_1st_RL, color="indianred", label=f"ETF mode_{mode_2}_1st_{label_RL}")

    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_2_freq_t_1st_int[idx_CL], tETF_mode_2_1st_int, '--', color="cornflowerblue", label=f"ETF mode_{mode_2}_1st_{label_int}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_int[idx_CL], tETF_mode_2_2nd_int, color="blue", label=f"ETF mode_{mode_2}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_int, tETF_mode_2_1st_int, color="cornflowerblue", label=f"ETF mode_{mode_2}_1st_{label_int}")


    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_2_freq_t_1st_ideal[idx_CL], tETF_mode_2_1st_ideal, '--', label=f"ETF mode_{mode_2}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_2_freq_t_2nd_ideal[idx_CL], tETF_mode_2_2nd_ideal, label=f"ETF mode_{mode_2}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_2_freq_t_1st_ideal, tETF_mode_2_1st_ideal, label=f"ETF mode_{mode_2}_1st_{label_ideal}")
    plt.title("temporal error transfer functions")
    plt.ylabel("ETF")
    plt.xlabel("frequency Hz")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(right=np.max(freq_lim))
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()

    #100
    plt.figure()

    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_3_freq_t_1st_RL[idx_CL], tETF_mode_3_1st_RL, '--', color="indianred", label=f"ETF mode_{mode_3}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_RL[idx_CL], tETF_mode_3_2nd_RL, color="red", label=f"ETF mode_{mode_3}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_RL, tETF_mode_3_1st_RL, color="indianred", label=f"ETF mode_{mode_3}_1st_{label_RL}")


    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_3_freq_t_1st_int[idx_CL], tETF_mode_3_1st_int, '--', color="cornflowerblue", label=f"ETF mode_{mode_3}_1st_{label_int}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_int[idx_CL], tETF_mode_3_2nd_int, color="blue", label=f"ETF mode_{mode_3}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_int, tETF_mode_3_1st_int, color="cornflowerblue", label=f"ETF mode_{mode_3}_1st_{label_int}")


    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_3_freq_t_1st_ideal[idx_CL], tETF_mode_3_1st_ideal, '--', label=f"ETF mode_{mode_3}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_3_freq_t_2nd_ideal[idx_CL], tETF_mode_3_2nd_ideal, label=f"ETF mode_{mode_3}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_3_freq_t_1st_ideal, tETF_mode_3_1st_ideal, label=f"ETF mode_{mode_3}_1st_{label_ideal}")

    plt.title("temporal error transfer functions")
    plt.ylabel("ETF")
    plt.xlabel("frequency Hz")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(right=np.max(freq_lim))
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()

    #200
    plt.figure()

    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_4_freq_t_1st_RL[idx_CL], tETF_mode_4_1st_RL, '--', color="indianred", label=f"ETF mode_{mode_4}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_RL[idx_CL], tETF_mode_4_2nd_RL, color="red", label=f"ETF mode_{mode_4}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_RL, tETF_mode_4_1st_RL, color="indianred", label=f"ETF mode_{mode_4}_1st_{label_RL}")

    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_4_freq_t_1st_int[idx_CL], tETF_mode_4_1st_int, '--', color="cornflowerblue", label=f"ETF mode_{mode_4}_1st_{label_int}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_int[idx_CL], tETF_mode_4_2nd_int, color="blue", label=f"ETF mode_{mode_4}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_int, tETF_mode_4_1st_int, color="cornflowerblue", label=f"ETF mode_{mode_4}_1st_{label_int}")

    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_4_freq_t_1st_ideal[idx_CL], tETF_mode_4_1st_ideal, '--', label=f"ETF mode_{mode_4}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_4_freq_t_2nd_ideal[idx_CL], tETF_mode_4_2nd_ideal, label=f"ETF mode_{mode_4}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_4_freq_t_1st_ideal, tETF_mode_4_1st_ideal, label=f"ETF mode_{mode_4}_1st_{label_ideal}")

    plt.title("temporal error transfer functions")
    plt.ylabel("ETF")
    plt.xlabel("frequency Hz")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(right=np.max(freq_lim))
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()


    #5
    plt.figure()


    if RL:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_5_freq_t_1st_RL[idx_CL], tETF_mode_5_1st_RL, '--', color="indianred", label=f"ETF mode_{mode_5}_1st_{label_RL}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_RL[idx_CL], tETF_mode_5_2nd_RL, color="red", label=f"ETF mode_{mode_5}_2nd_{label_RL}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_RL, tETF_mode_5_1st_RL, color="indianred", label=f"ETF mode_{mode_5}_1st_{label_RL}")

    if integrator:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_5_freq_t_1st_int[idx_CL], tETF_mode_5_1st_int, '--', color="cornflowerblue", label=f"ETF mode_{mode_5}_1st_{label_int}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_int[idx_CL], tETF_mode_5_2nd_int, color="blue", label=f"ETF mode_{mode_5}_2nd_{label_int}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_int, tETF_mode_5_1st_int, color="cornflowerblue", label=f"ETF mode_{mode_5}_1st_{label_int}")

    if ideal:
        if stage_2:
            if stage_1_plus_stage_2:
                plt.plot(PSD_residual_mode_5_freq_t_1st_ideal[idx_CL], tETF_mode_5_1st_ideal, '--', label=f"ETF mode_{mode_5}_1st_{label_ideal}")
            plt.plot(PSD_residual_mode_5_freq_t_2nd_ideal[idx_CL], tETF_mode_5_2nd_ideal, label=f"ETF mode_{mode_5}_2nd_{label_ideal}")
        else: plt.plot(PSD_residual_mode_5_freq_t_1st_ideal, tETF_mode_5_1st_ideal, label=f"ETF mode_{mode_5}_1st_{label_ideal}")


    plt.title("temporal error transfer functions")
    plt.ylabel("ETF")
    plt.xlabel("frequency Hz")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(right=np.max(freq_lim))
    plt.grid(True, which='both', alpha=0.5)
    plt.minorticks_on()
    plt.legend()

"""


plt.show()





























