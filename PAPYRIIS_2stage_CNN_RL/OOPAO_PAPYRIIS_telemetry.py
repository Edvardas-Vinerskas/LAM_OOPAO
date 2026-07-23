"""
OOPAO PAPYRIIS two-stage telemetry — metric plotting
====================================================
Post-processing and visualisation for the two-stage PAPYRIIS AO *simulation*
results. Same philosophy as the on-sky OOPAO_metric_plotting_bench_telemetry.py:
each controller's residuals become labelled, coloured StageData blocks, and every
plot is one loop over (mode, stage). The shared 1st-stage (CL1OL2) recording is
loaded once. See Config for all paths and switches.
 
Produces:
  * RL training-loss curves  (dynamics + policy)
  * Strehl-ratio time series with running-average overlay
  * KL-coefficient variance spectra
  * Per-mode temporal PSDs  (raw + cumulative)
  * Per-mode temporal error transfer functions  (residual / atmosphere PSD)
  * Per-mode 2nd/1st PSD ratio  (optional)
  * Per-mode residual time series  (optional)
 
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

from functions import mode_covariance, tPSD_calculator, second_stage_dm_builder

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
    run_date:       str  = "~2026-06-23"
    run_name_rl:    str  = "PAPYRIIS_arcturus_noise_quantisation_pwfs_calibration_pupil_EMCCD"
    run_name_ideal: str  = "PAPYRIIS_arcturus_nonoise_quantisation_pwfs_calibration_pupil"
    atm_filename:   str  = ("generated_atm_2nd_stage/"
                             "atm_OPDs_2nd_r0_0.050_V0_4.049_L0_30.000_tboil_5.000_multi_layer.npz")
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
    plot_cumulative: bool = False
    plot_etf:        bool = False    # temporal error transfer function per mode
    plot_ratio:      bool = False   # 2nd/1st PSD gain per mode

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
    # nm^2 unit conversion (2nd-stage KL basis), shared by every stage/atmosphere:
    #   m2c            : modes -> 2nd-stage DM commands (M2C_KL)
    #   dm_modes_meters: 2nd-stage DM influence functions in METRES, pupil-masked
    m2c:             Optional[np.ndarray] = None
    dm_modes_meters: Optional[np.ndarray] = None
    nperseg:         int = 256
    _G:         Optional[np.ndarray] = field(default=None, repr=False)
    _psd_cache: dict = field(default_factory=dict, repr=False)

    @property
    def time_axis(self) -> np.ndarray:
        return np.arange(len(self.strehl)) / self.frequency

    def gram(self) -> np.ndarray:
        """Mode-covariance matrix G [nm^2]; G[i, i] rescales mode i into nm^2."""
        if self._G is None:
            m2p_nm = (self.dm_modes_meters * 1e9) @ self.m2c   # metres -> nm
            self._G = mode_covariance(m2p_nm)
        return self._G

    def variance(self) -> np.ndarray:
        """Per-mode residual variance in nm^2 (the KL-decomposition curve)."""
        return np.var(self.modes, axis=0) * np.diag(self.gram())

    def psd(self, mode_idx: int, lo: int = 0, hi: Optional[int] = None
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Temporal PSD of one mode in nm^2/Hz over modes[lo:hi] (cached).

        Delegates to tPSD_calculator, which scales the raw modal PSD by the Gram
        diagonal to convert arbitrary modal units into nm^2.
        """
        key = (mode_idx, lo, hi)
        if key not in self._psd_cache:
            seg = self.modes[lo:hi]
            f, psd_nm2, _ = tPSD_calculator(seg, mode_idx, np.diag(self.gram()),
                                            self.frequency, nperseg=self.nperseg)
            self._psd_cache[key] = (f, psd_nm2)
        return self._psd_cache[key]
 
 
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


def common_freq_indices(f_a: np.ndarray,
                        f_b: np.ndarray,
                        decimals: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean masks selecting the frequency bins shared by two Welch grids.

    The closed-loop and atmosphere PSDs may be sampled at different rates, so the
    ETF can only be formed on the frequencies they have in common.
    """
    idx_a = np.isin(np.round(f_a, decimals), np.round(f_b, decimals))
    idx_b = np.isin(np.round(f_b, decimals), np.round(f_a, decimals))
    return idx_a, idx_b
 

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
    return {k: raw[k][-n:] if raw[k].ndim >= 1 else raw[k] for k in raw.files}
 
 
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


def load_unit_conversion(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Return (m2c_2nd, dm_2nd_meters) used to rescale 2nd-stage KL modes to nm^2.

    m2c_2nd = M2C_KL (modes -> 2nd-stage DM commands); the DM influence functions
    come from second_stage_dm_builder(), reshaped to its pixel grid, pupil-masked,
    and left in METRES (tPSD_calculator / mode_covariance do the *1e9 to nm).
    Every stage and the atmosphere live in this same 87-mode basis, so one Gram
    serves them all (the analog of the bench file's single 1st-stage Gram).
    """
    m2c_2nd = np.load(cfg.base_dir / "M2C_KL.npy")             # (97, 87)
    inf_2nd, pupil = second_stage_dm_builder()                # (pixels, 97), (S, S)
    pupil = np.asarray(pupil).astype(bool)
    n_act = inf_2nd.shape[1]
    side  = int(round(inf_2nd.shape[0] ** 0.5))               # 100
    dm_2nd_meters = inf_2nd.reshape(side, side, n_act)[pupil, :]   # (n_pupil, 97)
    return m2c_2nd, dm_2nd_meters


def build_atmosphere(cfg: Config,
                     pupil_mask: np.ndarray,
                     projector_kl: np.ndarray,
                     m2c: np.ndarray,
                     dm_meters: np.ndarray) -> StageData:
    """Open-loop atmosphere wrapped as a StageData (so it shares the nm^2 path)."""
    modes = load_atmosphere_modes(cfg, pupil_mask, projector_kl)   # (T, n_modes)
    return StageData(strehl=np.zeros(len(modes)), modes=modes,
                     frequency=cfg.freq_atm, label="Atmosphere (open loop)",
                     color="black", linestyle="--",
                     m2c=m2c, dm_modes_meters=dm_meters, nperseg=cfg.welch_nperseg)


def _build_stage(npz: Dict[str, np.ndarray],
                 C2M_2nd: np.ndarray,
                 pupil_mask: np.ndarray,
                 cfg: Config,
                 frequency: float,
                 label: str,
                 color: str,
                 m2c: Optional[np.ndarray]       = None,
                 dm_meters: Optional[np.ndarray] = None,
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
                     label=label, color=color, linestyle=linestyle,
                     m2c=m2c, dm_modes_meters=dm_meters, nperseg=cfg.welch_nperseg)



def load_controllers(cfg: Config,
                     C2M_2nd: np.ndarray,
                     pupil_mask: np.ndarray,
                     m2c: np.ndarray,
                     dm_meters: np.ndarray) -> Dict[str, ControllerResult]:
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
        label      = "1st stage",
        color      = "indianred",
        m2c        = m2c,
        dm_meters  = dm_meters,
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
                                   m2c        = m2c,
                                   dm_meters  = dm_meters,
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
                                   m2c        = m2c,
                                   dm_meters  = dm_meters,
                                   strehl_key = "all_2nd_stage_strehl"),
        )
 
    if cfg.use_ideal:
        path = cfg.ideal_dir / "results_2nd_stage.npz"
        d    = _load_tail(path, cfg.n_frames)
        controllers["ideal"] = ControllerResult(
            name   = "No-noise RL",
            stage1 = stage1_shared,
            stage2 = _build_stage(d, C2M_2nd, pupil_mask, cfg,
                                   frequency  = cfg.freq_ideal,
                                   label      = "No-noise RL",
                                   color      = "darkgreen",
                                   m2c        = m2c,
                                   dm_meters  = dm_meters,
                                   strehl_key = "all_2nd_stage_strehl",
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
    """Apply consistent styling to an Axes object (matches the bench file's style)."""
    if title:  ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if log_x:  ax.set_xscale("log")
    if log_y:  ax.set_yscale("log")
    ax.grid(True, which="both" if (log_x or log_y) else "major", alpha=0.5)
    ax.minorticks_on()
    ax.legend()


def _shared_stage1(controllers: Dict[str, ControllerResult]) -> List[StageData]:
    """The distinct 1st-stage recordings, de-duplicated by identity.

    Every controller references the same shared CL1OL2 1st-stage data, so this
    normally returns a single StageData — letting callers draw the 1st stage once
    instead of redrawing the identical curve for every controller.
    """
    seen: set = set()
    unique: List[StageData] = []
    for ctrl in controllers.values():
        s1 = ctrl.stage1
        if s1 is not None and id(s1) not in seen:
            seen.add(id(s1))
            unique.append(s1)
    return unique



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
 
    fig, (ax1, ax2) = plt.subplots(1, 2)
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
    fig, ax = plt.subplots()
 
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
                     atm: StageData,
                     cfg: Config) -> None:
    """
    Log–log KL-mode variance spectrum for each controller.
 
    The open-loop atmospheric variance is shown as a black dashed reference.
    When cfg.overlay_stage1 is True, the shared 1st-stage variance is also
    overlaid (dashed) on the same axes.
    """
    fig, ax = plt.subplots()
    ax.plot(atm.variance(), color=atm.color, linestyle=atm.linestyle, label=atm.label)

    # shared 1st stage: drawn once (identical across controllers)
    if cfg.overlay_stage1:
        for s1 in _shared_stage1(controllers):
            ax.plot(s1.variance(),
                    color=s1.color, linestyle=s1.linestyle, label=s1.label)

    for ctrl in controllers.values():
        if ctrl.stage2 is not None:
            s2 = ctrl.stage2
            ax.plot(s2.variance(),
                    color=s2.color, linestyle=s2.linestyle,
                    label=f"2nd stage — {s2.label}")
 
    _style(ax,
           title="KL-coefficient variance spectrum",
           xlabel="KL mode index",
           ylabel="Variance  [nm$^2$]",
           log_x=True, log_y=True)
    fig.tight_layout()
 
 
def _psd_single_mode(controllers: Dict[str, ControllerResult],
                     atm: StageData,
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
    fig, ax = plt.subplots()
 
    # ── atmosphere reference ────────────────────────────────────────────────
    fa, pa = atm.psd(mode_idx)
    ax.plot(fa, np.cumsum(pa) if cumulative else pa,
            color=atm.color, linestyle="-", label=f"Atmosphere — mode {mode_idx}")
 
    # ── 2nd-stage, one curve per controller ─────────────────────────────────
    for ctrl in controllers.values():
        if ctrl.stage2 is not None:
            s2    = ctrl.stage2
            i0, i1 = find_stable_segment(s2.strehl, cfg.sr_threshold)
            try:
                f2, p2 = s2.psd(mode_idx, i0, i1)
                ax.plot(f2, np.cumsum(p2) if cumulative else p2,
                        color=s2.color, linestyle=s2.linestyle,
                        label=f"2nd — {s2.label}")
            except Exception as exc:
                warnings.warn(f"PSD failed [{ctrl.name} stage2 mode {mode_idx}]: {exc}")

    # ── shared 1st-stage: drawn once, dashed, clipped to stage1_freq_lim ─────
    if cfg.overlay_stage1:
        for s1 in _shared_stage1(controllers):
            i0, i1 = find_stable_segment(s1.strehl, cfg.sr_threshold)
            try:
                f1, p1 = s1.psd(mode_idx, i0, i1)
                mask   = f1 <= cfg.stage1_freq_lim
                ax.plot(f1[mask],
                        (np.cumsum(p1) if cumulative else p1)[mask],
                        color=s1.color, linestyle=s1.linestyle,
                        label=s1.label)
            except Exception as exc:
                warnings.warn(f"PSD failed [stage1 mode {mode_idx}]: {exc}")
 
    _style(ax,
           title=f"{kind} — mode {mode_idx}  (gain = {cfg.cl_gain_pyr})",
           xlabel="Frequency (Hz)",
           ylabel="Cumulative PSD  [nm$^2$]" if cumulative else "PSD  [nm$^2$/Hz]",
           log_x=True, log_y=True)
    fig.tight_layout()
 
 
def plot_psds(controllers: Dict[str, ControllerResult],
              atm: StageData,
              cfg: Config,
              cumulative: bool = False) -> None:
    """Generate one PSD figure per entry in cfg.plot_modes."""
    for mode_idx in cfg.plot_modes:
        _psd_single_mode(controllers, atm, mode_idx, cfg,
                         cumulative=cumulative)
 
 
def plot_timeseries(controllers: Dict[str, ControllerResult],
                    cfg: Config) -> None:
    """
    Raw residual time-series for each mode in cfg.plot_modes.
    One figure per mode; both stages overlaid when cfg.overlay_stage1 is True.
    """
    for mode_idx in cfg.plot_modes:
        fig, ax = plt.subplots()

        # shared 1st stage: drawn once
        if cfg.overlay_stage1:
            for s1 in _shared_stage1(controllers):
                ax.plot(s1.time_axis, s1.modes[:, mode_idx],
                        color=s1.color, linestyle=s1.linestyle,
                        alpha=0.85, label=s1.label)

        # 2nd stage, one curve per controller
        for ctrl in controllers.values():
            if ctrl.stage2 is None:
                continue
            s2 = ctrl.stage2
            ax.plot(s2.time_axis, s2.modes[:, mode_idx],
                    color=s2.color, linestyle=s2.linestyle,
                    alpha=0.85, label=s2.label)
 
        _style(ax,
               title=f"Residual time series — mode {mode_idx}",
               xlabel="Time (s)",
               ylabel=f"KL mode {mode_idx}  coefficient")
        fig.tight_layout()


def _etf_single_mode(controllers: Dict[str, ControllerResult],
                     atm: StageData,
                     mode_idx: int,
                     cfg: Config) -> None:
    """
    Temporal error transfer function for one KL mode: residual PSD divided by the
    open-loop atmospheric PSD, on the frequencies the two grids share.

    2nd-stage is solid; the shared 1st-stage is dashed (when cfg.overlay_stage1).
    The PSD uses the longest segment with SR >= cfg.sr_threshold (full series if 0).
    """
    fig, ax = plt.subplots()
    fa, pa = atm.psd(mode_idx)

    def _draw_etf(sd: StageData, label: str) -> None:
        i0, i1 = find_stable_segment(sd.strehl, cfg.sr_threshold)
        try:
            f, p = sd.psd(mode_idx, i0, i1)
            idx_cl, idx_atm = common_freq_indices(f, fa)
            ax.plot(f[idx_cl], p[idx_cl] / pa[idx_atm],
                    color=sd.color, linestyle=sd.linestyle, label=label)
        except Exception as exc:
            warnings.warn(f"ETF failed [{label} mode {mode_idx}]: {exc}")

    # 2nd stage, one curve per controller
    for ctrl in controllers.values():
        if ctrl.stage2 is not None:
            _draw_etf(ctrl.stage2, f"2nd — {ctrl.stage2.label}")

    # shared 1st stage, drawn once
    if cfg.overlay_stage1:
        for s1 in _shared_stage1(controllers):
            _draw_etf(s1, s1.label)

    _style(ax,
           title=f"Temporal error transfer function — mode {mode_idx}",
           xlabel="Frequency (Hz)",
           ylabel="ETF",
           log_x=True, log_y=True)
    ax.set_xlim(right=cfg.freq_lim)
    fig.tight_layout()


def plot_etf(controllers: Dict[str, ControllerResult],
             atm: StageData,
             cfg: Config) -> None:
    """One ETF figure per entry in cfg.plot_modes."""
    for mode_idx in cfg.plot_modes:
        _etf_single_mode(controllers, atm, mode_idx, cfg)


def _ratio_single_mode(controllers: Dict[str, ControllerResult],
                       mode_idx: int,
                       cfg: Config) -> None:
    """
    2nd-stage / 1st-stage PSD ratio for one KL mode — the extra rejection the 2nd
    stage buys on top of the 1st. Requires both stages.
    """
    fig, ax = plt.subplots()

    for ctrl in controllers.values():
        if ctrl.stage1 is None or ctrl.stage2 is None:
            continue
        j0, j1 = find_stable_segment(ctrl.stage1.strehl, cfg.sr_threshold)
        i0, i1 = find_stable_segment(ctrl.stage2.strehl, cfg.sr_threshold)
        try:
            _,  p1 = ctrl.stage1.psd(mode_idx, j0, j1)
            f2, p2 = ctrl.stage2.psd(mode_idx, i0, i1)
            ax.plot(f2, p1 / p2, color=ctrl.stage2.color, label=ctrl.name)
        except Exception as exc:
            warnings.warn(f"Ratio failed [{ctrl.name} mode {mode_idx}]: {exc}")

    _style(ax,
           title=f"2nd/1st-stage gain — mode {mode_idx}",
           xlabel="Frequency (Hz)",
           ylabel="PSD ratio",
           log_x=True, log_y=True)
    fig.tight_layout()


def plot_ratio(controllers: Dict[str, ControllerResult],
               cfg: Config) -> None:
    """One 2nd/1st gain figure per entry in cfg.plot_modes."""
    for mode_idx in cfg.plot_modes:
        _ratio_single_mode(controllers, mode_idx, cfg)
 
 
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

    # ── nm^2 unit conversion (2nd-stage Gram inputs, shared by all traces) ──
    m2c_2nd, dm_2nd_meters = load_unit_conversion(cfg)

    # ── open-loop atmosphere (wrapped as a StageData) ───────────────────────
    atm = build_atmosphere(cfg, pupil_mask, projector_kl, m2c_2nd, dm_2nd_meters)

    # ── controller results ──────────────────────────────────────────────────
    controllers = load_controllers(cfg, C2M_2nd, pupil_mask,
                                   m2c_2nd, dm_2nd_meters)
 
    # ── plots ───────────────────────────────────────────────────────────────
    if cfg.use_rl:
        plot_training_losses(cfg)
 
    plot_strehl(controllers, cfg)
    plot_kl_variance(controllers, atm, cfg)

    if cfg.plot_tpsd:
        plot_psds(controllers, atm, cfg, cumulative=False)

    if cfg.plot_cumulative:
        plot_psds(controllers, atm, cfg, cumulative=True)

    if cfg.plot_tpsd and cfg.plot_etf:
        plot_etf(controllers, atm, cfg)

    if cfg.plot_ratio:
        plot_ratio(controllers, cfg)

    if cfg.plot_timeseries:
        plot_timeseries(controllers, cfg)
 
    plt.show()
 

if __name__ == "__main__":
    main()

