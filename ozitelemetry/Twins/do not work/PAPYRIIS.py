# -*- coding: utf-8 -*-
"""
PAPYRIIS.py

General two-stage PAPY--OZI simulation driver.

This module intentionally does not modify Papyrus, OZIRIIS, or the existing
parallel utilities. It wraps the current cell-based workflow into a reusable
class and stores all relevant intermediate products as attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import tqdm

try:
    from pymatreader import read_mat
except Exception:  # pragma: no cover - optional dependency at import time
    read_mat = None

from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix


ArrayLike = Union[np.ndarray, Sequence[float]]


def _import_local_classes():
    """Import local project classes without imposing a single folder layout."""
    try:
        from Twins.Papyrus import Papyrus
        from Twins.OZIRIIS import OZIRIIS
    except Exception:
        from Papyrus import Papyrus
        from OZIRIIS import OZIRIIS
    return Papyrus, OZIRIIS


def _as_bool_pupil(pupil: np.ndarray) -> np.ndarray:
    """Return a boolean pupil mask."""
    return np.asarray(pupil).astype(bool)


def _demean_columns(a: np.ndarray) -> np.ndarray:
    """Remove the temporal mean from columns stored as ``(n_modes, n_frames)``."""
    a = np.asarray(a)
    return a - np.nanmean(a, axis=0, keepdims=True)


@dataclass
class FirstStageLoopConfig:
    """Configuration used by the first-stage closed loop."""

    nLoop: Optional[int] = None
    gainCL: float = 0.5
    leak: float = 0.995
    frame_delay: int = 2
    photon_noise: bool = False
    progress: bool = True


@dataclass
class SecondStageLoopConfig:
    """Configuration used by the second-stage closed loop."""

    nLoop: Optional[int] = None
    nmodes: Optional[int] = None
    gainCL_2nd: float = 0.0
    leak_2nd: float = 0.98
    frame_delay_2nd: int = 2
    progress: bool = True


class PAPYRIIS:
    """
    Driver class for a two-stage PAPYRUS--OZIRIIS simulation.

    The class mirrors the original script logic while exposing explicit methods
    for initialization, calibration, OPD generation/projection, control loops,
    modal projections, and PSD computations.

    Notes
    -----
    The current OZIRIIS implementation accepts ``is_onsky`` but internally
    forces ``self.is_onsky = True``. This driver does not patch OZIRIIS; it only
    forwards the argument and keeps the resulting object state.
    """

    def __init__(
        self,
        *,
        param: Optional[dict] = None,
        dtype: Any = np.float32,
        auto_init_first_stage: bool = True,
        auto_init_second_stage: bool = True,
        first_stage_calibration_pupil: bool = True,
        first_stage_sky_offset: Sequence[int] = (2, 2),
        second_stage_is_onsky: bool = True,
        second_stage_controlled_modes: int = 35,
    ) -> None:
        self.dtype = dtype
        self.param = param

        self.Papytwin = None
        self.OZItwin = None

        # First-stage OOPAO objects.
        self.tel = None
        self.ngs = None
        self.dm = None
        self.wfs = None
        self.atm = None
        self.slow_tt = None
        self.src = None

        # Second-stage shortcuts.
        self.tel_2nd = None
        self.src_2nd = None
        self.dm_2nd = None
        self.atm_2nd = None
        self.vzwfs = None
        self.zwfs1 = None
        self.zwfs2 = None

        # Calibration products.
        self.M2C_1rst = None
        self.M2C_CL = None
        self.valid_pixel = None
        self.valid_pixel_binned = None
        self.int_mat_1rst = None
        self.int_mat_binned = None
        self.calib_1rst = None
        self.calib_1rst_M = None
        self.reconstructor_1rst = None
        self.first_stage_calibration_params: Dict[str, Any] = {}

        self.IM_z1 = None
        self.IM_z2 = None
        self.calib_2nd = None
        self.calib_2nd_M = None
        self.reconstructor_2nd = None
        self.second_stage_calibration_params: Dict[str, Any] = {}

        # OPD and loop products.
        self.atm_OPDs_2nd = None
        self.atm_OPDs_1rst = None
        self.first_stage_results: Dict[str, Any] = {}
        self.second_stage_results: Dict[str, Any] = {}

        # Modal and temporal products.
        self.modal: Dict[str, Any] = {}
        self.modal_psd: Dict[str, np.ndarray] = {}
        self.temporal_psd: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        if auto_init_first_stage:
            self.initialize_first_stage(
                calibration_pupil=first_stage_calibration_pupil,
                sky_offset=first_stage_sky_offset,
            )

        if auto_init_second_stage:
            self.initialize_second_stage(
                is_onsky=second_stage_is_onsky,
                controlled_modes=second_stage_controlled_modes,
            )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize_first_stage(
        self,
        *,
        calibration_pupil: bool = True,
        sky_offset: Sequence[int] = (2, 2),
    ) -> "PAPYRIIS":
        """
        Initialize the PAPYRUS first-stage model and store useful objects.

        Parameters
        ----------
        calibration_pupil : bool
            If True, use the calibration pupil. If False, use the sky/T152 pupil.
        sky_offset : sequence of int
            Offset passed to ``Papyrus.set_pupil`` when selecting the sky pupil.
        """
        Papyrus, _ = _import_local_classes()
        self.Papytwin = Papyrus()

        self.tel = self.Papytwin.tel
        self.ngs = self.Papytwin.ngs
        self.dm = self.Papytwin.dm
        self.wfs = self.Papytwin.wfs
        self.atm = self.Papytwin.atm
        self.slow_tt = self.Papytwin.slow_tt
        self.param = self.Papytwin.param if self.param is None else self.param

        self.Papytwin.set_pupil(
            calibration=calibration_pupil,
            sky_offset=list(sky_offset),
        )
        return self

    def initialize_second_stage(
        self,
        *,
        is_onsky: bool = True,
        controlled_modes: int = 35,
    ) -> "PAPYRIIS":
        """
        Initialize the OZIRIIS second-stage model and store useful objects.

        Parameters
        ----------
        is_onsky : bool
            Forwarded to ``OZIRIIS``. The current OZIRIIS class may override it.
        controlled_modes : int
            Forwarded to ``OZIRIIS``.
        """
        _, OZIRIIS = _import_local_classes()
        self.OZItwin = OZIRIIS(
            is_onsky=is_onsky,
            param=self.param,
            controlled_modes=controlled_modes,
        )

        self.tel_2nd = self.OZItwin.tel
        self.src_2nd = self.OZItwin.src
        self.dm_2nd = self.OZItwin.dm
        self.atm_2nd = self.OZItwin.atm
        self.vzwfs = self.OZItwin.vzwfs
        self.zwfs1 = self.OZItwin.zwfs1
        self.zwfs2 = self.OZItwin.zwfs2
        return self

    def set_first_stage_pupil(
        self,
        *,
        calibration: bool,
        sky_offset: Sequence[int] = (2, 2),
    ) -> "PAPYRIIS":
        """Switch the first-stage pupil using ``Papyrus.set_pupil``."""
        self._require_first_stage()
        self.Papytwin.set_pupil(calibration=calibration, sky_offset=list(sky_offset))
        return self

    def initialize_first_stage_propagation(self) -> "PAPYRIIS":
        """Propagate the first-stage NGS to the PWFS once after pupil selection."""
        self._require_first_stage()
        self.ngs * self.tel * self.wfs
        return self

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def load_first_stage_inputs(
        self,
        *,
        M2C_path: Optional[Union[str, Path]] = None,
        M2C: Optional[np.ndarray] = None,
        valid_pixel_path: Optional[Union[str, Path]] = None,
        valid_pixel: Optional[np.ndarray] = None,
        interaction_matrix_path: Optional[Union[str, Path]] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        mat_key: str = "matrix_inf",
        bin_bench_data: bool = True,
    ) -> "PAPYRIIS":
        """
        Load or assign first-stage external calibration inputs.

        Paths are explicit on purpose; no hard-coded bench path is used here.
        """
        self._require_first_stage()

        if M2C is None and M2C_path is not None:
            M2C = np.load(Path(M2C_path))
        if valid_pixel is None and valid_pixel_path is not None:
            valid_pixel = np.load(Path(valid_pixel_path))
        if interaction_matrix is None and interaction_matrix_path is not None:
            interaction_matrix = self._load_interaction_matrix_from_file(
                interaction_matrix_path,
                mat_key=mat_key,
            )

        if M2C is not None:
            self.M2C_1rst = np.asarray(M2C)
        if valid_pixel is not None:
            self.valid_pixel = np.asarray(valid_pixel)
        if interaction_matrix is not None:
            self.int_mat_1rst = np.asarray(interaction_matrix)

        if (
            bin_bench_data
            and self.valid_pixel is not None
            and self.int_mat_1rst is not None
        ):
            self.valid_pixel_binned, self.int_mat_binned = self.Papytwin.bin_bench_data(
                valid_pixel=self.valid_pixel,
                full_int_mat=self.int_mat_1rst,
                ratio=self.param["ratio"],
            )

        return self

    def check_first_stage_pwfs_pupils(
        self,
        *,
        valid_pixel_map: Optional[np.ndarray] = None,
        threshold: float = 0.005,
        correct: bool = True,
        n_it: int = 6,
    ) -> "PAPYRIIS":
        """
        Compare/correct PWFS pupil positions using the same logic as the script.

        If ``valid_pixel_map`` is not provided, it is inferred from the variance
        of the loaded full interaction matrix.
        """
        self._require_first_stage()
        if valid_pixel_map is None:
            if self.int_mat_1rst is None:
                raise ValueError("Provide valid_pixel_map or load int_mat_1rst first.")
            valid_pixel_map = np.var(self.int_mat_1rst, axis=1).reshape(240, 240)
            valid_pixel_map = valid_pixel_map / np.nanmax(valid_pixel_map)
            valid_pixel_map = valid_pixel_map > threshold

        self.Papytwin.check_pwfs_pupils(
            valid_pixel_map=valid_pixel_map,
            correct=correct,
            n_it=n_it,
        )
        return self

    def calibrate_first_stage(
        self,
        *,
        end_mode: int = 195,
        M2C: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        use_binned_interaction_matrix: bool = False,
        compute_synthetic_IM: bool = False,
        stroke: float = 1e-4,
        display: bool = True,
        switch_to_sky: bool = True,
        sky_offset: Sequence[int] = (2, 2),
    ) -> "PAPYRIIS":
        """
        Build the first-stage calibration vault and reconstructor.

        By default this reproduces the script path:
        ``CalibrationVault(im[:, :end_mode])`` followed by
        ``M2C[:, :end_mode] @ calib.M``.
        """
        self._require_first_stage()

        if M2C is not None:
            self.M2C_1rst = np.asarray(M2C)
        if interaction_matrix is not None:
            self.int_mat_1rst = np.asarray(interaction_matrix)

        if self.M2C_1rst is None:
            raise ValueError("First-stage M2C is missing. Call load_first_stage_inputs.")
        int_mat = self.int_mat_binned if use_binned_interaction_matrix else self.int_mat_1rst

        self.M2C_CL = self.M2C_1rst[:, :end_mode]

        if compute_synthetic_IM:
            self.wfs.modulation = self.param.get("modulation", 5)
            self.calib_1rst = InteractionMatrix(
                ngs=self.ngs,
                atm=self.atm,
                tel=self.tel,
                dm=self.dm,
                wfs=self.wfs,
                M2C=self.M2C_1rst,
                stroke=stroke,
                phaseOffset=0,
                nMeasurements=1,
                noise="off",
                print_time=False,
                display=display,
            )
            self.calib_1rst_M = getattr(self.calib_1rst, "M", None)
            if self.calib_1rst_M is None:
                raise AttributeError("Synthetic InteractionMatrix did not expose .M.")
            self.reconstructor_1rst = self.M2C_CL @ self.calib_1rst_M[:end_mode, :]
        else:
            if int_mat is None:
                raise ValueError("First-stage interaction matrix is missing.")
            self.calib_1rst = CalibrationVault(int_mat[:, :end_mode])
            self.calib_1rst_M = self.calib_1rst.M
            self.reconstructor_1rst = self.M2C_CL @ self.calib_1rst_M

        self.first_stage_calibration_params = {
            "end_mode": end_mode,
            "stroke": stroke,
            "compute_synthetic_IM": compute_synthetic_IM,
            "use_binned_interaction_matrix": use_binned_interaction_matrix,
        }

        if switch_to_sky:
            self.Papytwin.set_pupil(calibration=False, sky_offset=list(sky_offset))

        return self

    def calibrate_second_stage(
        self,
        *,
        nmodes: int = 35,
        stroke_nm: float = 12,
        use_zwfs: int = 1,
    ) -> "PAPYRIIS":
        """
        Compute the synthetic second-stage IM and build its reconstructor.

        Parameters
        ----------
        nmodes : int
            Number of modes retained in the second-stage reconstructor.
        stroke_nm : float
            Stroke passed to ``OZItwin.compute_synth_IM``.
        use_zwfs : int
            Use 1 for ``IM_z1`` or 2 for ``IM_z2``.
        """
        self._require_second_stage()

        self.IM_z1, self.IM_z2 = self.OZItwin.compute_synth_IM(stroke_nm=stroke_nm)
        IM = self.IM_z1 if use_zwfs == 1 else self.IM_z2

        self.calib_2nd = CalibrationVault(IM[:, :nmodes])
        self.calib_2nd_M = self.calib_2nd.M
        self.reconstructor_2nd = self.OZItwin.M2C[:, :nmodes] @ self.calib_2nd_M
        self.second_stage_calibration_params = {
            "nmodes": nmodes,
            "stroke_nm": stroke_nm,
            "use_zwfs": use_zwfs,
        }
        return self

    # ------------------------------------------------------------------
    # OPD generation and pupil projection
    # ------------------------------------------------------------------
    def generate_second_stage_atmosphere(
        self,
        *,
        nLoop: int,
        seed: Optional[int] = 15,
        use_no_pupil: bool = True,
        progress: bool = True,
    ) -> np.ndarray:
        """
        Generate a second-stage atmospheric OPD cube.

        The generated cube is stored in ``self.atm_OPDs_2nd``.
        """
        self._require_second_stage()
        self.nLoop = nLoop
        self.OZItwin.atm.initializeAtmosphere(self.OZItwin.tel)
        if seed is not None:
            self.OZItwin.atm.generateNewPhaseScreen(seed=seed)
        self.OZItwin.atm * self.OZItwin.tel

        sample = self.OZItwin.atm.OPD_no_pupil if use_no_pupil else self.OZItwin.atm.OPD
        out = np.zeros((nLoop, sample.shape[0], sample.shape[1]), dtype=self.dtype)

        iterator = range(nLoop)
        if progress:
            iterator = tqdm.tqdm(iterator, desc="Generate second-stage atmosphere")

        for i in iterator:
            self.OZItwin.atm.update()
            out[i] = (
                self.OZItwin.atm.OPD_no_pupil.copy()
                if use_no_pupil
                else self.OZItwin.atm.OPD.copy()
            )

        self.atm_OPDs_2nd = out
        return out

    def project_atmosphere_to_first_stage(self) -> np.ndarray:
        """
        Project ``self.atm_OPDs_2nd`` into the first-stage pupil.

        The result is stored in ``self.atm_OPDs_1rst``.
        """
        self._require_first_stage()
        self._require_second_stage()
        if self.atm_OPDs_2nd is None:
            raise ValueError("Generate or assign self.atm_OPDs_2nd first.")

        self.atm_OPDs_1rst = self.project_opd_between_pupils(
            self.atm_OPDs_2nd,
            input_pupil=None,
            output_pupil=self.tel.pupil,
            output_shape=self.tel.pupil.shape,
        )
        return self.atm_OPDs_1rst

    def inject_atmosphere_opds(
        self,
        *,
        atm_OPDs_2nd: Optional[np.ndarray] = None,
        atm_OPDs_1rst: Optional[np.ndarray] = None,
        project_missing: bool = True,
    ) -> "PAPYRIIS":
        """Inject externally generated OPD cubes."""
        if atm_OPDs_2nd is not None:
            self.atm_OPDs_2nd = np.asarray(atm_OPDs_2nd, dtype=self.dtype)
        if atm_OPDs_1rst is not None:
            self.atm_OPDs_1rst = np.asarray(atm_OPDs_1rst, dtype=self.dtype)
        elif project_missing and self.atm_OPDs_2nd is not None:
            self.project_atmosphere_to_first_stage()
        return self

    def project_opd_between_pupils(
        self,
        OPDs: np.ndarray,
        *,
        input_pupil: Optional[np.ndarray] = None,
        output_pupil: Optional[np.ndarray] = None,
        output_shape: Optional[Tuple[int, int]] = None,
        dtype: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Crop an OPD cube on an input pupil, rescale it, and inject it into
        an output pupil.

        This is the class version of ``project_opd_between_pupils`` from the
        script. It deliberately relies on OZIRIIS crop/rescale utilities.
        """
        self._require_second_stage()

        if dtype is None:
            dtype = self.dtype

        OPDs = np.asarray(OPDs)
        if output_pupil is None:
            output_pupil = self.OZItwin.tel.pupil
        output_pupil = _as_bool_pupil(output_pupil)

        if output_shape is None:
            output_shape = output_pupil.shape

        if input_pupil is None:
            _, opd_cropped = self.OZItwin._crop_opd(OPDs)
        else:
            _, opd_cropped = self.OZItwin._crop_opd(OPDs, _as_bool_pupil(input_pupil))

        pupil_cropped, _ = self.OZItwin._crop_pupil(output_pupil)
        pupil_cropped = _as_bool_pupil(pupil_cropped)

        opd_rescaled = self.OZItwin._rescale_matrix(
            opd_cropped,
            pupil_cropped.shape[0],
            pupil_cropped.shape[1],
        )

        if OPDs.ndim == 3:
            out = np.zeros((OPDs.shape[0], output_shape[0], output_shape[1]), dtype=dtype)
            out[:, output_pupil] = opd_rescaled[:, pupil_cropped]
        elif OPDs.ndim == 2:
            out = np.zeros(output_shape, dtype=dtype)
            out[output_pupil] = opd_rescaled[pupil_cropped]
        else:
            raise ValueError(f"OPDs must be 2D or 3D, got ndim={OPDs.ndim}.")

        return out

    # ------------------------------------------------------------------
    # First-stage loop
    # ------------------------------------------------------------------
    def run_first_stage_loop(
        self,
        *,
        nLoop: Optional[int] = None,
        gainCL: float = 0.5,
        leak: float =1,
        frame_delay: int = 2,
        photon_noise: bool = False,
        progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the first-stage loop and store the result dictionary.

        
        """
        self._require_first_stage()
        self._require_second_stage()
        if self.reconstructor_1rst is None:
            raise ValueError("First-stage reconstructor is missing.")
        if self.atm_OPDs_1rst is None:
            raise ValueError("First-stage atmospheric OPDs are missing.")

        if nLoop is None:
            nLoop = self.atm_OPDs_1rst.shape[0]

        self._prepare_first_stage_science_source()

        self.atm.initializeAtmosphere(self.tel)
        self.tel.resetOPD()
        self.dm.coefs = 0
        self.ngs * self.tel * self.dm * self.wfs
        self.wfs * self.wfs.focal_plane_camera
        self.atm * self.ngs * self.tel
        self.atm * self.src * self.tel

        ratio_samp = self._sampling_ratio()
        update_mask = self._build_sampling_mask(nLoop, ratio_samp)

        self.wfs.cam.photonNoise = photon_noise

        total = np.zeros(nLoop, dtype=self.dtype)
        residual_NGS = np.zeros(nLoop, dtype=self.dtype)
        residual_SRC = np.zeros(nLoop, dtype=self.dtype)
        dm_commands = np.zeros((nLoop, self.dm.nValidAct), dtype=self.dtype)
        reconstructed_cmd = np.zeros((nLoop, self.dm.nValidAct), dtype=self.dtype)
        dm_opds = np.zeros((nLoop, self.tel.pupil.shape[0], self.tel.pupil.shape[1]), dtype=self.dtype)
        src_opds = np.zeros_like(dm_opds)
        wfs_signal_history = np.zeros((nLoop, self.wfs.nSignal), dtype=self.dtype)

        wfsSignal = np.arange(0, self.wfs.nSignal) * 0

        iterator = range(nLoop)
        

        for i in iterator:
            self.atm.update(self.atm_OPDs_1rst[i])

            total[i] = np.std(self.tel.OPD[np.where(self.tel.pupil > 0)]) * 1e9

            self.atm * self.ngs * self.tel * self.dm * self.slow_tt * self.wfs
            dm_opds[i] = self.dm.OPD.copy()
            self.wfs * self.wfs.focal_plane_camera
            residual_NGS[i] = np.std(self.tel.OPD[np.where(self.tel.pupil > 0)]) * 1e9

            self.atm * self.src * self.tel * self.dm * self.slow_tt
            dm_commands[i, :] = self.dm.coefs.copy()
            residual_SRC[i] = np.std(self.tel.OPD[np.where(self.tel.pupil > 0)]) * 1e9
            src_opds[i] = self.tel.mean_removed_OPD.copy()

            if update_mask[i]:
                if frame_delay == 1:
                    wfsSignal = self.wfs.signal.copy()

                reconstructed_cmd[i] = np.matmul(self.reconstructor_1rst, wfsSignal)

                self.dm.coefs = leak * self.dm.coefs - gainCL * reconstructed_cmd[i]
                
                if frame_delay == 2:
                    wfsSignal = self.wfs.signal.copy()

            wfs_signal_history[i] = wfsSignal
            print(
            f'\rLoop {i+1}/{nLoop} '
            f'NGS: {residual_NGS[i]:.3f} '
            f'-- SRC: {residual_SRC[i]:.3f}',
            end='',
            flush=True
            )
        residuals_opds_1rst = (
            self.project_opd_between_pupils(
                dm_opds,
                input_pupil=self.tel.pupil,
                output_pupil=self.OZItwin.tel.pupil,
                output_shape=self.OZItwin.tel.pupil.shape,
            )
            + self.atm_OPDs_2nd[:nLoop]
        )

        self.first_stage_results = {
            "config": FirstStageLoopConfig(
                nLoop=nLoop,
                gainCL=gainCL,
                leak=leak,
                frame_delay=frame_delay,
                photon_noise=photon_noise,
                progress=progress,
            ),
            "ratio_samp": ratio_samp,
            "update_mask": update_mask,
            "total": total,
            "residual_NGS": residual_NGS,
            "residual_SRC": residual_SRC,
            "dm_commands": dm_commands,
            "reconstructed_cmd": reconstructed_cmd,
            "dm_opds": dm_opds,
            "src_opds": src_opds,
            "wfs_signal_history": wfs_signal_history,
            "residuals_opds_1rst": residuals_opds_1rst,
        }
        self.residuals_opds_1rst = residuals_opds_1rst
        return self.first_stage_results

    # ------------------------------------------------------------------
    # Second-stage loop
    # ------------------------------------------------------------------
    def run_second_stage_loop(
        self,
        *,
        nLoop: Optional[int] = None,
        nmodes: Optional[int] = None,
        gainCL_2nd: float = 0.0,
        leak_2nd: float = 0.98,
        frame_delay_2nd: int = 2,
        progress: bool = True,
        reconstructor: str = 'linear',
    ) -> Dict[str, Any]:
        """
        Run the second-stage loop and store the result dictionary.
        """
        self._require_second_stage()
        if self.reconstructor_2nd is None:
            raise ValueError("Second-stage reconstructor is missing.")
        if not hasattr(self, "residuals_opds_1rst"):
            raise ValueError("Run first-stage loop or assign residuals_opds_1rst first.")

        if nLoop is None:
            nLoop = self.residuals_opds_1rst.shape[0]
        if nmodes is None:
            nmodes = self.second_stage_calibration_params.get("nmodes", None)

        total = np.zeros(nLoop, dtype=self.dtype)
        residual_NGS = np.zeros(nLoop, dtype=self.dtype)
        residual_SRC = np.zeros(nLoop, dtype=self.dtype)
        dm_commands = np.zeros((nLoop, self.OZItwin.dm.nValidAct), dtype=self.dtype)
        reconstructed_cmd = np.zeros((nLoop, self.OZItwin.dm.nValidAct), dtype=self.dtype)
        opds = np.zeros_like(self.residuals_opds_1rst[:nLoop], dtype=self.dtype)
        ngs_opds = np.zeros_like(opds, dtype=self.dtype)
        src_opds = np.zeros_like(opds, dtype=self.dtype)

        wfs_signal_len = self.OZItwin.vzwfs.zwfs1.nSignal
        wfs_signal_history = np.zeros((nLoop, wfs_signal_len), dtype=self.dtype)
        wfsSignal = np.arange(0, wfs_signal_len) * 0

        self.OZItwin.dm.coefs = 0
        self.OZItwin.atm.initializeAtmosphere(self.OZItwin.tel)

        iterator = range(nLoop)
        
        for i in iterator:
            self.OZItwin.atm.update(self.residuals_opds_1rst[i])
            total[i] = np.std(self.OZItwin.tel.OPD[np.where(self.OZItwin.tel.pupil > 0)]) * 1e9

            self.OZItwin.atm * self.OZItwin.src * self.OZItwin.tel * self.OZItwin.dm
            self.OZItwin.tel * self.OZItwin.vzwfs

            residual_NGS[i] = np.std(self.OZItwin.tel.OPD[np.where(self.OZItwin.tel.pupil > 0)]) * 1e9
            ngs_opds[i] = self.OZItwin.tel.mean_removed_OPD.copy()
            opds[i] = self.OZItwin.tel.OPD.copy()

            self.OZItwin.atm * self.OZItwin.src * self.OZItwin.tel * self.OZItwin.dm
            dm_commands[i, :] = self.OZItwin.dm.coefs.copy()
            residual_SRC[i] = np.std(self.OZItwin.tel.OPD[np.where(self.OZItwin.tel.pupil > 0)]) * 1e9
            src_opds[i] = self.OZItwin.tel.mean_removed_OPD.copy()

            if frame_delay_2nd == 1:
                wfsSignal = self.OZItwin.vzwfs.zwfs1.signal.copy()
            if reconstructor == 'linear':
                reconstructed_cmd[i] = np.matmul(self.reconstructor_2nd, wfsSignal)
            elif reconstructor == 'atan':
                reconstructed_cmd[i]
            self.OZItwin.dm.coefs = leak_2nd * self.OZItwin.dm.coefs- gainCL_2nd * reconstructed_cmd[i]
            

            if frame_delay_2nd == 2:
                wfsSignal = self.OZItwin.vzwfs.zwfs1.signal.copy()

            wfs_signal_history[i] = wfsSignal
            print(
            f'\rLoop {i+1}/{nLoop} '
            f'NGS: {residual_NGS[i]:.3f} '
            f'-- SRC: {residual_SRC[i]:.3f}',
            end='',
            flush=True
            )
        self.second_stage_results = {
            "config": SecondStageLoopConfig(
                nLoop=nLoop,
                nmodes=nmodes,
                gainCL_2nd=gainCL_2nd,
                leak_2nd=leak_2nd,
                frame_delay_2nd=frame_delay_2nd,
                progress=progress,
            ),
            "total": total,
            "residual_NGS": residual_NGS,
            "residual_SRC": residual_SRC,
            "dm_commands": dm_commands,
            "reconstructed_cmd": reconstructed_cmd,
            "opds": opds,
            "ngs_opds": ngs_opds,
            "src_opds": src_opds,
            "wfs_signal_history": wfs_signal_history,
            "reconstructor_type": reconstructor,
        }
        self.opds_2nd = src_opds
        return self.second_stage_results

    # ------------------------------------------------------------------
    # Reconstruction and modal projections
    # ------------------------------------------------------------------
    
    
    def reconstruct_second_stage_atan(
        self,
        *,
        opds: Optional[np.ndarray] = None,
        nmodes: Optional[int] = None,
        parallel: bool = True,
        parall_njob: int = 6,
        method: str = "atan",
        iteration: int = 20,
        damping: float = 0.5,
    ) -> np.ndarray:
        """
        Create ZWFS images from OPD maps, reconstruct phase, convert to OPD,
        and optionally filter on the requested modal basis.
        """
        self._require_second_stage()

        if opds is None:
            if not hasattr(self, "opds_2nd"):
                raise ValueError("Provide opds or run the second-stage loop first.")
            opds = self.opds_2nd
        if nmodes is None:
            nmodes = self.second_stage_calibration_params.get("nmodes", 35)

        phase_rad = opds / self.OZItwin.src.wavelength * 2 * np.pi
        self.OZItwin.create_images_from_phase(phase_rad)
        self.OZItwin.reconstruct_all_phase(
            method=method,
            parallel=parallel,
            parall_njob=parall_njob,
            iteration=iteration,
            damping=damping,
        )

        self.OPDs_atan = self.OZItwin.OPDs.copy()
        self.OPDs_atan_filtered = self.OZItwin.filter_OPDs(
            self.OPDs_atan.copy(),
            nmodes=nmodes,
        )
        return self.OPDs_atan_filtered

    def compute_modal_quantities(
        self,
        *,
        nmodes: Optional[int] = None,
        include_atan: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute modal projections used by the original PSD cells.

        Stored keys:
        ``proj_modes_1rst``, ``proj_modes_2nd``, ``opd_true``, ``opd_atm``,
        and, if available/requested, ``opd_atan``.
        """
        self._require_first_stage()
        self._require_second_stage()

        if nmodes is None:
            nmodes = self.second_stage_calibration_params.get("nmodes", 35)

        if not self.first_stage_results or not self.second_stage_results:
            raise ValueError("Run both loops before computing modal quantities.")

        nLoop = self.second_stage_results["reconstructed_cmd"].shape[0]

        phase_2nd = self.OZItwin.dm.modes @ self.second_stage_results["reconstructed_cmd"].T
        phase_2nd = _demean_columns(phase_2nd)
        proj_modes_2nd = self.OZItwin.proj_M2C @ phase_2nd

        update_mask = self.first_stage_results["update_mask"]
        cmd_1rst = self.first_stage_results["reconstructed_cmd"][update_mask]
        phase_1rst_native = self.dm.modes @ cmd_1rst.T
        phase_1rst_cube = phase_1rst_native.T.reshape(
            cmd_1rst.shape[0],
            self.tel.pupil.shape[0],
            self.tel.pupil.shape[1],
        )
        phase_1rst_projected = self.project_opd_between_pupils(
            phase_1rst_cube,
            input_pupil=self.tel.pupil,
            output_pupil=self.OZItwin.tel.pupil,
            output_shape=self.OZItwin.tel.pupil.shape,
        ).reshape(cmd_1rst.shape[0], -1).T
        phase_1rst_projected = _demean_columns(phase_1rst_projected)
        proj_modes_1rst = self.OZItwin.proj_M2C @ phase_1rst_projected

        residuals_opds = self.residuals_opds_1rst[:nLoop].reshape(nLoop, -1).T
        residuals_opds = _demean_columns(residuals_opds)
        opd_true = self.OZItwin.proj_M2C @ residuals_opds

        atm_opd = self.atm_OPDs_2nd[:nLoop].reshape(nLoop, -1).T
        atm_opd = _demean_columns(atm_opd)
        opd_atm = self.OZItwin.proj_M2C @ atm_opd

        out = {
            "nmodes": nmodes,
            "proj_modes_1rst": proj_modes_1rst,
            "proj_modes_2nd": proj_modes_2nd,
            "opd_true": opd_true,
            "opd_atm": opd_atm,
        }

        if include_atan and hasattr(self, "OPDs_atan_filtered"):
            opds_at = self.OPDs_atan_filtered.reshape(nLoop, -1).T
            opds_at = _demean_columns(opds_at)
            out["opd_atan"] = self.OZItwin.proj_M2C @ opds_at

        self.modal.update(out)
        return out

    def compute_modal_psd(
        self,
        *,
        scale_nm2: bool = True,
        keys: Optional[Iterable[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute modal variances from already-projected modal coefficients.

        Parameters
        ----------
        scale_nm2 : bool
            If True, return values in nm^2/mode. Otherwise return SI units.
        keys : iterable of str, optional
            Modal dictionary keys to process.
        """
        if keys is None:
            keys = [
                "proj_modes_1rst",
                "proj_modes_2nd",
                "opd_atan",
                "opd_true",
                "opd_atm",
            ]

        factor = 1e18 if scale_nm2 else 1.0
        out: Dict[str, np.ndarray] = {}
        for key in keys:
            if key not in self.modal:
                continue
            # modal[key] is stored as (n_modes, n_frames).
            coeff = self.modal[key].T.copy()
            coeff -= np.nanmean(coeff, axis=0, keepdims=True)
            out[key] = np.nanvar(coeff, axis=0) * factor

        self.modal_psd.update(out)
        return out

    # ------------------------------------------------------------------
    # Temporal PSD
    # ------------------------------------------------------------------
    def compute_temporal_psd(
        self,
        name: str,
        data: np.ndarray,
        *,
        t: Optional[np.ndarray] = None,
        fs: Optional[float] = None,
        nperseg: int = 4096,
        noverlap: Optional[int] = None,
        detrend: str = "constant",
        window: str = "hann",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute and store a temporal PSD using ``OZItwin.psd``.

        ``data`` must be ``(n_samples, n_signals)``.
        """
        self._require_second_stage()

        data = np.asarray(data)
        if data.ndim == 1:
            data = data[:, None]

        if t is None:
            if fs is None:
                fs = 1.0 / self.OZItwin.tel.samplingTime
            t = np.arange(data.shape[0]) / fs

        f, psd = self.OZItwin.psd(
            t,
            data,
            fs=fs,
            nperseg=min(nperseg, data.shape[0]),
            noverlap=noverlap,
            detrend=detrend,
            window=window,
        )
        self.temporal_psd[name] = (f, psd)
        return f, psd

    def compute_temporal_psds(
        self,
        *,
        nperseg_2nd: int = 1000,
        nperseg_1rst: int = 500,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute the temporal PSDs used most often in the original analysis.
        """
        out = {}

        if "opd_atan" in self.modal:
            out["atan"] = self.compute_temporal_psd(
                "atan",
                self.modal["opd_atan"].T,
                fs=1.0 / self.OZItwin.tel.samplingTime,
                nperseg=nperseg_2nd,
            )

        if "proj_modes_2nd" in self.modal:
            out["second_stage_IM"] = self.compute_temporal_psd(
                "second_stage_IM",
                self.modal["proj_modes_2nd"].T,
                fs=1.0 / self.OZItwin.tel.samplingTime,
                nperseg=nperseg_2nd,
            )

        if "proj_modes_1rst" in self.modal:
            out["first_stage_IM"] = self.compute_temporal_psd(
                "first_stage_IM",
                self.modal["proj_modes_1rst"].T,
                fs=1.0 / self.tel.samplingTime,
                nperseg=nperseg_1rst,
            )

        if "opd_true" in self.modal:
            out["true_residuals"] = self.compute_temporal_psd(
                "true_residuals",
                self.modal["opd_true"].T,
                fs=1.0 / self.OZItwin.tel.samplingTime,
                nperseg=nperseg_2nd,
            )

        if "opd_atm" in self.modal:
            out["atmosphere"] = self.compute_temporal_psd(
                "atmosphere",
                self.modal["opd_atm"].T,
                fs=1.0 / self.OZItwin.tel.samplingTime,
                nperseg=nperseg_2nd,
            )

        return out

    # ------------------------------------------------------------------
    # Convenience workflow
    # ------------------------------------------------------------------
    def run_full_stage(
        self,
        *,
        M2C_path: Union[str, Path],
        valid_pixel_path: Union[str, Path],
        interaction_matrix_path: Union[str, Path],
        mat_key: str = "matrix_inf",
        nLoop: int = 1000,
        first_stage_end_mode: int = 195,
        second_stage_nmodes: int = 35,
        atmosphere_seed: Optional[int] = 15,
        first_gain: float = 0.5,
        first_leak: float = 0.995,
        first_frame_delay: int = 2,
        second_gain: float = 0.0,
        second_leak: float = 0.98,
        second_frame_delay: int = 2,
        reconstruct_atan: bool = True,
        atan_parallel: bool = True,
        atan_njobs: int = 6,
    ) -> "PAPYRIIS":
        """
        Run generation of the classes, calibration, closed loop.
        """
        self.set_first_stage_pupil(calibration=True)
        self.initialize_first_stage_propagation()
        self.load_first_stage_inputs(
            M2C_path=M2C_path,
            valid_pixel_path=valid_pixel_path,
            interaction_matrix_path=interaction_matrix_path,
            mat_key=mat_key,
        )
        self.calibrate_first_stage(end_mode=first_stage_end_mode, switch_to_sky=True)
        self.calibrate_second_stage(nmodes=second_stage_nmodes)
        self.generate_second_stage_atmosphere(nLoop=nLoop, seed=atmosphere_seed)
        self.project_atmosphere_to_first_stage()
        self.run_first_stage_loop(
            nLoop=nLoop,
            gainCL=first_gain,
            leak=first_leak,
            frame_delay=first_frame_delay,
        )
        self.run_second_stage_loop(
            nLoop=nLoop,
            nmodes=second_stage_nmodes,
            gainCL_2nd=second_gain,
            leak_2nd=second_leak,
            frame_delay_2nd=second_frame_delay,
        )
        if reconstruct_atan:
            self.reconstruct_second_stage_atan(
                nmodes=second_stage_nmodes,
                parallel=atan_parallel,
                parall_njob=atan_njobs,
            )
        self.compute_modal_quantities(
            nmodes=second_stage_nmodes,
            include_atan=reconstruct_atan,
        )
        self.compute_modal_psd()
        self.compute_temporal_psds()
        return self

    # ------------------------------------------------------------------
    # Save/load helpers
    # ------------------------------------------------------------------
    def save_npz(
        self,
        path: Union[str, Path],
        *,
        include_large_cubes: bool = False,
    ) -> Path:
        """
        Save selected numerical products to an ``.npz`` file.

        Large cubes are excluded by default to avoid accidental huge files.
        """
        path = Path(path)
        payload: Dict[str, Any] = {}

        for prefix, dct in [
            ("modal_psd", self.modal_psd),
            ("modal", self.modal),
        ]:
            for key, value in dct.items():
                if isinstance(value, np.ndarray):
                    payload[f"{prefix}_{key}"] = value

        for key, value in self.temporal_psd.items():
            payload[f"temporal_psd_{key}_f"] = value[0]
            payload[f"temporal_psd_{key}_psd"] = value[1]

        if self.first_stage_results:
            for key in ["total", "residual_NGS", "residual_SRC", "dm_commands", "reconstructed_cmd"]:
                payload[f"first_{key}"] = self.first_stage_results[key]

        if self.second_stage_results:
            for key in ["total", "residual_NGS", "residual_SRC", "dm_commands", "reconstructed_cmd"]:
                payload[f"second_{key}"] = self.second_stage_results[key]

        if include_large_cubes:
            for name in [
                "atm_OPDs_2nd",
                "atm_OPDs_1rst",
                "residuals_opds_1rst",
                "opds_2nd",
                "OPDs_atan",
                "OPDs_atan_filtered",
            ]:
                if hasattr(self, name):
                    payload[name] = getattr(self, name)

        np.savez_compressed(path, **payload)
        return path

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _load_interaction_matrix_from_file(
        self,
        path: Union[str, Path],
        *,
        mat_key: str = "matrix_inf",
    ) -> np.ndarray:
        """Load an interaction matrix from ``.npy``, ``.npz``, or ``.mat``."""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".npy":
            return np.load(path)
        if suffix == ".npz":
            data = np.load(path)
            if mat_key in data:
                return data[mat_key]
            if len(data.files) == 1:
                return data[data.files[0]]
            raise KeyError(f"{mat_key!r} not found in {path}. Keys: {data.files}")
        if suffix == ".mat":
            if read_mat is None:
                raise ImportError("pymatreader is required to read .mat files.")
            data = read_mat(str(path))
            if mat_key not in data:
                raise KeyError(f"{mat_key!r} not found in {path}. Keys: {list(data.keys())}")
            return data[mat_key]

        raise ValueError(f"Unsupported interaction matrix file extension: {suffix}")

    def _prepare_first_stage_science_source(self) -> None:
        """Create and propagate the first-stage scientific source if needed."""
        if self.src is not None:
            return

        try:
            from OOPAO.Source import Source
        except Exception as exc:
            raise ImportError("Could not import OOPAO.Source.Source.") from exc

        self.src = Source("IR1310", 0)
        self.src * self.tel

    def _sampling_ratio(self) -> float:
        """Return second-stage sampling frequency divided by first-stage one."""
        return (1.0 / self.OZItwin.tel.samplingTime) / (1.0 / self.tel.samplingTime)

    @staticmethod
    def _build_sampling_mask(n: int, ratio: float) -> np.ndarray:
        """
        Build a boolean update mask for the first-stage loop.

        If the ratio is close to an integer, this reproduces ``i % ratio == 0``.
        Otherwise, it uses the nearest sample-time accumulator.
        """
        ratio_round = int(round(ratio))
        mask = np.zeros(n, dtype=bool)

        if np.isclose(ratio, ratio_round):
            mask[::max(ratio_round, 1)] = True
            return mask

        # General non-integer fallback.
        next_update = 0.0
        for i in range(n):
            if i + 1e-12 >= next_update:
                mask[i] = True
                next_update += ratio
        return mask

    def _require_first_stage(self) -> None:
        if self.Papytwin is None:
            raise RuntimeError("First stage is not initialized.")

    def _require_second_stage(self) -> None:
        if self.OZItwin is None:
            raise RuntimeError("Second stage is not initialized.")
