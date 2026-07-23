
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


# returns the distance to pixel value [x, y] from the centre at position [x, y]
def circle_radius(shape):
    x, y = np.indices(shape) # returns indices of a grid of shape == shape
    center = [shape[0] // 2, shape[1] // 2]
    radius = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    radius = radius.astype(int)
    return radius


# calculates the circular average
# need shape of the array and the PSF/OTF
def circular_average(shape, psf):
    radius = circle_radius(shape)
    pixel_sum_at_r = np.bincount(radius.ravel(), psf.ravel())  # weighted sum of pixels at the same radius
    number_of_pixels_at_r = np.bincount(radius.ravel())
    average_sum = pixel_sum_at_r / number_of_pixels_at_r
    radial_number = np.arange(len(average_sum)) / len(average_sum)
    return radial_number, average_sum


#no averaging for the PSD to conserve the variance
def circular_sum_PSD(shape, psd, freq_max):
    radius = circle_radius(shape)
    #radius_freq = radius / np.max(radius) * freq_max
    pixel_sum_at_r = np.bincount(radius.ravel(), psd.ravel())  # weighted sum of pixels at the same radius
    radial_number = freq_max * np.arange(len(pixel_sum_at_r)) / len(pixel_sum_at_r)
    return radial_number, pixel_sum_at_r



#PSD fitting

from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Literal

# ── model definitions ──────────────────────────────────────────────────────────
 
def psd_kolmogorov(f: np.ndarray, A: float, alpha: float) -> np.ndarray:
    """Power-law PSD:  A * f^{-alpha}."""
    return A * np.abs(f) ** (-alpha)
 
 
def psd_von_karman(f: np.ndarray, A: float, alpha: float, f0: float) -> np.ndarray:
    """Von Kármán PSD:  A * (f^2 + f0^2)^{-alpha/2}.
 
    f0 is the outer-scale knee frequency V/(2*pi*L0).
    """
    return A * (f ** 2 + f0 ** 2) ** (-alpha / 2)
 

@dataclass
class PSDFitResult:
    model: str          # "kolmogorov" or "vk"
    A: float            # amplitude
    alpha: float        # power-law exponent  (canonical: 8/3 ≈ 2.667)
    f0: float | None    # knee frequency [Hz] — Von Kármán only
    A_err: float
    alpha_err: float
    f0_err: float | None
    residual_rms: float # RMS of log10 residuals
 
    def __str__(self) -> str:
        lines = [
            f"Model     : {self.model}",
            f"A         : {self.A:.4e}  ± {self.A_err:.2e}",
            f"alpha     : {self.alpha:.4f}  ± {self.alpha_err:.4f}",
        ]
        if self.f0 is not None:
            lines.append(f"f0        : {self.f0:.4f} Hz  ± {self.f0_err:.4f}")
        lines.append(f"RMS(log10): {self.residual_rms:.4f} dex")
        return "\n".join(lines)
    



def fit_psd(
    freq: np.ndarray,
    psd: np.ndarray,
    model: Literal["kolmogorov", "vk"] = "vk",
    freq_range: tuple[float, float] | None = None,
    p0_alpha: float = 8 / 3,
    p0_f0: float | None = None,
    fit_in_log: bool = True,
) -> PSDFitResult:
    """Fit a temporal PSD measurement.
 
    Parameters
    ----------
    freq : array_like
        Temporal frequencies [Hz], positive values only.
    psd : array_like
        PSD values (one-sided).  Units are arbitrary.
    model : {"kolmogorov", "vk"}
        "kolmogorov" = pure power law.
        "vk"         = Von Kármán (adds outer-scale knee).
    freq_range : (f_min, f_max) or None
        Restrict the fit to this frequency band.  Useful to exclude
        the noise floor or very low frequencies dominated by L0 effects.
    p0_alpha : float
        Initial guess for the exponent (default 8/3).
    p0_f0 : float or None
        Initial guess for the knee frequency [Hz].  If None the code
        estimates it from the lowest retained frequency.
    fit_in_log : bool
        If True (default) fit log10(PSD) vs log10(freq); gives equal
        weight per decade rather than over-weighting high-PSD points.
 
    Returns
    -------
    PSDFitResult
    """
    freq = np.asarray(freq, dtype=float)
    psd  = np.asarray(psd,  dtype=float)
 
    # --- keep only positive, finite, positive-valued points ---
    mask = (freq > 0) & np.isfinite(freq) & (psd > 0) & np.isfinite(psd)
    if freq_range is not None:
        mask &= (freq >= freq_range[0]) & (freq <= freq_range[1])
    freq, psd = freq[mask], psd[mask]
 
    if len(freq) < 3:
        raise ValueError("Too few valid data points after masking.")
 
    # --- build fit arrays (log or linear) ---
    if fit_in_log:
        x = np.log10(freq)
        y = np.log10(psd)
 
        def _kol_log(lf, logA, alpha):
            return logA - alpha * lf
 
        def _vk_log(lf, logA, alpha, logf0):
            f0 = 10 ** logf0
            return logA - (alpha / 2) * np.log10(10 ** (2 * lf) + f0 ** 2)
 
        # initial guesses in log space
        p0_logA = np.median(y) + p0_alpha * np.median(x)
        if p0_f0 is None:
            p0_f0 = freq[0]
        p0_logf0 = np.log10(p0_f0)
 
        if model == "kolmogorov":
            func   = _kol_log
            p0     = [p0_logA, p0_alpha]
            bounds = ([-np.inf, -1], [np.inf, 10.0])
        else:
            func   = _vk_log
            p0     = [p0_logA, p0_alpha, p0_logf0]
            bounds = ([-np.inf, -1, -4.0], [np.inf, 10.0, np.log10(freq.max())])
 
        popt, pcov = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=10_000)
        perr = np.sqrt(np.diag(pcov))
        residuals = y - func(x, *popt)
 
        if model == "kolmogorov":
            A, alpha   = 10 ** popt[0], popt[1]
            A_err      = A * np.log(10) * perr[0]
            alpha_err  = perr[1]
            f0 = f0_err = None
        else:
            A, alpha, f0 = 10 ** popt[0], popt[1], 10 ** popt[2]
            A_err     = A * np.log(10) * perr[0]
            alpha_err = perr[1]
            f0_err    = f0 * np.log(10) * perr[2]
 
    else:  # linear-space fit
        if model == "kolmogorov":
            p0     = [psd.max() * freq[0] ** p0_alpha, p0_alpha]
            bounds = ([0, 0.5], [np.inf, 10.0])
            popt, pcov = curve_fit(
                psd_kolmogorov, freq, psd, p0=p0, bounds=bounds, maxfev=10_000
            )
            perr = np.sqrt(np.diag(pcov))
            A, alpha = popt
            A_err, alpha_err = perr
            f0 = f0_err = None
            residuals = np.log10(psd) - np.log10(psd_kolmogorov(freq, *popt))
        else:
            if p0_f0 is None:
                p0_f0 = freq[0]
            p0     = [psd.max() * freq[0] ** p0_alpha, p0_alpha, p0_f0]
            bounds = ([0, 0.5, 0], [np.inf, 10.0, freq.max()])
            popt, pcov = curve_fit(
                psd_von_karman, freq, psd, p0=p0, bounds=bounds, maxfev=10_000
            )
            perr = np.sqrt(np.diag(pcov))
            A, alpha, f0 = popt
            A_err, alpha_err, f0_err = perr
            residuals = np.log10(psd) - np.log10(psd_von_karman(freq, *popt))
 
    return PSDFitResult(
        model      = model,
        A          = A,
        alpha      = alpha,
        f0         = f0,
        A_err      = A_err,
        alpha_err  = alpha_err,
        f0_err     = f0_err,
        residual_rms = float(np.sqrt(np.mean(residuals ** 2))),
    )




def mode_calculator_fromOPD(OPD_screen, m2c, dm_modes_masked):
    """
    Calculates the piston subtracted modes 
    (piston is usually not included in M2C so it can infect other modes)

    Atmosphere OPD and dm_modes_masked should have units of meters
    """

    
    dm_modes_masked = dm_modes_masked * 1e9 #nm conversion

    m2p = dm_modes_masked @ m2c
    p2m = np.linalg.pinv(m2p)
    mode_projector = m2p @ p2m 

    #converts KL mode units into nm^2
    G = mode_covariance(m2p)
    
    OPD_screen = OPD_screen * 1e9 #nm conversion

    OPD_screen_no_piston = OPD_screen - OPD_screen.mean(axis=1, keepdims=True)
    print(OPD_screen_no_piston.shape)
    print(p2m.T.shape)
    modes = OPD_screen_no_piston @ p2m.T
    OPD_reprojected = OPD_screen_no_piston @ mode_projector

    var_from_modes = np.mean(np.sum((modes @ G) * modes, axis=1))
    atm_reprojected_variance = np.mean(np.var(OPD_reprojected, axis = 1))

    #if not np.isclose(atm_reprojected_variance, var_from_modes, rtol=1e-5):
    print(
        f"Reprojected variance {atm_reprojected_variance} "
        f"!= {var_from_modes} KL mode variance"
    )

    return modes



def mode_calculator_fromDM(dm_coefs, m2c):
    """
    converts dm coefficients into modes
    """
  
    c2m = np.linalg.pinv(m2c)
    dm_modes = dm_coefs @ c2m.T

    return dm_modes



def tPSD_calculator(modes, mode_counter, G_diag, frequency, nperseg = 256):
    """
    Calculate temporal PSD of modes in nm^2.

    modes shape: (timeseries length, no_of_modes)
    G_diag: precomputed diagonal of mode_covariance(dm_modes_masked * 1e9 @ m2c)
    """
    G_mm = G_diag[mode_counter]

    modes_psd_f, modes_psd = welch_method_scipy(modes[:, mode_counter], frequency, nperseg = nperseg)
    modes_psd = modes_psd * G_mm
    modes_psd_boxcar_f, modes_psd_boxcar = welch_method_scipy(modes[:, mode_counter], frequency, nperseg = len(modes[:, mode_counter]), window = 'boxcar')
    modes_psd_boxcar = modes_psd_boxcar * G_mm

    var_from_PSD = np.trapezoid(modes_psd_boxcar, modes_psd_boxcar_f)
    var_from_modes = np.var(modes[:, mode_counter]) * G_mm

    rtol = 1e-3  # 0.1% tolerance
    if not np.isclose(var_from_PSD, var_from_modes, rtol=rtol):
        print(
            f"Mode {mode_counter}: variance from PSD ({var_from_PSD}) "
            f"!= variance from modes ({var_from_modes}) "
            f"beyond tolerance {rtol*100}%"
        )

    return modes_psd_f, modes_psd, var_from_PSD



def welch_method_scipy(data, fs, nperseg=256, window='hann'):
    """
    welch method to mitigate the windowing effects when calculating FFTs
    """
    frequencies, psd = signal.welch(
        data,
        fs=fs,
        window=window,  #windowing
        nperseg=nperseg,
        scaling='density'
    )
    return frequencies, psd


#So we learned 3 things:
#   1. the total phase variance is a @ G @ a (for a single phase screen), where a are the modes and G is the mode covariance matrix
#   2. when calculating the KL coefficient variance, naturally you only calculate the variance for single coefficients and only G diagonal terms appear (you lose some phase variance)
        #np.var(atm_simulated_2nd_modes, axis = 0) * np.diag(G)
#   3. temporal PSDs just need to be multiplied by the diagional G elements to retain their nm^2 normalisation:
        #atm_simulated_2nd_modes_1_psd_2 * G[i, i]
        #integrating it you get the spatial variance for the selected mode
        # any small errors are due to windowing in welch and there might be an error in np.trapezoid function (did not observe during calculations)

def mode_covariance(m2p):
    """
    Calculates the mode covariance matrix
    (i.e. how much each mode influences other modes, ideally should be a diagonal matrix)
    m2p - mode to phase/OPD matrix
    """
    mode_opds_centered = m2p - m2p.mean(axis=0)
    G = mode_opds_centered.T @ mode_opds_centered / len(m2p)
    return G




from OOPAO.DeformableMirror import DeformableMirror, MisRegistration
from OOPAO.Telescope import Telescope
from OOPAO.tools.interpolateGeometricalTransformation import interpolate_cube
from PAPYRIIS_2stage_CNN_RL.parameterFile_papyriis import initializeParameterFile
def first_stage_dm_builder():
    '''
    Temporary function for building the 1st stage DM influence functions
    '''
    param = initializeParameterFile()
    T152onDM_size       = 35.5 # mm
    PapyrusOnDM_size    = 37.5 # mm 
    ratio_sky_calib = T152onDM_size/PapyrusOnDM_size
    from OOPAO.Telescope import Telescope
    tel = Telescope(resolution    = int(100),
                        diameter            = param['diameter']/ratio_sky_calib,
                        samplingTime        = param['samplingTime'],
                        centralObstruction  = 0,
                        fov                 = 0)

    # mis-registrations object
    misReg          = MisRegistration(param)
    pitch           = 2.5 #mm
    DM_diag_size    = param['nActuator'] * pitch #mm
    scale_T152DM = DM_diag_size / T152onDM_size
    D_T152 = 1.52

    x = np.linspace(-scale_T152DM * D_T152/2, scale_T152DM * D_T152/2, param['nActuator'])
    [X,Y] = np.meshgrid(x,x)

    DM_coordinates = np.asarray([X.reshape(17**2),Y.reshape(17**2)]).T
    dist           = np.sqrt(DM_coordinates[:,0]**2 + DM_coordinates[:,1]**2)
    DM_coordinates = DM_coordinates[dist <= D_T152/2 + 2.2 *pitch * D_T152 / T152onDM_size, :]
    DM_pitch       = pitch * D_T152 / T152onDM_size

    # hardcoded for now
    alpao_unit     = 30*7591.024876

    param['dm_coordinates'] = DM_coordinates
    param['pitch']          = DM_pitch

    dm_1st=DeformableMirror(telescope    = tel,\
                        nSubap       = 16,\
                        mechCoupling = 0.36,\
                        misReg       = misReg, \
                        coordinates  = DM_coordinates,\
                        pitch        = DM_pitch,\
                        modes        = None,
                        flip_lr      = True,
                        sign         = -1/alpao_unit)
    return dm_1st.modes, tel.pupil



def second_stage_dm_builder():
    tel_sky_2 = Telescope(resolution      = 100,
                            diameter            = 1.52,
                            samplingTime        = 1/400,
                            centralObstruction  = 0)
            
    pupil_sky_2 = tel_sky_2.pupil.copy()


    tel_sky_2 = Telescope(resolution      = 100,
                            diameter            = 1.52,
                            samplingTime        = 1/400,
                            pupil = pupil_sky_2)
    param_misreg = np.load("PAPYRIIS_2stage_CNN_RL/dm_second_stage_misreg_dict.npy", allow_pickle=True).item()
    m = MisRegistration(param_misreg)
    dm_2nd = DeformableMirror(telescope=tel_sky_2, 
                                    nSubap=10, 
                                    mechCoupling=0.35, 
                                    print_dm_properties=False, 
                                    pitch=0.11, 
                                    misReg=m,
                                    sign=-1e-5)
    return dm_2nd.modes, tel_sky_2.pupil

