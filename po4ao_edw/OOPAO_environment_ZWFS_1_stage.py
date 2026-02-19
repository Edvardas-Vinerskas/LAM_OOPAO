import gymnasium as gym
import numpy as np
from numpy.fft import fftshift, fft, fft2, fftfreq, rfft, rfftfreq
import torch
from functions import *


import OOPAO
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Pyramid import Pyramid
from OOPAO.ZWFS2 import ZWFS2
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.calibration.CalibrationVault import CalibrationVault


class OOPAO_environment_ZWFS_1_stage(gym.Env):
    def __init__(self):

        self.device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        self.N_SUBAPERTURE_zer = 9
        self.DIAMETER = 1.5
        self.RESOLUTION = 20 * 8
        self.FREQUENCY = 1000
        self.r_0 = 0.15
        self.L_0 = 25
        self.WIND_SPEED = [10, 20]
        self.WIND_DIRECTION = [0, 100]
        self.FRACTIONAL_C_N2 = [0.6, 0.4]
        self.ALTITUDE = [0, 4500]
        self.NGS_MAGNITUDE = 2
        self.NGS_OptBand   = "I"
        self.SRC_MAGNITUDE = 2
        self.SRC_OptBand = "I"
        self.CENTRAL_OBSTRUCTION = 0
        self.MECH_COUPLING = 0.35
        self.KL_coefs_zer = 81
        self.zeroPaddingFactor = 6
        self.seed = 10



        #---------------------------------------------------SOURCE---------------------------------------------------#

        self.NGS = Source(optBand  = self.NGS_OptBand,
                     magnitude     = self.NGS_MAGNITUDE)

        self.SRC = Source(optBand  = self.SRC_OptBand,
                     magnitude     = self.SRC_MAGNITUDE)

        #---------------------------------------------------TELESCOPE---------------------------------------------------#
        self.TEL = Telescope(resolution    = self.RESOLUTION,
                        diameter           = self.DIAMETER,
                        samplingTime       = 1 / self.FREQUENCY,
                        centralObstruction = self.CENTRAL_OBSTRUCTION)

        self.SRC * self.TEL

        self.TEL.computePSF(self.zeroPaddingFactor)

        # diffraction limited OTF calculation
        """OTF_dl = fftshift(fft2(fftshift(self.TEL.PSF / np.sum(self.TEL.PSF))))
        self.x_axis, self.OTF_dl_averaged = circular_average((np.abs(OTF_dl)).shape, np.abs(OTF_dl))"""

        #---------------------------------------------------ATMOSPHERE---------------------------------------------------#
        self.ATM = Atmosphere(telescope       = self.TEL,
                                r0            = self.r_0,
                                L0            = self.L_0,
                                windSpeed     = self.WIND_SPEED,
                                windDirection = self.WIND_DIRECTION,
                                fractionalR0  = self.FRACTIONAL_C_N2,
                                altitude      = self.ALTITUDE
                                )

        self.ATM.initializeAtmosphere(telescope=self.TEL)



        #---------------------------------------------------DEFORMABLE_MIRROR---------------------------------------------------#

        self.DM_zer = DeformableMirror(telescope    = self.TEL,
                                       nSubap       = self.N_SUBAPERTURE_zer,
                                       mechCoupling = self.MECH_COUPLING)
        self.data_shape = self.DM_zer.nAct
        #---------------------------------------------------WFS---------------------------------------------------#
        """self.vZWFS = ZWFS2(tel         = self.TEL,
                           zpf         = 30,
                           diameter    = 2.14,
                           phase_shift = [-0.75 * np.pi,0.3 * np.pi])"""

        self.vZWFS = ZWFS2(tel          = self.TEL,
                           zpf          = 30,
                           diameter     = 1.06,
                           phase_shift  = [-np.pi/2,np.pi/2])


        #---------------------------------------------------KL_MODAL_BASIS---------------------------------------------------#
        M2C_KL_zer   = compute_KL_basis(tel = self.TEL, atm = self.ATM, dm = self.DM_zer)
        self.M2C_zer = M2C_KL_zer[:, :self.KL_coefs_zer]

        KL_basis_dm = (self.DM_zer.modes @ M2C_KL_zer) * np.tile(self.TEL.pupil.flatten()[:, None], M2C_KL_zer.shape[1])
        self.projector_kl = np.linalg.pinv(KL_basis_dm)

        modes_zer    = self.DM_zer.modes @ self.M2C_zer

        self.M2C_ = self.M2C_zer

        stroke = self.SRC.wavelength / 16
        #---------------------------------------------------INTERACTION MATRIX---------------------------------------------------#
        #ZWFS calibration
        CALIB_zer = np.zeros((self.vZWFS.signal_cam.shape[0], self.M2C_zer.shape[1]))

        #zernike_wfs calibration with DM_zer
        for i in range(self.M2C_zer.shape[1]):
            v_plus  = self.M2C_zer[:, i] * stroke
            v_minus = -self.M2C_zer[:, i] * stroke

            self.SRC.reset()
            self.DM_zer.coefs = v_plus
            self.SRC ** self.TEL * self.DM_zer * self.vZWFS
            w_plus = self.vZWFS.signal_cam

            self.SRC.reset()
            self.DM_zer.coefs = v_minus
            self.SRC ** self.TEL * self.DM_zer * self.vZWFS
            w_minus = self.vZWFS.signal_cam

            CALIB_zer[:, i] = (w_plus - w_minus) / (2 * stroke)

        # takes in modes and outputs wfs signal
        self.CALIB_zer_obj = CalibrationVault(CALIB_zer)

        """# ---------------------------------------------------FITTING ERROR CALC---------------------------------------------------#
        self.modes_inv_pyr = np.linalg.pinv(modes_pyr)
        self.modes_inv_zer = np.linalg.pinv(modes_zer)"""

        #---------------------------------------------------INITIALIZATION---------------------------------------------------#
        self.ATM.generateNewPhaseScreen(seed = self.seed)
        self.atm_update_tracker = 0
        self.SRC.reset()
        self.DM_zer.coefs = 0
        self.DM_zer_copy  = 0
        self.SRC ** self.ATM * self.TEL * self.DM_zer * self.vZWFS
        self.TEL.print_optical_path()


        self.CL_gain_zer = 0.4


        self.CURRENT_STEPS = 0 #current steps seems to just be for tracking loop progress
        self.scale_up = 1e7
        self.scale_down = 1e-7


        # CRUCIAL PIXEL SIZE CHECK#
        self.pixel_size = self.DIAMETER / self.RESOLUTION
        if (3 * self.pixel_size) > self.r_0:
            raise SystemExit("ERROR: pixel size is too big for r_0 value")


        #takes in wfs signal and outputs controle
        self.reconstructor_zer = self.M2C_zer @ self.CALIB_zer_obj.M

        mask_1D = self.DM_zer.validAct.copy()
        mask_reshaped = mask_1D.reshape((self.N_SUBAPERTURE_zer + 1, self.N_SUBAPERTURE_zer + 1))
        self.mask = torch.from_numpy(mask_reshaped).bool()

        self.strehl_1st      = 0
        self.modes_atm       = 0
        self.modes_1st_stage = 0
        self.modes_2nd_stage = 0


    def flatten_dm(self):
        self.DM_zer.coefs = 0
        self.DM_zer_copy  = 0

        # propagate to wfs and apply new dm commands to the dm
        self.SRC ** self.ATM * self.TEL * self.DM_zer * self.vZWFS

        vzwfs_signal_proj = self.reconstructor_zer @ self.vZWFS.signal_cam

        # redoing in 2D
        vzwfs_signal_proj_2D = torch.zeros((self.N_SUBAPERTURE_zer + 1, self.N_SUBAPERTURE_zer + 1), dtype=torch.float32)
        vzwfs_signal_proj_2D[self.mask] = torch.from_numpy(vzwfs_signal_proj).float()
        next_state = vzwfs_signal_proj_2D


        return next_state

    def reset(self, seed = None, options = None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        if seed is None:
            seed = np.random.randint(1e9)

        self.ATM.generateNewPhaseScreen(seed=seed)
        self.atm_update_tracker = 0
        self.ATM.update()


        self.CURRENT_STEPS      = 0
        self.vzwfssignal_buffer = [torch.zeros((self.N_SUBAPERTURE_zer + 1, self.N_SUBAPERTURE_zer + 1))] #2 frame delay zwfs

        self.DM_zer.coefs = 0
        self.DM_zer_copy = 0
        self.DM_prev_coefs_zer = self.DM_zer.coefs.copy()



        # propagate to wfs and apply new dm commands to the dm
        self.SRC ** self.ATM * self.TEL * self.DM_zer * self.vZWFS

        vzwfs_signal_proj = self.reconstructor_zer @ self.vZWFS.signal_cam
        # redoing in 2D
        vzwfs_signal_proj_2D = torch.zeros((self.N_SUBAPERTURE_zer + 1, self.N_SUBAPERTURE_zer + 1), dtype=torch.float32)
        vzwfs_signal_proj_2D[self.mask] = torch.from_numpy(vzwfs_signal_proj).float()
        next_state = vzwfs_signal_proj_2D


        #change to "tt_modes"
        INFO        = {"KL_modes": "vzwfs_KL_modes"} #tip tilt zernike coefs

        return next_state, INFO


    @torch.no_grad()
    def step(self, action):
        self.ATM.update()
        self.atm_update_tracker = self.atm_update_tracker + 1
        atm_opd = self.ATM.OPD.copy()

        total_error = np.std(self.ATM.OPD[np.where(self.TEL.pupil > 0)]) * 1e9

        # frame delay implementation (zernike)
        self.vzwfssignal_buffer.append(action)
        self.vzwfs_delayed_signal = self.vzwfssignal_buffer[0]
        self.vzwfssignal_buffer.pop(0)


        self.DM_zer_coefs = self.DM_zer_copy - self.vzwfs_delayed_signal[self.mask].detach().cpu().numpy()
        self.DM_zer_copy = self.DM_zer_coefs.copy()

        self.SRC ** self.ATM * self.TEL
        self.modes_atm = self.projector_kl @ self.SRC.OPD.flatten()

        # propagate to wfs and apply new dm commands to the dm
        #only activate the second stage after the first stage closes the loop
        self.SRC ** self.ATM * self.TEL * self.DM_zer * self.vZWFS
        self.modes_1st_stage = self.projector_kl @ self.SRC.OPD.flatten()


        vzwfs_signal_torch_proj = self.reconstructor_zer @ self.vZWFS.signal_cam

        #redoing in 2D the zwfs signal
        vzwfs_signal_torch_proj_2D = torch.zeros((self.N_SUBAPERTURE_zer + 1, self.N_SUBAPERTURE_zer + 1), dtype = torch.float32)
        vzwfs_signal_torch_proj_2D[self.mask] = torch.from_numpy(vzwfs_signal_torch_proj).float()
        next_state = vzwfs_signal_torch_proj_2D


        strehl = np.exp(-np.var(self.TEL.src.phase[np.where(self.TEL.pupil == 1)]))
        residual_error = np.std(self.TEL.OPD[np.where(self.TEL.pupil > 0)]) * 1e9

        self.CURRENT_STEPS += 1

        self.TEL.computePSF(self.zeroPaddingFactor)
        INFO = {"strehl": strehl, "strehl_1st": strehl,
                "residual_error": residual_error, "total_error": total_error,
                "modes_1st_stage": self.modes_1st_stage, "modes_2nd_stage": self.modes_1st_stage,
                "modes_atm": self.modes_atm, "tracker": self.atm_update_tracker}


        return next_state, INFO




























