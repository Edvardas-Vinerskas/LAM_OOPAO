import gymnasium as gym
import numpy as np
from numpy.fft import fftshift, fft, fft2, fftfreq, rfft, rfftfreq
import torch

from OOPAO.tools.tools import strehlMeter
from functions import *


import OOPAO
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Pyramid import Pyramid
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis




class OOPAO_environment_PWFS(gym.Env):
    def __init__(self):

        self.device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.xp = torch if RL else np

        self.N_SUBAPERTURE_pyr = 20
        self.DIAMETER = 1.52
        self.RESOLUTION = self.N_SUBAPERTURE_pyr * 8
        self.FREQUENCY = 500
        self.MODULATION = 3
        self.LIGHT_RATIO = 0.1
        self.POST_PROCESS = "slopesMaps"
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
        self.KL_no = 250
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
        OTF_dl = fftshift(fft2(fftshift(self.TEL.PSF / np.sum(self.TEL.PSF))))
        self.x_axis, self.OTF_dl_averaged = circular_average((np.abs(OTF_dl)).shape, np.abs(OTF_dl))

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

        self.DM_pyr = DeformableMirror(telescope = self.TEL,
                              nSubap             = self.N_SUBAPERTURE_pyr,
                              mechCoupling       =self.MECH_COUPLING)

        self.data_shape = self.DM_pyr.nAct
        #---------------------------------------------------WFS---------------------------------------------------#

        self.PWFS = Pyramid(nSubap    = self.N_SUBAPERTURE_pyr,
                       telescope      = self.TEL,
                       modulation     = self.MODULATION,
                       lightRatio     = self.LIGHT_RATIO,
                       postProcessing = self.POST_PROCESS)


        #---------------------------------------------------KL_MODAL_BASIS---------------------------------------------------#
        M2C_KL_pyr   = compute_KL_basis(tel = self.TEL, atm = self.ATM, dm = self.DM_pyr)
        self.M2C_pyr = M2C_KL_pyr[:, :self.KL_no]
        modes_pyr    = self.DM_pyr.modes @ self.M2C_pyr

        self.M2C_     = self.M2C_pyr

        stroke = self.SRC.wavelength / 16
        #---------------------------------------------------INTERACTION MATRIX---------------------------------------------------#

        self.CALIB_pyr = InteractionMatrix(ngs               = self.NGS,
                                               tel           = self.TEL,
                                               dm            = self.DM_pyr,
                                               wfs           = self.PWFS,
                                               M2C           = self.M2C_pyr,
                                               atm           = self.ATM,
                                               nMeasurements = 1,
                                               stroke        = stroke)



        # ---------------------------------------------------FITTING ERROR CALC---------------------------------------------------#
        self.modes_inv_pyr = np.linalg.pinv(np.squeeze(modes_pyr[self.TEL.pupilLogical, :]))

        #---------------------------------------------------INITIALIZATION---------------------------------------------------#
        self.ATM.generateNewPhaseScreen(seed = self.seed)
        self.SRC.reset()
        self.DM_pyr.coefs = np.zeros(self.DM_pyr.coefs.shape)
        self.SRC ** self.ATM * self.TEL * self.DM_pyr * self.PWFS
        self.TEL.print_optical_path()


        self.CL_gain_pyr = 0.4


        self.CURRENT_STEPS = 0 #current steps seems to just be for tracking loop progress
        self.scale_up = 1e7
        self.scale_down = 1e-7

        # CRUCIAL PIXEL SIZE CHECK#
        self.pixel_size = self.DIAMETER / self.RESOLUTION
        if (3 * self.pixel_size) > self.r_0:
            raise SystemExit("ERROR: pixel size is too big for r_0 value")


        #takes in wfs signal and outputs controle
        self.reconstructor_pyr = self.M2C_pyr @ self.CALIB_pyr.M

        mask_1D = self.DM_pyr.validAct.copy()
        mask_reshaped = mask_1D.reshape((self.N_SUBAPERTURE_pyr + 1, self.N_SUBAPERTURE_pyr + 1))
        self.mask = torch.from_numpy(mask_reshaped).bool()



    def flatten_dm(self):
        self.DM_pyr.coefs = 0

        # propagate to wfs and apply new dm commands to the dm
        self.SRC * self.TEL * self.DM_pyr * self.PWFS

        pwfs_signal_proj = self.reconstructor_pyr @ self.PWFS.signal

        # redoing in 2D
        pwfs_signal_proj_2D = torch.zeros((self.N_SUBAPERTURE_pyr + 1, self.N_SUBAPERTURE_pyr + 1), dtype=torch.float32)
        pwfs_signal_proj_2D[self.mask] = torch.from_numpy(pwfs_signal_proj).float()
        next_state = pwfs_signal_proj_2D


        return next_state

    def reset(self, seed = None, options = None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        if seed is None:
            seed = np.random.randint(1e9)

        self.ATM.generateNewPhaseScreen(seed=seed)
        self.ATM.update()


        self.CURRENT_STEPS     = 0
        self.pwfssignal_buffer = [torch.zeros((self.N_SUBAPERTURE_pyr + 1, self.N_SUBAPERTURE_pyr + 1))] #2frame delay pwfs


        self.DM_pyr.coefs = 0
        self.DM_prev_coefs_pyr = self.DM_pyr.coefs.copy()




        # propagate to wfs and apply new dm commands to the dm
        self.SRC ** self.ATM * self.TEL * self.DM_pyr * self.PWFS

        pwfs_signal_proj = self.reconstructor_pyr @ self.PWFS.signal
        # redoing in 2D
        pwfs_signal_proj_2D = torch.zeros((self.N_SUBAPERTURE_pyr + 1, self.N_SUBAPERTURE_pyr + 1),
                                          dtype=torch.float32)
        pwfs_signal_proj_2D[self.mask] = torch.from_numpy(pwfs_signal_proj).float()
        next_state = pwfs_signal_proj_2D


        INFO        = {"KL_modes": "vzwfs_KL_modes"} #tip tilt zernike coefs

        return next_state, INFO


    @torch.no_grad()
    def step(self, action):
        self.ATM.update()
        atm_opd = self.ATM.OPD.copy()


        total_error = np.std(self.TEL.OPD[np.where(self.TEL.pupil > 0)]) * 1e9


        # frame delay implementation
        self.pwfssignal_buffer.append(action)
        self.pwfs_delayed_signal = self.pwfssignal_buffer[0]
        self.pwfssignal_buffer.pop(0)


        self.DM_pyr.coefs = self.DM_pyr.coefs - self.pwfs_delayed_signal[self.mask].detach().cpu().numpy()
        self.DM_prev_coefs_pyr = self.DM_pyr.coefs.copy()



        # propagate to wfs and apply new dm commands to the dm
        self.SRC * self.TEL * self.DM_pyr * self.PWFS


        pwfs_signal_torch_proj = self.reconstructor_pyr @ self.PWFS.signal

        #redoing in 2D
        pwfs_signal_torch_proj_2D = torch.zeros((self.N_SUBAPERTURE_pyr + 1, self.N_SUBAPERTURE_pyr + 1), dtype = torch.float32)
        pwfs_signal_torch_proj_2D[self.mask] = torch.from_numpy(pwfs_signal_torch_proj).float()
        next_state = pwfs_signal_torch_proj_2D


        strehl = np.exp(-np.var(self.TEL.src.phase[np.where(self.TEL.pupil == 1)]))
        residual_error = np.std(self.TEL.OPD[np.where(self.TEL.pupil > 0)]) * 1e9



        self.CURRENT_STEPS += 1

        self.TEL.computePSF(self.zeroPaddingFactor)
        INFO = {"strehl": strehl, "strehl_1st": strehl,  # the 1st is just a placeholder for standardisation
                "residual_error": residual_error, "total_error": total_error,
                "TEL_PSF": self.TEL.PSF, "residual_OPD": self.TEL.OPD,
                "atm_OPD": atm_opd}


        return next_state, INFO












