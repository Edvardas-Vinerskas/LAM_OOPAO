import gymnasium as gym
import numpy as np
from numpy.fft import fftshift, fft, fft2, fftfreq, rfft, rfftfreq
import torch
from functions import *
import matplotlib.pyplot as plt


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


class OOPAO_environment_ZWFS_perfect(gym.Env):
    def __init__(self):

        self.device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.N_SUBAPERTURE_pyr = 20
        self.N_SUBAPERTURE_zer = 9
        self.DIAMETER = 1.5
        self.RESOLUTION = self.N_SUBAPERTURE_pyr * 8
        self.FREQUENCY = 1500
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
        self.KL_coefs_pyr = 250
        self.KL_coefs_zer = 81
        self.zeroPaddingFactor = 6
        self.seed = 10

        self.pwfs_photNoise = True
        self.pwfs_readNoise = 4
        self.pwfs_quanteff = 0.8
        self.noise_calibration = 'on'



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
        self.DM = DeformableMirror(telescope    =self.TEL,
                                   nSubap       =self.N_SUBAPERTURE_pyr,
                                   mechCoupling = self.MECH_COUPLING)

        self.DM_pyr = DeformableMirror(telescope    = self.TEL,
                                       nSubap       = self.N_SUBAPERTURE_pyr,
                                       mechCoupling = self.MECH_COUPLING)

        self.DM_zer = DeformableMirror(telescope    = self.TEL,
                                       nSubap       = self.N_SUBAPERTURE_zer,
                                       mechCoupling = self.MECH_COUPLING)
        self.data_shape = self.DM_zer.nAct

        N_1 = np.squeeze(self.DM_pyr.modes[self.TEL.pupilLogical, :])
        N_2_inv = np.linalg.pinv(np.squeeze(self.DM_zer.modes[self.TEL.pupilLogical, :]))
        self.N2N1 = np.matmul(N_2_inv, N_1)
        #---------------------------------------------------WFS---------------------------------------------------#

        self.PWFS = Pyramid(nSubap    = self.N_SUBAPERTURE_pyr,
                       telescope      = self.TEL,
                       modulation     = self.MODULATION,
                       lightRatio     = self.LIGHT_RATIO,
                       postProcessing = self.POST_PROCESS)


        #---------------------------------------------------KL_MODAL_BASIS---------------------------------------------------#
        full_M2C = compute_KL_basis(self.TEL,
                                    self.ATM,
                                    self.DM,
                                    lim=0)  # inversion stability criterion

        # compute KL modal projector truncating the modes by the pupil used
        KL_basis_dm = (self.DM.modes @ full_M2C) * np.tile(self.TEL.pupil.flatten()[:, None], full_M2C.shape[1])
        self.projector_kl = np.linalg.pinv(KL_basis_dm)

        M2C_KL_pyr   = compute_KL_basis(tel = self.TEL, atm = self.ATM, dm = self.DM_pyr)
        self.M2C_pyr = M2C_KL_pyr[:, :self.KL_coefs_pyr]
        modes_pyr    = self.DM_pyr.modes @ self.M2C_pyr

        stroke = self.SRC.wavelength / 16
        #---------------------------------------------------INTERACTION MATRIX---------------------------------------------------#

        self.PWFS.cam.photonNoise = self.pwfs_photNoise
        self.PWFS.cam.readoutNoise = self.pwfs_readNoise
        # self.PWFS.cam.darkCurrent = 100 #for some reason the integration time is not automatically retrieved from the telescope object
        self.PWFS.cam.QE = self.pwfs_quanteff

        self.CALIB_pyr = InteractionMatrix(ngs               = self.NGS,
                                               tel           = self.TEL,
                                               dm            = self.DM_pyr,
                                               wfs           = self.PWFS,
                                               M2C           = self.M2C_pyr,
                                               atm           = self.ATM,
                                               nMeasurements = 1,
                                               stroke        = stroke,
                                               noise         = self.noise_calibration)

        """# ---------------------------------------------------FITTING ERROR CALC---------------------------------------------------#
        self.modes_inv_pyr = np.linalg.pinv(modes_pyr)
        self.modes_inv_zer = np.linalg.pinv(modes_zer)"""

        #---------------------------------------------------INITIALIZATION---------------------------------------------------#
        self.ATM.generateNewPhaseScreen(seed = self.seed)
        self.atm_update_tracker = 0
        self.SRC.reset()
        self.DM_pyr.coefs = 0
        self.DM_zer.coefs = 0
        self.DM_zer_copy  = 0
        self.SRC ** self.ATM * self.TEL * self.DM_pyr * self.PWFS
        self.SRC ** self.ATM * self.TEL * self.DM_pyr * self.DM_zer
        self.TEL.print_optical_path()


        self.CL_gain_pyr = 0.4
        self.CL_gain_zer = 0.4


        self.CURRENT_STEPS = 0 #current steps seems to just be for tracking loop progress
        self.scale_up = 1e7
        self.scale_down = 1e-7


        # CRUCIAL PIXEL SIZE CHECK#
        self.pixel_size = self.DIAMETER / self.RESOLUTION
        if (3 * self.pixel_size) > self.r_0:
            raise SystemExit("ERROR: pixel size is too big for r_0 value")


        #takes in wfs signal and outputs controle
        self.reconstructor_pyr = self.M2C_pyr @ self.CALIB_pyr.M
        self.reconstructor_zer = np.linalg.pinv(self.DM_zer.modes * np.tile(self.TEL.pupil.flatten()[:,None], self.DM_zer.modes.shape[1]))

        self.PWFS_signal_avg = 0
        self.PWFS_signal_avg_norm = 0

        self.PWFS.cam.photonNoise = self.pwfs_photNoise
        self.PWFS.cam.readoutNoise = self.pwfs_readNoise
        # self.PWFS.cam.darkCurrent = 100 #for some reason the integration time is not automatically retrieved from the telescope object
        self.PWFS.cam.QE = self.pwfs_quanteff

        mask_1D = self.DM_zer.validAct.copy()
        mask_reshaped = mask_1D.reshape((self.N_SUBAPERTURE_zer + 1, self.N_SUBAPERTURE_zer + 1))
        self.mask = torch.from_numpy(mask_reshaped).bool()

        self.strehl_1st      = 0
        self.modes_atm       = 0
        self.modes_1st_stage = 0
        self.modes_2nd_stage = 0


    def flatten_dm(self):
        self.DM_pyr.coefs = 0
        self.DM_zer.coefs = 0
        self.DM_zer_copy  = 0

        # propagate to wfs and apply new dm commands to the dm
        self.SRC ** self.ATM * self.TEL * self.DM_pyr * self.PWFS
        self.SRC ** self.ATM * self.TEL * self.DM_pyr * self.DM_zer

        vzwfs_signal_proj = self.reconstructor_zer @ self.SRC.OPD.flatten()

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

        print(f'RESET IS BEING CALLED WITH SEED {seed}')
        self.ATM.generateNewPhaseScreen(seed=seed)
        self.atm_update_tracker = 0
        self.ATM.update()


        self.CURRENT_STEPS      = 0
        self.pwfssignal_buffer  = [np.zeros(self.PWFS.nSignal)] #2frame delay pwfs
        self.vzwfssignal_buffer = [torch.zeros((self.N_SUBAPERTURE_zer + 1, self.N_SUBAPERTURE_zer + 1))] #2 frame delay zwfs

        self.DM_pyr.coefs = 0
        self.DM_zer.coefs = 0
        self.DM_zer_copy = 0
        self.DM_prev_coefs_zer = self.DM_zer.coefs.copy()


        self.PWFS_signal_avg = 0
        self.PWFS_signal_avg_norm = 0


        # propagate to wfs and apply new dm commands to the dm
        self.SRC ** self.ATM * self.TEL * self.DM_pyr * self.PWFS
        self.SRC ** self.ATM * self.TEL * self.DM_pyr * self.DM_zer

        vzwfs_signal_proj = self.reconstructor_zer @ self.SRC.OPD.flatten()
        # redoing in 2D
        vzwfs_signal_proj_2D = torch.zeros((self.N_SUBAPERTURE_zer + 1, self.N_SUBAPERTURE_zer + 1), dtype=torch.float32)
        vzwfs_signal_proj_2D[self.mask] = torch.from_numpy(vzwfs_signal_proj).float()
        next_state = vzwfs_signal_proj_2D


        #change to "tt_modes"
        INFO        = {"KL_modes": "vzwfs_KL_modes"} #tip tilt zernike coefs

        return next_state, INFO


    @torch.no_grad()
    def step(self, action, pyramid_noise):
        self.ATM.update()
        self.atm_update_tracker = self.atm_update_tracker + 1
        atm_opd = self.ATM.OPD.copy()


        total_error = np.std(self.ATM.OPD[np.where(self.TEL.pupil > 0)]) * 1e9


        # PWFS slope averaging over 3 frames
        self.PWFS_signal_avg += (self.PWFS.signal_2D + self.PWFS.referenceSignal_2D) * self.PWFS.norma
        self.PWFS_signal_avg_norm += self.PWFS.norma

        if (self.CURRENT_STEPS + 1) % 3 == 0:
            # PWFS slope averaging over 3 frames (should be implemented correctly, but if something goes wrong you can turn this off and check)
            self.PWFS_signal_avg = self.PWFS_signal_avg / self.PWFS_signal_avg_norm - self.PWFS.referenceSignal_2D
            self.PWFS_signal_avg = self.PWFS_signal_avg[np.where(self.PWFS.validSignal == 1)]

            # frame delay implementation
            self.pwfssignal_buffer.append(self.PWFS_signal_avg)
            pwfs_delayed_signal = self.pwfssignal_buffer[0]
            self.pwfssignal_buffer.pop(0)

            self.DM_pyr.coefs = self.DM_pyr.coefs - (self.CL_gain_pyr * np.matmul(self.reconstructor_pyr, pwfs_delayed_signal) + pyramid_noise)


            self.PWFS_signal_avg = 0
            self.PWFS_signal_avg_norm = 0


        # frame delay implementation (zernike)
        self.vzwfssignal_buffer.append(action)
        self.vzwfs_delayed_signal = self.vzwfssignal_buffer[0]
        self.vzwfssignal_buffer.pop(0)


        self.DM_zer_coefs = self.DM_zer_copy - self.vzwfs_delayed_signal[self.mask].detach().cpu().numpy()
        self.DM_zer_copy = self.DM_zer_coefs.copy()
        #woofer-tweeter
        self.DM_zer.coefs = self.DM_zer_copy - np.matmul(self.N2N1, self.DM_pyr.coefs)

        self.SRC ** self.ATM * self.TEL
        self.modes_atm = self.projector_kl @ self.SRC.OPD.flatten()

        # propagate to wfs and apply new dm commands to the dm
        #only activate the second stage after the first stage closes the loop
        self.SRC ** self.ATM * self.TEL * self.DM_pyr * self.PWFS
        self.modes_1st_stage = self.projector_kl @ self.SRC.OPD.flatten()
        if (self.CURRENT_STEPS + 1) % 3 == 0:
            self.strehl_1st = np.exp(-np.var(self.TEL.src.phase[np.where(self.TEL.pupil == 1)]))
        self.SRC ** self.ATM * self.TEL * self.DM_pyr * self.DM_zer
        self.modes_2nd_stage = self.projector_kl @ self.SRC.OPD.flatten()
        DM_zer_OPD = self.DM_zer.OPD * self.TEL.pupil


        vzwfs_signal_torch_proj = self.reconstructor_zer @ self.SRC.OPD.flatten()

        #redoing in 2D the zwfs signal
        vzwfs_signal_torch_proj_2D = torch.zeros((self.N_SUBAPERTURE_zer + 1, self.N_SUBAPERTURE_zer + 1), dtype = torch.float32)
        vzwfs_signal_torch_proj_2D[self.mask] = torch.from_numpy(vzwfs_signal_torch_proj).float()
        next_state = vzwfs_signal_torch_proj_2D


        strehl = np.exp(-np.var(self.TEL.src.phase[np.where(self.TEL.pupil == 1)]))
        residual_error = np.std(self.TEL.OPD[np.where(self.TEL.pupil > 0)]) * 1e9

        self.CURRENT_STEPS += 1

        self.TEL.computePSF(self.zeroPaddingFactor)
        INFO = {"strehl": strehl, "strehl_1st": self.strehl_1st,
                "residual_error": residual_error, "total_error": total_error,
                "modes_1st_stage": self.modes_1st_stage, "modes_2nd_stage": self.modes_2nd_stage,
                "modes_atm": self.modes_atm, "tracker": self.atm_update_tracker,
                "DM_zer_OPD": DM_zer_OPD}


        return next_state, INFO




























