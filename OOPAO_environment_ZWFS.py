import gymnasium as gym
import numpy as np
from numpy.fft import fftshift, fft, fft2, fftfreq, rfft, rfftfreq
import torch
import os
from functions import *
import matplotlib.pyplot as plt

import OOPAO
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Pyramid import Pyramid
from OOPAO.ZWFS import ZWFS
from OOPAO.ZWFS2 import ZWFS2
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.Zernike import Zernike
from OOPAO.Detector import Detector
from OOPAO.calibration.CalibrationVault import CalibrationVault

"""
code adapted from "Self-optimizing adaptive optics control with Reinforcement Learning"
"""


class OOPAO_environment_ZWFS(gym.Env):
    def __init__(self):
        """
        * need to initialise all of the variables
        * clear observation space and action space definitions
        * what variables do my functions need
        * how do I implement the pwfs signal and DM coefficients as input due to shape mismatch?
        * need to keep track of rewards
        * again initialise all of the OOPAO variables
        """

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.N_SUBAPERTURE_pyr = 20
        self.N_SUBAPERTURE_zer = 9
        self.DIAMETER = 1.52
        self.RESOLUTION = self.N_SUBAPERTURE_pyr * 8
        self.FREQUENCY = 1500
        self.MODULATION = 3
        self.LIGHT_RATIO = 0.1
        self.POST_PROCESS = "slopesMaps"
        self.r_0 = 0.15
        self.L_0 = 25
        self.WIND_SPEED = [10, 20, 60]
        self.WIND_DIRECTION = [0, 100, 160]
        self.FRACTIONAL_C_N2 = [0.6, 0.3, 0.1]
        self.ALTITUDE = [0, 4500, 10000]
        self.NGS_MAGNITUDE = 2
        self.NGS_OptBand   = "I"
        self.SRC_MAGNITUDE = 2
        self.SRC_OptBand = "I"
        self.CENTRAL_OBSTRUCTION = 0
        self.MECH_COUPLING = 0.35
        self.KL_no = 250
        self.nLOOP = 1536
        self.zeroPaddingFactor = 4
        self.seed = 2

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
        x_axis, OTF_dl_averaged = circular_average((np.abs(OTF_dl)).shape, np.abs(OTF_dl))

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

        self.DM_zer = DeformableMirror(telescope    = self.TEL,
                                       nSubap       = self.N_SUBAPERTURE_zer,
                                       mechCoupling = self.MECH_COUPLING)

        #---------------------------------------------------WFS---------------------------------------------------#

        self.PWFS = Pyramid(nSubap    = self.N_SUBAPERTURE_pyr,
                       telescope      = self.TEL,
                       modulation     = self.MODULATION,
                       lightRatio     = self.LIGHT_RATIO,
                       postProcessing = self.POST_PROCESS)

        self.vZWFS = ZWFS2(tel     =self.TEL,
                                 zpf     =8,
                                 diameter=1.06)

        #---------------------------------------------------KL_MODAL_BASIS---------------------------------------------------#
        M2C_KL_pyr   = compute_KL_basis(tel = self.TEL, atm = self.ATM, dm = self.DM_pyr)
        self.M2C_pyr = M2C_KL_pyr[:, :self.KL_no]
        modes_pyr    = self.DM_pyr.modes @ self.M2C_pyr

        M2C_KL_zer   = compute_KL_basis(tel = self.TEL, atm = self.ATM, dm = self.DM_zer)
        self.M2C_zer = M2C_KL_zer[:, :self.KL_no]
        modes_zer    = self.DM_zer.modes @ self.M2C_zer

        stroke = self.SRC.wavelength / 16
        #---------------------------------------------------INTERACTION MATRIX---------------------------------------------------#

        self.CALIB_pyr = InteractionMatrix(ngs      =self.NGS,
                                               tel           =self.TEL,
                                               dm            =self.DM_pyr,
                                               wfs           =self.PWFS,
                                               M2C           =self.M2C_pyr,
                                               atm           =self.ATM,
                                               nMeasurements = 1,
                                               stroke        = stroke)

        #ZWFS calibration
        self.TEL - self.ATM
        CALIB_zer = np.zeros((self.vZWFS.signal.shape[0], self.M2C_zer.shape[1]))

        #zernike_wfs calibration with DM_zer
        for i in range(self.M2C_zer.shape[1]):
            v_plus = self.M2C_zer[:, i] * stroke
            v_minus = -self.M2C_zer[:, i] * stroke

            self.TEL.resetOPD()
            self.DM_zer.coefs = v_plus
            self.SRC * self.TEL * self.DM_zer * self.vZWFS
            w_plus = self.vZWFS.signal

            self.TEL.resetOPD()
            self.DM_zer.coefs = v_minus
            self.SRC * self.TEL * self.DM_zer * self.vZWFS
            w_minus = self.vZWFS.signal

            CALIB_zer[:, i] = (w_plus - w_minus) / (2 * stroke)

        # takes in modes and outputs wfs signal
        self.CALIB_zer_obj = CalibrationVault(CALIB_zer)

        # ---------------------------------------------------FITTING ERROR CALC---------------------------------------------------#
        self.modes_inv = np.linalg.pinv(np.squeeze(modes_pyr[self.TEL.pupilLogical, :]))

        #---------------------------------------------------INITIALIZATION---------------------------------------------------#
        self.ATM.generateNewPhaseScreen(seed = self.seed)
        self.TEL.resetOPD()
        self.DM_pyr.coefs = np.zeros(self.DM_pyr.coefs.shape)
        self.DM_zer.coefs = np.zeros(self.DM_zer.coefs.shape)
        self.TEL + self.ATM
        self.SRC * self.TEL * self.DM_pyr * self.PWFS * self.DM_zer * self.vZWFS
        self.TEL.print_optical_path()






        self.CL_gain_pyr = 0.4
        self.CL_gain_zer = 0.4
        self.DELAY = 2  # 2 frame delay



        self.N_HISTORY = 15 #past DM_zer + vZWFS measurements (2x factor is in the code later)
        self.CURRENT_STEPS = 0 #current steps seems to just be for tracking loop progress
        self.scale_up = 1e7
        self.scale_down = 1e-7

        # CRUCIAL PIXEL SIZE CHECK#
        self.pixel_size = self.DIAMETER / self.RESOLUTION

        if (3 * self.pixel_size) > self.r_0:
            print("WARNING: pixel size is too big for r_0 value")



        #define what the agent can observe (PHASE + DM commands)
        #this is policy only btw
        self.observation_space = gym.spaces.Box(
            low   = -np.inf,
            high  = np.inf,
            shape = (2 * self.N_HISTORY, self.DM_zer.nAct, self.DM_zer.nAct),
            dtype = np.float32
        )

        self.action_space = gym.spaces.Box(
            low   = -1,
            high  = 1,
            shape = (self.DM_zer.nAct, self.DM_zer.nAct), # either number of actuators or zernike coefs
            dtype = np.float32
        )

        #takes in wfs signal and outputs controle
        self.reconstructor_pyr = self.M2C_pyr @ self.CALIB_pyr.M
        self.reconstructor_zer = self.M2C_zer @ self.CALIB_zer_obj.M

    def reset(self, seed = None, options = None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        if seed is None:
            seed = np.random.randint(1e9)

        self.ATM.generateNewPhaseScreen(seed=seed)
        self.TEL * self.PWFS

        self.REWARD            = 0
        self.TIME              = 0
        self.CURRENT_STEPS     = 0
        self.pwfssignal_buffer = [np.zeros(self.PWFS.nSignal)] #2frame delay
        self.N_HISTORY_BUFFER  = torch.zeros((2 * self.N_HISTORY, self.DM_zer.nAct, self.DM_zer.nAct)).to(self.device) #this will act as the input into the actor network

        self.DM_pyr.coefs = 0
        self.DM_zer.coefs = 0
        self.DM_prev_coefs_zer = self.DM_zer.coefs.copy()


        vzwfs_KL_modes = self.CALIB_zer_obj.M @ self.vZWFS.signal


        self.PWFS.signal  = 0  # state
        self.DM_pyr.coefs = 0  # action
        self.DM_zer.coefs = 0  # action

        # propagate to wfs and apply new dm commands to the dm
        self.SRC * self.TEL * self.DM_pyr * self.PWFS
        self.TEL * self.DM_zer * self.vZWFS
        self.SRC * self.TEL

        OBSERVATION = self.N_HISTORY_BUFFER.clone().detach().cpu().numpy()
        #change to "tt_modes"
        INFO        = {"KL_modes": vzwfs_KL_modes} #tip tilt zernike coefs

        return OBSERVATION, INFO

    def step(self, action, predicted_state):
        self.ATM.update()
        atm_opd = self.ATM.OPD.copy()

        total_error = np.std(self.TEL.OPD[np.where(self.TEL.pupil > 0)]) * 1e9

        if self.CURRENT_STEPS % 3 == 0:
            self.pwfssignal_buffer.append(self.PWFS.signal)
            pwfs_delayed_signal = self.pwfssignal_buffer[0]
            self.pwfssignal_buffer.pop(0)

            self.DM_pyr.coefs = self.DM_pyr.coefs - self.CL_gain_pyr * np.matmul(self.reconstructor_pyr, pwfs_delayed_signal)
            DM_pyr_copy = self.DM_pyr.coefs.copy()



        #the mask is applied by the network
        #but it is still 2D output so I reapply the mask here and, thus get a flat one
        #the action here should have a shape of (10, 10) and if not then uhm...
        mask = np.reshape(self.DM_zer.validAct, (10, 10))
        self.DM_zer.coefs = self.DM_zer.coefs + action[mask] * self.scale_down
                                #the policy should output the 2D of actuator values so we just flatten them here (joining rows)
                                #also apply the mask somewhere hmm
                                #check for shape mismatch
                                #should you rewrite this as a residual change to the DM commands
                                #instead of the actual DM commands?
        self.DM_prev_coefs_zer = self.DM_zer.coefs.copy()


        # propagate to wfs and apply new dm commands to the dm
        self.SRC * self.TEL * self.DM_pyr * self.PWFS
        self.TEL * self.DM_zer * self.vZWFS
        self.SRC * self.TEL


        #updates the history buffer with new pwfs signal and the previous dm commands
        """self.N_HISTORY_BUFFER    = torch.roll(self.N_HISTORY_BUFFER, 2, 0)
        vzwfs_signal_torch_proj  = self.reconstructor_zer @ self.vZWFS.signal
        vzwfs_KL_modes           = self.CALIB_zer_obj.M @ self.vZWFS.signal
        vzwfs_signal_torch       = torch.tensor(vzwfs_signal_torch_proj, dtype=torch.float32, device=self.device)
        DM_prev_coefs_zer_torch  = torch.tensor(self.DM_prev_coefs_zer, dtype = torch.float32, device = self.device)
        self.N_HISTORY_BUFFER[0] = vzwfs_signal_torch #projected onto DM
        self.N_HISTORY_BUFFER[1] = DM_prev_coefs_zer_torch  #DM commands sent by the policy"""


        vzwfs_signal_torch_proj = self.reconstructor_zer @ self.vZWFS.signal
        vzwfs_KL_modes = self.CALIB_zer_obj.M @ self.vZWFS.signal

        #redoing in 2D
        vzwfs_signal_torch_proj_2D = np.zeros((10, 10))
        vzwfs_signal_torch_proj_2D[mask] = vzwfs_signal_torch_proj
        next_state = vzwfs_signal_torch_proj_2D

        #past obs and actions + current?
        OBSERVATION = self.N_HISTORY_BUFFER.clone().detach()

        #reward time (do you need to turn it into a tensor? the backprop is not going through it so no?)
        #so the reward here should be at least a sum over the horizon (and better over every state)
        residual_volt = predicted_state #output of the dynamics model
        REWARD        = - np.linalg.norm(residual_volt.cpu()) ** 2


        STREHL = np.exp(-np.var(self.TEL.src.phase[np.where(self.TEL.pupil == 1)]))
        residual_error = np.std(self.TEL.OPD[np.where(self.TEL.pupil > 0)]) * 1e9

        self.CURRENT_STEPS += 1

        TERMINATED = 0
        TRUNCATED = self.CURRENT_STEPS >= self.nLOOP

        self.TEL.computePSF(self.zeroPaddingFactor)
        INFO = {"KL_modes": vzwfs_KL_modes, "strehl": STREHL,
                "residual_err": residual_error, "total_err": total_error,
                "TEL_PSF": self.TEL.PSF, "residual_OPD": self.TEL.OPD,
                "atm_OPD": atm_opd}

        if TRUNCATED:
            self.CURRENT_STEPS = 0


        return OBSERVATION, REWARD, bool(TERMINATED), bool(TRUNCATED), INFO


    def step_mod(self, action, integrator = False):
        self.ATM.update()
        atm_opd = self.ATM.OPD.copy()

        total_error = np.std(self.TEL.OPD[np.where(self.TEL.pupil > 0)]) * 1e9

        if self.CURRENT_STEPS % 3 == 0:
            self.pwfssignal_buffer.append(self.PWFS.signal)
            pwfs_delayed_signal = self.pwfssignal_buffer[0]
            self.pwfssignal_buffer.pop(0)

            self.DM_pyr.coefs = self.DM_pyr.coefs - self.CL_gain_pyr * np.matmul(self.reconstructor_pyr, pwfs_delayed_signal)
            DM_pyr_copy = self.DM_pyr.coefs.copy()



        #the mask is applied by the network
        #but it is still 2D output so I reapply the mask here and, thus get a flat one
        #the action here should have a shape of (10, 10) and if not then uhm...
        mask = np.reshape(self.DM_zer.validAct, (10, 10))
        self.DM_zer.coefs = self.DM_zer.coefs + action[mask].detach().cpu().numpy()
        self.DM_prev_coefs_zer = self.DM_zer.coefs.copy()


        # propagate to wfs and apply new dm commands to the dm
        self.SRC * self.TEL * self.DM_pyr * self.PWFS
        self.TEL * self.DM_zer * self.vZWFS
        self.SRC * self.TEL


        vzwfs_signal_torch_proj = self.reconstructor_zer @ self.vZWFS.signal
        vzwfs_KL_modes = self.CALIB_zer_obj.M @ self.vZWFS.signal


        #redoing in 2D the zwfs signal
        mask_torch = torch.from_numpy(mask)
        vzwfs_signal_torch_proj_2D = torch.zeros((10, 10), dtype = torch.float32)
        vzwfs_signal_torch_proj_2D[mask_torch] = torch.from_numpy(vzwfs_signal_torch_proj).float()
        next_state = vzwfs_signal_torch_proj_2D



        #so the reward here should be at least a sum over the horizon (and better over every state)
        REWARD = 0#- np.linalg.norm(next_state.cpu()) ** 2


        STREHL = np.exp(-np.var(self.TEL.src.phase[np.where(self.TEL.pupil == 1)]))
        residual_error = np.std(self.TEL.OPD[np.where(self.TEL.pupil > 0)]) * 1e9

        self.CURRENT_STEPS += 1

        TERMINATED = 0
        TRUNCATED = self.CURRENT_STEPS >= self.nLOOP

        self.TEL.computePSF(self.zeroPaddingFactor)
        INFO = {"KL_modes": vzwfs_KL_modes, "strehl": STREHL,
                "residual_err": residual_error, "total_err": total_error,
                "TEL_PSF": self.TEL.PSF, "residual_OPD": self.TEL.OPD,
                "atm_OPD": atm_opd}

        if TRUNCATED:
            self.CURRENT_STEPS = 0


        return next_state, REWARD, bool(TERMINATED), bool(TRUNCATED), INFO


























