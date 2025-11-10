import gymnasium as gym
import numpy as np
import OOPAO
import torch
import os

import OOPAO
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Pyramid import Pyramid
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.Zernike import Zernike
from OOPAO.Detector import Detector
from OOPAO.calibration.CalibrationVault import CalibrationVault

"""
code adapted from "Self-optimizing adaptive optics control with Reinforcement Learning"
"""


class OOPAO_environment(gym.Env):
    def __init__(self):
        """
        * need to initialise all of the variables
        * clear observation space and action space definitions
        * what variables do my functions need
        * how do I implement the pwfs signal and DM coefficients as input due to shape mismatch?
        * need to keep track of rewards
        * the training takes place outside of the environment
        * again initialise all of the OOPAO variables

        * NOT SURE IF THE FRAME DELAY HAS BEEN IMPLEMENTED CORRECTLY
        """

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.N_SUBAPERTURE = 20
        self.DIAMETER = 1.52
        self.RESOLUTION = self.N_SUBAPERTURE * 8
        self.FREQUENCY = 1000
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
        self.J_zer = 300
        self.J_corr= 50
        self.nLOOP = 500
        self.zeroPaddingFactor = 6

        # SOURCE#

        self.NGS = Source(optBand=self.NGS_OptBand,
                     magnitude=self.NGS_MAGNITUDE)

        self.SRC = Source(optBand=self.SRC_OptBand,
                     magnitude=self.SRC_MAGNITUDE)

        # TELESCOPE#
        self.TEL = Telescope(resolution=self.RESOLUTION,
                        diameter=self.DIAMETER,
                        samplingTime=1 / self.FREQUENCY,
                        centralObstruction=self.CENTRAL_OBSTRUCTION)

        self.SRC * self.TEL

        # compute PSF
        #self.TEL.computePSF(zeroPaddingFactor=self.zeroPaddingFactor)

        # ATMOSPHERE#
        self.ATM = Atmosphere(telescope=self.TEL,
                                r0=self.r_0,
                                L0=self.L_0,
                                windSpeed=self.WIND_SPEED,
                                windDirection=self.WIND_DIRECTION,
                                fractionalR0=self.FRACTIONAL_C_N2,
                                altitude=self.ALTITUDE
                                )

        self.ATM.initializeAtmosphere(telescope=self.TEL)
        self.TEL + self.ATM

        # SCIENCE DETECTOR



        # define a detector with its properties (see Detector class for further documentation)
        self.CAM = Detector(integrationTime=self.TEL.samplingTime,  # integration time of the detector
                            photonNoise=False,  # enable photon noise
                            readoutNoise=0,  # readout of the detector in [e-/pixel]
                            QE=1,  # quantum efficiency
                            psf_sampling=2,  # sampling for the PSF computation 2 = Shannon sampling
                            binning=1)  # Binning factor of the PSF

        self.CAM_BINNED = Detector(integrationTime=self.TEL.samplingTime,  # integration time of the detector
                                   photonNoise=True,  # enable photon noise
                                   readoutNoise=2,  # readout of the detector in [e-/pixel]
                                   QE=0.8,  # quantum efficiency
                                   psf_sampling=2,  # sampling for the PSF computation 2 = Shannon sampling
                                   binning=4)  # Binning factor of the PSF

        # computation of a PSF on the detector using the '*' operator
        self.SRC * self.TEL * self.CAM * self.CAM_BINNED

        # %%         PROPAGATE THE LIGHT THROUGH THE ATMOSPHERE
        # The Telescope and Atmosphere can be combined using the '+' operator (Propagation through the atmosphere):
        self.TEL + self.ATM  # This operations makes that the tel.OPD is automatically over-written by the value of atm.OPD when atm.OPD is updated.

        # computation of a PSF on the detector using the '*' operator
        self.ATM * self.SRC * self.TEL * self.CAM * self.CAM_BINNED

        # The Telescope and Atmosphere can be separated using the '-' operator (Free space propagation)
        self.TEL - self.ATM





        # DEFORMABLE_MIRROR #

        self.DM = DeformableMirror(telescope=self.TEL,
                              nSubap=self.N_SUBAPERTURE,
                              mechCoupling=self.MECH_COUPLING)

        self.DM.coefs = 0
        self.DM_prev_coefs = self.DM.coefs.copy()

        #need to separate tel and atm?
        self.TEL - self.ATM
        self.TEL.resetOPD()

        # Pyramid wfs #

        self.PWFS = Pyramid(nSubap=self.N_SUBAPERTURE,
                       telescope=self.TEL,
                       modulation=self.MODULATION,
                       lightRatio=self.LIGHT_RATIO,
                       postProcessing=self.POST_PROCESS)
        self.TEL * self.PWFS

        # ZONAL/MODAL FUNCTIONS #
        self.ZERNIKE = Zernike(telObject=self.TEL,
                          J=self.J_zer)

        self.ZERNIKE.computeZernike(telObject2=self.TEL)

        self.M2C = np.linalg.pinv(np.squeeze(self.DM.modes[self.TEL.pupilLogical, :])) @ self.ZERNIKE.modes

        stroke = self.SRC.wavelength / 16
        # CALIBRATION MATRIX #
        self.CALIBRATION_MATRIX = InteractionMatrix(ngs=self.NGS,
                                               tel=self.TEL,
                                               dm=self.DM,
                                               wfs=self.PWFS,
                                               M2C=self.M2C,
                                               atm=self.ATM,
                                               nMeasurements=5,
                                               stroke = stroke)



        # THIS SPLITTING ONLY WORKS IF YOUR MODES ARE ORTHONORMAL (they are not when you have a central obstruction)
        # doesn't matter that much in the end because RL will take care of this
        #takes in slopes outputs controle
        self.RECONSTRUCTION_wTT = self.M2C[:, self.J_corr:] @ self.CALIBRATION_MATRIX.M[self.J_corr:, :]
        self.RECONSTRUCTION_TT  = self.M2C[:, :self.J_corr] @ self.CALIBRATION_MATRIX.M[:self.J_corr, :]
        self.M2C_TT = self.M2C[:, :self.J_corr]

        # initialize DM commands
        self.TEL.resetOPD()
        self.DM.coefs = 0
        self.DM_prev_coefs = self.DM.coefs.copy()
        self.SRC * self.TEL * self.DM * self.PWFS
        self.PWFS * self.PWFS.focal_plane_camera

        self.ATM.generateNewPhaseScreen(seed=10)

        self.TEL + self.ATM

        self.TEL.computePSF(zeroPaddingFactor=self.zeroPaddingFactor)


        self.GAIN = 0.4
        self.DELAY = 2  # 2 frame delay



        self.N_HISTORY = 5
        self.N_SLOPES  = self.PWFS.nSignal
        self.CURRENT_STEPS = 0
        self.SCALE_DOWN = 1e-6
        self.SCALE_UP = 1e6
        self.SR = []
        self.SR_running = np.zeros(self.nLOOP)
        self.TOTAL_ERROR = np.zeros(self.nLOOP)
        self.RESIDUAL_ERROR = np.zeros(self.nLOOP)
        self.SE_PSF = []
        #self.LE_PSF = np.log10(self.TEL.PSF)
        self.LE_PSFs = []
        self.TIME = 0

        # CRUCIAL PIXEL SIZE CHECK#
        self.pixel_size = self.DIAMETER / self.RESOLUTION

        if (3 * self.pixel_size) > self.r_0:
            print("WARNING: pixel size is too big for r_0 value")



        #define what the agent can observe (PHASE + DM commands)
        self.observation_space = gym.spaces.Box(
            low   = -np.inf,
            high  = np.inf,
            shape = (self.N_HISTORY, (self.N_SLOPES + self.DM.coefs.shape[0])),
            dtype = np.float32
        )

        self.action_space = gym.spaces.Box(
            low   = -1,
            high  = 1,
            shape = (self.J_corr, ), # either number of actuators or zernike coefs
            dtype = np.float32
        )

        self.ACTION_DELAY_LIST = [torch.zeros(self.J_corr).to(self.device) for i in range(self.DELAY)]





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
        self.ACTION_DELAY_LIST = [torch.zeros(self.J_corr).to(self.device, dtype=torch.float32) for i in range(self.DELAY)]
        self.N_HISTORY_BUFFER  = torch.zeros((self.N_HISTORY, (self.N_SLOPES + self.DM.nValidAct))).to(self.device) #this will act as the input into the actor network

        self.DM.coefs = 0
        self.DM_prev_coefs = self.DM.coefs.copy()

        #self.TEL + self.ATM

        tt_modes_residual = torch.tensor(self.CALIBRATION_MATRIX.M[:self.J_corr, :] @ self.PWFS.signal, dtype=torch.float32).to(
            self.device) * self.SCALE_UP

        #self.SRC * self.TEL * self.DM * self.PWFS
        #self.SRC * self.TEL


        self.PWFS.signal = 0  # state
        self.DM.coefs    = 0  # action



        OBSERVATION = self.N_HISTORY_BUFFER.clone().detach().cpu().numpy() #self.PWFS.signal + previous DM commands
        #change to "tt_modes"
        INFO        = {"tt_modes": tt_modes_residual.cpu().numpy()} #tip tilt zernike coefs

        return OBSERVATION, INFO

    def step(self, action):
        self.TIME += 1

        action_tensor = torch.tensor(action, dtype = torch.float32, device = self.device)
        self.ACTION_DELAY_LIST.pop(0)
        self.ACTION_DELAY_LIST.append(action_tensor)
        delayed_action = self.ACTION_DELAY_LIST[0] #these are zernike modes

        self.DM.coefs = self.DM_prev_coefs - self.GAIN * (self.M2C_TT @ delayed_action.cpu().numpy() * self.SCALE_DOWN)
        self.DM_prev_coefs = self.DM.coefs.copy()

        # propagate to wfs and apply new dm commands to the dm
        self.SRC * self.TEL * self.DM * self.PWFS
        self.SRC * self.TEL


        #updates the history buffer with new pwfs signal and the previous dm commands
        self.N_HISTORY_BUFFER    = torch.roll(self.N_HISTORY_BUFFER, 1, 0)
        pwfs_signal_torch        = torch.tensor(self.PWFS.signal, dtype = torch.float32, device = self.device)
        DM_prev_coefs            = torch.tensor(self.DM_prev_coefs, dtype = torch.float32, device = self.device)
        self.N_HISTORY_BUFFER[0] = torch.concatenate((pwfs_signal_torch, DM_prev_coefs))

        OBSERVATION = self.N_HISTORY_BUFFER.clone().detach() #current action and previous dm commands


        tt_modes_residual = torch.tensor(self.CALIBRATION_MATRIX.M[:self.J_corr, :] @ self.PWFS.signal, dtype = torch.float32).to(self.device) * self.SCALE_UP
        REWARD      = - np.linalg.norm(tt_modes_residual.cpu()) ** 2 / self.J_corr #centered PSF or reconstructed phase projected zernike are 0 for tip tilt


        STREHL = np.exp(-np.var(self.TEL.src.phase[np.where(self.TEL.pupil == 1)]))
        self.SR.append(STREHL)

        self.CURRENT_STEPS += 1

        TERMINATED = 0
        TRUNCATED = self.CURRENT_STEPS >= self.nLOOP
        # change to "tt_modes" and then change it in RL_model_test too
        INFO = {"tt_modes": tt_modes_residual.cpu().numpy(), "strehl": STREHL}

        if TRUNCATED:
            self.CURRENT_STEPS = 0

        self.ATM.update()

        return OBSERVATION, REWARD, bool(TERMINATED), bool(TRUNCATED), INFO



"""
update the atmosphere
get the action
update the action buffer
get the delayed action
use the delayed action to calculate the new DM commands
copy the new DM commands to a new variable for safe storage? (seems a bit redundant hmmm)
get new PWFS signal by propagating through the system

"""

























