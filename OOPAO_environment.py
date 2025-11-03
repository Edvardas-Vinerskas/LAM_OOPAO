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
        self.MODULATION = 2
        self.LIGHT_RATIO = 0.1
        self.POST_PROCESS = "slopesMaps"
        self.r_0 = 0.1
        self.L_0 = 25
        self.WIND_SPEED = [10, 20, 60]
        self.WIND_DIRECTION = [0, 100, 160]
        self.FRACTIONAL_C_N2 = [0.6, 0.3, 0.1]
        self.ALTITUDE = [0, 4500, 10000]
        self.NGS_MAGNITUDE = 2
        self.NGS_OptBand   = "I"
        self.SRC_MAGNITUDE = 2
        self.SRC_OptBand = "I"
        self.CENTRAL_OBSTRUCTION = 0.1
        self.MECH_COUPLING = 0.35
        self.J_zer = 300
        self.J_corr= 2
        self.nLOOP = 500
        self.zeroPaddingFactor = 2

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

        self.NGS * self.TEL

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

        # DEFORMABLE_MIRROR#

        self.DM = DeformableMirror(telescope=self.TEL,
                              nSubap=self.N_SUBAPERTURE,
                              mechCoupling=self.MECH_COUPLING)

        self.PWFS = Pyramid(nSubap=self.N_SUBAPERTURE,
                       telescope=self.TEL,
                       modulation=self.MODULATION,
                       lightRatio=self.LIGHT_RATIO,
                       postProcessing=self.POST_PROCESS)

        # ZONAL/MODAL FUNCTIONS#
        self.ZERNIKE = Zernike(telObject=self.TEL,
                          J=self.J_zer)

        self.ZERNIKE.computeZernike(telObject2=self.TEL)

        self.M2C = np.linalg.pinv(np.squeeze(self.DM.modes[self.TEL.pupilLogical, :])) @ self.ZERNIKE.modes


        self.CALIBRATION_MATRIX = InteractionMatrix(ngs=self.NGS,
                                               tel=self.TEL,
                                               dm=self.DM,
                                               wfs=self.PWFS,
                                               M2C=self.M2C,
                                               atm=self.ATM,
                                               nMeasurements=5)


        #REDO THIS BECAUSE THIS SMELS
        self.RECONSTRUCTION_wTT = CalibrationVault(self.CALIBRATION_MATRIX @ self.M2C[:, 2:], invert=True)
        self.RECONSTRUCTION_TT  = CalibrationVault(self.CALIBRATION_MATRIX @ self.M2C[:, :2], invert=True)

        self.GAIN = 0.4
        self.DELAY = 2  # 2 frame delay



        self.N_HISTORY = 5
        self.N_SLOPES  = self.PWFS.nSignal
        self.CURRENT_STEPS = 0

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

        self.REWARD            = 0
        self.TIME              = 0
        self.N_HISTORY_BUFFER  = torch.zeros((self.N_HISTORY, (self.N_SLOPES + self.DM.nValidAct))) #this will act as the input into the actor network
        self.ACTION_DELAY_LIST = [torch.zeros(self.J_corr).to(self.device, dtype=torch.float32) for i in range(self.DELAY)]


        self.TEL + self.ATM

        self.SRC * self.TEL * self.DM * self.PWFS
        self.SRC * self.TEL


        self.PWFS.signal = 0  # state
        self.DM.coefs    = 0  # action



        OBSERVATION = self.N_HISTORY_BUFFER.clone().detach() #self.PWFS.signal + previous DM commands
        INFO        = self.PWFS.signal #tip tilt zernike coefs

        return OBSERVATION, INFO

    def step(self, action):
        #update the atmosphere
        #actor gets its slope and dm measurements
        #critic outputs the residual phase
        #get the reward
        #terminate and truncate conditions
        #additional info
        self.CURRENT_STEPS += 1
        self.ATM.update()

        # propagate to wfs and apply new dm commands to the dm
        self.SRC * self.TEL * self.DM * self.PWFS
        self.SRC * self.TEL


        action_tensor = torch.tensor(action, dtype = torch.float32, device = self.device)


        self.ACTION_DELAY_LIST.pop(0)
        self.ACTION_DELAY_LIST.append(action_tensor)
        delayed_action = self.ACTION_DELAY_LIST[0] #these are zernike modes
        self.DM.coefs = self.DM_prev_coefs - self.GAIN * (self.M2C @ delayed_action) #for now imagine the actor outputs zernike modes
        self.DM_prev_coefs = self.DM.coefs.copy()


        self.N_HISTORY_BUFFER    = torch.roll(self.N_HISTORY_BUFFER, 1, 0)
        pwfs_signal_torch        = torch.tensor(self.PWFS.signal, dtype = torch.float32, device = self.device)
        DM_prev_coefs            = torch.tensor(self.DM_prev_coefs, dtype = torch.float32, device = self.device)
        self.N_HISTORY_BUFFER[0] = torch.concatenate(pwfs_signal_torch, DM_prev_coefs)

        OBSERVATION = self.DM_prev_coefs #current action and previous dm commands
        REWARD      = #centered PSF or reconstructed phase projected zernike are 0 for tip tilt

        TERMINATED = 0
        self.CURRENT_STEPS +=1
        TRUNCATED = self.CURRENT_STEPS >= self.nLOOP

        INFO = "placeholder"


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

























