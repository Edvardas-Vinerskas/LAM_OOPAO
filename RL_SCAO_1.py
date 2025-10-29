import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing
import numpy as np
import torch.optim as optim
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from stable_baselines3.common.buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs import GymEnv, set_gym_backend
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from OOPAOEnv.IM_delayEnv import OOPAO
import gymnasium as gym
import random
from torch.utils.tensorboard import SummaryWriter
import os
from OOPAOEnv.__load__oopao import load_oopao

from tensordict import TensorDict

from numpy.fft import fftshift, fft2 #need to shift just because of formatting

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

import time
from OOPAO.MisRegistration import MisRegistration
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.tools.displayTools import cl_plot, displayMap

import tyro
from dataclasses import dataclass


N_SUBAPERTURE  = 30
DIAMETER       = 1.52
RESOLUTION     = N_SUBAPERTURE * 6
FREQUENCY      = 1000
SAMPLING_TIME  = 1 / FREQUENCY
CENTRAL_OBSTRUCTION = 0.1
FOV            = 10
OPTICAL_BAND_NGS = "I"
MAGNITUDE_NGS  = 4
OPTICAL_BAND_SRC = "I"
MAGNITUDE_SRC  = 8
MODULATION     = 3
LIGHT_RATIO    = 0.1
POST_PROCESS   = "slopesMaps"
R_0            = 0.1
L_0            = 25
WIND_SPEED     = [10, 20, 60]
WIND_DIRECTION = [0, 100, 160]
FRACTIONAL_C_N2= [0.5, 0.3, 0.2]
ALTITUDE       = [0, 4500, 10000]
MECHANICAL_COUPLING = 0.35
nACTUATORS     = N_SUBAPERTURE + 1
N_PIX_SEPARATION = 6
nLoop = 1000
nMODES = 2
DELAY = 2





class OOPAO_env(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    # --------------------------Core gym funtions--------------------------------
    # init BASICALLY CREATES THE ENVIRONMENT YOUR RL WILL TRAIN IN? I.E. THE SIMULATION
    def __init__(self, T=1, seed=0):



        self.T = T
        self.seed = seed
        self.t = 0
        self.n = nMODES #self.args.nModes

        # OOPAO Modules
        self.gainCL = None
        self.atm = None
        self.wfs = None
        self.dm = None
        self.misReg = None
        self.tel = None
        self.src = None
        self.ngs = None
        self.imat = None  # zonal imat
        self.M2C_CL = None
        self.calib_CL = None
        self.mode = None
        self.Z = None
        self.plot_obj = None
        self.display = None
        self.cam = None
        self.cam_binned = None
        self.reconstructor = None
        # OOPAO saved info
        self.SE_PSF = None
        self.LE_PSF = None
        self.LE_PSFs = None
        self.SR = None
        self.total = None
        self.residual = None
        self.wfsSignal = None
        self.OPD = None

        # Jalo
        self.action_buffer = []
        self.done = False
        self.param_file = ""
        self.oopao_path = ""
        self.delay = 1
        self.S2V = None
        self.V2S = None
        self.F = 1
        self.pmat = None
        self.infmat = None
        self.calibConst = 1
        self.name = "OOPAO"

        self.dm_mask = None
        self.nActuator = None
        self.xvalid = None
        self.yvalid = None
        self.slope_res = 1456 #664

        self.leak = 0.99
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = None
        self.net_gain = 0.5
        self.scale_down = 1e-6
        self.scale_up = 1e6

        self.n_history = T
        self.obs_history = torch.zeros((self.n_history, self.slope_res)).to(self.device)

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_history, self.slope_res),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(nMODES,),
            dtype=np.float32
        )

        #self.args.modulation = 3
        #self.args.delay = 2
        #self.args.nLoop = 500

        self.current_steps = 0

        # Set the parameters
        #self.set_params_file(self.args.param_file, self.args.oopao_path)  # set parameter file
        self.set_params()

        self.xvalid, self.yvalid = np.where(self.tel.pupil == 1)
        self.d = DELAY
        self.action_buffer = [torch.zeros((nMODES)).to(device=self.device, dtype=torch.float32)] * self.d

    # reset IS THE FUNCTION RUN AFTER COMPLETING THE MAX NUMBER OF EXPLORATORY STEPS OR SMTH? BASICALLY PUTS YOU BACK AT THE START TO EXPLORE NEW PATHS
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = 0
        self.current_steps = 0
        self.episode_reward_sum = 0  # Initialize in reset
        self.action_buffer = [torch.zeros((nMODES)).to(device=self.device, dtype=torch.float32)] * self.d
        self.obs_history = torch.zeros((self.n_history, self.slope_res)).to(self.device)

        self.dm.coefs = 0
        self.dm_prev = self.dm.coefs.copy()

        if seed is None:
            seed = np.random.randint(1e9)
        self.atm.generateNewPhaseScreen(seed=seed)
        self.tel * self.wfs

        tt_modes = torch.tensor(np.matmul(self.reconstructor_tt, self.get_slopes()), dtype=torch.float32).to(
            self.device)
        tt_modes *= self.scale_up

        slopes = torch.tensor(self.get_slopes(), dtype=torch.float32).to(self.device)

        self.obs_history[0] = slopes

        obs = self.obs_history.clone().detach()

        info = {"tt_modes": tt_modes.cpu().numpy()}

        return obs.cpu().numpy(), info

    # step IS A FUNCTION THAT IS RUN EVERY TIME YOU EXECUTE A NEWLY SELECTED ACTION
    def step(self, action):

        self.t += 1
        # Convert action to tensor and apply delay
        action_tensor = torch.tensor(action, device=self.device, dtype=torch.float32)
        self.action_buffer.append(action_tensor)
        delayed_action = self.action_buffer.pop(0)

        action4DM = self.M2C_tt.cpu().numpy() @ delayed_action.cpu().numpy() * self.scale_down

        self.dm.coefs = (self.dm_prev * self.leak) + self.tensor_to_numpy(action4DM) * self.gainCL
        self.dm_prev = self.dm.coefs.copy()
        dm_shape_modal = self.C2M_tt @ self.dm.coefs.copy()

        self.tel * self.dm * self.wfs
        self.tel * self.wfs

        slopes = torch.tensor(self.get_slopes(), dtype=torch.float32).to(self.device)

        tt_modes = torch.tensor(np.matmul(self.reconstructor_tt, self.get_slopes()), dtype=torch.float32).to(
            self.device)
        tt_modes *= self.scale_up

        self.obs_history = self.roll_buffer(self.obs_history, slopes)

        obs = self.obs_history.clone().detach()

        strehl = self.get_strehl()
        self.SR.append(strehl)

        self.current_steps += 1

        done = self.current_steps >= nLoop
        truncated = done

        reward = -np.linalg.norm(tt_modes.cpu()) ** 2 / nMODES  # Normalize by number of signals
        # reward = np.clip(reward, -1, 1)

        info = {"tt_modes": tt_modes.cpu().numpy(), "dm_shape": dm_shape_modal, "strehl": strehl}
        terminated = 0

        if done:
            self.current_steps = 0

        self.atm.update()

        return obs.cpu().numpy(), reward, bool(terminated), bool(truncated), info

    # ---------------- Simulation initialisation functions ---------------------
    """def set_params_file(self, param_file, oopao_path):
        if param_file != self.param_file:
            self.param_file = param_file
        if oopao_path != self.oopao_path:
            self.oopao_path = oopao_path"""

    def set_params(self, wfs_type="pyramid", modal_basis="zernike", gainCL=0.5):

        self.gainCL = gainCL


        import importlib

        # The file gets executed upon import, as expected.
        #config = importlib.import_module(self.param_file)

        #param = config.initializeParameterFile(args)

        #self.param = param



        # %% -----------------------     TELESCOPE   ----------------------------------

        # create the Telescope object
        self.tel = Telescope(resolution=RESOLUTION,  # resolution of the telescope in [pix]
                             diameter=DIAMETER,  # diameter in [m]
                             samplingTime=SAMPLING_TIME,  # Sampling time in [s] of the AO loop
                             centralObstruction=CENTRAL_OBSTRUCTION,
                             # Central obstruction in [%] of a diameter
                             display_optical_path=False,  # Flag to display optical path
                             fov=FOV)

        # %% -----------------------     NGS   ----------------------------------
        # create the Natural Guide Star object
        self.ngs = Source(optBand=OPTICAL_BAND_NGS,  # Optical band (see photometry.py)
                          magnitude=MAGNITUDE_NGS,  # Source Magnitude
                          coordinates=[0, 0])  # Source coordinated [arcsec,deg]

        # combine the NGS to the telescope using '*'
        self.ngs * self.tel

        # create the Scientific Target object located at 10 arcsec from the  ngs
        self.src = Source(optBand=OPTICAL_BAND_SRC,  # Optical band (see photometry.py)
                          magnitude=MAGNITUDE_SRC,  # Source Magnitude
                          coordinates=[0, 0])  # Source coordinated [arcsec,deg]

        # combine the SRC to the telescope using '*'
        self.src * self.tel

        # check that the ngs and tel.src objects are the same
        self.tel.src.print_properties()

        # compute PSF
        self.tel.computePSF(zeroPaddingFactor=6)

        # %% -----------------------     ATMOSPHERE   ----------------------------------

        # create the Atmosphere object
        self.atm = Atmosphere(telescope=self.tel, \
                              r0=R_0, \
                              L0=L_0, \
                              windSpeed=WIND_SPEED, \
                              fractionalR0=FRACTIONAL_C_N2, \
                              windDirection=WIND_DIRECTION, \
                              altitude=ALTITUDE)
        # initialize atmosphere
        self.atm.initializeAtmosphere(self.tel)

        # The phase screen can be updated using atm.update method (Temporal sampling given by tel.samplingTime)
        self.atm.update()

        # display the atmosphere layers for the sources specified in list_src:
        # self.atm.display_atm_layers(list_src=[self.ngs,self.src])

        # the sources coordinates can be updated on the fly:
        self.src.coordinates = [0, 0]
        # self.atm.display_atm_layers(list_src=[self.ngs,self.src])

        # %% -----------------------     Scientific Detector   ----------------------------------
        from OOPAO.Detector import Detector

        # define a detector with its properties (see Detector class for further documentation)
        self.cam = Detector(integrationTime=self.tel.samplingTime,  # integration time of the detector
                            photonNoise=False,  # enable photon noise
                            readoutNoise=0,  # readout of the detector in [e-/pixel]
                            QE=1,  # quantum efficiency
                            psf_sampling=2,  # sampling for the PSF computation 2 = Shannon sampling
                            binning=1)  # Binning factor of the PSF

        self.cam_binned = Detector(integrationTime=self.tel.samplingTime,  # integration time of the detector
                                   photonNoise=True,  # enable photon noise
                                   readoutNoise=2,  # readout of the detector in [e-/pixel]
                                   QE=0.8,  # quantum efficiency
                                   psf_sampling=2,  # sampling for the PSF computation 2 = Shannon sampling
                                   binning=4)  # Binning factor of the PSF

        # computation of a PSF on the detector using the '*' operator
        self.src * self.tel * self.cam * self.cam_binned

        # %%         PROPAGATE THE LIGHT THROUGH THE ATMOSPHERE
        # The Telescope and Atmosphere can be combined using the '+' operator (Propagation through the atmosphere):
        self.tel + self.atm  # This operations makes that the tel.OPD is automatically over-written by the value of atm.OPD when atm.OPD is updated.

        # computation of a PSF on the detector using the '*' operator
        self.atm * self.ngs * self.tel * self.cam * self.cam_binned

        # The Telescope and Atmosphere can be separated using the '-' operator (Free space propagation)
        self.tel - self.atm

        # %% -----------------------     DEFORMABLE MIRROR   ----------------------------------
        # mis-registrations object
        """misReg = MisRegistration()
        misReg.shiftX = param['MisReg_shiftX']  # in [m]
        misReg.shiftY = param['MisReg_shiftY']  # in [m]
        misReg.rotationAngle = param['MisReg_rotationAngle']  # in [deg]"""

        # Get valid Actuators
        self.nActuator = nACTUATORS #param['nActuator']

        # if no coordonates specified, create a cartesian dm
        self.dm = DeformableMirror(telescope=self.tel,  # Telescope
                                   nSubap=N_SUBAPERTURE,
                                   # number of subaperture of the system considered (by default the DM has n_subaperture + 1 actuators to be in a Fried Geometry)
                                   mechCoupling=MECHANICAL_COUPLING,
                                   # Mechanical Coupling for the influence functions
                                   #misReg=misReg,  # Mis-registration associated
                                   coordinates=None,
                                   # coordinates in [m]. Should be input as an array of size [n_actuators, 2]
                                   pitch=self.tel.D / self.nActuator)  # inter actuator distance. Only used to compute the influence function coupling. The default is based on the n_subaperture value.

        # Get Valid Actuators mask (X and Y axis) this si important for the RL
        self.dm_mask = np.reshape(self.dm.validAct, (self.nActuator, self.nActuator))
        (self.xvalid, self.yvalid) = np.nonzero(self.dm_mask)
        # %% -----------------------     PYRAMID WFS   ----------------------------------

        # make sure tel and atm are separated to initialize the PWFS
        self.tel.isPaired = False
        self.tel.resetOPD()

        self.wfs = Pyramid(nSubap=N_SUBAPERTURE,
                           # number of subaperture = number of pixel accros the pupil diameter
                           telescope=self.tel,  # telescope object
                           lightRatio=LIGHT_RATIO,  # flux threshold to select valid sub-subaperture
                           modulation=MODULATION,  # Tip tilt modulation radius
                           binning=1,  # binning factor (applied only on the )
                           n_pix_separation=N_PIX_SEPARATION,
                           # number of pixel separating the different pupils
                           n_pix_edge=2,  # number of pixel on the edges of the pupils
                           postProcessing=POST_PROCESS)  # slopesMaps,

        # Propagate the light to the Wave-Front Sensor
        self.tel * self.wfs
        # %% -----------------------     Modal Basis - Zernike  ----------------------------------
        # %% ZERNIKE Polynomials
        # create Zernike Object
        Z = Zernike(self.tel, 50)
        # compute polynomials for given telescope
        Z.computeZernike(self.tel)

        # mode to command matrix to project Zernike Polynomials on DM
        # M2C_zernike = np.linalg.pinv(np.squeeze(self.dm.modes[self.tel.pupilLogical,:]))@Z.modes

        #M2C_zernike = np.load(os.path.dirname(__file__) + '/manual_m2c.npy')[:, :50]
        M2C_zernike = np.linalg.pinv(np.squeeze(self.dm.modes[self.tel.pupilLogical, :])) @ Z.modes

        #self.M2OPD = np.load(os.path.dirname(__file__) + '/../wf_recon/M2OPD_500modes.npy')[:, :self.args.nModes]
        # %% -----------------------     Calibration: Interaction Matrix  ----------------------------------

        # amplitude of the modes in m
        stroke = self.ngs.wavelength / 16
        # zonal Interaction Matrix
        M2C_zonal = np.eye(self.dm.nValidAct)
        # modal Interaction Matrix for 300 modes
        M2C_modal = M2C_zernike[:, :300]

        self.tel - self.atm
        # zonal interaction matrix
        calib_modal = InteractionMatrix(ngs=self.ngs,
                                        atm=self.atm,
                                        tel=self.tel,
                                        dm=self.dm,
                                        wfs=self.wfs,
                                        M2C=M2C_zonal,  # M2C matrix used
                                        stroke=stroke,  # stroke for the push/pull in M2C units
                                        nMeasurements=6,  # number of simultaneous measurements
                                        noise='off',  # disable wfs.cam noise
                                        display=True,  # display the time using tqdm
                                        single_pass=True)  # only push to compute the interaction matrix instead of push-pull

        # Modal interaction matrix
        # calib_modal.D is the interaction matrix
        # CalibrationVault calculates the pseudo inverse using the M2C_zernike
        # couldn't I calculate the calib_modal with zernike in the first place?
        # so calib_zernike takes in zernike and outputs phase modes
        calib_zernike = CalibrationVault(calib_modal.D @ M2C_zernike)

        #they split the tt for RL control?
        calib_tt = CalibrationVault(calib_modal.D @ M2C_zernike[:, :nMODES])
        # %%
        # %%  ----------------------- Define instrument and WFS path detectors  -----------------------

        # instrument path
        src_cam = Detector(self.tel.resolution * 4)
        src_cam.psf_sampling = 4
        src_cam.integrationTime = self.tel.samplingTime * 1
        # put the scientific target off-axis to simulate anisoplanetism (set to  [0,0] to remove anisoplanetism)
        self.src.coordinates = [0, 0]

        # WFS path
        ngs_cam = Detector(self.tel.resolution)
        ngs_cam.psf_sampling = 4
        ngs_cam.integrationTime = self.tel.samplingTime

        # initialize DM commands
        self.tel.resetOPD()
        self.dm.coefs = 0
        self.dm_prev = self.dm.coefs.copy()
        self.ngs * self.tel * self.dm * self.wfs
        self.wfs * self.wfs.focal_plane_camera

        # Update the r0 parameter, generate a new phase screen for the atmosphere and combine it with the Telescope
        # atm.r0 = 0.15
        self.atm.generateNewPhaseScreen(seed=10)

        self.tel + self.atm

        self.tel.computePSF(4)

        # These are the calibration data used to close the loop
        calib_CL = calib_zernike
        M2C_CL = M2C_zernike
        self.M2C_CL = M2C_CL

        #clearly this will be RL since they are using torch
        self.M2C_tt = torch.from_numpy(M2C_zernike[:, :nMODES]).to(device=self.device, dtype=torch.float32)

        #takes in phase and outputs zernike (I assume tip-tilt modes)
        self.reconstructor_tt = calib_tt.M

        # combine telescope with atmosphere
        self.tel + self.atm
        # initialize DM commands
        self.atm * self.ngs * self.tel * ngs_cam
        self.atm * self.src * self.tel * src_cam

        # allocate memory to save data
        self.SR = []  # np.zeros(param['nLoop'])
        self.total = np.zeros((nLoop))
        self.residual = np.zeros((nLoop))
        self.wfsSignal = np.arange(0, self.wfs.nSignal) * 0
        self.SE_PSF = []
        self.LE_PSF = np.log10(self.tel.PSF)
        self.LE_PSFs = []

        self.reconstructor = M2C_CL @ calib_CL.M
        self.modal_CM = calib_CL.M
        self.C2M_tt = np.linalg.pinv(M2C_CL[:, :nMODES])
        self.F = M2C_CL @ np.linalg.pinv(M2C_CL)

    def set_wfs(self, type="pyramid"):
        if type == "pyramid":
            from OOPAO.Pyramid import Pyramid
            # make sure tel and atm are separated to initialize the PWFS
            self.tel - self.atm

            self.wfs = Pyramid(nSubap=N_SUBAPERTURE, \
                               telescope=self.tel, \
                               modulation=MODULATION, \
                               lightRatio=LIGHT_RATIO, \
                               n_pix_separation=N_PIX_SEPARATION, \
                               #psfCentering=, \
                               postProcessing=POST_PROCESS)

            self.tel * self.wfs

    def set_modalBasis(self, mode="zernike"):
        if mode == "zernike":
            from OOPAO.Zernike import Zernike
            from OOPAO.calibration.CalibrationVault import CalibrationVault
            from OOPAO.calibration.InteractionMatrix import InteractionMatrix

            # create Zernike Object
            Z = Zernike(self.tel, 50)
            # compute polynomials for given telescope
            Z.computeZernike(self.tel)

            # mode to command matrix to project Zernike Polynomials on DM
            M2C_zernike = np.linalg.pinv(np.squeeze(self.dm.modes[self.tel.pupilLogical, :])) @ Z.modes

            # self.dm.coefs = M2C_zernike[:,:10]
            # self.tel*self.dm

            M2C_zonal = np.eye(self.dm.nValidAct)
            # zonal interaction matrix
            self.imat = InteractionMatrix(ngs=self.source, \
                                          atm=self.atm, \
                                          tel=self.tel, \
                                          dm=self.dm, \
                                          wfs=self.wfs, \
                                          M2C=M2C_zonal, \
                                          stroke=1e-9, \
                                          nMeasurements=25, \
                                          noise='off')
            # Modal interaction matrix
            calib_zernike = CalibrationVault(self.imat.D @ M2C_zernike)

            self.M2C_CL = M2C_zernike
            self.calib_CL = calib_zernike

    def set_display(self):
        from OOPAO.tools.displayTools import cl_plot
        self.SE_PSF = []
        self.LE_PSF = np.log10(self.tel.PSF)
        self.plot_obj = cl_plot(list_fig=[self.atm.OPD, self.tel.mean_removed_OPD, self.wfs.cam.frame,
                                          np.log10(self.wfs.get_modulation_frame(radius=1)), [[0, 0], [0, 0]],
                                          [self.dm.coordinates[:, 0], np.flip(self.dm.coordinates[:, 1]),
                                           self.dm.coefs], np.log10(self.tel.PSF), np.log10(self.tel.PSF)], \
                                type_fig=['imshow', 'imshow', 'imshow', 'imshow', 'plot', 'scatter', 'imshow',
                                          'imshow'], \
                                list_title=['Turbulence OPD', 'Residual OPD', 'WFS Detector', 'WFS Modulation Camera',
                                            None, None, None, None], \
                                list_lim=[None, None, None, [-3, 0], None, None, [-4, 0], [-4, 0]], \
                                list_label=[None, None, None, None, ['Time', 'WFE [nm]'], ['DM Commands', ''],
                                            ['Short Exposure PSF', ''], ['Long Exposure_PSF', '']], \
                                n_subplot=[4, 2], \
                                list_display_axis=[None, None, None, None, True, None, None, None], \
                                list_ratio=[[0.95, 0.95, 0.1], [1, 1, 1, 1]], s=20)

    def render(self, current_i, mode='rgb_array'):
        """
        Render and display the images of the WFS in real time.
        """

        return

    def calculate_strehl_AVG(self):
        """Calculates the average strehl ratio for each episode.
        Cleans the strehl array after each episode.
        """
        avg = np.mean(self.SR)
        std = np.std(self.SR)
        self.SR = []

        return avg, std

    def integrator(self):
        return -self.gainCL * np.matmul(self.reconstructor, self.wfsSignal)

    def get_slopes(self):
        return self.wfs.signal

    def get_strehl(self):
        return np.exp(-np.var(self.tel.src.phase[np.where(self.tel.pupil == 1)]))

    def _get_reward(self, slopes, type="volt"):
        if self.S2V is not None and type != "sh":
            res_volt = np.matmul(self.S2V, slopes)
            reward = -1 * np.linalg.norm(res_volt)
        else:
            reward = self.get_strehl()

        return reward

    def vec_to_img(self, action_vec, use_torch=True):
        if use_torch:
            valid_actus = torch.zeros((self.nActuator, self.nActuator)).float().to(self.device)

        else:
            valid_actus = np.zeros((self.nActuator, self.nActuator))

        if len(action_vec.shape) == 2:
            batch_size = action_vec.shape[1]

            valid_actus = torch.zeros((batch_size, self.nActuator, self.nActuator), dtype=torch.float32).to(self.device)

            # Expand indices for batch assignment
            batch_indices = torch.arange(batch_size).unsqueeze(1).to(self.device)  # Shape: (batch_size, 1)

            # Assign each action vector to its respective actuator positions
            valid_actus[batch_indices, self.xvalid, self.yvalid] = action_vec.T  # Transpose to align with batch dim

        # valid_actus[self.xvalid, self.yvalid] = action_vec.clone().detach()

        return valid_actus

    def img_to_vec(self, action):
        # assert len(action.shape) == 2
        if len(action.shape) == 4:
            action_out = action[:, :, self.xvalid, self.yvalid]
        else:
            action_out = action[self.xvalid, self.yvalid]

        return action_out

    def roll_buffer(self, history_tensor, new_image):
        """
        Updates the history tensor with a new image at index 0, shifting the rest.

        Args:
            history_tensor (torch.Tensor): Current history tensor of shape (history, height, width).
            new_image (torch.Tensor): New image tensor of shape (height, width).

        Returns:
            torch.Tensor: Updated history tensor.
        """
        # Shift the tensor elements along the 0th dimension to make space for the new image
        history_tensor = torch.roll(history_tensor, shifts=1, dims=0)

        # Insert the new image at the 0th position
        history_tensor[0] = new_image

        return history_tensor

    def tensor_to_numpy(self, obj):
        """Convert a PyTorch tensor to a NumPy array if it's a tensor."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy()
        return obj

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CL_OOPAO-v0"
    """the environment id of the task"""
    total_timesteps: int = 6000#1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(5e4)
    """the replay memory buffer size"""
    gamma: float = 0.90  #0.88
    """the discount factor gamma"""
    tau: float = 0.00385
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 1e3
    """timestep to start learning"""
    policy_lr: float = 0.00001
    """the learning rate of the policy network optimizer"""
    q_lr: float = 0.001
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 5#2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 9  #3 Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.02#0.01
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    hidden_dim: int = 256




# Change for custom environment
def make_env():
    def thunk():
        env = OOPAO_env()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, hidden_dim=256):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.n = 664#self.env.get_attr("n")[0]
        self.T = 5#self.env.get_attr("T")[0]

        self.hidden_dim = hidden_dim

        self.input_dim = self.n * self.T + 2

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x, a):
        x = torch.cat([x.view(x.shape[0], -1), a], 1)
        x = x.view(x.shape[0], -1)
        x = self.net(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -10


class Actor(nn.Module):
    def __init__(self, env, hidden_dim=256):
        super().__init__()

        self.env = env
        self.n = 664 #self.env.get_attr("n")[0]
        self.T = 5#self.env.get_attr("T")[0]
        self.hidden_dim = hidden_dim

        self.input_dim = self.n * self.T

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )


        self.fc_mean = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, np.prod(env.single_action_space.shape))
        )


        self.fc_logstd = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, np.prod(env.single_action_space.shape))
        )

        # Learnable residual scaling factor
        self.residual_scale = nn.Parameter(1e-4 * torch.ones(1))  # Initialized to 1.0


        # self.fc_mean = nn.Linear(64, np.prod(env.single_action_space.shape))
        # self.fc_logstd = nn.Linear(64, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):

        batch_size, T, n = x.shape
        assert n == self.n, f"Expected input last dim {self.n}, got {n}"
        assert T == self.T, f"Expected input time dim {self.T}, got {T}"

        # Flatten (T, n) into (T * n)
        x = x.view(batch_size, -1)

        x = self.net(x)
        x = x.view(batch_size, -1)

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() # for reparameterization trick (mean + std * N(0,1))


        action = x_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)

        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        # print(f"base_action: {base_action}, {base_action.shape}")
        # print(f"residual_action: {self.residual_scale * residual_action}, {residual_action.shape}")

        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
    poetry run pip install "stable_baselines3==2.0.0a1"
    """
        )

    num_runs = 1
    seeds = [167640, 813868, 168772, 214449,
             9498, 398085, 753264, 331695,
             950521, 715051]

    envs = gym.vector.SyncVectorEnv([make_env()])

    for i in range(num_runs):

        args = tyro.cli(Args, args=[])
        run_name = f"IM_delay_{args.env_id}__{args.exp_name}__{args.seed}__run_{i}__{int(time.time())}"
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"./runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        # Random seed for multiple runs
        args.seed = seeds[i]  # logged seed values

        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # env setup
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        max_action = float(envs.single_action_space.high[0])

        actor = Actor(envs).to(device)
        qf1 = SoftQNetwork(envs).to(device)
        qf2 = SoftQNetwork(envs).to(device)
        qf1_target = SoftQNetwork(envs).to(device)
        qf2_target = SoftQNetwork(envs).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

        best_reward = -np.inf
        # Automatic entropy tuning
        if args.autotune:
            target_entropy = - torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha = log_alpha.exp().item()
            a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
        else:
            alpha = args.alpha

        envs.single_observation_space.dtype = np.float32
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
        )
        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs, _ = envs.reset(seed=args.seed)
        for global_step in range(args.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < args.learning_starts:
                if global_step % 100 == 0:
                    print(f"WARMUP: {global_step}/{int(args.learning_starts)}")
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
                # Warmup with IM actions
                # actions = np.array([-1 * (obs[i]) for i in range(envs.num_envs)])
            else:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for info in infos["final_info"]:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    with open("./train_returns.txt", "a") as f:  # 'a' mode appends to the file
                        f.write(f"global_step={global_step}, episodic_return={info['episode']['r']} \n")

                    if info['episode']['r'] > best_reward and global_step > args.learning_starts:
                        best_reward = info['episode']['r']
                        torch.save({
                            'epoch': global_step + 1,
                            'model_state_dict': actor.state_dict(),
                            # 'ema_model_state_dict': ema_reconstructor.module.state_dict(),
                            'optimizer_state_dict': actor_optimizer.state_dict(),
                            'reward': best_reward,
                        }, os.path.dirname(__file__) + f"/../models/best_model_delay_run_{i}.pth")
                        with open("./train_returns.txt", "a") as f:  # 'a' mode appends to the file
                            f.write(f"Saving Model \n")

                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    #real_next_obs[idx] = infos["final_observation"][idx]
                    real_next_obs[idx] = infos.get("final_observation", next_obs)[idx]
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                        min_qf_next_target).view(-1)

                qf1_a_values = qf1(data.observations, data.actions).view(-1)
                qf2_a_values = qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                nn.utils.clip_grad_norm_(qf1.parameters(), args.max_grad_norm)
                q_optimizer.step()

                if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                            args.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = actor.get_action(data.observations)
                        qf1_pi = qf1(data.observations, pi)
                        qf2_pi = qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                        actor_optimizer.step()

                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(data.observations)
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()

                # update the target networks
                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                if global_step % 1000 == 0:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    print(f"Total steps: {global_step}")
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    if args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        # envs.close()
        writer.close()
# %%


















