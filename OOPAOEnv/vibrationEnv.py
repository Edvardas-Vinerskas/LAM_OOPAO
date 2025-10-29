# -*- coding: utf-8 -*-
"""
Same as twoSin except that the sin waves are given
as modal coeffs to the OPD, then measured by the WFS
and reconstructed into modal coefs for the DM.
"""

from .__load__oopao import load_oopao
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch
import os
from types import SimpleNamespace
import yaml


class OOPAO(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    #--------------------------Core gym funtions--------------------------------
    def __init__(self, T=5, seed=0, delay=3):

        # Load in configuration file
        self.conf_path = os.path.dirname(os.path.abspath(__file__)) + '/../Conf/papyrus_config.yaml'
        
        with open(self.conf_path, 'r') as file:
            conf = yaml.safe_load(file)

        self.args = SimpleNamespace(**conf)

        self.T = T
        self.seed = seed
        self.t = 0
        self.n = self.args.nModes


        # OOPAO Modules
        self.gainCL = None
        self.atm = None
        self.wfs = None
        self.dm = None
        self.misReg = None
        self.tel = None
        self.src = None
        self.ngs = None
        self.imat = None          # zonal imat
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
        self.SR        = None
        self.total     = None
        self.residual  = None
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
        self.slope_res = 664

        self.leak = 0.99
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = None
        self.net_gain = 0.5
        self.scale_down = 1e-7
        self.scale_up = 1e7

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
            shape=(self.args.nModes,),
            dtype=np.float32
            )
        
        self.args.modulation = 3
        self.args.delay = delay
        self.args.nLoop = 500

        self.current_steps = 0


        # Set the parameters
        self.set_params_file(self.args.param_file,self.args.oopao_path) # set parameter file
        self.set_params(self.args)

        self.xvalid, self.yvalid = np.where(self.tel.pupil == 1)
        self.d = delay
        self.action_buffer = [torch.zeros((self.args.nModes)).to(device=self.device, dtype=torch.float32)] * self.d

        # Set up vibrations
        self.vibration = np.zeros((self.args.nModes, self.args.nLoop))
        self.freq_x1, self.freq_x2, self.freq_x3 = 13, 37, 91
        self.freq_y1, self.freq_y2, self.freq_y3 = 11, 43, 87



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = 0
        self.current_steps = 0
        self.episode_reward_sum = 0  # Initialize in reset
        self.action_buffer = [torch.zeros((self.args.nModes)).to(device=self.device, dtype=torch.float32)] * self.d
        self.obs_history = torch.zeros((self.n_history, self.slope_res)).to(self.device)

        self.dm.coefs = 0
        self.dm_prev = self.dm.coefs.copy()

        if seed is None:
            seed = np.random.randint(1e9)

        # Set up the vibration signals
        time = np.arange(self.args.nLoop * 2) / (self.args.nLoop * 2)
        
        self.vibration = np.zeros((self.args.nModes, self.args.nLoop * 2))

        # Tip vibrations
        self.vibration[0, :] = (
            np.sin(2 * np.pi * self.freq_x1 * time + 2 * np.pi * np.random.rand()) +
            np.sin(2 * np.pi * self.freq_x2 * time + 2 * np.pi * np.random.rand()) +
            np.sin(2 * np.pi * self.freq_x3 * time + 2 * np.pi * np.random.rand())
        )

        # Tilt vibrations
        self.vibration[1, :] = (
            np.sin(2 * np.pi * self.freq_y1 * time + 2 * np.pi * np.random.rand()) +
            np.sin(2 * np.pi * self.freq_y2 * time + 2 * np.pi * np.random.rand()) +
            np.sin(2 * np.pi * self.freq_y3 * time + 2 * np.pi * np.random.rand())
        )

        # Scale down the vibrations
        self.vibration *= self.scale_down

        # Set the initial frame of OPD
        tmpOPD = np.zeros_like(self.tel.OPD.copy())
        tmpOPD[self.xvalid, self.yvalid] = self.M2OPD @ self.vibration[:, self.t]
        self.tel.OPD = tmpOPD
        self.tel*self.wfs


        tt_modes = torch.tensor(np.matmul(self.reconstructor_tt,self.get_slopes()), dtype=torch.float32).to(self.device)
        tt_modes *= self.scale_up

        slopes = torch.tensor(self.get_slopes(), dtype=torch.float32).to(self.device)
        self.obs_history[0] = slopes

        obs = self.obs_history.clone().detach()

        info = {"tt_modes":tt_modes.cpu().numpy()}

        return obs.cpu().numpy(), info
    

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

        # Step the OPD
        tmpOPD = np.zeros_like(self.tel.OPD.copy())
        tmpOPD[self.xvalid, self.yvalid] = self.M2OPD @ self.vibration[:, self.t]
        self.tel.OPD = tmpOPD
        
        self.tel*self.dm*self.wfs

        slopes = torch.tensor(self.get_slopes(), dtype=torch.float32).to(self.device)

        print(f"Compted in statement: {np.matmul(self.reconstructor_tt,self.wfs.signal)}")
        print(f"vibration: {self.vibration[:, self.t]}")

        tt_modes = torch.tensor(np.matmul(self.reconstructor_tt,self.get_slopes()), dtype=torch.float32).to(self.device)
        tt_modes *= self.scale_up

        print(f"tt_modes: {tt_modes.cpu().numpy()}")

        self.obs_history = self.roll_buffer(self.obs_history, slopes)

        obs = self.obs_history.clone().detach()

        strehl = self.get_strehl()
        self.SR.append(strehl)

        self.current_steps += 1

        done = self.current_steps >= self.args.nLoop
        truncated = done
       
        reward = -np.linalg.norm(tt_modes.cpu()) ** 2 / self.args.nModes # Normalize by number of signals
        # reward = np.clip(reward, -1, 1)

        info = {"tt_modes": tt_modes.cpu().numpy(), "dm_shape": dm_shape_modal ,"strehl":strehl}
        terminated = 0

        if done:
            self.current_steps = 0

        # Step the OPD
        # tmpOPD = np.zeros_like(self.tel.OPD.copy())
        # tmpOPD[self.xvalid, self.yvalid] = self.M2OPD @ self.vibration[:, self.t]
        # self.tel.OPD = tmpOPD
        # self.tel*self.wfs

        # print(f"[{self.t} tt_modes[0]: {tt_modes[0].cpu().numpy():.2e} | tt_modes[1]: {tt_modes[1].cpu().numpy():.2e} | computes tt: {np.matmul(self.reconstructor_tt,self.get_slopes())}")

        return obs.cpu().numpy(), reward, bool(terminated), bool(truncated), info

    
    # ---------------- Simulation initialisation functions ---------------------
    def set_params_file(self,param_file,oopao_path):
        if param_file != self.param_file:
            self.param_file = param_file
        if oopao_path != self.oopao_path:
            self.oopao_path = oopao_path
    def set_params(self, args,wfs_type = "pyramid", modal_basis = "zernike",gainCL = 0.5):

        self.gainCL = gainCL
        load_oopao(self.oopao_path) # 

        import importlib

        # The file gets executed upon import, as expected.
        config = importlib.import_module(self.param_file)

        param = config.initializeParameterFile(args)

        self.param = param
        
        import time
        from OOPAO.Atmosphere import Atmosphere
        from OOPAO.DeformableMirror import DeformableMirror
        from OOPAO.MisRegistration import MisRegistration
        from OOPAO.Pyramid import Pyramid
        from OOPAO.Source import Source
        from OOPAO.Telescope import Telescope
        from OOPAO.Zernike import Zernike
        from OOPAO.calibration.CalibrationVault import CalibrationVault
        from OOPAO.calibration.InteractionMatrix import InteractionMatrix
        from OOPAO.tools.displayTools import cl_plot, displayMap



        #%% -----------------------     TELESCOPE   ----------------------------------

        # create the Telescope object
        self.tel = Telescope(resolution           = param['resolution'],                      # resolution of the telescope in [pix]
                             diameter             = param['diameter'],                        # diameter in [m]        
                             samplingTime         = param['samplingTime'],                    # Sampling time in [s] of the AO loop
                             centralObstruction   = param['centralObstruction'],              # Central obstruction in [%] of a diameter 
                             display_optical_path = False,                                    # Flag to display optical path
                             fov                  = 1 )      

        #%% -----------------------     NGS   ----------------------------------
        # create the Natural Guide Star object
        self.ngs = Source(optBand     = param['opticalBand'],        # Optical band (see photometry.py)
                          magnitude   = param['magnitude'],        # Source Magnitude
                          coordinates = [0,0])                       # Source coordinated [arcsec,deg]

        # combine the NGS to the telescope using '*'
        self.ngs*self.tel

        # create the Scientific Target object located at 10 arcsec from the  ngs
        self.src = Source(optBand     = param['opticalBand'],           # Optical band (see photometry.py)
                          magnitude   = param['magnitude'],           # Source Magnitude
                          coordinates = [1,0])                          # Source coordinated [arcsec,deg]

        # combine the SRC to the telescope using '*'
        self.src*self.tel

        # check that the ngs and tel.src objects are the same
        self.tel.src.print_properties()

        # compute PSF 
        self.tel.computePSF(zeroPaddingFactor = 6)

        #%% -----------------------     ATMOSPHERE   ----------------------------------

        # create the Atmosphere object
        self.atm=Atmosphere(telescope     = self.tel,\
                            r0            = param['r0'],\
                            L0            = param['L0'],\
                            windSpeed     = param['windSpeed'],\
                            fractionalR0  = param['fractionnalR0'],\
                            windDirection = param['windDirection'],\
                            altitude      = param['altitude'])
        # initialize atmosphere
        self.atm.initializeAtmosphere(self.tel)

        # The phase screen can be updated using atm.update method (Temporal sampling given by tel.samplingTime)
        self.atm.update()

        # display the atmosphere layers for the sources specified in list_src: 
        # self.atm.display_atm_layers(list_src=[self.ngs,self.src])

        # the sources coordinates can be updated on the fly: 
        self.src.coordinates = [0,0]
        # self.atm.display_atm_layers(list_src=[self.ngs,self.src])

        
        #%% -----------------------     Scientific Detector   ----------------------------------
        from OOPAO.Detector import Detector

        # define a detector with its properties (see Detector class for further documentation)
        self.cam = Detector(integrationTime = self.tel.samplingTime,      # integration time of the detector
                       photonNoise     = False,                  # enable photon noise
                       readoutNoise    = 0,                     # readout of the detector in [e-/pixel]
                       QE              = 1,                   # quantum efficiency
                       psf_sampling    = 2,                     # sampling for the PSF computation 2 = Shannon sampling
                       binning         = 1)                     # Binning factor of the PSF

        self.cam_binned = Detector( integrationTime = self.tel.samplingTime,      # integration time of the detector
                               photonNoise     = True,                  # enable photon noise
                               readoutNoise    = 2,                     # readout of the detector in [e-/pixel]
                               QE              = 0.8,                   # quantum efficiency
                               psf_sampling    = 2,                     # sampling for the PSF computation 2 = Shannon sampling
                               binning         = 4)                     # Binning factor of the PSF


        # computation of a PSF on the detector using the '*' operator
        self.src*self.tel*self.cam*self.cam_binned

        
        #%%         PROPAGATE THE LIGHT THROUGH THE ATMOSPHERE
        # The Telescope and Atmosphere can be combined using the '+' operator (Propagation through the atmosphere): 
        self.tel+self.atm # This operations makes that the tel.OPD is automatically over-written by the value of atm.OPD when atm.OPD is updated. 


        # computation of a PSF on the detector using the '*' operator
        self.atm*self.ngs*self.tel*self.cam*self.cam_binned

        # The Telescope and Atmosphere can be separated using the '-' operator (Free space propagation) 
        self.tel-self.atm

        #%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
        # mis-registrations object
        misReg = MisRegistration()
        misReg.shiftX = param['MisReg_shiftX']                  # in [m]
        misReg.shiftY = param['MisReg_shiftY']                  # in [m]
        misReg.rotationAngle = param['MisReg_rotationAngle']    # in [deg]

        # Get valid Actuators
        self.nActuator = param['nActuator']

        # if no coordonates specified, create a cartesian dm
        self.dm=DeformableMirror(telescope    = self.tel,                     # Telescope
                                 nSubap       = param['nSubaperture'],        # number of subaperture of the system considered (by default the DM has n_subaperture + 1 actuators to be in a Fried Geometry)                   
                                 mechCoupling = param['mechanicalCoupling'],  # Mechanical Coupling for the influence functions
                                 misReg       = misReg,                       # Mis-registration associated
                                 coordinates  = None,                         # coordinates in [m]. Should be input as an array of size [n_actuators, 2] 
                                 pitch        = self.tel.D/self.nActuator)    # inter actuator distance. Only used to compute the influence function coupling. The default is based on the n_subaperture value. 
        
        # Get Valid Actuators mask (X and Y axis) this si important for the RL      
        self.dm_mask = np.reshape(self.dm.validAct,(self.nActuator,self.nActuator))
        (self.xvalid, self.yvalid) = np.nonzero(self.dm_mask)
        #%% -----------------------     PYRAMID WFS   ----------------------------------

        # make sure tel and atm are separated to initialize the PWFS
        self.tel.isPaired = False
        self.tel.resetOPD()

        self.wfs = Pyramid(nSubap       = param['nSubaperture'],        # number of subaperture = number of pixel accros the pupil diameter       
                      telescope         = self.tel,                     # telescope object
                      lightRatio        = param['lightThreshold'],      # flux threshold to select valid sub-subaperture
                      modulation        = param['modulation'],          # Tip tilt modulation radius
                      binning           = 1,                            # binning factor (applied only on the )
                      n_pix_separation  = param['n_pix_separation'],    # number of pixel separating the different pupils
                      n_pix_edge        = 2,                            # number of pixel on the edges of the pupils
                      postProcessing    = param['postProcessing'])  # slopesMaps,

        # Propagate the light to the Wave-Front Sensor
        self.tel*self.wfs
        #%% -----------------------     Modal Basis - Zernike  ----------------------------------
        #%% ZERNIKE Polynomials
        # create Zernike Object
        Z = Zernike(self.tel,50)
        # compute polynomials for given telescope
        Z.computeZernike(self.tel)

        # mode to command matrix to project Zernike Polynomials on DM
        # M2C_zernike = np.linalg.pinv(np.squeeze(self.dm.modes[self.tel.pupilLogical,:]))@Z.modes

        M2C_zernike = np.load(os.path.dirname(__file__)+'/manual_m2c.npy')[:, :50]

        self.M2OPD = np.load(os.path.dirname(__file__)+'/../wf_recon/M2OPD_500modes.npy')[:, :self.args.nModes]
        #%% -----------------------     Calibration: Interaction Matrix  ----------------------------------

        # amplitude of the modes in m
        stroke=self.ngs.wavelength/16
        # zonal Interaction Matrix
        M2C_zonal = np.eye(self.dm.nValidAct)
        # modal Interaction Matrix for 300 modes
        M2C_modal = M2C_zernike[:,:300]

        self.tel-self.atm
        # zonal interaction matrix
        calib_modal = InteractionMatrix(ngs            = self.ngs,
                                        atm            = self.atm,
                                        tel            = self.tel,
                                        dm             = self.dm,
                                        wfs            = self.wfs,   
                                        M2C            = M2C_zonal, # M2C matrix used 
                                        stroke         = stroke,    # stroke for the push/pull in M2C units
                                        nMeasurements  = 6,        # number of simultaneous measurements
                                        noise          = 'off',     # disable wfs.cam noise 
                                        display        = True,      # display the time using tqdm
                                        single_pass    = True)      # only push to compute the interaction matrix instead of push-pull


    


        # Modal interaction matrix
        calib_zernike = CalibrationVault(calib_modal.D@M2C_zernike)

        calib_tt = CalibrationVault(calib_modal.D@M2C_zernike[:,:self.args.nModes])
        #%%
        #%%  ----------------------- Define instrument and WFS path detectors  -----------------------

        # instrument path
        src_cam = Detector(self.tel.resolution*4)
        src_cam.psf_sampling = 4
        src_cam.integrationTime = self.tel.samplingTime*1
        # put the scientific target off-axis to simulate anisoplanetism (set to  [0,0] to remove anisoplanetism)
        self.src.coordinates = [0.4,0]

        # WFS path
        ngs_cam = Detector(self.tel.resolution)
        ngs_cam.psf_sampling = 4
        ngs_cam.integrationTime = self.tel.samplingTime

        # initialize DM commands
        self.tel.resetOPD()
        self.dm.coefs=0
        self.dm_prev = self.dm.coefs.copy()
        self.ngs*self.tel*self.dm*self.wfs
        self.wfs*self.wfs.focal_plane_camera

        # Update the r0 parameter, generate a new phase screen for the atmosphere and combine it with the Telescope
        # atm.r0 = 0.15
        self.atm.generateNewPhaseScreen(seed = 10)

        self.tel+self.atm


        self.tel.computePSF(4)
        
        # These are the calibration data used to close the loop
        calib_CL    = calib_zernike
        M2C_CL      = M2C_zernike
        self.M2C_CL = M2C_CL

        self.M2C_tt = torch.from_numpy(M2C_zernike[:, :self.args.nModes]).to(device=self.device, dtype=torch.float32)

        self.reconstructor_tt = calib_tt.M


        # combine telescope with atmosphere
        self.tel+self.atm
        # initialize DM commands
        self.atm*self.ngs*self.tel*ngs_cam
        self.atm*self.src*self.tel*src_cam
        
        # allocate memory to save data
        self.SR                      = []   #np.zeros(param['nLoop'])
        self.total                   = np.zeros(param['nLoop'])
        self.residual                = np.zeros(param['nLoop'])
        self.wfsSignal               = np.arange(0,self.wfs.nSignal)*0
        self.SE_PSF = []
        self.LE_PSF = np.log10(self.tel.PSF)
        self.LE_PSFs = []

        self.reconstructor = M2C_CL@calib_CL.M
        self.modal_CM = calib_CL.M
        self.C2M_tt = np.linalg.pinv(M2C_CL[:, :self.args.nModes])
        self.F = M2C_CL @ np.linalg.pinv(M2C_CL)   
        
        




    def set_wfs(self,param,type = "pyramid"):
        if type == "pyramid":

            from OOPAO.Pyramid          import Pyramid
            # make sure tel and atm are separated to initialize the PWFS
            self.tel-self.atm
            
            self.wfs = Pyramid(nSubap              = param['nSubaperture'],\
                        telescope             = self.tel,\
                        modulation            = param['modulation'],\
                        lightRatio            = param['lightThreshold'],\
                        n_pix_separation      = param['n_pix_separation'],\
                        psfCentering          = param['psfCentering'],\
                        postProcessing        = param['postProcessing'])

            self.tel*self.wfs

        

    def set_modalBasis(self, mode = "zernike"):
        if mode == "zernike":
            from OOPAO.Zernike import Zernike
            from OOPAO.calibration.CalibrationVault import CalibrationVault
            from OOPAO.calibration.InteractionMatrix import InteractionMatrix

            # create Zernike Object
            Z = Zernike(self.tel,50)
            # compute polynomials for given telescope
            Z.computeZernike(self.tel)

            # mode to command matrix to project Zernike Polynomials on DM
            M2C_zernike  = np.linalg.pinv(np.squeeze(self.dm.modes[self.tel.pupilLogical,:]))@Z.modes

            #self.dm.coefs = M2C_zernike[:,:10]
            #self.tel*self.dm

            M2C_zonal = np.eye(self.dm.nValidAct)
            # zonal interaction matrix
            self.imat = InteractionMatrix(  ngs      = self.source,\
                                            atm            = self.atm,\
                                            tel            = self.tel,\
                                            dm             = self.dm,\
                                            wfs            = self.wfs,\
                                            M2C            = M2C_zonal,\
                                            stroke         = 1e-9,\
                                            nMeasurements  = 25,\
                                            noise          = 'off')
            # Modal interaction matrix
            calib_zernike = CalibrationVault(self.imat.D@M2C_zernike)

            self.M2C_CL = M2C_zernike
            self.calib_CL = calib_zernike

    def set_display(self):
        from OOPAO.tools.displayTools import cl_plot
        self.SE_PSF = []
        self.LE_PSF = np.log10(self.tel.PSF)
        self.plot_obj = cl_plot(list_fig  = [self.atm.OPD,self.tel.mean_removed_OPD,self.wfs.cam.frame,np.log10(self.wfs.get_modulation_frame(radius = 1)),[[0,0],[0,0]],[self.dm.coordinates[:,0],np.flip(self.dm.coordinates[:,1]),self.dm.coefs],np.log10(self.tel.PSF),np.log10(self.tel.PSF)],\
                   type_fig          = ['imshow','imshow','imshow','imshow','plot','scatter','imshow','imshow'],\
                   list_title        = ['Turbulence OPD','Residual OPD','WFS Detector','WFS Modulation Camera',None,None,None,None],\
                   list_lim          = [None,None,None,[-3,0],None,None,[-4,0],[-4,0]],\
                   list_label        = [None,None,None,None,['Time','WFE [nm]'],['DM Commands',''],['Short Exposure PSF',''],['Long Exposure_PSF','']],\
                   n_subplot         = [4,2],\
                   list_display_axis = [None,None,None,None,True,None,None,None],\
                   list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)
    def render(self, current_i,mode='rgb_array'):
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
        return -self.gainCL*np.matmul(self.reconstructor,self.wfsSignal)  
        
    def get_slopes(self):
        return self.wfs.signal
    
    def get_strehl(self):
        return  np.exp(-np.var(self.tel.src.phase[np.where(self.tel.pupil==1)]))

    def _get_reward(self,slopes,type = "volt"):
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
            action_out = action[:,:,self.xvalid, self.yvalid]
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

