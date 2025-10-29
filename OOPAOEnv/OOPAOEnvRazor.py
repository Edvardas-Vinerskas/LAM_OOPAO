"""
OOPAO module for FitAO
@author: Raissa Camelo (LAM) git: @srtacamelo
"""

from OOPAOEnv.__load__oopao import load_oopao
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch

class OOPAO(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    #--------------------------Core gym funtions--------------------------------
    def __init__(self):

        # OOPAO Modules
        self.gainCL = None
        self.atm = None
        self.wfs = None
        self.dm = None
        self.misReg = None
        self.tel = None
        self.source = None
        self.imat = None          # zonal imat
        self.M2C_CL = None
        self.calib_CL = None
        self.mode = None
        self.Z = None
        self.plot_obj = None
        self.display = None
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

        self.leak = 0.99
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def reset(self):
        self.set_params()
        self.action_buffer = []
        obs = -np.matmul(self.reconstructor,self.get_slopes())
        obs = self.vec_to_img(obs)*1e6
        return obs
    
    def reset_soft(self):
        self.action_buffer = []

        obs = -np.matmul(self.reconstructor,self.get_slopes())
        obs = self.vec_to_img(obs)*1e6
        return obs

    def reset_soft_wfs(self):
        self.action_buffer = []
        obs = self.wfs.cam.frame.copy()
        return obs
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
        from OOPAO.ShackHartmann import ShackHartmann
        from OOPAO.Source import Source
        from OOPAO.Telescope import Telescope
        from OOPAO.Zernike import Zernike
        from OOPAO.calibration.CalibrationVault import CalibrationVault
        from OOPAO.calibration.InteractionMatrix import InteractionMatrix
        from OOPAO.tools.displayTools import cl_plot, displayMap


        #%% -----------------------     TELESCOPE   ----------------------------------

        if wfs_type == "pyramid":

            self.tel = Telescope(resolution      = 64*6,\
                             diameter            = param['diameter'],\
                             samplingTime        = param['samplingTime'],\
                             centralObstruction  = param['centralObstruction'])

        else:
        # create the Telescope object
            self.tel = Telescope(resolution          = param['resolution'],\
                                diameter            = param['diameter'],\
                                samplingTime        = param['samplingTime'],\
                                centralObstruction  = param['centralObstruction'])

        #%% -----------------------     NGS   ----------------------------------
        # create the Source object
        self.source=Source(optBand   = param['opticalBand'],\
                           magnitude = param['magnitude'])
        

        ###

        # combine the NGS to the telescope using '*' operator:
        self.source*self.tel

        self.tel.computePSF(zeroPaddingFactor = 6)

        #%% -----------------------     ATMOSPHERE   ----------------------------------

        # create the Atmosphere object
        self.atm=Atmosphere(telescope     = self.tel,\
                    r0            = param['r0'],\
                    L0            = param['L0'],\
                    windSpeed     = param['windSpeed'],\
                    fractionalR0  = param['fractionalR0'],\
                    windDirection = param['windDirection'],\
                    altitude      = param['altitude'])
        # initialize atmosphere
        self.atm.initializeAtmosphere(self.tel)

        self.atm.update()



        self.tel+self.atm
        self.tel.computePSF(8)
        #%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
        # mis-registrations object
        misReg = MisRegistration(param)
        # if no coordonates specified, create a cartesian dm



        nAct = param['nActuator']
        mask = param['actMask']

        x = np.linspace(-(self.tel.D)/2,(self.tel.D)/2,nAct)
        X,Y= np.meshgrid(x,x)*mask            
                    
        # compute the initial set of coordinates
        xc = np.reshape(X,[nAct**2])
        yc = np.reshape(Y,[nAct**2])

        coords = np.array(list(zip(xc,yc)))

        coords0=[]

        for i in coords:
            if np.abs(np.array(i)[1]) < 1e3 and np.abs(np.array(i)[0]) < 1e3:
                coords0.append(i)

        self.dm = DeformableMirror(telescope = self.tel, 
                               nSubap = param['nActuator'], 
                               mechCoupling = param['mechanicalCoupling'],
                               M4_param = param, 
                               coordinates = np.array(coords0))


        # self.dm=DeformableMirror(telescope    = self.tel,\
        #                     nSubap       = param['nSubaperture'],\
        #                     mechCoupling = param['mechanicalCoupling'],\
        #                     misReg       = misReg)
        # Get valid Actuators
        self.nActuator = param['nActuator']
        #print(self.dm.nValidAct)
        #print(self.dm.validAct)
        self.dm_mask = param['boolActMask'].astype(int) # np.reshape(self.dm.validAct,(self.nActuator,self.nActuator))
        (self.xvalid, self.yvalid) = np.nonzero(self.dm_mask)
        #%% -----------------------     PYRAMID WFS   ----------------------------------

        # make sure tel and atm are separated to initialize the PWFS
        self.tel-self.atm

        if wfs_type == "pyramid":

            self.wfs = Pyramid(nSubap = 64, 
                            telescope = self.tel, 
                            modulation = param['modulation'], 
                            lightRatio = 0.95, 
                            psfCentering = param['psfCentering'], 
                            postProcessing = param['postProcessing'])

            self.wfs.cam.sensor = 'CMOS'
            self.wfs.cam.FWC = 10000
            self.wfs.cam.bits = 10

            self.wfs.cam.QE = 0.56
            self.wfs.cam.darkCurrent = 5

            self.wfs.cam.integrationTime = param['samplingTime']


        elif wfs_type == "shackhartmann":

            self.wfs = ShackHartmann(telescope = self.tel,
                    nSubap = param['nSubaperture'],
                    lightRatio = 0.5,
                    is_geometric = False,
                    shannon_sampling = True)#,
                    #threshold_cog = 0.1)

            self.wfs.valid_subapertures = param['validMask']

            self.wfs.cam.sensor = 'CMOS'
            self.wfs.cam.FWC = 10000
            self.wfs.cam.bits = 10

            self.wfs.cam.QE = 0.56
            self.wfs.cam.darkCurrent = 5

            self.wfs.cam.integrationTime = param['samplingTime']

        self.tel*self.wfs

        #%% ZERNIKE Polynomials
        # create Zernike Object
        Z = Zernike(self.tel,50)
        # compute polynomials for given telescope
        Z.computeZernike(self.tel)

        # mode to command matrix to project Zernike Polynomials on DM
        M2C_zernike = np.linalg.pinv(np.squeeze(self.dm.modes[self.tel.pupilLogical,:]))@Z.modes

        # show the first 10 zernikes
        self.dm.coefs = M2C_zernike[:,:10]
        self.tel*self.dm
        displayMap(self.tel.OPD)

        #%% to manually measure the interaction matrix

        # amplitude of the modes in m
        stroke=1e-9
        # Modal Interaction Matrix 

        #%%
        M2C_zonal = np.eye(self.dm.nValidAct)

        # zonal interaction matrix
        calib_zonal = InteractionMatrix(  ngs            = self.source,\
                                          atm            = self.atm,\
                                          tel            = self.tel,\
                                          dm             = self.dm,\
                                          wfs            = self.wfs,\
                                          M2C            = M2C_zonal,\
                                          stroke         = stroke,\
                                          nMeasurements  = 25,\
                                          noise          = 'off')


        # Modal interaction matrix
        calib_zernike = CalibrationVault(calib_zonal.D@M2C_zernike)
        #%%
        self.tel.resetOPD()
        # initialize DM commands
        self.dm.coefs=0
        self.dm_prev = self.dm.coefs.copy()
        self.source*self.tel*self.dm*self.wfs
        self.tel+self.atm

        # dm.coefs[100] = -1

        self.tel.computePSF(4)
            
        # These are the calibration data used to close the loop
        calib_CL    = calib_zernike
        self.M2C_CL      = M2C_zernike


        # combine telescope with atmosphere
        self.tel+self.atm

        # allocate memory to save data
        self.SR                      = []   #np.zeros(param['nLoop'])
        self.total                   = np.zeros(param['nLoop'])
        self.residual                = np.zeros(param['nLoop'])
        self.wfsSignal               = np.arange(0,self.wfs.nSignal)*0
        self.SE_PSF = []
        self.LE_PSF = np.log10(self.tel.PSF)
        self.LE_PSFs = []

        from OOPAO.tools.displayTools import cl_plot

        # self.plot_obj = cl_plot(list_fig  = [self.atm.OPD,self.tel.mean_removed_OPD,self.wfs.cam.frame,np.log10(self.wfs.get_modulation_frame(radius = 10)),[[0,0],[0,0]],[self.dm.coordinates[:,0],np.flip(self.dm.coordinates[:,1]),self.dm.coefs],np.log10(self.tel.PSF_norma_zoom),np.log10(self.tel.PSF_norma_zoom)],\
        #                 type_fig          = ['imshow','imshow','imshow','imshow','plot','scatter','imshow','imshow'],\
        #                 list_title        = ['Turbulence OPD','Residual OPD','WFS Detector','WFS Modulation Camera',None,None,None,None],\
        #                 list_lim          = [None,None,None,[-3,0],None,None,[-4,0],[-4,0]],\
        #                 list_label        = [None,None,None,None,['Time','WFE [nm]'],['DM Commands',''],['Short Exposure PSF',''],['Long Exposure_PSF','']],\
        #                 n_subplot         = [4,2],\
        #                 list_display_axis = [None,None,None,None,True,None,None,None],\
        #                 list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)
        # loop parameters
        gainCL                  = 0.2
        self.wfs.cam.photonNoise     = True
        self.wfs.cam.readoutNoise    = 14
        self.display                 = True

        self.reconstructor = self.M2C_CL@calib_CL.M
        self.F = self.M2C_CL @ np.linalg.pinv(self.M2C_CL)

        self.dm_proj = self.compute_dm_proj()


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

        elif type == "shackhartmann":

            from OOPAO.ShackHartmann import ShackHartmann

            self.tel-self.atm

            self.wfs = ShackHartmann(telescope = self.tel,
                                    nSubap = param['nSubaperture'],
                                    lightRatio = 0.5,
                                    is_geometric = False,
                                    shannon_sampling = True,
                                    threshold_cog = 0.1)

            self.wfs.valid_subapertures = param['validMask']

            self.wfs.cam.sensor = 'CMOS'
            self.wfs.cam.FWC = 10000
            self.wfs.cam.bits = 10

            self.wfs.cam.QE = 0.56
            self.wfs.cam.darkCurrent = 5

            self.wfs.cam.integrationTime = param['samplingTime']

            self.wfs.cam.photonNoise     = True
            self.wfs.cam.readoutNoise    = 14

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
        self.LE_PSF = np.log10(self.tel.PSF_norma_zoom)
        self.plot_obj = cl_plot(list_fig  = [self.atm.OPD,self.tel.mean_removed_OPD,self.wfs.cam.frame,np.log10(self.wfs.get_modulation_frame(radius = 1)),[[0,0],[0,0]],[self.dm.coordinates[:,0],np.flip(self.dm.coordinates[:,1]),self.dm.coefs],np.log10(self.tel.PSF_norma_zoom),np.log10(self.tel.PSF_norma_zoom)],\
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
        from OOPAO.tools.displayTools import cl_plot
        # update displays if required
        if self.display==True:        
            self.tel.computePSF(4)
            if current_i>15:
                self.SE_PSF.append(np.log10(self.tel.PSF_norma_zoom))
                self.LE_PSF = np.mean(self.SE_PSF, axis=0)
            
            plot = cl_plot(list_fig   = [self.atm.OPD,self.tel.mean_removed_OPD,self.wfs.cam.frame,
                                np.log10(self.wfs.get_modulation_frame(radius=10)),
                                [np.arange(current_i+1),self.residual[:current_i+1]],
                                self.dm.coefs,np.log10(self.tel.PSF_norma_zoom), self.LE_PSF],
                                plt_obj = self.plot_obj)
                                
            plt.pause(0.1)
        return self.LE_PSF, np.log10(self.tel.PSF_norma_zoom)
    

    
    def render4plot(self,current_i):
        self.tel.computePSF(4)
        if current_i>15:
            self.SE_PSF.append(np.log10(self.tel.PSF_norma_zoom.copy()))
            self.LE_PSF = np.mean(self.SE_PSF, axis=0)
        #plt.imshow(self.LE_PSF)
        #plt.show()
        #plt.pause(3)
        #plt.close()
        return self.LE_PSF,np.log10(self.tel.PSF_norma_zoom)


    def step(self,i, action ): # action, showAtmos = True
        # Single AO step. Action defines DM shape and function returns WFS
        # slopes, reward (-1*norm of slopes), done as false (no current condition)
        # and (currently empty) info dictionary, where one could store useful
        # data about the simulation
        action = self.img_to_vec(action)*1e-6

        # update phase screens => overwrite tel.OPD and consequently tel.src.phase
        self.atm.update()
        # save phase variance
        self.total[i]=np.std(self.tel.OPD[np.where(self.tel.pupil>0)])*1e9
        # save turbulent phase
        turbPhase = self.tel.src.phase
        # propagate to the WFS with the CL commands applied
        self.tel*self.dm*self.wfs

        # Integrator
        # self.dm.coefs=self.dm.coefs-self.gainCL*np.matmul(self.reconstructor,self.wfsSignal)  
        self.dm.coefs = (self.dm_prev * self.leak) + action 
        self.dm_prev = self.dm.coefs.copy()

        # store the slopes after computing the commands => 2 frames delay
        self.wfsSignal=self.wfs.signal

        slopes = self.wfsSignal 
        obs = -np.matmul(self.reconstructor,slopes)
        obs = self.vec_to_img(obs)*1e6

        self.residual[i]=np.std(self.tel.OPD[np.where(self.tel.pupil>0)])*1e9
        self.OPD=self.tel.OPD[np.where(self.tel.pupil>0)]

        # Save Strehl ratio to calculate Episode Average
        strehl = self.get_strehl()
        self.SR.append(strehl)

        # Extra
        info = {"strehl":strehl}
        done = 0


        return obs, -1 * np.linalg.norm(obs), strehl,bool(done), info


    def step_wfs(self, i, action):
        action = self.img_to_vec(action)*1e-6

        # update phase screens => overwrite tel.OPD and consequently tel.src.phase
        self.atm.update()
        # save phase variance
        self.total[i]=np.std(self.tel.OPD[np.where(self.tel.pupil>0)])*1e9
        # save turbulent phase
        turbPhase = self.tel.src.phase
        # propagate to the WFS with the CL commands applied
        self.tel*self.dm*self.wfs

        # Integrator
        # self.dm.coefs=self.dm.coefs-self.gainCL*np.matmul(self.reconstructor,self.wfsSignal)  
        self.dm.coefs += action 

        # store the slopes after computing the commands => 2 frames delay
        self.wfsSignal=self.wfs.cam.frame.copy()

        obs = self.wfsSignal 

        self.residual[i]=np.std(self.tel.OPD[np.where(self.tel.pupil>0)])*1e9
        self.OPD=self.tel.OPD[np.where(self.tel.pupil>0)]

        # Save Strehl ratio to calculate Episode Average
        strehl = self.get_strehl()
        self.SR.append(strehl)

        # Extra
        info = {"strehl":strehl}
        done = 0


        return obs, -1 * np.linalg.norm(obs), strehl,bool(done), info

    
    def step_wfs(self, i, action):
        action = self.img_to_vec(action)

        # update phase screens => overwrite tel.OPD and consequently tel.src.phase
        self.atm.update()
        # save phase variance
        self.total[i]=np.std(self.tel.OPD[np.where(self.tel.pupil>0)])*1e9
        # save turbulent phase
        turbPhase = self.tel.src.phase
        # propagate to the WFS with the CL commands applied
        self.tel*self.dm*self.wfs

        # Integrator
        # self.dm.coefs=self.dm.coefs-self.gainCL*np.matmul(self.reconstructor,self.wfsSignal)  
        self.dm.coefs += action.detach().numpy()

        # store the slopes after computing the commands => 2 frames delay
        self.wfsSignal=self.wfs.cam.frame.copy()

        obs = self.wfsSignal 

        self.residual[i]=np.std(self.tel.OPD[np.where(self.tel.pupil>0)])*1e9
        self.OPD=self.tel.OPD[np.where(self.tel.pupil>0)]

        # Save Strehl ratio to calculate Episode Average
        strehl = self.get_strehl()
        self.SR.append(strehl)

        # Extra
        info = {"strehl":strehl}
        done = 0


        return obs, -1 * np.linalg.norm(obs), strehl,bool(done), info


    def calculate_strehl_AVG(self):
        """Calculates the average strehl ratio for each episode.
        Cleans the strehl array after each episode.
        """
        avg = np.mean(self.SR)
        self.SR = []

        return avg
    
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
    
    def sample_noise(self, sigma):

        noise = self.F @ (sigma * np.random.normal(0,1 , size = (int(self.dm.nValidAct),))) # 357
        return self.vec_to_img(noise)
    
    def vec_to_img(self, action_vec, use_torch=False):
        if use_torch:
            valid_actus = torch.zeros((self.nActuator, self.nActuator)).to(self.device)

        else:
            valid_actus = np.zeros((self.nActuator, self.nActuator))

        valid_actus[self.xvalid, self.yvalid] = action_vec

        return valid_actus



    def img_to_vec(self, action):
        # assert len(action.shape) == 2
        if len(action.shape) == 4:
            action_out = action[:,:,self.xvalid, self.yvalid]
        else:
            action_out = action[self.xvalid, self.yvalid]
        
        return action_out
    

    def change_mag(self, mag):

        self.source.nPhoton = self.source.zeroPoint*10**(-0.4*mag)
        self.source * self.tel*self.wfs
    
    def change_src_fluxmap(self, lam):
            pupil = self.tel.pupil.copy()
            intensity = np.random.exponential(lam, pupil.shape)
            intensity /= np.max(intensity)

            self.source.fluxMap[pupil] = intensity[pupil]

            self.source*self.tel*self.wfs


    def plot_wfs(self, threshold=None):
        self.atm.update()

        self.tel*self.wfs

        frame = self.wfs.cam.frame.copy()

        bins = np.arange(np.min(frame), np.max(frame)+ 3)

        plt.hist(frame.flatten(), bins=bins)

        if threshold is not None:
            plt.axvline(threshold * np.max(frame), color='r')

        plt.yscale('log')
        plt.show()

    def compute_dm_proj(self):
        modes = self.dm.modes.copy()
        inv_proj = np.linalg.inv(np.matmul(modes.T, modes))
        
        return np.matmul(inv_proj, modes.T)


    def OPD_on_dm(self):
        res = self.dm.resolution
        phase = self.tel.OPD.copy().reshape(res*res)
        proj1 = np.matmul(self.dm_proj, phase)
        dm_OPD =  np.reshape(np.matmul(self.dm.modes, proj1), (res,res))

        return dm_OPD
        
