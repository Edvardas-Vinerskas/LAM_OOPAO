"""
Created 2026-05-24

Numerical twin for PAPYRIIS + CNN + RL

Heavily inspired by PAPYRIIS from Mathieu Motte
"""

import gymnasium as gym #(this environment is not and never was gym compliant)
import numpy as np
import torch
from functions import *
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, r"C:\Users\evinerskas\PycharmProjects\LAM_OOPAO\ozitelemetry")

import OOPAO
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Pyramid import Pyramid
from OOPAO.ZWFS2_mine import ZWFS2
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.calibration.CalibrationVault import CalibrationVault
from PAPYRIIS_2stage_CNN_RL.PAPYRUS.Papyrus import Papyrus
from PAPYRIIS_2stage_CNN_RL.OZIRIIS import OZIRIIS
from PAPYRIIS_2stage_CNN_RL.parameterFile_papyriis import initializeParameterFile
from Papyrus2ndStage import Papyrus2ndStage #the cnn
from pymatreader import read_mat
from OOPAO.MisRegistration import MisRegistration
from skimage.transform import resize
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
import tqdm


@dataclass
class FirstStageLoopConfig:
    """Configuration used by the first-stage closed loop."""

    nLoop: Optional[int] = None
    gainCL: float = 0.7
    leak: float = 0.995
    frame_delay: int = 2
    photon_noise: bool = True
    progress: bool = True



class OOPAO_environment_PAPYRIIS(gym.Env):
    def __init__(self):
        self.device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #--------------------------------------------------- Params ---------------------------------------------------#
        # src =   Source('IR1310', 0)
        # src*tel
        #TODO a big difference in performance is due to using your own interaction matrix vs imported one
        #       papyrus uses the imported so I will too (EMCCD does not work with synthetic? )

        self.modes_1st  = 195
        self.modes_2nd  = 50
        self.seed = 10

        self.FIRST_GAIN          = 0.7
        self.FIRST_LEAK          = 0.995
        #self.FIRST_FRAME_DELAY   = 2
        self.FIRST_SKY_OFFSET    = (2, 2)

        self.SECOND_GAIN         = "not applicable"
        self.SECOND_LEAK         = 0.95 #TODO fix this so it satisfies the po4ao_config (oopsie innit)
        #self.SECOND_FRAME_DELAY  = 3 #at least I think this is the frame delay encountered


        #---------------------------------------------------1st stage---------------------------------------------------#

        from PAPYRIIS_2stage_CNN_RL.PAPYRUS.parameter_files.OCAM2K  import OCAM_param
        self.OCAM_param = OCAM_param
        self.Papytwin = Papyrus()
        self.tel_1st    = self.Papytwin.tel
        self.ngs_1st    = self.Papytwin.ngs
        self.dm_1st     = self.Papytwin.dm
        self.pwfs       = self.Papytwin.wfs
        self.atm_1st    = self.Papytwin.atm     
        self.slow_tt    = self.Papytwin.slow_tt
        self.pwfs.cam   = self.Papytwin.perfect_OCAM

        #---------------------------------------------------2nd stage---------------------------------------------------#
        self.param  = initializeParameterFile()
        self.OZItwin = OZIRIIS(is_onsky = True,
                               param = self.param, #parameterFile_papyriis
                               controlled_modes = 50,
                               )

        self.tel_2nd    = self.OZItwin.tel
        self.src_2nd = Source(optBand='H', magnitude=-2.81)
        self.src_2nd.wavelength = 1.6e-06
        param_misreg = np.load("PAPYRIIS_2stage_CNN_RL/dm_second_stage_misreg_dict.npy", allow_pickle=True).item()
        m = MisRegistration(param_misreg)
        self.dm_2nd = DeformableMirror(telescope=self.tel_2nd, 
                                   nSubap=10, 
                                   mechCoupling=0.35, 
                                   print_dm_properties=False, 
                                   pitch=0.11, 
                                   misReg=m,
                                   sign=-1e-5)

        self.atm_2nd    = self.OZItwin.atm #every parameter for atm is set for 500nm
        self.vzwfs  = ZWFS2(tel = self.tel_2nd, 
                            diameter = self.param['d_z_sky'], 
                            phase_shift = self.param['depth'], 
                            zpf = self.param['mask_px_size'])



        #--------------------------------------------------- Calibration ---------------------------------------------------#
        #1st stage
        self.Papytwin.set_pupil(calibration=True)
        self.ngs_1st * self.tel_1st * self.dm_1st * self.pwfs
        self.M2C_1st = - np.load("PAPYRIIS_2stage_CNN_RL/M2C_1rst.npy") #obtained from on-sky data
        self.M2C_1st_CL = self.M2C_1st[:, :self.modes_1st]
        self.im_1st = read_mat('PAPYRIIS_2stage_CNN_RL/intMat_klOOPAO_synthetic_bin=1_F=500_rMod=5_20250604_0307.mat')['matrix_inf']
        self.pwfs.modulation = 5
        self.pwfs.cam.sensor = "CCD"
        self.calib_1st = InteractionMatrix(  ngs = self.ngs_1st,\
                            atm            = self.atm_1st,\
                            tel            = self.tel_1st,\
                            dm             = self.dm_1st,\
                            wfs            = self.pwfs,\
                            M2C            = self.M2C_1st_CL,\
                            stroke         = 0.0001,\
                            phaseOffset    = 0,\
                            nMeasurements  = 1,\
                            noise          = 'off',
                            print_time=False,
                            display=True)
        self.pwfs.cam.sensor = "EMCCD"
        # self.calib_1st = CalibrationVault(self.im_1st[:, :self.modes_1st])
        self.calib_1st_M = self.calib_1st.M
        self.reconstructor_1st = self.M2C_1st_CL @ self.calib_1st_M

        KL_basis_dm_1st = (self.dm_1st.modes @ self.M2C_1st) * np.tile(self.tel_1st.pupil.flatten()[:, None], self.M2C_1st.shape[1])
        self.projector_kl_1st = np.linalg.pinv(KL_basis_dm_1st) 
        #--------------------------------------------------- CNN ---------------------------------------------------#
        from Frame_Preprocess import Frame_Preprocess

        self.cnn = Papyrus2ndStage().to(device = self.device)
        checkpoint_path = 'PAPYRIIS_2stage_CNN_RL/OziNewDMOnSky.pth'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
        self.cnn.load_state_dict(checkpoint['PhaseEstimator_state_dict'])

        #pupil_centers = np.array([[ 81, 128],
        #                        [175, 128]])
        
        pupil_centers = np.array([[ 48, 48],
                                [48, 144]])
        self.frame_preprocessor = Frame_Preprocess(pupil_centers)

        #reference_intensity = np.load('PAPYRIIS_2stage_CNN_RL/reference_intensity.npy')
        #this should act as a reference innit
        self.dm_2nd.coefs = 0
        self.src_2nd * self.tel_2nd * self.dm_2nd * self.vzwfs
        self.frame_preprocessor.ProcessReference(self.vzwfs.signal_2D_cam_padded/np.sum(self.vzwfs.signal_2D_cam_padded))


        self.M2C_2nd = np.load('PAPYRIIS_2stage_CNN_RL/M2C_KL.npy') #obtained from Francisco
        self.M2C_ = self.M2C_2nd
        KL_basis_dm_2nd = (self.dm_2nd.modes @ self.M2C_2nd) * np.tile(self.tel_2nd.pupil.flatten()[:, None], self.M2C_2nd.shape[1])
        self.projector_kl_2nd = np.linalg.pinv(KL_basis_dm_2nd) 

        #---------------------------------------------------INITIALIZATION---------------------------------------------------#

        #science camera would add noise (better way to calculate the strehl?)
        self.atm_2nd.initializeAtmosphere(telescope = self.tel_2nd)
        self.atm_2nd.generateNewPhaseScreen(seed = self.seed)
        self.atm_update_tracker = 0
        self.src_2nd.reset()
        self.dm_2nd.coefs = 0
        self.dm_2nd_copy  = 0
        self.src_2nd ** self.atm_2nd * self.tel_2nd * self.dm_2nd * self.vzwfs
        self.tel_2nd.print_optical_path()
        self.CURRENT_STEPS = 0 #current steps seems to just be for tracking loop progress


        # CRUCIAL PIXEL SIZE CHECK#
        '''self.pixel_size = self.DIAMETER / self.RESOLUTION
        if (3 * self.pixel_size) > self.r_0:
            raise SystemExit("ERROR: pixel size is too big for r_0 value")'''

        mask_1D = self.dm_2nd.validAct.copy()
        mask_reshaped = mask_1D.reshape((11, 11))
        self.mask_cpu = mask_reshaped.astype(bool)
        self.mask = torch.from_numpy(mask_reshaped).bool().to(device = self.device)
        


    def flatten_dm(self):
        self.dm_1st.coefs = 0
        self.dm_2nd.coefs = 0
        self.dm_2nd_copy  = 0

        self.src_2nd ** self.atm_2nd * self.tel_2nd * self.dm_2nd * self.vzwfs

        cnn_input = self.frame_preprocessor.ProcessFrame(self.vzwfs.signal_2D_cam_padded/np.sum(self.vzwfs.signal_2D_cam_padded))
        cnn_input = torch.from_numpy(cnn_input).unsqueeze(0).float().to(device = self.device)
        cnn_output = - self.cnn(cnn_input).detach().cpu().numpy().squeeze()
        vzwfs_signal_proj = self.M2C_2nd[:, :self.modes_2nd] @ cnn_output
        vzwfs_signal_proj = torch.from_numpy(vzwfs_signal_proj).float().to(device = self.device)

        # redoing in 2D
        vzwfs_signal_proj_2D = torch.zeros((11, 11), dtype=torch.float32).to(device = self.device)
        vzwfs_signal_proj_2D[self.mask] = vzwfs_signal_proj
        next_state = vzwfs_signal_proj_2D


        return next_state

    def reset(self, atm_OPD_1st_residual, seed = None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        if seed is None:
            seed = np.random.randint(1e9)

        print(f'RESET IS BEING CALLED WITH SEED {seed}')
        self.atm_2nd.generateNewPhaseScreen(seed=seed)
        self.atm_update_tracker = 0
        self.atm_2nd.update(atm_OPD_1st_residual[0])
        self.CURRENT_STEPS      = 0
        self.vzwfssignal_buffer = [torch.zeros((11, 11)), torch.zeros((11, 11))] #3 frame delay zwfs

        self.dm_1st.coefs = 0
        self.dm_2nd.coefs = 0
        self.dm_2nd_copy = 0
        self.dm_prev_coefs_2nd = self.dm_2nd.coefs.copy()

        self.src_2nd ** self.atm_2nd * self.tel_2nd * self.dm_2nd * self.vzwfs
        self.atm_opd = self.atm_2nd.OPD.copy()

        
        cnn_input = self.frame_preprocessor.ProcessFrame(self.vzwfs.signal_2D_cam_padded/np.sum(self.vzwfs.signal_2D_cam_padded))
        cnn_input = torch.from_numpy(cnn_input).unsqueeze(0).float().to(device = self.device)
        self.cnn_output = - self.cnn(cnn_input).detach().cpu().numpy().squeeze()
        vzwfs_signal_proj = self.M2C_2nd[:, :self.modes_2nd] @ self.cnn_output
        vzwfs_signal_proj = torch.from_numpy(vzwfs_signal_proj).float().to(device = self.device)
        # redoing in 2D


        vzwfs_signal_proj_2D = torch.zeros((11, 11), dtype=torch.float32).to(device = self.device)
        vzwfs_signal_proj_2D[self.mask] = vzwfs_signal_proj
        next_state = vzwfs_signal_proj_2D


        INFO        = {"KL_modes": self.modes_2nd}

        return next_state, INFO


    @torch.no_grad()
    def step(self, action, atm_OPD_1st_residual):
        self.atm_2nd.update(atm_OPD_1st_residual[self.CURRENT_STEPS])

        # frame delay implementation (zernike)
        self.vzwfssignal_buffer.append(action)
        self.vzwfs_delayed_signal = self.vzwfssignal_buffer[0]
        self.vzwfssignal_buffer.pop(0)

        self.dm_2nd.coefs = self.SECOND_LEAK * self.dm_2nd_copy - self.vzwfs_delayed_signal[self.mask].detach().cpu().numpy()
        self.dm_2nd_copy = self.dm_2nd.coefs.copy()


        self.src_2nd ** self.atm_2nd * self.tel_2nd * self.dm_2nd * self.vzwfs
        #self.src_2nd ** self.atm_2nd * self.tel_2nd * self.vzwfs


        cnn_input = self.frame_preprocessor.ProcessFrame(self.vzwfs.signal_2D_cam_padded/np.sum(self.vzwfs.signal_2D_cam_padded))
        cnn_input = torch.from_numpy(cnn_input).unsqueeze(0).float().to(device = self.device)
        cnn_output = - self.cnn(cnn_input).detach().cpu().numpy().squeeze()
        vzwfs_signal_proj = self.M2C_2nd[:, :self.modes_2nd] @ cnn_output
        vzwfs_signal_proj = torch.from_numpy(vzwfs_signal_proj).float().to(device = self.device)

        #redoing in 2D the zwfs signal
        vzwfs_signal_torch_proj_2D = torch.zeros((11, 11), dtype = torch.float32).to(device = self.device)
        vzwfs_signal_torch_proj_2D[self.mask] = vzwfs_signal_proj
        next_state = vzwfs_signal_torch_proj_2D


        strehl = np.exp(-np.var(self.tel_2nd.src.phase[np.where(self.tel_2nd.pupil == 1)]))

        self.CURRENT_STEPS += 1

        dm_commands = self.dm_2nd.coefs.copy()
        src_opd = self.src_2nd.OPD_no_pupil.copy()
        atm_opd = self.atm_2nd.OPD.copy()
        atm_opd_2 = atm_OPD_1st_residual[self.CURRENT_STEPS]


        INFO = {
            "2nd_stage_strehl": strehl,
            "telescope_pupil": self.tel_2nd.pupil, 
            "dm_commands": dm_commands,
            "reconstructed_cmd": vzwfs_signal_proj,
            "src_opd": src_opd,
            "projector_kl_2nd": self.projector_kl_2nd,
        }

        return next_state, INFO



    def generate_second_stage_atmosphere(self, nLoop = 1000):
        """2nd stage atm generation"""
        self.atm_2nd.initializeAtmosphere(telescope = self.tel_2nd)
        self.atm_2nd.generateNewPhaseScreen(seed = self.seed)

        self.src_2nd ** self.atm_2nd * self.tel_2nd * self.dm_2nd * self.vzwfs

        sample = self.atm_2nd.OPD
        out = np.zeros((nLoop, sample.shape[0], sample.shape[1]))

        for i in tqdm.tqdm(range(nLoop), desc="Generate second-stage atmosphere"):
            self.atm_2nd.update()
            out[i] = self.atm_2nd.OPD

            self.src_2nd ** self.atm_2nd * self.tel_2nd * self.dm_2nd * self.vzwfs

        self.atm_OPDs_2nd = out
        return out






    def run_first_stage_loop(
        self, nLoop, atm_OPD_1st, gainCL = 0.4, leak = 0.995, photon_noise = False
    ):
        self.Papytwin.set_pupil(
            calibration = True)       
        #     sky_offset = list(self.FIRST_SKY_OFFSET), 
        # )
        self.atm_1st.initializeAtmosphere(telescope = self.tel_1st)
        self.atm_1st.generateNewPhaseScreen(seed = self.seed)
        self.ngs_1st.reset()
        self.dm_1st.coefs = 0
        self.PWFS_signal_avg = 0
        self.PWFS_signal_avg_norm = 0
        self.pwfs.cam  = self.Papytwin.OCAM
        

        self.ngs_1st ** self.atm_1st * self.tel_1st * self.dm_1st * self.pwfs


        dm_commands = np.zeros((nLoop, self.dm_1st.nValidAct))
        reconstructed_cmd = np.zeros((int(nLoop/2), self.dm_1st.nValidAct))
        src_opds = np.zeros((nLoop, self.tel_1st.pupil.shape[0], self.tel_1st.pupil.shape[1]))
        self.pwfssignal_buffer  = [np.zeros(self.pwfs.nSignal)] #2frame delay pwfs
        strehlers = np.zeros(100)

        #TODO papyrus ngs is R, src is src =   Source('IR1310', 0)
        #TODO > 1.55 - 1.7 µm goes to 2nd stage wavefront sensing
        #TODO so we are closing the loop on the ngs and src is I guess if I plan to attach a science camera?
        #TODO setting for the 2nd stage self.src = Source(optBand='H', magnitude=-2.5)
            # self.src.wavelength = 1.6e-06
            # self.src.bandwidth = 2e-07
        #TODO for now I haven't changed the wavelengths because for OPD it does not matter
        #TODO if you add cameras later then it will matter (I don't think Mathieu had any cameras?)

        for i in range(nLoop):
            self.atm_1st.update(atm_OPD_1st[i])
            
            self.PWFS_signal_avg += (self.pwfs.signal_2D + self.pwfs.referenceSignal_2D) * self.pwfs.norma #fullFrameMaps
            self.PWFS_signal_avg_norm += self.pwfs.norma

            if (i + 1) % 2 == 0:
                # PWFS slope averaging over 2 frames (should be implemented correctly, but if something goes wrong you can turn this off and check)
                self.PWFS_signal_avg = self.PWFS_signal_avg / self.PWFS_signal_avg_norm - self.pwfs.referenceSignal_2D
                self.PWFS_signal_avg = self.PWFS_signal_avg[np.where(self.pwfs.validSignal == 1)]

                # frame delay implementation
                self.pwfssignal_buffer.append(self.PWFS_signal_avg)
                pwfs_delayed_signal = self.pwfssignal_buffer[0]
                self.pwfssignal_buffer.pop(0)
                reconstructed_cmd[int((i-1)/2)] = np.matmul(self.reconstructor_1st, pwfs_delayed_signal)
                self.dm_1st.coefs = leak * self.dm_1st.coefs - (gainCL * reconstructed_cmd[int((i-1)/2)])


                self.PWFS_signal_avg = 0
                self.PWFS_signal_avg_norm = 0


            self.ngs_1st ** self.atm_1st * self.tel_1st * self.dm_1st * self.slow_tt * self.pwfs
            strehl_1st = np.exp(-np.var(self.tel_1st.src.phase[np.where(self.tel_1st.pupil == 1)]))
            strehlers[i%100] = strehl_1st


            dm_commands[i]  = self.dm_1st.coefs.copy()
            src_opds[i]  = self.ngs_1st.OPD_no_pupil.copy()
            
            if i % 100 == 99:
                print(f'First stage strehl ratio: {np.mean(strehlers)}, iteration {i}')



        

        residuals_opd_1st = self.project_opd_between_pupils(src_opds, input_pupil = self.tel_1st.pupil,
                                                                output_pupil = self.tel_2nd.pupil, 
                                                                output_shape = self.tel_2nd.pupil.shape)
            
        first_stage_results = {
        "config": FirstStageLoopConfig(
            nLoop=nLoop,
            gainCL=gainCL,
            leak=leak,
            frame_delay=2,
            photon_noise=photon_noise,
        ),
        "telescope_pupil": self.tel_1st.pupil, 
        "dm_commands": dm_commands,
        "reconstructed_cmd": reconstructed_cmd,
        "src_opds": src_opds,
        "residuals_opds_1rst": residuals_opd_1st,
        "projector_kl_1st": self.projector_kl_1st,
        }
        return first_stage_results


    

    def _crop_pupil(self, pupil = None):
        if pupil is None:
            pupil = self.tel_1st.pupil
        where_pupil = np.where(pupil)
        x_, x = where_pupil[0].min(),where_pupil[0].max() #gets the minimum and maximum index values
        y_, y = where_pupil[1].min(),where_pupil[1].max()
        final_mask = pupil[x_:x+1,y_:y+1] 
        
        return final_mask, (x_, x,y_, y)
    

    def _crop_opd(self, OPDs:np.ndarray= None, pupil = None):
        if OPDs is None:
            OPDs = self.OPDs
        if pupil is None:
            pupil = self.tel_2nd.pupil #gets the telescope pupil
        final_mask, _ = self._crop_pupil(pupil) #so this function crops the pupil if there is any outside pixels that are False
        if OPDs.ndim == 3:
            OPDs_crop = np.zeros((OPDs.shape[0], final_mask.shape[0], final_mask.shape[1])).astype(np.float32)
            OPDs_crop[:, final_mask] = OPDs[:, pupil] #assigns the old OPD values to the newly cropped OPDs_crop
        else:
            OPDs_crop = np.zeros_like(final_mask).astype(np.float32)
            OPDs_crop[final_mask] = OPDs[pupil] #assigns the old OPD values to the newly cropped OPDs_crop
        return final_mask, OPDs_crop


    def _rescale_matrix(self, A, j, k, anti_aliasing=True):
        """
        Resample a 2D, 3D, or 4D array to a target spatial shape.

        Parameters
        ----------
        A : ndarray
            Input array to resize. Supported dimensions are 2, 3, and 4.
        j : int
            Target size along the first spatial axis.
        k : int
            Target size along the second spatial axis.
        anti_aliasing : bool, optional
            Whether to apply anti-aliasing during resizing.

        Returns
        -------
        ndarray
            Resized array with the requested spatial dimensions.
        """
        if A.ndim == 3:
            l, m, n = A.shape
            return resize(A, (l, j, k), order=5, anti_aliasing=anti_aliasing)
        elif A.ndim == 4:
            o, l, m, n = A.shape
            return resize(A, (o, l, j, k), order=5, anti_aliasing=anti_aliasing)
        elif A.ndim == 2:
            m, n = A.shape
            return resize(A, (j, k), order=5, anti_aliasing=anti_aliasing) #resizes the image to specified size




    def project_opd_between_pupils(
        self,
        OPDs: np.ndarray,
        input_pupil = None,
        output_pupil = None,
        output_shape = None
    ):
        OPDs = np.asarray(OPDs)
        output_pupil = np.asarray(output_pupil).astype(bool)

        opd_rescaled = self._rescale_matrix(
            OPDs,
            output_shape[0],
            output_shape[1],
        )

        out = opd_rescaled

        return out


    def project_atmosphere_to_first_stage(self, atm_OPD_2nd):

        self.atm_OPDs_1rst = self.project_opd_between_pupils(
            atm_OPD_2nd,
            input_pupil = None,
            output_pupil = self.tel_1st.pupil,
            output_shape = self.tel_1st.pupil.shape
        )


        out = self.atm_OPDs_1rst.copy()
        return out









