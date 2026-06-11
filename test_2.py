import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\Users\evinerskas\ffmpeg-2026-03-01-git-862338fe31-essentials_build\ffmpeg-2026-03-01-git-862338fe31-essentials_build\bin\ffmpeg.exe'
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Circle
from matplotlib import cm
import matplotlib
from numpy.fft import fft2, fftshift
from functions import *
import time
from matplotlib.colors import Normalize

import OOPAO
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Detector import Detector
from OOPAO.Pyramid import Pyramid
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Zernike import Zernike
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.tools.displayTools import displayMap, makeSquareAxes
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.tools.displayTools import cl_plot, displayMap
from OOPAO.MisRegistration import MisRegistration


from OOPAO.ZWFS2 import ZWFS2
from PIL import Image
from PAPYRIIS_2stage_CNN_RL.parameterFile_papyriis import initializeParameterFile

import imageio
from astropy.io import fits
param = initializeParameterFile()

from PAPYRIIS_2stage_CNN_RL.PAPYRUS.Papyrus import Papyrus

Papytwin = Papyrus()
tel = Papytwin.tel
ngs = Papytwin.ngs
dm_1st = Papytwin.dm
pwfs = Papytwin.wfs
atm = Papytwin.atm


"""
pwfs = Pyramid(nSubap            = (80)//1,
              telescope             = tel,
              modulation            = 5,
              lightRatio            = 0,
              n_pix_separation      = 40//1,
              n_pix_edge            = 20//1,
              psfCentering          = True,
              postProcessing        = 'fullFrame',)




T152onDM_size       = 35.5
PapyrusOnDM_size    = 37.5



tel = Telescope(resolution    = 80,
                    diameter            = 1.52,
                    samplingTime        = 1/200,
                    centralObstruction  = 0,
                    fov                 = 0)

ngs=Source(optBand   = 'R',\
               magnitude = -0.05)




ngs * tel

atm=Atmosphere(telescope     = tel,\
                   r0            = 0.05,\
                   L0            = 30,\
                   windSpeed     = np.array([4, 3, 5, 5, 2]),\
                   fractionalR0  = np.array([0.45,0.1,0.1,0.25,0.1]),\
                   windDirection = np.array([4, 3, 5, 5, 2]),\
                   altitude      = np.array([0, 1000,5000,10000,12000]))


misReg          = MisRegistration(param)
pitch           = 2.5 #mm
DM_diag_size    = 17 * pitch #mm
scale_T152DM = DM_diag_size / T152onDM_size
D_T152 = 1.52

x = np.linspace(-scale_T152DM * D_T152/2, scale_T152DM * D_T152/2, param['nActuator'])
[X,Y] = np.meshgrid(x,x)

DM_coordinates = np.asarray([X.reshape(17**2),Y.reshape(17**2)]).T
dist           = np.sqrt(DM_coordinates[:,0]**2 + DM_coordinates[:,1]**2)
DM_coordinates = DM_coordinates[dist <= D_T152/2 + 2.2 *pitch * D_T152 / T152onDM_size, :]
DM_pitch       = pitch * D_T152 / T152onDM_size

# hardcoded for now
alpao_unit     = 30*7591.024876


dm_1st = DeformableMirror(telescope    = tel,\
                    nSubap       = 16,\
                    mechCoupling = param['mechanicalCoupling'],\
                    misReg       = misReg, \
                    coordinates  = DM_coordinates,\
                    pitch        = DM_pitch,\
                    modes        = None,
                    flip_lr      = True,
                    sign         = -1/alpao_unit)


pwfs = Pyramid(nSubap            = (80)//1,
              telescope             = tel,
              modulation            = 5,
              lightRatio            = 0,
              n_pix_separation      = 40//1,
              n_pix_edge            = 20//1,
              psfCentering          = True,
              postProcessing        = 'fullFrame',)



"""
M2C_1st = np.load("PAPYRIIS_2stage_CNN_RL/M2C_1rst.npy")
'''self.M2C_1st    = compute_KL_basis(self.tel_1st,
                            self.atm_1st,
                            self.dm_1st,
                            lim=0)       #TODO np.load("PAPYRIIS_2stage_CNN_RL/M2C_1rst.npy")'''
M2C_1st_CL = M2C_1st[:, :195]
valid_pixels  = np.load("PAPYRIIS_2stage_CNN_RL/useful_pix.npy")

#self.im_1st = read_mat('PAPYRIIS_2stage_CNN_RL/intMat_klOOPAO_synthetic_bin=1_F=500_rMod=5_20250604_0307.mat')['matrix_inf']
pwfs.modulation = 5
calib_1st = InteractionMatrix(  ngs = ngs,\
                    atm            = atm,\
                    tel            = tel,\
                    dm             = dm_1st,\
                    wfs            = pwfs,\
                    M2C            = M2C_1st_CL,\
                    stroke         = 0.0001,\
                    phaseOffset    = 0,\
                    nMeasurements  = 1,\
                    noise          = 'off',
                    print_time=False,
                    display=True)





calib_1st_M = calib_1st.M
reconstructor_1st = M2C_1st_CL @ calib_1st_M

KL_basis_dm = (dm_1st.modes @ M2C_1st_CL) * np.tile(tel.pupil.flatten()[:, None], M2C_1st_CL.shape[1])
projector_kl = np.linalg.pinv(KL_basis_dm)


from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
@dataclass
class FirstStageLoopConfig:
    """Configuration used by the first-stage closed loop."""

    nLoop: Optional[int] = None
    gainCL: float = 0.5
    leak: float = 0.995
    frame_delay: int = 2
    photon_noise: bool = False
    progress: bool = True

def run_first_stage_loop(
    nLoop, atm_OPD_1st, gainCL = 0.7, leak = 0.995, photon_noise = False
):

    atm.initializeAtmosphere(telescope = tel)
    dm_1st.coefs = 0
    ngs ** atm * tel * dm_1st * pwfs

    PWFS_signal_avg = 0
    PWFS_signal_avg_norm = 0
    dm_commands = np.zeros((nLoop, dm_1st.nValidAct))
    reconstructed_cmd = np.zeros((nLoop, dm_1st.nValidAct))
    src_opds = np.zeros((nLoop, tel.pupil.shape[0], tel.pupil.shape[1]))
    atm_opds = np.zeros((nLoop, tel.pupil.shape[0], tel.pupil.shape[1]))
    atm_opds_2 = np.zeros((nLoop, tel.pupil.shape[0], tel.pupil.shape[1]))

    pwfssignal_buffer  = [np.zeros(pwfs.nSignal)] #2frame delay pwfs




    for i in range(nLoop):
        atm.update(atm_OPD_1st[i])        
        atm_opds[i] = atm.OPD.copy()

        PWFS_signal_avg += (pwfs.signal_2D + pwfs.referenceSignal_2D) * pwfs.norma #fullFrameMaps
        PWFS_signal_avg_norm += pwfs.norma
        
        if (i + 1) % 2 == 0:
            # PWFS slope averaging over 2 frames (should be implemented correctly, but if something goes wrong you can turn this off and check)
            PWFS_signal_avg = PWFS_signal_avg / PWFS_signal_avg_norm - pwfs.referenceSignal_2D
            PWFS_signal_avg = PWFS_signal_avg[np.where(pwfs.validSignal == 1)]

            # frame delay implementation
            pwfssignal_buffer.append(PWFS_signal_avg)
            pwfs_delayed_signal = pwfssignal_buffer[0]
            pwfssignal_buffer.pop(0)
            reconstructed_cmd[i] = np.matmul(reconstructor_1st, pwfs_delayed_signal)
            dm_1st.coefs = 0.995 * dm_1st.coefs - (0.7 * reconstructed_cmd[i])


            PWFS_signal_avg = 0
            PWFS_signal_avg_norm = 0
        

        ngs ** atm * tel * dm_1st * pwfs
        strehl_1st = np.exp(-np.var(tel.src.phase[np.where(tel.pupil == 1)]))
        print(f'strehl ratio {strehl_1st}')
        dm_commands[i]  = dm_1st.coefs.copy()
        src_opds[i]  = ngs.OPD_no_pupil.copy()
        atm_opds_2[i] = dm_1st.OPD.copy()
        
    #residuals_opd_1st = self.project_opd_between_pupils(src_opds, input_pupil = tel_calib.pupil,
    #                                                    output_pupil = tel_2nd.pupil, 
    #                                                    output_shape = tel_2nd.pupil.shape)
    first_stage_results = {
    "config": FirstStageLoopConfig(
                nLoop=nLoop,
                gainCL=gainCL,
                leak=leak,
                frame_delay=2,
                photon_noise=photon_noise,
            ),
    "telescope_pupil": tel.pupil, 
    "dm_commands": dm_commands, #total dm commands (what is the point of saving this?)
    "reconstructed_cmd": reconstructed_cmd, #wfs measurement on dm space (what I need)
    "src_opds": src_opds, #residual OPDs (residuals without the wfs measurement error; needed for 2nd stage)
    "atm_opds": atm_opds,
    "atm_opds_2": atm_opds_2,
    #"residuals_opds_1rst": residuals_opd_1st,
    "projector_kl": projector_kl,
    }
    return first_stage_results



atm_OPD_1st = np.load("PAPYRIIS_2stage_CNN_RL/pre_saved_data/atm_OPDs_1rst.npy")





a = time.perf_counter()
first_stage_results = run_first_stage_loop(120, atm_OPD_1st)
b = time.perf_counter()
print('time', (b-a))


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)  # 1 row, 3 columns
plt.ion()

for frame1, frame2, frame3 in zip(first_stage_results["atm_opds"], first_stage_results["atm_opds_2"], first_stage_results["src_opds"]):
    ax1.clear()
    ax2.clear()
    ax3.clear()

    ax1.imshow(frame1 - frame2)
    ax1.axis("off")
    ax1.set_title("atm_opd")

    ax2.imshow(frame2)
    ax2.axis("off")
    ax2.set_title("dm_opds_2")

    ax3.imshow(frame3)
    ax3.axis("off")
    ax3.set_title("residuals_opds_1rst")

    plt.pause(0.01)

plt.ioff()
plt.show()


