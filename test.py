import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from numpy.fft import fft2, fftshift
from functions import *

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

import torch

from po4ao_edw.OOPAO_environment_ZWFS import OOPAO_environment_ZWFS
from po4ao_edw.OOPAO_environment_PWFS import OOPAO_environment_PWFS
import matplotlib.animation as animation
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Circle
import glob
from PIL import Image

#n_pix_separation = 2
#n_pix_edge = 0
#self.zeroPaddingFactor = 2.1
#n extra pixel
print(int(np.round((2/20) * 160/(160/20)/2/1))) #1
#n centre pixel
print(int(np.round((42/1)/2))) #21
#self.cam.resolution = 44
#self.resolution = 352
    #resolution seems to refer to pyramid pupil plane detector size that is arbitrarily scaled?
    #resolution of the pyramid which then is binned?
    #why would you need this shouldn't the telescope resolution be key then?
print(int((20*2+2 +0*2)*160/20)) #36
#n_pixels = 20

#this is the boolean array of valid pixels where the pixel values exceed the lightRatio
#self.validI4Q = (self.I4Q >= self.lightRatio*self.I4Q.max())

"""
so indeed the self.resolution is 352 which is the resolution of the raw PWFS data, only after
is this shit binned into the detector which has a resolution of 44

so indeed this resolution corresponds to the resolution of the telescope for a single quadrant
it all makes sense now
"""
env = OOPAO_environment_ZWFS()

DM_fake = DeformableMirror(telescope = env.TEL,
                           nSubap    = 40,
                           mechCoupling = env.MECH_COUPLING)

M2C_KL_fake = compute_KL_basis(tel = env.TEL, atm = env.ATM, dm = DM_fake)
M2C_fake    = M2C_KL_fake[:, :400]
modes_fake  = DM_fake.modes @ M2C_fake
print(modes_fake.shape)

modes_fake_inv = np.linalg.pinv(np.squeeze(modes_fake[env.TEL.pupilLogical, :]))


directory_name_int = 'vZWFS_metrics_integrator'
residual_OPD_array_int = np.load(f"temp_save_dir/{directory_name_int}/residual_OPD_array.npy") #for use in spatial PSD/KL modes var and correlation

strehl_array_1st_int = np.load(f"temp_save_dir/{directory_name_int}/strehl_array_1st.npy")

strehl_array_2nd_int = np.load(f"temp_save_dir/{directory_name_int}/strehl_array_2nd.npy")
strehl_mask_int = strehl_array_2nd_int

coefficient_matrix_res_list_int        = []
coefficient_matrix_res_list_int_full   = []
residual_OPD_array_05_int = residual_OPD_array_int[strehl_mask_int > 0.5]
#for i in range(len(residual_OPD_array_05_int)):
final_residual_phase_int = 2 * np.pi * residual_OPD_array_05_int[0] / env.SRC.wavelength
coefficient_matrix_res_int = modes_fake_inv @ final_residual_phase_int[np.where(env.TEL.pupil > 0)]
coefficient_matrix_res_list_int.append(coefficient_matrix_res_int)
coefficient_matrix_res_int_tip = np.zeros_like(coefficient_matrix_res_int)
coefficient_matrix_res_int_tip[0] = coefficient_matrix_res_int[0]
coefficient_matrix_res_int_tilt = np.zeros_like(coefficient_matrix_res_int)
coefficient_matrix_res_int_tilt[1] = coefficient_matrix_res_int[1]
coefficient_matrix_res_int_2 = np.zeros_like(coefficient_matrix_res_int)
coefficient_matrix_res_int_2[2] = coefficient_matrix_res_int[2]
coefficient_matrix_res_int_3 = np.zeros_like(coefficient_matrix_res_int)
coefficient_matrix_res_int_3[3] = coefficient_matrix_res_int[3]


final_residual_phase_int_v2 = final_residual_phase_int[np.where(env.TEL.pupil > 0)] - np.squeeze(modes_fake[env.TEL.pupilLogical, :])@coefficient_matrix_res_int_tilt
tip_residual = modes_fake_inv @ final_residual_phase_int_v2
print(coefficient_matrix_res_int_tilt[1])
print(tip_residual[1])


