import matplotlib.pyplot as plt
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

import torch

from po4ao_edw.OOPAO_environment_ZWFS import OOPAO_environment_ZWFS
from po4ao_edw.OOPAO_environment_PWFS import OOPAO_environment_PWFS
from po4ao_edw.OOPAO_environment_ZWFS_1_stage import OOPAO_environment_ZWFS_1_stage
from OOPAO.ZWFS2 import ZWFS2
from PIL import Image


import imageio


def array_to_video(images, output_video, fps=30, cmap='viridis'):
    # Use the new matplotlib API
    colormap = matplotlib.colormaps[cmap]

    num_frames, height, width = images.shape
    images_rgb = np.zeros((num_frames, height, width, 3), dtype=np.uint8)

    for i, frame in enumerate(images):
        # Handle case where all values are the same (avoid division by zero)
        frame_min = frame.min()
        frame_max = frame.max()

        if frame_max - frame_min == 0:
            # If all values are the same, set to middle gray
            frame_norm = np.ones_like(frame) * 0.5
        else:
            frame_norm = (frame - frame_min) / (frame_max - frame_min)

        # Apply colormap
        frame_colored = colormap(frame_norm)
        images_rgb[i] = (frame_colored[:, :, :3] * 255).astype(np.uint8)

    imageio.mimsave(output_video, images_rgb, fps=fps, bitrate='5000k')


# Usage
images = np.load('temp_save_dir/vZWFS_1st_2nd_noise_03_phnoise1_read_4_4_QE08_mag2_epis20_onlinetrain_40/DM_zer_OPD_array.npy')
print(images.shape)
array_to_video(images, 'temp_save_dir/vZWFS_1st_2nd_noise_03_phnoise1_read_4_4_QE08_mag2_epis20_onlinetrain_40/DM_zer_OPD_array_video_epis6_3seconds.mp4', fps=150)