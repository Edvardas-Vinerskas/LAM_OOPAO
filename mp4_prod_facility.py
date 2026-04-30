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

"""import OOPAO
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
"""
import torch

"""from po4ao_edw.OOPAO_environment_ZWFS import OOPAO_environment_ZWFS
from po4ao_edw.OOPAO_environment_PWFS import OOPAO_environment_PWFS
from po4ao_edw.OOPAO_environment_ZWFS_1_stage import OOPAO_environment_ZWFS_1_stage
from OOPAO.ZWFS2 import ZWFS2
from PIL import Image"""


import imageio
#from astropy.io import fits


def array_to_video(images, output_video, fps=30, cmap='viridis'):
    colormap = matplotlib.colormaps[cmap]

    num_frames, height, width = images.shape
    images_rgb = np.zeros((num_frames, height, width, 3), dtype=np.uint8)

    global_min = images.min()
    global_max = images.max()
    print(f"Global min: {global_min:.3f}, Global max: {global_max:.3f}")

    for i, frame in enumerate(images):
        if global_max - global_min == 0:
            frame_norm = np.ones_like(frame) * 0.5
        else:
            frame_norm = (frame - global_min) / (global_max - global_min)

        frame_colored = colormap(frame_norm)
        images_rgb[i] = (frame_colored[:, :, :3] * 255).astype(np.uint8)

    imageio.mimsave(output_video, images_rgb, fps=fps, bitrate='5000k')


# Usage
# Load data
"""hdul = fits.open('RL_TEST_0-CL-2026-02-23T16_43_44-Cube.fits')
hdul.info()"""

RL_telemetry_2nd = np.load('bench_loss_5_0324/test_warmup50_1st200_2nd400_scaling1e2_noise2ndstageonly_losspenalty01_iters300_v21/2026-03-30T20_00_35_telemetry_2nd_data.npy', allow_pickle = True)
data = RL_telemetry_2nd.item()['CRED2Cube'] - RL_telemetry_2nd.item()['credDark']#hdul[0].data.astype(np.float32)
xxx = RL_telemetry_2nd.item()['validPixels'].copy()
xxx[xxx < 1e-5] = 0
xxx[xxx >= 1e-5] = 1
data = data * xxx[np.newaxis, :, :]

plt.figure()
plt.imshow(data[0])
plt.show()

print(data.shape)

"""# Crop all frames and zero out the first row of every frame
corrected = data[:, 175:-175, 225:-225]
corrected[:, 0, :] = 0  # zero first row of ALL frames

# Take corner crop and compute background reference
corner_crop = data[:, 0:200, 0:225]  # shape: (5000, 200, 225)
background = corner_crop.mean(axis=0)  # shape: (200, 225) - averaged over all frames

# Subtract background mean (scalar) from every frame
background_level = background.mean()  # single scalar value
corrected -= background_level"""

array_to_video(data, 'RL_ZWFS_cam.mp4', fps=150)

#images = np.load('temp_save_dir/vZWFS_1st_2nd_noise_03_phnoise1_read_4_4_QE08_mag2_epis20_onlinetrain_40/DM_zer_OPD_array.npy')
#print(images.shape)
#array_to_video(images, 'temp_save_dir/vZWFS_1st_2nd_noise_03_phnoise1_read_4_4_QE08_mag2_epis20_onlinetrain_40/DM_zer_OPD_array_video_epis6_3seconds.mp4', fps=150)