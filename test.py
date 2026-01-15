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


arcsec_per_pixel = 206265 * (790e-9/1.52) / 6

directory_name_RL = "vZWFS_1st_2nd_noise_03_wooftw_seed_chang"
frames_array = np.load(f"temp_save_dir/{directory_name_RL}/tel_psf_array.npy")
NNN = 300
frames_array = frames_array[:, NNN:-NNN, NNN:-NNN]

h, w = frames_array.shape[1:]
cx = (w - 1) / 2
cy = (h - 1) / 2
print(frames_array.shape)
print(cx, cy)

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(frames_array[0], norm=SymLogNorm(1e-1), origin='lower')
ax.axis('off')


#fix the correction zone problem
# Add main circles on top
corr_zone_1 = Circle([cx, cy], (1.125635899881358/arcsec_per_pixel),
                      fc='none', ec='lime', ls='-', linewidth=2)
corr_zone_2 = Circle([cx, cy], (0.536017095181599/arcsec_per_pixel),
                      fc='none', ec='red', ls='-', linewidth=2)

# Add shadows first, then main circles
ax.add_patch(corr_zone_1)
ax.add_patch(corr_zone_2)


def update(frame_idx):
    im.set_array(frames_array[frame_idx])
    return [im, corr_zone_1, corr_zone_2]

ani = animation.FuncAnimation(fig, update, frames=2000, interval=1, blit=True)
ani.save('vZWFS_1st_2nd_noise_03_wooftw_seed_chang.gif', writer='pillow', fps=1500)
plt.show()