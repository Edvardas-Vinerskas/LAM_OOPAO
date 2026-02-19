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
import os

from po4ao_edw.OOPAO_environment_ZWFS import OOPAO_environment_ZWFS
from po4ao_edw.OOPAO_environment_PWFS import OOPAO_environment_PWFS
from po4ao_edw.OOPAO_environment_ZWFS_perfect import OOPAO_environment_ZWFS_perfect
from po4ao_edw.OOPAO_environment_ZWFS_1_stage import OOPAO_environment_ZWFS_1_stage
from OOPAO.ZWFS2 import ZWFS2
from PIL import Image
from po4ao_PAPYRUS.po4ao_util_PAPYRUS import EfficientExperienceReplay, SharedAdam

"""replay = EfficientExperienceReplay((10, 10), (10, 10), 20 * 750)
replay_warmup = EfficientExperienceReplay((10, 10), (10, 10), 20 * 750)

loaddir_buffer = 'temp_save_dir/vZWFS_1st_2nd_noise_03_phnoise1_read_4_4_QE08_mag2_epis20_warmupperc05/buffer'
loaddir_models = 'temp_save_dir/vZWFS_1st_2nd_noise_03_phnoise1_read_4_4_QE08_mag2_epis20_warmupperc05/models'

replay_warmup.states = torch.load(os.path.join(loaddir_buffer, f"states_warmup.pt"))
replay_warmup.next_states = torch.load(os.path.join(loaddir_buffer, f"next_states_warmup.pt"))
replay_warmup.actions = torch.load(os.path.join(loaddir_buffer, f"actions_warmup.pt"))

replay_warmup.set_len(20 * 750 - 1)

replay.states = torch.load(os.path.join(loaddir_buffer, f"states.pt"))
replay.next_states = torch.load(os.path.join(loaddir_buffer, f"next_states.pt"))
replay.actions = torch.load(os.path.join(loaddir_buffer, f"actions.pt"))
"""

print(1 @ 1)





