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

import torch

from po4ao_edw.OOPAO_environment_ZWFS import OOPAO_environment_ZWFS
from po4ao_edw.OOPAO_environment_PWFS import OOPAO_environment_PWFS
from po4ao_edw.OOPAO_environment_ZWFS_1_stage import OOPAO_environment_ZWFS_1_stage
from OOPAO.ZWFS2 import ZWFS2
from PIL import Image


import imageio
from astropy.io import fits

mvieif = True


scaling_factor = 1e2

directory_name = 'test_5_warmup50_2ndstageonly_scaling1e2'

next_state_dynamics_tensor = torch.load(f"bench_loss/{directory_name}/next_state_dynamics_tensor.pt").detach().cpu().squeeze(1).numpy() #dynamics model ground truth
pred_tensor = torch.load(f"bench_loss/{directory_name}/pred_tensor.pt").detach().cpu().squeeze(1).numpy() #dynamics model predictions
actions = torch.load(f"bench_loss/{directory_name}/actions.pt").cpu().numpy() #policy output
next_states_2nd = torch.load(f"bench_loss/{directory_name}/next_states_2nd.pt").cpu().numpy() #2nd stage wfs measurement in DM space

actionsANDnext_states_2nd = np.concatenate([actions, next_states_2nd], axis = 2)
pred_tensorANDnext_state_dynamics_tensor = np.concatenate([pred_tensor, next_state_dynamics_tensor], axis = 2)

next_state_dynamics_tensor_mean = np.mean(np.abs(next_state_dynamics_tensor), axis = (1, 2))# * scaling_factor
next_state_dynamics_tensor_max = np.max(next_state_dynamics_tensor, axis = (1, 2))#* scaling_factor

pred_tensor_mean = np.mean(np.abs(pred_tensor), axis = (1, 2))# * scaling_factor
pred_tensor_max = np.max(pred_tensor, axis = (1, 2))#* scaling_factor

plt.figure()
plt.title('dynamics_groundtruth vs dynamics prediction mean')
plt.plot(next_state_dynamics_tensor_mean)
plt.plot(pred_tensor_mean)
plt.figure()
plt.title('dynamics_groundtruth vs dynamics prediction max')
plt.plot(next_state_dynamics_tensor_max)
plt.plot(pred_tensor_max)

plt.show()



def make_movie(data, output_file=f"bench_loss/{directory_name}/movie.mp4", frame_step=1, fps=150, colormap="viridis"):
    """
    Make a movie from a numpy array of shape (T, H, W).

    Args:
        data:        numpy array, shape (T, H, W)
        output_file: ".mp4" (requires ffmpeg) or ".gif" (requires pillow)
        frame_step:  sample every Nth frame
        fps:         frames per second
        colormap:    matplotlib colormap, e.g. "viridis", "plasma", "RdBu"
    """
    frames = data[::frame_step]
    norm = Normalize(vmin=data.min(), vmax=data.max())

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#0f0f0f")
    ax.axis("off")

    im = ax.imshow(frames[0], cmap=colormap, norm=norm,
                   interpolation="nearest", aspect="equal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.outline.set_edgecolor("white")

    time_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes,
        color="white", fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#222", alpha=0.7)
    )
    fig.tight_layout()

    def update(i):
        im.set_data(frames[i])
        time_text.set_text(f"{i * frame_step:,} / {len(data):,}")
        return im, time_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=True
    )

    if output_file.endswith(".gif"):
        ani.save(output_file, writer="pillow", fps=fps)
    else:
        ani.save(output_file, writer="ffmpeg", fps=fps, dpi=150)

    print(f"Saved → {output_file}  ({len(frames)} frames @ {fps} fps)")
    plt.close()



#if mvieif:
    #make_movie(actionsANDnext_states_2nd[:10000], output_file=f"bench_loss/{directory_name}/actionsANDnext_states_2nd_movie.mp4")
    #make_movie(pred_tensorANDnext_state_dynamics_tensor[-10000:], output_file=f"bench_loss/{directory_name}/pred_tensorANDnext_state_dynamics_tensor_movie.mp4")
    #make_movie(actions[:10000], output_file=f"bench_loss/{directory_name}/policy_output_movie.mp4")
    #make_movie(next_states_2nd[:10000], output_file=f"bench_loss/{directory_name}/next_states_2nd_movie.mp4")
    #make_movie(pred_tensor[-10000:], output_file=f"bench_loss/{directory_name}/pred_tensor_movie.mp4")
    #make_movie(next_state_dynamics_tensor[-10000:], output_file=f"bench_loss/{directory_name}/next_state_dynamics_tensor_movie.mp4")






